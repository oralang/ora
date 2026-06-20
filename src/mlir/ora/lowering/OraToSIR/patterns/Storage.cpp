#include "patterns/Storage.h"
#include "patterns/AdtCarrierHelpers.h"
#include "patterns/EVMConstants.h"
#include "patterns/ErrorUnionCarrierHelpers.h"
#include "patterns/LoweringHelpers.h"
#include "patterns/StorageLayout.h"
#include "OraMaterializationKinds.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;

using mlir::ora::lowering::addStorageWordOffset;
using mlir::ora::lowering::coerceToU256;
using mlir::ora::lowering::constU256;
using mlir::ora::lowering::createPtrViewMaterializationCast;
using mlir::ora::lowering::ensureU256;
using mlir::ora::lowering::getElementWordCount;
using mlir::ora::lowering::getStaticMemRefWordCount;
using mlir::ora::lowering::getStorageWordCount;
using mlir::ora::lowering::getStructFieldStorageOffset;
using mlir::ora::lowering::getStructFieldAttrs;
using mlir::ora::lowering::kStorageMemRefViewKind;
using mlir::ora::lowering::kStorageStructCarrierKind;
using mlir::ora::lowering::kStorageStructViewFieldsAttr;

namespace {
    static bool getConstU64(Value v, uint64_t &out)
    {
        if (auto cst = v.getDefiningOp<sir::ConstOp>())
        {
            if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
            {
                out = intAttr.getValue().getZExtValue();
                return true;
            }
        }
        return false;
    }

    static MapHashKey makeMapHashKey(Operation *funcOp, Value mapSlot, Value key)
    {
        MapHashKey k;
        k.func = funcOp ? reinterpret_cast<void *>(funcOp) : nullptr;
        k.keyVal = key.getAsOpaquePointer();
        uint64_t constVal = 0;
        if (getConstU64(mapSlot, constVal))
        {
            k.mapIsConst = true;
            k.mapConst = constVal;
        }
        else
        {
            k.mapIsConst = false;
            k.mapVal = mapSlot.getAsOpaquePointer();
        }
        return k;
    }

    static Value lookupCachedMapHash(MapHashCache &cache, Operation *funcOp, Operation *anchor, Value mapSlot, Value key)
    {
        MapHashKey k = makeMapHashKey(funcOp, mapSlot, key);
        auto it = cache.hashes.find(k);
        if (it == cache.hashes.end())
            return Value();
        Value hash = it->second;
        if (!anchor || !hash)
            return hash;
        if (auto *def = hash.getDefiningOp())
        {
            auto func = anchor->getParentOfType<func::FuncOp>();
            if (func)
            {
                DominanceInfo dom(func);
                if (dom.dominates(def, anchor))
                    return hash;
            }
        }
        return Value();
    }

    static void storeCachedMapHash(MapHashCache &cache, Operation *funcOp, Value mapSlot, Value key, Value hash)
    {
        MapHashKey k = makeMapHashKey(funcOp, mapSlot, key);
        cache.hashes[k] = hash;
    }

    static void emitEmptyRevert(PatternRewriter &rewriter, Location loc)
    {
        auto *ctx = rewriter.getContext();
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace=*/1);
        Value zero = constU256(rewriter, loc, 0);
        Value zeroPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, zero);
        rewriter.create<sir::RevertOp>(loc, zeroPtr, zero);
    }
} // namespace

// Debug logging macro
#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

// Helper: Set result name on an operation (for better readability in SIR MLIR)
static void setResultName(Operation *op, unsigned resultIndex, StringRef name)
{
    auto nameAttr = StringAttr::get(op->getContext(), name);
    std::string attrName = "sir.result_name_" + std::to_string(resultIndex);
    op->setAttr(attrName, nameAttr);
}

static Value computeWordCount(Location loc, Value lengthU256, PatternRewriter &rewriter)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    Value addend = constU256(rewriter, loc, 31);
    Value divisor = constU256(rewriter, loc, 32);
    Value sum = rewriter.create<sir::AddOp>(loc, u256Type, lengthU256, addend);
    return rewriter.create<sir::DivOp>(loc, u256Type, sum, divisor);
}

static Value dynamicStorageDataBase(Location loc, Value slot, PatternRewriter &rewriter)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    Value wordSize = constU256(rewriter, loc, 32);
    Value tmp = rewriter.create<sir::MallocOp>(loc, ptrType, wordSize);
    rewriter.create<sir::StoreOp>(loc, tmp, slot);
    return rewriter.create<sir::KeccakOp>(loc, u256Type, tmp, wordSize);
}

static FailureOr<Value> materializeStorageMapKey(Operation *op,
                                                 Value key,
                                                 Type originalKeyType,
                                                 ConversionPatternRewriter &rewriter,
                                                 StringRef context)
{
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    if (llvm::isa<ora::StringType, ora::BytesType>(originalKeyType))
    {
        if (!llvm::isa<sir::PtrType>(key.getType()))
            return rewriter.notifyMatchFailure(op, llvm::Twine(context) + " dynamic key did not lower to pointer");
        Value length = rewriter.create<sir::LoadOp>(loc, u256Type, key);
        Value wordSize = constU256(rewriter, loc, 32);
        Value payload = rewriter.create<sir::AddPtrOp>(loc, ptrType, key, wordSize);
        return rewriter.create<sir::KeccakOp>(loc, u256Type, payload, length).getResult();
    }

    Value keyU256 = ensureU256(rewriter, loc, op, key, context);
    if (!keyU256)
        return failure();
    return keyU256;
}

static FailureOr<Value> materializeDynamicBytesLoadFromStorageRoot(Operation *op,
                                                                   Value slot,
                                                                   Type convertedResultType,
                                                                   PatternRewriter &rewriter)
{
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    if (!convertedResultType)
        convertedResultType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    if (!llvm::isa<sir::PtrType>(convertedResultType))
        return failure();

    Value length = rewriter.create<sir::SLoadOp>(loc, u256Type, slot);
    Value wordSize = constU256(rewriter, loc, 32);
    Value totalSize = rewriter.create<sir::AddOp>(loc, u256Type, length, wordSize);
    Value basePtr = rewriter.create<sir::MallocOp>(loc, convertedResultType, totalSize);
    rewriter.create<sir::StoreOp>(loc, basePtr, length);

    Value wordCount = computeWordCount(loc, length, rewriter);
    Value storageDataBase = dynamicStorageDataBase(loc, slot, rewriter);
    Value zero = constU256(rewriter, loc, 0);
    Value one = constU256(rewriter, loc, 1);

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
    auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});

    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::BrOp>(loc, ValueRange{zero}, condBlock);

    rewriter.setInsertionPointToStart(condBlock);
    Value iv = condBlock->getArgument(0);
    Value lt = rewriter.create<sir::LtOp>(loc, u256Type, iv, wordCount);
    rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv}, ValueRange{}, bodyBlock, afterBlock);

    rewriter.setInsertionPointToStart(bodyBlock);
    Value ivU256 = bodyBlock->getArgument(0);
    Value wordSlot = rewriter.create<sir::AddOp>(loc, u256Type, storageDataBase, ivU256);
    Value wordVal = rewriter.create<sir::SLoadOp>(loc, u256Type, wordSlot);

    Value wordBytes = rewriter.create<sir::MulOp>(loc, u256Type, ivU256, wordSize);
    Value dataOffset = rewriter.create<sir::AddOp>(loc, u256Type, wordBytes, wordSize);
    Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, convertedResultType, basePtr, dataOffset);
    rewriter.create<sir::StoreOp>(loc, dataPtr, wordVal);

    Value next = rewriter.create<sir::AddOp>(loc, u256Type, ivU256, one);
    rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

    rewriter.setInsertionPoint(op);
    return basePtr;
}

static LogicalResult replaceWithDynamicBytesLoadFromStorageRoot(Operation *op,
                                                                Value slot,
                                                                Type convertedResultType,
                                                                ConversionPatternRewriter &rewriter)
{
    auto loaded = materializeDynamicBytesLoadFromStorageRoot(op, slot, convertedResultType, rewriter);
    if (failed(loaded))
        return rewriter.notifyMatchFailure(op, "dynamic bytes storage value did not lower to pointer");
    rewriter.replaceOp(op, *loaded);
    return success();
}

static LogicalResult storeDynamicBytesValueToStorageRoot(Operation *op,
                                                         Value value,
                                                         Value slot,
                                                         PatternRewriter &rewriter,
                                                         const TypeConverter *typeConverter)
{
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    Value basePtr = value;
    if (!llvm::isa<sir::PtrType>(basePtr.getType()) &&
        llvm::isa<ora::StringType, ora::BytesType>(basePtr.getType()))
    {
        // Dynamic bytes/string have a fixed runtime pointer representation.
        Type converted = typeConverter ? typeConverter->convertType(basePtr.getType()) : ptrType;
        if (!converted)
            return rewriter.notifyMatchFailure(op, "failed to convert dynamic bytes storage pointer view type");
        if (!llvm::isa<sir::PtrType>(converted))
            return rewriter.notifyMatchFailure(op, "dynamic bytes storage value did not lower to pointer");
        basePtr = createPtrViewMaterializationCast(rewriter, loc, converted, basePtr);
    }
    if (!llvm::isa<sir::PtrType>(basePtr.getType()) && typeConverter)
    {
        Type converted = typeConverter->convertType(basePtr.getType());
        if (!converted || !llvm::isa<sir::PtrType>(converted))
            return rewriter.notifyMatchFailure(op, "dynamic bytes storage value did not lower to pointer");
        if (converted != basePtr.getType())
            basePtr = rewriter.create<sir::BitcastOp>(loc, converted, basePtr);
    }
    if (!llvm::isa<sir::PtrType>(basePtr.getType()))
        return rewriter.notifyMatchFailure(op, "dynamic bytes storage value is not a pointer");

    Value length = rewriter.create<sir::LoadOp>(loc, u256Type, basePtr);
    rewriter.create<sir::SStoreOp>(loc, slot, length);

    Value writeCount = computeWordCount(loc, length, rewriter);
    Value storageDataBase = dynamicStorageDataBase(loc, slot, rewriter);
    Value zero = constU256(rewriter, loc, 0);
    Value one = constU256(rewriter, loc, 1);
    Value wordSize = constU256(rewriter, loc, 32);

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
    auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});

    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::BrOp>(loc, ValueRange{zero}, condBlock);

    rewriter.setInsertionPointToStart(condBlock);
    Value iv = condBlock->getArgument(0);
    Value lt = rewriter.create<sir::LtOp>(loc, u256Type, iv, writeCount);
    rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv}, ValueRange{}, bodyBlock, afterBlock);

    rewriter.setInsertionPointToStart(bodyBlock);
    Value ivU256 = bodyBlock->getArgument(0);
    Value wordSlot = rewriter.create<sir::AddOp>(loc, u256Type, storageDataBase, ivU256);
    Value wordBytes = rewriter.create<sir::MulOp>(loc, u256Type, ivU256, wordSize);
    Value dataOffset = rewriter.create<sir::AddOp>(loc, u256Type, wordBytes, wordSize);
    Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, dataOffset);
    Value wordVal = rewriter.create<sir::LoadOp>(loc, u256Type, dataPtr);
    rewriter.create<sir::SStoreOp>(loc, wordSlot, wordVal);

    Value next = rewriter.create<sir::AddOp>(loc, u256Type, ivU256, one);
    rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

    rewriter.eraseOp(op);
    rewriter.setInsertionPointToStart(afterBlock);
    return success();
}

static Value getStorageMemRefViewSlot(Value value);

static LogicalResult copyStaticMemRefValueToStorageRoot(Operation *op,
                                                        Value value,
                                                        Value slot,
                                                        mlir::MemRefType memrefType,
                                                        ConversionPatternRewriter &rewriter)
{
    auto wordCount = getStaticMemRefWordCount(op, memrefType);
    if (!wordCount)
        return rewriter.notifyMatchFailure(op, "static memref storage value has dynamic shape");

    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    if (Value sourceSlot = getStorageMemRefViewSlot(value))
    {
        if (sourceSlot == slot)
            return success();

        for (uint64_t i = 0; i < *wordCount; ++i)
        {
            Value src = addStorageWordOffset(loc, sourceSlot, i, rewriter);
            Value dst = addStorageWordOffset(loc, slot, i, rewriter);
            Value word = rewriter.create<sir::SLoadOp>(loc, u256Type, src);
            rewriter.create<sir::SStoreOp>(loc, dst, word);
        }
        return success();
    }

    Value basePtr = value;
    if (!llvm::isa<sir::PtrType>(basePtr.getType()))
        basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, basePtr);

    for (uint64_t i = 0; i < *wordCount; ++i)
    {
        Value dst = addStorageWordOffset(loc, slot, i, rewriter);
        Value offset = constU256(rewriter, loc, i * 32ULL);
        Value wordPtr = i == 0 ? basePtr : rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, offset).getResult();
        Value word = rewriter.create<sir::LoadOp>(loc, u256Type, wordPtr);
        rewriter.create<sir::SStoreOp>(loc, dst, word);
    }
    return success();
}

static Value getStorageMemRefViewSlot(Value value)
{
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() == 1)
        {
            Value operand = cast.getOperand(0);
            auto viewKind = cast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
            if (viewKind && viewKind.getValue() == kStorageMemRefViewKind &&
                llvm::isa<sir::U256Type>(operand.getType()))
                return operand;
            if (operand.getDefiningOp())
                return getStorageMemRefViewSlot(operand);
        }
    }

    if (auto bitcast = value.getDefiningOp<sir::BitcastOp>())
    {
        Value operand = bitcast.getOperand();
        auto viewKind = bitcast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
        if (viewKind && viewKind.getValue() == kStorageMemRefViewKind &&
            llvm::isa<sir::U256Type>(operand.getType()))
            return operand;
        if (operand.getDefiningOp())
            return getStorageMemRefViewSlot(operand);
    }
    return Value();
}

static FailureOr<Value> loadStructValueFromStorageRoot(Operation *anchor,
                                                       Location loc,
                                                       Value slot,
                                                       ora::StructType structType,
                                                       ConversionPatternRewriter &rewriter,
                                                       const TypeConverter *typeConverter)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    ArrayAttr fieldNamesAttr;
    ArrayAttr fieldTypesAttr;
    if (!getStructFieldAttrs(anchor, structType, fieldNamesAttr, fieldTypesAttr))
        return failure();

    SmallVector<Value> fieldValues;
    fieldValues.reserve(fieldNamesAttr.size());

    uint64_t offset = 0;
    for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
    {
        Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
        Value fieldSlot = addStorageWordOffset(loc, slot, offset, rewriter);

        Value fieldValue;
        if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(fieldType))
        {
            FailureOr<Value> nested = loadStructValueFromStorageRoot(anchor, loc, fieldSlot, nestedStructType, rewriter, typeConverter);
            if (failed(nested))
                return failure();
            fieldValue = *nested;
        }
        else
        {
            auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            if (fieldMemRefType)
            {
                Type convertedFieldType = typeConverter ? typeConverter->convertType(fieldType) : Type();
                if (!convertedFieldType)
                    convertedFieldType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
                fieldValue = rewriter.create<sir::BitcastOp>(loc, convertedFieldType, fieldSlot);
                fieldValue.getDefiningOp()->setAttr(
                    kOraMaterializationKindAttr,
                    StringAttr::get(ctx, kStorageMemRefViewKind));
            }
            else
            {
                fieldValue = rewriter.create<sir::SLoadOp>(loc, u256Type, fieldSlot);

                Type convertedFieldType = typeConverter ? typeConverter->convertType(fieldType) : Type();
                if (!convertedFieldType)
                    convertedFieldType = u256Type;
                if (convertedFieldType != u256Type)
                {
                    fieldValue = rewriter.create<sir::BitcastOp>(loc, convertedFieldType, fieldValue);
                }
            }
        }

        fieldValues.push_back(fieldValue);
        offset += getStorageWordCount(anchor, fieldType);
    }

    auto structInitOp = rewriter.create<ora::StructInitOp>(loc, structType, fieldValues);
    structInitOp->setAttr(
        kOraMaterializationKindAttr,
        StringAttr::get(ctx, kStorageStructCarrierKind));
    return structInitOp.getResult();
}

static bool structTypeContainsDynamicMemRefField(Operation *anchor, ora::StructType structType)
{
    ArrayAttr fieldNamesAttr;
    ArrayAttr fieldTypesAttr;
    if (!getStructFieldAttrs(anchor, structType, fieldNamesAttr, fieldTypesAttr))
        return false;

    for (Attribute fieldTypeAttr : fieldTypesAttr)
    {
        Type fieldType = cast<TypeAttr>(fieldTypeAttr).getValue();
        if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            fieldMemRefType && !fieldMemRefType.hasStaticShape())
        {
            return true;
        }
        if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(fieldType);
            nestedStructType && structTypeContainsDynamicMemRefField(anchor, nestedStructType))
        {
            return true;
        }
    }

    return false;
}

static LogicalResult storeMemoryStructValueToStorageRoot(Operation *op,
                                                         Location loc,
                                                         Value structDataBase,
                                                         Value slotBase,
                                                         ora::StructType structType,
                                                         PatternRewriter &rewriter,
                                                         Region *parentRegion,
                                                         Block *insertBeforeBlock)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    ArrayAttr fieldNamesAttr;
    ArrayAttr fieldTypesAttr;
    if (!getStructFieldAttrs(op, structType, fieldNamesAttr, fieldTypesAttr))
        return failure();

    Value wordSize = constU256(rewriter, loc, 32);
    Value zero = constU256(rewriter, loc, 0);
    Value one = constU256(rewriter, loc, 1);

    uint64_t offset = 0;
    for (Attribute fieldTypeAttr : fieldTypesAttr)
    {
        Type fieldType = cast<TypeAttr>(fieldTypeAttr).getValue();
        uint64_t fieldWords = getStorageWordCount(op, fieldType);
        Value fieldSlot = addStorageWordOffset(loc, slotBase, offset, rewriter);
        Value fieldByteOffset = constU256(rewriter, loc, offset * 32ULL);
        Value fieldPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, structDataBase, fieldByteOffset);

        if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(fieldType))
        {
            if (failed(storeMemoryStructValueToStorageRoot(
                    op, loc, fieldPtr, fieldSlot, nestedStructType, rewriter, parentRegion, insertBeforeBlock)))
            {
                return failure();
            }
            offset += fieldWords;
            continue;
        }

        if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            fieldMemRefType && !fieldMemRefType.hasStaticShape())
        {
            Value nestedPtrWord = rewriter.create<sir::LoadOp>(loc, u256Type, fieldPtr);
            Value nestedPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, nestedPtrWord);
            Value nestedLength = rewriter.create<sir::LoadOp>(loc, u256Type, nestedPtr);
            rewriter.create<sir::SStoreOp>(loc, fieldSlot, nestedLength);

            Value nestedTmp = rewriter.create<sir::MallocOp>(loc, ptrType, wordSize);
            rewriter.create<sir::StoreOp>(loc, nestedTmp, fieldSlot);
            Value nestedStorageBase = rewriter.create<sir::KeccakOp>(loc, u256Type, nestedTmp, wordSize);

            uint64_t nestedElemWords = getElementWordCount(op, fieldMemRefType.getElementType());
            Value nestedWriteCount = nestedLength;
            if (nestedElemWords != 1)
            {
                Value nestedElemWordsConst = constU256(rewriter, loc, nestedElemWords);
                nestedWriteCount = rewriter.create<sir::MulOp>(loc, u256Type, nestedLength, nestedElemWordsConst);
            }

            Block *dynamicFieldBlock = rewriter.getInsertionBlock();
            auto innerCondBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator(), {u256Type}, {loc});
            auto innerBodyBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator(), {u256Type}, {loc});
            auto innerAfterBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
            rewriter.setInsertionPointToEnd(dynamicFieldBlock);
            rewriter.create<sir::BrOp>(loc, ValueRange{zero}, innerCondBlock);

            rewriter.setInsertionPointToStart(innerCondBlock);
            Value innerIv = innerCondBlock->getArgument(0);
            Value innerLt = rewriter.create<sir::LtOp>(loc, u256Type, innerIv, nestedWriteCount);
            rewriter.create<sir::CondBrOp>(loc, innerLt, ValueRange{innerIv}, ValueRange{}, innerBodyBlock, innerAfterBlock);

            rewriter.setInsertionPointToStart(innerBodyBlock);
            Value innerIndex = innerBodyBlock->getArgument(0);
            Value innerSlot = rewriter.create<sir::AddOp>(loc, u256Type, nestedStorageBase, innerIndex);
            Value innerBytes = rewriter.create<sir::MulOp>(loc, u256Type, innerIndex, wordSize);
            Value innerDataOffset = rewriter.create<sir::AddOp>(loc, u256Type, innerBytes, wordSize);
            Value innerDataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, nestedPtr, innerDataOffset);
            Value innerWordVal = rewriter.create<sir::LoadOp>(loc, u256Type, innerDataPtr);
            rewriter.create<sir::SStoreOp>(loc, innerSlot, innerWordVal);
            Value nextInner = rewriter.create<sir::AddOp>(loc, u256Type, innerIndex, one);
            rewriter.create<sir::BrOp>(loc, ValueRange{nextInner}, innerCondBlock);

            rewriter.setInsertionPointToStart(innerAfterBlock);
            offset += fieldWords;
            continue;
        }

        for (uint64_t fieldWord = 0; fieldWord < fieldWords; ++fieldWord)
        {
            Value wordSlot = addStorageWordOffset(loc, slotBase, offset + fieldWord, rewriter);
            Value wordByteOffset = constU256(rewriter, loc, (offset + fieldWord) * 32ULL);
            Value wordPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, structDataBase, wordByteOffset);
            Value wordVal = rewriter.create<sir::LoadOp>(loc, u256Type, wordPtr);
            rewriter.create<sir::SStoreOp>(loc, wordSlot, wordVal);
        }
        offset += fieldWords;
    }

    return success();
}

static LogicalResult storeDynamicMemRefToStorageRoot(Operation *op,
                                                     Value value,
                                                     Value slot,
                                                     mlir::MemRefType memrefType,
                                                     PatternRewriter &rewriter,
                                                     bool eraseSourceOp = true)
{
    auto loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    Value basePtr = value;
    if (!llvm::isa<sir::PtrType>(basePtr.getType()))
        basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, basePtr);

    Value length = rewriter.create<sir::LoadOp>(loc, u256Type, basePtr);
    rewriter.create<sir::SStoreOp>(loc, slot, length);

    uint64_t elemWords = getElementWordCount(op, memrefType.getElementType());
    Value writeCount = length;
    Value wordSize = constU256(rewriter, loc, 32);
    if (elemWords != 1)
    {
        Value elemWordsConst = constU256(rewriter, loc, elemWords);
        writeCount = rewriter.create<sir::MulOp>(loc, u256Type, length, elemWordsConst);
    }

    Value tmp = rewriter.create<sir::MallocOp>(loc, ptrType, wordSize);
    rewriter.create<sir::StoreOp>(loc, tmp, slot);
    Value storageDataBase = rewriter.create<sir::KeccakOp>(loc, u256Type, tmp, wordSize);

    Value zero = constU256(rewriter, loc, 0);
    Value one = constU256(rewriter, loc, 1);

    if (auto elementStructType = llvm::dyn_cast<ora::StructType>(memrefType.getElementType()))
    {
        ArrayAttr fieldNamesAttr;
        ArrayAttr fieldTypesAttr;
        if (!getStructFieldAttrs(op, elementStructType, fieldNamesAttr, fieldTypesAttr))
            return failure();

        bool hasDynamicField = structTypeContainsDynamicMemRefField(op, elementStructType);

        if (hasDynamicField)
        {
            Block *parentBlock = op->getBlock();
            Region *parentRegion = parentBlock->getParent();
            auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
            auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
            auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});

            rewriter.setInsertionPointToEnd(parentBlock);
            rewriter.create<sir::BrOp>(loc, ValueRange{zero}, condBlock);

            rewriter.setInsertionPointToStart(condBlock);
            Value row = condBlock->getArgument(0);
            Value lt = rewriter.create<sir::LtOp>(loc, u256Type, row, length);
            rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{row}, ValueRange{}, bodyBlock, afterBlock);

            rewriter.setInsertionPointToStart(bodyBlock);
            Value rowIndex = bodyBlock->getArgument(0);
            Value rowWordOffset = rowIndex;
            if (elemWords != 1)
            {
                Value elemWordsConst = constU256(rewriter, loc, elemWords);
                rowWordOffset = rewriter.create<sir::MulOp>(loc, u256Type, rowIndex, elemWordsConst);
            }
            Value rowSlotBase = rewriter.create<sir::AddOp>(loc, u256Type, storageDataBase, rowWordOffset);
            Value rowBytes = rewriter.create<sir::MulOp>(loc, u256Type, rowWordOffset, wordSize);
            Value rowDataOffset = rewriter.create<sir::AddOp>(loc, u256Type, rowBytes, wordSize);
            Value rowDataBase = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, rowDataOffset);

            if (failed(storeMemoryStructValueToStorageRoot(
                    op, loc, rowDataBase, rowSlotBase, elementStructType, rewriter, parentRegion, afterBlock)))
            {
                return failure();
            }

            Value next = rewriter.create<sir::AddOp>(loc, u256Type, rowIndex, one);
            rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

            if (eraseSourceOp)
                rewriter.eraseOp(op);
            rewriter.setInsertionPointToStart(afterBlock);
            return success();
        }
    }

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
    auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});

    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::BrOp>(loc, ValueRange{zero}, condBlock);

    rewriter.setInsertionPointToStart(condBlock);
    Value iv = condBlock->getArgument(0);
    Value lt = rewriter.create<sir::LtOp>(loc, u256Type, iv, writeCount);
    rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv}, ValueRange{}, bodyBlock, afterBlock);

    rewriter.setInsertionPointToStart(bodyBlock);
    Value ivU256 = bodyBlock->getArgument(0);
    Value slotAddr = rewriter.create<sir::AddOp>(loc, u256Type, storageDataBase, ivU256);
    Value wordBytes = rewriter.create<sir::MulOp>(loc, u256Type, ivU256, wordSize);
    Value dataOffset = rewriter.create<sir::AddOp>(loc, u256Type, wordBytes, wordSize);
    Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, dataOffset);

    if (auto nestedMemRefType = llvm::dyn_cast<mlir::MemRefType>(memrefType.getElementType());
        nestedMemRefType && !nestedMemRefType.hasStaticShape())
    {
        Value rowPtrWord = rewriter.create<sir::LoadOp>(loc, u256Type, dataPtr);
        Value rowPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, rowPtrWord);
        Value rowLength = rewriter.create<sir::LoadOp>(loc, u256Type, rowPtr);
        rewriter.create<sir::SStoreOp>(loc, slotAddr, rowLength);

        Value rowTmp = rewriter.create<sir::MallocOp>(loc, ptrType, wordSize);
        rewriter.create<sir::StoreOp>(loc, rowTmp, slotAddr);
        Value rowStorageBase = rewriter.create<sir::KeccakOp>(loc, u256Type, rowTmp, wordSize);

        auto innerCondBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
        auto innerBodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
        auto innerAfterBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());
        rewriter.setInsertionPointToEnd(bodyBlock);
        rewriter.create<sir::BrOp>(loc, ValueRange{zero}, innerCondBlock);

        rewriter.setInsertionPointToStart(innerCondBlock);
        Value innerIv = innerCondBlock->getArgument(0);
        Value innerLt = rewriter.create<sir::LtOp>(loc, u256Type, innerIv, rowLength);
        rewriter.create<sir::CondBrOp>(loc, innerLt, ValueRange{innerIv}, ValueRange{}, innerBodyBlock, innerAfterBlock);

        rewriter.setInsertionPointToStart(innerBodyBlock);
        Value innerIvU256 = innerBodyBlock->getArgument(0);
        Value innerSlotAddr = rewriter.create<sir::AddOp>(loc, u256Type, rowStorageBase, innerIvU256);
        Value innerWordBytes = rewriter.create<sir::MulOp>(loc, u256Type, innerIvU256, wordSize);
        Value innerDataOffset = rewriter.create<sir::AddOp>(loc, u256Type, innerWordBytes, wordSize);
        Value innerDataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, rowPtr, innerDataOffset);
        Value innerWordVal = rewriter.create<sir::LoadOp>(loc, u256Type, innerDataPtr);
        rewriter.create<sir::SStoreOp>(loc, innerSlotAddr, innerWordVal);
        Value nextInner = rewriter.create<sir::AddOp>(loc, u256Type, innerIvU256, one);
        rewriter.create<sir::BrOp>(loc, ValueRange{nextInner}, innerCondBlock);

        rewriter.setInsertionPointToStart(innerAfterBlock);
        Value next = rewriter.create<sir::AddOp>(loc, u256Type, ivU256, one);
        rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

        rewriter.eraseOp(op);
        rewriter.setInsertionPointToStart(afterBlock);
        return success();
    }

    Value wordVal = rewriter.create<sir::LoadOp>(loc, u256Type, dataPtr);
    rewriter.create<sir::SStoreOp>(loc, slotAddr, wordVal);

    Value next = rewriter.create<sir::AddOp>(loc, u256Type, ivU256, one);
    rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

    if (eraseSourceOp)
        rewriter.eraseOp(op);
    rewriter.setInsertionPointToStart(afterBlock);
    return success();
}

struct DynamicStructFieldStore
{
    Value value;
    Value slot;
    mlir::MemRefType type;
};

static LogicalResult storeStructValueToStorageRoot(Operation *anchor,
                                                   Location loc,
                                                   Value value,
                                                   Value slot,
                                                   ora::StructType structType,
                                                   ConversionPatternRewriter &rewriter,
                                                   const TypeConverter *typeConverter,
                                                   SmallVectorImpl<DynamicStructFieldStore> *deferredDynamicStores = nullptr)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    ArrayAttr fieldNamesAttr;
    ArrayAttr fieldTypesAttr;
    if (!getStructFieldAttrs(anchor, structType, fieldNamesAttr, fieldTypesAttr))
        return failure();

    if (auto update = value.getDefiningOp<ora::StructFieldUpdateOp>())
    {
        size_t updatedFieldIndex = 0;
        auto fieldOffset = getStructFieldStorageOffset(
            anchor, structType, update.getFieldName(), &updatedFieldIndex);
        if (!fieldOffset)
            return failure();

        Value fieldSlot = addStorageWordOffset(loc, slot, *fieldOffset, rewriter);

        Type fieldType = cast<TypeAttr>(fieldTypesAttr[updatedFieldIndex]).getValue();
        Value updatedValue = update.getValue();
        if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(fieldType))
        {
            return storeStructValueToStorageRoot(
                anchor, loc, updatedValue, fieldSlot, nestedStructType, rewriter, typeConverter);
        }
        if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            fieldMemRefType && fieldMemRefType.hasStaticShape())
        {
            return copyStaticMemRefValueToStorageRoot(
                anchor, updatedValue, fieldSlot, fieldMemRefType, rewriter);
        }
        if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            fieldMemRefType && !fieldMemRefType.hasStaticShape())
        {
            return storeDynamicMemRefToStorageRoot(
                anchor, updatedValue, fieldSlot, fieldMemRefType, rewriter, /*eraseSourceOp=*/false);
        }

        if (!llvm::isa<sir::U256Type>(updatedValue.getType()))
        {
            Type convertedFieldType = typeConverter ? typeConverter->convertType(updatedValue.getType()) : Type();
            if (convertedFieldType != updatedValue.getType() && convertedFieldType && llvm::isa<sir::U256Type>(convertedFieldType))
                updatedValue = rewriter.create<sir::BitcastOp>(loc, convertedFieldType, updatedValue);
            else
                updatedValue = rewriter.create<sir::BitcastOp>(loc, u256Type, updatedValue);
        }
        rewriter.create<sir::SStoreOp>(loc, fieldSlot, updatedValue);
        return success();
    }

    SmallVector<Value> fieldValues;
    if (auto structInit = value.getDefiningOp<ora::StructInitOp>())
    {
        fieldValues.append(structInit.getFieldValues().begin(), structInit.getFieldValues().end());
    }
    else if (auto structInstantiate = value.getDefiningOp<ora::StructInstantiateOp>())
    {
        fieldValues.append(structInstantiate.getFieldValues().begin(), structInstantiate.getFieldValues().end());
    }
    else
    {
        fieldValues.reserve(fieldNamesAttr.size());
        for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
        {
            auto fieldName = cast<StringAttr>(fieldNamesAttr[i]).getValue();
            Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
            fieldValues.push_back(rewriter.create<ora::StructFieldExtractOp>(loc, fieldType, value, fieldName));
        }
    }

    if (fieldValues.size() != fieldNamesAttr.size())
        return failure();

    auto dynamicFieldIsPreservedStorageView = [&](size_t fieldIndex) {
        Operation *definingOp = value.getDefiningOp();
        if (!definingOp)
            return false;
        auto attr = definingOp->getAttrOfType<ArrayAttr>(kStorageStructViewFieldsAttr);
        if (!attr)
            return false;
        for (Attribute entry : attr)
        {
            auto intAttr = dyn_cast<IntegerAttr>(entry);
            if (intAttr && intAttr.getUInt() == fieldIndex)
                return true;
        }
        return false;
    };

    SmallVector<DynamicStructFieldStore, 4> localDynamicFieldStores;
    SmallVectorImpl<DynamicStructFieldStore> &dynamicFieldStores =
        deferredDynamicStores ? *deferredDynamicStores : localDynamicFieldStores;

    uint64_t offset = 0;
    for (size_t i = 0; i < fieldValues.size(); ++i)
    {
        Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
        Value fieldSlot = addStorageWordOffset(loc, slot, offset, rewriter);
        Value stored = fieldValues[i];

        if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(fieldType))
        {
            if (failed(storeStructValueToStorageRoot(
                    anchor, loc, stored, fieldSlot, nestedStructType, rewriter, typeConverter, &dynamicFieldStores)))
                return failure();
            offset += getStorageWordCount(anchor, fieldType);
            continue;
        }

        if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            fieldMemRefType && fieldMemRefType.hasStaticShape())
        {
            if (failed(copyStaticMemRefValueToStorageRoot(anchor, stored, fieldSlot, fieldMemRefType, rewriter)))
                return failure();
            offset += getStorageWordCount(anchor, fieldType);
            continue;
        }

        if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            fieldMemRefType && !fieldMemRefType.hasStaticShape())
        {
            if (dynamicFieldIsPreservedStorageView(i))
            {
                offset += getStorageWordCount(anchor, fieldType);
                continue;
            }
            if (Value sourceSlot = getStorageMemRefViewSlot(stored))
            {
                if (sourceSlot == fieldSlot)
                {
                    offset += getStorageWordCount(anchor, fieldType);
                    continue;
                }
                return failure();
            }
            dynamicFieldStores.push_back({stored, fieldSlot, fieldMemRefType});
            offset += getStorageWordCount(anchor, fieldType);
            continue;
        }

        if (!llvm::isa<sir::U256Type>(stored.getType()))
        {
            Type convertedFieldType = typeConverter ? typeConverter->convertType(stored.getType()) : Type();
            if (convertedFieldType != stored.getType() && convertedFieldType && llvm::isa<sir::U256Type>(convertedFieldType))
                stored = rewriter.create<sir::BitcastOp>(loc, convertedFieldType, stored);
            else
                stored = rewriter.create<sir::BitcastOp>(loc, u256Type, stored);
        }

        rewriter.create<sir::SStoreOp>(loc, fieldSlot, stored);
        offset += getStorageWordCount(anchor, fieldType);
    }

    if (deferredDynamicStores)
        return success();

    for (const auto &dynamicField : dynamicFieldStores)
    {
        if (failed(storeDynamicMemRefToStorageRoot(
                anchor, dynamicField.value, dynamicField.slot, dynamicField.type, rewriter, /*eraseSourceOp=*/false)))
        {
            return failure();
        }
    }

    return success();
}

static Value buildIndexFromU256(ConversionPatternRewriter &rewriter, Location loc, Operation *anchor, Value value)
{
    return ensureU256(rewriter, loc, anchor, value, "storage index");
}

// -----------------------------------------------------------------------------
// Helper: Find existing slot constant in function to avoid duplicates
// -----------------------------------------------------------------------------
static Value findOrCreateSlotConstant(Operation *op, uint64_t slotIndex,
                                      StringRef globalName,
                                      PatternRewriter &rewriter)
{
    auto loc = op->getLoc();
    auto ctx = rewriter.getContext();

    std::string slotName = "slot_" + globalName.str();
    if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>())
    {
        Block &entry = parentFunc.getBody().front();
        for (Operation &entryOp : entry)
        {
            auto constOp = llvm::dyn_cast<sir::ConstOp>(entryOp);
            if (!constOp)
                continue;
            auto nameAttr = constOp->getAttrOfType<StringAttr>("sir.result_name_0");
            if (nameAttr && nameAttr.getValue() == slotName)
                return constOp.getResult();
        }

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&entry);
        Value slotConst = constU256(rewriter, loc, slotIndex);
        slotConst.getDefiningOp()->setAttr("sir.result_name_0", StringAttr::get(ctx, slotName));
        return slotConst;
    }

    // Fallback for non-function contexts.
    Value slotConst = constU256(rewriter, loc, slotIndex);
    slotConst.getDefiningOp()->setAttr("sir.result_name_0", StringAttr::get(ctx, slotName));
    return slotConst;
}

static LogicalResult storeAdtCarrierToStorage(ora::SStoreOp op,
                                              ArrayRef<Value> operands,
                                              Value baseSlot,
                                              PatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    SmallVector<Value, 2> parts;
    if (operands.size() == 2)
    {
        Value tag = ensureU256(rewriter, loc, op.getOperation(), operands[0], "ADT storage tag");
        Value payload = ensureU256(rewriter, loc, op.getOperation(), operands[1], "ADT storage payload");
        if (!tag || !payload)
            return failure();
        parts.push_back(tag);
        parts.push_back(payload);
    }
    else
    {
        ora::adt_helpers::AdtConstructCarrierOptions options;
        options.requireExistingScalarCarrier = true;
        options.acceptExistingAggregateCarrier = true;

        auto normalized = ora::adt_helpers::getAdtPartsFromConstructOrNormalized(
            rewriter, loc, op.getValue(), options);
        if (failed(normalized))
            return rewriter.notifyMatchFailure(op, "expected normalized ADT carrier for storage store");
        parts.push_back(normalized->first);
        parts.push_back(normalized->second);
    }

    ora::adt_helpers::storeAdtPartsToStorageRoot(rewriter, loc, baseSlot, parts[0], parts[1]);
    rewriter.eraseOp(op);
    return success();
}

static bool errorUnionStoragePayloadUsesPointer(Type successType)
{
    return llvm::isa<ora::TupleType, ora::StructType, ora::AnonymousStructType,
                     ora::StringType, ora::BytesType, ora::AdtType, ora::MapType,
                     mlir::MemRefType, mlir::UnrankedMemRefType>(successType);
}

static bool errorUnionStorageSupportsPointerPayload(Type successType)
{
    return llvm::isa<ora::StringType, ora::BytesType, mlir::MemRefType>(successType);
}

static bool errorUnionStoragePayloadUnsupported(Type successType)
{
    return errorUnionStoragePayloadUsesPointer(successType) &&
           !errorUnionStorageSupportsPointerPayload(successType);
}

static bool errorDeclHasPayload(Operation *anchor, StringRef name)
{
    ModuleOp module = anchor ? anchor->getParentOfType<ModuleOp>() : ModuleOp();
    if (!module)
        return true;

    bool found = false;
    bool hasPayload = true;
    module.walk([&](Operation *decl) {
        if (found)
            return;
        if (!isa<ora::ErrorDeclOp>(decl) && !isa<sir::ErrorDeclOp>(decl))
            return;
        auto sym = decl->getAttrOfType<StringAttr>("sym_name");
        if (!sym || sym.getValue() != name)
            return;
        found = true;
        auto paramTypes = decl->getAttrOfType<ArrayAttr>("ora.param_types");
        if (!paramTypes)
            paramTypes = decl->getAttrOfType<ArrayAttr>("sir.param_types");
        hasPayload = paramTypes && !paramTypes.empty();
    });

    // Unknown declarations should not be treated as payloadless at a storage boundary.
    return !found || hasPayload;
}

static bool errorUnionStorageHasPayloadBearingErrors(ora::ErrorUnionType errorUnionType, Operation *anchor)
{
    for (Type errorType : errorUnionType.getErrorTypes())
    {
        auto structType = llvm::dyn_cast<ora::StructType>(errorType);
        if (!structType || errorDeclHasPayload(anchor, structType.getName()))
            return true;
    }
    return false;
}

static mlir::IntegerAttr lookupErrorIdAttr(Operation *anchor, StringRef name)
{
    ModuleOp module = anchor ? anchor->getParentOfType<ModuleOp>() : ModuleOp();
    if (!module)
        return {};

    mlir::IntegerAttr foundId;
    module.walk([&](Operation *decl) {
        if (foundId)
            return;
        if (!isa<ora::ErrorDeclOp>(decl) && !isa<sir::ErrorDeclOp>(decl))
            return;
        auto sym = decl->getAttrOfType<mlir::StringAttr>("sym_name");
        if (!sym || sym.getValue() != name)
            return;
        if (auto attr = decl->getAttrOfType<mlir::IntegerAttr>("ora.error_id"))
            foundId = attr;
        else if (auto attr = decl->getAttrOfType<mlir::IntegerAttr>("sir.error_id"))
            foundId = attr;
    });
    return foundId;
}

static Value storageErrorIdConst(PatternRewriter &rewriter, Location loc, mlir::IntegerAttr value)
{
    Value result = constU256(rewriter, loc, value);
    result.getDefiningOp()->setAttr("ora.error_id", value);
    return result;
}

static Value materializeStorageResultMemRefView(Location loc,
                                                Value payloadSlot,
                                                mlir::MemRefType successType,
                                                PatternRewriter &rewriter,
                                                const TypeConverter *typeConverter)
{
    auto *ctx = rewriter.getContext();
    if (!typeConverter)
    {
        auto view = rewriter.create<mlir::UnrealizedConversionCastOp>(
            loc, successType, payloadSlot);
        view->setAttr(
            kOraMaterializationKindAttr,
            StringAttr::get(ctx, kStorageMemRefViewKind));
        return view.getResult(0);
    }

    Type convertedPayloadType = typeConverter ? typeConverter->convertType(successType) : Type();
    if (!convertedPayloadType)
        convertedPayloadType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    if (!llvm::isa<sir::PtrType>(convertedPayloadType))
        return Value();

    Value payloadView = rewriter.create<sir::BitcastOp>(loc, convertedPayloadType, payloadSlot);
    payloadView.getDefiningOp()->setAttr(
        kOraMaterializationKindAttr,
        StringAttr::get(ctx, kStorageMemRefViewKind));
    return payloadView;
}

static Value materializeStorageResultDynamicBytesView(Location loc,
                                                      Value payloadSlot,
                                                      Type successType,
                                                      PatternRewriter &rewriter,
                                                      const TypeConverter *typeConverter)
{
    auto *ctx = rewriter.getContext();
    if (!typeConverter)
    {
        auto view = rewriter.create<mlir::UnrealizedConversionCastOp>(
            loc, successType, payloadSlot);
        view->setAttr(
            kOraMaterializationKindAttr,
            StringAttr::get(ctx, kStorageMemRefViewKind));
        return view.getResult(0);
    }

    Type convertedPayloadType = typeConverter->convertType(successType);
    if (!convertedPayloadType)
        convertedPayloadType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    if (!llvm::isa<sir::PtrType>(convertedPayloadType))
        return Value();

    Value payloadView = rewriter.create<sir::BitcastOp>(loc, convertedPayloadType, payloadSlot);
    payloadView.getDefiningOp()->setAttr(
        kOraMaterializationKindAttr,
        StringAttr::get(ctx, kStorageMemRefViewKind));
    return payloadView;
}

static std::pair<Value, Value> splitPackedErrorUnionForStorage(
    PatternRewriter &rewriter,
    Location loc,
    Value packed)
{
    return ora::error_union_helpers::splitNarrowPackedCarrier(rewriter, loc, packed);
}

static FailureOr<std::pair<Value, Value>> getErrorUnionPartsForStorage(
    ora::SStoreOp op,
    ora::ErrorUnionType errorUnionType,
    PatternRewriter &rewriter)
{
    auto loc = op.getLoc();
    if (errorUnionStoragePayloadUsesPointer(errorUnionType.getSuccessType()) ||
        errorUnionStorageHasPayloadBearingErrors(errorUnionType, op.getOperation()))
    {
        return failure();
    }

    Value value = op.getValue();
    if (auto ok = value.getDefiningOp<ora::ErrorOkOp>())
        return std::pair<Value, Value>{constU256(rewriter, loc, 0), coerceToU256(rewriter, loc, ok.getValue())};

    if (auto err = value.getDefiningOp<ora::ErrorErrOp>())
        return std::pair<Value, Value>{constU256(rewriter, loc, 1), coerceToU256(rewriter, loc, err.getValue())};

    if (auto ret = value.getDefiningOp<ora::ErrorReturnOp>())
    {
        if (ret.getNumOperands() != 0)
            return failure();
        auto errorId = lookupErrorIdAttr(op.getOperation(), ret.getSymName());
        if (!errorId)
            return failure();
        return std::pair<Value, Value>{
            constU256(rewriter, loc, 1),
            storageErrorIdConst(rewriter, loc, errorId)};
    }

    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() == 2)
            return std::pair<Value, Value>{
                coerceToU256(rewriter, loc, cast.getOperand(0)),
                coerceToU256(rewriter, loc, cast.getOperand(1))};
        if (cast.getNumOperands() == 1)
            return splitPackedErrorUnionForStorage(
                rewriter, loc, coerceToU256(rewriter, loc, cast.getOperand(0)));
    }

    return failure();
}

static LogicalResult storePointerBackedErrorUnionCarrierToStorage(
    ora::SStoreOp op,
    ora::ErrorUnionType errorUnionType,
    ArrayRef<Value> operands,
    Value baseSlot,
    PatternRewriter &rewriter,
    const TypeConverter *typeConverter = nullptr)
{
    auto loc = op.getLoc();
    auto memrefType = llvm::dyn_cast<mlir::MemRefType>(errorUnionType.getSuccessType());
    const bool dynamicBytesPayload = llvm::isa<ora::StringType, ora::BytesType>(errorUnionType.getSuccessType());
    if (!memrefType && !dynamicBytesPayload)
        return rewriter.notifyMatchFailure(
            op, "Result storage with aggregate payload is not yet supported");

    Value payloadSlot = ora::adt_helpers::adtStoragePayloadSlot(rewriter, loc, baseSlot);

    if (auto ok = op.getValue().getDefiningOp<ora::ErrorOkOp>())
    {
        rewriter.create<sir::SStoreOp>(loc, baseSlot, constU256(rewriter, loc, 0));
        if (dynamicBytesPayload)
            return storeDynamicBytesValueToStorageRoot(
                op.getOperation(), ok.getValue(), payloadSlot, rewriter, typeConverter);
        return storeDynamicMemRefToStorageRoot(
            op.getOperation(), ok.getValue(), payloadSlot, memrefType, rewriter);
    }

    if (auto err = op.getValue().getDefiningOp<ora::ErrorErrOp>())
    {
        Value payload = coerceToU256(rewriter, loc, err.getValue());
        rewriter.create<sir::SStoreOp>(loc, baseSlot, constU256(rewriter, loc, 1));
        rewriter.create<sir::SStoreOp>(loc, payloadSlot, payload);
        rewriter.eraseOp(op);
        return success();
    }

    if (auto ret = op.getValue().getDefiningOp<ora::ErrorReturnOp>())
    {
        if (ret.getNumOperands() != 0)
            return failure();
        auto errorId = lookupErrorIdAttr(op.getOperation(), ret.getSymName());
        if (!errorId)
            return failure();
        rewriter.create<sir::SStoreOp>(loc, baseSlot, constU256(rewriter, loc, 1));
        rewriter.create<sir::SStoreOp>(loc, payloadSlot, storageErrorIdConst(rewriter, loc, errorId));
        rewriter.eraseOp(op);
        return success();
    }

    if (auto cast = op.getValue().getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if ((ora::hasMaterializationKind(cast, ora::mat_kind::kWideErrorUnionJoin) ||
             ora::hasMaterializationKind(cast, ora::mat_kind::kNormalizedErrorUnion)) &&
            cast.getNumOperands() == 2)
        {
            Value tag = ensureU256(
                rewriter, loc, op.getOperation(), cast.getOperand(0), "Result storage tag");
            if (!tag)
                return failure();

            uint64_t tagValue = 0;
            if (!getConstU64(tag, tagValue))
                return rewriter.notifyMatchFailure(
                    op, "dynamic Result storage tag for aggregate payload is not yet supported");

            rewriter.create<sir::SStoreOp>(loc, baseSlot, tag);
            if (tagValue == 0)
            {
                if (dynamicBytesPayload)
                    return storeDynamicBytesValueToStorageRoot(
                        op.getOperation(), cast.getOperand(1), payloadSlot, rewriter, typeConverter);
                return storeDynamicMemRefToStorageRoot(
                    op.getOperation(), cast.getOperand(1), payloadSlot, memrefType, rewriter);
            }
            if (tagValue == 1)
            {
                Value payload = coerceToU256(rewriter, loc, cast.getOperand(1));
                rewriter.create<sir::SStoreOp>(loc, payloadSlot, payload);
                rewriter.eraseOp(op);
                return success();
            }
        }
    }

    if (operands.size() == 2)
    {
        Value tag = ensureU256(rewriter, loc, op.getOperation(), operands[0], "Result storage tag");
        if (!tag)
            return failure();

        uint64_t tagValue = 0;
        if (!getConstU64(tag, tagValue))
            return rewriter.notifyMatchFailure(
                op, "dynamic Result storage tag for aggregate payload is not yet supported");

        rewriter.create<sir::SStoreOp>(loc, baseSlot, tag);
        if (tagValue == 0)
        {
            if (dynamicBytesPayload)
                return storeDynamicBytesValueToStorageRoot(
                    op.getOperation(), operands[1], payloadSlot, rewriter, typeConverter);
            return storeDynamicMemRefToStorageRoot(
                op.getOperation(), operands[1], payloadSlot, memrefType, rewriter);
        }
        if (tagValue == 1)
        {
            Value payload = coerceToU256(rewriter, loc, operands[1]);
            rewriter.create<sir::SStoreOp>(loc, payloadSlot, payload);
            rewriter.eraseOp(op);
            return success();
        }
    }

    return rewriter.notifyMatchFailure(
        op, "Result storage with aggregate payload must store a known Ok or Err carrier");
}

static LogicalResult storeErrorUnionCarrierToStorage(
    ora::SStoreOp op,
    ora::ErrorUnionType errorUnionType,
    ArrayRef<Value> operands,
    Value baseSlot,
    PatternRewriter &rewriter)
{
    auto loc = op.getLoc();
    if (errorUnionStorageHasPayloadBearingErrors(errorUnionType, op.getOperation()) ||
        errorUnionStoragePayloadUnsupported(errorUnionType.getSuccessType()))
    {
        return rewriter.notifyMatchFailure(
            op, "Result storage with aggregate payload is not yet supported");
    }
    if (errorUnionStoragePayloadUsesPointer(errorUnionType.getSuccessType()))
        return storePointerBackedErrorUnionCarrierToStorage(op, errorUnionType, operands, baseSlot, rewriter);

    Value tag;
    Value payload;
    if (operands.empty())
    {
        auto parts = getErrorUnionPartsForStorage(op, errorUnionType, rewriter);
        if (failed(parts))
        {
            return rewriter.notifyMatchFailure(
                op, "Result storage value must lower to one or two carrier words");
        }
        tag = parts->first;
        payload = parts->second;
    }
    else if (operands.size() == 2)
    {
        tag = ensureU256(rewriter, loc, op.getOperation(), operands[0], "Result storage tag");
        payload = ensureU256(rewriter, loc, op.getOperation(), operands[1], "Result storage payload");
        if (!tag || !payload)
            return failure();
    }
    else if (operands.size() == 1)
    {
        Value packed = ensureU256(rewriter, loc, op.getOperation(), operands[0], "Result storage packed value");
        if (!packed)
            return failure();
        auto parts = splitPackedErrorUnionForStorage(rewriter, loc, packed);
        tag = parts.first;
        payload = parts.second;
    }
    else
    {
        return rewriter.notifyMatchFailure(
            op, "Result storage value must lower to one or two carrier words");
    }

    ora::adt_helpers::storeAdtPartsToStorageRoot(rewriter, loc, baseSlot, tag, payload);
    rewriter.eraseOp(op);
    return success();
}

static LogicalResult storeConvertedErrorUnionCarrierToStorage(
    ora::SStoreOp op,
    ora::ErrorUnionType errorUnionType,
    ArrayRef<Value> operands,
    Value baseSlot,
    ConversionPatternRewriter &rewriter,
    const TypeConverter *typeConverter)
{
    if (errorUnionStorageHasPayloadBearingErrors(errorUnionType, op.getOperation()) ||
        errorUnionStoragePayloadUnsupported(errorUnionType.getSuccessType()))
    {
        return rewriter.notifyMatchFailure(
            op, "Result storage with aggregate payload is not yet supported");
    }

    if (errorUnionStoragePayloadUsesPointer(errorUnionType.getSuccessType()))
        return storePointerBackedErrorUnionCarrierToStorage(
            op, errorUnionType, operands, baseSlot, rewriter, typeConverter);

    return storeErrorUnionCarrierToStorage(op, errorUnionType, operands, baseSlot, rewriter);
}

LogicalResult NormalizeAdtSLoadOp::matchAndRewrite(
    ora::SLoadOp op,
    PatternRewriter &rewriter) const
{
    Type resultType = op.getResult().getType();
    if (auto errorUnionType = llvm::dyn_cast<ora::ErrorUnionType>(resultType))
    {
        if (errorUnionStorageHasPayloadBearingErrors(errorUnionType, op.getOperation()) ||
            errorUnionStoragePayloadUnsupported(errorUnionType.getSuccessType()))
        {
            return rewriter.notifyMatchFailure(
                op, "Result storage with aggregate payload is not yet supported");
        }

        auto slotIndexOpt = computeGlobalSlot(op.getGlobalName(), op.getOperation());
        if (!slotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for Result sload");

        Value slot = findOrCreateSlotConstant(op.getOperation(), *slotIndexOpt, op.getGlobalName(), rewriter);
        if (auto slotOp = slot.getDefiningOp<sir::ConstOp>())
            setResultName(slotOp, 0, ("slot_" + op.getGlobalName()).str());

        if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(errorUnionType.getSuccessType()))
        {
            auto u256 = sir::U256Type::get(rewriter.getContext());
            Value payloadSlot = ora::adt_helpers::adtStoragePayloadSlot(rewriter, op.getLoc(), slot);
            Value tag = rewriter.create<sir::SLoadOp>(op.getLoc(), u256, slot);
            Value payload = materializeStorageResultMemRefView(
                op.getLoc(), payloadSlot, memrefType, rewriter, nullptr);
            if (!payload)
                return rewriter.notifyMatchFailure(op, "failed to materialize Result storage memref payload view");
            Value normalized = ora::createMaterializationCast(
                rewriter, op.getLoc(), resultType, ValueRange{tag, payload}, ora::mat_kind::kNormalizedErrorUnion);
            rewriter.replaceOp(op, normalized);
            return success();
        }
        if (llvm::isa<ora::StringType, ora::BytesType>(errorUnionType.getSuccessType()))
        {
            auto u256 = sir::U256Type::get(rewriter.getContext());
            Value payloadSlot = ora::adt_helpers::adtStoragePayloadSlot(rewriter, op.getLoc(), slot);
            Value tag = rewriter.create<sir::SLoadOp>(op.getLoc(), u256, slot);
            Value payload = materializeStorageResultDynamicBytesView(
                op.getLoc(), payloadSlot, errorUnionType.getSuccessType(), rewriter, nullptr);
            if (!payload)
                return rewriter.notifyMatchFailure(op, "failed to materialize Result storage dynamic payload view");
            Value normalized = ora::createMaterializationCast(
                rewriter, op.getLoc(), resultType, ValueRange{tag, payload}, ora::mat_kind::kNormalizedErrorUnion);
            rewriter.replaceOp(op, normalized);
            return success();
        }

        auto [tag, payload] = ora::adt_helpers::loadAdtPartsFromStorageRoot(rewriter, op.getLoc(), slot);
        Value normalized = ora::createMaterializationCast(
            rewriter, op.getLoc(), resultType, ValueRange{tag, payload}, ora::mat_kind::kNormalizedErrorUnion);
        rewriter.replaceOp(op, normalized);
        return success();
    }

    if (!llvm::isa<ora::AdtType>(resultType))
        return failure();

    auto slotIndexOpt = computeGlobalSlot(op.getGlobalName(), op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for ADT sload");

    Value slot = findOrCreateSlotConstant(op.getOperation(), *slotIndexOpt, op.getGlobalName(), rewriter);
    if (auto slotOp = slot.getDefiningOp<sir::ConstOp>())
        setResultName(slotOp, 0, ("slot_" + op.getGlobalName()).str());

    auto loc = op.getLoc();
    auto [tag, payload] = ora::adt_helpers::loadAdtPartsFromStorageRoot(rewriter, loc, slot);
    Value normalized = ora::createMaterializationCast(
        rewriter, loc, resultType, ValueRange{tag, payload}, ora::mat_kind::kNormalizedAdt);
    rewriter.replaceOp(op, normalized);
    return success();
}

LogicalResult NormalizeAdtSStoreOp::matchAndRewrite(
    ora::SStoreOp op,
    PatternRewriter &rewriter) const
{
    if (auto errorUnionType = llvm::dyn_cast<ora::ErrorUnionType>(op.getValue().getType()))
    {
        auto slotIndexOpt = computeGlobalSlot(op.getGlobalName(), op.getOperation());
        if (!slotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for Result sstore");

        Value slot = findOrCreateSlotConstant(op.getOperation(), *slotIndexOpt, op.getGlobalName(), rewriter);
        if (auto slotOp = slot.getDefiningOp<sir::ConstOp>())
            setResultName(slotOp, 0, ("slot_" + op.getGlobalName()).str());

        return storeErrorUnionCarrierToStorage(op, errorUnionType, ArrayRef<Value>{}, slot, rewriter);
    }

    if (!llvm::isa<ora::AdtType>(op.getValue().getType()))
        return failure();

    auto slotIndexOpt = computeGlobalSlot(op.getGlobalName(), op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for ADT sstore");

    Value slot = findOrCreateSlotConstant(op.getOperation(), *slotIndexOpt, op.getGlobalName(), rewriter);
    if (auto slotOp = slot.getDefiningOp<sir::ConstOp>())
        setResultName(slotOp, 0, ("slot_" + op.getGlobalName()).str());

    return storeAdtCarrierToStorage(op, ArrayRef<Value>{}, slot, rewriter);
}

// -----------------------------------------------------------------------------
// Lower ora.sload → sir.sload
// -----------------------------------------------------------------------------
LogicalResult ConvertSLoadOp::matchAndRewrite(
    ora::SLoadOp op,
    typename ora::SLoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    LLVM_DEBUG(llvm::dbgs() << "[OraToSIR] ConvertSLoadOp: " << op.getGlobalName()
                           << " op=" << op.getOperation()
                           << " block=" << op.getOperation()->getBlock() << "\n");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    ctx->getOrLoadDialect<sir::SIRDialect>();
    StringRef globalName = op.getGlobalName();

    if (llvm::isa<mlir::RankedTensorType>(op.getResult().getType()) ||
        llvm::isa<mlir::MemRefType>(op.getResult().getType()))
    {
        auto slotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
        if (!slotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for array sload");
        uint64_t slotIndex = *slotIndexOpt;
        Value slotConst = findOrCreateSlotConstant(op.getOperation(), slotIndex, globalName, rewriter);
        if (auto slotOp = slotConst.getDefiningOp<sir::ConstOp>())
        {
            setResultName(slotOp, 0, ("slot_" + globalName).str());
        }
        if (llvm::isa<mlir::MemRefType>(op.getResult().getType()))
        {
            // Storage memref users consume this view in MemRef.cpp to
            // recover the concrete base slot before lowering load/store.
            auto slotView = rewriter.create<mlir::UnrealizedConversionCastOp>(
                loc, op.getResult().getType(), slotConst);
            rewriter.replaceOp(op, slotView.getResult(0));
            return success();
        }
        rewriter.replaceOp(op, slotConst);
        return success();
    }

    // If this is a map type, return the base slot handle (do not load storage).
    if (llvm::isa<ora::MapType>(op.getResult().getType()))
    {
        auto slotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
        if (!slotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for map sload");
        uint64_t slotIndex = *slotIndexOpt;
        Value slotConst = findOrCreateSlotConstant(op.getOperation(), slotIndex, globalName, rewriter);
        if (auto slotOp = slotConst.getDefiningOp<sir::ConstOp>())
        {
            setResultName(slotOp, 0, ("slot_" + globalName).str());
        }
        rewriter.replaceOp(op, slotConst);
        return success();
    }

    // Check if result type is dynamic bytes (string, bytes, or enum with string repr)
    Type resultType = op.getResult().getType();
    bool isDynamicBytes = llvm::isa<ora::StringType, ora::BytesType>(resultType);
    LLVM_DEBUG(llvm::dbgs() << "[OraToSIR]   resultType: " << resultType
                            << ", isDynamicBytes(initial): " << isDynamicBytes << "\n");

    // Check if enum type has string/bytes representation
    if (auto enumType = llvm::dyn_cast<ora::EnumType>(resultType))
    {
        Type reprType = enumType.getReprType();
        LLVM_DEBUG(llvm::dbgs() << "[OraToSIR]   enum reprType: " << reprType << "\n");
        if (llvm::isa<ora::StringType, ora::BytesType>(reprType))
        {
            isDynamicBytes = true;
        }
    }
    if (!isDynamicBytes)
    {
        if (auto opaque = llvm::dyn_cast<mlir::OpaqueType>(resultType))
        {
            if (opaque.getDialectNamespace() == "ora" &&
                (opaque.getTypeData() == "string" || opaque.getTypeData() == "bytes"))
            {
                isDynamicBytes = true;
            }
        }
    }

    // Compute storage slot index from ora.global operation
    auto slotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for sload");
    uint64_t slotIndex = *slotIndexOpt;
    DBG("  -> slot index: " << slotIndex);

    // Find or create slot constant (reuse if already exists in function)
    Value slotConst = findOrCreateSlotConstant(op.getOperation(), slotIndex, globalName, rewriter);
    // Set name: "slot_" + globalName
    if (auto slotOp = slotConst.getDefiningOp<sir::ConstOp>())
    {
        setResultName(slotOp, 0, ("slot_" + globalName).str());
    }

    auto u256 = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    if (llvm::isa<ora::AdtType>(resultType))
    {
        auto [tag, payload] = ora::adt_helpers::loadAdtPartsFromStorageRoot(rewriter, loc, slotConst);
        rewriter.replaceOp(op, ValueRange{tag, payload});
        return success();
    }

    if (auto errorUnionType = llvm::dyn_cast<ora::ErrorUnionType>(resultType))
    {
        if (errorUnionStorageHasPayloadBearingErrors(errorUnionType, op.getOperation()) ||
            errorUnionStoragePayloadUnsupported(errorUnionType.getSuccessType()))
        {
            return rewriter.notifyMatchFailure(
                op, "Result storage with aggregate payload is not yet supported");
        }

        if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(errorUnionType.getSuccessType()))
        {
            Value payloadSlot = ora::adt_helpers::adtStoragePayloadSlot(rewriter, loc, slotConst);
            Value tag = rewriter.create<sir::SLoadOp>(loc, u256, slotConst);
            Value payload = materializeStorageResultMemRefView(
                loc, payloadSlot, memrefType, rewriter, this->getTypeConverter());
            if (!payload)
                return rewriter.notifyMatchFailure(op, "failed to materialize Result storage memref payload view");
            Value normalized = ora::createMaterializationCast(
                rewriter, loc, resultType, ValueRange{tag, payload}, ora::mat_kind::kNormalizedErrorUnion);
            rewriter.replaceOp(op, normalized);
            return success();
        }
        if (llvm::isa<ora::StringType, ora::BytesType>(errorUnionType.getSuccessType()))
        {
            Value payloadSlot = ora::adt_helpers::adtStoragePayloadSlot(rewriter, loc, slotConst);
            Value tag = rewriter.create<sir::SLoadOp>(loc, u256, slotConst);
            Value payload = materializeStorageResultDynamicBytesView(
                loc, payloadSlot, errorUnionType.getSuccessType(), rewriter, this->getTypeConverter());
            if (!payload)
                return rewriter.notifyMatchFailure(op, "failed to materialize Result storage dynamic payload view");
            Value normalized = ora::createMaterializationCast(
                rewriter, loc, resultType, ValueRange{tag, payload}, ora::mat_kind::kNormalizedErrorUnion);
            rewriter.replaceOp(op, normalized);
            return success();
        }

        auto [tag, payload] = ora::adt_helpers::loadAdtPartsFromStorageRoot(rewriter, loc, slotConst);
        Value normalized = ora::createMaterializationCast(
            rewriter, loc, resultType, ValueRange{tag, payload}, ora::mat_kind::kNormalizedErrorUnion);
        rewriter.replaceOp(op, normalized);
        return success();
    }

    if (auto structType = llvm::dyn_cast<ora::StructType>(resultType))
    {
        FailureOr<Value> loaded = loadStructValueFromStorageRoot(
            op.getOperation(), loc, slotConst, structType, rewriter, this->getTypeConverter());
        if (failed(loaded))
            return rewriter.notifyMatchFailure(op, "invalid struct field attributes for storage sload");

        rewriter.replaceOp(op, *loaded);
        return success();
    }

    // Precompute converted type so we can treat pointer results as dynamic bytes.
    Type convertedResultType = this->getTypeConverter()->convertType(resultType);
    if (!convertedResultType)
    {
        convertedResultType = ptrType;
    }
    LLVM_DEBUG(llvm::dbgs() << "[OraToSIR]   convertedResultType: " << convertedResultType << "\n");
    if (llvm::isa<sir::PtrType>(convertedResultType))
    {
        isDynamicBytes = true;
    }
    // Fallback: if this is an Ora type that isn't a known scalar, treat as dynamic bytes.
    if (!isDynamicBytes && resultType.getDialect().getNamespace() == "ora" &&
        !llvm::isa<ora::IntegerType, ora::BoolType, ora::AddressType, ora::MapType, ora::StructType, ora::EnumType,
                   ora::MinValueType, ora::MaxValueType, ora::InRangeType, ora::ScaledType, ora::ExactType, ora::NonZeroAddressType>(resultType))
    {
        isDynamicBytes = true;
    }
    LLVM_DEBUG(llvm::dbgs() << "[OraToSIR]   isDynamicBytes(final): " << isDynamicBytes << "\n");

    if (isDynamicBytes)
    {
        LogicalResult loaded = replaceWithDynamicBytesLoadFromStorageRoot(
            op.getOperation(), slotConst, convertedResultType, rewriter);
        LLVM_DEBUG(llvm::dbgs() << "[OraToSIR]   replaced with dynamic bytes load\n");
        return loaded;
    }

    // Replace the ora.sload with sir.sload for scalar values
    // Get the converted result type from type converter (enum -> u256, etc.)
    if (!convertedResultType)
    {
        LLVM_DEBUG(llvm::dbgs() << "[OraToSIR]   failed to convert result type for scalar sload\n");
        return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    // SIR storage loads always produce u256; cast after loading if needed.
    Value loaded = rewriter.create<sir::SLoadOp>(loc, u256, slotConst);
    if (convertedResultType != u256)
    {
        loaded = rewriter.create<sir::BitcastOp>(loc, convertedResultType, loaded);
    }
    setResultName(loaded.getDefiningOp(), 0, "value");
    rewriter.replaceOp(op, loaded);

    LLVM_DEBUG(llvm::dbgs() << "[OraToSIR]   replaced with sir.sload\n");
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.sstore → sir.sstore
// -----------------------------------------------------------------------------
LogicalResult ConvertSStoreOp::matchAndRewrite(
    ora::SStoreOp op,
    typename ora::SStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    Value value = adaptor.getValue();
    StringRef globalName = op.getGlobalName();

    if (llvm::isa<mlir::RankedTensorType>(op.getValue().getType()) ||
        (llvm::isa<mlir::MemRefType>(op.getValue().getType()) &&
         llvm::cast<mlir::MemRefType>(op.getValue().getType()).hasStaticShape()))
    {
        // Storage array element writes are lowered from tensor.insert/memref.store
        // to sir.sstore. The enclosing whole-array ora.sstore is then a no-op.
        rewriter.eraseOp(op);
        return success();
    }

    auto valueMemRefType = llvm::dyn_cast<mlir::MemRefType>(op.getValue().getType());
    const bool isDynamicStorageMemRef = valueMemRefType && !valueMemRefType.hasStaticShape();
    const bool isDynamicBytes = llvm::isa<ora::StringType, ora::BytesType>(op.getValue().getType());

    // Compute storage slot index from ora.global operation
    auto slotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for sstore");
    uint64_t slotIndex = *slotIndexOpt;

    // Find or create slot constant (reuse if already exists in function)
    Value slot = findOrCreateSlotConstant(op.getOperation(), slotIndex, globalName, rewriter);
    // Set name: "slot_" + globalName
    if (auto slotOp = slot.getDefiningOp<sir::ConstOp>())
    {
        setResultName(slotOp, 0, ("slot_" + globalName).str());
    }

    Type u256Type = sir::U256Type::get(ctx);

    if (llvm::isa<ora::AdtType>(op.getValue().getType()))
    {
        SmallVector<Value, 4> operands(adaptor.getOperands().begin(), adaptor.getOperands().end());
        return storeAdtCarrierToStorage(op, operands, slot, rewriter);
    }

    if (auto errorUnionType = llvm::dyn_cast<ora::ErrorUnionType>(op.getValue().getType()))
    {
        SmallVector<Value, 4> operands(adaptor.getOperands().begin(), adaptor.getOperands().end());
        return storeConvertedErrorUnionCarrierToStorage(
            op, errorUnionType, operands, slot, rewriter, this->getTypeConverter());
    }

    if (auto structType = llvm::dyn_cast<ora::StructType>(op.getValue().getType()))
    {
        ArrayAttr fieldNamesAttr;
        ArrayAttr fieldTypesAttr;
        if (!getStructFieldAttrs(op.getOperation(), structType, fieldNamesAttr, fieldTypesAttr))
            return rewriter.notifyMatchFailure(op, "invalid struct field attributes for storage sstore");

        if (auto update = op.getValue().getDefiningOp<ora::StructFieldUpdateOp>())
        {
            size_t updatedFieldIndex = 0;
            auto fieldOffset = getStructFieldStorageOffset(
                op.getOperation(), structType, update.getFieldName(), &updatedFieldIndex);
            if (!fieldOffset)
                return rewriter.notifyMatchFailure(op, "unknown struct field in storage field update");

            Value fieldSlot = addStorageWordOffset(loc, slot, *fieldOffset, rewriter);

            Type fieldType = cast<TypeAttr>(fieldTypesAttr[updatedFieldIndex]).getValue();
            Value updatedValue = update.getValue();
            if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(fieldType))
            {
                if (failed(storeStructValueToStorageRoot(
                        op.getOperation(), loc, updatedValue, fieldSlot, nestedStructType, rewriter, this->getTypeConverter())))
                {
                    return rewriter.notifyMatchFailure(op, "invalid nested struct field update for storage sstore");
                }
                rewriter.eraseOp(op);
                return success();
            }
            if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
                fieldMemRefType && fieldMemRefType.hasStaticShape())
            {
                if (failed(copyStaticMemRefValueToStorageRoot(op.getOperation(), updatedValue, fieldSlot, fieldMemRefType, rewriter)))
                    return rewriter.notifyMatchFailure(op, "failed to store static storage memref update");
                rewriter.eraseOp(op);
                return success();
            }
            if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
                fieldMemRefType && !fieldMemRefType.hasStaticShape())
            {
                return storeDynamicMemRefToStorageRoot(op.getOperation(), updatedValue, fieldSlot, fieldMemRefType, rewriter);
            }

            if (!llvm::isa<sir::U256Type>(updatedValue.getType()))
            {
                Type convertedFieldType = this->getTypeConverter()->convertType(updatedValue.getType());
                if (convertedFieldType != updatedValue.getType() && llvm::isa<sir::U256Type>(convertedFieldType))
                    updatedValue = rewriter.create<sir::BitcastOp>(loc, convertedFieldType, updatedValue);
                else
                    updatedValue = rewriter.create<sir::BitcastOp>(loc, u256Type, updatedValue);
            }
            rewriter.create<sir::SStoreOp>(loc, fieldSlot, updatedValue);
            rewriter.eraseOp(op);
            return success();
        }

        if (failed(storeStructValueToStorageRoot(
                op.getOperation(), loc, value, slot, structType, rewriter, this->getTypeConverter())))
        {
            return rewriter.notifyMatchFailure(op, "invalid struct field values for storage sstore");
        }

        rewriter.eraseOp(op);
        return success();
    }

    if (isDynamicStorageMemRef)
        return storeDynamicMemRefToStorageRoot(op.getOperation(), value, slot, valueMemRefType, rewriter);

    if (isDynamicBytes)
        return storeDynamicBytesValueToStorageRoot(
            op.getOperation(), value, slot, rewriter, this->getTypeConverter());

    // Convert value to SIR u256 - ALL Ora types must become SIR u256
    Value convertedValue = value;
    if (!llvm::isa<sir::U256Type>(value.getType()))
    {
        if (llvm::isa<ora::IntegerType>(value.getType()))
        {
            // Direct Ora int -> SIR u256 conversion
            convertedValue = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
        }
        else
        {
            Type valueConverted = this->getTypeConverter()->convertType(value.getType());
            if (valueConverted != value.getType() && llvm::isa<sir::U256Type>(valueConverted))
            {
                convertedValue = rewriter.create<sir::BitcastOp>(loc, valueConverted, value);
            }
            else
            {
                // Force to u256
                convertedValue = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
            }
        }
    }

    rewriter.replaceOpWithNewOp<sir::SStoreOp>(op, slot, convertedValue);
    return success();
}

LogicalResult ConvertStorageDeriveOp::matchAndRewrite(
    ora::StorageDeriveOp op,
    typename ora::StorageDeriveOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    auto namespaceHashAttr = llvm::dyn_cast<IntegerAttr>(op.getNamespaceHash());
    if (!namespaceHashAttr)
        return rewriter.notifyMatchFailure(op, "storage.derive missing integer namespace_hash");

    ValueRange keys = adaptor.getKeys();

    // Domain-separate computed storage from all other physical storage layouts.
    // The preimage is always:
    //   [ORA_CST_V1, key_count, namespace_hash, key0, key1, ...]
    // including the zero-key case. Returning namespace_hash directly would put
    // computed storage in the same flat slot space as ordinary sstore globals.
    constexpr uint64_t kComputedStorageDomainPrefix = 0x4f72614353545631ULL; // "OraCSTV1"
    uint64_t wordCount = 3 + static_cast<uint64_t>(keys.size());
    Value byteLen = constU256(rewriter, loc, wordCount * 32);
    Value buffer = rewriter.create<sir::MallocOp>(loc, ptrType, byteLen);
    rewriter.create<sir::StoreOp>(loc, buffer, constU256(rewriter, loc, kComputedStorageDomainPrefix));
    Value keyCountPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, buffer, constU256(rewriter, loc, 32));
    rewriter.create<sir::StoreOp>(loc, keyCountPtr, constU256(rewriter, loc, static_cast<uint64_t>(keys.size())));
    Value namespacePtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, buffer, constU256(rewriter, loc, 64));
    rewriter.create<sir::StoreOp>(loc, namespacePtr, constU256(rewriter, loc, namespaceHashAttr));

    for (auto indexedKey : llvm::enumerate(keys))
    {
        Value keyWord = ensureU256(rewriter, loc, op.getOperation(), indexedKey.value(), "storage.derive key");
        if (!keyWord)
            return failure();
        Value offset = constU256(rewriter, loc, (indexedKey.index() + 3) * 32);
        Value ptr = rewriter.create<sir::AddPtrOp>(loc, ptrType, buffer, offset);
        rewriter.create<sir::StoreOp>(loc, ptr, keyWord);
    }

    Value slot = rewriter.create<sir::KeccakOp>(loc, u256Type, buffer, byteLen);
    rewriter.replaceOp(op, slot);
    return success();
}

LogicalResult ConvertStorageWordLoadOp::matchAndRewrite(
    ora::StorageWordLoadOp op,
    typename ora::StorageWordLoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    Value slot = ensureU256(rewriter, loc, op.getOperation(), adaptor.getSlot(), "storage.word_load slot");
    if (!slot)
        return failure();
    Value offset = ensureU256(rewriter, loc, op.getOperation(), adaptor.getOffset(), "storage.word_load offset");
    if (!offset)
        return failure();

    Value physicalSlot = rewriter.create<sir::AddOp>(loc, u256Type, slot, offset);
    Value wrapped = rewriter.create<sir::LtOp>(loc, u256Type, physicalSlot, slot);
    Value notWrapped = rewriter.create<sir::IsZeroOp>(loc, u256Type, wrapped);

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto okBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());
    auto revertBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());
    auto resultArg = afterBlock->addArgument(u256Type, loc);
    op.getResult().replaceAllUsesWith(resultArg);
    rewriter.eraseOp(op);

    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::CondBrOp>(loc, notWrapped, ValueRange{}, ValueRange{}, okBlock, revertBlock);

    rewriter.setInsertionPointToStart(okBlock);
    Value loaded = rewriter.create<sir::SLoadOp>(loc, u256Type, physicalSlot);
    rewriter.create<sir::BrOp>(loc, ValueRange{loaded}, afterBlock);

    rewriter.setInsertionPointToStart(revertBlock);
    emitEmptyRevert(rewriter, loc);
    return success();
}

LogicalResult ConvertStorageWordStoreOp::matchAndRewrite(
    ora::StorageWordStoreOp op,
    typename ora::StorageWordStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    Value slot = ensureU256(rewriter, loc, op.getOperation(), adaptor.getSlot(), "storage.word_store slot");
    if (!slot)
        return failure();
    Value offset = ensureU256(rewriter, loc, op.getOperation(), adaptor.getOffset(), "storage.word_store offset");
    if (!offset)
        return failure();
    Value value = ensureU256(rewriter, loc, op.getOperation(), adaptor.getValue(), "storage.word_store value");
    if (!value)
        return failure();

    Value physicalSlot = rewriter.create<sir::AddOp>(loc, u256Type, slot, offset);
    Value wrapped = rewriter.create<sir::LtOp>(loc, u256Type, physicalSlot, slot);
    Value notWrapped = rewriter.create<sir::IsZeroOp>(loc, u256Type, wrapped);

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto okBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());
    auto revertBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());
    rewriter.eraseOp(op);

    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::CondBrOp>(loc, notWrapped, ValueRange{}, ValueRange{}, okBlock, revertBlock);

    rewriter.setInsertionPointToStart(okBlock);
    rewriter.create<sir::SStoreOp>(loc, physicalSlot, value);
    rewriter.create<sir::BrOp>(loc, ValueRange{}, afterBlock);

    rewriter.setInsertionPointToStart(revertBlock);
    emitEmptyRevert(rewriter, loc);
    return success();
}

LogicalResult ConvertStorageRangeEraseOp::matchAndRewrite(
    ora::StorageRangeEraseOp op,
    typename ora::StorageRangeEraseOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    Value slot = ensureU256(rewriter, loc, op.getOperation(), adaptor.getSlot(), "storage.range_erase slot");
    if (!slot)
        return failure();
    auto wordCountAttr = op->getAttrOfType<IntegerAttr>("word_count");
    if (!wordCountAttr)
        return rewriter.notifyMatchFailure(op, "storage.range_erase missing bounded word_count attribute");
    if (wordCountAttr.getValue().isNegative())
        return rewriter.notifyMatchFailure(op, "storage.range_erase word_count must be non-negative");
    if (wordCountAttr.getValue().isZero())
    {
        rewriter.eraseOp(op);
        return success();
    }
    Value len = constU256(rewriter, loc, wordCountAttr);

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
    auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
    auto revertBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());

    rewriter.setInsertionPointToEnd(parentBlock);
    Value zero = constU256(rewriter, loc, 0);
    Value maxOffset = constU256(rewriter, loc, wordCountAttr.getValue() - 1);
    Value lastSlot = rewriter.create<sir::AddOp>(loc, u256Type, slot, maxOffset);
    Value wrapped = rewriter.create<sir::LtOp>(loc, u256Type, lastSlot, slot);
    Value notWrapped = rewriter.create<sir::IsZeroOp>(loc, u256Type, wrapped);
    rewriter.create<sir::CondBrOp>(loc, notWrapped, ValueRange{zero}, ValueRange{}, condBlock, revertBlock);

    rewriter.setInsertionPointToStart(revertBlock);
    emitEmptyRevert(rewriter, loc);

    rewriter.setInsertionPointToStart(condBlock);
    Value iv = condBlock->getArgument(0);
    Value hasElement = rewriter.create<sir::LtOp>(loc, u256Type, iv, len);
    rewriter.create<sir::CondBrOp>(loc, hasElement, ValueRange{iv}, ValueRange{}, bodyBlock, afterBlock);

    rewriter.setInsertionPointToStart(bodyBlock);
    Value bodyIv = bodyBlock->getArgument(0);
    Value physicalSlot = rewriter.create<sir::AddOp>(loc, u256Type, slot, bodyIv);
    Value storeZero = constU256(rewriter, loc, 0);
    rewriter.create<sir::SStoreOp>(loc, physicalSlot, storeZero);
    Value one = constU256(rewriter, loc, 1);
    Value nextIv = rewriter.create<sir::AddOp>(loc, u256Type, bodyIv, one);
    rewriter.create<sir::BrOp>(loc, ValueRange{nextIv}, condBlock);

    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.tload → sir.tload
// -----------------------------------------------------------------------------
LogicalResult ConvertTLoadOp::matchAndRewrite(
    ora::TLoadOp op,
    typename ora::TLoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256 = sir::U256Type::get(ctx);

    auto keyAttr = op->getAttrOfType<StringAttr>("key");
    if (!keyAttr)
    {
        return rewriter.notifyMatchFailure(op, "tload missing key attribute");
    }

    auto slotIndexOpt = computeGlobalSlot(keyAttr.getValue(), op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for tload");
    uint64_t slotIndex = *slotIndexOpt;
    Value slotConst = constU256(rewriter, loc, slotIndex);

    Value result = rewriter.create<sir::TLoadOp>(loc, u256, slotConst);
    rewriter.replaceOp(op, result);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.tstore → sir.tstore
// -----------------------------------------------------------------------------
LogicalResult ConvertTStoreOp::matchAndRewrite(
    ora::TStoreOp op,
    typename ora::TStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256 = sir::U256Type::get(ctx);

    auto keyAttr = op->getAttrOfType<StringAttr>("key");
    if (!keyAttr)
    {
        return rewriter.notifyMatchFailure(op, "tstore missing key attribute");
    }

    auto slotIndexOpt = computeGlobalSlot(keyAttr.getValue(), op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for tstore");
    uint64_t slotIndex = *slotIndexOpt;
    Value slotConst = constU256(rewriter, loc, slotIndex);

    Value value = adaptor.getValue();
    if (!llvm::isa<sir::U256Type>(value.getType()))
    {
        value = rewriter.create<sir::BitcastOp>(loc, u256, value);
    }

    rewriter.replaceOpWithNewOp<sir::TStoreOp>(op, slotConst, value);
    return success();
}

// Lock/unlock/guard use a tx-scoped "locked set" stored in TSTORE at key = LOCK_PREFIX + slot.
// Sensei text has no tstore.lock/unlock/guard; we expand to const/add/tstore/tload/cond_br/revert.
constexpr unsigned kLockPrefixBit = 255;
static llvm::APInt getLockPrefixAPInt()
{
    return llvm::APInt(256, 1).shl(kLockPrefixBit);
}

static llvm::StringRef rootFromPathKey(llvm::StringRef key)
{
    size_t dot = key.find('.');
    size_t bracket = key.find('[');
    size_t end = llvm::StringRef::npos;
    if (dot != llvm::StringRef::npos)
        end = dot;
    if (bracket != llvm::StringRef::npos && (end == llvm::StringRef::npos || bracket < end))
        end = bracket;
    return end == llvm::StringRef::npos ? key : key.take_front(end);
}

static bool keyIsIndexed(llvm::StringRef key)
{
    return key.find('[') != llvm::StringRef::npos;
}

static Value deriveMapElementSlot(
    Location loc,
    ConversionPatternRewriter &rewriter,
    Value key,
    Value mapSlot,
    Type u256Type,
    Type ptrType)
{
    Value keyU256 = key;
    if (!llvm::isa<sir::U256Type>(keyU256.getType()))
        keyU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, keyU256);

    Value size64 = constU256(rewriter, loc, 64);
    Value slotKey = rewriter.create<sir::MallocOp>(loc, ptrType, size64);
    rewriter.create<sir::StoreOp>(loc, slotKey, keyU256);

    Value offset32 = constU256(rewriter, loc, 32);
    Value slotKeyPlus32 = rewriter.create<sir::AddPtrOp>(loc, ptrType, slotKey, offset32);
    rewriter.create<sir::StoreOp>(loc, slotKeyPlus32, mapSlot);

    return rewriter.create<sir::KeccakOp>(loc, u256Type, slotKey, size64);
}

static llvm::StringRef getGlobalNameFromMapOperand(mlir::Value mapOperand, mlir::Operation *currentOp);

static bool isSupportedResourceCarrier(Type type)
{
    if (auto oraInt = llvm::dyn_cast<ora::IntegerType>(type))
        return oraInt.getWidth() == 256;
    if (auto builtinInt = llvm::dyn_cast<mlir::IntegerType>(type))
        return builtinInt.getWidth() == 256;
    return false;
}

static Block *emitResourceRevertUnless(Location loc, ConversionPatternRewriter &rewriter, Value condition)
{
    Block *guardBlock = rewriter.getInsertionBlock();
    Region *parentRegion = guardBlock->getParent();
    Block *afterBlock = rewriter.splitBlock(guardBlock, rewriter.getInsertionPoint());

    Block *revertBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());
    rewriter.setInsertionPointToStart(revertBlock);
    emitEmptyRevert(rewriter, loc);

    rewriter.setInsertionPointToEnd(guardBlock);
    rewriter.create<sir::CondBrOp>(loc, condition, ValueRange{}, ValueRange{}, afterBlock, revertBlock);
    rewriter.setInsertionPointToStart(afterBlock);
    return afterBlock;
}

static void emitResourceUnsignedBalanceGuard(
    Location loc,
    ConversionPatternRewriter &rewriter,
    Value current,
    Value amount,
    Type u256Type)
{
    Value underflow = rewriter.create<sir::LtOp>(loc, u256Type, current, amount);
    Value ok = rewriter.create<sir::IsZeroOp>(loc, u256Type, underflow);
    emitResourceRevertUnless(loc, rewriter, ok);
}

static void emitResourceUnsignedAddGuard(
    Location loc,
    ConversionPatternRewriter &rewriter,
    Value updated,
    Value previous,
    Type u256Type)
{
    Value overflow = rewriter.create<sir::LtOp>(loc, u256Type, updated, previous);
    Value ok = rewriter.create<sir::IsZeroOp>(loc, u256Type, overflow);
    emitResourceRevertUnless(loc, rewriter, ok);
}

static FailureOr<Value> deriveResourcePlaceSlot(
    Operation *op,
    OperandRange originalPlace,
    ValueRange convertedPlace,
    Type carrierType,
    ConversionPatternRewriter &rewriter)
{
    if (originalPlace.size() != convertedPlace.size() || originalPlace.empty())
        return rewriter.notifyMatchFailure(op, "resource place must be a storage root or a map root plus at least one key");

    auto loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    llvm::StringRef globalName = getGlobalNameFromMapOperand(originalPlace.front(), op);
    if (globalName.empty())
        return rewriter.notifyMatchFailure(op, "could not extract resource place root name");

    auto slotIndex = computeGlobalSlot(globalName, op);
    if (!slotIndex)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for resource place root");

    Value mapSlot = findOrCreateSlotConstant(op, *slotIndex, globalName, rewriter);
    if (auto slotOp = mapSlot.getDefiningOp<sir::ConstOp>())
        setResultName(slotOp, 0, ("slot_" + globalName).str());

    if (originalPlace.size() == 1)
    {
        if (originalPlace.front().getType() != carrierType)
            return rewriter.notifyMatchFailure(op, "direct resource place type does not match carrier type");
        return mapSlot;
    }

    auto mapType = llvm::dyn_cast<ora::MapType>(originalPlace.front().getType());
    if (!mapType)
        return rewriter.notifyMatchFailure(op, "resource place root is not a map");

    Value slot = mapSlot;
    for (unsigned i = 1; i < originalPlace.size(); ++i)
    {
        Type originalKeyType = mapType.getKeyType();
        Value key = convertedPlace[i];
        FailureOr<Value> materializedKey = materializeStorageMapKey(op, key, originalKeyType, rewriter, "resource place key");
        if (failed(materializedKey))
            return failure();

        slot = deriveMapElementSlot(loc, rewriter, *materializedKey, slot, u256Type, ptrType);
        if (!globalName.empty() && i == 1)
        {
            if (auto hashOp = slot.getDefiningOp())
                setResultName(hashOp, 0, ("hash_" + globalName).str());
        }

        Type valueType = mapType.getValueType();
        if (i + 1 < originalPlace.size())
        {
            mapType = llvm::dyn_cast<ora::MapType>(valueType);
            if (!mapType)
                return rewriter.notifyMatchFailure(op, "resource place has extra keys after scalar map value");
        }
    }

    return slot;
}

static bool resourcePlacesHaveDistinctRoots(
    Operation *op,
    OperandRange sourcePlace,
    OperandRange destinationPlace)
{
    if (sourcePlace.empty() || destinationPlace.empty())
        return false;

    llvm::StringRef sourceRoot = getGlobalNameFromMapOperand(sourcePlace.front(), op);
    llvm::StringRef destinationRoot = getGlobalNameFromMapOperand(destinationPlace.front(), op);
    return !sourceRoot.empty() && !destinationRoot.empty() && sourceRoot != destinationRoot;
}

static LogicalResult lowerResourceCreateOrDestroy(
    Operation *op,
    OperandRange originalPlace,
    ValueRange convertedPlace,
    Value amount,
    Type carrierType,
    bool isCreate,
    ConversionPatternRewriter &rewriter)
{
    if (!isSupportedResourceCarrier(carrierType))
        return rewriter.notifyMatchFailure(op, "resource lowering currently supports only 256-bit integer carriers");

    auto loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    Value amountU256 = ensureU256(rewriter, loc, op, amount, "resource amount");
    if (!amountU256)
        return failure();

    FailureOr<Value> slot = deriveResourcePlaceSlot(op, originalPlace, convertedPlace, carrierType, rewriter);
    if (failed(slot))
        return failure();

    rewriter.setInsertionPoint(op);
    Value current = rewriter.create<sir::SLoadOp>(loc, u256Type, *slot);
    if (isCreate)
    {
        Value updated = rewriter.create<sir::AddOp>(loc, u256Type, current, amountU256);
        emitResourceUnsignedAddGuard(loc, rewriter, updated, current, u256Type);
        rewriter.create<sir::SStoreOp>(loc, *slot, updated);
    }
    else
    {
        emitResourceUnsignedBalanceGuard(loc, rewriter, current, amountU256, u256Type);
        Value updated = rewriter.create<sir::SubOp>(loc, u256Type, current, amountU256);
        rewriter.create<sir::SStoreOp>(loc, *slot, updated);
    }

    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertResourceCreateOp::matchAndRewrite(
    ora::CreateOp op,
    typename ora::CreateOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerResourceCreateOrDestroy(
        op.getOperation(),
        op.getPlace(),
        adaptor.getPlace(),
        adaptor.getAmount(),
        op.getCarrierType(),
        /*isCreate=*/true,
        rewriter);
}

LogicalResult ConvertResourceDestroyOp::matchAndRewrite(
    ora::DestroyOp op,
    typename ora::DestroyOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerResourceCreateOrDestroy(
        op.getOperation(),
        op.getPlace(),
        adaptor.getPlace(),
        adaptor.getAmount(),
        op.getCarrierType(),
        /*isCreate=*/false,
        rewriter);
}

static void emitResourceMoveSamePlace(
    Location loc,
    ConversionPatternRewriter &rewriter,
    Value slot,
    Value amount,
    Type u256Type,
    Block *mergeBlock)
{
    Value current = rewriter.create<sir::SLoadOp>(loc, u256Type, slot);
    emitResourceUnsignedBalanceGuard(loc, rewriter, current, amount, u256Type);
    if (mergeBlock)
        rewriter.create<sir::BrOp>(loc, ValueRange{}, mergeBlock);
}

static void emitResourceMoveDistinctPlaces(
    Location loc,
    ConversionPatternRewriter &rewriter,
    Value sourceSlot,
    Value destinationSlot,
    Value amount,
    Type u256Type,
    Block *mergeBlock)
{
    Value sourceCurrent = rewriter.create<sir::SLoadOp>(loc, u256Type, sourceSlot);
    Value destinationCurrent = rewriter.create<sir::SLoadOp>(loc, u256Type, destinationSlot);

    emitResourceUnsignedBalanceGuard(loc, rewriter, sourceCurrent, amount, u256Type);
    Value sourceUpdated = rewriter.create<sir::SubOp>(loc, u256Type, sourceCurrent, amount);

    Value destinationUpdated = rewriter.create<sir::AddOp>(loc, u256Type, destinationCurrent, amount);
    emitResourceUnsignedAddGuard(loc, rewriter, destinationUpdated, destinationCurrent, u256Type);

    rewriter.create<sir::SStoreOp>(loc, sourceSlot, sourceUpdated);
    rewriter.create<sir::SStoreOp>(loc, destinationSlot, destinationUpdated);
    if (mergeBlock)
        rewriter.create<sir::BrOp>(loc, ValueRange{}, mergeBlock);
}

LogicalResult ConvertResourceMoveOp::matchAndRewrite(
    ora::MoveOp op,
    typename ora::MoveOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    if (!isSupportedResourceCarrier(op.getCarrierType()))
        return rewriter.notifyMatchFailure(op, "resource lowering currently supports only 256-bit integer carriers");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    Value amountU256 = ensureU256(rewriter, loc, op.getOperation(), adaptor.getAmount(), "resource amount");
    if (!amountU256)
        return failure();

    FailureOr<Value> sourceSlot = deriveResourcePlaceSlot(op.getOperation(), op.getSourcePlace(), adaptor.getSourcePlace(), op.getCarrierType(), rewriter);
    if (failed(sourceSlot))
        return failure();

    FailureOr<Value> destinationSlot = deriveResourcePlaceSlot(op.getOperation(), op.getDestinationPlace(), adaptor.getDestinationPlace(), op.getCarrierType(), rewriter);
    if (failed(destinationSlot))
        return failure();

    rewriter.setInsertionPoint(op);
    if (resourcePlacesHaveDistinctRoots(op.getOperation(), op.getSourcePlace(), op.getDestinationPlace()))
    {
        emitResourceMoveDistinctPlaces(loc, rewriter, *sourceSlot, *destinationSlot, amountU256, u256Type, nullptr);
        rewriter.eraseOp(op);
        return success();
    }

    Value sameSlot = rewriter.create<sir::EqOp>(loc, u256Type, *sourceSlot, *destinationSlot);
    Block *branchBlock = op->getBlock();
    Region *parentRegion = branchBlock->getParent();
    Block *mergeBlock = rewriter.splitBlock(branchBlock, Block::iterator(op));
    Block *sameBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
    Block *distinctBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());

    rewriter.setInsertionPointToEnd(branchBlock);
    rewriter.create<sir::CondBrOp>(loc, sameSlot, ValueRange{}, ValueRange{}, sameBlock, distinctBlock);

    rewriter.setInsertionPointToStart(sameBlock);
    emitResourceMoveSamePlace(loc, rewriter, *sourceSlot, amountU256, u256Type, mergeBlock);

    rewriter.setInsertionPointToStart(distinctBlock);
    emitResourceMoveDistinctPlaces(loc, rewriter, *sourceSlot, *destinationSlot, amountU256, u256Type, mergeBlock);

    rewriter.eraseOp(op);
    return success();
}

static Block *emitLockedKeyRevertGuard(
    Operation *op,
    Location loc,
    ConversionPatternRewriter &rewriter,
    Value lockKey,
    Type u256,
    Type ptrType)
{
    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    Block *afterBlock = rewriter.splitBlock(parentBlock, std::next(Block::iterator(op)));

    Block *revertBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());
    rewriter.setInsertionPointToStart(revertBlock);
    Value zeroU256 = constU256(rewriter, loc, 0);
    Value zeroPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, zeroU256);
    Value zeroLen = constU256(rewriter, loc, 0);
    rewriter.create<sir::RevertOp>(loc, zeroPtr, zeroLen);

    rewriter.setInsertionPoint(op);
    Value val = rewriter.create<sir::TLoadOp>(loc, u256, lockKey);
    rewriter.create<sir::CondBrOp>(loc, val, ValueRange{}, ValueRange{}, revertBlock, afterBlock);
    return afterBlock;
}

// -----------------------------------------------------------------------------
// Lower ora.tstore.guard -> key = LOCK_PREFIX+slot; if TLOAD(key) != 0 then REVERT(0,0)
// -----------------------------------------------------------------------------
LogicalResult ConvertTStoreGuardOp::matchAndRewrite(
    ora::TStoreGuardOp op,
    typename ora::TStoreGuardOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256 = sir::U256Type::get(ctx);

    auto keyAttr = op->getAttrOfType<StringAttr>("key");
    if (!keyAttr)
        return rewriter.notifyMatchFailure(op, "tstore.guard missing key");
    llvm::StringRef key = keyAttr.getValue();
    llvm::StringRef root = rootFromPathKey(key);
    auto slotIndexOpt = computeGlobalSlot(root, op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing slot for tstore.guard key");
    Value slotBase = constU256(rewriter, loc, *slotIndexOpt);
    auto ptrType = sir::PtrType::get(ctx, 1);
    Value slot = slotBase;
    if (keyIsIndexed(key))
    {
        slot = deriveMapElementSlot(loc, rewriter, adaptor.getResource(), slotBase, u256, ptrType);
    }

    rewriter.setInsertionPoint(op);
    Value lockPrefix = constU256(rewriter, loc, getLockPrefixAPInt());
    Value lockKey = rewriter.create<sir::AddOp>(loc, u256, lockPrefix, slot);
    emitLockedKeyRevertGuard(op.getOperation(), loc, rewriter, lockKey, u256, ptrType);
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.lock -> if TLOAD(LOCK_PREFIX+slot) != 0 then REVERT(0,0), else TSTORE(..., 1)
// -----------------------------------------------------------------------------
LogicalResult ConvertLockOp::matchAndRewrite(
    ora::LockOp op,
    typename ora::LockOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256 = sir::U256Type::get(ctx);

    auto keyAttr = op->getAttrOfType<StringAttr>("key");
    if (!keyAttr)
        return rewriter.notifyMatchFailure(op, "ora.lock missing key attribute");
    llvm::StringRef key = keyAttr.getValue();
    llvm::StringRef root = rootFromPathKey(key);
    auto slotIndexOpt = computeGlobalSlot(root, op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing slot for lock key");
    Value slotBase = constU256(rewriter, loc, *slotIndexOpt);
    auto ptrType = sir::PtrType::get(ctx, 1);
    Value slot = slotBase;
    if (keyIsIndexed(key))
    {
        slot = deriveMapElementSlot(loc, rewriter, adaptor.getResource(), slotBase, u256, ptrType);
    }

    rewriter.setInsertionPoint(op);
    Value lockPrefix = constU256(rewriter, loc, getLockPrefixAPInt());
    Value lockKey = rewriter.create<sir::AddOp>(loc, u256, lockPrefix, slot);
    Block *afterBlock = emitLockedKeyRevertGuard(op.getOperation(), loc, rewriter, lockKey, u256, ptrType);
    rewriter.setInsertionPointToStart(afterBlock);
    Value one = constU256(rewriter, loc, 1);
    rewriter.create<sir::TStoreOp>(loc, lockKey, one);
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.unlock -> TSTORE(LOCK_PREFIX+slot, 0)
// -----------------------------------------------------------------------------
LogicalResult ConvertUnlockOp::matchAndRewrite(
    ora::UnlockOp op,
    typename ora::UnlockOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256 = sir::U256Type::get(ctx);

    auto keyAttr = op->getAttrOfType<StringAttr>("key");
    if (!keyAttr)
        return rewriter.notifyMatchFailure(op, "ora.unlock missing key attribute");
    llvm::StringRef key = keyAttr.getValue();
    llvm::StringRef root = rootFromPathKey(key);
    auto slotIndexOpt = computeGlobalSlot(root, op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing slot for unlock key");
    Value slotBase = constU256(rewriter, loc, *slotIndexOpt);
    auto ptrType = sir::PtrType::get(ctx, 1);
    Value slot = slotBase;
    if (keyIsIndexed(key))
    {
        slot = deriveMapElementSlot(loc, rewriter, adaptor.getResource(), slotBase, u256, ptrType);
    }

    rewriter.setInsertionPoint(op);
    Value lockPrefix = constU256(rewriter, loc, getLockPrefixAPInt());
    Value lockKey = rewriter.create<sir::AddOp>(loc, u256, lockPrefix, slot);
    Value zero = constU256(rewriter, loc, 0);
    rewriter.create<sir::TStoreOp>(loc, lockKey, zero);
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.global - convert type attribute from ora.int to u256
// Also assigns sequential slot indices to globals
// -----------------------------------------------------------------------------
LogicalResult ConvertGlobalOp::matchAndRewrite(
    ora::GlobalOp op,
    typename ora::GlobalOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    (void)adaptor;
    // Contract-level metadata for globals may be absent in syntax-only samples.
    // Globals are compile-time declarations only; lowering can safely erase them.
    // If metadata exists and is inconsistent, fail closed: storage layout is
    // part of the verifier/codegen soundness boundary.
    auto module = op->getParentOfType<ModuleOp>();
    const bool requireSlotMetadata = module && module->hasAttr("ora.global_slots_built");
    auto nameAttr = op->getAttrOfType<StringAttr>("sym_name");
    if (auto slotAttr = op->getAttrOfType<IntegerAttr>("ora.slot_index"))
    {
        if (nameAttr)
        {
            if (module)
            {
                bool ambiguousName = false;
                if (auto ambiguousAttr = module->getAttrOfType<ArrayAttr>("ora.global_slot_ambiguous_names"))
                {
                    for (Attribute entry : ambiguousAttr)
                    {
                        if (auto entryName = llvm::dyn_cast<StringAttr>(entry))
                        {
                            if (entryName.getValue() == nameAttr.getValue())
                            {
                                ambiguousName = true;
                                break;
                            }
                        }
                    }
                }
                if (ambiguousName)
                {
                    rewriter.eraseOp(op);
                    return success();
                }

                if (auto slotsAttr = module->getAttrOfType<DictionaryAttr>("ora.global_slots"))
                {
                    auto entry = slotsAttr.get(nameAttr.getValue());
                    auto entryInt = llvm::dyn_cast_or_null<IntegerAttr>(entry);
                    if (!entryInt || entryInt.getUInt() != slotAttr.getUInt())
                    {
                        return rewriter.notifyMatchFailure(op, "ora.global slot metadata mismatch");
                    }
                }
                else if (requireSlotMetadata)
                {
                    return rewriter.notifyMatchFailure(op, "ora.global slot manifest missing");
                }
            }
        }
        else if (requireSlotMetadata)
        {
            return rewriter.notifyMatchFailure(op, "ora.global name metadata missing");
        }
    }
    else if (requireSlotMetadata)
    {
        return rewriter.notifyMatchFailure(op, "ora.global slot metadata missing");
    }

    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Helper: Extract global name from map operand (tensor from ora.sload)
// -----------------------------------------------------------------------------
static llvm::StringRef getGlobalNameFromMapOperand(mlir::Value mapOperand, mlir::Operation *currentOp)
{
    // First, try to find the defining operation
    mlir::Operation *definingOp = mapOperand.getDefiningOp();
    if (!definingOp)
        return llvm::StringRef();
    if (auto sloadOp = llvm::dyn_cast<ora::SLoadOp>(definingOp))
    {
        return sloadOp.getGlobalName();
    }
    // If it's a cast, try to follow it
    if (auto castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(definingOp))
    {
        auto inputs = castOp.getInputs();
        if (inputs.size() == 1)
        {
            mlir::Operation *inputOp = inputs[0].getDefiningOp();
            if (auto sloadOp = llvm::dyn_cast<ora::SLoadOp>(inputOp))
            {
                return sloadOp.getGlobalName();
            }
        }
    }

    // If not found, look backwards in the block for the most recent ora.sload
    // This handles the case where ora.sload was already converted to sir.sload
    if (currentOp)
    {
        mlir::Block *block = currentOp->getBlock();
        auto it = mlir::Block::iterator(currentOp);
        // Walk backwards from current operation
        while (it != block->begin())
        {
            --it;
            if (auto sloadOp = llvm::dyn_cast<ora::SLoadOp>(*it))
            {
                // Check if this sload produces a value of the same type as mapOperand
                // (or if it's a tensor type, which is what arrays use)
                if (llvm::isa<mlir::RankedTensorType>(sloadOp.getResult().getType()))
                {
                    return sloadOp.getGlobalName();
                }
            }
        }
    }

    return llvm::StringRef(); // Empty if not found
}

// -----------------------------------------------------------------------------
// Lower ora.map_get → sir.keccak256 + sir.sload
// Pattern: %slot_key = sir.malloc 64
//          sir.store %slot_key, %key
//          %slot_key_plus_32 = sir.addptr %slot_key, 32
//          sir.store %slot_key_plus_32, %mapSlot
//          %hash = sir.keccak256 %slot_key, 64
//          %result = sir.sload %hash
// -----------------------------------------------------------------------------
LogicalResult ConvertMapGetOp::matchAndRewrite(
    ora::MapGetOp op,
    typename ora::MapGetOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    LLVM_DEBUG(llvm::dbgs() << "[OraToSIR] ConvertMapGetOp: " << op->getName()
                           << " at " << op.getLoc() << "\n");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    Type expectedResultType = op.getResult().getType();
    const bool returnsMapHandle = llvm::isa<ora::MapType>(expectedResultType);
    const bool returnsDynamicBytes = llvm::isa<ora::StringType, ora::BytesType>(expectedResultType);
    auto memRefResultType = llvm::dyn_cast<mlir::MemRefType>(expectedResultType);
    auto structResultType = llvm::dyn_cast<ora::StructType>(expectedResultType);
    Type convertedResultType;
    Type storageMemRefResultType;
    Type dynamicBytesResultType;

    if (returnsDynamicBytes)
    {
        dynamicBytesResultType = this->getTypeConverter()->convertType(expectedResultType);
        if (!dynamicBytesResultType)
            return rewriter.notifyMatchFailure(op, "map_get dynamic bytes result type conversion failed");
        if (!llvm::isa<sir::PtrType>(dynamicBytesResultType))
            return rewriter.notifyMatchFailure(op, "map_get dynamic bytes result did not lower to pointer");
    }
    else if (memRefResultType)
    {
        storageMemRefResultType = this->getTypeConverter()->convertType(expectedResultType);
        if (!storageMemRefResultType || !llvm::isa<sir::PtrType>(storageMemRefResultType))
            storageMemRefResultType = ptrType;
    }
    else if (!returnsMapHandle && !structResultType)
    {
        convertedResultType = this->getTypeConverter()->convertType(expectedResultType);
        if (!convertedResultType)
            return rewriter.notifyMatchFailure(op, "map_get result type conversion failed");
        if (!llvm::isa<sir::U256Type>(convertedResultType))
            return rewriter.notifyMatchFailure(op, "map_get result type did not lower to SIR storage value");
    }

    // Get the map operand and key
    Value originalMapOperand = op.getMap();
    Value convertedMapOperand = adaptor.getMap();
    Value key = adaptor.getKey();
    Type originalKeyType = op.getKey().getType();

    FailureOr<Value> materializedKey = materializeStorageMapKey(
        op.getOperation(), key, originalKeyType, rewriter, "map_get key");
    if (failed(materializedKey))
        return failure();
    key = *materializedKey;
    Value keyForCache = key;

    Value mapSlot = Value();
    llvm::StringRef globalName;
    if (llvm::isa<sir::U256Type>(convertedMapOperand.getType()))
    {
        mapSlot = convertedMapOperand;
    }
    else
    {
        // Extract global name from original map operand (before conversion)
        globalName = getGlobalNameFromMapOperand(originalMapOperand, op.getOperation());
        if (globalName.empty())
        {
            DBG("ConvertMapGetOp: failed to find global name from map operand");
            return rewriter.notifyMatchFailure(op, "could not extract global name from map operand");
        }

        // Compute storage slot for the map/array from ora.global operation
        auto mapSlotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
        if (!mapSlotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for map_get");
        uint64_t mapSlotIndex = *mapSlotIndexOpt;
        mapSlot = findOrCreateSlotConstant(op.getOperation(), mapSlotIndex, globalName, rewriter);
        // Set name: "slot_" + globalName
        if (auto slotOp = mapSlot.getDefiningOp<sir::ConstOp>())
        {
            setResultName(slotOp, 0, ("slot_" + globalName).str());
        }
    }

    auto funcOp = op->getParentOfType<func::FuncOp>();
    Value hash = lookupCachedMapHash(*mapHashCache, funcOp, op.getOperation(), mapSlot, keyForCache);
    if (hash)
    {
        DBG("ConvertMapGetOp: reused cached keccak256 hash");
    }
    else
    {
        // Allocate 64 bytes for key + slot
        Value size64 = constU256(rewriter, loc, 64);
        Value slotKey = rewriter.create<sir::MallocOp>(loc, ptrType, size64);
        setResultName(slotKey.getDefiningOp(), 0, "ptr");

        // Store key at offset 0
        rewriter.create<sir::StoreOp>(loc, slotKey, key);

        // Store map slot at offset 32
        Value offset32 = constU256(rewriter, loc, 32);
        Value slotKeyPlus32 = rewriter.create<sir::AddPtrOp>(loc, ptrType, slotKey, offset32);
        setResultName(slotKeyPlus32.getDefiningOp(), 0, "ptr_off");
        rewriter.create<sir::StoreOp>(loc, slotKeyPlus32, mapSlot);

        // Compute keccak256 hash
        hash = rewriter.create<sir::KeccakOp>(loc, u256Type, slotKey, size64);
        if (!globalName.empty())
        {
            setResultName(hash.getDefiningOp(), 0, ("hash_" + globalName).str());
        }
        storeCachedMapHash(*mapHashCache, funcOp, mapSlot, keyForCache, hash);
    }

    // If the map value is another map, return the derived slot hash as a map handle
    if (returnsMapHandle)
    {
        rewriter.replaceOp(op, hash);
        DBG("ConvertMapGetOp: map-of-map, returning derived slot hash");
        return success();
    }

    if (returnsDynamicBytes)
    {
        DBG("ConvertMapGetOp: loading dynamic bytes map value from storage root");
        return replaceWithDynamicBytesLoadFromStorageRoot(
            op.getOperation(), hash, dynamicBytesResultType, rewriter);
    }

    if (memRefResultType)
    {
        (void)storageMemRefResultType;
        Value storageView = rewriter.create<mlir::UnrealizedConversionCastOp>(
            loc, expectedResultType, hash).getResult(0);
        storageView.getDefiningOp()->setAttr(
            kOraMaterializationKindAttr,
            StringAttr::get(ctx, kStorageMemRefViewKind));
        rewriter.replaceOp(op, storageView);
        DBG("ConvertMapGetOp: map memref value returned as storage view");
        return success();
    }

    // If the result type is a struct, handle it specially by loading multiple storage slots
    if (structResultType)
    {
        FailureOr<Value> loaded = loadStructValueFromStorageRoot(
            op.getOperation(), loc, hash, structResultType, rewriter, this->getTypeConverter());
        if (failed(loaded))
            return rewriter.notifyMatchFailure(op, "invalid struct field attributes for map_get");

        rewriter.replaceOp(op, *loaded);
        return success();
    }

    // SIR storage loads always produce u256; cast after loading if needed.
    Value result = rewriter.create<sir::SLoadOp>(loc, u256Type, hash);
    if (convertedResultType != u256Type)
    {
        result = rewriter.create<sir::BitcastOp>(loc, convertedResultType, result);
    }
    setResultName(result.getDefiningOp(), 0, "value");

    // Replace the map_get with the result
    rewriter.replaceOp(op, result);
    DBG("ConvertMapGetOp: replaced with keccak256 + sload");
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.map_store → sir.keccak256 + sir.sstore
// Pattern: %slot_key = sir.malloc 64
//          sir.store %slot_key, %key
//          %slot_key_plus_32 = sir.addptr %slot_key, 32
//          sir.store %slot_key_plus_32, %mapSlot
//          %hash = sir.keccak256 %slot_key, 64
//          sir.sstore %hash, %value
// -----------------------------------------------------------------------------
LogicalResult ConvertMapStoreOp::matchAndRewrite(
    ora::MapStoreOp op,
    typename ora::MapStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    LLVM_DEBUG(llvm::dbgs() << "[OraToSIR] ConvertMapStoreOp: " << op->getName()
                           << " at " << op.getLoc() << "\n");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    // Get the map operand, key, and value
    Value originalMapOperand = op.getMap();
    Value convertedMapOperand = adaptor.getMap();
    Value key = adaptor.getKey();
    Value value = adaptor.getValue();
    Type originalKeyType = op.getKey().getType();
    Type originalValueType = op.getValue().getType();

    // Normalize the key first. Value conversion is delayed until the final
    // scalar store path so aggregate values can expand into storage slots.
    FailureOr<Value> materializedKey = materializeStorageMapKey(
        op.getOperation(), key, originalKeyType, rewriter, "map_store key");
    if (failed(materializedKey))
        return failure();
    key = *materializedKey;
    Value keyForCache = key;

    if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(originalMapOperand.getType()))
    {
        Value baseSlot = convertedMapOperand;
        if (!llvm::isa<sir::U256Type>(baseSlot.getType()))
        {
            // Try to recover base slot from global name if tensor wasn't converted.
            llvm::StringRef globalName = getGlobalNameFromMapOperand(originalMapOperand, op.getOperation());
            if (globalName.empty())
            {
                // Look backwards for most recent ora.sload if needed.
                mlir::Block *block = op->getBlock();
                auto it = mlir::Block::iterator(op.getOperation());
                while (it != block->begin())
                {
                    --it;
                    if (auto sloadOp = llvm::dyn_cast<ora::SLoadOp>(*it))
                    {
                        globalName = sloadOp.getGlobalName();
                        break;
                    }
                }
            }
            if (globalName.empty())
            {
                return rewriter.notifyMatchFailure(op, "array base slot is not u256");
            }
            auto slotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
            if (!slotIndexOpt)
            {
                return rewriter.notifyMatchFailure(op, "missing ora.slot_index for array map_store");
            }
            uint64_t slotIndex = *slotIndexOpt;
            baseSlot = findOrCreateSlotConstant(op.getOperation(), slotIndex, globalName, rewriter);
            if (auto slotOp = baseSlot.getDefiningOp<sir::ConstOp>())
            {
                setResultName(slotOp, 0, ("slot_" + globalName).str());
            }
        }
        Value indexU256 = ensureU256(rewriter, loc, op.getOperation(), key, "array map_store index");
        if (!indexU256)
            return failure();
        uint64_t elemWords = getElementWordCount(op.getOperation(), tensorType.getElementType());
        Value slot = Value();
        if (tensorType.hasStaticShape())
        {
            if (elemWords != 1)
            {
                Value elemWordsConst = constU256(rewriter, loc, elemWords);
                Value offset = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, elemWordsConst);
                slot = rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, offset);
            }
            else
            {
                slot = rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, indexU256);
            }
        }
        else
        {
            Value size32 = constU256(rewriter, loc, 32);
            auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
            Value tmp = rewriter.create<sir::MallocOp>(loc, ptrType, size32);
            rewriter.create<sir::StoreOp>(loc, tmp, baseSlot);
            Value hash = rewriter.create<sir::KeccakOp>(loc, u256Type, tmp, size32);
            if (elemWords != 1)
            {
                Value elemWordsConst = constU256(rewriter, loc, elemWords);
                Value offset = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, elemWordsConst);
                slot = rewriter.create<sir::AddOp>(loc, u256Type, hash, offset);
            }
            else
            {
                slot = rewriter.create<sir::AddOp>(loc, u256Type, hash, indexU256);
            }
        }
        rewriter.replaceOpWithNewOp<sir::SStoreOp>(op, slot, value);
        return success();
    }

    Value mapSlot = Value();
    llvm::StringRef globalName;
    if (llvm::isa<sir::U256Type>(convertedMapOperand.getType()))
    {
        mapSlot = convertedMapOperand;
    }
    else
    {
        // Extract global name from original map operand (before conversion)
        globalName = getGlobalNameFromMapOperand(originalMapOperand, op.getOperation());
        DBG("ConvertMapStoreOp: globalName = " << (globalName.empty() ? "<empty>" : globalName.str()));
        if (globalName.empty())
        {
            DBG("ConvertMapStoreOp: failed to find global name from map operand, trying backwards search");
            // Try a simpler approach: look for the most recent ora.sload in the block
            mlir::Block *block = op->getBlock();
            auto it = mlir::Block::iterator(op.getOperation());
            while (it != block->begin())
            {
                --it;
                if (auto sloadOp = llvm::dyn_cast<ora::SLoadOp>(*it))
                {
                    globalName = sloadOp.getGlobalName();
                    DBG("ConvertMapStoreOp: found global name via backwards search: " << globalName);
                    break;
                }
            }
        }
        if (globalName.empty())
        {
            DBG("ConvertMapStoreOp: failed to find global name from map operand");
            return rewriter.notifyMatchFailure(op, "could not extract global name from map operand");
        }

        // Compute storage slot for the map/array from ora.global operation
        auto mapSlotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
        if (!mapSlotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for map_store");
        uint64_t mapSlotIndex = *mapSlotIndexOpt;
        mapSlot = findOrCreateSlotConstant(op.getOperation(), mapSlotIndex, globalName, rewriter);
        // Set name: "slot_" + globalName
        if (auto slotOp = mapSlot.getDefiningOp<sir::ConstOp>())
        {
            setResultName(slotOp, 0, ("slot_" + globalName).str());
        }
    }

    auto funcOp = op->getParentOfType<func::FuncOp>();
    Value hash = lookupCachedMapHash(*mapHashCache, funcOp, op.getOperation(), mapSlot, keyForCache);
    if (hash)
    {
        DBG("ConvertMapStoreOp: reused cached keccak256 hash");
    }
    else
    {
        // Allocate 64 bytes for key + slot
        Value size64 = constU256(rewriter, loc, 64);
        Value slotKey = rewriter.create<sir::MallocOp>(loc, ptrType, size64);
        setResultName(slotKey.getDefiningOp(), 0, "ptr");

        // Store key at offset 0
        rewriter.create<sir::StoreOp>(loc, slotKey, key);

        // Store map slot at offset 32
        Value offset32 = constU256(rewriter, loc, 32);
        Value slotKeyPlus32 = rewriter.create<sir::AddPtrOp>(loc, ptrType, slotKey, offset32);
        setResultName(slotKeyPlus32.getDefiningOp(), 0, "ptr_off");
        rewriter.create<sir::StoreOp>(loc, slotKeyPlus32, mapSlot);

        // Compute keccak256 hash
        hash = rewriter.create<sir::KeccakOp>(loc, u256Type, slotKey, size64);
        if (!globalName.empty())
        {
            setResultName(hash.getDefiningOp(), 0, ("hash_" + globalName).str());
        }
        storeCachedMapHash(*mapHashCache, funcOp, mapSlot, keyForCache, hash);
    }

    if (llvm::isa<ora::StringType, ora::BytesType>(originalValueType))
    {
        return storeDynamicBytesValueToStorageRoot(
            op.getOperation(), value, hash, rewriter, this->getTypeConverter());
    }

    if (auto valueMemRefType = llvm::dyn_cast<mlir::MemRefType>(originalValueType))
    {
        if (valueMemRefType.hasStaticShape())
        {
            if (failed(copyStaticMemRefValueToStorageRoot(op.getOperation(), value, hash, valueMemRefType, rewriter)))
                return failure();
            rewriter.eraseOp(op);
            return success();
        }

        if (Value sourceSlot = getStorageMemRefViewSlot(value))
        {
            if (sourceSlot == hash)
            {
                rewriter.eraseOp(op);
                return success();
            }
            return rewriter.notifyMatchFailure(op, "copying dynamic map values between storage roots is not yet supported");
        }
        return storeDynamicMemRefToStorageRoot(op.getOperation(), value, hash, valueMemRefType, rewriter);
    }

    if (auto structType = llvm::dyn_cast<ora::StructType>(originalValueType))
    {
        if (failed(storeStructValueToStorageRoot(
                op.getOperation(), loc, value, hash, structType, rewriter, this->getTypeConverter())))
        {
            return rewriter.notifyMatchFailure(op, "invalid struct field values for map_store");
        }

        rewriter.eraseOp(op);
        DBG("ConvertMapStoreOp: replaced struct with keccak256 + field sstores");
        return success();
    }

    Type convertedValueType = this->getTypeConverter()->convertType(originalValueType);
    if (!convertedValueType || !llvm::isa<sir::U256Type>(convertedValueType))
        return rewriter.notifyMatchFailure(op, "map_store value type did not lower to SIR storage value");

    if (!llvm::isa<sir::U256Type>(value.getType()))
    {
        value = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
    }

    // Store to storage using the hash
    rewriter.create<sir::SStoreOp>(loc, hash, value);

    // Erase the map_store operation (it has no results)
    rewriter.eraseOp(op);
    DBG("ConvertMapStoreOp: replaced with keccak256 + sstore");
    return success();
}

// -----------------------------------------------------------------------------
// Lower tensor.insert for storage arrays -> sir.sstore base_slot + index
// -----------------------------------------------------------------------------
LogicalResult ConvertTensorInsertOp::matchAndRewrite(
    mlir::tensor::InsertOp op,
    typename mlir::tensor::InsertOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(op.getDest().getType());
    if (!tensorType)
        return rewriter.notifyMatchFailure(op, "tensor.insert destination is not ranked tensor");
    if (static_cast<int64_t>(adaptor.getIndices().size()) != tensorType.getRank())
        return rewriter.notifyMatchFailure(op, "tensor.insert index count mismatch");

    Value base = adaptor.getDest();
    if (auto cast = base.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() == 1)
            base = cast.getOperand(0);
    }
    if (!llvm::isa<sir::U256Type>(base.getType()))
        return rewriter.notifyMatchFailure(op, "tensor.insert destination is not backed by storage slot");

    Value indexU256;
    if (tensorType.getRank() == 1)
    {
        indexU256 = ensureU256(rewriter, loc, op.getOperation(), adaptor.getIndices()[0], "tensor.insert index");
        if (!indexU256)
            return failure();
    }
    else
    {
        indexU256 = constU256(rewriter, loc, 0);

        auto shape = tensorType.getShape();
        int64_t stride = 1;
        for (int64_t i = tensorType.getRank() - 1; i >= 0; --i)
        {
            Value idx = ensureU256(rewriter, loc, op.getOperation(), adaptor.getIndices()[i], "tensor.insert index");
            if (!idx)
                return failure();
            Value strideConst = constU256(rewriter, loc, static_cast<uint64_t>(stride));
            Value scaled = rewriter.create<sir::MulOp>(loc, u256Type, idx, strideConst);
            indexU256 = rewriter.create<sir::AddOp>(loc, u256Type, indexU256, scaled);
            stride *= shape[i];
        }
    }

    uint64_t elemWords = getElementWordCount(op.getOperation(), tensorType.getElementType());
    Value slot = base;
    if (elemWords != 1)
    {
        Value elemWordsConst = constU256(rewriter, loc, elemWords);
        Value offset = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, elemWordsConst);
        slot = rewriter.create<sir::AddOp>(loc, u256Type, base, offset);
    }
    else
    {
        slot = rewriter.create<sir::AddOp>(loc, u256Type, base, indexU256);
    }

    if (auto structType = llvm::dyn_cast<ora::StructType>(tensorType.getElementType()))
    {
        if (failed(storeStructValueToStorageRoot(
                op.getOperation(), loc, adaptor.getScalar(), slot, structType, rewriter, this->getTypeConverter())))
        {
            return rewriter.notifyMatchFailure(op, "invalid struct element for tensor.insert");
        }

        // Preserve SSA flow; the enclosing ora.sstore(tensor, global) is a no-op.
        rewriter.replaceOp(op, adaptor.getDest());
        return success();
    }

    Value storedValue = ensureU256(rewriter, loc, op.getOperation(), adaptor.getScalar(), "tensor.insert scalar");
    if (!storedValue)
        return failure();
    rewriter.create<sir::SStoreOp>(loc, slot, storedValue);

    // Preserve SSA flow; the enclosing ora.sstore(tensor, global) is a no-op.
    rewriter.replaceOp(op, adaptor.getDest());
    return success();
}

// -----------------------------------------------------------------------------
// Lower tensor.extract for storage arrays -> sir.sload base_slot + index
// -----------------------------------------------------------------------------
LogicalResult ConvertTensorExtractOp::matchAndRewrite(
    mlir::tensor::ExtractOp op,
    typename mlir::tensor::ExtractOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(op.getTensor().getType());
    if (!tensorType)
        return rewriter.notifyMatchFailure(op, "tensor.extract without ranked tensor");
    Value base = adaptor.getTensor();
    if (!llvm::isa<sir::U256Type>(base.getType()))
        return rewriter.notifyMatchFailure(op, "array base is not u256");

    if (static_cast<int64_t>(adaptor.getIndices().size()) != tensorType.getRank())
        return rewriter.notifyMatchFailure(op, "index count does not match tensor rank");

    // Compute linearized index for multi-dim tensors (row-major).
    Value indexU256 = Value();
    if (tensorType.getRank() == 1)
    {
        indexU256 = ensureU256(rewriter, loc, op.getOperation(), adaptor.getIndices()[0], "tensor.extract index");
        if (!indexU256)
            return failure();
    }
    else
    {
        if (!tensorType.hasStaticShape())
            return rewriter.notifyMatchFailure(op, "non-static tensor shape not supported");

        Value linear = constU256(rewriter, loc, 0);

        auto shape = tensorType.getShape();
        int64_t stride = 1;
        for (int64_t i = tensorType.getRank() - 1; i >= 0; --i)
        {
            Value idx = ensureU256(rewriter, loc, op.getOperation(), adaptor.getIndices()[i], "tensor.extract index");
            if (!idx)
                return failure();
            Value strideConst = constU256(rewriter, loc, static_cast<uint64_t>(stride));
            Value scaled = rewriter.create<sir::MulOp>(loc, u256Type, idx, strideConst);
            linear = rewriter.create<sir::AddOp>(loc, u256Type, linear, scaled);
            stride *= shape[i];
        }
        indexU256 = linear;
    }
    uint64_t elemWords = getElementWordCount(op.getOperation(), tensorType.getElementType());

    Value loaded = Value();
    bool isStorageArray = false;
    if (auto def = op.getTensor().getDefiningOp())
    {
        if (llvm::isa<ora::SLoadOp>(def))
            isStorageArray = true;
    }

    if (tensorType.hasStaticShape() || isStorageArray)
    {
        Value baseSlot = base;
        if (!tensorType.hasStaticShape())
        {
            Value size32 = constU256(rewriter, loc, 32);
            auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
            Value tmp = rewriter.create<sir::MallocOp>(loc, ptrType, size32);
            rewriter.create<sir::StoreOp>(loc, tmp, baseSlot);
            baseSlot = rewriter.create<sir::KeccakOp>(loc, u256Type, tmp, size32);
        }
        Value slot = baseSlot;
        if (elemWords != 1)
        {
            Value elemWordsConst = constU256(rewriter, loc, elemWords);
            Value offset = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, elemWordsConst);
            slot = rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, offset);
        }
        else
        {
            slot = rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, indexU256);
        }
        loaded = rewriter.create<sir::SLoadOp>(loc, u256Type, slot);
    }
    else
    {
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        Value ptr = rewriter.create<sir::BitcastOp>(loc, ptrType, base);
        Value wordSize = constU256(rewriter, loc, 32);
        Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, ptr, wordSize);
        Value offsetBytes = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, wordSize);
        Value elemPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, dataPtr, offsetBytes);
        loaded = rewriter.create<sir::LoadOp>(loc, u256Type, elemPtr);
    }

    Type desiredType = u256Type;
    if (auto *tc = getTypeConverter())
    {
        if (Type converted = tc->convertType(op.getType()))
            desiredType = converted;
    }
    // Force address elements to remain as sir.u256 to avoid back-materialization.
    if (llvm::isa<ora::AddressType, ora::NonZeroAddressType>(op.getType()))
        desiredType = u256Type;
    if (desiredType != u256Type)
        loaded = rewriter.create<sir::BitcastOp>(loc, desiredType, loaded);

    rewriter.replaceOp(op, loaded);
    return success();
}

// -----------------------------------------------------------------------------
// Lower tensor.dim for arrays/slices
// -----------------------------------------------------------------------------
LogicalResult ConvertTensorDimOp::matchAndRewrite(
    mlir::tensor::DimOp op,
    typename mlir::tensor::DimOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(op.getSource().getType());
    if (!tensorType)
        return rewriter.notifyMatchFailure(op, "tensor.dim on non-ranked tensor");

    if (tensorType.hasStaticShape())
    {
        int64_t dim = tensorType.getDimSize(0);
        Value dimConst = constU256(rewriter, loc, static_cast<uint64_t>(dim));
        Value idx = buildIndexFromU256(rewriter, loc, op.getOperation(), dimConst);
        if (!idx)
            return failure();
        rewriter.replaceOp(op, idx);
        return success();
    }

    Value base = adaptor.getSource();
    if (!llvm::isa<sir::U256Type>(base.getType()))
        return rewriter.notifyMatchFailure(op, "slice base is not u256");
    auto u256Type = sir::U256Type::get(ctx);

    bool isStorageArray = false;
    if (auto def = op.getSource().getDefiningOp())
    {
        if (llvm::isa<ora::SLoadOp>(def))
            isStorageArray = true;
    }

    if (isStorageArray)
    {
        Value length = rewriter.create<sir::SLoadOp>(loc, u256Type, base);
        Value idx = buildIndexFromU256(rewriter, loc, op.getOperation(), length);
        if (!idx)
            return failure();
        rewriter.replaceOp(op, idx);
        return success();
    }

    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    Value ptr = rewriter.create<sir::BitcastOp>(loc, ptrType, base);
    Value length = rewriter.create<sir::LoadOp>(loc, u256Type, ptr);
    Value idx = buildIndexFromU256(rewriter, loc, op.getOperation(), length);
    if (!idx)
        return failure();
    rewriter.replaceOp(op, idx);
    return success();
}
