#include "patterns/MemRef.h"
#include "patterns/Naming.h"
#include "patterns/Storage.h"
#include "OraMaterializationKinds.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"
#include "OraDialect.h"

#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;

// Debug logging macro
#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

static constexpr llvm::StringLiteral kStorageMemRefViewKind{"storage_memref_view"};
static constexpr llvm::StringLiteral kStorageStructCarrierKind{"storage_struct_carrier"};
static constexpr llvm::StringLiteral kStorageStructViewFieldsAttr{"ora.storage_struct_view_fields"};

static std::map<Operation *, SIRNamingHelper> memrefHelperMap;
static SIRNamingHelper memrefFallbackHelper;

static bool storageStructCarrierPreservesField(Value carrier, size_t fieldIndex)
{
    bool changed = true;
    while (changed)
    {
        changed = false;
        if (auto bitcast = carrier.getDefiningOp<sir::BitcastOp>())
        {
            carrier = bitcast.getOperand();
            changed = true;
            continue;
        }
        if (auto cast = carrier.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (cast.getNumOperands() == 1)
            {
                carrier = cast.getOperand(0);
                changed = true;
            }
        }
    }
    Operation *definingOp = carrier.getDefiningOp();
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
}

static bool isNarrowErrorUnionType(Type type)
{
    auto errType = llvm::dyn_cast<mlir::ora::ErrorUnionType>(type);
    if (!errType)
        return false;
    auto successType = errType.getSuccessType();
    return llvm::isa<mlir::ora::IntegerType, mlir::IntegerType, mlir::NoneType, mlir::ora::AddressType, mlir::ora::NonZeroAddressType>(successType);
}

static bool hasForceWideErrorUnionAttr(Operation *op)
{
    if (!op)
        return false;
    if (auto attr = op->getAttrOfType<mlir::BoolAttr>("ora.force_wide_error_union"))
        return attr.getValue();
    if (auto func = op->getParentOfType<mlir::func::FuncOp>())
    {
        if (auto attr = func->getAttrOfType<mlir::BoolAttr>("ora.force_wide_error_union"))
            return attr.getValue();
    }
    return false;
}

static bool valueHasForceWideErrorUnion(Value value)
{
    if (!value)
        return false;
    if (Operation *def = value.getDefiningOp())
        return hasForceWideErrorUnionAttr(def);
    return false;
}

static bool isScalarErrorUnionMemRefCarrier(Type type)
{
    auto errType = llvm::dyn_cast<mlir::ora::ErrorUnionType>(type);
    if (!errType)
        return false;
    auto successType = errType.getSuccessType();
    return llvm::isa<mlir::IntegerType, mlir::ora::IntegerType, mlir::NoneType, mlir::ora::AddressType, mlir::ora::NonZeroAddressType>(successType);
}

static Value ensureU256Value(PatternRewriter &rewriter, Location loc, Value value)
{
    if (llvm::isa<sir::U256Type>(value.getType()))
        return value;
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    return rewriter.create<sir::BitcastOp>(loc, u256Type, value);
}

static Value unwrapIndexCastInput(Value value)
{
    while (true)
    {
        if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (cast.getNumOperands() == 1)
            {
                value = cast.getOperand(0);
                continue;
            }
        }
        if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>())
        {
            value = cast.getIn();
            continue;
        }
        if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
        {
            value = cast.getIn();
            continue;
        }
        break;
    }
    return value;
}

static ora::StructDeclOp findStructDeclForName(Operation *op, StringRef structName)
{
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
        return nullptr;

    ora::StructDeclOp structDecl = nullptr;
    module.walk([&](ora::StructDeclOp declOp)
                {
        auto declNameAttr = declOp->getAttrOfType<StringAttr>("sym_name");
        if (declNameAttr && declNameAttr.getValue() == structName)
        {
            structDecl = declOp;
            return WalkResult::interrupt();
        }
        return WalkResult::advance(); });
    return structDecl;
}

static bool getStructFieldAttrs(Operation *anchor,
                                ora::StructType structType,
                                ArrayAttr &fieldNamesAttr,
                                ArrayAttr &fieldTypesAttr)
{
    auto structDecl = findStructDeclForName(anchor, structType.getName());
    if (!structDecl)
        return false;
    fieldNamesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_names");
    fieldTypesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_types");
    return fieldNamesAttr && fieldTypesAttr && fieldNamesAttr.size() == fieldTypesAttr.size();
}

static uint64_t getMemRefElementWordCount(Operation *anchor, Type elementType)
{
    if (auto structType = llvm::dyn_cast<ora::StructType>(elementType))
    {
        ArrayAttr fieldNamesAttr;
        ArrayAttr fieldTypesAttr;
        if (getStructFieldAttrs(anchor, structType, fieldNamesAttr, fieldTypesAttr))
        {
            uint64_t words = 0;
            for (Attribute fieldTypeAttr : fieldTypesAttr)
            {
                Type fieldType = cast<TypeAttr>(fieldTypeAttr).getValue();
                words += getMemRefElementWordCount(anchor, fieldType);
            }
            return words == 0 ? 1 : words;
        }
    }

    if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(elementType);
        memrefType && !memrefType.hasStaticShape())
    {
        return 1;
    }

    // v0.1: word-aligned elements only.
    return 1;
}

static Value addStorageWordOffset(Location loc,
                                  Value slot,
                                  uint64_t offset,
                                  ConversionPatternRewriter &rewriter)
{
    if (offset == 0)
        return slot;

    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    Value offsetValue = rewriter.create<sir::ConstOp>(
        loc, u256Type, mlir::IntegerAttr::get(ui64Type, offset));
    return rewriter.create<sir::AddOp>(loc, u256Type, slot, offsetValue);
}

static Value subtractStorageWordOffset(Location loc,
                                       Value slot,
                                       uint64_t offset,
                                       ConversionPatternRewriter &rewriter)
{
    if (offset == 0)
        return slot;

    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    Value offsetValue = rewriter.create<sir::ConstOp>(
        loc, u256Type, mlir::IntegerAttr::get(ui64Type, offset));
    return rewriter.create<sir::SubOp>(loc, u256Type, slot, offsetValue);
}

static Value getSlotFromStorageFieldValue(Value value)
{
    while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() != 1)
            break;
        value = cast.getOperand(0);
    }

    if (auto bitcast = value.getDefiningOp<sir::BitcastOp>())
    {
        auto viewKind = bitcast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
        if (viewKind && viewKind.getValue() == kStorageMemRefViewKind &&
            llvm::isa<sir::U256Type>(bitcast.getOperand().getType()))
            return bitcast.getOperand();
    }

    if (auto sload = value.getDefiningOp<sir::SLoadOp>())
        return sload.getSlot();

    return Value();
}

static FailureOr<Value> loadStructValueFromStorageMemRefRoot(Operation *anchor,
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
            FailureOr<Value> nested = loadStructValueFromStorageMemRefRoot(
                anchor, loc, fieldSlot, nestedStructType, rewriter, typeConverter);
            if (failed(nested))
                return failure();
            fieldValue = *nested;
        }
        else
        {
            auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            const bool isDynamicMemRefField = fieldMemRefType && !fieldMemRefType.hasStaticShape();
            fieldValue = isDynamicMemRefField
                ? fieldSlot
                : rewriter.create<sir::SLoadOp>(loc, u256Type, fieldSlot).getResult();

            Type convertedFieldType = typeConverter ? typeConverter->convertType(fieldType) : Type();
            if (!convertedFieldType)
                convertedFieldType = u256Type;
            if (convertedFieldType != u256Type)
            {
                fieldValue = rewriter.create<sir::BitcastOp>(loc, convertedFieldType, fieldValue);
                if (isDynamicMemRefField)
                {
                    fieldValue.getDefiningOp()->setAttr(
                        kOraMaterializationKindAttr,
                        StringAttr::get(ctx, kStorageMemRefViewKind));
                }
            }
        }

        fieldValues.push_back(fieldValue);
        offset += getMemRefElementWordCount(anchor, fieldType);
    }

    auto structInitOp = rewriter.create<ora::StructInitOp>(loc, structType, fieldValues);
    structInitOp->setAttr(
        kOraMaterializationKindAttr,
        StringAttr::get(ctx, kStorageStructCarrierKind));
    return structInitOp.getResult();
}

static LogicalResult storeDynamicMemRefValueToStorageMemRefRoot(Operation *anchor,
                                                                Location loc,
                                                                Value value,
                                                                Value slot,
                                                                mlir::MemRefType memrefType,
                                                                ConversionPatternRewriter &rewriter)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    Value basePtr = value;
    if (!llvm::isa<sir::PtrType>(basePtr.getType()))
        basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, basePtr);

    Value length = rewriter.create<sir::LoadOp>(loc, u256Type, basePtr);
    rewriter.create<sir::SStoreOp>(loc, slot, length);

    uint64_t elemWords = getMemRefElementWordCount(anchor, memrefType.getElementType());
    Value writeCount = length;
    Value wordSize = rewriter.create<sir::ConstOp>(
        loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
    if (elemWords != 1)
    {
        Value elemWordsConst = rewriter.create<sir::ConstOp>(
            loc, u256Type, mlir::IntegerAttr::get(ui64Type, elemWords));
        writeCount = rewriter.create<sir::MulOp>(loc, u256Type, length, elemWordsConst);
    }

    Value tmp = rewriter.create<sir::MallocOp>(loc, ptrType, wordSize);
    rewriter.create<sir::StoreOp>(loc, tmp, slot);
    Value storageDataBase = rewriter.create<sir::KeccakOp>(loc, u256Type, tmp, wordSize);

    Value zero = rewriter.create<sir::ConstOp>(
        loc, u256Type, mlir::IntegerAttr::get(ui64Type, 0));
    Value one = rewriter.create<sir::ConstOp>(
        loc, u256Type, mlir::IntegerAttr::get(ui64Type, 1));

    Block *parentBlock = anchor->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(anchor));
    auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
    auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});

    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::BrOp>(loc, ValueRange{zero}, condBlock);

    rewriter.setInsertionPointToStart(condBlock);
    Value iv = condBlock->getArgument(0);
    Value lt = rewriter.create<sir::LtOp>(loc, u256Type, iv, writeCount);
    rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv}, ValueRange{}, bodyBlock, afterBlock);

    rewriter.setInsertionPointToStart(bodyBlock);
    Value wordIndex = bodyBlock->getArgument(0);
    Value wordSlot = rewriter.create<sir::AddOp>(loc, u256Type, storageDataBase, wordIndex);
    Value wordBytes = rewriter.create<sir::MulOp>(loc, u256Type, wordIndex, wordSize);
    Value dataOffset = rewriter.create<sir::AddOp>(loc, u256Type, wordBytes, wordSize);
    Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, dataOffset);
    Value wordValue = rewriter.create<sir::LoadOp>(loc, u256Type, dataPtr);
    rewriter.create<sir::SStoreOp>(loc, wordSlot, wordValue);
    Value next = rewriter.create<sir::AddOp>(loc, u256Type, wordIndex, one);
    rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

    rewriter.setInsertionPointToStart(afterBlock);
    return success();
}

static LogicalResult storeStructValueToStorageMemRefRoot(Operation *anchor,
                                                         Location loc,
                                                         Value value,
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

    if (auto update = value.getDefiningOp<ora::StructFieldUpdateOp>())
    {
        std::optional<size_t> updatedFieldIndex;
        for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
        {
            if (cast<StringAttr>(fieldNamesAttr[i]).getValue() == update.getFieldName())
            {
                updatedFieldIndex = i;
                break;
            }
        }
        if (!updatedFieldIndex)
            return failure();

        uint64_t fieldOffset = 0;
        for (size_t i = 0; i < *updatedFieldIndex; ++i)
            fieldOffset += getMemRefElementWordCount(anchor, cast<TypeAttr>(fieldTypesAttr[i]).getValue());
        Value fieldSlot = addStorageWordOffset(loc, slot, fieldOffset, rewriter);

        Type fieldType = cast<TypeAttr>(fieldTypesAttr[*updatedFieldIndex]).getValue();
        Value updatedValue = update.getValue();
        if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(fieldType))
        {
            return storeStructValueToStorageMemRefRoot(
                anchor, loc, updatedValue, fieldSlot, nestedStructType, rewriter, typeConverter);
        }
        if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            fieldMemRefType && !fieldMemRefType.hasStaticShape())
        {
            return storeDynamicMemRefValueToStorageMemRefRoot(
                anchor, loc, updatedValue, fieldSlot, fieldMemRefType, rewriter);
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

    uint64_t offset = 0;
    for (size_t i = 0; i < fieldValues.size(); ++i)
    {
        Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
        Value fieldSlot = addStorageWordOffset(loc, slot, offset, rewriter);
        Value stored = fieldValues[i];

        if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(fieldType))
        {
            if (failed(storeStructValueToStorageMemRefRoot(anchor, loc, stored, fieldSlot, nestedStructType, rewriter, typeConverter)))
                return failure();
            offset += getMemRefElementWordCount(anchor, fieldType);
            continue;
        }

        if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(fieldType);
            fieldMemRefType && !fieldMemRefType.hasStaticShape())
        {
            return failure();
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
        offset += getMemRefElementWordCount(anchor, fieldType);
    }

    return success();
}

static Value buildStaticStorageMemRefSlot(ConversionPatternRewriter &rewriter,
                                          Location loc,
                                          mlir::MemRefType memrefType,
                                          Value baseSlot,
                                          ValueRange indices);

static Value buildDynamicStorageMemRefSlot(ConversionPatternRewriter &rewriter,
                                           Location loc,
                                           Operation *anchor,
                                           mlir::MemRefType memrefType,
                                           Value baseSlot,
                                           ValueRange indices);

static std::optional<uint64_t> getStructFieldStorageOffset(Operation *anchor,
                                                           ora::StructType structType,
                                                           StringRef fieldName,
                                                           size_t *fieldIndex = nullptr)
{
    ArrayAttr fieldNamesAttr;
    ArrayAttr fieldTypesAttr;
    if (!getStructFieldAttrs(anchor, structType, fieldNamesAttr, fieldTypesAttr))
        return std::nullopt;

    uint64_t offset = 0;
    for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
    {
        if (cast<StringAttr>(fieldNamesAttr[i]).getValue() == fieldName)
        {
            if (fieldIndex)
                *fieldIndex = i;
            return offset;
        }
        Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
        offset += getMemRefElementWordCount(anchor, fieldType);
    }

    return std::nullopt;
}

static std::optional<uint64_t> getStructFieldStorageOffsetByScan(Operation *anchor,
                                                                 StringRef fieldName,
                                                                 Type resultType)
{
    ModuleOp module = anchor ? anchor->getParentOfType<ModuleOp>() : ModuleOp();
    if (!module)
        return std::nullopt;

    std::optional<uint64_t> foundOffset;
    bool ambiguous = false;
    module.walk([&](ora::StructDeclOp declOp)
                {
        ArrayAttr fieldNamesAttr = declOp->getAttrOfType<ArrayAttr>("ora.field_names");
        ArrayAttr fieldTypesAttr = declOp->getAttrOfType<ArrayAttr>("ora.field_types");
        if (!fieldNamesAttr || !fieldTypesAttr || fieldNamesAttr.size() != fieldTypesAttr.size())
            return WalkResult::advance();

        uint64_t offset = 0;
        for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
        {
            Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
            if (cast<StringAttr>(fieldNamesAttr[i]).getValue() == fieldName &&
                (!resultType || fieldType == resultType))
            {
                if (foundOffset && *foundOffset != offset)
                {
                    ambiguous = true;
                    return WalkResult::interrupt();
                }
                foundOffset = offset;
            }
            offset += getMemRefElementWordCount(anchor, fieldType);
        }

        return WalkResult::advance(); });

    if (ambiguous)
        return std::nullopt;
    return foundOffset;
}

static Value getStorageStructValueBaseSlot(Value structValue,
                                           ConversionPatternRewriter &rewriter,
                                           Location loc)
{
    while (auto cast = structValue.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() != 1)
            break;
        structValue = cast.getOperand(0);
    }

    if (auto sload = structValue.getDefiningOp<ora::SLoadOp>())
    {
        auto slotIndexOpt = computeGlobalSlot(sload.getGlobalName(), sload.getOperation());
        if (!slotIndexOpt)
            return Value();

        auto *ctx = rewriter.getContext();
        auto u256Type = sir::U256Type::get(ctx);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
        return rewriter.create<sir::ConstOp>(
            loc, u256Type, mlir::IntegerAttr::get(ui64Type, *slotIndexOpt));
    }

    if (auto extract = structValue.getDefiningOp<ora::StructFieldExtractOp>())
    {
        auto parentStructType = llvm::dyn_cast<ora::StructType>(extract.getStructValue().getType());
        std::optional<uint64_t> fieldOffset = parentStructType
            ? getStructFieldStorageOffset(extract.getOperation(), parentStructType, extract.getFieldName())
            : getStructFieldStorageOffsetByScan(
                  extract.getOperation(), extract.getFieldName(), extract.getResult().getType());
        if (!fieldOffset)
            return Value();

        Value parentSlot = getStorageStructValueBaseSlot(extract.getStructValue(), rewriter, loc);
        if (!parentSlot)
            return Value();

        return addStorageWordOffset(loc, parentSlot, *fieldOffset, rewriter);
    }

    if (auto load = structValue.getDefiningOp<mlir::memref::LoadOp>())
    {
        auto loadMemRefType = llvm::dyn_cast<mlir::MemRefType>(load.getMemRefType());
        if (!loadMemRefType)
            return Value();

        Value loadMemRef = load.getMemref();
        while (auto cast = loadMemRef.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (cast.getNumOperands() != 1)
                break;
            loadMemRef = cast.getOperand(0);
        }

        Value baseSlot = loadMemRef;
        if (auto sourceSLoad = loadMemRef.getDefiningOp<ora::SLoadOp>())
        {
            auto slotIndexOpt = computeGlobalSlot(sourceSLoad.getGlobalName(), load.getOperation());
            if (!slotIndexOpt)
                return Value();

            auto *ctx = rewriter.getContext();
            auto u256Type = sir::U256Type::get(ctx);
            auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
            baseSlot = rewriter.create<sir::ConstOp>(
                loc, u256Type, mlir::IntegerAttr::get(ui64Type, *slotIndexOpt));
        }
        else if (!llvm::isa<sir::U256Type>(baseSlot.getType()))
        {
            return Value();
        }

        SmallVector<Value> loadIndices;
        loadIndices.reserve(load.getIndices().size());
        for (Value index : load.getIndices())
            loadIndices.push_back(unwrapIndexCastInput(index));

        return loadMemRefType.hasStaticShape()
            ? buildStaticStorageMemRefSlot(rewriter, loc, loadMemRefType, baseSlot, loadIndices)
            : buildDynamicStorageMemRefSlot(rewriter, loc, load.getOperation(), loadMemRefType, baseSlot, loadIndices);
    }

    if (auto init = structValue.getDefiningOp<ora::StructInitOp>())
    {
        auto carrierKind = init->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
        if (!carrierKind || carrierKind.getValue() != kStorageStructCarrierKind)
            return Value();

        auto structType = llvm::dyn_cast<ora::StructType>(init.getResult().getType());
        if (!structType)
            return Value();

        ArrayAttr fieldNamesAttr;
        ArrayAttr fieldTypesAttr;
        if (!getStructFieldAttrs(init.getOperation(), structType, fieldNamesAttr, fieldTypesAttr))
            return Value();
        if (init.getFieldValues().size() != fieldTypesAttr.size())
            return Value();

        uint64_t offset = 0;
        for (size_t i = 0; i < fieldTypesAttr.size(); ++i)
        {
            Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
            Value fieldValue = init.getFieldValues()[i];
            Value fieldSlot;
            if (llvm::isa<ora::StructType>(fieldType))
                fieldSlot = getStorageStructValueBaseSlot(fieldValue, rewriter, loc);
            else
                fieldSlot = getSlotFromStorageFieldValue(fieldValue);
            if (fieldSlot)
                return subtractStorageWordOffset(loc, fieldSlot, offset, rewriter);
            offset += getMemRefElementWordCount(init.getOperation(), fieldType);
        }
    }

    return Value();
}

static Value getStorageBaseSlot(Value originalMemRef,
                                Value convertedMemRef,
                                ConversionPatternRewriter &rewriter,
                                Location loc)
{
    if (llvm::isa<sir::U256Type>(convertedMemRef.getType()))
        return convertedMemRef;

    if (auto bitcast = convertedMemRef.getDefiningOp<sir::BitcastOp>())
    {
        Value operand = bitcast.getOperand();
        auto viewKind = bitcast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
        if (viewKind && viewKind.getValue() == kStorageMemRefViewKind &&
            llvm::isa<sir::U256Type>(operand.getType()))
            return operand;
        if (viewKind && viewKind.getValue() == kStorageMemRefViewKind)
            return getStorageBaseSlot(originalMemRef, operand, rewriter, loc);
        if (llvm::isa<sir::U256Type>(operand.getType()))
            return operand;
        if (operand.getDefiningOp())
            return getStorageBaseSlot(originalMemRef, operand, rewriter, loc);
    }

    if (auto extract = originalMemRef.getDefiningOp<ora::StructFieldExtractOp>())
    {
        auto memrefType = llvm::dyn_cast<mlir::MemRefType>(extract.getResult().getType());
        auto structType = llvm::dyn_cast<ora::StructType>(extract.getStructValue().getType());
        size_t fieldIndex = 0;
        std::optional<uint64_t> fieldOffset;
        if (memrefType && !memrefType.hasStaticShape() && structType)
            fieldOffset = getStructFieldStorageOffset(
                extract.getOperation(), structType, extract.getFieldName(), &fieldIndex);
        if (memrefType && !memrefType.hasStaticShape() && !fieldOffset)
            fieldOffset = getStructFieldStorageOffsetByScan(
                extract.getOperation(), extract.getFieldName(), extract.getResult().getType());

        if (fieldOffset && structType)
        {
            if (auto init = extract.getStructValue().getDefiningOp<ora::StructInitOp>())
            {
                if (fieldIndex < init.getFieldValues().size())
                {
                    Value fieldValue = init.getFieldValues()[fieldIndex];
                    if (auto bitcast = fieldValue.getDefiningOp<sir::BitcastOp>())
                    {
                        auto viewKind = bitcast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
                        if (viewKind && viewKind.getValue() == kStorageMemRefViewKind &&
                            llvm::isa<sir::U256Type>(bitcast.getOperand().getType()))
                            return bitcast.getOperand();
                    }
                }
            }
        }

        if (fieldOffset)
        {
            Value structSlot = getStorageStructValueBaseSlot(extract.getStructValue(), rewriter, loc);
            if (structSlot)
                return addStorageWordOffset(loc, structSlot, *fieldOffset, rewriter);
        }

        Value structValue = extract.getStructValue();
        while (auto cast = structValue.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (cast.getNumOperands() != 1)
                break;
            structValue = cast.getOperand(0);
        }
        auto sload = structValue.getDefiningOp<ora::SLoadOp>();
        if (fieldOffset && sload)
        {
            auto slotIndexOpt = computeGlobalSlot(sload.getGlobalName(), extract.getOperation());
            if (!slotIndexOpt)
                return Value();

            auto ctx = rewriter.getContext();
            auto u256Type = sir::U256Type::get(ctx);
            auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
            Value baseSlot = rewriter.create<sir::ConstOp>(
                loc, u256Type, mlir::IntegerAttr::get(ui64Type, *slotIndexOpt));
            return addStorageWordOffset(loc, baseSlot, *fieldOffset, rewriter);
        }

        if (fieldOffset)
        {
            if (auto load = structValue.getDefiningOp<mlir::memref::LoadOp>())
            {
                auto loadMemRefType = llvm::dyn_cast<mlir::MemRefType>(load.getMemRefType());
                Value loadMemRef = load.getMemref();
                while (auto cast = loadMemRef.getDefiningOp<mlir::UnrealizedConversionCastOp>())
                {
                    if (cast.getNumOperands() != 1)
                        break;
                    loadMemRef = cast.getOperand(0);
                }
                auto sourceSLoad = loadMemRef.getDefiningOp<ora::SLoadOp>();
                if (loadMemRefType && (sourceSLoad || llvm::isa<sir::U256Type>(loadMemRef.getType())))
                {
                    auto ctx = rewriter.getContext();
                    auto u256Type = sir::U256Type::get(ctx);
                    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
                    Value baseSlot = loadMemRef;
                    if (sourceSLoad)
                    {
                        auto slotIndexOpt = computeGlobalSlot(sourceSLoad.getGlobalName(), extract.getOperation());
                        if (!slotIndexOpt)
                            return Value();
                        baseSlot = rewriter.create<sir::ConstOp>(
                            loc, u256Type, mlir::IntegerAttr::get(ui64Type, *slotIndexOpt));
                    }
                    SmallVector<Value> loadIndices;
                    loadIndices.reserve(load.getIndices().size());
                    for (Value index : load.getIndices())
                        loadIndices.push_back(unwrapIndexCastInput(index));
                    Value elementSlot = loadMemRefType.hasStaticShape()
                        ? buildStaticStorageMemRefSlot(rewriter, loc, loadMemRefType, baseSlot, loadIndices)
                        : buildDynamicStorageMemRefSlot(rewriter, loc, load.getOperation(), loadMemRefType, baseSlot, loadIndices);
                    if (!elementSlot)
                        return Value();
                    return addStorageWordOffset(loc, elementSlot, *fieldOffset, rewriter);
                }
            }
        }
    }

    if (auto cast = originalMemRef.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() == 1 && llvm::isa<sir::U256Type>(cast.getOperand(0).getType()))
            return cast.getOperand(0);
        if (cast.getNumOperands() == 1)
            return getStorageBaseSlot(cast.getOperand(0), convertedMemRef, rewriter, loc);
    }

    return Value();
}

static Value getStorageBaseSlotFromConvertedValue(Value convertedMemRef)
{
    if (!convertedMemRef)
        return Value();
    if (llvm::isa<sir::U256Type>(convertedMemRef.getType()))
        return convertedMemRef;

    if (auto bitcast = convertedMemRef.getDefiningOp<sir::BitcastOp>())
    {
        Value operand = bitcast.getOperand();
        auto viewKind = bitcast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
        if (viewKind && viewKind.getValue() == kStorageMemRefViewKind &&
            llvm::isa<sir::U256Type>(operand.getType()))
            return operand;

        if (Value nested = getStorageBaseSlotFromConvertedValue(operand))
            return nested;
    }

    if (auto cast = convertedMemRef.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() == 1)
            return getStorageBaseSlotFromConvertedValue(cast.getOperand(0));
    }

    return Value();
}

static Value buildDynamicStorageMemRefBase(ConversionPatternRewriter &rewriter,
                                           Location loc,
                                           Value baseSlot)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    Value size32 = rewriter.create<sir::ConstOp>(
        loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
    Value tmp = rewriter.create<sir::MallocOp>(loc, ptrType, size32);
    rewriter.create<sir::StoreOp>(loc, tmp, baseSlot);
    return rewriter.create<sir::KeccakOp>(loc, u256Type, tmp, size32);
}

static Value buildStaticStorageMemRefSlot(ConversionPatternRewriter &rewriter,
                                          Location loc,
                                          mlir::MemRefType memrefType,
                                          Value baseSlot,
                                          ValueRange indices)
{
    if (!memrefType.hasStaticShape())
        return Value();

    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    auto shape = memrefType.getShape();
    const int64_t rank = static_cast<int64_t>(shape.size());
    if (static_cast<int64_t>(indices.size()) != rank)
        return Value();

    Value linear;
    int64_t stride = 1;
    for (int64_t i = rank - 1; i >= 0; --i)
    {
        Value idx = ensureU256Value(rewriter, loc, indices[i]);
        Value strideConst = rewriter.create<sir::ConstOp>(
            loc, u256Type, mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(stride)));
        Value scaled = rewriter.create<sir::MulOp>(loc, u256Type, idx, strideConst);
        linear = linear ? rewriter.create<sir::AddOp>(loc, u256Type, linear, scaled) : scaled;
        stride *= shape[i];
    }

    if (!linear)
        return baseSlot;
    return rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, linear);
}

static Value buildDynamicStorageMemRefSlot(ConversionPatternRewriter &rewriter,
                                           Location loc,
                                           Operation *anchor,
                                           mlir::MemRefType memrefType,
                                           Value baseSlot,
                                           ValueRange indices)
{
    if (memrefType.hasStaticShape() || memrefType.getRank() != 1 || indices.size() != 1)
        return Value();

    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    Value dataBase = buildDynamicStorageMemRefBase(rewriter, loc, baseSlot);
    Value indexU256 = ensureU256Value(rewriter, loc, indices[0]);
    uint64_t elemWords = getMemRefElementWordCount(anchor, memrefType.getElementType());
    if (elemWords != 1)
    {
        Value elemWordsConst = rewriter.create<sir::ConstOp>(
            loc, u256Type, mlir::IntegerAttr::get(ui64Type, elemWords));
        Value offset = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, elemWordsConst);
        return rewriter.create<sir::AddOp>(loc, u256Type, dataBase, offset);
    }
    return rewriter.create<sir::AddOp>(loc, u256Type, dataBase, indexU256);
}

static mlir::MemRefType remapMemRefElementType(mlir::MemRefType type, Type elementType)
{
    return mlir::MemRefType::get(type.getShape(), elementType, type.getLayout(), type.getMemorySpace());
}

static Value unwrapPackedErrorUnionCarrier(Value value)
{
    while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() == 1)
        {
            value = cast.getOperand(0);
            continue;
        }
        break;
    }
    return value;
}

static Value stripDimIndexCasts(Value value)
{
    while (true)
    {
        if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (cast.getNumOperands() == 1)
            {
                value = cast.getOperand(0);
                continue;
            }
        }
        if (auto bitcast = value.getDefiningOp<sir::BitcastOp>())
        {
            value = bitcast.getOperand();
            continue;
        }
        break;
    }
    return value;
}

static mlir::IntegerAttr findErrorIdByName(Operation *anchor, StringRef name)
{
    if (auto module = anchor->getParentOfType<mlir::ModuleOp>())
    {
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
    return {};
}

static SIRNamingHelper &getNamingHelper(Operation *op)
{
    Operation *parentFunc = op->getParentOfType<mlir::func::FuncOp>();
    if (!parentFunc)
        return memrefFallbackHelper;

    auto it = memrefHelperMap.find(parentFunc);
    if (it == memrefHelperMap.end())
    {
        SIRNamingHelper newHelper;
        newHelper.reset();
        memrefHelperMap[parentFunc] = newHelper;
        return memrefHelperMap[parentFunc];
    }
    return it->second;
}

void clearMemRefHelperMap()
{
    memrefHelperMap.clear();
    memrefFallbackHelper.reset();
}

LogicalResult NormalizeNarrowErrorUnionMemRefLoadOp::matchAndRewrite(
    mlir::memref::LoadOp op,
    PatternRewriter &rewriter) const
{
    auto memrefType = op.getMemRefType();
    if (!memrefType)
        return failure();
    bool narrowCarrier = isNarrowErrorUnionType(memrefType.getElementType());
    bool scalarCarrier = isScalarErrorUnionMemRefCarrier(memrefType.getElementType());
    if (!narrowCarrier && !scalarCarrier)
        return failure();
    if (valueHasForceWideErrorUnion(op.getMemref()) && !scalarCarrier)
        return failure();
    if (!narrowCarrier && hasForceWideErrorUnionAttr(op) && !scalarCarrier)
        return failure();
    if (!isNarrowErrorUnionType(op.getType()) && !scalarCarrier)
        return failure();
    if (hasForceWideErrorUnionAttr(op) && !scalarCarrier)
        return failure();

    auto loc = op.getLoc();
    if (mlir::ora::isDebugEnabled())
    {
        llvm::errs() << "[OraToSIR] NormalizeNarrowErrorUnionMemRefLoadOp loc=" << loc
                     << " memrefType=" << memrefType << " resultType=" << op.getType() << "\n";
    }
    auto i256Type = mlir::IntegerType::get(rewriter.getContext(), 256);
    auto packedMemRefType = remapMemRefElementType(memrefType, i256Type);
    Value packedMemRef = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, packedMemRefType, op.getMemref()).getResult(0);

    auto packedLoad = rewriter.create<mlir::memref::LoadOp>(loc, packedMemRef, op.getIndices());
    Value restored = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, op.getType(), packedLoad.getResult()).getResult(0);
    rewriter.replaceOp(op, restored);
    return success();
}

LogicalResult NormalizeNarrowErrorUnionMemRefStoreOp::matchAndRewrite(
    mlir::memref::StoreOp op,
    PatternRewriter &rewriter) const
{
    auto memrefType = op.getMemRefType();
    if (!memrefType)
        return failure();
    bool narrowCarrier = isNarrowErrorUnionType(memrefType.getElementType());
    bool scalarCarrier = isScalarErrorUnionMemRefCarrier(memrefType.getElementType());
    if (!narrowCarrier && !scalarCarrier)
        return failure();
    if (valueHasForceWideErrorUnion(op.getMemref()) && !scalarCarrier)
        return failure();
    if (!isNarrowErrorUnionType(op.getValue().getType()) && !scalarCarrier)
        return failure();
    if (valueHasForceWideErrorUnion(op.getValue()) && !scalarCarrier)
        return failure();

    auto loc = op.getLoc();
    if (mlir::ora::isDebugEnabled())
    {
        llvm::errs() << "[OraToSIR] NormalizeNarrowErrorUnionMemRefStoreOp loc=" << loc
                     << " memrefType=" << memrefType << " valueType=" << op.getValue().getType() << "\n";
    }
    auto i256Type = mlir::IntegerType::get(rewriter.getContext(), 256);
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    auto ensureI256 = [&](Value value) -> Value {
        if (auto intType = llvm::dyn_cast<mlir::IntegerType>(value.getType()))
        {
            if (intType.getWidth() == 256)
                return value;
        }
        if (llvm::isa<sir::U256Type>(value.getType()))
            return rewriter.create<sir::BitcastOp>(loc, i256Type, value);
        if (llvm::isa<sir::PtrType>(value.getType()))
        {
            Value asU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
            return rewriter.create<sir::BitcastOp>(loc, i256Type, asU256);
        }
        return Value();
    };
    auto packedMemRefType = remapMemRefElementType(memrefType, i256Type);
    Value packedMemRef = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, packedMemRefType, op.getMemref()).getResult(0);
    Value packedValue = op.getValue();
    Operation *consumedCast = nullptr;
    if (auto cast = packedValue.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() == 2)
        {
            if (!narrowCarrier && !hasMaterializationKind(cast, mat_kind::kNormalizedErrorUnion))
                return failure();
            consumedCast = cast;
            Value tag = ensureI256(cast.getOperand(0));
            Value payload = ensureI256(cast.getOperand(1));
            if (!payload)
            {
                auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
                if (auto materialized = ora::materializePtrCarrierFromOraValue(
                        rewriter, loc, ptrType, cast.getOperand(1)))
                {
                    payload = ensureI256(*materialized);
                }
            }
            if (!tag || !payload)
                return failure();
            Value one = rewriter.create<mlir::arith::ConstantOp>(loc, i256Type, mlir::IntegerAttr::get(i256Type, 1));
            Value shifted = rewriter.create<mlir::arith::ShLIOp>(loc, payload, one);
            packedValue = rewriter.create<mlir::arith::OrIOp>(loc, shifted, tag);
        }
        else if (cast.getNumOperands() == 1)
        {
            packedValue = cast.getOperand(0);
        }
    }
    packedValue = unwrapPackedErrorUnionCarrier(packedValue);
    if (Operation *def = packedValue.getDefiningOp())
    {
        if (auto okOp = llvm::dyn_cast<ora::ErrorOkOp>(def))
        {
            if (llvm::isa<mlir::NoneType>(okOp.getValue().getType()))
            {
                packedValue = rewriter.create<mlir::arith::ConstantOp>(
                    loc,
                    i256Type,
                    mlir::IntegerAttr::get(i256Type, 0)
                );
            }
        }
        if (def->getName().getStringRef() == "ora.error.return")
        {
            auto sym = def->getAttrOfType<mlir::StringAttr>("sym_name");
            if (!sym)
                return failure();
            auto errorId = findErrorIdByName(op, sym.getValue());
            if (!errorId)
                return failure();

            auto idVal = errorId.getValue().zextOrTrunc(256);
            Value idConst = rewriter.create<mlir::arith::ConstantOp>(loc, i256Type, mlir::IntegerAttr::get(i256Type, idVal));
            idConst.getDefiningOp()->setAttr("ora.error_id", errorId);
            Value one = rewriter.create<mlir::arith::ConstantOp>(loc, i256Type, mlir::IntegerAttr::get(i256Type, 1));
            Value shifted = rewriter.create<mlir::arith::ShLIOp>(loc, idConst, one);
            packedValue = rewriter.create<mlir::arith::OrIOp>(loc, shifted, one);
        }
    }
    if (llvm::isa<sir::U256Type>(packedValue.getType()))
        packedValue = rewriter.create<sir::BitcastOp>(loc, i256Type, packedValue);
    else if (llvm::isa<sir::PtrType>(packedValue.getType()))
        packedValue = ensureI256(packedValue);
    if (!llvm::isa<mlir::IntegerType>(packedValue.getType()) ||
        llvm::cast<mlir::IntegerType>(packedValue.getType()).getWidth() != 256)
    {
        if (mlir::ora::isDebugEnabled())
        {
            llvm::errs() << "[OraToSIR] NormalizeNarrowErrorUnionMemRefStoreOp unsupported carrier loc=" << loc
                         << " packedValueType=" << packedValue.getType() << "\n";
        }
        return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, packedValue, packedMemRef, op.getIndices());
    if (consumedCast && consumedCast->use_empty())
        rewriter.eraseOp(consumedCast);
    return success();
}

// -----------------------------------------------------------------------------
// Convert memref.alloca → sir.malloc
// -----------------------------------------------------------------------------
LogicalResult ConvertMemRefAllocOp::matchAndRewrite(
    mlir::memref::AllocaOp op,
    typename mlir::memref::AllocaOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    DBG("ConvertMemRefAllocOp: matching alloca");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto memrefType = op.getType();

    // Get memref shape and element type
    if (!memrefType.hasStaticShape())
    {
        DBG("ConvertMemRefAllocOp: dynamic shape not supported");
        return failure();
    }

    // Calculate total size: num_elements * element_size (32 bytes for u256)
    int64_t numElements = 1;
    for (int64_t dim : memrefType.getShape())
    {
        numElements *= dim;
    }

    // Element size is always 32 bytes (256 bits) for SIR
    const uint64_t elementSize = 32;
    uint64_t totalSize = numElements * elementSize;

    // Create distinct location for allocation block
    auto allocLoc = mlir::NameLoc::get(
        mlir::StringAttr::get(ctx, "alloc"),
        loc);

    // Create size constant
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    auto sizeAttr = mlir::IntegerAttr::get(ui64Type, totalSize);
    auto u256Type = sir::U256Type::get(ctx);
    Value sizeConst = rewriter.create<sir::ConstOp>(allocLoc, u256Type, sizeAttr);

    // Name size constant
    auto &naming = getNamingHelper(op);
    naming.nameConst(sizeConst.getDefiningOp(), 0, totalSize, "alloc_size");

    // Create malloc - detect if this is an array allocation (multiple elements)
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    Value mallocResult = rewriter.create<sir::MallocOp>(allocLoc, ptrType, sizeConst);

    // Determine context: array allocation if numElements > 1
    SIRNamingHelper::Context allocCtx = (numElements > 1)
                                            ? SIRNamingHelper::Context::ArrayAllocation
                                            : SIRNamingHelper::Context::General;
    naming.nameMalloc(mallocResult.getDefiningOp(), 0, allocCtx);

    // Replace alloca with malloc result
    rewriter.replaceOp(op, mallocResult);
    DBG("ConvertMemRefAllocOp: converted alloca to malloc");
    return success();
}

// -----------------------------------------------------------------------------
// Convert memref.dim → sir.const for static memrefs, or load the leading length
// word for the supported dynamic 1-D memref representation [len | data...].
// -----------------------------------------------------------------------------
LogicalResult ConvertMemRefDimOp::matchAndRewrite(
    mlir::memref::DimOp op,
    typename mlir::memref::DimOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    DBG("ConvertMemRefDimOp: matching dim");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto memrefType = llvm::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    if (!memrefType)
    {
        DBG("ConvertMemRefDimOp: source is not memref");
        return failure();
    }

    // only handle constant dimension index
    int64_t dimIndex = -1;
    Value dimIndexValue = stripDimIndexCasts(adaptor.getIndex());
    if (auto indexConst = dimIndexValue.getDefiningOp<sir::ConstOp>())
    {
        auto indexAttr = llvm::dyn_cast<mlir::IntegerAttr>(indexConst.getValueAttr());
        if (!indexAttr)
        {
            DBG("ConvertMemRefDimOp: index not integer");
            return failure();
        }
        dimIndex = indexAttr.getInt();
    }
    else if (auto indexConst = dimIndexValue.getDefiningOp<mlir::arith::ConstantOp>())
    {
        auto indexAttr = llvm::dyn_cast<mlir::IntegerAttr>(indexConst.getValue());
        if (!indexAttr)
        {
            DBG("ConvertMemRefDimOp: arith index not integer");
            return failure();
        }
        dimIndex = indexAttr.getInt();
    }
    else
    {
        DBG("ConvertMemRefDimOp: non-constant index");
        return failure();
    }
    if (dimIndex < 0 || dimIndex >= static_cast<int64_t>(memrefType.getRank()))
    {
        DBG("ConvertMemRefDimOp: index out of bounds");
        return failure();
    }

    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    if (memrefType.hasStaticShape())
    {
        int64_t dimSize = memrefType.getDimSize(dimIndex);
        auto sizeAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(dimSize));
        auto sizeConst = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
        rewriter.replaceOp(op, sizeConst.getResult());
        return success();
    }

    if (memrefType.getRank() != 1 || dimIndex != 0 || !memrefType.isDynamicDim(0))
    {
        DBG("ConvertMemRefDimOp: unsupported dynamic memref shape");
        return failure();
    }

    Value base = adaptor.getSource();
    Value storageBaseSlot = getStorageBaseSlot(op.getSource(), base, rewriter, loc);
    if (storageBaseSlot)
    {
        Value length = rewriter.create<sir::SLoadOp>(loc, u256Type, storageBaseSlot);
        rewriter.replaceOp(op, length);
        return success();
    }

    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    if (!llvm::isa<sir::PtrType>(base.getType()))
        base = rewriter.create<sir::BitcastOp>(loc, ptrType, base);

    Value length = rewriter.create<sir::LoadOp>(loc, u256Type, base);
    rewriter.replaceOp(op, length);
    return success();
}

// -----------------------------------------------------------------------------
// Convert memref.load → sir.addptr + sir.load
// -----------------------------------------------------------------------------
LogicalResult ConvertMemRefLoadOp::matchAndRewrite(
    mlir::memref::LoadOp op,
    typename mlir::memref::LoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    DBG("ConvertMemRefLoadOp: matching load");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    // Get the converted memref (type converter should have converted memref to pointer)
    Value memref = adaptor.getMemref();
    auto memrefType = op.getMemRefType();
    auto indices = adaptor.getIndices();
    Value storageBaseSlot = memrefType ? getStorageBaseSlot(op.getMemref(), memref, rewriter, loc) : Value();
    if (!storageBaseSlot && memrefType)
        storageBaseSlot = getStorageBaseSlotFromConvertedValue(memref);
    if (storageBaseSlot)
    {
        Value slot = memrefType.hasStaticShape()
            ? buildStaticStorageMemRefSlot(rewriter, loc, memrefType, storageBaseSlot, indices)
            : buildDynamicStorageMemRefSlot(rewriter, loc, op.getOperation(), memrefType, storageBaseSlot, indices);
        if (!slot)
        {
            DBG("ConvertMemRefLoadOp: invalid storage memref indices");
            return failure();
        }

        auto u256Type = sir::U256Type::get(ctx);
        Type desiredType = u256Type;
        if (auto *tc = getTypeConverter())
        {
            if (Type converted = tc->convertType(op.getType()))
            {
                desiredType = converted;
            }
        }

        if (auto structType = llvm::dyn_cast<ora::StructType>(memrefType.getElementType()))
        {
            FailureOr<Value> loadedStruct = loadStructValueFromStorageMemRefRoot(
                op.getOperation(), loc, slot, structType, rewriter, getTypeConverter());
            if (failed(loadedStruct))
                return rewriter.notifyMatchFailure(op, "invalid struct field attributes for storage memref load");

            rewriter.replaceOp(op, *loadedStruct);
            DBG("ConvertMemRefLoadOp: converted struct storage memref load to struct init");
            return success();
        }

        if (auto nestedMemRefType = llvm::dyn_cast<mlir::MemRefType>(memrefType.getElementType());
            nestedMemRefType && !nestedMemRefType.hasStaticShape())
        {
            auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
            Type viewType = desiredType;
            if (!llvm::isa<sir::PtrType>(viewType))
                viewType = ptrType;
            Value storageView = rewriter.create<sir::BitcastOp>(loc, viewType, slot);
            storageView.getDefiningOp()->setAttr(
                kOraMaterializationKindAttr,
                StringAttr::get(ctx, kStorageMemRefViewKind));
            rewriter.replaceOp(op, storageView);
            DBG("ConvertMemRefLoadOp: converted nested storage memref load to storage view");
            return success();
        }

        Value loadResult = rewriter.create<sir::SLoadOp>(loc, u256Type, slot);
        if (desiredType != u256Type)
        {
            loadResult = rewriter.create<sir::BitcastOp>(loc, desiredType, loadResult);
        }
        rewriter.replaceOp(op, loadResult);
        DBG("ConvertMemRefLoadOp: converted storage memref load to sir.sload");
        return success();
    }

    if (!llvm::isa<sir::PtrType>(memref.getType()))
    {
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        memref = rewriter.create<sir::BitcastOp>(loc, ptrType, memref);
    }

    // Get indices (if any)
    Value basePtr = memref;

    auto &naming = getNamingHelper(op);

    if (!indices.empty())
    {
        if (!memrefType)
        {
            DBG("ConvertMemRefLoadOp: memref type missing");
            return failure();
        }

        // Extract element index for naming (best-effort from first index)
        int64_t elemIndex = naming.extractElementIndex(indices.front());
        if (elemIndex < 0)
        {
            elemIndex = naming.getNextElemIndex();
        }

        // Create distinct location for this element block
        std::string elemLocName = "elem" + std::to_string(elemIndex);
        auto elemLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, elemLocName),
            loc);

        auto u256Type = sir::U256Type::get(ctx);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

        // Create element size constant (32 bytes)
        auto elementSizeAttr = mlir::IntegerAttr::get(ui64Type, 32ULL);
        Value elementSize = rewriter.create<sir::ConstOp>(elemLoc, u256Type, elementSizeAttr);
        naming.nameConst(elementSize.getDefiningOp(), 0, 32, "elem_size");

        // Compute linearized offset in bytes: ((i0 * stride0) + (i1 * stride1) + ...) * elem_size
        Value linear = rewriter.create<sir::ConstOp>(
            elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, 0));

        auto shape = memrefType.getShape();
        const int64_t rank = static_cast<int64_t>(shape.size());
        if (static_cast<int64_t>(indices.size()) != rank)
        {
            DBG("ConvertMemRefLoadOp: index count does not match memref rank");
            return failure();
        }

        if (!memrefType.hasStaticShape())
        {
            if (rank != 1 || !memrefType.isDynamicDim(0))
            {
                DBG("ConvertMemRefLoadOp: non-static memref shape not supported");
                return failure();
            }

            Value idx = indices.front();
            if (!llvm::isa<sir::U256Type>(idx.getType()))
                idx = rewriter.create<sir::BitcastOp>(elemLoc, u256Type, idx);
            linear = idx;
        }
        else
        {
            int64_t stride = 1;
            for (int64_t i = rank - 1; i >= 0; --i)
            {
                Value idx = indices[i];
                if (!llvm::isa<sir::U256Type>(idx.getType()))
                {
                    idx = rewriter.create<sir::BitcastOp>(elemLoc, u256Type, idx);
                }

                Value strideConst = rewriter.create<sir::ConstOp>(
                    elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(stride)));
                Value scaled = rewriter.create<sir::MulOp>(elemLoc, u256Type, idx, strideConst);
                linear = rewriter.create<sir::AddOp>(elemLoc, u256Type, linear, scaled);

                stride *= shape[i];
            }
        }

        unsigned offsetIndex = naming.getNextOffsetIndex();
        Value offset = rewriter.create<sir::MulOp>(elemLoc, u256Type, linear, elementSize);
        if (!memrefType.hasStaticShape())
        {
            Value headerSize = rewriter.create<sir::ConstOp>(
                elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, 32ULL));
            offset = rewriter.create<sir::AddOp>(elemLoc, u256Type, offset, headerSize);
        }
        naming.nameOffset(offset.getDefiningOp(), 0, offsetIndex);

        // Add offset to base pointer
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        basePtr = rewriter.create<sir::AddPtrOp>(elemLoc, ptrType, memref, offset);
        naming.nameAddPtr(basePtr.getDefiningOp(), 0, elemIndex);
    }

    // Load from the pointer
    auto u256Type = sir::U256Type::get(ctx);
    Type desiredType = u256Type;
    if (auto *tc = getTypeConverter())
    {
        if (Type converted = tc->convertType(op.getType()))
        {
            desiredType = converted;
        }
    }
    int64_t elemIndex = naming.extractElementIndex(indices.empty() ? Value() : indices[0]);
    if (elemIndex < 0)
    {
        elemIndex = naming.getNextElemIndex();
    }

    // Use element location if we have an index, otherwise use original location
    Location loadLoc = loc;
    if (!indices.empty() && elemIndex >= 0)
    {
        std::string elemLocName = "elem" + std::to_string(elemIndex);
        loadLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, elemLocName),
            loc);
    }

    Value loadResult = rewriter.create<sir::LoadOp>(loadLoc, u256Type, basePtr);
    naming.nameLoad(loadResult.getDefiningOp(), 0, elemIndex);

    if (desiredType != u256Type)
    {
        loadResult = rewriter.create<sir::BitcastOp>(loadLoc, desiredType, loadResult);
    }

    rewriter.replaceOp(op, loadResult);
    DBG("ConvertMemRefLoadOp: converted load to sir.load");
    return success();
}

// -----------------------------------------------------------------------------
// Convert memref.store → sir.addptr + sir.store
// -----------------------------------------------------------------------------
LogicalResult ConvertMemRefStoreOp::matchAndRewrite(
    mlir::memref::StoreOp op,
    typename mlir::memref::StoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    DBG("ConvertMemRefStoreOp: matching store");
    if (mlir::ora::isDebugEnabled())
    {
        llvm::errs() << "[OraToSIR] ConvertMemRefStoreOp loc=" << op.getLoc()
                     << " valueType=" << op.getValue().getType()
                     << " memrefType=" << op.getMemref().getType() << "\n";
    }

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    // Get the converted memref and value (type converter should have converted memref to pointer)
    Value memref = adaptor.getMemref();
    Value value = adaptor.getValue();
    auto memrefType = llvm::dyn_cast<mlir::MemRefType>(op.getMemref().getType());
    auto indices = adaptor.getIndices();

    auto u256Type = sir::U256Type::get(ctx);
    Type desiredValueType = u256Type;
    if (auto *tc = getTypeConverter())
    {
        if (Type converted = tc->convertType(op.getValue().getType()))
        {
            desiredValueType = converted;
        }
    }

    if (value.getType() != desiredValueType)
    {
        if (auto *tc = getTypeConverter())
        {
            if (Value converted = tc->materializeTargetConversion(rewriter, loc, desiredValueType, value))
            {
                value = converted;
            }
            else if (llvm::isa<sir::U256Type, sir::PtrType>(desiredValueType))
            {
                if (mlir::ora::isDebugEnabled())
                {
                    llvm::errs() << "[OraToSIR] ConvertMemRefStoreOp fallback bitcast loc=" << op.getLoc()
                                 << " desiredType=" << desiredValueType
                                 << " currentType=" << value.getType() << "\n";
                }
                value = rewriter.create<sir::BitcastOp>(loc, desiredValueType, value);
            }
            else
            {
                if (mlir::ora::isDebugEnabled())
                {
                    llvm::errs() << "[OraToSIR] ConvertMemRefStoreOp failed materialization loc=" << op.getLoc()
                                 << " desiredType=" << desiredValueType
                                 << " currentType=" << value.getType() << "\n";
                }
                return failure();
            }
        }
        else if (llvm::isa<sir::U256Type, sir::PtrType>(desiredValueType))
        {
            if (mlir::ora::isDebugEnabled())
            {
                llvm::errs() << "[OraToSIR] ConvertMemRefStoreOp no typeConverter fallback bitcast loc=" << op.getLoc()
                             << " desiredType=" << desiredValueType
                             << " currentType=" << value.getType() << "\n";
            }
            value = rewriter.create<sir::BitcastOp>(loc, desiredValueType, value);
        }
        else
        {
            if (mlir::ora::isDebugEnabled())
            {
                llvm::errs() << "[OraToSIR] ConvertMemRefStoreOp no typeConverter failure loc=" << op.getLoc()
                             << " desiredType=" << desiredValueType
                             << " currentType=" << value.getType() << "\n";
            }
            return failure();
        }
    }

    Value storageBaseSlot = memrefType ? getStorageBaseSlot(op.getMemref(), memref, rewriter, loc) : Value();
    if (!storageBaseSlot && memrefType)
        storageBaseSlot = getStorageBaseSlotFromConvertedValue(memref);
    if (storageBaseSlot)
    {
        Value slot = memrefType.hasStaticShape()
            ? buildStaticStorageMemRefSlot(rewriter, loc, memrefType, storageBaseSlot, indices)
            : buildDynamicStorageMemRefSlot(rewriter, loc, op.getOperation(), memrefType, storageBaseSlot, indices);
        if (!slot)
        {
            DBG("ConvertMemRefStoreOp: invalid storage memref indices");
            return failure();
        }

        if (auto structType = llvm::dyn_cast<ora::StructType>(memrefType.getElementType()))
        {
            ArrayAttr fieldNamesAttr;
            ArrayAttr fieldTypesAttr;
            if (!getStructFieldAttrs(op.getOperation(), structType, fieldNamesAttr, fieldTypesAttr))
                return rewriter.notifyMatchFailure(op, "invalid struct field attributes for storage memref store");

            if (auto update = op.getValue().getDefiningOp<ora::StructFieldUpdateOp>())
            {
                std::optional<size_t> updatedFieldIndex;
                for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
                {
                    if (cast<StringAttr>(fieldNamesAttr[i]).getValue() == update.getFieldName())
                    {
                        updatedFieldIndex = i;
                        break;
                    }
                }
                if (!updatedFieldIndex)
                    return rewriter.notifyMatchFailure(op, "unknown struct field in storage memref update");

                uint64_t updatedFieldOffset = 0;
                for (size_t i = 0; i < *updatedFieldIndex; ++i)
                    updatedFieldOffset += getMemRefElementWordCount(op.getOperation(), cast<TypeAttr>(fieldTypesAttr[i]).getValue());
                Value updatedFieldSlot = addStorageWordOffset(loc, slot, updatedFieldOffset, rewriter);

                Type updatedFieldType = cast<TypeAttr>(fieldTypesAttr[*updatedFieldIndex]).getValue();
                Value updatedValue = update.getValue();
                if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(updatedFieldType))
                {
                    if (failed(storeStructValueToStorageMemRefRoot(
                            op.getOperation(), loc, updatedValue, updatedFieldSlot, nestedStructType,
                            rewriter, getTypeConverter())))
                    {
                        return rewriter.notifyMatchFailure(op, "failed to store nested struct storage memref update");
                    }
                    rewriter.eraseOp(op);
                    return success();
                }

                if (auto fieldMemRefType = llvm::dyn_cast<mlir::MemRefType>(updatedFieldType);
                    fieldMemRefType && !fieldMemRefType.hasStaticShape())
                {
                    if (failed(storeDynamicMemRefValueToStorageMemRefRoot(
                            op.getOperation(), loc, updatedValue, updatedFieldSlot, fieldMemRefType, rewriter)))
                    {
                        return rewriter.notifyMatchFailure(op, "failed to store dynamic storage memref update");
                    }
                    rewriter.eraseOp(op);
                    return success();
                }

                if (!llvm::isa<sir::U256Type>(updatedValue.getType()))
                {
                    Type convertedFieldType = getTypeConverter() ? getTypeConverter()->convertType(updatedValue.getType()) : Type();
                    if (convertedFieldType != updatedValue.getType() && convertedFieldType && llvm::isa<sir::U256Type>(convertedFieldType))
                        updatedValue = rewriter.create<sir::BitcastOp>(loc, convertedFieldType, updatedValue);
                    else
                        updatedValue = rewriter.create<sir::BitcastOp>(loc, u256Type, updatedValue);
                }
                rewriter.create<sir::SStoreOp>(loc, updatedFieldSlot, updatedValue);
                rewriter.eraseOp(op);
                return success();
            }

            auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
            auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
            Value carrier = value;
            if (llvm::isa<sir::U256Type>(carrier.getType()))
                carrier = rewriter.create<sir::BitcastOp>(loc, ptrType, carrier);
            else if (!llvm::isa<sir::PtrType>(carrier.getType()))
                carrier = rewriter.create<sir::BitcastOp>(loc, ptrType, carrier);

            bool needsDynamicCopy = false;
            for (size_t i = 0; i < fieldTypesAttr.size(); ++i)
            {
                Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
                if (auto fieldMemRefType = dyn_cast<mlir::MemRefType>(fieldType);
                    fieldMemRefType && !fieldMemRefType.hasStaticShape() &&
                    !storageStructCarrierPreservesField(carrier, i))
                {
                    needsDynamicCopy = true;
                    break;
                }
            }

            Block *afterBlock = nullptr;
            Region *parentRegion = nullptr;
            if (needsDynamicCopy)
            {
                Block *parentBlock = op->getBlock();
                parentRegion = parentBlock->getParent();
                afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
                rewriter.setInsertionPointToEnd(parentBlock);
            }

            SmallVector<Value> fieldValues;
            Value originalValue = op.getValue();
            if (auto structInit = originalValue.getDefiningOp<ora::StructInitOp>())
            {
                fieldValues.append(structInit.getFieldValues().begin(), structInit.getFieldValues().end());
            }
            else if (auto structInstantiate = originalValue.getDefiningOp<ora::StructInstantiateOp>())
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
                    fieldValues.push_back(rewriter.create<ora::StructFieldExtractOp>(
                        loc, fieldType, originalValue, fieldName));
                }
            }

            if (fieldValues.size() != fieldNamesAttr.size())
                return rewriter.notifyMatchFailure(op, "struct storage memref store field count mismatch");

            uint64_t fieldOffsetWords = 0;
            for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
            {
                Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
                Value fieldSlot = addStorageWordOffset(loc, slot, fieldOffsetWords, rewriter);
                Value carrierFieldPtr = carrier;
                if (i > 0)
                {
                    Value fieldOffset = rewriter.create<sir::ConstOp>(
                        loc, u256Type, mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(i) * 32ULL));
                    carrierFieldPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, carrier, fieldOffset);
                }

                if (auto nestedStructType = llvm::dyn_cast<ora::StructType>(fieldType))
                {
                    if (failed(storeStructValueToStorageMemRefRoot(
                            op.getOperation(), loc, fieldValues[i], fieldSlot, nestedStructType,
                            rewriter, getTypeConverter())))
                        return rewriter.notifyMatchFailure(op, "failed to store nested struct storage memref field");
                    fieldOffsetWords += getMemRefElementWordCount(op.getOperation(), fieldType);
                    continue;
                }

                if (auto fieldMemRefType = dyn_cast<mlir::MemRefType>(fieldType);
                    fieldMemRefType && !fieldMemRefType.hasStaticShape())
                {
                    Value fieldValue = rewriter.create<sir::LoadOp>(loc, u256Type, carrierFieldPtr);
                    StringRef fieldName = cast<StringAttr>(fieldNamesAttr[i]).getValue();
                    if (auto update = op.getValue().getDefiningOp<ora::StructFieldUpdateOp>())
                    {
                        if (update.getFieldName() != fieldName)
                        {
                            fieldOffsetWords += getMemRefElementWordCount(op.getOperation(), fieldType);
                            continue;
                        }
                    }
                    if (storageStructCarrierPreservesField(carrier, i))
                    {
                        fieldOffsetWords += getMemRefElementWordCount(op.getOperation(), fieldType);
                        continue;
                    }

                    Value sourcePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, fieldValue);
                    Value length = rewriter.create<sir::LoadOp>(loc, u256Type, sourcePtr);
                    rewriter.create<sir::SStoreOp>(loc, fieldSlot, length);

                    Value wordSize = rewriter.create<sir::ConstOp>(
                        loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
                    Value tmp = rewriter.create<sir::MallocOp>(loc, ptrType, wordSize);
                    rewriter.create<sir::StoreOp>(loc, tmp, fieldSlot);
                    Value storageDataBase = rewriter.create<sir::KeccakOp>(loc, u256Type, tmp, wordSize);

                    uint64_t elemWords = getMemRefElementWordCount(
                        op.getOperation(), fieldMemRefType.getElementType());

                    Value writeCount = length;
                    if (elemWords != 1)
                    {
                        Value elemWordsConst = rewriter.create<sir::ConstOp>(
                            loc, u256Type, mlir::IntegerAttr::get(ui64Type, elemWords));
                        writeCount = rewriter.create<sir::MulOp>(loc, u256Type, length, elemWordsConst);
                    }

                    Value zero = rewriter.create<sir::ConstOp>(
                        loc, u256Type, mlir::IntegerAttr::get(ui64Type, 0));
                    Value one = rewriter.create<sir::ConstOp>(
                        loc, u256Type, mlir::IntegerAttr::get(ui64Type, 1));
                    Block *dynamicFieldBlock = rewriter.getInsertionBlock();
                    auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
                    auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
                    auto copyAfterBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());

                    rewriter.setInsertionPointToEnd(dynamicFieldBlock);
                    rewriter.create<sir::BrOp>(loc, ValueRange{zero}, condBlock);

                    rewriter.setInsertionPointToStart(condBlock);
                    Value iv = condBlock->getArgument(0);
                    Value lt = rewriter.create<sir::LtOp>(loc, u256Type, iv, writeCount);
                    rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv}, ValueRange{}, bodyBlock, copyAfterBlock);

                    rewriter.setInsertionPointToStart(bodyBlock);
                    Value wordIndex = bodyBlock->getArgument(0);
                    Value wordSlot = rewriter.create<sir::AddOp>(loc, u256Type, storageDataBase, wordIndex);
                    Value wordBytes = rewriter.create<sir::MulOp>(loc, u256Type, wordIndex, wordSize);
                    Value dataOffset = rewriter.create<sir::AddOp>(loc, u256Type, wordBytes, wordSize);
                    Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, sourcePtr, dataOffset);
                    Value wordValue = rewriter.create<sir::LoadOp>(loc, u256Type, dataPtr);
                    rewriter.create<sir::SStoreOp>(loc, wordSlot, wordValue);
                    Value next = rewriter.create<sir::AddOp>(loc, u256Type, wordIndex, one);
                    rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

                    rewriter.setInsertionPointToStart(copyAfterBlock);
                    fieldOffsetWords += getMemRefElementWordCount(op.getOperation(), fieldType);
                    continue;
                }

                Value fieldValue = fieldValues[i];
                if (!llvm::isa<sir::U256Type>(fieldValue.getType()))
                    fieldValue = rewriter.create<sir::BitcastOp>(loc, u256Type, fieldValue);
                rewriter.create<sir::SStoreOp>(loc, fieldSlot, fieldValue);
                fieldOffsetWords += getMemRefElementWordCount(op.getOperation(), fieldType);
            }

            if (needsDynamicCopy)
                rewriter.create<sir::BrOp>(loc, ValueRange{}, afterBlock);
            rewriter.eraseOp(op);
            DBG("ConvertMemRefStoreOp: converted struct storage memref store to field sstores");
            return success();
        }

        if (!llvm::isa<sir::U256Type>(value.getType()))
            value = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
        rewriter.create<sir::SStoreOp>(loc, slot, value);
        rewriter.eraseOp(op);
        DBG("ConvertMemRefStoreOp: converted storage memref store to sir.sstore");
        return success();
    }

    // sir.store always stores a 256-bit word. After any memref-specific
    // normalization (including narrow error-union packing), cast the element
    // carrier back to sir.u256 before emitting the actual store.
    if (!llvm::isa<sir::U256Type>(value.getType()))
    {
        value = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
    }

    if (!llvm::isa<sir::PtrType>(memref.getType()))
    {
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        memref = rewriter.create<sir::BitcastOp>(loc, ptrType, memref);
    }

    auto &naming = getNamingHelper(op);

    // Get indices (if any)
    Value storePtr = memref;

    if (!indices.empty())
    {
        if (!memrefType)
        {
            DBG("ConvertMemRefStoreOp: memref type missing");
            return failure();
        }

        // Extract element index for naming (best-effort from first index)
        int64_t elemIndex = naming.extractElementIndex(indices.front());
        if (elemIndex < 0)
        {
            elemIndex = naming.getNextElemIndex();
        }

        // Create distinct location for this element block
        std::string elemLocName = "elem" + std::to_string(elemIndex);
        auto elemLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, elemLocName),
            loc);

        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

        // Create element size constant (32 bytes)
        auto elementSizeAttr = mlir::IntegerAttr::get(ui64Type, 32ULL);
        Value elementSize = rewriter.create<sir::ConstOp>(elemLoc, u256Type, elementSizeAttr);
        naming.nameConst(elementSize.getDefiningOp(), 0, 32, "elem_size");

        // Compute linearized offset in bytes: ((i0 * stride0) + (i1 * stride1) + ...) * elem_size
        Value linear = rewriter.create<sir::ConstOp>(
            elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, 0));

        auto shape = memrefType.getShape();
        const int64_t rank = static_cast<int64_t>(shape.size());
        if (static_cast<int64_t>(indices.size()) != rank)
        {
            DBG("ConvertMemRefStoreOp: index count does not match memref rank");
            return failure();
        }

        if (!memrefType.hasStaticShape())
        {
            if (rank != 1 || !memrefType.isDynamicDim(0))
            {
                DBG("ConvertMemRefStoreOp: non-static memref shape not supported");
                return failure();
            }

            Value idx = indices.front();
            if (!llvm::isa<sir::U256Type>(idx.getType()))
                idx = rewriter.create<sir::BitcastOp>(elemLoc, u256Type, idx);
            linear = idx;
        }
        else
        {
            int64_t stride = 1;
            for (int64_t i = rank - 1; i >= 0; --i)
            {
                Value idx = indices[i];
                if (!llvm::isa<sir::U256Type>(idx.getType()))
                {
                    idx = rewriter.create<sir::BitcastOp>(elemLoc, u256Type, idx);
                }

                Value strideConst = rewriter.create<sir::ConstOp>(
                    elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(stride)));
                Value scaled = rewriter.create<sir::MulOp>(elemLoc, u256Type, idx, strideConst);
                linear = rewriter.create<sir::AddOp>(elemLoc, u256Type, linear, scaled);

                stride *= shape[i];
            }
        }

        unsigned offsetIndex = naming.getNextOffsetIndex();
        Value offset = rewriter.create<sir::MulOp>(elemLoc, u256Type, linear, elementSize);
        if (!memrefType.hasStaticShape())
        {
            Value headerSize = rewriter.create<sir::ConstOp>(
                elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, 32ULL));
            offset = rewriter.create<sir::AddOp>(elemLoc, u256Type, offset, headerSize);
        }
        naming.nameOffset(offset.getDefiningOp(), 0, offsetIndex);

        // Add offset to base pointer
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        storePtr = rewriter.create<sir::AddPtrOp>(elemLoc, ptrType, memref, offset);
        naming.nameAddPtr(storePtr.getDefiningOp(), 0, elemIndex);
    }

    // Store the value - use element location if we have an index
    Location storeLoc = loc;
    if (!indices.empty())
    {
        int64_t elemIndex = naming.extractElementIndex(indices[0]);
        if (elemIndex >= 0)
        {
            std::string elemLocName = "elem" + std::to_string(elemIndex);
            storeLoc = mlir::NameLoc::get(
                mlir::StringAttr::get(ctx, elemLocName),
                loc);
        }
    }
    rewriter.create<sir::StoreOp>(storeLoc, storePtr, value);

    // Erase the original store operation
    rewriter.eraseOp(op);
    DBG("ConvertMemRefStoreOp: converted store to sir.store");
    return success();
}
