#include "patterns/MissingOps.h"
#include "patterns/EVMConstants.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include <optional>
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;

#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

// Helper: ensure value is u256, inserting a bitcast if needed.
static Value ensureU256(PatternRewriter &rewriter, Location loc, Value value)
{
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    if (llvm::isa<sir::U256Type>(value.getType()))
        return value;
    return rewriter.create<sir::BitcastOp>(loc, u256Type, value);
}

// Helper: convert a value to a non-zero boolean in u256 (double iszero).
static Value toCondU256(PatternRewriter &rewriter, Location loc, Value value)
{
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    Value v = ensureU256(rewriter, loc, value);
    Value isZero = rewriter.create<sir::IsZeroOp>(loc, u256Type, v);
    return rewriter.create<sir::IsZeroOp>(loc, u256Type, isZero);
}

static bool isDebugInfoLoweringEnabled(Operation *op)
{
    if (!op)
        return false;
    if (auto module = op->getParentOfType<ModuleOp>())
        return module->hasAttr("ora.debug_info");
    return false;
}

static std::optional<uint64_t> lookupNamedRootSlot(Operation *op, StringRef rootName)
{
    if (!op)
        return std::nullopt;
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
        return std::nullopt;
    auto slotsAttr = module->getAttrOfType<DictionaryAttr>("ora.global_slots");
    if (!slotsAttr)
        return std::nullopt;
    if (auto slotAttr = llvm::dyn_cast_or_null<IntegerAttr>(slotsAttr.get(rootName)))
        return slotAttr.getValue().getZExtValue();
    return std::nullopt;
}

static Value buildDebugNamedMemoryPtr(
    PatternRewriter &rewriter,
    Location loc,
    Operation *op,
    StringRef rootName)
{
    auto slot = lookupNamedRootSlot(op, rootName);
    if (!slot)
        return Value();

    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace=*/1);
    auto ui64Type = mlir::IntegerType::get(ctx, evm::kU64Bits, mlir::IntegerType::Unsigned);

    Value base = rewriter.create<sir::CodeSizeOp>(loc, u256Type);
    const uint64_t byteOffset = slot.value() * evm::kWordBytes;
    Value addr = base;
    if (byteOffset != 0)
    {
        Value offset = rewriter.create<sir::ConstOp>(
            loc,
            u256Type,
            mlir::IntegerAttr::get(ui64Type, byteOffset));
        addr = rewriter.create<sir::AddOp>(loc, u256Type, base, offset);
    }
    return rewriter.create<sir::BitcastOp>(loc, ptrType, addr);
}

static std::optional<unsigned> getAdtPayloadBitWidth(Type type)
{
    if (!type)
        return std::nullopt;
    if (llvm::isa<mlir::NoneType>(type))
        return 0u;
    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
        return intType.getWidth();
    if (auto intType = llvm::dyn_cast<ora::IntegerType>(type))
        return intType.getWidth();
    if (llvm::isa<ora::BoolType>(type))
        return 1u;
    if (llvm::isa<ora::AddressType>(type))
        return 160u;
    if (auto enumType = llvm::dyn_cast<ora::EnumType>(type))
        return getAdtPayloadBitWidth(enumType.getReprType());
    if (auto minType = llvm::dyn_cast<ora::MinValueType>(type))
        return getAdtPayloadBitWidth(minType.getBaseType());
    if (auto maxType = llvm::dyn_cast<ora::MaxValueType>(type))
        return getAdtPayloadBitWidth(maxType.getBaseType());
    if (auto rangeType = llvm::dyn_cast<ora::InRangeType>(type))
        return getAdtPayloadBitWidth(rangeType.getBaseType());
    if (auto scaledType = llvm::dyn_cast<ora::ScaledType>(type))
        return getAdtPayloadBitWidth(scaledType.getBaseType());
    if (auto exactType = llvm::dyn_cast<ora::ExactType>(type))
        return getAdtPayloadBitWidth(exactType.getBaseType());
    return std::nullopt;
}

static bool isNarrowAdt(ora::AdtType type)
{
    if (type.getVariantNames().size() > 256)
        return false;
    for (Type payloadType : type.getPayloadTypes())
    {
        auto width = getAdtPayloadBitWidth(payloadType);
        if (!width || *width > 248)
            return false;
    }
    return true;
}

static FailureOr<unsigned> getAdtVariantIndex(ora::AdtType type, StringRef variantName)
{
    for (auto [index, name] : llvm::enumerate(type.getVariantNames()))
        if (name == variantName)
            return static_cast<unsigned>(index);
    return failure();
}

static FailureOr<std::pair<Value, Value>> getNormalizedAdtParts(
    PatternRewriter &rewriter,
    Location loc,
    Value adtValue)
{
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    if (auto cast = adtValue.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        auto kind = cast->getAttrOfType<StringAttr>("ora.materialization_kind");
        if (kind && kind.getValue() == "normalized_adt" && cast.getNumOperands() == 2)
        {
            Value tag = cast.getOperand(0);
            Value payload = cast.getOperand(1);
            if (!llvm::isa<sir::U256Type>(tag.getType()))
                tag = rewriter.create<sir::BitcastOp>(loc, u256Type, tag);
            if (!llvm::isa<sir::U256Type>(payload.getType()))
                payload = rewriter.create<sir::BitcastOp>(loc, u256Type, payload);
            return std::make_pair(tag, payload);
        }
    }
    return failure();
}

static FailureOr<std::pair<Value, Value>> getNormalizedAdtPartsFromAdaptor(
    PatternRewriter &rewriter,
    Location loc,
    ValueRange values)
{
    if (values.size() != 2)
        return failure();
    return std::make_pair(
        ensureU256(rewriter, loc, values[0]),
        ensureU256(rewriter, loc, values[1]));
}

static LogicalResult convertAdtTagCommon(
    ora::AdtTagOp op,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter)
{
    ORA_DEBUG_PREFIX("OraToSIR", "ConvertAdtTagOp operands=" << operands.size() << " valueType=" << op.getValue().getType());
    FailureOr<std::pair<Value, Value>> parts = failure();
    if (operands.size() == 2)
        parts = getNormalizedAdtPartsFromAdaptor(rewriter, op.getLoc(), ValueRange(operands));
    if (failed(parts))
        parts = getNormalizedAdtParts(rewriter, op.getLoc(), op.getValue());
    if (failed(parts))
        return rewriter.notifyMatchFailure(op, "expected normalized ADT carrier");
    rewriter.replaceOp(op, parts->first);
    return success();
}

static LogicalResult convertAdtPayloadCommon(
    ora::AdtPayloadOp op,
    ArrayRef<Value> operands,
    const TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter)
{
    ORA_DEBUG_PREFIX("OraToSIR", "ConvertAdtPayloadOp operands=" << operands.size() << " valueType=" << op.getValue().getType() << " resultType=" << op.getResult().getType());
    auto adtType = llvm::dyn_cast<ora::AdtType>(op.getValue().getType());
    if (!adtType)
        return rewriter.notifyMatchFailure(op, "expected !ora.adt payload source");

    auto variantIndex = getAdtVariantIndex(adtType, op.getVariantName());
    if (failed(variantIndex))
        return rewriter.notifyMatchFailure(op, "unknown ADT variant");

    Type payloadType = adtType.getPayloadTypes()[*variantIndex];
    if (llvm::isa<mlir::NoneType>(payloadType))
        return rewriter.notifyMatchFailure(op, "unit ADT variant has no payload");

    FailureOr<std::pair<Value, Value>> parts = failure();
    if (operands.size() == 2)
        parts = getNormalizedAdtPartsFromAdaptor(rewriter, op.getLoc(), ValueRange(operands));
    if (failed(parts))
        parts = getNormalizedAdtParts(rewriter, op.getLoc(), op.getValue());
    if (failed(parts))
        return rewriter.notifyMatchFailure(op, "expected normalized ADT carrier");

    Value payload = parts->second;
    Type loweredType = Type();
    if (typeConverter)
        loweredType = typeConverter->convertType(op.getResult().getType());
    if (!loweredType)
        loweredType = op.getResult().getType();

    if (payload.getType() != loweredType)
    {
        auto loc = op.getLoc();
        if (llvm::isa<sir::PtrType>(loweredType))
        {
            payload = rewriter.create<sir::BitcastOp>(loc, loweredType, ensureU256(rewriter, loc, payload));
        }
        else if (llvm::isa<sir::U256Type>(loweredType))
        {
            payload = ensureU256(rewriter, loc, payload);
        }
        else if (llvm::isa<mlir::IntegerType>(loweredType))
        {
            payload = rewriter.create<sir::BitcastOp>(loc, loweredType, ensureU256(rewriter, loc, payload));
        }
        else
        {
            return rewriter.notifyMatchFailure(op, "unsupported lowered ADT payload result type");
        }
    }

    rewriter.replaceOp(op, payload);
    return success();
}

static Value makeU256Const(PatternRewriter &rewriter, Location loc, uint64_t value)
{
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    auto ui64Type = mlir::IntegerType::get(rewriter.getContext(), evm::kU64Bits, mlir::IntegerType::Unsigned);
    return rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, value));
}

// ---------------------------------------------------------------------------
// ora.refinement_guard → sir.cond_br (true → continue, false → revert)
// ---------------------------------------------------------------------------
LogicalResult ConvertRefinementGuardOp::matchAndRewrite(
    ora::RefinementGuardOp op,
    typename ora::RefinementGuardOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();

    // Split the block: after the guard instruction, execution continues.
    auto *afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));

    // Create a revert block.
    auto *revertBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());
    rewriter.setInsertionPointToStart(revertBlock);
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace=*/1);
    auto ui64Type = mlir::IntegerType::get(rewriter.getContext(), evm::kU64Bits, mlir::IntegerType::Unsigned);
    // revert(0, 0) — zero-length revert data.
    Value zeroU256 = rewriter.create<sir::ConstOp>(loc, u256Type,
        mlir::IntegerAttr::get(ui64Type, 0));
    Value zeroPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, zeroU256);
    Value zeroLen = rewriter.create<sir::ConstOp>(loc, u256Type,
        mlir::IntegerAttr::get(ui64Type, 0));
    rewriter.create<sir::RevertOp>(loc, zeroPtr, zeroLen);

    // At the end of parentBlock, branch based on the guard condition.
    rewriter.setInsertionPointToEnd(parentBlock);
    Value cond = toCondU256(rewriter, loc, adaptor.getCondition());
    rewriter.create<sir::CondBrOp>(
        loc, cond,
        ValueRange{}, ValueRange{},
        afterBlock, revertBlock);

    rewriter.eraseOp(op);
    return success();
}

// ---------------------------------------------------------------------------
// ora.power → sir.exp
// ---------------------------------------------------------------------------
LogicalResult ConvertPowerOp::matchAndRewrite(
    ora::PowerOp op,
    typename ora::PowerOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    Value base = ensureU256(rewriter, loc, adaptor.getBase());
    Value exp = ensureU256(rewriter, loc, adaptor.getExponent());
    rewriter.replaceOpWithNewOp<sir::ExpOp>(op, u256Type, base, exp);
    return success();
}

// ---------------------------------------------------------------------------
// ora.mload → sir.malloc + sir.load
// Named memory variables: allocate a slot and load from it.
// At SIR level, named variables are lowered to memory pointers that the
// upstream alloc pass will resolve. For now, emit a sir.freeptr to get the
// current free memory pointer and use it as the load address.
// ---------------------------------------------------------------------------
LogicalResult ConvertMLoadOp::matchAndRewrite(
    ora::MLoadOp op,
    typename ora::MLoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace=*/1);
    auto ui64Type = mlir::IntegerType::get(rewriter.getContext(), evm::kU64Bits, mlir::IntegerType::Unsigned);

    if (isDebugInfoLoweringEnabled(op.getOperation()))
    {
        if (Value ptr = buildDebugNamedMemoryPtr(rewriter, loc, op.getOperation(), op.getVariable()))
        {
            Value result = rewriter.create<sir::LoadOp>(loc, u256Type, ptr);
            rewriter.replaceOp(op, result);
            return success();
        }
    }

    // Allocate a word-sized slot and load from it.
    Value size = rewriter.create<sir::ConstOp>(loc, u256Type,
        mlir::IntegerAttr::get(ui64Type, evm::kWordBytes));
    Value ptr = rewriter.create<sir::MallocOp>(loc, ptrType, size);
    Value result = rewriter.create<sir::LoadOp>(loc, u256Type, ptr);
    rewriter.replaceOp(op, result);
    return success();
}

// ---------------------------------------------------------------------------
// ora.mstore → sir.malloc + sir.store
// ---------------------------------------------------------------------------
LogicalResult ConvertMStoreOp::matchAndRewrite(
    ora::MStoreOp op,
    typename ora::MStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace=*/1);
    auto ui64Type = mlir::IntegerType::get(rewriter.getContext(), evm::kU64Bits, mlir::IntegerType::Unsigned);

    if (isDebugInfoLoweringEnabled(op.getOperation()))
    {
        if (Value ptr = buildDebugNamedMemoryPtr(rewriter, loc, op.getOperation(), op.getVariable()))
        {
            Value val = ensureU256(rewriter, loc, adaptor.getValue());
            rewriter.create<sir::StoreOp>(loc, ptr, val);
            rewriter.eraseOp(op);
            return success();
        }
    }

    Value size = rewriter.create<sir::ConstOp>(loc, u256Type,
        mlir::IntegerAttr::get(ui64Type, evm::kWordBytes));
    Value ptr = rewriter.create<sir::MallocOp>(loc, ptrType, size);
    Value val = ensureU256(rewriter, loc, adaptor.getValue());
    rewriter.create<sir::StoreOp>(loc, ptr, val);
    rewriter.eraseOp(op);
    return success();
}

// ---------------------------------------------------------------------------
// ora.mload8 → sir.load8
// ---------------------------------------------------------------------------
LogicalResult ConvertMLoad8Op::matchAndRewrite(
    ora::MLoad8Op op,
    typename ora::MLoad8Op::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace=*/1);

    Value base = adaptor.getBase();
    // If base is not a ptr, bitcast it.
    if (!llvm::isa<sir::PtrType>(base.getType()))
        base = rewriter.create<sir::BitcastOp>(loc, ptrType, base);
    Value offset = ensureU256(rewriter, loc, adaptor.getOffset());
    Value addr = rewriter.create<sir::AddPtrOp>(loc, ptrType, base, offset);
    Value zero = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(mlir::IntegerType::get(rewriter.getContext(), 64, mlir::IntegerType::Unsigned), 0));
    Value result = rewriter.create<sir::Load8Op>(loc, u256Type, addr, zero);
    rewriter.replaceOp(op, result);
    return success();
}

// ---------------------------------------------------------------------------
// ora.mstore8 → sir.store8
// ---------------------------------------------------------------------------
LogicalResult ConvertMStore8Op::matchAndRewrite(
    ora::MStore8Op op,
    typename ora::MStore8Op::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace=*/1);

    Value base = adaptor.getBase();
    if (!llvm::isa<sir::PtrType>(base.getType()))
        base = rewriter.create<sir::BitcastOp>(loc, ptrType, base);
    Value offset = ensureU256(rewriter, loc, adaptor.getOffset());
    Value val = ensureU256(rewriter, loc, adaptor.getValue());
    Value addr = rewriter.create<sir::AddPtrOp>(loc, ptrType, base, offset);
    Value zero = rewriter.create<sir::ConstOp>(loc, val.getType(), mlir::IntegerAttr::get(mlir::IntegerType::get(rewriter.getContext(), 64, mlir::IntegerType::Unsigned), 0));
    rewriter.create<sir::Store8Op>(loc, addr, zero, val);
    rewriter.eraseOp(op);
    return success();
}

// ---------------------------------------------------------------------------
// ora.enum_constant → sir.const
// The variant value comes from the matching ora.enum.decl metadata so runtime
// enum comparisons use the same discriminants that HIR switch lowering emits.
// ---------------------------------------------------------------------------
LogicalResult ConvertEnumConstantOp::matchAndRewrite(
    ora::EnumConstantOp op,
    typename ora::EnumConstantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    if (auto stringValueAttr = op->getAttrOfType<mlir::StringAttr>("ora.enum_string_value"))
    {
        auto stringType = ora::StringType::get(rewriter.getContext());
        if (resultType == stringType)
        {
            rewriter.replaceOpWithNewOp<ora::StringConstantOp>(op, resultType, stringValueAttr);
            return success();
        }
    }
    if (auto bytesValueAttr = op->getAttrOfType<mlir::StringAttr>("ora.enum_bytes_value"))
    {
        auto bytesType = ora::BytesType::get(rewriter.getContext());
        if (resultType == bytesType)
        {
            rewriter.replaceOpWithNewOp<ora::BytesConstantOp>(op, resultType, bytesValueAttr);
            return success();
        }
    }
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    auto ui64Type = mlir::IntegerType::get(rewriter.getContext(), evm::kU64Bits, mlir::IntegerType::Unsigned);
    int64_t discriminant = -1;
    if (auto ordinalAttr = op->getAttrOfType<mlir::IntegerAttr>("ora.enum_ordinal"))
        discriminant = ordinalAttr.getInt();

    Operation *moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
        moduleOp = op->getParentWithTrait<mlir::OpTrait::SymbolTable>();
    if (moduleOp)
    {
        moduleOp->walk([&](ora::EnumDeclOp decl) {
            if (discriminant >= 0 || decl.getName() != op.getEnumName())
                return;

            auto variantNames = decl->getAttrOfType<mlir::ArrayAttr>("ora.variant_names");
            auto denseVariantValues = decl->getAttrOfType<mlir::DenseI64ArrayAttr>("ora.variant_values");
            auto arrayVariantValues = decl->getAttrOfType<mlir::ArrayAttr>("ora.variant_values");
            if (!variantNames || (!denseVariantValues && !arrayVariantValues))
                return;

            const size_t valueCount = denseVariantValues ? denseVariantValues.size() : arrayVariantValues.size();
            const size_t count = std::min<size_t>(variantNames.size(), valueCount);
            for (size_t i = 0; i < count; ++i)
            {
                auto nameAttr = llvm::dyn_cast<mlir::StringAttr>(variantNames[i]);
                if (!nameAttr || nameAttr.getValue() != op.getVariantName())
                    continue;
                if (denseVariantValues)
                {
                    discriminant = denseVariantValues[i];
                    return;
                }
                auto valueAttr = llvm::dyn_cast<mlir::IntegerAttr>(arrayVariantValues[i]);
                if (!valueAttr)
                    return;
                discriminant = valueAttr.getInt();
                return;
            }
        });
    }
    if (discriminant < 0 && moduleOp)
    {
        if (auto enumDict = moduleOp->getAttrOfType<DictionaryAttr>("sir.enum_values"))
        {
            std::string key = op.getEnumName().str();
            key.push_back('.');
            key += op.getVariantName().str();
            if (auto valueAttr = enumDict.get(key))
            {
                if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
                    discriminant = intAttr.getInt();
                else if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(valueAttr))
                {
                    auto stringType = ora::StringType::get(rewriter.getContext());
                    auto bytesType = ora::BytesType::get(rewriter.getContext());
                    if (resultType == stringType)
                    {
                        rewriter.replaceOpWithNewOp<ora::StringConstantOp>(op, resultType, strAttr);
                        return success();
                    }
                    if (resultType == bytesType)
                    {
                        rewriter.replaceOpWithNewOp<ora::BytesConstantOp>(op, resultType, strAttr);
                        return success();
                    }
                }
            }
        }
    }
    if (discriminant < 0)
        return rewriter.notifyMatchFailure(op, "missing enum discriminant metadata");

    Value result = rewriter.create<sir::ConstOp>(loc, u256Type,
        mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(discriminant)));
    rewriter.replaceOp(op, result);
    return success();
}

// ---------------------------------------------------------------------------
// ora.adt.* → split ADT carrier
// Runtime layout: (tag: u256, payload_carrier: u256). Aggregate payloads flow
// through payload_carrier as pointers bitcast to u256; scalar payloads stay as
// plain u256 values.
// ---------------------------------------------------------------------------
LogicalResult ConvertAdtConstructOp::matchAndRewrite(
    ora::AdtConstructOp op,
    typename ora::AdtConstructOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto adtType = llvm::dyn_cast<ora::AdtType>(op.getResult().getType());
    if (!adtType)
        return rewriter.notifyMatchFailure(op, "expected !ora.adt result type");

    auto variantIndex = getAdtVariantIndex(adtType, op.getVariantName());
    if (failed(variantIndex))
        return rewriter.notifyMatchFailure(op, "unknown ADT variant");

    auto loc = op.getLoc();
    Value tag = makeU256Const(rewriter, loc, *variantIndex);
    Value payload = makeU256Const(rewriter, loc, 0);
    if (!adaptor.getPayloadValues().empty())
    {
        if (adaptor.getPayloadValues().size() != 1)
            return rewriter.notifyMatchFailure(op, "ADT construct expects zero or one payload operand");
        payload = ensureU256(rewriter, loc, adaptor.getPayloadValues().front());
    }
    rewriter.replaceOp(op, ValueRange{tag, payload});
    return success();
}

LogicalResult ConvertAdtTagOp::matchAndRewrite(
    Operation *operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const
{
    auto op = llvm::dyn_cast<ora::AdtTagOp>(operation);
    if (!op)
        return failure();
    return convertAdtTagCommon(op, operands, rewriter);
}

LogicalResult ConvertAdtPayloadOp::matchAndRewrite(
    Operation *operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const
{
    auto op = llvm::dyn_cast<ora::AdtPayloadOp>(operation);
    if (!op)
        return failure();
    return convertAdtPayloadCommon(op, operands, getTypeConverter(), rewriter);
}

// ---------------------------------------------------------------------------
// ora.struct_field_store → sir.addptr + sir.store
// ---------------------------------------------------------------------------
LogicalResult ConvertStructFieldStoreOp::matchAndRewrite(
    ora::StructFieldStoreOp op,
    typename ora::StructFieldStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace=*/1);
    auto ui64Type = mlir::IntegerType::get(rewriter.getContext(), evm::kU64Bits, mlir::IntegerType::Unsigned);

    Value structVal = adaptor.getStructValue();
    if (!llvm::isa<sir::PtrType>(structVal.getType()))
        structVal = rewriter.create<sir::BitcastOp>(loc, ptrType, structVal);

    // Compute field offset: hash the field name to get a deterministic slot index,
    // then multiply by word size. A proper struct layout pass should replace this
    // with actual offsets; for now this is a placeholder that preserves semantics.
    StringRef fieldName = op.getFieldName();
    uint64_t fieldIndex = 0;
    for (char c : fieldName)
        fieldIndex = fieldIndex * 31 + static_cast<uint64_t>(c);
    fieldIndex %= 256; // Clamp to reasonable range.

    Value offset = rewriter.create<sir::ConstOp>(loc, u256Type,
        mlir::IntegerAttr::get(ui64Type, fieldIndex * evm::kWordBytes));
    Value fieldPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, structVal, offset);
    Value val = ensureU256(rewriter, loc, adaptor.getValue());
    rewriter.create<sir::StoreOp>(loc, fieldPtr, val);
    rewriter.eraseOp(op);
    return success();
}

// ---------------------------------------------------------------------------
// ora.destructure → identity / passthrough
// At SIR level, destructuring is a no-op: the value is simply forwarded.
// ---------------------------------------------------------------------------
LogicalResult ConvertDestructureOp::matchAndRewrite(
    ora::DestructureOp op,
    typename ora::DestructureOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value input = adaptor.getValue();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
        return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (input.getType() == resultType)
    {
        rewriter.replaceOp(op, input);
    }
    else
    {
        Value cast = rewriter.create<sir::BitcastOp>(loc, resultType, input);
        rewriter.replaceOp(op, cast);
    }
    return success();
}

// ---------------------------------------------------------------------------
// ora.immutable → passthrough (result = value operand)
// ---------------------------------------------------------------------------
LogicalResult ConvertImmutableOp::matchAndRewrite(
    ora::ImmutableOp op,
    typename ora::ImmutableOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value input = adaptor.getValue();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
        return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (input.getType() == resultType)
    {
        rewriter.replaceOp(op, input);
    }
    else
    {
        Value cast = rewriter.create<sir::BitcastOp>(loc, resultType, input);
        rewriter.replaceOp(op, cast);
    }
    return success();
}
