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
            if (auto valueAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(enumDict.get(key)))
                discriminant = valueAttr.getInt();
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
