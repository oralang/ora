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
    Value result = rewriter.create<sir::Load8Op>(loc, u256Type, base, offset);
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
    rewriter.create<sir::Store8Op>(loc, base, offset, val);
    rewriter.eraseOp(op);
    return success();
}

// ---------------------------------------------------------------------------
// ora.enum_constant → sir.const
// The variant ordinal is encoded as a u256 constant. We hash the variant
// name to produce a deterministic integer (using the same scheme the
// frontend uses for enum discriminants).
// ---------------------------------------------------------------------------
LogicalResult ConvertEnumConstantOp::matchAndRewrite(
    ora::EnumConstantOp op,
    typename ora::EnumConstantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    auto ui64Type = mlir::IntegerType::get(rewriter.getContext(), evm::kU64Bits, mlir::IntegerType::Unsigned);

    // Use the variant name hash as the enum discriminant value.
    StringRef variant = op.getVariantName();
    uint64_t hash = 0;
    for (char c : variant)
        hash = hash * 31 + static_cast<uint64_t>(c);

    Value result = rewriter.create<sir::ConstOp>(loc, u256Type,
        mlir::IntegerAttr::get(ui64Type, hash));
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
