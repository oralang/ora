#include "patterns/Arithmetic.h"
#include "patterns/Naming.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;

// Debug logging macro
#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

static SIRNamingHelper &getNamingHelper(Operation *op)
{
    static SIRNamingHelper helper;
    return helper;
}
static mlir::arith::ConstantOp findOriginalConstant(Value val)
{
    Operation *defOp = val.getDefiningOp();
    if (!defOp)
        return nullptr;

    if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(defOp))
        return constOp;

    if (dyn_cast<sir::ConstOp>(defOp))
        return nullptr;

    if (auto bitcastOp = dyn_cast<sir::BitcastOp>(defOp))
    {
        return findOriginalConstant(bitcastOp.getInput());
    }
    if (auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(defOp))
    {
        if (castOp.getInputs().size() == 1)
        {
            return findOriginalConstant(castOp.getInputs()[0]);
        }
    }

    return nullptr;
}

// -----------------------------------------------------------------------------
// Lower ora.add → sir.add
// -----------------------------------------------------------------------------
LogicalResult ConvertAddOp::matchAndRewrite(
    ora::AddOp op,
    typename ora::AddOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Convert operands to SIR u256 - ALL Ora types must become SIR u256
    Type u256Type = sir::U256Type::get(op.getContext());
    Value lhsU256 = lhs;
    Value rhsU256 = rhs;

    // Convert LHS to u256 if needed
    // ALWAYS check if LHS is an arith.constant first (even if type is already u256)
    // The type converter may have converted the type but not the operation
    if (auto constOp = findOriginalConstant(lhs))
    {
        // Convert arith.constant to sir.const
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            lhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            // If already u256 type, use as-is, otherwise bitcast
            if (llvm::isa<sir::U256Type>(lhs.getType()))
                lhsU256 = lhs;
            else
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(lhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(lhs.getType()))
        {
            // Direct Ora int -> SIR u256 conversion
            lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
        else
        {
            Type lhsConverted = this->getTypeConverter()->convertType(lhs.getType());
            if (lhsConverted != lhs.getType() && llvm::isa<sir::U256Type>(lhsConverted))
            {
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, lhsConverted, lhs);
            }
            else
            {
                // Force to u256
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
            }
        }
    }

    // Convert RHS to u256 if needed
    // ALWAYS check if RHS is an arith.constant first (even if type is already u256)
    // The type converter may have converted the type but not the operation
    if (auto constOp = findOriginalConstant(rhs))
    {
        // Convert arith.constant to sir.const
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            rhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            // If already u256 type, use as-is, otherwise bitcast
            if (llvm::isa<sir::U256Type>(rhs.getType()))
                rhsU256 = rhs;
            else
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(rhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(rhs.getType()))
        {
            // Direct Ora int -> SIR u256 conversion
            rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
        else
        {
            Type rhsConverted = this->getTypeConverter()->convertType(rhs.getType());
            if (rhsConverted != rhs.getType() && llvm::isa<sir::U256Type>(rhsConverted))
            {
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, rhsConverted, rhs);
            }
            else
            {
                // Force to u256
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
            }
        }
    }

    // Result type is always u256
    // Note: Constant folding is handled by MLIR's canonicalization pass, not here
    auto ctx = op.getContext();
    auto addOp = rewriter.create<sir::AddOp>(loc, u256Type, lhsU256, rhsU256);
    addOp->setAttr("sir.result_name_0", StringAttr::get(ctx, "sum"));
    rewriter.replaceOp(op, addOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.mul → sir.mul
// -----------------------------------------------------------------------------
LogicalResult ConvertMulOp::matchAndRewrite(
    ora::MulOp op,
    typename ora::MulOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Convert operands to SIR u256 - ALL Ora types must become SIR u256
    Type u256Type = sir::U256Type::get(op.getContext());
    Value lhsU256 = lhs;
    Value rhsU256 = rhs;

    // Convert LHS to u256 if needed
    // ALWAYS check if LHS is an arith.constant first (even if type is already u256)
    // The type converter may have converted the type but not the operation
    if (auto constOp = findOriginalConstant(lhs))
    {
        // Convert arith.constant to sir.const
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            lhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            // If already u256 type, use as-is, otherwise bitcast
            if (llvm::isa<sir::U256Type>(lhs.getType()))
                lhsU256 = lhs;
            else
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(lhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(lhs.getType()))
        {
            // Direct Ora int -> SIR u256 conversion
            lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
        else
        {
            Type lhsConverted = this->getTypeConverter()->convertType(lhs.getType());
            if (lhsConverted != lhs.getType() && llvm::isa<sir::U256Type>(lhsConverted))
            {
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, lhsConverted, lhs);
            }
            else
            {
                // Force to u256
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
            }
        }
    }

    // Convert RHS to u256 if needed
    // ALWAYS check if RHS is an arith.constant first (even if type is already u256)
    // The type converter may have converted the type but not the operation
    if (auto constOp = findOriginalConstant(rhs))
    {
        // Convert arith.constant to sir.const
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            rhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            // If already u256 type, use as-is, otherwise bitcast
            if (llvm::isa<sir::U256Type>(rhs.getType()))
                rhsU256 = rhs;
            else
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(rhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(rhs.getType()))
        {
            // Direct Ora int -> SIR u256 conversion
            rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
        else
        {
            Type rhsConverted = this->getTypeConverter()->convertType(rhs.getType());
            if (rhsConverted != rhs.getType() && llvm::isa<sir::U256Type>(rhsConverted))
            {
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, rhsConverted, rhs);
            }
            else
            {
                // Force to u256
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
            }
        }
    }

    auto ctx = op.getContext();
    auto mulOp = rewriter.create<sir::MulOp>(loc, u256Type, lhsU256, rhsU256);
    mulOp->setAttr("sir.result_name_0", StringAttr::get(ctx, "product"));
    rewriter.replaceOp(op, mulOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.sub → sir.sub
// -----------------------------------------------------------------------------
LogicalResult ConvertSubOp::matchAndRewrite(
    ora::SubOp op,
    typename ora::SubOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Convert operands to SIR u256 - ALL Ora types must become SIR u256
    Type u256Type = sir::U256Type::get(op.getContext());
    Value lhsU256 = lhs;
    Value rhsU256 = rhs;

    // Convert LHS to u256 if needed (same logic as ConvertAddOp)
    if (auto constOp = findOriginalConstant(lhs))
    {
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            lhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            if (llvm::isa<sir::U256Type>(lhs.getType()))
                lhsU256 = lhs;
            else
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(lhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(lhs.getType()))
        {
            lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
        else
        {
            Type lhsConverted = this->getTypeConverter()->convertType(lhs.getType());
            if (lhsConverted != lhs.getType() && llvm::isa<sir::U256Type>(lhsConverted))
            {
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, lhsConverted, lhs);
            }
            else
            {
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
            }
        }
    }

    if (auto constOp = findOriginalConstant(rhs))
    {
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            rhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            if (llvm::isa<sir::U256Type>(rhs.getType()))
                rhsU256 = rhs;
            else
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(rhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(rhs.getType()))
        {
            rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
        else
        {
            Type rhsConverted = this->getTypeConverter()->convertType(rhs.getType());
            if (rhsConverted != rhs.getType() && llvm::isa<sir::U256Type>(rhsConverted))
            {
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, rhsConverted, rhs);
            }
            else
            {
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
            }
        }
    }

    auto subOp = rewriter.create<sir::SubOp>(loc, u256Type, lhsU256, rhsU256);
    auto ctx = op.getContext();
    subOp->setAttr("sir.result_name_0", StringAttr::get(ctx, "difference"));
    rewriter.replaceOp(op, subOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.div → sir.div
// -----------------------------------------------------------------------------
LogicalResult ConvertDivOp::matchAndRewrite(
    ora::DivOp op,
    typename ora::DivOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Convert operands to SIR u256 - ALL Ora types must become SIR u256
    Type u256Type = sir::U256Type::get(op.getContext());
    Value lhsU256 = lhs;
    Value rhsU256 = rhs;

    // Convert LHS to u256 if needed (same logic as ConvertAddOp)
    if (auto constOp = findOriginalConstant(lhs))
    {
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            lhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            if (llvm::isa<sir::U256Type>(lhs.getType()))
                lhsU256 = lhs;
            else
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(lhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(lhs.getType()))
        {
            lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
        else
        {
            Type lhsConverted = this->getTypeConverter()->convertType(lhs.getType());
            if (lhsConverted != lhs.getType() && llvm::isa<sir::U256Type>(lhsConverted))
            {
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, lhsConverted, lhs);
            }
            else
            {
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
            }
        }
    }

    if (auto constOp = findOriginalConstant(rhs))
    {
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            rhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            if (llvm::isa<sir::U256Type>(rhs.getType()))
                rhsU256 = rhs;
            else
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(rhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(rhs.getType()))
        {
            rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
        else
        {
            Type rhsConverted = this->getTypeConverter()->convertType(rhs.getType());
            if (rhsConverted != rhs.getType() && llvm::isa<sir::U256Type>(rhsConverted))
            {
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, rhsConverted, rhs);
            }
            else
            {
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
            }
        }
    }

    auto divOp = rewriter.create<sir::DivOp>(loc, u256Type, lhsU256, rhsU256);
    auto ctx = op.getContext();
    divOp->setAttr("sir.result_name_0", StringAttr::get(ctx, "quotient"));
    rewriter.replaceOp(op, divOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.rem → sir.mod
// -----------------------------------------------------------------------------
LogicalResult ConvertRemOp::matchAndRewrite(
    ora::RemOp op,
    typename ora::RemOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Convert operands to SIR u256 - ALL Ora types must become SIR u256
    Type u256Type = sir::U256Type::get(op.getContext());
    Value lhsU256 = lhs;
    Value rhsU256 = rhs;

    // Convert LHS to u256 if needed (same logic as ConvertAddOp)
    if (auto constOp = findOriginalConstant(lhs))
    {
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            lhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            if (llvm::isa<sir::U256Type>(lhs.getType()))
                lhsU256 = lhs;
            else
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(lhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(lhs.getType()))
        {
            lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
        }
        else
        {
            Type lhsConverted = this->getTypeConverter()->convertType(lhs.getType());
            if (lhsConverted != lhs.getType() && llvm::isa<sir::U256Type>(lhsConverted))
            {
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, lhsConverted, lhs);
            }
            else
            {
                lhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
            }
        }
    }

    if (auto constOp = findOriginalConstant(rhs))
    {
        auto valueAttr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
        {
            int64_t intValue = intAttr.getInt();
            uint64_t uintValue = static_cast<uint64_t>(intValue);
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);
            rhsU256 = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);
        }
        else
        {
            if (llvm::isa<sir::U256Type>(rhs.getType()))
                rhsU256 = rhs;
            else
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
    }
    else if (!llvm::isa<sir::U256Type>(rhs.getType()))
    {
        if (llvm::isa<ora::IntegerType>(rhs.getType()))
        {
            rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
        }
        else
        {
            Type rhsConverted = this->getTypeConverter()->convertType(rhs.getType());
            if (rhsConverted != rhs.getType() && llvm::isa<sir::U256Type>(rhsConverted))
            {
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, rhsConverted, rhs);
            }
            else
            {
                rhsU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);
            }
        }
    }

    auto modOp = rewriter.create<sir::ModOp>(loc, u256Type, lhsU256, rhsU256);
    auto ctx = op.getContext();
    modOp->setAttr("sir.result_name_0", StringAttr::get(ctx, "remainder"));
    rewriter.replaceOp(op, modOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.constant with Ora types → sir.const with SIR types
// -----------------------------------------------------------------------------
LogicalResult ConvertArithConstantOp::matchAndRewrite(
    mlir::arith::ConstantOp op,
    typename mlir::arith::ConstantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();

    DBG("ConvertArithConstantOp: checking constant with type: " << resultType);

    // Only convert if the result type is an Ora type
    if (!llvm::isa<ora::IntegerType>(resultType))
    {
        // Not an Ora type, skip conversion
        DBG("ConvertArithConstantOp: not an Ora type, skipping");
        return failure();
    }

    DBG("ConvertArithConstantOp: converting Ora constant to sir.const");

    auto valueAttr = op.getValue();
    if (!valueAttr)
    {
        return rewriter.notifyMatchFailure(op, "missing value attribute");
    }

    int64_t intValue = 0;
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr))
    {
        intValue = intAttr.getInt();
    }
    else
    {
        return rewriter.notifyMatchFailure(op, "value attribute is not an integer");
    }

    auto u256Type = sir::U256Type::get(op.getContext());
    auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);

    uint64_t uintValue = static_cast<uint64_t>(intValue);
    auto valueAttrU64 = mlir::IntegerAttr::get(ui64Type, uintValue);

    auto constOp = rewriter.create<sir::ConstOp>(loc, u256Type, valueAttrU64);

    auto &naming = getNamingHelper(op);
    const int64_t MAX_NAMED_CONSTANT = 10000;
    if (intValue >= 0 && intValue <= MAX_NAMED_CONSTANT)
    {
        naming.nameConst(constOp.getOperation(), 0, intValue);
    }
    rewriter.replaceOp(op, constOp.getResult());
    return success();
}
