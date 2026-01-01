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
#include "llvm/ADT/APInt.h"
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

static Value ensureU256(ConversionPatternRewriter &rewriter, Location loc, Value value)
{
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    if (llvm::isa<sir::U256Type>(value.getType()))
        return value;
    return rewriter.create<sir::BitcastOp>(loc, u256Type, value);
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
// Convert ora.cmp → sir.eq/lt/gt (+ not/or for derived predicates)
// -----------------------------------------------------------------------------
LogicalResult ConvertCmpOp::matchAndRewrite(
    ora::CmpOp op,
    typename ora::CmpOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    auto predicate = op.getPredicate();
    Value lhs = ensureU256(rewriter, loc, adaptor.getLhs());
    Value rhs = ensureU256(rewriter, loc, adaptor.getRhs());
    auto u256Type = sir::U256Type::get(ctx);

    auto makeEq = [&]() -> Value {
        return rewriter.create<sir::EqOp>(loc, u256Type, lhs, rhs);
    };
    auto makeLt = [&]() -> Value {
        return rewriter.create<sir::LtOp>(loc, u256Type, lhs, rhs);
    };
    auto makeGt = [&]() -> Value {
        return rewriter.create<sir::GtOp>(loc, u256Type, lhs, rhs);
    };
    auto makeSLt = [&]() -> Value {
        return rewriter.create<sir::SLtOp>(loc, u256Type, lhs, rhs);
    };
    auto makeSGt = [&]() -> Value {
        return rewriter.create<sir::SGtOp>(loc, u256Type, lhs, rhs);
    };

    Value result;
    if (predicate == "eq")
    {
        result = makeEq();
    }
    else if (predicate == "ne" || predicate == "neq")
    {
        auto eqVal = makeEq();
        result = rewriter.create<sir::NotOp>(loc, u256Type, eqVal);
    }
    else if (predicate == "lt" || predicate == "ult")
    {
        result = makeLt();
    }
    else if (predicate == "gt" || predicate == "ugt")
    {
        result = makeGt();
    }
    else if (predicate == "le" || predicate == "lte" || predicate == "ule")
    {
        auto ltVal = makeLt();
        auto eqVal = makeEq();
        result = rewriter.create<sir::OrOp>(loc, u256Type, ltVal, eqVal);
    }
    else if (predicate == "ge" || predicate == "gte" || predicate == "uge")
    {
        auto gtVal = makeGt();
        auto eqVal = makeEq();
        result = rewriter.create<sir::OrOp>(loc, u256Type, gtVal, eqVal);
    }
    else if (predicate == "slt")
    {
        result = makeSLt();
    }
    else if (predicate == "sgt")
    {
        result = makeSGt();
    }
    else if (predicate == "sle")
    {
        auto ltVal = makeSLt();
        auto eqVal = makeEq();
        result = rewriter.create<sir::OrOp>(loc, u256Type, ltVal, eqVal);
    }
    else if (predicate == "sge")
    {
        auto gtVal = makeSGt();
        auto eqVal = makeEq();
        result = rewriter.create<sir::OrOp>(loc, u256Type, gtVal, eqVal);
    }
    else
    {
        return rewriter.notifyMatchFailure(op, "unsupported cmp predicate");
    }

    rewriter.replaceOp(op, result);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.const → arith.constant
// -----------------------------------------------------------------------------
LogicalResult ConvertConstOp::matchAndRewrite(
    ora::ConstOp op,
    typename ora::ConstOp::Adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto valueAttr = op.getValueAttr();

    auto typedAttr = llvm::dyn_cast<mlir::TypedAttr>(valueAttr);
    if (!typedAttr)
    {
        return rewriter.notifyMatchFailure(op, "const value is not a typed attribute");
    }

    auto constOp = rewriter.create<mlir::arith::ConstantOp>(loc, resultType, typedAttr);
    constOp->setAttr("ora.name", op.getNameAttr());
    rewriter.replaceOp(op, constOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.string.constant → ptr (dynamic bytes layout)
// Layout: [len: u256][bytes...]
// -----------------------------------------------------------------------------
LogicalResult ConvertStringConstantOp::matchAndRewrite(
    ora::StringConstantOp op,
    typename ora::StringConstantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto valueAttr = op.getValueAttr();
    auto strAttr = llvm::dyn_cast<mlir::StringAttr>(valueAttr);
    if (!strAttr)
    {
        return rewriter.notifyMatchFailure(op, "string constant missing value attribute");
    }

    auto bytes = strAttr.getValue();
    auto ctx = op.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    auto lengthAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(bytes.size()));
    Value lengthConst = rewriter.create<sir::ConstOp>(loc, u256Type, lengthAttr);
    Value wordSize = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
    Value totalSize = rewriter.create<sir::AddOp>(loc, u256Type, lengthConst, wordSize);
    Value base = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);

    // store length word at offset 0
    rewriter.create<sir::StoreOp>(loc, base, lengthConst);

    for (size_t i = 0; i < bytes.size(); ++i)
    {
        uint8_t byte = static_cast<uint8_t>(bytes[i]);
        auto offsetAttr = mlir::IntegerAttr::get(ui64Type, 32 + i);
        auto byteAttr = mlir::IntegerAttr::get(ui64Type, byte);
        Value offsetConst = rewriter.create<sir::ConstOp>(loc, u256Type, offsetAttr);
        Value byteConst = rewriter.create<sir::ConstOp>(loc, u256Type, byteAttr);
        rewriter.create<sir::Store8Op>(loc, base, offsetConst, byteConst);
    }

    rewriter.replaceOp(op, base);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.bytes.constant → ptr (dynamic bytes layout)
// -----------------------------------------------------------------------------
LogicalResult ConvertBytesConstantOp::matchAndRewrite(
    ora::BytesConstantOp op,
    typename ora::BytesConstantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto valueAttr = op.getValueAttr();
    auto strAttr = llvm::dyn_cast<mlir::StringAttr>(valueAttr);
    if (!strAttr)
    {
        return rewriter.notifyMatchFailure(op, "bytes constant missing value attribute");
    }

    auto raw = strAttr.getValue();
    if (raw.starts_with("0x") || raw.starts_with("0X"))
        raw = raw.drop_front(2);

    if (raw.size() % 2 != 0)
    {
        return rewriter.notifyMatchFailure(op, "bytes constant has odd-length hex string");
    }

    SmallVector<uint8_t> bytes;
    bytes.reserve(raw.size() / 2);
    for (size_t i = 0; i < raw.size(); i += 2)
    {
        auto hi = llvm::hexDigitValue(raw[i]);
        auto lo = llvm::hexDigitValue(raw[i + 1]);
        if (hi < 0 || lo < 0)
        {
            return rewriter.notifyMatchFailure(op, "bytes constant has invalid hex characters");
        }
        bytes.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }

    auto ctx = op.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    auto lengthAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(bytes.size()));
    Value lengthConst = rewriter.create<sir::ConstOp>(loc, u256Type, lengthAttr);
    Value wordSize = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
    Value totalSize = rewriter.create<sir::AddOp>(loc, u256Type, lengthConst, wordSize);
    Value base = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);

    rewriter.create<sir::StoreOp>(loc, base, lengthConst);

    for (size_t i = 0; i < bytes.size(); ++i)
    {
        auto offsetAttr = mlir::IntegerAttr::get(ui64Type, 32 + i);
        auto byteAttr = mlir::IntegerAttr::get(ui64Type, bytes[i]);
        Value offsetConst = rewriter.create<sir::ConstOp>(loc, u256Type, offsetAttr);
        Value byteConst = rewriter.create<sir::ConstOp>(loc, u256Type, byteAttr);
        rewriter.create<sir::Store8Op>(loc, base, offsetConst, byteConst);
    }

    rewriter.replaceOp(op, base);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.addr.to.i160 → sir.bitcast
// -----------------------------------------------------------------------------
LogicalResult ConvertAddrToI160Op::matchAndRewrite(
    ora::AddrToI160Op op,
    typename ora::AddrToI160Op::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value input = adaptor.getAddr();
    Type outType = op.getType();

    rewriter.replaceOpWithNewOp<sir::BitcastOp>(op, outType, input);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.i160.to.addr → sir.bitcast + mask to 160 bits
// -----------------------------------------------------------------------------
LogicalResult ConvertI160ToAddrOp::matchAndRewrite(
    ora::I160ToAddrOp op,
    typename ora::I160ToAddrOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    Value input = adaptor.getI160();

    auto u256Type = sir::U256Type::get(ctx);
    Value cast = rewriter.create<sir::BitcastOp>(loc, u256Type, input);

    llvm::APInt maskValue(256, 0);
    maskValue.setLowBits(160);
    auto maskAttr = mlir::IntegerAttr::get(u256Type, maskValue);
    Value mask = rewriter.create<sir::ConstOp>(loc, u256Type, maskAttr);
    Value masked = rewriter.create<sir::AndOp>(loc, u256Type, cast, mask);

    rewriter.replaceOp(op, masked);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.old → passthrough value
// -----------------------------------------------------------------------------
LogicalResult ConvertOldOp::matchAndRewrite(
    ora::OldOp op,
    typename ora::OldOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.refinement_to_base → passthrough value
// -----------------------------------------------------------------------------
LogicalResult ConvertRefinementToBaseOp::matchAndRewrite(
    ora::RefinementToBaseOp op,
    typename ora::RefinementToBaseOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
}

// -----------------------------------------------------------------------------
// Verification ops lowered away for Ora → SIR
// -----------------------------------------------------------------------------
LogicalResult ConvertInvariantOp::matchAndRewrite(
    ora::InvariantOp op,
    typename ora::InvariantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertRequiresOp::matchAndRewrite(
    ora::RequiresOp op,
    typename ora::RequiresOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertEnsuresOp::matchAndRewrite(
    ora::EnsuresOp op,
    typename ora::EnsuresOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertAssertOp::matchAndRewrite(
    ora::AssertOp op,
    typename ora::AssertOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertAssumeOp::matchAndRewrite(
    ora::AssumeOp op,
    typename ora::AssumeOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertDecreasesOp::matchAndRewrite(
    ora::DecreasesOp op,
    typename ora::DecreasesOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertIncreasesOp::matchAndRewrite(
    ora::IncreasesOp op,
    typename ora::IncreasesOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertHavocOp::matchAndRewrite(
    ora::HavocOp op,
    typename ora::HavocOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertQuantifiedOp::matchAndRewrite(
    ora::QuantifiedOp op,
    typename ora::QuantifiedOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.replaceOp(op, adaptor.getBody());
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

    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    // Only convert if the result type is not already legal in SIR
    if (typeConverter->isLegal(resultType))
    {
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

    auto convertedType = typeConverter->convertType(resultType);
    if (!convertedType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert constant type");
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

// -----------------------------------------------------------------------------
// Convert arith.cmpi → sir.{eq,lt,gt,sgt,slt} (+ combos for le/ge/ne)
// -----------------------------------------------------------------------------
LogicalResult ConvertArithCmpIOp::matchAndRewrite(
    mlir::arith::CmpIOp op,
    typename mlir::arith::CmpIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert cmp result type");
    }

    auto u256Type = sir::U256Type::get(op.getContext());
    auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
    auto oneAttr = mlir::IntegerAttr::get(ui64Type, 1);
    auto one = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    const auto pred = op.getPredicate();

    auto mkEq = [&]() { return rewriter.create<sir::EqOp>(loc, resultType, lhs, rhs).getResult(); };
    auto mkLt = [&]() { return rewriter.create<sir::LtOp>(loc, resultType, lhs, rhs).getResult(); };
    auto mkGt = [&]() { return rewriter.create<sir::GtOp>(loc, resultType, lhs, rhs).getResult(); };
    auto mkSLt = [&]() { return rewriter.create<sir::SLtOp>(loc, resultType, lhs, rhs).getResult(); };
    auto mkSGt = [&]() { return rewriter.create<sir::SGtOp>(loc, resultType, lhs, rhs).getResult(); };

    Value out;
    switch (pred)
    {
    case mlir::arith::CmpIPredicate::eq:
        out = mkEq();
        break;
    case mlir::arith::CmpIPredicate::ne:
    {
        auto eq = mkEq();
        out = rewriter.create<sir::XorOp>(loc, resultType, eq, one).getResult();
        break;
    }
    case mlir::arith::CmpIPredicate::ult:
        out = mkLt();
        break;
    case mlir::arith::CmpIPredicate::ugt:
        out = mkGt();
        break;
    case mlir::arith::CmpIPredicate::ule:
    {
        auto lt = mkLt();
        auto eq = mkEq();
        out = rewriter.create<sir::OrOp>(loc, resultType, lt, eq).getResult();
        break;
    }
    case mlir::arith::CmpIPredicate::uge:
    {
        auto gt = mkGt();
        auto eq = mkEq();
        out = rewriter.create<sir::OrOp>(loc, resultType, gt, eq).getResult();
        break;
    }
    case mlir::arith::CmpIPredicate::slt:
        out = mkSLt();
        break;
    case mlir::arith::CmpIPredicate::sgt:
        out = mkSGt();
        break;
    case mlir::arith::CmpIPredicate::sle:
    {
        auto lt = mkSLt();
        auto eq = mkEq();
        out = rewriter.create<sir::OrOp>(loc, resultType, lt, eq).getResult();
        break;
    }
    case mlir::arith::CmpIPredicate::sge:
    {
        auto gt = mkSGt();
        auto eq = mkEq();
        out = rewriter.create<sir::OrOp>(loc, resultType, gt, eq).getResult();
        break;
    }
    default:
        return rewriter.notifyMatchFailure(op, "unsupported cmp predicate");
    }

    rewriter.replaceOp(op, out);
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.addi → sir.add
// -----------------------------------------------------------------------------
LogicalResult ConvertArithAddIOp::matchAndRewrite(
    mlir::arith::AddIOp op,
    typename mlir::arith::AddIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert addi result type");
    }

    auto newOp = rewriter.create<sir::AddOp>(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.subi → sir.sub
// -----------------------------------------------------------------------------
LogicalResult ConvertArithSubIOp::matchAndRewrite(
    mlir::arith::SubIOp op,
    typename mlir::arith::SubIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert subi result type");
    }

    auto newOp = rewriter.create<sir::SubOp>(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.muli → sir.mul
// -----------------------------------------------------------------------------
LogicalResult ConvertArithMulIOp::matchAndRewrite(
    mlir::arith::MulIOp op,
    typename mlir::arith::MulIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert muli result type");
    }

    auto newOp = rewriter.create<sir::MulOp>(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.divui → sir.div
// -----------------------------------------------------------------------------
LogicalResult ConvertArithDivUIOp::matchAndRewrite(
    mlir::arith::DivUIOp op,
    typename mlir::arith::DivUIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert divui result type");
    }

    auto newOp = rewriter.create<sir::DivOp>(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.remui → sir.mod
// -----------------------------------------------------------------------------
LogicalResult ConvertArithRemUIOp::matchAndRewrite(
    mlir::arith::RemUIOp op,
    typename mlir::arith::RemUIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert remui result type");
    }

    auto newOp = rewriter.create<sir::ModOp>(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.divsi → sir.div
// -----------------------------------------------------------------------------
LogicalResult ConvertArithDivSIOp::matchAndRewrite(
    mlir::arith::DivSIOp op,
    typename mlir::arith::DivSIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert divsi result type");
    }

    auto newOp = rewriter.create<sir::DivOp>(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.andi → sir.and
// -----------------------------------------------------------------------------
LogicalResult ConvertArithAndIOp::matchAndRewrite(
    mlir::arith::AndIOp op,
    typename mlir::arith::AndIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert andi result type");
    }

    auto newOp = rewriter.create<sir::AndOp>(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.ori → sir.or
// -----------------------------------------------------------------------------
LogicalResult ConvertArithOrIOp::matchAndRewrite(
    mlir::arith::OrIOp op,
    typename mlir::arith::OrIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert ori result type");
    }

    auto newOp = rewriter.create<sir::OrOp>(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.xori → sir.xor
// -----------------------------------------------------------------------------
LogicalResult ConvertArithXOrIOp::matchAndRewrite(
    mlir::arith::XOrIOp op,
    typename mlir::arith::XOrIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert xori result type");
    }

    auto newOp = rewriter.create<sir::XorOp>(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.select → sir.select
// -----------------------------------------------------------------------------
LogicalResult ConvertArithSelectOp::matchAndRewrite(
    mlir::arith::SelectOp op,
    typename mlir::arith::SelectOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert select result type");
    }

    auto loc = op.getLoc();
    Value cond = ensureU256(rewriter, loc, adaptor.getCondition());
    Value trueVal = ensureU256(rewriter, loc, adaptor.getTrueValue());
    Value falseVal = ensureU256(rewriter, loc, adaptor.getFalseValue());

    auto newOp = rewriter.create<sir::SelectOp>(loc, resultType, cond, trueVal, falseVal);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.extui → sir.bitcast (u256)
// -----------------------------------------------------------------------------
LogicalResult ConvertArithExtUIOp::matchAndRewrite(
    mlir::arith::ExtUIOp op,
    typename mlir::arith::ExtUIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert extui result type");
    }

    auto newOp = rewriter.create<sir::BitcastOp>(op.getLoc(), resultType, adaptor.getIn());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.index_castui → sir.bitcast
// -----------------------------------------------------------------------------
LogicalResult ConvertArithIndexCastUIOp::matchAndRewrite(
    mlir::arith::IndexCastUIOp op,
    typename mlir::arith::IndexCastUIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert index_castui result type");
    }

    auto newOp = rewriter.create<sir::BitcastOp>(op.getLoc(), resultType, adaptor.getIn());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.trunci → sir.bitcast (u256)
// -----------------------------------------------------------------------------
LogicalResult ConvertArithTruncIOp::matchAndRewrite(
    mlir::arith::TruncIOp op,
    typename mlir::arith::TruncIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert trunci result type");
    }

    auto newOp = rewriter.create<sir::BitcastOp>(op.getLoc(), resultType, adaptor.getIn());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Fold redundant sir.bitcast chains
// -----------------------------------------------------------------------------
LogicalResult FoldRedundantBitcastOp::matchAndRewrite(
    sir::BitcastOp op,
    PatternRewriter &rewriter) const
{
    auto in = op.getInput();
    if (in.getType() == op.getType())
    {
        rewriter.replaceOp(op, in);
        return success();
    }

    if (auto inner = in.getDefiningOp<sir::BitcastOp>())
    {
        if (inner.getInput().getType() == op.getType())
        {
            rewriter.replaceOp(op, inner.getInput());
            return success();
        }
    }

    return failure();
}

// -----------------------------------------------------------------------------
// Fold sir.and with const 1
// -----------------------------------------------------------------------------
LogicalResult FoldAndOneOp::matchAndRewrite(
    sir::AndOp op,
    PatternRewriter &rewriter) const
{
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    if (auto lhs_const = lhs.getDefiningOp<sir::ConstOp>())
    {
        if (lhs_const.getValue() == 1)
        {
            rewriter.replaceOp(op, rhs);
            return success();
        }
    }

    if (auto rhs_const = rhs.getDefiningOp<sir::ConstOp>())
    {
        if (rhs_const.getValue() == 1)
        {
            rewriter.replaceOp(op, lhs);
            return success();
        }
    }

    return failure();
}
