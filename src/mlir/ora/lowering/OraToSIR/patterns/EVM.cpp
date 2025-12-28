#include "patterns/EVM.h"

#include "OraDebug.h"
#include "SIR/SIRDialect.h"

#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

using namespace mlir;
using namespace mlir::ora;

static LogicalResult lowerEvmOp(Operation *op, ConversionPatternRewriter &rewriter)
{
    auto name = op->getName().getStringRef();
    auto loc = op->getLoc();

    if (!name.starts_with("ora.evm."))
        return failure();

    auto resTy = sir::U256Type::get(rewriter.getContext());

    if (name == "ora.evm.origin")
    {
        auto newOp = rewriter.create<sir::OriginOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.caller")
    {
        auto newOp = rewriter.create<sir::CallerOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.gasprice")
    {
        auto newOp = rewriter.create<sir::GasPriceOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.callvalue")
    {
        auto newOp = rewriter.create<sir::CallValueOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.gas")
    {
        auto newOp = rewriter.create<sir::GasOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.timestamp")
    {
        auto newOp = rewriter.create<sir::TimestampOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.number")
    {
        auto newOp = rewriter.create<sir::NumberOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.coinbase")
    {
        auto newOp = rewriter.create<sir::CoinbaseOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.prevrandao" || name == "ora.evm.difficulty")
    {
        auto newOp = rewriter.create<sir::DifficultyOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.gaslimit")
    {
        auto newOp = rewriter.create<sir::GasLimitOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.chainid")
    {
        auto newOp = rewriter.create<sir::ChainIdOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }
    if (name == "ora.evm.basefee")
    {
        auto newOp = rewriter.create<sir::BaseFeeOp>(loc, resTy);
        rewriter.replaceOp(op, newOp.getResult());
        return success();
    }

    return failure();
}

LogicalResult ConvertEvmOp::matchAndRewrite(
    Operation *op,
    ArrayRef<Value> /*operands*/,
    ConversionPatternRewriter &rewriter) const
{
    return lowerEvmOp(op, rewriter);
}
