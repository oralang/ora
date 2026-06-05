#include "patterns/EVM.h"

#include "OraDebug.h"
#include "SIR/SIRDialect.h"

#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

using namespace mlir;
using namespace mlir::ora;

namespace
{
template <typename SirOp>
LogicalResult lowerEvmNullary(Operation *op, ConversionPatternRewriter &rewriter)
{
    auto loc = op->getLoc();
    auto resTy = sir::U256Type::get(rewriter.getContext());
    auto newOp = rewriter.create<SirOp>(loc, resTy);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

using EvmNullaryLoweringFn = LogicalResult (*)(Operation *, ConversionPatternRewriter &);

struct EvmNullaryLowering
{
    StringRef name;
    EvmNullaryLoweringFn lower;
};

static const EvmNullaryLowering kEvmNullaryLowerings[] = {
    {"ora.evm.origin", &lowerEvmNullary<sir::OriginOp>},
    {"ora.evm.caller", &lowerEvmNullary<sir::CallerOp>},
    {"ora.evm.gasprice", &lowerEvmNullary<sir::GasPriceOp>},
    {"ora.evm.callvalue", &lowerEvmNullary<sir::CallValueOp>},
    {"ora.evm.gas", &lowerEvmNullary<sir::GasOp>},
    {"ora.evm.timestamp", &lowerEvmNullary<sir::TimestampOp>},
    {"ora.evm.number", &lowerEvmNullary<sir::NumberOp>},
    {"ora.evm.coinbase", &lowerEvmNullary<sir::CoinbaseOp>},
    {"ora.evm.prevrandao", &lowerEvmNullary<sir::DifficultyOp>},
    {"ora.evm.difficulty", &lowerEvmNullary<sir::DifficultyOp>},
    {"ora.evm.gaslimit", &lowerEvmNullary<sir::GasLimitOp>},
    {"ora.evm.chainid", &lowerEvmNullary<sir::ChainIdOp>},
    {"ora.evm.basefee", &lowerEvmNullary<sir::BaseFeeOp>},
};
} // namespace

static LogicalResult lowerEvmOp(Operation *op, ConversionPatternRewriter &rewriter)
{
    auto name = op->getName().getStringRef();

    if (!name.starts_with("ora.evm."))
        return failure();

    for (const auto &entry : kEvmNullaryLowerings)
        if (name == entry.name)
            return entry.lower(op, rewriter);

    return failure();
}

LogicalResult ConvertEvmOp::matchAndRewrite(
    Operation *op,
    ArrayRef<Value> /*operands*/,
    ConversionPatternRewriter &rewriter) const
{
    return lowerEvmOp(op, rewriter);
}
