#include "patterns/Logs.h"
#include "patterns/LoweringHelpers.h"
#include "patterns/Naming.h"
#include "SIR/SIRDialect.h"

using namespace mlir;
using namespace mlir::ora;
using mlir::ora::lowering::constU256;

LogicalResult ConvertLogOp::matchAndRewrite(
    ora::LogOp op,
    typename ora::LogOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    SmallVector<Value> params;
    params.reserve(adaptor.getParameters().size());
    for (Value v : adaptor.getParameters())
        params.push_back(v);

    Value dataPtr;
    Value dataLen;

    if (params.empty())
    {
        Value zero = constU256(rewriter, loc, 0);
        dataPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, zero);
        dataLen = zero;
    }
    else
    {
        const uint64_t totalSizeBytes = static_cast<uint64_t>(params.size()) * 32;
        Value totalSize = constU256(rewriter, loc, totalSizeBytes);
        dataPtr = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);
        dataLen = totalSize;

        for (size_t i = 0; i < params.size(); ++i)
        {
            Value val = params[i];
            Value valU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, val);
            Value offset = constU256(rewriter, loc, static_cast<uint64_t>(i * 32));
            Value slotPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, dataPtr, offset);
            rewriter.create<sir::StoreOp>(loc, slotPtr, valU256);
        }
    }

    Operation *logOp = nullptr;
    if (auto topicAttr = op->getAttrOfType<StringAttr>("ora.event_topic0"))
    {
        auto topicText = topicAttr.getValue();
        if (!topicText.consume_front("0x"))
            topicText.consume_front("0X");
        Value topic0 = constU256(rewriter, loc, llvm::APInt(256, topicText, 16));
        logOp = rewriter.create<sir::Log1Op>(loc, dataPtr, dataLen, topic0);
    }
    else
    {
        logOp = rewriter.create<sir::Log0Op>(loc, dataPtr, dataLen);
    }
    if (auto nameAttr = op.getEventNameAttr())
    {
        logOp->setAttr("ora.event_name", nameAttr);
    }
    rewriter.eraseOp(op);
    return success();
}
