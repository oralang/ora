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

    SmallVector<bool> indexedFields(params.size(), false);
    if (auto indexedAttr = op->getAttrOfType<ArrayAttr>("ora.field_indexed"))
    {
        if (indexedAttr.size() != params.size())
            return rewriter.notifyMatchFailure(op, "log indexed field metadata does not match argument count");
        for (auto [index, attr] : llvm::enumerate(indexedAttr))
        {
            auto boolAttr = dyn_cast<BoolAttr>(attr);
            if (!boolAttr)
                return rewriter.notifyMatchFailure(op, "log indexed field metadata must be boolean");
            indexedFields[index] = boolAttr.getValue();
        }
    }

    SmallVector<Value> dataParams;
    dataParams.reserve(params.size());
    auto topicAttr = op->getAttrOfType<StringAttr>("ora.event_topic0");
    size_t topicCount = topicAttr ? 1 : 0;
    for (auto [index, val] : llvm::enumerate(params))
    {
        if (indexedFields[index])
        {
            ++topicCount;
        }
        else
        {
            dataParams.push_back(val);
        }
    }

    if (topicCount > 4)
        return rewriter.notifyMatchFailure(op, "event logs support at most four topics");

    Value dataPtr;
    Value dataLen;

    if (dataParams.empty())
    {
        Value zero = constU256(rewriter, loc, 0);
        dataPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, zero);
        dataLen = zero;
    }
    else
    {
        const uint64_t totalSizeBytes = static_cast<uint64_t>(dataParams.size()) * 32;
        Value totalSize = constU256(rewriter, loc, totalSizeBytes);
        dataPtr = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);
        dataLen = totalSize;

        for (size_t i = 0; i < dataParams.size(); ++i)
        {
            Value val = dataParams[i];
            Value valU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, val);
            Value offset = constU256(rewriter, loc, static_cast<uint64_t>(i * 32));
            Value slotPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, dataPtr, offset);
            rewriter.create<sir::StoreOp>(loc, slotPtr, valU256);
        }
    }

    SmallVector<Value> topics;
    topics.reserve(topicCount);
    if (topicAttr)
    {
        auto topicText = topicAttr.getValue();
        if (!topicText.consume_front("0x"))
            topicText.consume_front("0X");
        topics.push_back(constU256(rewriter, loc, llvm::APInt(256, topicText, 16)));
    }
    for (auto [index, val] : llvm::enumerate(params))
    {
        if (indexedFields[index])
            topics.push_back(rewriter.create<sir::BitcastOp>(loc, u256Type, val));
    }

    Operation *logOp = nullptr;
    switch (topics.size())
    {
    case 0:
        logOp = rewriter.create<sir::Log0Op>(loc, dataPtr, dataLen);
        break;
    case 1:
        logOp = rewriter.create<sir::Log1Op>(loc, dataPtr, dataLen, topics[0]);
        break;
    case 2:
        logOp = rewriter.create<sir::Log2Op>(loc, dataPtr, dataLen, topics[0], topics[1]);
        break;
    case 3:
        logOp = rewriter.create<sir::Log3Op>(loc, dataPtr, dataLen, topics[0], topics[1], topics[2]);
        break;
    case 4:
        logOp = rewriter.create<sir::Log4Op>(loc, dataPtr, dataLen, topics[0], topics[1], topics[2], topics[3]);
        break;
    default:
        llvm_unreachable("topic count checked before log op creation");
    }
    if (auto nameAttr = op.getEventNameAttr())
    {
        logOp->setAttr("ora.event_name", nameAttr);
    }
    rewriter.eraseOp(op);
    return success();
}
