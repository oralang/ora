#include "patterns/Logs.h"
#include "patterns/Naming.h"
#include "SIR/SIRDialect.h"

using namespace mlir;
using namespace mlir::ora;

LogicalResult ConvertLogOp::matchAndRewrite(
    ora::LogOp op,
    typename ora::LogOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    SmallVector<Value> params;
    params.reserve(adaptor.getParameters().size());
    for (Value v : adaptor.getParameters())
        params.push_back(v);

    Value dataPtr;
    Value dataLen;

    if (params.empty())
    {
        auto zeroAttr = mlir::IntegerAttr::get(ui64Type, 0);
        Value zero = rewriter.create<sir::ConstOp>(loc, u256Type, zeroAttr);
        dataPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, zero);
        dataLen = zero;
    }
    else
    {
        const uint64_t totalSizeBytes = static_cast<uint64_t>(params.size()) * 32;
        auto sizeAttr = mlir::IntegerAttr::get(ui64Type, totalSizeBytes);
        Value totalSize = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
        dataPtr = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);
        dataLen = totalSize;

        for (size_t i = 0; i < params.size(); ++i)
        {
            Value val = params[i];
            Value valU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, val);
            auto offsetAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(i * 32));
            Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, offsetAttr);
            Value slotPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, dataPtr, offset);
            rewriter.create<sir::StoreOp>(loc, slotPtr, valU256);
        }
    }

    auto logOp = rewriter.create<sir::Log0Op>(loc, dataPtr, dataLen);
    if (auto nameAttr = op.getEventNameAttr())
    {
        logOp->setAttr("ora.event_name", nameAttr);
    }
    rewriter.eraseOp(op);
    return success();
}
