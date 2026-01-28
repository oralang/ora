#include "patterns/MemRef.h"
#include "patterns/Naming.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;

// Debug logging macro
#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

// Helper: Get or create naming helper for current function
static SIRNamingHelper &getNamingHelper(Operation *op)
{
    // Get the parent function to use as a key for per-function helpers
    Operation *parentFunc = op->getParentOfType<mlir::func::FuncOp>();
    if (!parentFunc)
    {
        // Fallback to static helper if not in a function
        static SIRNamingHelper helper;
        return helper;
    }

    // Use a map to store per-function helpers (reset counters per function)
    static std::map<Operation *, SIRNamingHelper> helperMap;
    auto it = helperMap.find(parentFunc);
    if (it == helperMap.end())
    {
        // First time seeing this function - create new helper with reset counters
        SIRNamingHelper newHelper;
        newHelper.reset();
        helperMap[parentFunc] = newHelper;
        return helperMap[parentFunc];
    }
    return it->second;
}

// -----------------------------------------------------------------------------
// Convert memref.alloca → sir.malloc
// -----------------------------------------------------------------------------
LogicalResult ConvertMemRefAllocOp::matchAndRewrite(
    mlir::memref::AllocaOp op,
    typename mlir::memref::AllocaOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    DBG("ConvertMemRefAllocOp: matching alloca");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto memrefType = op.getType();

    // Get memref shape and element type
    if (!memrefType.hasStaticShape())
    {
        DBG("ConvertMemRefAllocOp: dynamic shape not supported");
        return failure();
    }

    // Calculate total size: num_elements * element_size (32 bytes for u256)
    int64_t numElements = 1;
    for (int64_t dim : memrefType.getShape())
    {
        numElements *= dim;
    }

    // Element size is always 32 bytes (256 bits) for SIR
    const uint64_t elementSize = 32;
    uint64_t totalSize = numElements * elementSize;

    // Create distinct location for allocation block
    auto allocLoc = mlir::NameLoc::get(
        mlir::StringAttr::get(ctx, "alloc"),
        loc);

    // Create size constant
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    auto sizeAttr = mlir::IntegerAttr::get(ui64Type, totalSize);
    auto u256Type = sir::U256Type::get(ctx);
    Value sizeConst = rewriter.create<sir::ConstOp>(allocLoc, u256Type, sizeAttr);

    // Name size constant
    auto &naming = getNamingHelper(op);
    naming.nameConst(sizeConst.getDefiningOp(), 0, totalSize, "alloc_size");

    // Create malloc - detect if this is an array allocation (multiple elements)
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    Value mallocResult = rewriter.create<sir::MallocOp>(allocLoc, ptrType, sizeConst);

    // Determine context: array allocation if numElements > 1
    SIRNamingHelper::Context allocCtx = (numElements > 1)
                                            ? SIRNamingHelper::Context::ArrayAllocation
                                            : SIRNamingHelper::Context::General;
    naming.nameMalloc(mallocResult.getDefiningOp(), 0, allocCtx);

    // Replace alloca with malloc result
    rewriter.replaceOp(op, mallocResult);
    DBG("ConvertMemRefAllocOp: converted alloca to malloc");
    return success();
}

// -----------------------------------------------------------------------------
// Convert memref.dim → sir.const (static shape only)
// -----------------------------------------------------------------------------
LogicalResult ConvertMemRefDimOp::matchAndRewrite(
    mlir::memref::DimOp op,
    typename mlir::memref::DimOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    DBG("ConvertMemRefDimOp: matching dim");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto memrefType = llvm::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    if (!memrefType || !memrefType.hasStaticShape())
    {
        DBG("ConvertMemRefDimOp: dynamic shape not supported");
        return failure();
    }

    // only handle constant dimension index
    int64_t dimIndex = -1;
    if (auto indexConst = adaptor.getIndex().getDefiningOp<sir::ConstOp>())
    {
        auto indexAttr = llvm::dyn_cast<mlir::IntegerAttr>(indexConst.getValueAttr());
        if (!indexAttr)
        {
            DBG("ConvertMemRefDimOp: index not integer");
            return failure();
        }
        dimIndex = indexAttr.getInt();
    }
    else if (auto indexConst = adaptor.getIndex().getDefiningOp<mlir::arith::ConstantOp>())
    {
        auto indexAttr = llvm::dyn_cast<mlir::IntegerAttr>(indexConst.getValue());
        if (!indexAttr)
        {
            DBG("ConvertMemRefDimOp: arith index not integer");
            return failure();
        }
        dimIndex = indexAttr.getInt();
    }
    else
    {
        DBG("ConvertMemRefDimOp: non-constant index");
        return failure();
    }
    if (dimIndex < 0 || dimIndex >= static_cast<int64_t>(memrefType.getRank()))
    {
        DBG("ConvertMemRefDimOp: index out of bounds");
        return failure();
    }

    int64_t dimSize = memrefType.getDimSize(dimIndex);
    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    auto sizeAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(dimSize));
    auto sizeConst = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
    rewriter.replaceOp(op, sizeConst.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert memref.load → sir.addptr + sir.load
// -----------------------------------------------------------------------------
LogicalResult ConvertMemRefLoadOp::matchAndRewrite(
    mlir::memref::LoadOp op,
    typename mlir::memref::LoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    DBG("ConvertMemRefLoadOp: matching load");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    // Get the converted memref (type converter should have converted memref to pointer)
    Value memref = adaptor.getMemref();
    if (!llvm::isa<sir::PtrType>(memref.getType()))
    {
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        memref = rewriter.create<sir::BitcastOp>(loc, ptrType, memref);
    }

    // Get indices (if any)
    auto indices = adaptor.getIndices();
    Value basePtr = memref;

    auto &naming = getNamingHelper(op);

    if (!indices.empty())
    {
        auto memrefType = llvm::dyn_cast<mlir::MemRefType>(op.getMemref().getType());
        if (!memrefType || !memrefType.hasStaticShape())
        {
            DBG("ConvertMemRefLoadOp: non-static memref shape not supported");
            return failure();
        }

        // Extract element index for naming (best-effort from first index)
        int64_t elemIndex = naming.extractElementIndex(indices.front());
        if (elemIndex < 0)
        {
            elemIndex = naming.getNextElemIndex();
        }

        // Create distinct location for this element block
        std::string elemLocName = "elem" + std::to_string(elemIndex);
        auto elemLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, elemLocName),
            loc);

        auto u256Type = sir::U256Type::get(ctx);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

        // Create element size constant (32 bytes)
        auto elementSizeAttr = mlir::IntegerAttr::get(ui64Type, 32ULL);
        Value elementSize = rewriter.create<sir::ConstOp>(elemLoc, u256Type, elementSizeAttr);
        naming.nameConst(elementSize.getDefiningOp(), 0, 32, "elem_size");

        // Compute linearized offset in bytes: ((i0 * stride0) + (i1 * stride1) + ...) * elem_size
        Value linear = rewriter.create<sir::ConstOp>(
            elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, 0));

        auto shape = memrefType.getShape();
        const int64_t rank = static_cast<int64_t>(shape.size());
        if (static_cast<int64_t>(indices.size()) != rank)
        {
            DBG("ConvertMemRefLoadOp: index count does not match memref rank");
            return failure();
        }

        int64_t stride = 1;
        for (int64_t i = rank - 1; i >= 0; --i)
        {
            Value idx = indices[i];
            if (!llvm::isa<sir::U256Type>(idx.getType()))
            {
                idx = rewriter.create<sir::BitcastOp>(elemLoc, u256Type, idx);
            }

            Value strideConst = rewriter.create<sir::ConstOp>(
                elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(stride)));
            Value scaled = rewriter.create<sir::MulOp>(elemLoc, u256Type, idx, strideConst);
            linear = rewriter.create<sir::AddOp>(elemLoc, u256Type, linear, scaled);

            stride *= shape[i];
        }

        unsigned offsetIndex = naming.getNextOffsetIndex();
        Value offset = rewriter.create<sir::MulOp>(elemLoc, u256Type, linear, elementSize);
        naming.nameOffset(offset.getDefiningOp(), 0, offsetIndex);

        // Add offset to base pointer
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        basePtr = rewriter.create<sir::AddPtrOp>(elemLoc, ptrType, memref, offset);
        naming.nameAddPtr(basePtr.getDefiningOp(), 0, elemIndex);
    }

    // Load from the pointer
    auto u256Type = sir::U256Type::get(ctx);
    Type desiredType = u256Type;
    if (auto *tc = getTypeConverter())
    {
        if (Type converted = tc->convertType(op.getType()))
        {
            desiredType = converted;
        }
    }
    int64_t elemIndex = naming.extractElementIndex(indices.empty() ? Value() : indices[0]);
    if (elemIndex < 0)
    {
        elemIndex = naming.getNextElemIndex();
    }

    // Use element location if we have an index, otherwise use original location
    Location loadLoc = loc;
    if (!indices.empty() && elemIndex >= 0)
    {
        std::string elemLocName = "elem" + std::to_string(elemIndex);
        loadLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, elemLocName),
            loc);
    }

    Value loadResult = rewriter.create<sir::LoadOp>(loadLoc, u256Type, basePtr);
    naming.nameLoad(loadResult.getDefiningOp(), 0, elemIndex);

    if (desiredType != u256Type)
    {
        loadResult = rewriter.create<sir::BitcastOp>(loadLoc, desiredType, loadResult);
    }

    rewriter.replaceOp(op, loadResult);
    DBG("ConvertMemRefLoadOp: converted load to sir.load");
    return success();
}

// -----------------------------------------------------------------------------
// Convert memref.store → sir.addptr + sir.store
// -----------------------------------------------------------------------------
LogicalResult ConvertMemRefStoreOp::matchAndRewrite(
    mlir::memref::StoreOp op,
    typename mlir::memref::StoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    DBG("ConvertMemRefStoreOp: matching store");

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    // Get the converted memref and value (type converter should have converted memref to pointer)
    Value memref = adaptor.getMemref();
    Value value = adaptor.getValue();

    if (!llvm::isa<sir::PtrType>(memref.getType()))
    {
        llvm::errs() << "[OraToSIR] ConvertMemRefStoreOp: memref type not lowered: "
                     << memref.getType() << " value type=" << value.getType()
                     << " at " << loc << "\n";
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        memref = rewriter.create<sir::BitcastOp>(loc, ptrType, memref);
    }

    // Ensure value is u256
    auto u256Type = sir::U256Type::get(ctx);
    if (!llvm::isa<sir::U256Type>(value.getType()))
    {
        value = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
    }

    auto &naming = getNamingHelper(op);

    // Get indices (if any)
    auto indices = adaptor.getIndices();
    Value storePtr = memref;

    if (!indices.empty())
    {
        auto memrefType = llvm::dyn_cast<mlir::MemRefType>(op.getMemref().getType());
        if (!memrefType || !memrefType.hasStaticShape())
        {
            DBG("ConvertMemRefStoreOp: non-static memref shape not supported");
            return failure();
        }

        // Extract element index for naming (best-effort from first index)
        int64_t elemIndex = naming.extractElementIndex(indices.front());
        if (elemIndex < 0)
        {
            elemIndex = naming.getNextElemIndex();
        }

        // Create distinct location for this element block
        std::string elemLocName = "elem" + std::to_string(elemIndex);
        auto elemLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, elemLocName),
            loc);

        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

        // Create element size constant (32 bytes)
        auto elementSizeAttr = mlir::IntegerAttr::get(ui64Type, 32ULL);
        Value elementSize = rewriter.create<sir::ConstOp>(elemLoc, u256Type, elementSizeAttr);
        naming.nameConst(elementSize.getDefiningOp(), 0, 32, "elem_size");

        // Compute linearized offset in bytes: ((i0 * stride0) + (i1 * stride1) + ...) * elem_size
        Value linear = rewriter.create<sir::ConstOp>(
            elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, 0));

        auto shape = memrefType.getShape();
        const int64_t rank = static_cast<int64_t>(shape.size());
        if (static_cast<int64_t>(indices.size()) != rank)
        {
            DBG("ConvertMemRefStoreOp: index count does not match memref rank");
            return failure();
        }

        int64_t stride = 1;
        for (int64_t i = rank - 1; i >= 0; --i)
        {
            Value idx = indices[i];
            if (!llvm::isa<sir::U256Type>(idx.getType()))
            {
                idx = rewriter.create<sir::BitcastOp>(elemLoc, u256Type, idx);
            }

            Value strideConst = rewriter.create<sir::ConstOp>(
                elemLoc, u256Type, mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(stride)));
            Value scaled = rewriter.create<sir::MulOp>(elemLoc, u256Type, idx, strideConst);
            linear = rewriter.create<sir::AddOp>(elemLoc, u256Type, linear, scaled);

            stride *= shape[i];
        }

        unsigned offsetIndex = naming.getNextOffsetIndex();
        Value offset = rewriter.create<sir::MulOp>(elemLoc, u256Type, linear, elementSize);
        naming.nameOffset(offset.getDefiningOp(), 0, offsetIndex);

        // Add offset to base pointer
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        storePtr = rewriter.create<sir::AddPtrOp>(elemLoc, ptrType, memref, offset);
        naming.nameAddPtr(storePtr.getDefiningOp(), 0, elemIndex);
    }

    // Store the value - use element location if we have an index
    Location storeLoc = loc;
    if (!indices.empty())
    {
        int64_t elemIndex = naming.extractElementIndex(indices[0]);
        if (elemIndex >= 0)
        {
            std::string elemLocName = "elem" + std::to_string(elemIndex);
            storeLoc = mlir::NameLoc::get(
                mlir::StringAttr::get(ctx, elemLocName),
                loc);
        }
    }
    rewriter.create<sir::StoreOp>(storeLoc, storePtr, value);

    // Erase the original store operation
    rewriter.eraseOp(op);
    DBG("ConvertMemRefStoreOp: converted store to sir.store");
    return success();
}
