#include "patterns/Storage.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;

// Debug logging macro
#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

// Helper: Set result name on an operation (for better readability in SIR MLIR)
static void setResultName(Operation *op, unsigned resultIndex, StringRef name)
{
    auto nameAttr = StringAttr::get(op->getContext(), name);
    std::string attrName = "sir.result_name_" + std::to_string(resultIndex);
    op->setAttr(attrName, nameAttr);
}

static Value computeWordCount(Location loc, Value lengthU256, ConversionPatternRewriter &rewriter)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    Value addend = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 31));
    Value divisor = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
    Value sum = rewriter.create<sir::AddOp>(loc, u256Type, lengthU256, addend);
    return rewriter.create<sir::DivOp>(loc, u256Type, sum, divisor);
}

static Value ensureU256Value(ConversionPatternRewriter &rewriter, Location loc, Value value)
{
    if (llvm::isa<sir::U256Type>(value.getType()))
        return value;
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    return rewriter.create<sir::BitcastOp>(loc, u256Type, value);
}

static uint64_t getElementWordCount(Type elementType)
{
    (void)elementType;
    // v0.1: word-aligned elements only.
    return 1;
}

static Value buildIndexFromU256(ConversionPatternRewriter &rewriter, Location loc, Value value)
{
    if (llvm::isa<mlir::IndexType>(value.getType()))
        return value;
    return rewriter.create<sir::BitcastOp>(loc, rewriter.getIndexType(), value);
}

// -----------------------------------------------------------------------------
// Helper: Find existing slot constant in function to avoid duplicates
// -----------------------------------------------------------------------------
static Value findOrCreateSlotConstant(Operation *op, uint64_t slotIndex,
                                      StringRef globalName,
                                      ConversionPatternRewriter &rewriter)
{
    auto loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto u256 = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    auto slotAttr = mlir::IntegerAttr::get(ui64Type, slotIndex);

    // Look for existing slot constant with the same name.
    Value existingConst;
    Operation *parentFunc = op->getParentOfType<mlir::func::FuncOp>();
    if (parentFunc)
    {
        parentFunc->walk([&](sir::ConstOp constOp)
                         {
            auto nameAttr = constOp->getAttrOfType<StringAttr>("sir.result_name_0");
            if (!nameAttr)
                return WalkResult::advance();
            if (nameAttr.getValue() != ("slot_" + globalName).str())
                return WalkResult::advance();
            // Found existing slot constant - reuse it
            existingConst = constOp.getResult();
            return WalkResult::interrupt();
            return WalkResult::advance(); });
    }

    // Reuse existing constant if found
    if (existingConst)
    {
        return existingConst;
    }

    // Create new constant if not found
    auto slotConst = rewriter.create<sir::ConstOp>(loc, u256, slotAttr);
    std::string slotName = "slot_" + globalName.str();
    slotConst->setAttr("sir.result_name_0", StringAttr::get(ctx, slotName));
    return slotConst.getResult();
}

// -----------------------------------------------------------------------------
// Lower ora.sload → sir.sload
// -----------------------------------------------------------------------------
LogicalResult ConvertSLoadOp::matchAndRewrite(
    ora::SLoadOp op,
    typename ora::SLoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    llvm::errs() << "[OraToSIR] ConvertSLoadOp: " << op.getGlobalName()
                 << " op=" << op.getOperation()
                 << " block=" << op.getOperation()->getBlock()
                 << "\n";
    llvm::errs().flush();

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    ctx->getOrLoadDialect<sir::SIRDialect>();
    StringRef globalName = op.getGlobalName();

    if (llvm::isa<mlir::RankedTensorType>(op.getResult().getType()))
    {
        auto slotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
        if (!slotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for array sload");
        uint64_t slotIndex = *slotIndexOpt;
        Value slotConst = findOrCreateSlotConstant(op.getOperation(), slotIndex, globalName, rewriter);
        if (auto slotOp = slotConst.getDefiningOp<sir::ConstOp>())
        {
            setResultName(slotOp, 0, ("slot_" + globalName).str());
        }
        rewriter.replaceOp(op, slotConst);
        return success();
    }

    // If this is a map type, return the base slot handle (do not load storage).
    if (llvm::isa<ora::MapType>(op.getResult().getType()))
    {
        auto slotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
        if (!slotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for map sload");
        uint64_t slotIndex = *slotIndexOpt;
        Value slotConst = findOrCreateSlotConstant(op.getOperation(), slotIndex, globalName, rewriter);
        if (auto slotOp = slotConst.getDefiningOp<sir::ConstOp>())
        {
            setResultName(slotOp, 0, ("slot_" + globalName).str());
        }
        rewriter.replaceOp(op, slotConst);
        return success();
    }

    // Check if result type is dynamic bytes (string, bytes, or enum with string repr)
    Type resultType = op.getResult().getType();
    bool isDynamicBytes = llvm::isa<ora::StringType, ora::BytesType>(resultType);
    llvm::errs() << "[OraToSIR]   resultType: " << resultType << "\n";
    llvm::errs() << "[OraToSIR]   dialect: " << resultType.getDialect().getNamespace() << "\n";
    llvm::errs() << "[OraToSIR]   is StringType: " << llvm::isa<ora::StringType>(resultType) << "\n";
    llvm::errs() << "[OraToSIR]   is BytesType: " << llvm::isa<ora::BytesType>(resultType) << "\n";
    llvm::errs() << "[OraToSIR]   is EnumType: " << llvm::isa<ora::EnumType>(resultType) << "\n";
    llvm::errs() << "[OraToSIR]   isDynamicBytes (initial): " << isDynamicBytes << "\n";

    // Check if enum type has string/bytes representation
    if (auto enumType = llvm::dyn_cast<ora::EnumType>(resultType))
    {
        Type reprType = enumType.getReprType();
        llvm::errs() << "[OraToSIR]   enum reprType: " << reprType << "\n";
        if (llvm::isa<ora::StringType, ora::BytesType>(reprType))
        {
            isDynamicBytes = true;
        }
    }
    if (!isDynamicBytes)
    {
        if (auto opaque = llvm::dyn_cast<mlir::OpaqueType>(resultType))
        {
            if (opaque.getDialectNamespace() == "ora" &&
                (opaque.getTypeData() == "string" || opaque.getTypeData() == "bytes"))
            {
                isDynamicBytes = true;
            }
        }
    }

    // Compute storage slot index from ora.global operation
    auto slotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for sload");
    uint64_t slotIndex = *slotIndexOpt;
    DBG("  -> slot index: " << slotIndex);

    // Find or create slot constant (reuse if already exists in function)
    Value slotConst = findOrCreateSlotConstant(op.getOperation(), slotIndex, globalName, rewriter);
    // Set name: "slot_" + globalName
    if (auto slotOp = slotConst.getDefiningOp<sir::ConstOp>())
    {
        setResultName(slotOp, 0, ("slot_" + globalName).str());
    }

    auto u256 = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    // Precompute converted type so we can treat pointer results as dynamic bytes.
    Type convertedResultType = this->getTypeConverter()->convertType(resultType);
    if (!convertedResultType)
    {
        convertedResultType = ptrType;
    }
    llvm::errs() << "[OraToSIR]   convertedResultType: " << convertedResultType << "\n";
    if (llvm::isa<sir::PtrType>(convertedResultType))
    {
        isDynamicBytes = true;
    }
    // Fallback: if this is an Ora type that isn't a known scalar, treat as dynamic bytes.
    if (!isDynamicBytes && resultType.getDialect().getNamespace() == "ora" &&
        !llvm::isa<ora::IntegerType, ora::BoolType, ora::AddressType, ora::MapType, ora::StructType, ora::EnumType>(resultType))
    {
        isDynamicBytes = true;
    }
    llvm::errs() << "[OraToSIR]   isDynamicBytes (final): " << isDynamicBytes << "\n";

    if (isDynamicBytes)
    {
        // Use the precomputed converted result type (ptr fallback for dynamic bytes).

        Value length = rewriter.create<sir::SLoadOp>(loc, u256, slotConst);
        Value wordSize = rewriter.create<sir::ConstOp>(loc, u256, mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned), 32));
        Value totalSize = rewriter.create<sir::AddOp>(loc, u256, length, wordSize);
        Value basePtr = rewriter.create<sir::MallocOp>(loc, convertedResultType, totalSize);
        rewriter.create<sir::StoreOp>(loc, basePtr, length);

        Value wordCount = computeWordCount(loc, length, rewriter);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
        Value zero = rewriter.create<sir::ConstOp>(loc, u256, mlir::IntegerAttr::get(ui64Type, 0));
        Value one = rewriter.create<sir::ConstOp>(loc, u256, mlir::IntegerAttr::get(ui64Type, 1));

        Block *parentBlock = op->getBlock();
        Region *parentRegion = parentBlock->getParent();
        auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
        auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256}, {loc});
        auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256}, {loc});

        rewriter.setInsertionPoint(op);
        rewriter.create<sir::BrOp>(loc, ValueRange{zero}, condBlock);

        rewriter.setInsertionPointToStart(condBlock);
        Value iv = condBlock->getArgument(0);
        Value lt = rewriter.create<sir::LtOp>(loc, u256, iv, wordCount);
        rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv}, ValueRange{}, bodyBlock, afterBlock);

        rewriter.setInsertionPointToStart(bodyBlock);
        Value ivU256 = bodyBlock->getArgument(0);
        Value slotOffset = rewriter.create<sir::AddOp>(loc, u256, ivU256, one);
        Value slot = rewriter.create<sir::AddOp>(loc, u256, slotConst, slotOffset);
        Value wordVal = rewriter.create<sir::SLoadOp>(loc, u256, slot);

        Value wordBytes = rewriter.create<sir::MulOp>(loc, u256, ivU256, wordSize);
        Value dataOffset = rewriter.create<sir::AddOp>(loc, u256, wordBytes, wordSize);
        Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, convertedResultType, basePtr, dataOffset);
        rewriter.create<sir::StoreOp>(loc, dataPtr, wordVal);

        Value next = rewriter.create<sir::AddOp>(loc, u256, ivU256, one);
        rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

        rewriter.replaceOp(op, basePtr);
        llvm::errs() << "[OraToSIR]   replaced with dynamic bytes load\n";
        return success();
    }

    // Replace the ora.sload with sir.sload for scalar values
    // Get the converted result type from type converter (enum -> u256, etc.)
    if (!convertedResultType)
    {
        llvm::errs() << "[OraToSIR]   failed to convert result type for scalar sload\n";
        return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    // Ensure we use u256 for storage loads (all scalar types become u256 in SIR)
    if (!llvm::isa<sir::U256Type>(convertedResultType))
    {
        convertedResultType = u256;
    }

    auto sloadOp = rewriter.replaceOpWithNewOp<sir::SLoadOp>(op, convertedResultType, slotConst);
    setResultName(sloadOp, 0, "value");

    llvm::errs() << "[OraToSIR]   replaced with sir.sload\n";
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.sstore → sir.sstore
// -----------------------------------------------------------------------------
LogicalResult ConvertSStoreOp::matchAndRewrite(
    ora::SStoreOp op,
    typename ora::SStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    Value value = adaptor.getValue();
    StringRef globalName = op.getGlobalName();

    if (llvm::isa<mlir::RankedTensorType>(value.getType()))
    {
        DBG("  -> skipping sstore of tensor type");
        return failure();
    }

    const bool isDynamicBytes = llvm::isa<ora::StringType, ora::BytesType>(op.getValue().getType());

    // Compute storage slot index from ora.global operation
    auto slotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for sstore");
    uint64_t slotIndex = *slotIndexOpt;

    // Find or create slot constant (reuse if already exists in function)
    Value slot = findOrCreateSlotConstant(op.getOperation(), slotIndex, globalName, rewriter);
    // Set name: "slot_" + globalName
    if (auto slotOp = slot.getDefiningOp<sir::ConstOp>())
    {
        setResultName(slotOp, 0, ("slot_" + globalName).str());
    }

    Type u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    if (isDynamicBytes)
    {
        Value basePtr = value;
        if (!llvm::isa<sir::PtrType>(basePtr.getType()))
        {
            Type converted = this->getTypeConverter()->convertType(basePtr.getType());
            if (converted && converted != basePtr.getType())
            {
                basePtr = rewriter.create<sir::BitcastOp>(loc, converted, basePtr);
            }
        }

        Value length = rewriter.create<sir::LoadOp>(loc, u256Type, basePtr);
        rewriter.create<sir::SStoreOp>(loc, slot, length);

        Value wordCount = computeWordCount(loc, length, rewriter);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
        Value zero = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 0));
        Value one = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 1));
        Value wordSize = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));

        Block *parentBlock = op->getBlock();
        Region *parentRegion = parentBlock->getParent();
        auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
        auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
        auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});

        rewriter.setInsertionPoint(op);
        rewriter.create<sir::BrOp>(loc, ValueRange{zero}, condBlock);

        rewriter.setInsertionPointToStart(condBlock);
        Value iv = condBlock->getArgument(0);
        Value lt = rewriter.create<sir::LtOp>(loc, u256Type, iv, wordCount);
        rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv}, ValueRange{}, bodyBlock, afterBlock);

        rewriter.setInsertionPointToStart(bodyBlock);
        Value ivU256 = bodyBlock->getArgument(0);
        Value slotOffset = rewriter.create<sir::AddOp>(loc, u256Type, ivU256, one);
        Value slotAddr = rewriter.create<sir::AddOp>(loc, u256Type, slot, slotOffset);

        Value wordBytes = rewriter.create<sir::MulOp>(loc, u256Type, ivU256, wordSize);
        Value dataOffset = rewriter.create<sir::AddOp>(loc, u256Type, wordBytes, wordSize);
        Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, dataOffset);
        Value wordVal = rewriter.create<sir::LoadOp>(loc, u256Type, dataPtr);

        rewriter.create<sir::SStoreOp>(loc, slotAddr, wordVal);

        Value next = rewriter.create<sir::AddOp>(loc, u256Type, ivU256, one);
        rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

        rewriter.eraseOp(op);
        return success();
    }

    // Convert value to SIR u256 - ALL Ora types must become SIR u256
    Value convertedValue = value;
    if (!llvm::isa<sir::U256Type>(value.getType()))
    {
        if (llvm::isa<ora::IntegerType>(value.getType()))
        {
            // Direct Ora int -> SIR u256 conversion
            convertedValue = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
        }
        else
        {
            Type valueConverted = this->getTypeConverter()->convertType(value.getType());
            if (valueConverted != value.getType() && llvm::isa<sir::U256Type>(valueConverted))
            {
                convertedValue = rewriter.create<sir::BitcastOp>(loc, valueConverted, value);
            }
            else
            {
                // Force to u256
                convertedValue = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
            }
        }
    }

    rewriter.replaceOpWithNewOp<sir::SStoreOp>(op, slot, convertedValue);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.tload → sir.tload
// -----------------------------------------------------------------------------
LogicalResult ConvertTLoadOp::matchAndRewrite(
    ora::TLoadOp op,
    typename ora::TLoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256 = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    auto keyAttr = op->getAttrOfType<StringAttr>("key");
    if (!keyAttr)
    {
        return rewriter.notifyMatchFailure(op, "tload missing key attribute");
    }

    auto slotIndexOpt = computeGlobalSlot(keyAttr.getValue(), op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for tload");
    uint64_t slotIndex = *slotIndexOpt;
    auto slotAttr = mlir::IntegerAttr::get(ui64Type, slotIndex);
    Value slotConst = rewriter.create<sir::ConstOp>(loc, u256, slotAttr);

    Value result = rewriter.create<sir::TLoadOp>(loc, u256, slotConst);
    rewriter.replaceOp(op, result);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.tstore → sir.tstore
// -----------------------------------------------------------------------------
LogicalResult ConvertTStoreOp::matchAndRewrite(
    ora::TStoreOp op,
    typename ora::TStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256 = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    auto keyAttr = op->getAttrOfType<StringAttr>("key");
    if (!keyAttr)
    {
        return rewriter.notifyMatchFailure(op, "tstore missing key attribute");
    }

    auto slotIndexOpt = computeGlobalSlot(keyAttr.getValue(), op.getOperation());
    if (!slotIndexOpt)
        return rewriter.notifyMatchFailure(op, "missing ora.slot_index for tstore");
    uint64_t slotIndex = *slotIndexOpt;
    auto slotAttr = mlir::IntegerAttr::get(ui64Type, slotIndex);
    Value slotConst = rewriter.create<sir::ConstOp>(loc, u256, slotAttr);

    Value value = adaptor.getValue();
    if (!llvm::isa<sir::U256Type>(value.getType()))
    {
        value = rewriter.create<sir::BitcastOp>(loc, u256, value);
    }

    rewriter.replaceOpWithNewOp<sir::TStoreOp>(op, slotConst, value);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.global - convert type attribute from ora.int to u256
// Also assigns sequential slot indices to globals
// -----------------------------------------------------------------------------
LogicalResult ConvertGlobalOp::matchAndRewrite(
    ora::GlobalOp op,
    typename ora::GlobalOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    (void)adaptor;
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Helper: Extract global name from map operand (tensor from ora.sload)
// -----------------------------------------------------------------------------
static llvm::StringRef getGlobalNameFromMapOperand(mlir::Value mapOperand, mlir::Operation *currentOp)
{
    // First, try to find the defining operation
    mlir::Operation *definingOp = mapOperand.getDefiningOp();
    if (auto sloadOp = llvm::dyn_cast<ora::SLoadOp>(definingOp))
    {
        return sloadOp.getGlobalName();
    }
    // If it's a cast, try to follow it
    if (auto castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(definingOp))
    {
        auto inputs = castOp.getInputs();
        if (inputs.size() == 1)
        {
            mlir::Operation *inputOp = inputs[0].getDefiningOp();
            if (auto sloadOp = llvm::dyn_cast<ora::SLoadOp>(inputOp))
            {
                return sloadOp.getGlobalName();
            }
        }
    }

    // If not found, look backwards in the block for the most recent ora.sload
    // This handles the case where ora.sload was already converted to sir.sload
    if (currentOp)
    {
        mlir::Block *block = currentOp->getBlock();
        auto it = mlir::Block::iterator(currentOp);
        // Walk backwards from current operation
        while (it != block->begin())
        {
            --it;
            if (auto sloadOp = llvm::dyn_cast<ora::SLoadOp>(*it))
            {
                // Check if this sload produces a value of the same type as mapOperand
                // (or if it's a tensor type, which is what arrays use)
                if (llvm::isa<mlir::RankedTensorType>(sloadOp.getResult().getType()))
                {
                    return sloadOp.getGlobalName();
                }
            }
        }
    }

    return llvm::StringRef(); // Empty if not found
}

// -----------------------------------------------------------------------------
// Lower ora.map_get → sir.keccak256 + sir.sload
// Pattern: %slot_key = sir.malloc 64
//          sir.store %slot_key, %key
//          %slot_key_plus_32 = sir.addptr %slot_key, 32
//          sir.store %slot_key_plus_32, %mapSlot
//          %hash = sir.keccak256 %slot_key, 64
//          %result = sir.sload %hash
// -----------------------------------------------------------------------------
LogicalResult ConvertMapGetOp::matchAndRewrite(
    ora::MapGetOp op,
    typename ora::MapGetOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    llvm::errs() << "[OraToSIR] ========================================\n";
    llvm::errs() << "[OraToSIR] ConvertMapGetOp::matchAndRewrite() CALLED!\n";
    llvm::errs() << "[OraToSIR]   Operation: " << op->getName() << "\n";
    llvm::errs() << "[OraToSIR]   Location: " << op.getLoc() << "\n";
    llvm::errs() << "[OraToSIR] ========================================\n";
    llvm::errs().flush();

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    // Get the map operand and key
    Value originalMapOperand = op.getMap();
    Value convertedMapOperand = adaptor.getMap();
    Value key = adaptor.getKey();

    // Convert key to SIR u256 if needed
    if (!llvm::isa<sir::U256Type>(key.getType()))
    {
        key = rewriter.create<sir::BitcastOp>(loc, u256Type, key);
    }

    Value mapSlot = Value();
    llvm::StringRef globalName;
    if (llvm::isa<sir::U256Type>(convertedMapOperand.getType()))
    {
        mapSlot = convertedMapOperand;
    }
    else
    {
        // Extract global name from original map operand (before conversion)
        globalName = getGlobalNameFromMapOperand(originalMapOperand, op.getOperation());
        if (globalName.empty())
        {
            DBG("ConvertMapGetOp: failed to find global name from map operand");
            return rewriter.notifyMatchFailure(op, "could not extract global name from map operand");
        }

        // Compute storage slot for the map/array from ora.global operation
        auto mapSlotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
        if (!mapSlotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for map_get");
        uint64_t mapSlotIndex = *mapSlotIndexOpt;
        mapSlot = findOrCreateSlotConstant(op.getOperation(), mapSlotIndex, globalName, rewriter);
        // Set name: "slot_" + globalName
        if (auto slotOp = mapSlot.getDefiningOp<sir::ConstOp>())
        {
            setResultName(slotOp, 0, ("slot_" + globalName).str());
        }
    }

    // Allocate 64 bytes for key + slot
    auto size64Attr = mlir::IntegerAttr::get(ui64Type, 64ULL);
    Value size64 = rewriter.create<sir::ConstOp>(loc, u256Type, size64Attr);
    Value slotKey = rewriter.create<sir::MallocOp>(loc, ptrType, size64);
    setResultName(slotKey.getDefiningOp(), 0, "ptr");

    // Store key at offset 0
    rewriter.create<sir::StoreOp>(loc, slotKey, key);

    // Store map slot at offset 32
    auto offset32Attr = mlir::IntegerAttr::get(ui64Type, 32ULL);
    Value offset32 = rewriter.create<sir::ConstOp>(loc, u256Type, offset32Attr);
    Value slotKeyPlus32 = rewriter.create<sir::AddPtrOp>(loc, ptrType, slotKey, offset32);
    setResultName(slotKeyPlus32.getDefiningOp(), 0, "ptr_off");
    rewriter.create<sir::StoreOp>(loc, slotKeyPlus32, mapSlot);

    // Compute keccak256 hash
    Value hash = rewriter.create<sir::KeccakOp>(loc, u256Type, slotKey, size64);
    if (!globalName.empty())
    {
        setResultName(hash.getDefiningOp(), 0, ("hash_" + globalName).str());
    }

    // Get the expected result type from ora.map_get and convert it
    Type expectedResultType = op.getResult().getType();

    // If the map value is another map, return the derived slot hash as a map handle
    if (llvm::isa<ora::MapType>(expectedResultType))
    {
        rewriter.replaceOp(op, hash);
        DBG("ConvertMapGetOp: map-of-map, returning derived slot hash");
        return success();
    }

    // If the result type is a struct, handle it specially by loading multiple storage slots
    if (auto structType = llvm::dyn_cast<ora::StructType>(expectedResultType))
    {
        llvm::StringRef structName = structType.getName();
        DBG("ConvertMapGetOp: Handling struct result type: " << structName);

        // Find the struct declaration in the module to get field information
        ModuleOp module = op->getParentOfType<ModuleOp>();
        if (!module)
        {
            return rewriter.notifyMatchFailure(op, "could not find module for struct field lookup");
        }

        // Look for ora.struct.decl operation with matching name
        ora::StructDeclOp structDecl = nullptr;
        module.walk([&](ora::StructDeclOp declOp)
                    {
            auto declNameAttr = declOp->getAttrOfType<StringAttr>("name");
            if (declNameAttr && declNameAttr.getValue() == structName)
            {
                structDecl = declOp;
                return WalkResult::interrupt();
            }
            return WalkResult::advance(); });

        if (!structDecl)
        {
            llvm::errs() << "[OraToSIR] ConvertMapGetOp: Could not find struct declaration for: " << structName << "\n";
            llvm::errs().flush();
            return rewriter.notifyMatchFailure(op, "could not find struct declaration");
        }

        // Get field names and types from attributes
        auto fieldNamesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_names");
        auto fieldTypesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_types");

        if (!fieldNamesAttr || !fieldTypesAttr ||
            fieldNamesAttr.size() != fieldTypesAttr.size())
        {
            llvm::errs() << "[OraToSIR] ConvertMapGetOp: Invalid field attributes for struct: " << structName << "\n";
            llvm::errs().flush();
            return rewriter.notifyMatchFailure(op, "invalid struct field attributes");
        }

        size_t numFields = fieldNamesAttr.size();
        DBG("ConvertMapGetOp: Found " << numFields << " fields for struct " << structName);

        // Load each field from consecutive storage slots
        SmallVector<Value> fieldValues;
        for (size_t i = 0; i < numFields; ++i)
        {
            // Calculate storage slot: base hash + field index
            // Each field occupies one storage slot (32 bytes) in EVM
            Value fieldSlot = hash;
            if (i > 0)
            {
                // Add field index to base hash for subsequent fields
                auto fieldIndexAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(i));
                Value fieldIndex = rewriter.create<sir::ConstOp>(loc, u256Type, fieldIndexAttr);
                fieldSlot = rewriter.create<sir::AddOp>(loc, u256Type, hash, fieldIndex);
            }

            // Get field type and convert it
            Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
            Type convertedFieldType = this->getTypeConverter()->convertType(fieldType);
            if (!convertedFieldType)
            {
                // Fallback to u256 if conversion fails
                convertedFieldType = u256Type;
            }

            // Load field from storage
            Value fieldValue = rewriter.create<sir::SLoadOp>(loc, convertedFieldType, fieldSlot);
            StringRef fieldName = cast<StringAttr>(fieldNamesAttr[i]).getValue();
            std::string fieldNameStr = "field_" + structName.str() + "_" + fieldName.str();
            setResultName(fieldValue.getDefiningOp(), 0, fieldNameStr);

            fieldValues.push_back(fieldValue);
        }

        // Reconstruct struct using ora.struct_init
        // Note: We need to convert field values back to Ora types for struct_init
        // For now, create the struct_init operation with the loaded values
        // The type converter will handle converting back to Ora types if needed
        auto structInitOp = rewriter.create<ora::StructInitOp>(
            loc,
            expectedResultType, // Keep original struct type
            fieldValues);

        DBG("ConvertMapGetOp: Reconstructed struct " << structName << " from " << numFields << " fields");

        // Replace the map_get with the reconstructed struct
        rewriter.replaceOp(op, structInitOp.getResult());
        return success();
    }

    // Non-struct types: convert and load normally
    Type convertedResultType = this->getTypeConverter()->convertType(expectedResultType);

    // If type conversion failed, use u256 as fallback
    if (!convertedResultType)
    {
        llvm::errs() << "[OraToSIR] ConvertMapGetOp: Type conversion failed for result type: " << expectedResultType << "\n";
        llvm::errs() << "[OraToSIR]   Using u256 as fallback\n";
        llvm::errs().flush();
        convertedResultType = u256Type;
    }

    // Load from storage using the hash
    Value result = rewriter.create<sir::SLoadOp>(loc, convertedResultType, hash);
    setResultName(result.getDefiningOp(), 0, "value");

    // Replace the map_get with the result
    rewriter.replaceOp(op, result);
    DBG("ConvertMapGetOp: replaced with keccak256 + sload");
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.map_store → sir.keccak256 + sir.sstore
// Pattern: %slot_key = sir.malloc 64
//          sir.store %slot_key, %key
//          %slot_key_plus_32 = sir.addptr %slot_key, 32
//          sir.store %slot_key_plus_32, %mapSlot
//          %hash = sir.keccak256 %slot_key, 64
//          sir.sstore %hash, %value
// -----------------------------------------------------------------------------
LogicalResult ConvertMapStoreOp::matchAndRewrite(
    ora::MapStoreOp op,
    typename ora::MapStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    llvm::errs() << "[OraToSIR] ========================================\n";
    llvm::errs() << "[OraToSIR] ConvertMapStoreOp::matchAndRewrite() CALLED!\n";
    llvm::errs() << "[OraToSIR]   Operation: " << op->getName() << "\n";
    llvm::errs() << "[OraToSIR]   Location: " << op.getLoc() << "\n";
    llvm::errs() << "[OraToSIR] ========================================\n";
    llvm::errs().flush();

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    // Get the map operand, key, and value
    Value originalMapOperand = op.getMap();
    Value convertedMapOperand = adaptor.getMap();
    Value key = adaptor.getKey();
    Value value = adaptor.getValue();

    // Convert key and value to SIR u256 if needed
    if (!llvm::isa<sir::U256Type>(key.getType()))
    {
        key = rewriter.create<sir::BitcastOp>(loc, u256Type, key);
    }
    if (!llvm::isa<sir::U256Type>(value.getType()))
    {
        value = rewriter.create<sir::BitcastOp>(loc, u256Type, value);
    }

    if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(originalMapOperand.getType()))
    {
        Value baseSlot = convertedMapOperand;
        if (!llvm::isa<sir::U256Type>(baseSlot.getType()))
        {
            return rewriter.notifyMatchFailure(op, "array base slot is not u256");
        }
        Value indexU256 = ensureU256Value(rewriter, loc, key);
        uint64_t elemWords = getElementWordCount(tensorType.getElementType());
        Value slot = Value();
        if (tensorType.hasStaticShape())
        {
            if (elemWords != 1)
            {
                auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
                Value elemWordsConst = rewriter.create<sir::ConstOp>(
                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, elemWords));
                Value offset = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, elemWordsConst);
                slot = rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, offset);
            }
            else
            {
                slot = rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, indexU256);
            }
        }
        else
        {
            auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
            Value size32 = rewriter.create<sir::ConstOp>(
                loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
            auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
            Value tmp = rewriter.create<sir::MallocOp>(loc, ptrType, size32);
            rewriter.create<sir::StoreOp>(loc, tmp, baseSlot);
            Value hash = rewriter.create<sir::KeccakOp>(loc, u256Type, tmp, size32);
            if (elemWords != 1)
            {
                auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
                Value elemWordsConst = rewriter.create<sir::ConstOp>(
                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, elemWords));
                Value offset = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, elemWordsConst);
                slot = rewriter.create<sir::AddOp>(loc, u256Type, hash, offset);
            }
            else
            {
                slot = rewriter.create<sir::AddOp>(loc, u256Type, hash, indexU256);
            }
        }
        rewriter.replaceOpWithNewOp<sir::SStoreOp>(op, slot, value);
        return success();
    }

    Value mapSlot = Value();
    llvm::StringRef globalName;
    if (llvm::isa<sir::U256Type>(convertedMapOperand.getType()))
    {
        mapSlot = convertedMapOperand;
    }
    else
    {
        // Extract global name from original map operand (before conversion)
        globalName = getGlobalNameFromMapOperand(originalMapOperand, op.getOperation());
        DBG("ConvertMapStoreOp: globalName = " << (globalName.empty() ? "<empty>" : globalName.str()));
        if (globalName.empty())
        {
            DBG("ConvertMapStoreOp: failed to find global name from map operand, trying backwards search");
            // Try a simpler approach: look for the most recent ora.sload in the block
            mlir::Block *block = op->getBlock();
            auto it = mlir::Block::iterator(op.getOperation());
            while (it != block->begin())
            {
                --it;
                if (auto sloadOp = llvm::dyn_cast<ora::SLoadOp>(*it))
                {
                    globalName = sloadOp.getGlobalName();
                    DBG("ConvertMapStoreOp: found global name via backwards search: " << globalName);
                    break;
                }
            }
        }
        if (globalName.empty())
        {
            DBG("ConvertMapStoreOp: failed to find global name from map operand");
            return rewriter.notifyMatchFailure(op, "could not extract global name from map operand");
        }

        // Compute storage slot for the map/array from ora.global operation
        auto mapSlotIndexOpt = computeGlobalSlot(globalName, op.getOperation());
        if (!mapSlotIndexOpt)
            return rewriter.notifyMatchFailure(op, "missing ora.slot_index for map_store");
        uint64_t mapSlotIndex = *mapSlotIndexOpt;
        mapSlot = findOrCreateSlotConstant(op.getOperation(), mapSlotIndex, globalName, rewriter);
        // Set name: "slot_" + globalName
        if (auto slotOp = mapSlot.getDefiningOp<sir::ConstOp>())
        {
            setResultName(slotOp, 0, ("slot_" + globalName).str());
        }
    }

    // Allocate 64 bytes for key + slot
    auto size64Attr = mlir::IntegerAttr::get(ui64Type, 64ULL);
    Value size64 = rewriter.create<sir::ConstOp>(loc, u256Type, size64Attr);
    Value slotKey = rewriter.create<sir::MallocOp>(loc, ptrType, size64);
    setResultName(slotKey.getDefiningOp(), 0, "ptr");

    // Store key at offset 0
    rewriter.create<sir::StoreOp>(loc, slotKey, key);

    // Store map slot at offset 32
    auto offset32Attr = mlir::IntegerAttr::get(ui64Type, 32ULL);
    Value offset32 = rewriter.create<sir::ConstOp>(loc, u256Type, offset32Attr);
    Value slotKeyPlus32 = rewriter.create<sir::AddPtrOp>(loc, ptrType, slotKey, offset32);
    setResultName(slotKeyPlus32.getDefiningOp(), 0, "ptr_off");
    rewriter.create<sir::StoreOp>(loc, slotKeyPlus32, mapSlot);

    // Compute keccak256 hash
    Value hash = rewriter.create<sir::KeccakOp>(loc, u256Type, slotKey, size64);
    if (!globalName.empty())
    {
        setResultName(hash.getDefiningOp(), 0, ("hash_" + globalName).str());
    }

    // Store to storage using the hash
    rewriter.create<sir::SStoreOp>(loc, hash, value);

    // Erase the map_store operation (it has no results)
    rewriter.eraseOp(op);
    DBG("ConvertMapStoreOp: replaced with keccak256 + sstore");
    return success();
}

// -----------------------------------------------------------------------------
// Lower tensor.extract for storage arrays -> sir.sload base_slot + index
// -----------------------------------------------------------------------------
LogicalResult ConvertTensorExtractOp::matchAndRewrite(
    mlir::tensor::ExtractOp op,
    typename mlir::tensor::ExtractOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(op.getTensor().getType());
    if (!tensorType)
        return rewriter.notifyMatchFailure(op, "tensor.extract without ranked tensor");
    if (tensorType.getRank() != 1)
        return rewriter.notifyMatchFailure(op, "only 1D arrays supported");

    Value base = adaptor.getTensor();
    if (!llvm::isa<sir::U256Type>(base.getType()))
        return rewriter.notifyMatchFailure(op, "array base is not u256");

    if (adaptor.getIndices().size() != 1)
        return rewriter.notifyMatchFailure(op, "expected single index");

    Value indexU256 = ensureU256Value(rewriter, loc, adaptor.getIndices()[0]);
    uint64_t elemWords = getElementWordCount(tensorType.getElementType());

    Value loaded = Value();
    bool isStorageArray = false;
    if (auto def = op.getTensor().getDefiningOp())
    {
        if (llvm::isa<ora::SLoadOp>(def))
            isStorageArray = true;
    }

    if (tensorType.hasStaticShape() || isStorageArray)
    {
        Value baseSlot = base;
        if (!tensorType.hasStaticShape())
        {
            auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
            Value size32 = rewriter.create<sir::ConstOp>(
                loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
            auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
            Value tmp = rewriter.create<sir::MallocOp>(loc, ptrType, size32);
            rewriter.create<sir::StoreOp>(loc, tmp, baseSlot);
            baseSlot = rewriter.create<sir::KeccakOp>(loc, u256Type, tmp, size32);
        }
        Value slot = baseSlot;
        if (elemWords != 1)
        {
            auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
            Value elemWordsConst = rewriter.create<sir::ConstOp>(
                loc, u256Type, mlir::IntegerAttr::get(ui64Type, elemWords));
            Value offset = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, elemWordsConst);
            slot = rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, offset);
        }
        else
        {
            slot = rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, indexU256);
        }
        loaded = rewriter.create<sir::SLoadOp>(loc, u256Type, slot);
    }
    else
    {
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        Value ptr = rewriter.create<sir::BitcastOp>(loc, ptrType, base);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
        Value wordSize = rewriter.create<sir::ConstOp>(
            loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
        Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, ptr, wordSize);
        Value offsetBytes = rewriter.create<sir::MulOp>(loc, u256Type, indexU256, wordSize);
        Value elemPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, dataPtr, offsetBytes);
        loaded = rewriter.create<sir::LoadOp>(loc, u256Type, elemPtr);
    }

    Type desiredType = u256Type;
    if (auto *tc = getTypeConverter())
    {
        if (Type converted = tc->convertType(op.getType()))
            desiredType = converted;
    }
    if (desiredType != u256Type)
        loaded = rewriter.create<sir::BitcastOp>(loc, desiredType, loaded);

    rewriter.replaceOp(op, loaded);
    return success();
}

// -----------------------------------------------------------------------------
// Lower tensor.dim for arrays/slices
// -----------------------------------------------------------------------------
LogicalResult ConvertTensorDimOp::matchAndRewrite(
    mlir::tensor::DimOp op,
    typename mlir::tensor::DimOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(op.getSource().getType());
    if (!tensorType)
        return rewriter.notifyMatchFailure(op, "tensor.dim on non-ranked tensor");

    if (tensorType.hasStaticShape())
    {
        int64_t dim = tensorType.getDimSize(0);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
        auto u256Type = sir::U256Type::get(ctx);
        Value dimConst = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, dim));
        Value idx = buildIndexFromU256(rewriter, loc, dimConst);
        rewriter.replaceOp(op, idx);
        return success();
    }

    Value base = adaptor.getSource();
    if (!llvm::isa<sir::U256Type>(base.getType()))
        return rewriter.notifyMatchFailure(op, "slice base is not u256");
    auto u256Type = sir::U256Type::get(ctx);

    bool isStorageArray = false;
    if (auto def = op.getSource().getDefiningOp())
    {
        if (llvm::isa<ora::SLoadOp>(def))
            isStorageArray = true;
    }

    if (isStorageArray)
    {
        Value length = rewriter.create<sir::SLoadOp>(loc, u256Type, base);
        Value idx = buildIndexFromU256(rewriter, loc, length);
        rewriter.replaceOp(op, idx);
        return success();
    }

    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    Value ptr = rewriter.create<sir::BitcastOp>(loc, ptrType, base);
    Value length = rewriter.create<sir::LoadOp>(loc, u256Type, ptr);
    Value idx = buildIndexFromU256(rewriter, loc, length);
    rewriter.replaceOp(op, idx);
    return success();
}
