#include "patterns/Naming.h"

#include "SIR/SIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace ora;

// =====================================================================
// Internal Helper
// =====================================================================

void SIRNamingHelper::setResultName(Operation *op, unsigned resultIndex, StringRef name) const
{
    if (!op)
        return;
    auto nameAttr = StringAttr::get(op->getContext(), name);
    std::string attrName = "sir.result_name_" + std::to_string(resultIndex);
    op->setAttr(attrName, nameAttr);
}

// =====================================================================
// Memory & Pointer Naming
// =====================================================================

void SIRNamingHelper::nameMalloc(Operation *op, unsigned resultIndex, Context ctx)
{
    StringRef name;
    switch (ctx)
    {
    case Context::ArrayAllocation:
        name = "arr_base";
        break;
    case Context::ReturnBuffer:
        name = "ret_ptr";
        break;
    case Context::General:
    default:
        name = "base";
        break;
    }
    setResultName(op, resultIndex, name);
}

void SIRNamingHelper::nameConst(Operation *op, unsigned resultIndex, int64_t value, StringRef purpose)
{
    StringRef name;

    // Special-purpose constants
    if (purpose == "elem_size")
    {
        name = "elem_size";
    }
    else if (purpose == "alloc_size")
    {
        name = "alloc_size";
    }
    else if (purpose == "ret_len")
    {
        name = "ret_len";
    }
    else if (purpose == "index")
    {
        // Index constants: %idx{i}
        unsigned idxNum = getNextIndexNum();
        std::string idxName = "idx" + std::to_string(idxNum);
        setResultName(op, resultIndex, idxName);
        return;
    }
    else
    {
        // Raw constants: %c{value} (only for reasonable-sized values)
        // This method is only called for values <= MAX_NAMED_CONSTANT
        std::string constName = "c" + std::to_string(value);
        setResultName(op, resultIndex, constName);
        return;
    }

    setResultName(op, resultIndex, name);
}

void SIRNamingHelper::nameAddPtr(Operation *op, unsigned resultIndex, unsigned elemIndex)
{
    std::string name = "elem" + std::to_string(elemIndex) + "_ptr";
    setResultName(op, resultIndex, name);
}

void SIRNamingHelper::nameLoad(Operation *op, unsigned resultIndex, unsigned elemIndex)
{
    std::string name = "elem" + std::to_string(elemIndex);
    setResultName(op, resultIndex, name);
}

void SIRNamingHelper::nameOffset(Operation *op, unsigned resultIndex, unsigned offsetIndex)
{
    std::string name = "offset" + std::to_string(offsetIndex);
    setResultName(op, resultIndex, name);
}

void SIRNamingHelper::nameIndex(Operation *op, unsigned resultIndex, unsigned indexNum)
{
    std::string name = "idx" + std::to_string(indexNum);
    setResultName(op, resultIndex, name);
}

// =====================================================================
// Computed Values
// =====================================================================

void SIRNamingHelper::nameComputedValue(Operation *op, unsigned resultIndex, StringRef baseName, unsigned variant)
{
    std::string name;
    if (variant == 0)
    {
        // First occurrence: %{baseName}
        name = baseName.str();
    }
    else
    {
        // Subsequent: %{baseName}{variant}
        name = baseName.str() + std::to_string(variant);
    }
    setResultName(op, resultIndex, name);
}

unsigned SIRNamingHelper::getNextVariant(StringRef baseName)
{
    std::string key = baseName.str();
    auto it = computedValueCounters.find(key);
    if (it == computedValueCounters.end())
    {
        computedValueCounters[key] = 2; // Next variant will be 2
        return 0;                        // First variant (no suffix)
    }
    else
    {
        unsigned variant = it->second;
        it->second++;
        return variant;
    }
}

// =====================================================================
// Storage Naming
// =====================================================================

void SIRNamingHelper::nameSlotConstant(Operation *op, unsigned resultIndex, StringRef globalName)
{
    std::string name = "slot_" + globalName.str();
    setResultName(op, resultIndex, name);
}

void SIRNamingHelper::nameStorageValue(Operation *op, unsigned resultIndex, StringRef globalName)
{
    std::string name = globalName.str() + "_val";
    setResultName(op, resultIndex, name);
}

void SIRNamingHelper::nameStorageNew(Operation *op, unsigned resultIndex, StringRef globalName)
{
    std::string name = globalName.str() + "_new";
    setResultName(op, resultIndex, name);
}

void SIRNamingHelper::nameHash(Operation *op, unsigned resultIndex, StringRef mapName)
{
    std::string name = "hash_" + mapName.str();
    setResultName(op, resultIndex, name);
}

// =====================================================================
// Return Values
// =====================================================================

void SIRNamingHelper::nameReturnPtr(Operation *op, unsigned resultIndex)
{
    setResultName(op, resultIndex, "ret_ptr");
}

void SIRNamingHelper::nameReturnLen(Operation *op, unsigned resultIndex)
{
    setResultName(op, resultIndex, "ret_len");
}

void SIRNamingHelper::nameReturnVal(Operation *op, unsigned resultIndex)
{
    setResultName(op, resultIndex, "ret_val");
}

// =====================================================================
// Utility Methods
// =====================================================================

int64_t SIRNamingHelper::extractElementIndex(Value indexValue) const
{
    // Try to extract constant index value
    Operation *defOp = indexValue.getDefiningOp();
    if (!defOp)
        return -1;

    // Check if it's a constant operation (arith.constant)
    if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(defOp))
    {
        auto attr = constOp.getValue();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
        {
            return intAttr.getInt();
        }
    }
    // Check if it's a sir.const operation
    else if (auto sirConstOp = dyn_cast<sir::ConstOp>(defOp))
    {
        auto attr = sirConstOp.getValueAttr();
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
        {
            return intAttr.getInt();
        }
    }
    // Check if it's a bitcast - follow to the source
    else if (auto bitcastOp = dyn_cast<sir::BitcastOp>(defOp))
    {
        Value input = bitcastOp.getInput();
        return extractElementIndex(input); // Recursively extract from source
    }

    return -1;
}

unsigned SIRNamingHelper::getNextElemIndex()
{
    return nextElemIndex++;
}

unsigned SIRNamingHelper::getNextOffsetIndex()
{
    return nextOffsetIndex++;
}

unsigned SIRNamingHelper::getNextIndexNum()
{
    return nextIndexNum++;
}

