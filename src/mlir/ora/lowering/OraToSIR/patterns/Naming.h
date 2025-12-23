#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <string>

namespace mlir
{
    namespace ora
    {

        /// Helper class for generating consistent, human-readable names for SIR operations
        class SIRNamingHelper
        {
        public:
            /// Context types for different allocation scenarios
            enum class Context
            {
                ArrayAllocation, /// memref.alloca for arrays
                ReturnBuffer,    /// malloc for return values
                General          /// Other allocations
            };

            SIRNamingHelper() = default;

            /// Set context for current operation
            void setContext(Context ctx) { currentContext = ctx; }

            /// Reset counters (call at start of each function)
            void reset()
            {
                nextElemIndex = 0;
                nextOffsetIndex = 0;
                nextIndexNum = 0;
                computedValueCounters.clear();
                currentContext = Context::General;
            }

            // =====================================================================
            // Memory & Pointer Naming
            // =====================================================================

            /// Name malloc operation based on context
            /// - ArrayAllocation: %arr_base
            /// - ReturnBuffer: %ret_ptr
            /// - General: %base
            void nameMalloc(Operation *op, unsigned resultIndex, Context ctx = Context::General);

            /// Name constant operation
            /// - Special values: %elem_size (32), %alloc_size, %ret_len (32)
            /// - Index constants: %idx{i}
            /// - Raw constants: %c{value}
            void nameConst(Operation *op, unsigned resultIndex, int64_t value, StringRef purpose = "");

            /// Name addptr operation as %elem{i}_ptr
            void nameAddPtr(Operation *op, unsigned resultIndex, unsigned elemIndex);

            /// Name load operation as %elem{i}
            void nameLoad(Operation *op, unsigned resultIndex, unsigned elemIndex);

            /// Name offset calculation as %offset{i}
            void nameOffset(Operation *op, unsigned resultIndex, unsigned offsetIndex);

            /// Name index constant as %idx{i}
            void nameIndex(Operation *op, unsigned resultIndex, unsigned indexNum);

            // =====================================================================
            // Computed Values
            // =====================================================================

            /// Name computed value (sum, product, etc.)
            /// Generates: %{baseName}, %{baseName}2, %{baseName}3, etc.
            void nameComputedValue(Operation *op, unsigned resultIndex, StringRef baseName, unsigned variant = 0);

            // =====================================================================
            // Storage Naming
            // =====================================================================

            /// Name storage slot constant as %slot_{name}
            void nameSlotConstant(Operation *op, unsigned resultIndex, StringRef globalName);

            /// Name loaded storage value as %{name}_val
            void nameStorageValue(Operation *op, unsigned resultIndex, StringRef globalName);

            /// Name updated storage value as %{name}_new
            void nameStorageNew(Operation *op, unsigned resultIndex, StringRef globalName);

            /// Name keccak hash as %hash_{name}
            void nameHash(Operation *op, unsigned resultIndex, StringRef mapName);

            // =====================================================================
            // Return Values
            // =====================================================================

            /// Name return pointer as %ret_ptr
            void nameReturnPtr(Operation *op, unsigned resultIndex);

            /// Name return length as %ret_len
            void nameReturnLen(Operation *op, unsigned resultIndex);

            /// Name return value as %ret_val
            void nameReturnVal(Operation *op, unsigned resultIndex);

            // =====================================================================
            // Utility Methods
            // =====================================================================

            /// Extract element index from constant index value
            /// Returns -1 if index cannot be extracted
            int64_t extractElementIndex(Value indexValue) const;

            /// Get next element index (auto-increment)
            unsigned getNextElemIndex();

            /// Get next offset index (auto-increment)
            unsigned getNextOffsetIndex();

            /// Get next index number (auto-increment)
            unsigned getNextIndexNum();

            /// Get variant number for computed value (auto-increment)
            unsigned getNextVariant(StringRef baseName);

        private:
            Context currentContext = Context::General;
            unsigned nextElemIndex = 0;
            unsigned nextOffsetIndex = 0;
            unsigned nextIndexNum = 0;
            std::map<std::string, unsigned> computedValueCounters;

            /// Internal helper to set result name attribute
            void setResultName(Operation *op, unsigned resultIndex, StringRef name) const;
        };

    } // namespace ora
} // namespace mlir

