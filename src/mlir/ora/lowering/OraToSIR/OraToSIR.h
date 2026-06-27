//===- OraToSIR.h - Ora to SIR Conversion Pass ----------------------===//
//
// This file declares the conversion pass from Ora dialect to SIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_LOWERING_ORATOSIR_H
#define ORA_LOWERING_ORATOSIR_H

#include "mlir/Pass/Pass.h"

#include <cstdint>

namespace llvm
{
    class raw_ostream;
}

namespace mlir
{
    namespace ora
    {
        struct OraMlirPassStatistics
        {
            uint64_t oraFunctionsCanonicalized = 0;
            uint64_t oraFunctionsCSEProcessed = 0;
            uint64_t oraStorageReadsReused = 0;
            uint64_t oraCallsInlined = 0;
            uint64_t oraSourceInlineFailures = 0;
            uint64_t sirConstantsDeduplicated = 0;
            uint64_t sirUnusedAllocasRemoved = 0;
            uint64_t sirUnusedLoadsRemoved = 0;
            uint64_t sirUnusedPureOpsRemoved = 0;
            uint64_t sirFrameworkFunctionsProcessed = 0;
            uint64_t oraSymbolsDCEd = 0;
        };

        enum class OraMlirPassStatistic
        {
            OraFunctionsCanonicalized,
            OraFunctionsCSEProcessed,
            OraStorageReadsReused,
            OraCallsInlined,
            OraSourceInlineFailures,
            SirConstantsDeduplicated,
            SirUnusedAllocasRemoved,
            SirUnusedLoadsRemoved,
            SirUnusedPureOpsRemoved,
            SirFrameworkFunctionsProcessed,
            OraSymbolsDCEd,
        };

        /// Install a per-thread statistics sink used by Ora-owned passes. This is
        /// independent of LLVM's optional statistics build flag so
        /// --mlir-debug=statistics remains useful in the default Ora toolchain.
        void setActiveOraMlirPassStatistics(OraMlirPassStatistics *statistics);
        void recordOraMlirPassStatistic(OraMlirPassStatistic statistic, uint64_t amount = 1);
        void printOraMlirPassStatistics(const OraMlirPassStatistics &statistics, llvm::raw_ostream &os, const char *pipelineName);

        /// Create a pass to convert Ora dialect operations to SIR dialect
        std::unique_ptr<Pass> createOraToSIRPass();

        /// Create a narrow deterministic SIR cleanup pass for release hygiene.
        std::unique_ptr<Pass> createSIRCleanupPass();

        /// Create an optimization pass for SIR operations
        std::unique_ptr<Pass> createSIROptimizationPass();
        std::unique_ptr<Pass> createSIRFrameworkCanonicalizerPass();

        /// Create a legalizer pass to validate SIR MLIR for text emission
        std::unique_ptr<Pass> createSIRTextLegalizerPass();

        /// Create a pass to build a Solidity-style dispatcher for public functions
        std::unique_ptr<Pass> createSIRDispatcherPass();

        /// Create a pass to inline functions marked with ora.inline attribute
        std::unique_ptr<Pass> createOraInliningPass();

        /// Create Ora-owned wrappers around framework SymbolDCE. The visibility
        /// pass translates Ora function/root metadata into temporary MLIR symbol
        /// visibility, and the cleanup pass removes that temporary metadata
        /// after the framework pass runs while recording deterministic counts.
        std::unique_ptr<Pass> createOraSymbolVisibilityPass();
        std::unique_ptr<Pass> createOraSymbolDCECleanupPass();

        /// Create a pass to run canonicalization on nested Ora MLIR functions
        std::unique_ptr<Pass> createOraFunctionCanonicalizerPass();

        /// Create a pass to run MLIR CSE on nested Ora MLIR functions
        std::unique_ptr<Pass> createOraFunctionCSEPass();

        /// Create a storage-aware pass to reuse repeated Ora storage loads
        std::unique_ptr<Pass> createOraStorageReadCSEPass();

    } // namespace ora
} // namespace mlir

#endif // ORA_LOWERING_ORATOSIR_H
