//===- OraToSIR.h - Ora to SIR Conversion Pass ----------------------===//
//
// This file declares the conversion pass from Ora dialect to SIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_LOWERING_ORATOSIR_H
#define ORA_LOWERING_ORATOSIR_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
    namespace ora
    {

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

        /// Create a pass to run canonicalization on nested Ora MLIR functions
        std::unique_ptr<Pass> createOraFunctionCanonicalizerPass();

        /// Create a pass to run MLIR CSE on nested Ora MLIR functions
        std::unique_ptr<Pass> createOraFunctionCSEPass();

        /// Create a storage-aware pass to reuse repeated Ora storage loads
        std::unique_ptr<Pass> createOraStorageReadCSEPass();

    } // namespace ora
} // namespace mlir

#endif // ORA_LOWERING_ORATOSIR_H
