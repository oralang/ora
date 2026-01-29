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

        /// Create a cleanup pass to remove dead operations (unused allocas, etc.)
        std::unique_ptr<Pass> createMemRefEliminationPass();
        std::unique_ptr<Pass> createSIRCleanupPass();

        /// Create an optimization pass for SIR operations
        std::unique_ptr<Pass> createSIROptimizationPass();
        std::unique_ptr<Pass> createSimpleDCEPass();

        /// Create a legalizer pass to validate SIR MLIR for text emission
        std::unique_ptr<Pass> createSIRTextLegalizerPass();

        /// Create a pass to optimize Ora operations (constant deduplication, constant folding)
        std::unique_ptr<Pass> createOraOptimizationPass();

        /// Create a pass to clean up unused Ora operations
        std::unique_ptr<Pass> createOraCleanupPass();

        /// Create a pass to run canonicalization and DCE on Ora MLIR functions
        std::unique_ptr<Pass> createSimpleOraOptimizationPass();

        /// Create a pass to inline functions marked with ora.inline attribute
        std::unique_ptr<Pass> createOraInliningPass();

        /// Legacy alias for backward compatibility
        std::unique_ptr<Pass> createOraCanonicalizationPass();

    } // namespace ora
} // namespace mlir

#endif // ORA_LOWERING_ORATOSIR_H
