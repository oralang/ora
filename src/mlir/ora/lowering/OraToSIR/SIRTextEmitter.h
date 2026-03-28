//===- SIRTextEmitter.h - SIR Text Emitter ---------------------------===//
//
// Emits Sensei SIR text from SIR MLIR. The emitter is a pure serializer:
// it assumes the input module already satisfies SIR text constraints.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_LOWERING_SIRTEXTEMITTER_H
#define ORA_LOWERING_SIRTEXTEMITTER_H

#include <string>

namespace mlir
{
    class ModuleOp;
}

namespace mlir
{
    namespace ora
    {
        /// Emit Sensei SIR text for a module. Input must be SIR-legal.
        std::string emitSIRText(ModuleOp module);

        /// Extract source locations from SIR MLIR ops as JSON.
        /// Returns a JSON array of {idx, file, line, col} entries.
        /// Op indices match the sequential order of emitSIRText() output.
        std::string extractSIRLocations(ModuleOp module);

        /// Extract debugger-oriented sidecar metadata from SIR MLIR ops as JSON.
        /// Returns a JSON object with per-op entries keyed by the same op indices
        /// used by emitSIRText() and extractSIRLocations().
        std::string extractSIRDebugInfo(ModuleOp module);
    } // namespace ora
} // namespace mlir

#endif // ORA_LOWERING_SIRTEXTEMITTER_H
