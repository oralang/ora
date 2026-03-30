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
        /// Op indices follow the serialized backend-execution order used by
        /// bytecode/source-map generation, not raw textual .sir line order.
        std::string extractSIRLocations(ModuleOp module);

        /// Extract debugger-oriented sidecar metadata from SIR MLIR ops as JSON.
        /// Returns a JSON object with per-op entries keyed by the same op indices
        /// used by extractSIRLocations() and bytecode/source-map generation.
        std::string extractSIRDebugInfo(ModuleOp module);

        /// Extract a map from serialized backend op indices to emitted SIR text lines.
        /// Returns a JSON array of {idx, line} entries. The idx values follow the same
        /// backend-execution ordering used by extractSIRLocations()/debug info, while
        /// the line values point into the textual .sir artifact emitted by emitSIRText().
        std::string extractSIRLineMap(ModuleOp module);
    } // namespace ora
} // namespace mlir

#endif // ORA_LOWERING_SIRTEXTEMITTER_H
