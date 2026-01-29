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
    } // namespace ora
} // namespace mlir

#endif // ORA_LOWERING_SIRTEXTEMITTER_H
