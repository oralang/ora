//===- SIRTextEmitter.cpp - SIR Text Emitter -------------------------===//
//
// Minimal stub emitter. Replace with full serializer once legalizer is in place.
//
//===----------------------------------------------------------------------===//

#include "SIRTextEmitter.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir
{
    namespace ora
    {
        std::string emitSIRText(ModuleOp module)
        {
            (void)module;
            std::string out;
            llvm::raw_string_ostream os(out);
            os << "// SIR text emitter not implemented yet\n";
            os.flush();
            return out;
        }
    } // namespace ora
} // namespace mlir
