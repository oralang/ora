//===- SIRTextLegalizer.cpp - SIR Text Legalizer Pass ----------------===//
//
// Validates SIR MLIR against constraints required by the Sensei text format.
// This pass should be run after Ora -> SIR conversion and before text emission.
//
//===----------------------------------------------------------------------===//

#include "OraToSIR.h"

#include "SIR/SIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir
{
    namespace ora
    {

        namespace
        {
            struct SIRTextLegalizerPass : public PassWrapper<SIRTextLegalizerPass, OperationPass<ModuleOp>>
            {
                void runOnOperation() override
                {
                    ModuleOp module = getOperation();
                    bool failed_any = false;

                    auto isAllowedDialect = [](Dialect *dialect) {
                        if (!dialect)
                            return false;
                        StringRef ns = dialect->getNamespace();
                        return ns == "builtin" || ns == "func" || ns == "sir";
                    };

                    module.walk([&](Operation *op) {
                        if (!isAllowedDialect(op->getDialect()))
                        {
                            op->emitError() << "illegal dialect for SIR text: "
                                            << (op->getDialect() ? op->getDialect()->getNamespace() : "null");
                            failed_any = true;
                        }
                    });

                    for (func::FuncOp func : module.getOps<func::FuncOp>())
                    {
                        for (Block &block : func.getBlocks())
                        {
                            if (block.empty())
                            {
                                func.emitError() << "empty block in function: " << func.getName();
                                failed_any = true;
                                continue;
                            }

                            Operation &terminator = block.back();
                            if (!terminator.hasTrait<OpTrait::IsTerminator>())
                            {
                                terminator.emitError() << "block missing terminator for SIR text";
                                failed_any = true;
                            }
                        }
                    }

                    if (failed_any)
                        signalPassFailure();
                }
            };
        } // namespace

        std::unique_ptr<Pass> createSIRTextLegalizerPass()
        {
            return std::make_unique<SIRTextLegalizerPass>();
        }

    } // namespace ora
} // namespace mlir
