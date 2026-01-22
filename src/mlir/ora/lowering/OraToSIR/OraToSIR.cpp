#include "OraToSIR.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

// Include pattern headers
#include "patterns/Arithmetic.h"
#include "patterns/Struct.h"
#include "patterns/Storage.h"
#include "patterns/MemRef.h"
#include "patterns/ControlFlow.h"
#include "patterns/EVM.h"
#include "patterns/Logs.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ora-to-sir"

#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

using namespace mlir;
using namespace ora;

static void normalizeFuncTerminators(mlir::func::FuncOp funcOp)
{
    mlir::IRRewriter rewriter(funcOp.getContext());
    for (Block &block : funcOp.getBody())
    {
        Operation *terminator = nullptr;
        for (Operation &op : block)
        {
            if (op.hasTrait<mlir::OpTrait::IsTerminator>())
            {
                terminator = &op;
                break;
            }
        }
        if (!terminator)
        {
            rewriter.setInsertionPointToEnd(&block);
            rewriter.create<sir::InvalidOp>(funcOp.getLoc());
            continue;
        }
        if (terminator->getNextNode())
        {
            llvm::errs() << "[OraToSIR] ERROR: Terminator has trailing ops in function "
                         << funcOp.getName() << " at " << terminator->getLoc() << "\n";
        }
    }
}

static void assignGlobalSlots(ModuleOp module)
{
    auto *ctx = module.getContext();
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    auto assignInBlock = [&](Block &block)
    {
        uint64_t slot = 0;
        for (Operation &op : block)
        {
            auto globalOp = dyn_cast<ora::GlobalOp>(op);
            if (!globalOp)
                continue;
            if (globalOp->getAttrOfType<IntegerAttr>("ora.slot_index"))
            {
                slot++;
                continue;
            }
            globalOp->setAttr("ora.slot_index", mlir::IntegerAttr::get(ui64Type, slot));
            slot++;
        }
    };

    module.walk([&](ora::ContractOp contractOp)
                { assignInBlock(contractOp.getBody().front()); });

    if (module.getBody()->empty())
        return;
    assignInBlock(module.getBodyRegion().front());
}

static void inlineContractsAndEraseDecls(ModuleOp module)
{
    // Inline contract bodies into the module block and erase contract wrappers.
    SmallVector<ora::ContractOp, 4> contracts;
    module.walk([&](ora::ContractOp contractOp) { contracts.push_back(contractOp); });

    if (!contracts.empty())
    {
        module->setLoc(contracts.front().getLoc());
    }

    for (auto contractOp : contracts)
    {
        Block &contractBlock = contractOp.getBody().front();
        for (auto it = contractBlock.begin(); it != contractBlock.end();)
        {
            Operation *inner = &*it++;
            if (llvm::isa<ora::YieldOp>(inner))
            {
                inner->erase();
                continue;
            }
            inner->moveBefore(contractOp);
        }
        contractOp.erase();
    }

    // Drop decl-only ops that are metadata-only at this stage.
    module.walk([&](Operation *op)
                {
        StringRef name = op->getName().getStringRef();
        if (name == "ora.enum.decl" || name == "ora.log.decl" ||
            name == "ora.import" || name == "ora.tstore.global" || name == "ora.memory.global")
        {
            op->erase();
        } });
}
class MemRefEliminationPass : public PassWrapper<MemRefEliminationPass, OperationPass<ModuleOp>>
{
public:
    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        ora::OraToSIRTypeConverter typeConverter;

        RewritePatternSet patterns(module.getContext());

        patterns.add<ConvertMemRefLoadOp>(typeConverter, module.getContext());
        patterns.add<ConvertMemRefStoreOp>(typeConverter, module.getContext());
        patterns.add<ConvertMemRefAllocOp>(typeConverter, module.getContext());
        patterns.add<ConvertMemRefDimOp>(typeConverter, module.getContext());

        ConversionTarget target(*module.getContext());
        target.addLegalDialect<mlir::BuiltinDialect>();
        target.addLegalDialect<sir::SIRDialect>();
        target.addLegalDialect<ora::OraDialect>();
        target.addLegalDialect<mlir::func::FuncDialect>();
        // Keep arith legal only in the memref elimination stage.
        // Keep arith legal only in the memref elimination stage.
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::cf::ControlFlowDialect>();
        target.addIllegalDialect<mlir::memref::MemRefDialect>();

        if (failed(applyFullConversion(module, target, std::move(patterns))))
        {
            signalPassFailure();
        }
    }
};

class SIRCleanupPass : public PassWrapper<SIRCleanupPass, OperationPass<ModuleOp>>
{
public:
    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        bool changed = true;

        while (changed)
        {
            changed = false;

            module.walk([&](mlir::memref::StoreOp storeOp)
                        {
                if (storeOp->use_empty())
                {
                    DBG("SIRCleanupPass: removing unused store");
                    storeOp->erase();
                    changed = true;
                } });

            module.walk([&](mlir::memref::AllocaOp allocaOp)
                        {
                    if (allocaOp->use_empty())
                    {
                        DBG("SIRCleanupPass: removing unused alloca");
                        allocaOp->erase();
                        changed = true;
                    } });

            module.walk([&](mlir::memref::LoadOp loadOp)
                        {
                if (loadOp->use_empty())
                {
                    DBG("SIRCleanupPass: removing unused load");
                    loadOp->erase();
                    changed = true;
                } });
        }

        DBG("SIRCleanupPass: cleanup completed");
    }
};

namespace
{
    class EraseOpByName final : public ConversionPattern
    {
    public:
        EraseOpByName(StringRef name, MLIRContext *ctx, PatternBenefit benefit = 1)
            : ConversionPattern(name, benefit, ctx) {}

        LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                                      ConversionPatternRewriter &rewriter) const override
        {
            rewriter.eraseOp(op);
            return success();
        }
    };
}
class OraToSIRPass : public PassWrapper<OraToSIRPass, OperationPass<ModuleOp>>
{
public:
    void runOnOperation() override
    {
        llvm::errs() << "[OraToSIR] ========================================\n";
        llvm::errs() << "[OraToSIR] runOnOperation() called!\n";
        llvm::errs().flush();

        ModuleOp module = getOperation();
        MLIRContext *ctx = module.getContext();
        if (ctx)
            ctx->printOpOnDiagnostic(false);

        assignGlobalSlots(module);
        inlineContractsAndEraseDecls(module);

        llvm::errs() << "[OraToSIR] Starting Ora → SIR conversion pass\n";
        llvm::errs() << "[OraToSIR] ========================================\n";
        llvm::errs().flush();

        ora::OraToSIRTypeConverter typeConverter;

        RewritePatternSet patterns(ctx);
        // Pattern toggles for crash bisecting (set to false to isolate)
        const bool enable_contract = true;
        const bool enable_func = true;
        const bool enable_arith = true;
        const bool enable_memref_alloc = true;
        const bool enable_memref_load = true;
        const bool enable_memref_store = true;
        const bool enable_storage = true;
        const bool enable_return = true;
        const bool enable_control_flow = true;

        if (enable_contract)
            patterns.add<ConvertContractOp>(typeConverter, ctx);
        if (enable_arith)
            patterns.add<ConvertArithConstantOp>(typeConverter, ctx);
        if (enable_arith)
            patterns.add<ConvertArithCmpIOp>(typeConverter, ctx);
        if (enable_arith)
        {
            patterns.add<ConvertArithAddIOp>(typeConverter, ctx);
            patterns.add<ConvertArithSubIOp>(typeConverter, ctx);
            patterns.add<ConvertArithMulIOp>(typeConverter, ctx);
            patterns.add<ConvertArithDivUIOp>(typeConverter, ctx);
            patterns.add<ConvertArithRemUIOp>(typeConverter, ctx);
            patterns.add<ConvertArithDivSIOp>(typeConverter, ctx);
            patterns.add<ConvertArithAndIOp>(typeConverter, ctx);
            patterns.add<ConvertArithOrIOp>(typeConverter, ctx);
            patterns.add<ConvertArithXOrIOp>(typeConverter, ctx);
            patterns.add<ConvertArithSelectOp>(typeConverter, ctx);
            patterns.add<ConvertArithExtUIOp>(typeConverter, ctx);
            patterns.add<ConvertArithIndexCastUIOp>(typeConverter, ctx);
            patterns.add<ConvertArithTruncIOp>(typeConverter, ctx);
            patterns.add<FoldRedundantBitcastOp>(ctx);
            patterns.add<FoldAndOneOp>(ctx);
        }
        if (enable_storage)
            patterns.add<ConvertGlobalOp>(typeConverter, ctx);
        if (enable_func)
            patterns.add<ConvertFuncOp>(typeConverter, ctx);
        if (enable_func)
            patterns.add<ConvertCallOp>(typeConverter, ctx);
        if (enable_arith)
        {
            patterns.add<ConvertAddOp>(typeConverter, ctx);
            patterns.add<ConvertSubOp>(typeConverter, ctx);
            patterns.add<ConvertMulOp>(typeConverter, ctx);
            patterns.add<ConvertDivOp>(typeConverter, ctx);
            patterns.add<ConvertRemOp>(typeConverter, ctx);
            patterns.add<ConvertCmpOp>(typeConverter, ctx);
            patterns.add<ConvertConstOp>(typeConverter, ctx);
            patterns.add<ConvertStringConstantOp>(typeConverter, ctx);
            patterns.add<ConvertBytesConstantOp>(typeConverter, ctx);
            patterns.add<ConvertHexConstantOp>(typeConverter, ctx);
            patterns.add<ConvertAddrToI160Op>(typeConverter, ctx);
            patterns.add<ConvertI160ToAddrOp>(typeConverter, ctx);
            patterns.add<ConvertOldOp>(typeConverter, ctx);
            patterns.add<ConvertInvariantOp>(typeConverter, ctx);
            patterns.add<ConvertRequiresOp>(typeConverter, ctx);
            patterns.add<ConvertEnsuresOp>(typeConverter, ctx);
            patterns.add<ConvertAssertOp>(typeConverter, ctx);
            patterns.add<ConvertAssumeOp>(typeConverter, ctx);
            patterns.add<ConvertDecreasesOp>(typeConverter, ctx);
            patterns.add<ConvertIncreasesOp>(typeConverter, ctx);
            patterns.add<ConvertHavocOp>(typeConverter, ctx);
            patterns.add<ConvertQuantifiedOp>(typeConverter, ctx);
            patterns.add<ConvertStructInstantiateOp>(typeConverter, ctx);
            patterns.add<ConvertStructInitOp>(typeConverter, ctx);
            patterns.add<ConvertStructFieldExtractOp>(typeConverter, ctx);
            patterns.add<ConvertStructFieldUpdateOp>(typeConverter, ctx);
            patterns.add<ConvertStructDeclOp>(typeConverter, ctx);
        }
        patterns.add<ConvertRefinementToBaseOp>(typeConverter, ctx);
        patterns.add<ConvertBaseToRefinementOp>(typeConverter, ctx);
        patterns.add<ConvertEvmOp>(typeConverter, ctx);
        if (enable_memref_store)
            patterns.add<ConvertMemRefStoreOp>(typeConverter, ctx, PatternBenefit(10));
        if (enable_memref_load)
            patterns.add<ConvertMemRefLoadOp>(typeConverter, ctx);
        if (enable_memref_load)
            patterns.add<ConvertMemRefDimOp>(typeConverter, ctx);
        if (enable_memref_alloc)
            patterns.add<ConvertMemRefAllocOp>(typeConverter, ctx);
        if (enable_storage)
        {
            patterns.add<ConvertSLoadOp>(typeConverter, ctx);
            patterns.add<ConvertSStoreOp>(typeConverter, ctx);
            patterns.add<ConvertTLoadOp>(typeConverter, ctx);
            patterns.add<ConvertTStoreOp>(typeConverter, ctx);
            patterns.add<ConvertMapGetOp>(typeConverter, ctx, PatternBenefit(5));
            patterns.add<ConvertMapStoreOp>(typeConverter, ctx, PatternBenefit(5));
            patterns.add<ConvertTensorExtractOp>(typeConverter, ctx);
            patterns.add<ConvertTensorDimOp>(typeConverter, ctx);
        }
        if (enable_return)
            patterns.add<ConvertReturnOp>(typeConverter, ctx);
        if (enable_control_flow)
            patterns.add<ConvertWhileOp>(typeConverter, ctx);
        if (enable_control_flow)
        {
            patterns.add<ConvertIfOp>(typeConverter, ctx);
            patterns.add<ConvertIsolatedIfOp>(typeConverter, ctx);
            patterns.add<ConvertBreakOp>(typeConverter, ctx);
            patterns.add<ConvertContinueOp>(typeConverter, ctx);
            patterns.add<ConvertSwitchExprOp>(typeConverter, ctx);
            patterns.add<ConvertSwitchOp>(typeConverter, ctx);
            patterns.add<ConvertTryCatchOp>(typeConverter, ctx);
            patterns.add<ConvertTryStmtOp>(typeConverter, ctx);
            patterns.add<ConvertCfBrOp>(typeConverter, ctx);
            patterns.add<ConvertCfCondBrOp>(typeConverter, ctx);
            patterns.add<ConvertCfAssertOp>(typeConverter, ctx);
            patterns.add<ConvertScfIfOp>(typeConverter, ctx);
            patterns.add<ConvertScfForOp>(typeConverter, ctx);
            patterns.add<ConvertErrorOkOp>(typeConverter, ctx);
            patterns.add<ConvertErrorErrOp>(typeConverter, ctx);
            patterns.add<ConvertErrorIsErrorOp>(typeConverter, ctx);
            patterns.add<ConvertErrorUnwrapOp>(typeConverter, ctx);
            patterns.add<ConvertErrorGetErrorOp>(typeConverter, ctx);
            patterns.add<ConvertRangeOp>(typeConverter, ctx);
        }
        patterns.add<ConvertErrorDeclOp>(typeConverter, ctx);
        patterns.add<ConvertLogOp>(typeConverter, ctx);
        patterns.add<EraseOpByName>("ora.enum.decl", ctx);
        patterns.add<EraseOpByName>("ora.log.decl", ctx);
        patterns.add<EraseOpByName>("ora.import", ctx);
        patterns.add<EraseOpByName>("ora.tstore.global", ctx);
        patterns.add<EraseOpByName>("ora.memory.global", ctx);

        ConversionTarget target(*ctx);
        // Mark SIR dialect as legal
        target.addLegalDialect<mlir::BuiltinDialect>();
        target.addLegalDialect<sir::SIRDialect>();
        DBG("Marked SIR dialect as legal");
        // Ora ops are illegal by default; no Ora ops should remain after conversion
        target.addIllegalDialect<ora::OraDialect>();
        target.addIllegalOp<ora::ContractOp>();
        target.addIllegalOp<ora::StructDeclOp>();
        // All sload/sstore must be legalized; no dynamic legality allowed.
        DBG("Marked Ora dialect as illegal");
        // SIR-only: cf/scf must be eliminated before final output.
        target.addIllegalDialect<mlir::cf::ControlFlowDialect>();
        DBG("Marked cf dialect as illegal");
        target.addIllegalDialect<mlir::scf::SCFDialect>();
        DBG("Marked scf dialect as illegal");
        target.addIllegalDialect<mlir::tensor::TensorDialect>();
        DBG("Marked tensor dialect as legal");
        target.addIllegalDialect<mlir::arith::ArithDialect>();

        target.addDynamicallyLegalDialect<mlir::func::FuncDialect>(
            [&](Operation *op)
            {
                if (auto funcOp = dyn_cast<mlir::func::FuncOp>(op))
                {
                    auto funcType = funcOp.getFunctionType();
                    for (Type inputType : funcType.getInputs())
                    {
                        if (!typeConverter.isLegal(inputType))
                            return false;
                    }
                    for (Type resultType : funcType.getResults())
                    {
                        if (!typeConverter.isLegal(resultType))
                            return false;
                    }
                    return true;
                }
                if (auto callOp = dyn_cast<mlir::func::CallOp>(op))
                {
                    for (Type operandType : callOp.getOperandTypes())
                    {
                        if (!typeConverter.isLegal(operandType))
                            return false;
                    }
                    for (Type resultType : callOp.getResultTypes())
                    {
                        if (!typeConverter.isLegal(resultType))
                            return false;
                    }
                    return true;
                }
                return true;
            });

        target.addIllegalOp<ora::AddOp, ora::SubOp, ora::MulOp, ora::DivOp, ora::RemOp, ora::MapGetOp, ora::MapStoreOp, ora::ReturnOp>();
        target.addIllegalOp<ora::GlobalOp>();

        // Count operations before conversion
        unsigned totalOps = 0;
        unsigned oraOps = 0;
        unsigned oraReturnOps = 0;
        module.walk([&](Operation *op)
                    {
                totalOps++;
                if (op->getDialect() && op->getDialect()->getNamespace() == "ora")
                {
                    oraOps++;
                    if (isa<ora::ContractOp>(op))
                    {
                        DBG("  Found Ora op: " << op->getName() << " in parent: " << op->getParentOp()->getName());
                    }
                }
                if (isa<ora::ReturnOp>(op))
                {
                    oraReturnOps++;
                } });
        DBG("Before conversion: " << totalOps << " total ops, " << oraOps << " Ora ops, " << oraReturnOps << " ora.return ops");

        // Debug: Walk the module to see what map operations exist
        module.walk([&](Operation *op)
                    {
            if (isa<ora::MapGetOp>(op) || isa<ora::MapStoreOp>(op))
            {
                DBG("Found map operation: " << op->getName() << " at " << op->getLoc());
                DBG("  Is illegal? " << (target.isIllegal(op) ? "YES" : "NO"));
                DBG("  Is legal? " << (target.isLegal(op) ? "YES" : "NO"));
            } });

        // Apply conversion
        if (failed(applyFullConversion(module, target, std::move(patterns))))
        {
            module.walk([&](Operation *op)
                        {
                if (op->getDialect() && op->getDialect()->getNamespace() == "ora")
                {
                    llvm::errs() << "[OraToSIR] Remaining ora op: " << op->getName()
                                 << " at " << op->getLoc()
                                 << " op=" << op
                                 << " block=" << op->getBlock();
                    if (op->getNumResults() > 0)
                    {
                        llvm::errs() << " result0=" << op->getResult(0).getType();
                    }
                    llvm::errs() << "\n";
                } });
            DBG("ERROR: Conversion failed!");
            signalPassFailure();
            return;
        }

        DBG("Conversion completed successfully!");

        // Guard: fail if any illegal dialect ops remain after conversion.
        bool illegalFound = false;
        module.walk([&](Operation *op)
                    {
            if (op->getDialect())
            {
                StringRef ns = op->getDialect()->getNamespace();
                if (ns == "ora" || ns == "cf" || ns == "scf" || ns == "tensor" || ns == "arith" || ns == "memref")
                {
                    llvm::errs() << "[OraToSIR] ERROR: Illegal op remains: " << op->getName()
                                 << " at " << op->getLoc() << "\n";
                    illegalFound = true;
                }
            } });
        if (illegalFound)
        {
            signalPassFailure();
            return;
        }

        // Guard: ensure every block in every function has a terminator.
        module.walk([&](mlir::func::FuncOp funcOp)
                    { normalizeFuncTerminators(funcOp); });

        bool missingTerminator = false;
        module.walk([&](mlir::func::FuncOp funcOp)
                    {
            for (Block &block : funcOp.getBody())
            {
                if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>())
                {
                    llvm::errs() << "[OraToSIR] ERROR: Missing terminator in function "
                                 << funcOp.getName() << " at " << funcOp.getLoc() << "\n";
                    llvm::errs() << "[OraToSIR]   Block contents:\n";
                    block.dump();
                    missingTerminator = true;
                }
            } });
        if (missingTerminator)
        {
            signalPassFailure();
            return;
        }

        DBG("Skipping post-conversion cleanup patterns (FoldRedundantBitcastOp/FoldAndOneOp)");

        // Remove gas_cost attributes from all operations (Ora MLIR specific, not SIR)
        module.walk([&](Operation *op)
                    {
                if (op->hasAttr("gas_cost"))
                {
                    op->removeAttr("gas_cost");
                } });

        // Check what Ora ops remain (should be none)
        module.walk([&](Operation *op)
                    {
                if (op->getDialect() && op->getDialect()->getNamespace() == "ora")
                {
                        DBG("  Remaining Ora op: " << op->getName() << " in parent: " << op->getParentOp()->getName());
                } });
    }
};

// -----------------------------------------------------------------------------
// Pass Registration
// -----------------------------------------------------------------------------
namespace mlir
{
    namespace ora
    {

        std::unique_ptr<Pass> createOraToSIRPass()
        {
            return std::make_unique<OraToSIRPass>();
        }

        std::unique_ptr<Pass> createMemRefEliminationPass()
        {
            return std::make_unique<MemRefEliminationPass>();
        }

        std::unique_ptr<Pass> createSIRCleanupPass()
        {
            return std::make_unique<SIRCleanupPass>();
        }

        // -----------------------------------------------------------------------------
        // SIR Optimization Pass
        // -----------------------------------------------------------------------------
        class SIROptimizationPass : public PassWrapper<SIROptimizationPass, OperationPass<ModuleOp>>
        {
        public:
            void runOnOperation() override
            {
                ModuleOp module = getOperation();
                bool changed = true;

                // Run optimizations iteratively until no more changes
                // This ensures constant folding propagates through chained operations
                while (changed)
                {
                    changed = false;

                    // Run optimizations in order
                    changed |= deduplicateConstants(module);
                    changed |= foldConstantArithmetic(module);
                }

                DBG("SIROptimizationPass: optimizations completed");
            }

        private:
            // Deduplicate constants within each function (SSA values cannot be shared across functions)
            bool deduplicateConstants(ModuleOp module)
            {
                bool changed = false;

                // Deduplicate constants per function, not across the entire module
                module.walk([&](mlir::func::FuncOp funcOp)
                            {
                    DenseMap<std::pair<uint64_t, Type>, sir::ConstOp> constantMap;

                    funcOp.walk([&](sir::ConstOp constOp)
                                {
                        // Get value from attribute
                        auto attr = constOp.getValueAttr();
                        if (!attr)
                            return;
                        uint64_t value = 0;
                        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
                        {
                            // getInt() returns APInt, getZExtValue() extracts uint64_t
                            value = intAttr.getValue().getZExtValue();
                        }
                        else
                        {
                            return;
                        }

                        Type type = constOp.getResult().getType();
                        auto key = std::make_pair(value, type);

                        auto it = constantMap.find(key);
                        if (it != constantMap.end())
                        {
                            // Replace all uses of this constant with the first one
                            constOp.getResult().replaceAllUsesWith(it->second.getResult());
                            constOp.erase();
                            changed = true;
                        }
                        else
                        {
                            constantMap[key] = constOp;
                        } }); });

                return changed;
            }

            // Fold constant arithmetic: evaluate constant expressions at compile time
            bool foldConstantArithmetic(ModuleOp module)
            {
                bool changed = false;

                module.walk([&](sir::AddOp addOp)
                            {
                    auto lhsConst = addOp.getLhs().getDefiningOp<sir::ConstOp>();
                    auto rhsConst = addOp.getRhs().getDefiningOp<sir::ConstOp>();

                    if (lhsConst && rhsConst)
                    {
                        auto lhsAttr = lhsConst.getValueAttr();
                        auto rhsAttr = rhsConst.getValueAttr();
                        if (lhsAttr && rhsAttr)
                        {
                            auto lhsInt = llvm::dyn_cast<mlir::IntegerAttr>(lhsAttr);
                            auto rhsInt = llvm::dyn_cast<mlir::IntegerAttr>(rhsAttr);
                            if (lhsInt && rhsInt)
                            {
                                uint64_t result = lhsInt.getValue().getZExtValue() + rhsInt.getValue().getZExtValue();
                                OpBuilder builder(addOp);
                                auto u256Type = sir::U256Type::get(addOp.getContext());
                                auto ui64Type = mlir::IntegerType::get(addOp.getContext(), 64, mlir::IntegerType::Unsigned);
                                auto valueAttr = mlir::IntegerAttr::get(ui64Type, result);
                                auto newConst = builder.create<sir::ConstOp>(
                                    addOp.getLoc(), u256Type, valueAttr);
                                addOp.getResult().replaceAllUsesWith(newConst.getResult());
                                addOp.erase();
                                changed = true;
                            }
                        }
                    } });

                module.walk([&](sir::MulOp mulOp)
                            {
                    auto lhsConst = mulOp.getLhs().getDefiningOp<sir::ConstOp>();
                    auto rhsConst = mulOp.getRhs().getDefiningOp<sir::ConstOp>();

                    if (lhsConst && rhsConst)
                    {
                        auto lhsAttr = lhsConst.getValueAttr();
                        auto rhsAttr = rhsConst.getValueAttr();
                        if (lhsAttr && rhsAttr)
                        {
                            auto lhsInt = llvm::dyn_cast<mlir::IntegerAttr>(lhsAttr);
                            auto rhsInt = llvm::dyn_cast<mlir::IntegerAttr>(rhsAttr);
                            if (lhsInt && rhsInt)
                            {
                                uint64_t result = lhsInt.getValue().getZExtValue() * rhsInt.getValue().getZExtValue();
                                OpBuilder builder(mulOp);
                                auto u256Type = sir::U256Type::get(mulOp.getContext());
                                auto ui64Type = mlir::IntegerType::get(mulOp.getContext(), 64, mlir::IntegerType::Unsigned);
                                auto valueAttr = mlir::IntegerAttr::get(ui64Type, result);
                                auto newConst = builder.create<sir::ConstOp>(
                                    mulOp.getLoc(), u256Type, valueAttr);
                                mulOp.getResult().replaceAllUsesWith(newConst.getResult());
                                mulOp.erase();
                                changed = true;
                            }
                        }
                    } });

                return changed;
            }

            StringRef getArgument() const override { return "sir-optimize"; }
            StringRef getDescription() const override { return "Optimize SIR operations"; }
        };

        std::unique_ptr<Pass> createSIROptimizationPass()
        {
            return std::make_unique<SIROptimizationPass>();
        }

        // Simple pass that runs canonicalization and DCE on each function in the module
        class SimpleDCEPass : public PassWrapper<SimpleDCEPass, OperationPass<ModuleOp>>
        {
        public:
            void runOnOperation() override
            {
                ModuleOp module = getOperation();

                llvm::errs() << "[SimpleDCE] Running canonicalization and DCE on module...\n";
                llvm::errs().flush();

                // Walk through all func.func operations and run passes on each
                module.walk([&](mlir::func::FuncOp funcOp)
                            {
                    llvm::errs() << "[SimpleDCE] Processing function: " << funcOp.getName() << "\n";
                    llvm::errs().flush();
                    
                    // Create a nested pass manager for this function
                    OpPassManager funcPM("func.func");
                    
                    // Run canonicalization first to fold constants
                    funcPM.addPass(mlir::createCanonicalizerPass());
                    llvm::errs() << "[SimpleDCE]   Added canonicalize pass\n";
                    llvm::errs().flush();
                    
                    // Then run DCE to remove dead code
                    funcPM.addPass(mlir::createRemoveDeadValuesPass());
                    llvm::errs() << "[SimpleDCE]   Added remove-dead-values pass\n";
                    llvm::errs().flush();
                    
                    // Run the pass manager on this function
                    if (failed(runPipeline(funcPM, funcOp)))
                    {
                        llvm::errs() << "[SimpleDCE] ERROR: Failed to run passes on function: " << funcOp.getName() << "\n";
                        llvm::errs().flush();
                        signalPassFailure();
                        return;
                    }
                    
                    llvm::errs() << "[SimpleDCE] Completed passes on function: " << funcOp.getName() << "\n";
                    llvm::errs().flush(); });

                llvm::errs() << "[SimpleDCE] All passes completed on all functions\n";
                llvm::errs().flush();
            }
        };

        std::unique_ptr<Pass> createSimpleDCEPass()
        {
            return std::make_unique<SimpleDCEPass>();
        }

        //===----------------------------------------------------------------------===//
        // Ora Inlining Pass
        //===----------------------------------------------------------------------===//

        // Pass that inlines functions marked with ora.inline attribute
        class OraInliningPass : public PassWrapper<OraInliningPass, OperationPass<ModuleOp>>
        {
        public:
            void runOnOperation() override
            {
                ModuleOp module = getOperation();
                bool changed = true;

                DBG("Running Ora inlining pass...");

                // Iterate until no more inlining opportunities
                while (changed)
                {
                    changed = false;

                    // Find all func.call operations and inline if the function has ora.inline attribute
                    module.walk([&](mlir::func::CallOp callOp)
                                {
                        // Get the function being called
                        auto callee = callOp.getCallableForCallee();
                        if (!callee)
                            return;

                        // Get the function symbol reference
                        auto symbolRef = llvm::dyn_cast<SymbolRefAttr>(callee);
                        if (!symbolRef)
                            return;

                        // Look up the function in the module
                        auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(symbolRef.getRootReference());
                        if (!funcOp)
                            return;

                        // Check if function has ora.inline attribute
                        auto inlineAttr = funcOp->getAttrOfType<BoolAttr>("ora.inline");
                        if (!inlineAttr || !inlineAttr.getValue())
                            return;

                        DBG("  Inlining function: " << funcOp.getName());

                        // Perform inlining
                        if (inlineCall(callOp, funcOp))
                        {
                            changed = true;
                            DBG("    Successfully inlined: " << funcOp.getName());
                        }
                        else
                        {
                            DBG("    Failed to inline: " << funcOp.getName());
                        } });

                    // If we made changes, run canonicalization to clean up
                    if (changed)
                    {
                        OpPassManager pm("builtin.module");
                        pm.addPass(createCanonicalizerPass());
                        if (failed(runPipeline(pm, module)))
                        {
                            signalPassFailure();
                            return;
                        }
                    }
                }

                DBG("Ora inlining pass completed");
            }

        private:
            // Inline a function call by cloning the function body
            bool inlineCall(mlir::func::CallOp callOp, mlir::func::FuncOp funcOp)
            {
                try
                {
                    // Get the function body
                    auto &funcBody = funcOp.getBody();
                    if (funcBody.empty())
                        return false;

                    // Get the entry block
                    Block *entryBlock = &funcBody.front();
                    if (entryBlock->empty())
                        return false;

                    // Create IR mapping for value substitution
                    IRMapping mapping;

                    // Map function arguments to call operands
                    for (unsigned i = 0; i < callOp.getNumOperands(); ++i)
                    {
                        if (i < entryBlock->getNumArguments())
                        {
                            mapping.map(entryBlock->getArgument(i), callOp.getOperand(i));
                        }
                    }

                    // Get the insertion point (before the call operation)
                    OpBuilder builder(callOp);

                    // Clone all operations from the function body (except the return)
                    for (auto &op : entryBlock->getOperations())
                    {
                        // Skip the return operation - we'll handle it separately
                        if (isa<mlir::func::ReturnOp>(op))
                        {
                            auto returnOp = cast<mlir::func::ReturnOp>(op);
                            // Map return values to call results
                            if (returnOp.getNumOperands() > 0)
                            {
                                // Clone the return operands and replace call results
                                SmallVector<Value> returnValues;
                                for (auto operand : returnOp.getOperands())
                                {
                                    returnValues.push_back(mapping.lookupOrDefault(operand));
                                }
                                // Replace call results with return values
                                if (returnValues.size() == callOp.getNumResults())
                                {
                                    for (unsigned i = 0; i < returnValues.size(); ++i)
                                    {
                                        callOp.getResult(i).replaceAllUsesWith(returnValues[i]);
                                    }
                                }
                            }
                            break; // Stop after processing return
                        }

                        // Clone the operation
                        builder.clone(op, mapping);
                    }

                    // Erase the call operation
                    callOp.erase();

                    return true;
                }
                catch (...)
                {
                    return false;
                }
            }
        };

        std::unique_ptr<Pass> createOraInliningPass()
        {
            return std::make_unique<OraInliningPass>();
        }

        //===----------------------------------------------------------------------===//
        // Ora Optimization Pass
        //===----------------------------------------------------------------------===//

        // Pass that performs Ora-specific optimizations (constant deduplication, constant folding fallback)
        class OraOptimizationPass : public PassWrapper<OraOptimizationPass, OperationPass<ModuleOp>>
        {
        public:
            void runOnOperation() override
            {
                ModuleOp module = getOperation();
                bool changed = true;

                DBG("Running Ora optimizations...");

                // Run optimizations iteratively until no more changes
                while (changed)
                {
                    changed = false;

                    // Run optimizations in order
                    changed |= deduplicateConstants(module);
                    changed |= foldConstantArithmetic(module);
                }

                DBG("Ora optimizations completed");
            }

        private:
            // Deduplicate constants: find all arith.constant with same value/type, replace uses
            // IMPORTANT: Only deduplicate constants in the SAME BLOCK to avoid dominance issues
            // Constants inside nested regions (ora.if, scf.if, etc.) cannot be deduplicated
            // with constants outside those regions.
            bool deduplicateConstants(ModuleOp module)
            {
                bool changed = false;

                module.walk([&](mlir::func::FuncOp funcOp)
                            {
                    // Process each block separately to avoid cross-region deduplication
                    for (Block &block : funcOp.getBlocks())
                    {
                        llvm::DenseMap<std::pair<uint64_t, Type>, sir::ConstOp> constantMap;
                        llvm::SmallVector<sir::ConstOp, 8> toErase;
                        
                        // Only process constants directly in this block, not in nested regions
                        for (Operation &op : block.getOperations())
                        {
                            auto constOp = dyn_cast<sir::ConstOp>(&op);
                            if (!constOp)
                                continue;
                                
                            // Get value from attribute
                            auto attr = constOp.getValueAttr();
                            if (!attr)
                                continue;
                            uint64_t value = 0;
                            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
                            {
                                value = intAttr.getValue().getZExtValue();
                            }
                            else
                            {
                                continue;
                            }

                            Type type = constOp.getResult().getType();
                            auto key = std::make_pair(value, type);

                            auto it = constantMap.find(key);
                            if (it != constantMap.end())
                            {
                                // Replace all uses of this constant with the first one
                                constOp.getResult().replaceAllUsesWith(it->second.getResult());
                                toErase.push_back(constOp);
                                changed = true;
                            }
                            else
                            {
                                constantMap[key] = constOp;
                            }
                        }
                        
                        // Erase duplicates after iteration is complete
                        for (auto constOp : toErase)
                        {
                            constOp.erase();
                        }
                    } });

                return changed;
            }

            // Fold constant arithmetic: evaluate constant expressions at compile time (fallback)
            bool foldConstantArithmetic(ModuleOp module)
            {
                bool changed = false;

                // Fold constant addition (fallback if canonicalization didn't catch it)
                module.walk([&](ora::AddOp addOp)
                            {
                    auto getConstantValue = [](Value val) -> std::optional<uint64_t> {
                        if (auto constOp = val.getDefiningOp<sir::ConstOp>()) {
                            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr())) {
                                return intAttr.getValue().getZExtValue();
                            }
                        }
                        return std::nullopt;
                    };

                    auto lhsVal = getConstantValue(addOp.getLhs());
                    auto rhsVal = getConstantValue(addOp.getRhs());

                    if (lhsVal && rhsVal) {
                        uint64_t result = *lhsVal + *rhsVal;
                        OpBuilder builder(addOp);
                        auto resultType = addOp.getResult().getType();
                        auto u256Type = sir::U256Type::get(addOp.getContext());
                        auto ui64Type = mlir::IntegerType::get(addOp.getContext(), 64, mlir::IntegerType::Unsigned);
                        auto valueAttr = mlir::IntegerAttr::get(ui64Type, result);
                        auto newConst = builder.create<sir::ConstOp>(addOp.getLoc(), u256Type, valueAttr);
                        Value resultVal = newConst.getResult();
                        if (resultType != u256Type)
                            resultVal = builder.create<sir::BitcastOp>(addOp.getLoc(), resultType, resultVal);
                        addOp.getResult().replaceAllUsesWith(resultVal);
                        addOp.erase();
                        changed = true;
                    } });

                // Fold constant multiplication (fallback)
                module.walk([&](ora::MulOp mulOp)
                            {
                    auto getConstantValue = [](Value val) -> std::optional<uint64_t> {
                        if (auto constOp = val.getDefiningOp<sir::ConstOp>()) {
                            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr())) {
                                return intAttr.getValue().getZExtValue();
                            }
                        }
                        return std::nullopt;
                    };

                    auto lhsVal = getConstantValue(mulOp.getLhs());
                    auto rhsVal = getConstantValue(mulOp.getRhs());

                    if (lhsVal && rhsVal) {
                        uint64_t result = *lhsVal * *rhsVal;
                        OpBuilder builder(mulOp);
                        auto resultType = mulOp.getResult().getType();
                        auto u256Type = sir::U256Type::get(mulOp.getContext());
                        auto ui64Type = mlir::IntegerType::get(mulOp.getContext(), 64, mlir::IntegerType::Unsigned);
                        auto valueAttr = mlir::IntegerAttr::get(ui64Type, result);
                        auto newConst = builder.create<sir::ConstOp>(mulOp.getLoc(), u256Type, valueAttr);
                        Value resultVal = newConst.getResult();
                        if (resultType != u256Type)
                            resultVal = builder.create<sir::BitcastOp>(mulOp.getLoc(), resultType, resultVal);
                        mulOp.getResult().replaceAllUsesWith(resultVal);
                        mulOp.erase();
                        changed = true;
                    } });

                return changed;
            }

            StringRef getArgument() const override { return "ora-optimize"; }
            StringRef getDescription() const override { return "Optimize Ora operations"; }
        };

        std::unique_ptr<Pass> createOraOptimizationPass()
        {
            return std::make_unique<OraOptimizationPass>();
        }

        //===----------------------------------------------------------------------===//
        // Ora Cleanup Pass
        //===----------------------------------------------------------------------===//

        // Pass that removes unused Ora operations
        class OraCleanupPass : public PassWrapper<OraCleanupPass, OperationPass<ModuleOp>>
        {
        public:
            void runOnOperation() override
            {
                ModuleOp module = getOperation();
                bool changed = true;

                DBG("Running Ora cleanup...");

                // Iterate until no more changes
                while (changed)
                {
                    changed = false;

                    // Remove unused storage stores (if the stored value is never used)
                    module.walk([&](ora::SStoreOp storeOp)
                                {
                        // Check if the stored value has any uses beyond this store
                        Value storedValue = storeOp.getValue();
                        if (storedValue.hasOneUse())
                        {
                            // Only used by this store, safe to remove
                            storeOp.erase();
                            changed = true;
                        } });

                    // Remove unused memory stores
                    module.walk([&](ora::MStoreOp storeOp)
                                {
                        Value storedValue = storeOp.getValue();
                        if (storedValue.hasOneUse())
                        {
                            storeOp.erase();
                            changed = true;
                        } });
                }

                DBG("Ora cleanup completed");
            }

            StringRef getArgument() const override { return "ora-cleanup"; }
            StringRef getDescription() const override { return "Clean up unused Ora operations"; }
        };

        std::unique_ptr<Pass> createOraCleanupPass()
        {
            return std::make_unique<OraCleanupPass>();
        }

        //===----------------------------------------------------------------------===//
        // Simple Ora Optimization Pass (canonicalize + DCE)
        //===----------------------------------------------------------------------===//

        // Pass that runs canonicalization and DCE on each function in Ora MLIR (before conversion)
        class SimpleOraOptimizationPass : public PassWrapper<SimpleOraOptimizationPass, OperationPass<ModuleOp>>
        {
        public:
            void runOnOperation() override
            {
                ModuleOp module = getOperation();

                DBG("Running canonicalization and DCE on Ora MLIR...");

                // Walk through all func.func operations and run passes on each
                module.walk([&](mlir::func::FuncOp funcOp)
                            {
                    DBG("Processing function: " << funcOp.getName());

                    bool hasNullOperand = false;
                    funcOp.walk([&](Operation *op)
                                {
                        for (auto operand : op->getOperands())
                        {
                            if (!operand)
                            {
                                hasNullOperand = true;
                                break;
                            }
                        } });
                    if (hasNullOperand || failed(mlir::verify(funcOp)))
                    {
                        DBG("Skipping canonicalization for function: " << funcOp.getName());
                        module->setAttr("ora.dce_invalid", mlir::UnitAttr::get(module.getContext()));
                        return;
                    }
                    
                    // Print IR BEFORE DCE to see what we start with
                    if (mlir::ora::isDebugEnabled())
                    {
                        llvm::errs() << "[SimpleOraOptimization] === IR BEFORE DCE ===\n";
                        funcOp.print(llvm::errs());
                        llvm::errs() << "\n[SimpleOraOptimization] === END IR BEFORE DCE ===\n";
                        llvm::errs().flush();
                        
                        // Track all operations before DCE
                        llvm::errs() << "[SimpleOraOptimization] Operations before DCE:\n";
                        funcOp.walk([&](Operation *op)
                                    {
                            llvm::errs() << "  - " << op->getName() << " at " << op->getLoc() << "\n";
                            if (op->getNumResults() > 0)
                            {
                                for (auto result : op->getResults())
                                {
                                    unsigned useCount = 0;
                                    for (auto &use : result.getUses())
                                    {
                                        useCount++;
                                        llvm::errs() << "      Used by: " << use.getOwner()->getName() << "\n";
                                    }
                                    llvm::errs() << "    Result: " << result << " has " << useCount << " uses\n";
                                }
                            } });
                        llvm::errs().flush();
                    }
                    
                    // Create a nested pass manager for this function
                    OpPassManager funcPM("func.func");
                    
                    // Run canonicalization first to fold constants (ora.add, ora.mul, etc.)
                    funcPM.addPass(mlir::createCanonicalizerPass());
                    DBG("  Added canonicalize pass");
                    
                    // DCE is temporarily disabled because it can invalidate
                    // func.call operands in Ora IR, producing null operands.
                    
                    // Run the pass manager on this function
                    if (failed(runPipeline(funcPM, funcOp)))
                    {
                        DBG("ERROR: Failed to run passes on function: " << funcOp.getName());
                        signalPassFailure();
                        return;
                    }
                    
                    // Print IR AFTER DCE to see what was removed
                    if (mlir::ora::isDebugEnabled())
                    {
                        llvm::errs() << "[SimpleOraOptimization] === IR AFTER DCE ===\n";
                        funcOp.print(llvm::errs());
                        llvm::errs() << "\n[SimpleOraOptimization] === END IR AFTER DCE ===\n";
                        llvm::errs().flush();
                        
                        // Check for null operands explicitly and show what func.return expects
                        llvm::errs() << "[SimpleOraOptimization] Checking for null operands...\n";
                        funcOp.walk([&](Operation *op)
                                    {
                            if (auto returnOp = dyn_cast<ora::ReturnOp>(op))
                            {
                                llvm::errs() << "  func.return at " << op->getLoc() << " has " << returnOp.getNumOperands() << " operands\n";
                                for (unsigned i = 0; i < returnOp.getNumOperands(); ++i)
                                {
                                    auto operand = returnOp.getOperand(i);
                                    if (!operand)
                                    {
                                        llvm::errs() << "    ERROR: Operand " << i << " is NULL\n";
                                    }
                                    else
                                    {
                                        llvm::errs() << "    Operand " << i << ": " << operand << " (type: " << operand.getType() << ")\n";
                                        if (auto defOp = operand.getDefiningOp())
                                        {
                                            llvm::errs() << "      Defined by: " << defOp->getName() << "\n";
                                        }
                                        else
                                        {
                                            llvm::errs() << "      WARNING: Has no defining operation!\n";
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (auto operand : op->getOperands())
                                {
                                    if (!operand)
                                    {
                                        llvm::errs() << "  ERROR: Found null operand in " << op->getName() << " at " << op->getLoc() << "\n";
                                    }
                                    else if (!operand.getDefiningOp())
                                    {
                                        llvm::errs() << "  WARNING: Operand " << operand << " has no defining op in " << op->getName() << "\n";
                                    }
                                }
                            } });
                        llvm::errs().flush();
                    }
                    
                    // Verify the function after DCE to catch any issues before printing
                    // If verification fails, the IR is invalid and printing will segfault
                    // CRITICAL: Mark the module as invalid so we can skip printing later
                    if (failed(mlir::verify(funcOp)))
                    {
                        DBG("ERROR: Function verification failed after DCE for: " << funcOp.getName());
                        DBG("  DCE left the IR in an invalid state");
                        
                        // Mark the module as invalid by setting an attribute
                        // This allows us to skip printing later
                        module->setAttr("ora.dce_invalid", mlir::UnitAttr::get(module.getContext()));
                        
                        // Don't fail the pass - we want to continue to see other errors
                        // But we'll skip printing to avoid segfault
                    }
                    
                    DBG("Completed passes on function: " << funcOp.getName()); });

                DBG("All passes completed on all functions");
            }

            StringRef getArgument() const override { return "ora-simple-optimize"; }
            StringRef getDescription() const override { return "Run canonicalization and DCE on Ora MLIR functions"; }
        };

        std::unique_ptr<Pass> createSimpleOraOptimizationPass()
        {
            return std::make_unique<SimpleOraOptimizationPass>();
        }

        // Legacy alias for backward compatibility
        std::unique_ptr<Pass> createOraCanonicalizationPass()
        {
            return createSimpleOraOptimizationPass();
        }

    } // namespace ora
} // namespace mlir
