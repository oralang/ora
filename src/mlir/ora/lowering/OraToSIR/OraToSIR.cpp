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
#include "patterns/MissingOps.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ora-to-sir"

#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

using namespace mlir;
using namespace ora;

namespace
{
    class RefinementErasureTypeConverter final : public TypeConverter
    {
    public:
        RefinementErasureTypeConverter()
        {
            addConversion([](Type type) { return type; });

            addConversion([&](ora::MinValueType type) { return convertType(type.getBaseType()); });
            addConversion([&](ora::MaxValueType type) { return convertType(type.getBaseType()); });
            addConversion([&](ora::InRangeType type) { return convertType(type.getBaseType()); });
            addConversion([&](ora::ScaledType type) { return convertType(type.getBaseType()); });
            addConversion([&](ora::ExactType type) { return convertType(type.getBaseType()); });
            addConversion([&](ora::NonZeroAddressType type) {
                return ora::AddressType::get(type.getContext());
            });

            addConversion([&](ora::ErrorUnionType type) -> Type {
                auto successType = convertType(type.getSuccessType());
                return ora::ErrorUnionType::get(type.getContext(), successType);
            });
            addConversion([&](ora::UnionType type) -> Type {
                SmallVector<Type> elems;
                elems.reserve(type.getElementTypes().size());
                for (Type elem : type.getElementTypes())
                    elems.push_back(convertType(elem));
                return ora::UnionType::get(type.getContext(), elems);
            });
            addConversion([&](ora::MapType type) -> Type {
                auto key = convertType(type.getKeyType());
                auto value = convertType(type.getValueType());
                return ora::MapType::get(type.getContext(), key, value);
            });
            addConversion([&](ora::TupleType type) -> Type {
                SmallVector<Type> elems;
                elems.reserve(type.getElementTypes().size());
                for (Type elem : type.getElementTypes())
                    elems.push_back(convertType(elem));
                return ora::TupleType::get(type.getContext(), elems);
            });

            addConversion([&](RankedTensorType type) -> Type {
                auto elem = convertType(type.getElementType());
                return RankedTensorType::get(type.getShape(), elem, type.getEncoding());
            });
            addConversion([&](UnrankedTensorType type) -> Type {
                auto elem = convertType(type.getElementType());
                return UnrankedTensorType::get(elem);
            });
            addConversion([&](MemRefType type) -> Type {
                auto elem = convertType(type.getElementType());
                return MemRefType::get(type.getShape(), elem, type.getLayout(), type.getMemorySpace());
            });
            addConversion([&](mlir::FunctionType type) -> Type {
                SmallVector<Type> inputs;
                SmallVector<Type> results;
                inputs.reserve(type.getInputs().size());
                results.reserve(type.getResults().size());
                for (Type in : type.getInputs())
                    inputs.push_back(convertType(in));
                for (Type out : type.getResults())
                    results.push_back(convertType(out));
                return mlir::FunctionType::get(type.getContext(), inputs, results);
            });
            addConversion([&](ora::FunctionType type) -> Type {
                SmallVector<Type> inputs;
                inputs.reserve(type.getParamTypes().size());
                for (Type in : type.getParamTypes())
                    inputs.push_back(convertType(in));
                Type result = convertType(type.getReturnType());
                return ora::FunctionType::get(type.getContext(), inputs, result);
            });

            addSourceMaterialization([&](OpBuilder &builder,
                                         Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value {
                if (inputs.size() != 1)
                    return Value();
                return builder
                    .create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
                    .getResult(0);
            });
            addTargetMaterialization([&](OpBuilder &builder,
                                         Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value {
                if (inputs.size() != 1)
                    return Value();
                return builder
                    .create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
                    .getResult(0);
            });
        }
    };

    class EraseRefinementToBaseOp final
        : public OpConversionPattern<ora::RefinementToBaseOp>
    {
    public:
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(ora::RefinementToBaseOp op,
                                      OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override
        {
            Value value = adaptor.getValue();
            Type targetType = typeConverter->convertType(op.getType());
            if (!targetType)
                return failure();
            if (value.getType() == targetType)
            {
                rewriter.replaceOp(op, value);
                return success();
            }
            rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, targetType, value);
            return success();
        }
    };

    class EraseBaseToRefinementOp final
        : public OpConversionPattern<ora::BaseToRefinementOp>
    {
    public:
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(ora::BaseToRefinementOp op,
                                      OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override
        {
            Value value = adaptor.getValue();
            Type targetType = typeConverter->convertType(op.getType());
            if (!targetType)
                return failure();
            if (value.getType() == targetType)
            {
                rewriter.replaceOp(op, value);
                return success();
            }
            rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, targetType, value);
            return success();
        }
    };

    class ConvertRefinementResultTypes final : public ConversionPattern
    {
    public:
        ConvertRefinementResultTypes(MLIRContext *ctx, const TypeConverter &converter)
            : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

        LogicalResult matchAndRewrite(Operation *op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            if (isa<ora::RefinementToBaseOp, ora::BaseToRefinementOp, ora::GlobalOp>(op))
                return failure();

            if (op->getNumResults() == 0)
                return failure();

            SmallVector<Type> newResultTypes;
            if (failed(typeConverter->convertTypes(op->getResultTypes(), newResultTypes)))
                return failure();

            if (llvm::equal(op->getResultTypes(), newResultTypes))
                return failure();

            auto newOp = convertOpResultTypes(op, operands, *typeConverter, rewriter);
            if (failed(newOp))
                return failure();
            rewriter.replaceOp(op, (*newOp)->getResults());
            return success();
        }
    };

    class ConvertCallTypeOp final : public OpConversionPattern<mlir::func::CallOp>
    {
    public:
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir::func::CallOp op,
                                      OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override
        {
            SmallVector<Value> newOperands;
            newOperands.reserve(op.getNumOperands());
            for (auto it : llvm::enumerate(adaptor.getOperands()))
            {
                Value operand = it.value();
                Type origType = op.getOperand(it.index()).getType();
                Type newType = typeConverter->convertType(origType);
                if (newType && newType != operand.getType())
                {
                    operand = rewriter
                                  .create<UnrealizedConversionCastOp>(op.getLoc(), newType, operand)
                                  .getResult(0);
                }
                newOperands.push_back(operand);
            }

            SmallVector<Type> newResultTypes;
            if (failed(typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
                return failure();

            if (llvm::equal(op.getResultTypes(), newResultTypes))
                return failure();

            auto newCall = rewriter.create<mlir::func::CallOp>(
                op.getLoc(),
                op.getCalleeAttr(),
                newResultTypes,
                newOperands);

            rewriter.replaceOp(op, newCall.getResults());
            return success();
        }
    };

    class ConvertGlobalTypeOp final : public OpConversionPattern<ora::GlobalOp>
    {
    public:
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(ora::GlobalOp op,
                                      OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override
        {
            (void)adaptor;
            Type oldType = op.getType();
            Type newType = typeConverter->convertType(oldType);
            if (!newType || newType == oldType)
                return success();
            op.setType(newType);
            return success();
        }
    };

    class ConvertFuncTypeAttrsOp final : public OpConversionPattern<mlir::func::FuncOp>
    {
    public:
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir::func::FuncOp op,
                                      OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override
        {
            (void)adaptor;
            bool changed = false;

            auto updateTypeAttr = [&](NamedAttrList &attrs, StringRef name) {
                if (auto typeAttr = mlir::dyn_cast<TypeAttr>(attrs.get(name)))
                {
                    Type newType = typeConverter->convertType(typeAttr.getValue());
                    if (newType && newType != typeAttr.getValue())
                    {
                        attrs.set(name, TypeAttr::get(newType));
                        changed = true;
                    }
                }
            };

            MLIRContext *ctx = op.getContext();

            rewriter.modifyOpInPlace(op, [&] {
                // Update argument attributes array.
                if (op.getNumArguments() > 0)
                {
                    SmallVector<Attribute> newArgAttrs;
                    newArgAttrs.reserve(op.getNumArguments());
                    ArrayRef<Attribute> argAttrs =
                        op.getArgAttrsAttr() ? op.getArgAttrsAttr().getValue()
                                             : ArrayRef<Attribute>();
                    for (unsigned i = 0; i < op.getNumArguments(); ++i)
                    {
                        DictionaryAttr dict = (i < argAttrs.size() && argAttrs[i])
                                                  ? mlir::dyn_cast<DictionaryAttr>(argAttrs[i])
                                                  : DictionaryAttr::get(ctx);
                        NamedAttrList attrs(dict ? dict : DictionaryAttr::get(ctx));
                        updateTypeAttr(attrs, "ora.type");
                        newArgAttrs.push_back(attrs.getDictionary(ctx));
                    }
                    if (changed)
                        op.setArgAttrsAttr(ArrayAttr::get(ctx, newArgAttrs));
                }

                // Update result attributes array.
                if (op.getNumResults() > 0)
                {
                    SmallVector<Attribute> newResAttrs;
                    newResAttrs.reserve(op.getNumResults());
                    ArrayRef<Attribute> resAttrs =
                        op.getResAttrsAttr() ? op.getResAttrsAttr().getValue()
                                             : ArrayRef<Attribute>();
                    for (unsigned i = 0; i < op.getNumResults(); ++i)
                    {
                        DictionaryAttr dict = (i < resAttrs.size() && resAttrs[i])
                                                  ? mlir::dyn_cast<DictionaryAttr>(resAttrs[i])
                                                  : DictionaryAttr::get(ctx);
                        NamedAttrList attrs(dict ? dict : DictionaryAttr::get(ctx));
                        updateTypeAttr(attrs, "ora.type");
                        newResAttrs.push_back(attrs.getDictionary(ctx));
                    }
                    if (changed)
                        op.setResAttrsAttr(ArrayAttr::get(ctx, newResAttrs));
                }
            });

            return success();
        }
    };
}

static void logModuleOps(ModuleOp module, StringRef tag)
{
    if (!mlir::ora::isDebugEnabled())
        return;
    llvm::errs() << "[OraToSIR] " << tag << " (per-op log)\n";
    module.walk([&](Operation *op) {
        llvm::errs() << "[OraToSIR]   op=" << op->getName() << " loc=" << op->getLoc() << "\n";
    });
    llvm::errs().flush();
}

static void dumpModuleOnFailure(ModuleOp module, StringRef phase)
{
    llvm::errs() << "[OraToSIR] ERROR: " << phase << " failed\n";
    llvm::errs().flush();
    // Note: module.dump() after a failed conversion can segfault if IR is
    // inconsistent, so we skip it.
}

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

static LogicalResult eraseRefinements(ModuleOp module)
{
    MLIRContext *ctx = module.getContext();
    RefinementErasureTypeConverter typeConverter;

    ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<ora::OraDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::cf::ControlFlowDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();

    target.addIllegalOp<ora::RefinementToBaseOp, ora::BaseToRefinementOp>();

    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<ora::GlobalOp>([&](ora::GlobalOp op) {
        return typeConverter.isLegal(op.getType());
    });
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        for (Type t : op->getOperandTypes())
            if (!typeConverter.isLegal(t))
                return false;
        for (Type t : op->getResultTypes())
            if (!typeConverter.isLegal(t))
                return false;
        return true;
    });

    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
    patterns.add<ConvertCallTypeOp>(typeConverter, ctx);
    patterns.add<ConvertFuncTypeAttrsOp>(typeConverter, ctx);
    patterns.add<EraseRefinementToBaseOp, EraseBaseToRefinementOp>(typeConverter, ctx);
    patterns.add<ConvertGlobalTypeOp>(typeConverter, ctx);
    patterns.add<ConvertRefinementResultTypes>(ctx, typeConverter);

    if (failed(applyFullConversion(module, target, std::move(patterns))))
        return failure();

    return success();
}

static void assignGlobalSlots(ModuleOp module)
{
    auto *ctx = module.getContext();
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    SmallVector<NamedAttribute> slotAttrs;

    auto assignInBlock = [&](Block &block)
    {
        uint64_t slot = 0;
        for (Operation &op : block)
        {
            if (auto globalOp = dyn_cast<ora::GlobalOp>(op))
            {
                if (globalOp->getAttrOfType<IntegerAttr>("ora.slot_index"))
                {
                    auto nameAttr = globalOp->getAttrOfType<StringAttr>("sym_name");
                    auto slotAttr = globalOp->getAttrOfType<IntegerAttr>("ora.slot_index");
                    if (nameAttr && slotAttr)
                    {
                        slotAttrs.push_back(NamedAttribute(nameAttr, slotAttr));
                    }
                    slot++;
                    continue;
                }
                auto slotAttr = mlir::IntegerAttr::get(ui64Type, slot);
                globalOp->setAttr("ora.slot_index", slotAttr);
                auto nameAttr = globalOp->getAttrOfType<StringAttr>("sym_name");
                if (nameAttr)
                {
                    slotAttrs.push_back(NamedAttribute(nameAttr, slotAttr));
                }
                slot++;
                continue;
            }

            // Also assign slots to non-ora.global storage-like declarations.
            auto opName = op.getName().getStringRef();
            if (opName == "ora.tstore.global" || opName == "ora.memory.global")
            {
                auto nameAttr = op.getAttrOfType<StringAttr>("sym_name");
                if (!nameAttr)
                {
                    continue;
                }
                if (op.getAttrOfType<IntegerAttr>("ora.slot_index"))
                {
                    auto slotAttr = op.getAttrOfType<IntegerAttr>("ora.slot_index");
                    if (slotAttr)
                    {
                        slotAttrs.push_back(NamedAttribute(nameAttr, slotAttr));
                    }
                    slot++;
                    continue;
                }
                auto slotAttr = mlir::IntegerAttr::get(ui64Type, slot);
                op.setAttr("ora.slot_index", slotAttr);
                slotAttrs.push_back(NamedAttribute(nameAttr, slotAttr));
                slot++;
            }
        }
    };

    module.walk([&](ora::ContractOp contractOp)
                { assignInBlock(contractOp.getBody().front()); });

    if (module.getBody()->empty())
        // Phase 1: proceed with Ora -> SIR conversion after normalization.
    assignInBlock(module.getBodyRegion().front());

    if (!slotAttrs.empty())
    {
        module->setAttr("ora.global_slots", DictionaryAttr::get(ctx, slotAttrs));
    }
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

        // Memref lowering happens in Phase 4; keep this pass free of memref patterns.

        ConversionTarget target(*module.getContext());
        target.addLegalDialect<mlir::BuiltinDialect>();
        target.addLegalOp<mlir::UnrealizedConversionCastOp>();
        target.addLegalDialect<sir::SIRDialect>();
        target.addLegalDialect<ora::OraDialect>();
        target.addLegalDialect<mlir::func::FuncDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::cf::ControlFlowDialect>();
        target.addIllegalDialect<mlir::memref::MemRefDialect>();

        if (failed(applyFullConversion(module, target, std::move(patterns))))
        {
            module.emitError("[MemRefElimination] memref lowering failed");
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

            module.walk([&](Block *block)
                        {
                llvm::DenseMap<Attribute, Value> consts;
                for (Operation &op : llvm::make_early_inc_range(*block))
                {
                    auto constOp = dyn_cast<sir::ConstOp>(&op);
                    if (!constOp)
                        continue;

                    Attribute key = constOp.getValueAttr();
                    auto it = consts.find(key);
                    if (it != consts.end())
                    {
                        constOp.replaceAllUsesWith(it->second);
                        constOp.erase();
                        changed = true;
                        continue;
                    }
                    consts.insert({key, constOp.getResult()});
                } });

            module.walk([&](Operation *op)
                        {
                if (!op->use_empty())
                    return;
                if (op->getNumRegions() != 0)
                    return;
                if (op->hasTrait<mlir::OpTrait::IsTerminator>())
                    return;

                if (auto iface = dyn_cast<MemoryEffectOpInterface>(op))
                {
                    if (!iface.hasNoEffect())
                        return;
                }
                else if (!op->hasTrait<mlir::OpTrait::ConstantLike>())
                {
                    return;
                }

                op->erase();
                changed = true;
            });
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
        ModuleOp module = getOperation();
        MLIRContext *ctx = module.getContext();
        if (ctx)
            ctx->printOpOnDiagnostic(false);

        assignGlobalSlots(module);
        inlineContractsAndEraseDecls(module);
        if (failed(eraseRefinements(module)))
        {
            module.emitError("[OraToSIR] Refinement erasure failed");
            return signalPassFailure();
        }

        // Single shared TypeConverter. Only tensor lowering enabled from the
        // start (needed for storage ops in Phase 1). MemRef and struct lowering
        // are enabled later when their respective phases run — enabling them too
        // early causes the TypeConverter to rewrite types inside regions (scf.if)
        // before the enclosing op is converted, leading to null type crashes.
        ora::OraToSIRTypeConverter typeConverter;
        typeConverter.setEnableTensorLowering(true);

        const bool enable_contract = true;
        const bool enable_func = true;
        const bool enable_arith = true;
        const bool enable_memref_alloc = false;   // Phase 4
        const bool enable_memref_load = false;    // Phase 4
        const bool enable_memref_store = false;   // Phase 4
        const bool enable_struct = false;          // Phase 4
        const bool enable_storage = true;
        const bool enable_control_flow = true;

        RewritePatternSet patterns(ctx);

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
            patterns.add<ConvertArithIndexCastOp>(typeConverter, ctx);
            patterns.add<ConvertArithTruncIOp>(typeConverter, ctx);
            patterns.add<FoldRedundantBitcastOp>(ctx);
        }
        if (enable_storage)
            patterns.add<ConvertGlobalOp>(typeConverter, ctx);
        if (enable_func)
            patterns.add<ConvertFuncOp>(typeConverter, ctx);
        if (enable_func)
            patterns.add<ConvertCallOp>(typeConverter, ctx);
        if (enable_arith)
        {
            // ora.add/sub/mul/div/rem no longer emitted; arith.* used directly.
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
        }
        if (enable_struct)
        {
            patterns.add<ConvertStructInstantiateOp>(typeConverter, ctx);
            patterns.add<ConvertStructInitOp>(typeConverter, ctx);
            patterns.add<ConvertStructFieldExtractOp>(typeConverter, ctx);
            patterns.add<ConvertStructFieldUpdateOp>(typeConverter, ctx);
            patterns.add<ConvertStructDeclOp>(typeConverter, ctx);
        }
        patterns.add<ConvertRefinementToBaseOp>(typeConverter, ctx);
        patterns.add<ConvertBaseToRefinementOp>(typeConverter, ctx);
        patterns.add<ConvertEvmOp>(typeConverter, ctx);
        // Memref lowering happens in Phase 4; do not add memref patterns here.
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
        // Defer ora.return lowering to phase 2 so scf.if results are split first.
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
            patterns.add<ConvertCfBrOp>(typeConverter, ctx);
            patterns.add<ConvertCfCondBrOp>(typeConverter, ctx);
            patterns.add<ConvertCfAssertOp>(typeConverter, ctx);
            patterns.add<ConvertScfForOp>(typeConverter, ctx);
            // Defer error union ops to phase 2.
            patterns.add<ConvertRangeOp>(typeConverter, ctx);
        }
        patterns.add<ConvertErrorDeclOp>(typeConverter, ctx);
        patterns.add<ConvertLogOp>(typeConverter, ctx);
        patterns.add<EraseOpByName>("ora.enum.decl", ctx);
        patterns.add<EraseOpByName>("ora.log.decl", ctx);
        patterns.add<EraseOpByName>("ora.import", ctx);
        patterns.add<EraseOpByName>("ora.tstore.global", ctx);
        patterns.add<EraseOpByName>("ora.memory.global", ctx);

        // Missing-op patterns (Step 1 of the Ora→SIR fix plan).
        patterns.add<ConvertRefinementGuardOp>(typeConverter, ctx);
        patterns.add<ConvertPowerOp>(typeConverter, ctx);
        patterns.add<ConvertMLoadOp>(typeConverter, ctx);
        patterns.add<ConvertMStoreOp>(typeConverter, ctx);
        patterns.add<ConvertMLoad8Op>(typeConverter, ctx);
        patterns.add<ConvertMStore8Op>(typeConverter, ctx);
        patterns.add<ConvertEnumConstantOp>(typeConverter, ctx);
        patterns.add<ConvertStructFieldStoreOp>(typeConverter, ctx);
        patterns.add<ConvertDestructureOp>(typeConverter, ctx);
        // Ops that pass through or erase.
        patterns.add<ConvertImmutableOp>(typeConverter, ctx);
        patterns.add<EraseOpByName>("ora.test", ctx);
        patterns.add<EraseOpByName>("ora.lock", ctx);
        patterns.add<EraseOpByName>("ora.unlock", ctx);
        patterns.add<EraseOpByName>("ora.move", ctx);
        patterns.add<EraseOpByName>("ora.for", ctx);

        ConversionTarget target(*ctx);
        // Mark SIR dialect as legal
        target.addLegalDialect<mlir::BuiltinDialect>();
        target.addLegalDialect<sir::SIRDialect>();
        DBG("Marked SIR dialect as legal");
        // Ora ops are illegal by default; no Ora ops should remain after conversion
        target.addIllegalDialect<ora::OraDialect>();

        if (enable_storage)
        {
            // Force storage-related tensor ops to lower when arrays/maps are enabled.
            target.addIllegalOp<mlir::tensor::ExtractOp, mlir::tensor::DimOp>();
        }
        target.addIllegalOp<ora::ContractOp>();
        target.addLegalOp<ora::ReturnOp>();
        target.addLegalOp<ora::ErrorOkOp>();
        target.addLegalOp<ora::ErrorErrOp>();
        target.addLegalOp<ora::ErrorIsErrorOp>();
        target.addLegalOp<ora::ErrorUnwrapOp>();
        target.addLegalOp<ora::ErrorGetErrorOp>();
        target.addLegalOp<ora::IfOp>();
        target.addLegalOp<ora::YieldOp>();
        target.addLegalOp<ora::BreakOp>();
        target.addLegalOp<ora::ContinueOp>();
        target.addLegalOp<ora::TryStmtOp>();
        target.addLegalOp<ora::SwitchOp>();
        target.addLegalOp<mlir::UnrealizedConversionCastOp>();
        if (!enable_struct)
        {
            target.addLegalOp<ora::StructInstantiateOp>();
            target.addLegalOp<ora::StructInitOp>();
            target.addLegalOp<ora::StructFieldExtractOp>();
            target.addLegalOp<ora::StructFieldUpdateOp>();
            target.addLegalOp<ora::StructDeclOp>();
        }
        // All sload/sstore must be legalized; no dynamic legality allowed.
        DBG("Marked Ora dialect as illegal");
        // Phase 1: keep cf/scf/tensor/arith legal; lower later.
        target.addLegalDialect<mlir::cf::ControlFlowDialect>();
        DBG("Marked cf dialect as legal");
        target.addLegalDialect<mlir::scf::SCFDialect>();
        DBG("Marked scf dialect as legal");
        target.addLegalDialect<mlir::tensor::TensorDialect>();
        DBG("Marked tensor dialect as legal");
        if (enable_memref_alloc || enable_memref_load || enable_memref_store)
        {
            target.addIllegalDialect<mlir::memref::MemRefDialect>();
        }
        else
        {
            target.addLegalDialect<mlir::memref::MemRefDialect>();
        }
        target.addLegalDialect<mlir::arith::ArithDialect>();

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

        target.addIllegalOp<ora::AddOp, ora::SubOp, ora::MulOp, ora::DivOp, ora::RemOp, ora::MapGetOp, ora::MapStoreOp>();
        target.addIllegalOp<ora::GlobalOp>();
        target.addLegalOp<mlir::UnrealizedConversionCastOp>();

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

        // Preserve error decl IDs as module attribute before any conversion
        // erases ora::ErrorDeclOp (sir::ErrorDeclOp lacks SymbolOpInterface).
        {
            SmallVector<NamedAttribute> errEntries;
            module.walk([&](ora::ErrorDeclOp decl) {
                auto id = decl->getAttrOfType<mlir::IntegerAttr>("ora.error_id");
                auto sym = decl->getAttrOfType<mlir::StringAttr>("sym_name");
                if (sym && id)
                    errEntries.push_back(NamedAttribute(sym, id));
            });
            if (!errEntries.empty())
                module->setAttr("sir.error_ids", DictionaryAttr::get(ctx, errEntries));
        }

        // Phase 0 only: normalize error_union ops into explicit packing/unpacking.
        {
            RewritePatternSet phase0Patterns(ctx);
            phase0Patterns.add<NormalizeErrorOkOp>(ctx);
            phase0Patterns.add<NormalizeErrorErrOp>(ctx);
            phase0Patterns.add<NormalizeErrorIsErrorOp>(ctx);
            phase0Patterns.add<NormalizeErrorUnwrapOp>(ctx);
            phase0Patterns.add<NormalizeErrorGetErrorOp>(ctx);
            phase0Patterns.add<NormalizeErrorUnionCastOp>(ctx);
            phase0Patterns.add<NormalizeScfYieldOp>(ctx);
            phase0Patterns.add<NormalizeOraYieldOp>(ctx);
            phase0Patterns.add<NormalizeReturnOp>(ctx);
            if (failed(applyPatternsGreedily(module, std::move(phase0Patterns))))
            {
                module.emitError("[OraToSIR] Phase 0: error-union normalization failed");
                signalPassFailure();
                return;
            }
        }

        // Debug: Walk the module to see what map operations exist
        module.walk([&](Operation *op)
                    {
            if (isa<ora::MapGetOp>(op) || isa<ora::MapStoreOp>(op))
            {
                DBG("Found map operation: " << op->getName() << " at " << op->getLoc());
                DBG("  Is illegal? " << (target.isIllegal(op) ? "YES" : "NO"));
                DBG("  Is legal? " << (target.isLegal(op) ? "YES" : "NO"));
            } });

        // Apply conversion (leave ora.return for second phase)
        logModuleOps(module, "Before Phase1 conversion");
        if (failed(applyFullConversion(module, target, std::move(patterns))))
        {
            dumpModuleOnFailure(module, "Phase1 conversion");
            module.emitError("[OraToSIR] Phase 1: main conversion failed (illegal ops remain)");
            signalPassFailure();
            return;
        }

        DBG("Conversion completed successfully!");
        logModuleOps(module, "After Phase1 conversion");

        // ---------------------------------------------------------------
        // Phase 2 (two sub-phases, consolidated from the original 4):
        //
        //  2a: Lower scf.if → CFG + error union ops + try_stmt
        //      (ora.return stays legal so returns inside scf.if regions
        //       don't crash before the enclosing scf.if is rewritten)
        //  2b: Lower ora.return → sir.return / sir.iret
        //      (now safe because scf.if regions are gone)
        // ---------------------------------------------------------------

        // Pre-pass: lower "safe" ora.return cases (ConvertReturnOpPre is guarded).
        {
            RewritePatternSet prePatterns(ctx);
            prePatterns.add<ConvertReturnOpPre>(&typeConverter, ctx);
            if (failed(applyPatternsGreedily(module, std::move(prePatterns))))
            {
                module.emitError("[OraToSIR] Pre-phase 2: safe return lowering failed");
                signalPassFailure();
                return;
            }
        }

        // Phase 2: error union ops (try_stmt, is_error, unwrap, get_error).
        // scf.if and ora.return are NOT lowered here.
        {
            RewritePatternSet phase2Patterns(ctx);
            phase2Patterns.add<ConvertTryStmtOp>(typeConverter, ctx);
            phase2Patterns.add<ConvertErrorIsErrorOp>(typeConverter, ctx);
            phase2Patterns.add<ConvertErrorUnwrapOp>(typeConverter, ctx);
            phase2Patterns.add<ConvertErrorGetErrorOp>(typeConverter, ctx);
            phase2Patterns.add<ConvertArithExtUIOp>(typeConverter, ctx);
            phase2Patterns.add<ConvertArithIndexCastUIOp>(typeConverter, ctx);
            phase2Patterns.add<ConvertArithIndexCastOp>(typeConverter, ctx);

            ConversionTarget phase2Target(*ctx);
            phase2Target.addLegalDialect<mlir::BuiltinDialect>();
            phase2Target.addLegalDialect<sir::SIRDialect>();
            phase2Target.addLegalDialect<mlir::func::FuncDialect>();
            phase2Target.addLegalDialect<mlir::cf::ControlFlowDialect>();
            phase2Target.addLegalDialect<mlir::tensor::TensorDialect>();
            phase2Target.addLegalDialect<mlir::memref::MemRefDialect>();
            phase2Target.addLegalOp<mlir::UnrealizedConversionCastOp>();
            phase2Target.addLegalDialect<mlir::scf::SCFDialect>();
            phase2Target.addLegalDialect<mlir::arith::ArithDialect>();
            phase2Target.addLegalOp<ora::ReturnOp>();
            phase2Target.addIllegalOp<ora::ErrorIsErrorOp>();
            phase2Target.addIllegalOp<ora::ErrorUnwrapOp>();
            phase2Target.addIllegalOp<ora::ErrorGetErrorOp>();
            phase2Target.addLegalOp<ora::ErrorOkOp>();
            phase2Target.addLegalOp<ora::ErrorErrOp>();
            phase2Target.addLegalOp<ora::IfOp>();
            phase2Target.addLegalOp<ora::YieldOp>();
            phase2Target.addLegalOp<ora::ContinueOp>();
            phase2Target.addIllegalOp<ora::TryStmtOp>();
            phase2Target.addLegalOp<ora::SwitchOp>();
            phase2Target.addLegalDialect<ora::OraDialect>();

            logModuleOps(module, "Before Phase2 conversion");
            if (failed(applyFullConversion(module, phase2Target, std::move(phase2Patterns))))
            {
                dumpModuleOnFailure(module, "Phase2 conversion");
                module.emitError("[OraToSIR] Phase 2: error-union lowering failed");
                signalPassFailure();
                return;
            }
            logModuleOps(module, "After Phase2 conversion");
        }

        // Phase 2b: re-run try_stmt/error lowering + scf.if → CFG + ora.if + returns.
        {
            RewritePatternSet phase2bPatterns(ctx);
            phase2bPatterns.add<ConvertTryStmtOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertErrorOkOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertErrorErrOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertErrorIsErrorOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertErrorUnwrapOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertErrorGetErrorOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertArithExtUIOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertArithIndexCastUIOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertArithIndexCastOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertScfIfOp>(typeConverter, ctx,
                /*lowerReturnsInMergeBlock=*/false, PatternBenefit(10));
            phase2bPatterns.add<ConvertIfOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertIsolatedIfOp>(typeConverter, ctx);

            ConversionTarget phase2bTarget(*ctx);
            phase2bTarget.addLegalDialect<mlir::BuiltinDialect>();
            phase2bTarget.addLegalDialect<sir::SIRDialect>();
            phase2bTarget.addLegalDialect<mlir::func::FuncDialect>();
            phase2bTarget.addLegalDialect<mlir::cf::ControlFlowDialect>();
            phase2bTarget.addLegalDialect<mlir::tensor::TensorDialect>();
            phase2bTarget.addLegalDialect<mlir::memref::MemRefDialect>();
            phase2bTarget.addLegalDialect<mlir::scf::SCFDialect>();
            phase2bTarget.addLegalDialect<mlir::arith::ArithDialect>();
            phase2bTarget.addLegalOp<mlir::UnrealizedConversionCastOp>();
            phase2bTarget.addIllegalOp<mlir::scf::IfOp>();
            phase2bTarget.addIllegalOp<ora::TryStmtOp>();
            phase2bTarget.addIllegalOp<ora::ErrorOkOp>();
            phase2bTarget.addIllegalOp<ora::ErrorErrOp>();
            phase2bTarget.addIllegalOp<ora::ErrorIsErrorOp>();
            phase2bTarget.addIllegalOp<ora::ErrorUnwrapOp>();
            phase2bTarget.addIllegalOp<ora::ErrorGetErrorOp>();
            phase2bTarget.addIllegalOp<ora::IfOp>();
            // ora.return stays legal — lowered in Phase 3a/3b.
            phase2bTarget.addLegalOp<ora::ReturnOp>();
            phase2bTarget.addLegalOp<ora::YieldOp>();
            phase2bTarget.addLegalOp<ora::ContinueOp>();
            phase2bTarget.addLegalOp<ora::SwitchOp>();
            phase2bTarget.addLegalDialect<ora::OraDialect>();

            logModuleOps(module, "Before Phase2b conversion");
            if (failed(applyFullConversion(module, phase2bTarget, std::move(phase2bPatterns))))
            {
                dumpModuleOnFailure(module, "Phase2b conversion");
                module.emitError("[OraToSIR] Phase 2b: scf.if/error-union/return lowering failed");
                signalPassFailure();
                return;
            }
            logModuleOps(module, "After Phase2b conversion");

            // Drop any dead blocks introduced by try_stmt inlining.
            {
                mlir::IRRewriter cleanupRewriter(ctx);
                (void)mlir::eraseUnreachableBlocks(cleanupRewriter, module.getOperation()->getRegions());
            }
        }

        // Phase 3: lower all remaining ora.return via greedy rewrite.
        // We cannot use the conversion framework here because the TypeConverter
        // splits error_union into 2x u256, and the framework cannot adapt the
        // ora.return operands through unrealized_conversion_casts. Instead, the
        // greedy ConvertReturnOpPre pattern reads operands directly.
        {
            RewritePatternSet phase3Patterns(ctx);
            phase3Patterns.add<ConvertReturnOpPre>(&typeConverter, ctx);
            logModuleOps(module, "Before Phase3 (greedy return) conversion");
            if (failed(applyPatternsGreedily(module, std::move(phase3Patterns))))
            {
                module.emitError("[OraToSIR] Phase 3: greedy return lowering failed");
                signalPassFailure();
                return;
            }

            // Check for any remaining ora.return ops — these need the conversion
            // framework with a full ConversionTarget.
            bool hasRemainingReturns = false;
            module.walk([&](ora::ReturnOp) { hasRemainingReturns = true; });
            if (hasRemainingReturns)
            {
                RewritePatternSet phase3bPatterns(ctx);
                phase3bPatterns.add<ConvertReturnOp>(typeConverter, ctx);
                phase3bPatterns.add<ConvertScfIfOp>(typeConverter, ctx,
                    /*lowerReturnsInMergeBlock=*/false, PatternBenefit(10));

                ConversionTarget phase3bTarget(*ctx);
                phase3bTarget.addLegalDialect<mlir::BuiltinDialect>();
                phase3bTarget.addLegalDialect<sir::SIRDialect>();
                phase3bTarget.addLegalDialect<mlir::func::FuncDialect>();
                phase3bTarget.addLegalDialect<mlir::cf::ControlFlowDialect>();
                phase3bTarget.addLegalDialect<mlir::tensor::TensorDialect>();
                phase3bTarget.addLegalDialect<mlir::memref::MemRefDialect>();
                phase3bTarget.addLegalDialect<mlir::arith::ArithDialect>();
                phase3bTarget.addLegalDialect<mlir::scf::SCFDialect>();
                phase3bTarget.addLegalOp<mlir::UnrealizedConversionCastOp>();
                phase3bTarget.addIllegalOp<mlir::scf::IfOp>();
                phase3bTarget.addIllegalOp<ora::ReturnOp>();
                phase3bTarget.addLegalOp<ora::IfOp>();
                phase3bTarget.addLegalOp<ora::YieldOp>();
                phase3bTarget.addLegalOp<ora::ContinueOp>();
                phase3bTarget.addLegalOp<ora::TryStmtOp>();
                phase3bTarget.addLegalOp<ora::SwitchOp>();
                phase3bTarget.addLegalDialect<ora::OraDialect>();

                logModuleOps(module, "Before Phase3b conversion");
                if (failed(applyFullConversion(module, phase3bTarget, std::move(phase3bPatterns))))
                {
                    dumpModuleOnFailure(module, "Phase3b conversion");
                    module.emitError("[OraToSIR] Phase 3b: final return lowering failed");
                    signalPassFailure();
                    return;
                }
            }
            logModuleOps(module, "After Phase3 conversion");
        }

        // Phase 4: lower scf.for, scf.while, memref ops (stack temps) to SIR.
        typeConverter.setEnableMemRefLowering(true);
        {
            RewritePatternSet phase3Patterns(ctx);
            phase3Patterns.add<ConvertScfForOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertScfWhileOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertIfOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertIsolatedIfOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertMemRefAllocOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertMemRefLoadOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertMemRefStoreOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertMemRefDimOp>(typeConverter, ctx);

            ConversionTarget phase3Target(*ctx);
            phase3Target.addLegalDialect<mlir::BuiltinDialect>();
            phase3Target.addLegalDialect<sir::SIRDialect>();
            phase3Target.addLegalDialect<mlir::func::FuncDialect>();
            phase3Target.addLegalDialect<mlir::cf::ControlFlowDialect>();
            phase3Target.addLegalDialect<mlir::arith::ArithDialect>();
            phase3Target.addLegalDialect<mlir::scf::SCFDialect>();
            phase3Target.addLegalDialect<mlir::tensor::TensorDialect>();
            phase3Target.addLegalOp<mlir::UnrealizedConversionCastOp>();
            phase3Target.addIllegalDialect<mlir::memref::MemRefDialect>();
            phase3Target.addIllegalOp<mlir::scf::ForOp>();
            phase3Target.addIllegalOp<mlir::scf::WhileOp>();
            phase3Target.addIllegalOp<ora::ReturnOp>();
            phase3Target.addIllegalOp<ora::IfOp>();
            phase3Target.addLegalOp<ora::YieldOp>();
            phase3Target.addLegalOp<ora::ContinueOp>();
            phase3Target.addLegalOp<ora::TryStmtOp>();
            phase3Target.addLegalOp<ora::SwitchOp>();
            phase3Target.addLegalDialect<ora::OraDialect>();

            ConversionConfig phase3Config;
            // Avoid ptr<->memref materializations; memref ops must be fully rewritten.
            phase3Config.buildMaterializations = false;
            logModuleOps(module, "Before Phase3 conversion");
            if (failed(applyFullConversion(module, phase3Target, std::move(phase3Patterns), phase3Config)))
            {
                dumpModuleOnFailure(module, "Phase3 conversion");
                module.emitError("[OraToSIR] Phase 4: scf.for/memref lowering failed");
                signalPassFailure();
                return;
            }
            logModuleOps(module, "After Phase3 conversion");
        }

        // Phase 5: lower remaining Ora control flow + structs + cleanup.
        // Enable struct lowering now that scf.if regions are gone.
        typeConverter.setEnableStructLowering(true);
        {
            ORA_DEBUG_PREFIX("OraToSIR", "Phase4 start");
            RewritePatternSet phase4Patterns(ctx);
            phase4Patterns.add<ConvertFuncOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertCallOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertReturnOpPre>(&typeConverter, ctx);
            phase4Patterns.add<ConvertReturnOp>(typeConverter, ctx, PatternBenefit(1));
            phase4Patterns.add<ConvertIfOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertContinueOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertSwitchOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertTryStmtOp>(typeConverter, ctx);
            phase4Patterns.add<NormalizeOraYieldOp>(ctx);
            phase4Patterns.add<NormalizeErrorUnionCastOp>(ctx);
            phase4Patterns.add<ConvertUnrealizedConversionCastOp>(typeConverter, ctx);
            phase4Patterns.add<StripNormalizedErrorUnionCastOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertCfBrOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertCfCondBrOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertCfAssertOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithConstantOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithCmpIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithAddIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithSubIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithMulIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithDivUIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithRemUIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithDivSIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithAndIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithOrIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithXOrIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithShlIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithShrUIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithSelectOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithExtUIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithIndexCastUIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithIndexCastOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertArithTruncIOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertTensorExtractOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertTensorDimOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertBaseToRefinementOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertRefinementToBaseOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructInitOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructInstantiateOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructFieldExtractOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructFieldUpdateOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructDeclOp>(typeConverter, ctx);
            phase4Patterns.add<StripStructMaterializeOp>(typeConverter, ctx);
            phase4Patterns.add<StripAddressMaterializeOp>(typeConverter, ctx);
            phase4Patterns.add<StripBytesMaterializeOp>(typeConverter, ctx);

            ConversionTarget phase4Target(*ctx);
            phase4Target.addLegalDialect<mlir::BuiltinDialect>();
            phase4Target.addLegalDialect<sir::SIRDialect>();
            phase4Target.addLegalDialect<mlir::func::FuncDialect>();
            phase4Target.addIllegalDialect<mlir::cf::ControlFlowDialect>();
            phase4Target.addIllegalDialect<mlir::arith::ArithDialect>();
            phase4Target.addLegalDialect<mlir::scf::SCFDialect>();
            phase4Target.addIllegalDialect<mlir::tensor::TensorDialect>();
            phase4Target.addIllegalDialect<mlir::memref::MemRefDialect>();
            phase4Target.addIllegalOp<mlir::UnrealizedConversionCastOp>();
            phase4Target.addIllegalOp<mlir::func::CallOp>();
            phase4Target.addIllegalOp<ora::ReturnOp>();
            phase4Target.addLegalOp<mlir::func::FuncOp>();
            phase4Target.addIllegalOp<ora::IfOp>();
            phase4Target.addIllegalOp<ora::YieldOp>();
            phase4Target.addIllegalOp<ora::ContinueOp>();
            phase4Target.addIllegalOp<ora::TryStmtOp>();
            phase4Target.addIllegalOp<ora::SwitchOp>();
            phase4Target.addIllegalOp<ora::StructInitOp>();
            phase4Target.addIllegalOp<ora::StructInstantiateOp>();
            phase4Target.addIllegalOp<ora::StructFieldExtractOp>();
            phase4Target.addIllegalOp<ora::StructFieldUpdateOp>();
            phase4Target.addIllegalOp<ora::StructDeclOp>();
            phase4Target.addIllegalOp<ora::BaseToRefinementOp>();
            phase4Target.addIllegalOp<ora::RefinementToBaseOp>();
            phase4Target.addLegalDialect<ora::OraDialect>();

            // Debug: report any unrealized casts still present before Phase 4.
            if (mlir::ora::isDebugEnabled())
            {
                for (auto castOp : module.getOps<mlir::UnrealizedConversionCastOp>())
                {
                    llvm::errs() << "[OraToSIR] Phase4 pre-scan: unrealized cast at "
                                 << castOp.getLoc() << " operands=" << castOp.getNumOperands()
                                 << " results=" << castOp.getNumResults() << "\n";
                }
                llvm::errs().flush();
            }

            // (error IDs already preserved as module attr before Phase 1)

            logModuleOps(module, "Before Phase4 conversion");
            if (failed(applyFullConversion(module, phase4Target, std::move(phase4Patterns))))
            {
                dumpModuleOnFailure(module, "Phase4 conversion");
                module.emitError("[OraToSIR] Phase 5: final control-flow/struct lowering failed");
                signalPassFailure();
                return;
            }
            logModuleOps(module, "After Phase4 conversion");

            // Cleanup: strip any remaining normalized error_union casts.
            SmallVector<mlir::UnrealizedConversionCastOp, 8> normalizedCasts;
            module.walk([&](mlir::UnrealizedConversionCastOp op) {
                if (!op->hasAttr("ora.normalized_error_union"))
                    return;
                if (op.getNumOperands() != 1 || op.getNumResults() != 1)
                    return;
                if (!llvm::isa<sir::U256Type>(op.getOperand(0).getType()))
                    return;
                normalizedCasts.push_back(op);
            });
            for (auto castOp : normalizedCasts)
            {
                castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
                castOp.erase();
            }

            // Cleanup: strip residual refinement ops (materializations may create them).
            SmallVector<ora::BaseToRefinementOp, 8> residualB2R;
            SmallVector<ora::RefinementToBaseOp, 8> residualR2B;
            module.walk([&](ora::BaseToRefinementOp op) { residualB2R.push_back(op); });
            module.walk([&](ora::RefinementToBaseOp op) { residualR2B.push_back(op); });
            for (auto op : residualB2R)
            {
                op.getResult().replaceAllUsesWith(op.getValue());
                op.erase();
            }
            for (auto op : residualR2B)
            {
                op.getResult().replaceAllUsesWith(op.getValue());
                op.erase();
            }

            // Cleanup: strip sir.bitcast ops whose result is an Ora refinement type.
            // These are created by source materializations during conversion.
            auto isOraRefinement = [](Type t) {
                return llvm::isa<ora::MinValueType, ora::MaxValueType, ora::InRangeType,
                                 ora::ScaledType, ora::ExactType, ora::NonZeroAddressType>(t);
            };
            SmallVector<sir::BitcastOp, 16> refinementBitcasts;
            module.walk([&](sir::BitcastOp op) {
                if (isOraRefinement(op.getResult().getType()))
                    refinementBitcasts.push_back(op);
            });
            for (auto op : refinementBitcasts)
            {
                op.getResult().replaceAllUsesWith(op.getOperand());
                op.erase();
            }

            // Cleanup: strip all remaining 1:1 unrealized_conversion_casts.
            SmallVector<mlir::UnrealizedConversionCastOp, 16> residualCasts;
            module.walk([&](mlir::UnrealizedConversionCastOp op) {
                if (op.getNumOperands() == 1 && op.getNumResults() == 1)
                    residualCasts.push_back(op);
            });
            for (auto castOp : residualCasts)
            {
                castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
                castOp.erase();
            }

            // Cleanup: replace sir.icall to error constructors with sir.const.
            // Error decl IDs preserved as module attribute before Phase 1.
            {
                llvm::StringMap<int64_t> errorIds;
                if (auto errDict = module->getAttrOfType<DictionaryAttr>("sir.error_ids"))
                {
                    for (auto entry : errDict)
                    {
                        if (auto id = dyn_cast<IntegerAttr>(entry.getValue()))
                            errorIds[entry.getName()] = id.getInt();
                    }
                }

                SmallVector<sir::ICallOp, 4> errorIcalls;
                module.walk([&](sir::ICallOp op) {
                    if (auto callee = op.getCalleeAttr())
                    {
                        StringRef name = callee.getValue();
                        if (errorIds.count(name))
                            errorIcalls.push_back(op);
                    }
                });

                for (auto op : errorIcalls)
                {
                    int64_t id = errorIds[op.getCalleeAttr().getValue()];
                    OpBuilder b(op);
                    auto u256 = sir::U256Type::get(ctx);
                    auto ui256 = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
                    auto idConst = b.create<sir::ConstOp>(
                        op.getLoc(), u256, mlir::IntegerAttr::get(ui256, id));
                    // Replace all results with the error ID constant or zero.
                    for (unsigned i = 0; i < op.getNumResults(); ++i)
                    {
                        Value oldRes = op.getResult(i);
                        if (oldRes.use_empty())
                            continue;
                        Value replacement = idConst;
                        if (oldRes.getType() != u256)
                            replacement = b.create<sir::BitcastOp>(op.getLoc(), oldRes.getType(), idConst);
                        oldRes.replaceAllUsesWith(replacement);
                    }
                    op.erase();
                }
            }

            // Verify no unrealized casts remain.
            bool leftoverUnrealized = false;
            if (mlir::ora::isDebugEnabled())
            {
                for (auto castOp : module.getOps<mlir::UnrealizedConversionCastOp>())
                {
                    leftoverUnrealized = true;
                    llvm::errs() << "[OraToSIR] Phase4 post-scan: unrealized cast at "
                                 << castOp.getLoc() << " operands=" << castOp.getNumOperands()
                                 << " results=" << castOp.getNumResults() << "\n";
                }
                llvm::errs().flush();
            }
            else
            {
                for (auto castOp : module.getOps<mlir::UnrealizedConversionCastOp>())
                {
                    (void)castOp;
                    leftoverUnrealized = true;
                    break;
                }
            }
            if (leftoverUnrealized)
            {
                module.emitError("[OraToSIR] Phase4 post-scan: unrealized casts remain");
                signalPassFailure();
                return;
            }

            // Debug: dump final module.
            if (mlir::ora::isDebugEnabled())
            {
                llvm::errs() << "\n//===----------------------------------------------------------------------===//\n";
                llvm::errs() << "// SIR MLIR (after Phase4)\n";
                llvm::errs() << "//===----------------------------------------------------------------------===//\n\n";
                module.print(llvm::errs());
                llvm::errs() << "\n";
                llvm::errs().flush();
            }

            // Extra guard: detect any remaining unrealized casts by name.
            int64_t unrealizedByName = 0;
            module.walk([&](Operation *op) {
                if (op->getName().getStringRef() == "builtin.unrealized_conversion_cast")
                {
                    ++unrealizedByName;
                    if (mlir::ora::isDebugEnabled())
                    {
                        llvm::errs() << "[OraToSIR] Phase4 name-scan: unrealized cast at "
                                     << op->getLoc() << " operands=" << op->getNumOperands()
                                     << " results=" << op->getNumResults();
                        if (op->getNumOperands() > 0)
                            llvm::errs() << " in=" << op->getOperand(0).getType();
                        if (op->getNumResults() > 0)
                            llvm::errs() << " out=" << op->getResult(0).getType();
                        llvm::errs() << "\n";
                    }
                }
            });
            if (mlir::ora::isDebugEnabled())
            {
                llvm::errs().flush();
            }
            if (unrealizedByName > 0)
            {
                module.emitError("[OraToSIR] Phase4 name-scan: unrealized casts remain");
                signalPassFailure();
                return;
            }

        }

        // Guard: fail if any ops remain that should have been lowered by this stage.
        bool illegalFound = false;
        module.walk([&](Operation *op)
                    {
            if (op->getDialect())
            {
                StringRef ns = op->getDialect()->getNamespace();
                // Control-flow and high-level ops are allowed to remain for later phases.
                if (ns == "cf" || ns == "scf" || ns == "tensor" || ns == "arith" || ns == "memref")
                    return;

                if (ns == "ora")
                {
                    // These should be gone by now.
                    if (isa<ora::ReturnOp,
                            ora::ErrorOkOp,
                            ora::ErrorErrOp,
                            ora::ErrorIsErrorOp,
                            ora::ErrorUnwrapOp,
                            ora::ErrorGetErrorOp>(op))
                    {
                        llvm::errs() << "[OraToSIR] ERROR: Illegal op remains: " << op->getName()
                                     << " at " << op->getLoc() << "\n";
                        illegalFound = true;
                    }
                }
            } });
        if (illegalFound)
        {
            module.emitError("[OraToSIR] post-conversion: illegal Ora ops remain after all phases");
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
            module.emitError("[OraToSIR] post-conversion: blocks missing terminators");
            signalPassFailure();
            return;
        }

        {
            RewritePatternSet cleanupPatterns(ctx);
            cleanupPatterns.add<FoldRedundantBitcastOp>(ctx);
            cleanupPatterns.add<FoldEqSameOp>(ctx);
            cleanupPatterns.add<FoldEqConstOp>(ctx);
            cleanupPatterns.add<FoldIsZeroConstOp>(ctx);
            cleanupPatterns.add<FoldCondBrSameDestOp>(ctx);
            cleanupPatterns.add<NormalizeCondBrOperandsOp>(ctx);
            cleanupPatterns.add<FoldCondBrDoubleIsZeroOp>(ctx);
            cleanupPatterns.add<FoldCondBrConstOp>(ctx);
            cleanupPatterns.add<FoldBrToBrOp>(ctx);
            (void)applyPatternsGreedily(module, std::move(cleanupPatterns));
        }

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

                module.walk([&](mlir::func::FuncOp funcOp)
                            {
                    OpPassManager funcPM("func.func");
                    funcPM.addPass(mlir::createCanonicalizerPass());
                    funcPM.addPass(mlir::createRemoveDeadValuesPass());

                    if (failed(runPipeline(funcPM, funcOp)))
                    {
                        funcOp.emitError("[SimpleDCE] canonicalize+DCE failed");
                        signalPassFailure();
                        return;
                    } });
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
                            module.emitError("[OraInlining] post-inline canonicalization failed");
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
                (void)getOperation();
                bool changed = true;

                DBG("Running Ora cleanup...");

                // Iterate until no more changes
                while (changed)
                {
                    changed = false;

                    // Do not remove stores: they are side-effecting even if their
                    // input value has no other uses.
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
