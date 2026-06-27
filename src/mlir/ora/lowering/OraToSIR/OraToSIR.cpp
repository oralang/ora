#include "OraToSIR.h"
#include "OraToSIRTypeConverter.h"
#include "OraMaterializationKinds.h"
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
#include "patterns/ErrorUnionCarrierHelpers.h"
#include "patterns/LoweringHelpers.h"
#include "patterns/StorageLayout.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

#define DEBUG_TYPE "ora-to-sir"

#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

using namespace mlir;
using namespace ora;
namespace euh = mlir::ora::error_union_helpers;
using mlir::ora::lowering::constU256;

namespace mlir
{
    namespace ora
    {
        namespace
        {
            thread_local OraMlirPassStatistics *activeOraMlirPassStatistics = nullptr;

            static void printOraMlirPassStatisticLine(llvm::raw_ostream &os, llvm::StringRef name, uint64_t value)
            {
                os << "  " << name << " = " << value << "\n";
            }
        }

        void setActiveOraMlirPassStatistics(OraMlirPassStatistics *statistics)
        {
            activeOraMlirPassStatistics = statistics;
        }

        void recordOraMlirPassStatistic(OraMlirPassStatistic statistic, uint64_t amount)
        {
            if (!activeOraMlirPassStatistics || amount == 0)
                return;

            switch (statistic)
            {
            case OraMlirPassStatistic::OraFunctionsCanonicalized:
                activeOraMlirPassStatistics->oraFunctionsCanonicalized += amount;
                return;
            case OraMlirPassStatistic::OraFunctionsCSEProcessed:
                activeOraMlirPassStatistics->oraFunctionsCSEProcessed += amount;
                return;
            case OraMlirPassStatistic::OraStorageReadsReused:
                activeOraMlirPassStatistics->oraStorageReadsReused += amount;
                return;
            case OraMlirPassStatistic::OraCallsInlined:
                activeOraMlirPassStatistics->oraCallsInlined += amount;
                return;
            case OraMlirPassStatistic::OraSourceInlineFailures:
                activeOraMlirPassStatistics->oraSourceInlineFailures += amount;
                return;
            case OraMlirPassStatistic::SirConstantsDeduplicated:
                activeOraMlirPassStatistics->sirConstantsDeduplicated += amount;
                return;
            case OraMlirPassStatistic::SirUnusedAllocasRemoved:
                activeOraMlirPassStatistics->sirUnusedAllocasRemoved += amount;
                return;
            case OraMlirPassStatistic::SirUnusedLoadsRemoved:
                activeOraMlirPassStatistics->sirUnusedLoadsRemoved += amount;
                return;
            case OraMlirPassStatistic::SirUnusedPureOpsRemoved:
                activeOraMlirPassStatistics->sirUnusedPureOpsRemoved += amount;
                return;
            case OraMlirPassStatistic::SirFrameworkFunctionsProcessed:
                activeOraMlirPassStatistics->sirFrameworkFunctionsProcessed += amount;
                return;
            case OraMlirPassStatistic::OraSymbolsDCEd:
                activeOraMlirPassStatistics->oraSymbolsDCEd += amount;
                return;
            }
        }

        void printOraMlirPassStatistics(const OraMlirPassStatistics &statistics, llvm::raw_ostream &os, const char *pipelineName)
        {
            os << "===-------------------------------------------------------------------------===\n";
            os << "                  ... Ora MLIR pass statistics: " << pipelineName << " ...\n";
            os << "===-------------------------------------------------------------------------===\n";
            printOraMlirPassStatisticLine(os, "ora-function-canonicalize.functions-processed", statistics.oraFunctionsCanonicalized);
            printOraMlirPassStatisticLine(os, "ora-function-cse.functions-processed", statistics.oraFunctionsCSEProcessed);
            printOraMlirPassStatisticLine(os, "ora-storage-read-cse.storage-reads-reused", statistics.oraStorageReadsReused);
            printOraMlirPassStatisticLine(os, "ora-inline.calls-inlined", statistics.oraCallsInlined);
            printOraMlirPassStatisticLine(os, "ora-inline.source-inline-failures", statistics.oraSourceInlineFailures);
            printOraMlirPassStatisticLine(os, "sir-optimize.constants-deduplicated", statistics.sirConstantsDeduplicated);
            printOraMlirPassStatisticLine(os, "sir-cleanup.unused-allocas-removed", statistics.sirUnusedAllocasRemoved);
            printOraMlirPassStatisticLine(os, "sir-cleanup.unused-loads-removed", statistics.sirUnusedLoadsRemoved);
            printOraMlirPassStatisticLine(os, "sir-cleanup.unused-pure-ops-removed", statistics.sirUnusedPureOpsRemoved);
            printOraMlirPassStatisticLine(os, "sir-framework-canonicalize.functions-processed", statistics.sirFrameworkFunctionsProcessed);
            printOraMlirPassStatisticLine(os, "ora-symbol-dce.symbols-removed", statistics.oraSymbolsDCEd);
            os << "\n";
        }
    } // namespace ora
} // namespace mlir

namespace
{
    constexpr llvm::StringLiteral kPhase0SkipManualBitcastFoldAttr =
        "ora.phase0.skip_manual_bitcast_fold";

    template <typename... Dialects>
    static void addLegalDialects(ConversionTarget &target)
    {
        (target.addLegalDialect<Dialects>(), ...);
    }

    static void addOraToSirBaseLegalDialects(ConversionTarget &target)
    {
        addLegalDialects<mlir::BuiltinDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect,
                         mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect,
                         mlir::tensor::TensorDialect, mlir::memref::MemRefDialect>(target);
    }

    static void addOraToSirBaseLegalDialectsWithSir(ConversionTarget &target)
    {
        addLegalDialects<mlir::BuiltinDialect, sir::SIRDialect, mlir::func::FuncDialect,
                         mlir::cf::ControlFlowDialect, mlir::tensor::TensorDialect,
                         mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                         mlir::arith::ArithDialect>(target);
    }

    static void addArithExtensionCastPatterns(
        RewritePatternSet &patterns,
        OraToSIRTypeConverter &typeConverter,
        MLIRContext *ctx)
    {
        patterns.add<ConvertArithExtUIOp>(typeConverter, ctx);
        patterns.add<ConvertArithExtSIOp>(typeConverter, ctx);
        patterns.add<ConvertArithIndexCastUIOp>(typeConverter, ctx);
        patterns.add<ConvertArithIndexCastOp>(typeConverter, ctx);
    }

    static void addArithmeticLoweringPatterns(
        RewritePatternSet &patterns,
        OraToSIRTypeConverter &typeConverter,
        MLIRContext *ctx)
    {
        patterns.add<ConvertArithConstantOp>(typeConverter, ctx);
        patterns.add<ConvertArithCmpIOp>(typeConverter, ctx);
        patterns.add<ConvertArithAddIOp>(typeConverter, ctx);
        patterns.add<ConvertAddWrappingOp>(typeConverter, ctx);
        patterns.add<ConvertSubWrappingOp>(typeConverter, ctx);
        patterns.add<ConvertMulWrappingOp>(typeConverter, ctx);
        patterns.add<ConvertArithSubIOp>(typeConverter, ctx);
        patterns.add<ConvertArithMulIOp>(typeConverter, ctx);
        patterns.add<ConvertArithDivUIOp>(typeConverter, ctx);
        patterns.add<ConvertArithRemUIOp>(typeConverter, ctx);
        patterns.add<ConvertArithDivSIOp>(typeConverter, ctx);
        patterns.add<ConvertArithRemSIOp>(typeConverter, ctx);
        patterns.add<ConvertArithAndIOp>(typeConverter, ctx);
        patterns.add<ConvertArithOrIOp>(typeConverter, ctx);
        patterns.add<ConvertArithXOrIOp>(typeConverter, ctx);
        patterns.add<ConvertArithShlIOp>(typeConverter, ctx);
        patterns.add<ConvertShlWrappingOp>(typeConverter, ctx);
        patterns.add<ConvertArithShrUIOp>(typeConverter, ctx);
        patterns.add<ConvertArithShrSIOp>(typeConverter, ctx);
        patterns.add<ConvertShrWrappingOp>(typeConverter, ctx);
        patterns.add<ConvertArithSelectOp>(typeConverter, ctx);
        addArithExtensionCastPatterns(patterns, typeConverter, ctx);
        patterns.add<ConvertArithTruncIOp>(typeConverter, ctx);
    }

    static void addOraRegionControlLegalOps(ConversionTarget &target)
    {
        target.addLegalOp<ora::YieldOp>();
        target.addLegalOp<ora::ContinueOp>();
        target.addLegalOp<ora::SwitchOp>();
        target.addLegalDialect<ora::OraDialect>();
    }

    static Value getStorageMemRefViewRootSlot(Value value)
    {
        if (!value)
            return Value();

        if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (cast.getNumOperands() != 1)
                return Value();
            Value operand = cast.getOperand(0);
            if (ora::hasMaterializationKind(cast, lowering::kStorageMemRefViewKind) &&
                llvm::isa<sir::U256Type>(operand.getType()))
                return operand;
            return getStorageMemRefViewRootSlot(operand);
        }

        if (auto bitcast = value.getDefiningOp<sir::BitcastOp>())
        {
            Value operand = bitcast.getInput();
            auto viewKind = bitcast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
            if (viewKind && viewKind.getValue() == lowering::kStorageMemRefViewKind &&
                llvm::isa<sir::U256Type>(operand.getType()))
                return operand;
            return getStorageMemRefViewRootSlot(operand);
        }

        return Value();
    }

    static std::optional<APInt> constU256(Value value)
    {
        if (auto constOp = value.getDefiningOp<sir::ConstOp>())
            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr()))
                return intAttr.getValue().zextOrTrunc(256);
        return std::nullopt;
    }

    static bool isLowBitsMask(Value value, unsigned width)
    {
        if (width == 0 || width > 256)
            return false;
        auto mask = constU256(value);
        return mask && *mask == APInt::getLowBitsSet(256, width);
    }

    static bool isIntegerNormalizationProducer(Operation *op, unsigned width)
    {
        if (llvm::isa<sir::SignExtendOp>(op))
            return true;
        if (llvm::isa<sir::SDivOp, sir::SModOp, sir::SarOp>(op))
            return true;
        if (auto andOp = llvm::dyn_cast<sir::AndOp>(op))
            return isLowBitsMask(andOp.getLhs(), width) || isLowBitsMask(andOp.getRhs(), width);
        return false;
    }

    static bool canDropExplicitIntegerCarrierRoundTrip(sir::BitcastOp inner, sir::BitcastOp outer)
    {
        if (inner->getNumOperands() != 1 || inner->getNumResults() != 1 ||
            outer->getNumOperands() != 1 || outer->getNumResults() != 1)
            return false;
        if (!llvm::isa<sir::U256Type>(inner.getInput().getType()) ||
            !llvm::isa<sir::U256Type>(outer.getResult().getType()))
            return false;
        auto middleInt = llvm::dyn_cast<mlir::IntegerType>(inner.getResult().getType());
        if (!middleInt || middleInt.getWidth() <= 1 || middleInt.getWidth() > 256)
            return false;
        if (!inner.getResult().hasOneUse())
            return false;
        Operation *producer = inner.getInput().getDefiningOp();
        return producer && isIntegerNormalizationProducer(producer, middleInt.getWidth());
    }

    static bool canDropSirBitcastRoundTrip(sir::BitcastOp inner, sir::BitcastOp outer)
    {
        if (inner->getNumOperands() != 1 || inner->getNumResults() != 1 ||
            outer->getNumOperands() != 1 || outer->getNumResults() != 1)
            return false;
        Type middle = inner.getResult().getType();
        const bool middleIsWordCarrier =
            llvm::isa<sir::PtrType, sir::U256Type>(middle) ||
            (llvm::isa<mlir::IntegerType>(middle) &&
             llvm::cast<mlir::IntegerType>(middle).getWidth() == 256);
        return middleIsWordCarrier &&
               inner.getInput().getType() == outer.getResult().getType() &&
               inner.getResult().hasOneUse();
    }

    static void foldExplicitIntegerCarrierRoundTripBitcasts(ModuleOp module)
    {
        // Retain only representational SIR-bitcast cleanup here. Same-type
        // identities and round trips with an explicit normalization producer
        // are safe to erase. Arbitrary carrier-changing A->B->A shapes stay
        // visible unless a kind-specific lowering owns them.
        bool localChanged = true;
        while (localChanged)
        {
            localChanged = false;

            sir::BitcastOp identityToErase;
            SmallVector<sir::BitcastOp, 32> bitcasts;
            module.walk([&](sir::BitcastOp op) { bitcasts.push_back(op); });
            for (auto op : bitcasts)
            {
                if (identityToErase)
                    break;
                if (!op->getBlock())
                    continue;
                if (op->getNumOperands() != 1 || op->getNumResults() != 1)
                    continue;
                if (op.getInput().getType() == op.getResult().getType())
                    identityToErase = op;
            }
            if (identityToErase)
            {
                identityToErase.getResult().replaceAllUsesWith(identityToErase.getInput());
                identityToErase.erase();
                localChanged = true;
                continue;
            }

            sir::BitcastOp outerToFold;
            sir::BitcastOp innerToFold;
            bitcasts.clear();
            module.walk([&](sir::BitcastOp op) { bitcasts.push_back(op); });
            for (auto op : bitcasts)
            {
                if (outerToFold)
                    break;
                if (!op->getBlock())
                    continue;
                if (op->getNumOperands() != 1 || op->getNumResults() != 1)
                    continue;
                auto inner = op.getInput().getDefiningOp<sir::BitcastOp>();
                if (!inner || !inner->getBlock() || inner.getResult() != op.getInput())
                    continue;
                if (inner->getNumOperands() != 1 || inner->getNumResults() != 1)
                    continue;
                if (canDropSirBitcastRoundTrip(inner, op) ||
                    canDropExplicitIntegerCarrierRoundTrip(inner, op))
                {
                    outerToFold = op;
                    innerToFold = inner;
                }
            }
            if (outerToFold)
            {
                outerToFold.getResult().replaceAllUsesWith(innerToFold.getInput());
                outerToFold.erase();
                localChanged = true;

                if (innerToFold.getResult().use_empty())
                    innerToFold.erase();
            }
        }
    }

    class RefinementErasureTypeConverter final : public TypeConverter
    {
    public:
        RefinementErasureTypeConverter()
        {
            addConversion([](Type type)
                          { return type; });

            addConversion([&](ora::MinValueType type)
                          { return convertType(type.getBaseType()); });
            addConversion([&](ora::MaxValueType type)
                          { return convertType(type.getBaseType()); });
            addConversion([&](ora::InRangeType type)
                          { return convertType(type.getBaseType()); });
            addConversion([&](ora::ScaledType type)
                          { return convertType(type.getBaseType()); });
            addConversion([&](ora::ExactType type)
                          { return convertType(type.getBaseType()); });
            addConversion([&](ora::NonZeroAddressType type)
                          { return ora::AddressType::get(type.getContext()); });

            addConversion([&](ora::ErrorUnionType type) -> Type
                          {
                auto successType = convertType(type.getSuccessType());
                llvm::SmallVector<Type> errorTypes;
                for (auto errorType : type.getErrorTypes())
                    errorTypes.push_back(convertType(errorType));
                return ora::ErrorUnionType::get(type.getContext(), successType, errorTypes); });
            addConversion([&](ora::MapType type) -> Type
                          {
                auto key = convertType(type.getKeyType());
                auto value = convertType(type.getValueType());
                return ora::MapType::get(type.getContext(), key, value); });
            addConversion([&](ora::TupleType type) -> Type
                          {
                SmallVector<Type> elems;
                elems.reserve(type.getElementTypes().size());
                for (Type elem : type.getElementTypes())
                    elems.push_back(convertType(elem));
                return ora::TupleType::get(type.getContext(), elems); });

            addConversion([&](RankedTensorType type) -> Type
                          {
                auto elem = convertType(type.getElementType());
                return RankedTensorType::get(type.getShape(), elem, type.getEncoding()); });
            addConversion([&](UnrankedTensorType type) -> Type
                          {
                auto elem = convertType(type.getElementType());
                return UnrankedTensorType::get(elem); });
            addConversion([&](MemRefType type) -> Type
                          {
                auto elem = convertType(type.getElementType());
                return MemRefType::get(type.getShape(), elem, type.getLayout(), type.getMemorySpace()); });
            addConversion([&](mlir::FunctionType type) -> Type
                          {
                SmallVector<Type> inputs;
                SmallVector<Type> results;
                inputs.reserve(type.getInputs().size());
                results.reserve(type.getResults().size());
                for (Type in : type.getInputs())
                    inputs.push_back(convertType(in));
                for (Type out : type.getResults())
                    results.push_back(convertType(out));
                return mlir::FunctionType::get(type.getContext(), inputs, results); });
            addConversion([&](ora::FunctionType type) -> Type
                          {
                SmallVector<Type> inputs;
                inputs.reserve(type.getParamTypes().size());
                for (Type in : type.getParamTypes())
                    inputs.push_back(convertType(in));
                Type result = convertType(type.getReturnType());
                return ora::FunctionType::get(type.getContext(), inputs, result); });

            addSourceMaterialization([&](OpBuilder &builder,
                                         Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value
                                     {
                if (inputs.empty())
                    return Value();
                return ora::createMaterializationCast(
                    builder, loc, resultType, inputs, "refinement_forward"); });
            addTargetMaterialization([&](OpBuilder &builder,
                                         Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value
                                     {
                if (inputs.empty())
                    return Value();
                return ora::createMaterializationCast(
                    builder, loc, resultType, inputs, "refinement_forward"); });
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
            Value materialized = ora::createMaterializationCast(
                rewriter, op.getLoc(), targetType, value, "refinement_forward");
            rewriter.replaceOp(op, materialized);
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
            Value materialized = ora::createMaterializationCast(
                rewriter, op.getLoc(), targetType, value, "refinement_forward");
            rewriter.replaceOp(op, materialized);
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
            (void)adaptor;
            SmallVector<Value> newOperands;
            newOperands.reserve(op.getNumOperands());
            bool changed = false;
            for (auto it : llvm::enumerate(op.getOperands()))
            {
                Value operand = it.value();
                Type origType = operand.getType();
                SmallVector<Type> convertedOperandTypes;
                if (llvm::isa<ora::AdtType>(origType))
                {
                    auto *ctx = origType.getContext();
                    convertedOperandTypes.push_back(sir::U256Type::get(ctx));
                    convertedOperandTypes.push_back(sir::U256Type::get(ctx));
                }
                else if (failed(typeConverter->convertType(origType, convertedOperandTypes)) || convertedOperandTypes.empty())
                {
                    if (Type converted = typeConverter->convertType(origType))
                        convertedOperandTypes.push_back(converted);
                    else if (llvm::isa<mlir::IntegerType>(origType))
                        convertedOperandTypes.push_back(origType);
                }

                if (convertedOperandTypes.empty())
                    return failure();

                if (convertedOperandTypes.size() == 1)
                {
                    Type newType = convertedOperandTypes[0];
                    if (newType != origType)
                        changed = true;
                    if (newType != operand.getType())
                    {
                        changed = true;
                        if (Value materialized = typeConverter->materializeTargetConversion(
                                rewriter, op.getLoc(), newType, operand))
                            operand = materialized;
                        else
                            operand = ora::createMaterializationCast(
                                rewriter, op.getLoc(), newType, operand, "call_forward");
                    }
                    newOperands.push_back(operand);
                    continue;
                }

                changed = true;
                StringRef materializationKind;
                if (llvm::isa<ora::ErrorUnionType>(origType))
                    materializationKind = mat_kind::kWideErrorUnionSplit;
                else if (llvm::isa<ora::AdtType>(origType))
                    materializationKind = mat_kind::kNormalizedAdt;
                else
                    return failure();

                auto cast = ora::createMaterializationCastOp(
                    rewriter, op.getLoc(), convertedOperandTypes, ValueRange{operand}, materializationKind);
                for (Value split : cast.getResults())
                    newOperands.push_back(split);
            }

            if (!changed)
                return failure();

            auto newCall = rewriter.create<mlir::func::CallOp>(
                op.getLoc(),
                op.getCalleeAttr(),
                op.getResultTypes(),
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
                return failure(); // no conversion needed
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

            auto updateTypeAttr = [&](NamedAttrList &attrs, StringRef name)
            {
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

            rewriter.modifyOpInPlace(op, [&]
                                     {
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
                } });

            return success();
        }
    };
}

static void logModuleOps(ModuleOp module, StringRef tag)
{
    if (!mlir::ora::isDebugEnabled())
        return;
    llvm::errs() << "[OraToSIR] " << tag << " (per-op log)\n";
    module.walk([&](Operation *op)
                {
                    llvm::errs() << "[OraToSIR]   ";
                    op->print(llvm::errs());
                    llvm::errs() << "\n";
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

static LogicalResult applyFullConversionWithDiagnostics(
    ModuleOp module, ConversionTarget &target, RewritePatternSet &&patterns,
    StringRef failureMessage, StringRef failureDump = StringRef(),
    StringRef beforeLog = StringRef(), StringRef afterLog = StringRef(),
    StringRef failureLog = StringRef(), const ConversionConfig *config = nullptr)
{
    if (!beforeLog.empty())
        logModuleOps(module, beforeLog);
    LogicalResult result = config ? applyFullConversion(module, target, std::move(patterns), *config)
                                  : applyFullConversion(module, target, std::move(patterns));
    if (failed(result))
    {
        if (!failureLog.empty())
            logModuleOps(module, failureLog);
        if (!failureDump.empty())
            dumpModuleOnFailure(module, failureDump);
        module.emitError(failureMessage);
        return failure();
    }
    if (!afterLog.empty())
        logModuleOps(module, afterLog);
    return success();
}

static bool normalizeFuncTerminators(mlir::func::FuncOp funcOp)
{
    mlir::IRRewriter rewriter(funcOp.getContext());
    bool hadMalformedBlock = false;
    SmallVector<Block *, 4> blocksToErase;
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
            llvm::errs() << "[OraToSIR] ERROR: Missing terminator in function "
                         << funcOp.getName() << " at " << block.getParent()->getLoc() << "\n";
            if (&block != &funcOp.front() && block.empty() && block.hasNoPredecessors())
            {
                blocksToErase.push_back(&block);
                continue;
            }
            if (Block *next = block.getNextNode())
            {
                if (next->getNumArguments() == 0)
                {
                    rewriter.setInsertionPointToEnd(&block);
                    rewriter.create<sir::BrOp>(funcOp.getLoc(), ValueRange{}, next);
                    continue;
                }
            }
            hadMalformedBlock = true;
            rewriter.setInsertionPointToEnd(&block);
            rewriter.create<sir::InvalidOp>(funcOp.getLoc());
            continue;
        }
        if (terminator->getNextNode())
        {
            llvm::errs() << "[OraToSIR] ERROR: Terminator has trailing ops in function "
                         << funcOp.getName() << " at " << terminator->getLoc() << "\n";
            // Keep IR valid for downstream passes by dropping trailing ops
            // that were left after a terminator.
            Operation *extra = terminator->getNextNode();
            while (extra)
            {
                Operation *next = extra->getNextNode();
                extra->erase();
                extra = next;
            }
        }
    }
    for (Block *block : blocksToErase)
        rewriter.eraseBlock(block);
    return hadMalformedBlock;
}

static LogicalResult normalizeResidualAdtExtractOps(ModuleOp module)
{
    auto *ctx = module.getContext();
    SmallVector<ora::AdtTagOp, 8> tagOps;
    SmallVector<ora::AdtPayloadOp, 8> payloadOps;

    module.walk([&](ora::AdtTagOp op) { tagOps.push_back(op); });
    module.walk([&](ora::AdtPayloadOp op) { payloadOps.push_back(op); });

    mlir::IRRewriter rewriter(ctx);
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    for (auto op : tagOps)
    {
        if (!op || op->getBlock() == nullptr)
            continue;
        rewriter.setInsertionPoint(op);
        auto split = ora::createMaterializationCastOp(
            rewriter, op.getLoc(), TypeRange{u256Type, u256Type}, ValueRange{op.getValue()}, mat_kind::kNormalizedAdt);
        Value replacement = ora::createMaterializationCast(
            rewriter, op.getLoc(), op.getType(), split.getResult(0), mat_kind::kPayloadForward);
        rewriter.replaceOp(op, replacement);
    }

    for (auto op : payloadOps)
    {
        if (!op || op->getBlock() == nullptr)
            continue;
        rewriter.setInsertionPoint(op);
        auto split = ora::createMaterializationCastOp(
            rewriter, op.getLoc(), TypeRange{u256Type, u256Type}, ValueRange{op.getValue()}, mat_kind::kNormalizedAdt);
        Value payload = split.getResult(1);
        Type resultType = op.getType();

        // Aggregate ADT payloads are carried as compiler-managed ptr-backed
        // values encoded into the payload word at the ADT boundary.
        if (llvm::isa<ora::TupleType, ora::StructType, ora::AnonymousStructType, ora::StringType, ora::BytesType,
                      mlir::MemRefType, mlir::UnrankedMemRefType>(resultType))
        {
            Value payloadPtr = rewriter.create<sir::BitcastOp>(op.getLoc(), ptrType, payload);
            auto view = lowering::createPtrViewMaterializationCast(rewriter, op.getLoc(), resultType, payloadPtr);
            rewriter.replaceOp(op, view);
            continue;
        }

        Value replacement = ora::createMaterializationCast(
            rewriter, op.getLoc(), resultType, payload, mat_kind::kPayloadForward);
        rewriter.replaceOp(op, replacement);
    }

    return success();
}

static LogicalResult eraseRefinements(ModuleOp module)
{
    MLIRContext *ctx = module.getContext();
    RefinementErasureTypeConverter typeConverter;

    ConversionTarget target(*ctx);
    addOraToSirBaseLegalDialects(target);
    target.addLegalDialect<ora::OraDialect>();

    target.addIllegalOp<ora::RefinementToBaseOp, ora::BaseToRefinementOp>();

    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op)
                                                     { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<ora::GlobalOp>([&](ora::GlobalOp op)
                                                { return typeConverter.isLegal(op.getType()); });
    target.markUnknownOpDynamicallyLegal([&](Operation *op)
                                         {
        for (Type t : op->getOperandTypes())
            if (!typeConverter.isLegal(t))
                return false;
        for (Type t : op->getResultTypes())
            if (!typeConverter.isLegal(t))
                return false;
        return true; });

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

static bool isPreservedUnrealizedMaterialization(mlir::UnrealizedConversionCastOp castOp)
{
    return ora::hasMaterializationKind(castOp, mat_kind::kNormalizedErrorUnion) ||
           ora::hasMaterializationKind(castOp, mat_kind::kNormalizedAdt) ||
           ora::hasMaterializationKind(castOp, mat_kind::kAdtHandleView) ||
           ora::hasMaterializationKind(castOp, mat_kind::kPtrView) ||
           ora::hasMaterializationKind(castOp, mat_kind::kAddressForward) ||
           ora::hasMaterializationKind(castOp, mat_kind::kWideErrorUnionJoin) ||
           ora::hasMaterializationKind(castOp, mat_kind::kWideErrorUnionSplit);
}

static LogicalResult verifyNoUnexpectedUnrealizedCasts(ModuleOp module)
{
    const bool debug = mlir::ora::isDebugEnabled();
    bool leftoverUnrealized = false;
    for (auto castOp : module.getOps<mlir::UnrealizedConversionCastOp>())
    {
        if (isPreservedUnrealizedMaterialization(castOp))
            continue;
        leftoverUnrealized = true;
        if (!debug)
            break;
        llvm::errs() << "[OraToSIR] Phase4 post-scan: unrealized cast at "
                     << castOp.getLoc() << " operands=" << castOp.getNumOperands()
                     << " results=" << castOp.getNumResults() << "\n";
    }
    if (debug)
        llvm::errs().flush();
    if (leftoverUnrealized)
    {
        module.emitError("[OraToSIR] Phase4 post-scan: unrealized casts remain");
        return failure();
    }

    if (mlir::ora::isDebugEnabled())
    {
        llvm::errs() << "[OraToSIR] Post-Phase4: name-scan start\n";
        llvm::errs().flush();
    }
    int64_t unrealizedByName = 0;
    module.walk([&](mlir::UnrealizedConversionCastOp castOp) {
        if (isPreservedUnrealizedMaterialization(castOp))
            return;
        ++unrealizedByName;
        if (!debug)
            return;
        llvm::errs() << "[OraToSIR] Phase4 name-scan: unrealized cast at "
                     << castOp.getLoc() << " operands=" << castOp.getNumOperands()
                     << " results=" << castOp.getNumResults();
        if (castOp.getNumOperands() > 0)
            llvm::errs() << " in=" << castOp.getOperand(0).getType();
        if (castOp.getNumResults() > 0)
            llvm::errs() << " out=" << castOp.getResult(0).getType();
        llvm::errs() << "\n";
    });
    if (mlir::ora::isDebugEnabled())
    {
        llvm::errs() << "[OraToSIR] Post-Phase4: name-scan done (count=" << unrealizedByName << ")\n";
        llvm::errs().flush();
    }
    if (unrealizedByName > 0)
    {
        module.emitError("[OraToSIR] Phase4 name-scan: unrealized casts remain");
        return failure();
    }
    return success();
}

static void assignGlobalSlots(ModuleOp module)
{
    auto *ctx = module.getContext();
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    SmallVector<NamedAttribute> slotAttrs;
    const bool requireExistingSlotMetadata = module->hasAttr("ora.global_slots_built");

    auto storageWordCount = [](Operation &op) -> uint64_t
    {
        Type type;
        if (auto globalOp = dyn_cast<ora::GlobalOp>(op))
        {
            type = globalOp.getGlobalType();
        }
        else if (auto typeAttr = op.getAttrOfType<TypeAttr>("type"))
        {
            type = typeAttr.getValue();
        }

        return mlir::ora::lowering::getStorageWordCount(&op, type);
    };

    auto advanceSlot = [](uint64_t &slot, uint64_t start, uint64_t words)
    {
        uint64_t next = start + (words == 0 ? 1 : words);
        if (next > slot)
            slot = next;
    };

    auto assignInBlock = [&](Block &block)
    {
        uint64_t slot = 0;
        for (Operation &op : block)
        {
            if (auto globalOp = dyn_cast<ora::GlobalOp>(op))
            {
                uint64_t wordCount = storageWordCount(op);
                if (globalOp->getAttrOfType<IntegerAttr>("ora.slot_index"))
                {
                    auto nameAttr = globalOp->getAttrOfType<StringAttr>("sym_name");
                    auto slotAttr = globalOp->getAttrOfType<IntegerAttr>("ora.slot_index");
                    if (nameAttr && slotAttr)
                    {
                        slotAttrs.push_back(NamedAttribute(nameAttr, slotAttr));
                    }
                    advanceSlot(slot, slotAttr.getUInt(), wordCount);
                    continue;
                }
                if (requireExistingSlotMetadata)
                {
                    advanceSlot(slot, slot, wordCount);
                    continue;
                }
                auto slotAttr = mlir::IntegerAttr::get(ui64Type, slot);
                globalOp->setAttr("ora.slot_index", slotAttr);
                auto nameAttr = globalOp->getAttrOfType<StringAttr>("sym_name");
                if (nameAttr)
                {
                    slotAttrs.push_back(NamedAttribute(nameAttr, slotAttr));
                }
                advanceSlot(slot, slotAttr.getUInt(), wordCount);
                continue;
            }

            // Also assign slots to non-ora.global storage-like declarations.
            auto opName = op.getName().getStringRef();
            if (opName == "ora.tstore.global" || opName == "ora.memory.global")
            {
                uint64_t wordCount = storageWordCount(op);
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
                    advanceSlot(slot, slotAttr.getUInt(), wordCount);
                    continue;
                }
                if (requireExistingSlotMetadata)
                {
                    advanceSlot(slot, slot, wordCount);
                    continue;
                }
                auto slotAttr = mlir::IntegerAttr::get(ui64Type, slot);
                op.setAttr("ora.slot_index", slotAttr);
                slotAttrs.push_back(NamedAttribute(nameAttr, slotAttr));
                advanceSlot(slot, slotAttr.getUInt(), wordCount);
            }
        }
    };

    module.walk([&](ora::ContractOp contractOp)
                { assignInBlock(contractOp.getBody().front()); });

    if (!module.getBody()->empty())
        assignInBlock(module.getBodyRegion().front());

    SmallVector<Attribute> ambiguousNames;
    for (size_t i = 0; i < slotAttrs.size(); i++)
    {
        auto name = slotAttrs[i].getName();
        bool duplicate = false;
        for (size_t j = 0; j < slotAttrs.size(); j++)
        {
            if (i != j && slotAttrs[j].getName() == name)
            {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
            continue;

        bool alreadyRecorded = false;
        for (Attribute existing : ambiguousNames)
        {
            if (existing == name)
            {
                alreadyRecorded = true;
                break;
            }
        }
        if (!alreadyRecorded)
            ambiguousNames.push_back(name);
    }

    if (!slotAttrs.empty() && !requireExistingSlotMetadata)
    {
        module->setAttr("ora.global_slots", DictionaryAttr::get(ctx, slotAttrs));
    }
    if (!requireExistingSlotMetadata && !slotAttrs.empty())
    {
        module->setAttr("ora.global_slots_built", UnitAttr::get(ctx));
    }
    if (!ambiguousNames.empty())
    {
        module->setAttr("ora.global_slot_ambiguous_names", ArrayAttr::get(ctx, ambiguousNames));
    }
}

static void inlineContractsAndEraseDecls(ModuleOp module)
{
    // Inline contract bodies into the module block and erase contract wrappers.
    SmallVector<ora::ContractOp, 4> contracts;
    module.walk([&](ora::ContractOp contractOp)
                { contracts.push_back(contractOp); });

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

static void preserveEnumDiscriminants(ModuleOp module, MLIRContext *ctx)
{
    // Preserve enum discriminants before ora.enum.decl is erased. Enum
    // constants can then lower deterministically even if the declaration
    // was removed earlier in the greedy conversion.
    SmallVector<NamedAttribute> enumEntries;
    module.walk([&](ora::EnumDeclOp decl) {
        auto variantNames = decl->getAttrOfType<mlir::ArrayAttr>("ora.variant_names");
        auto arrayVariantValues = decl->getAttrOfType<mlir::ArrayAttr>("ora.variant_values");
        auto denseVariantValues = decl->getAttrOfType<mlir::DenseI64ArrayAttr>("ora.variant_values");
        if (!variantNames || (!arrayVariantValues && !denseVariantValues))
            return;

        const size_t valueCount = denseVariantValues ? denseVariantValues.size() : arrayVariantValues.size();
        const size_t count = std::min<size_t>(variantNames.size(), valueCount);
        for (size_t i = 0; i < count; ++i)
        {
            auto nameAttr = dyn_cast<mlir::StringAttr>(variantNames[i]);
            Attribute valueAttr;
            if (denseVariantValues)
            {
                valueAttr = mlir::IntegerAttr::get(
                    mlir::IntegerType::get(ctx, 64),
                    denseVariantValues[i]);
            }
            else
            {
                valueAttr = arrayVariantValues[i];
            }
            if (!nameAttr || !valueAttr)
                continue;

            std::string key = decl.getName().str();
            key.push_back('.');
            key += nameAttr.getValue().str();
            enumEntries.push_back(NamedAttribute(StringAttr::get(ctx, key), valueAttr));
        }
    });
    if (!enumEntries.empty())
        module->setAttr("sir.enum_values", DictionaryAttr::get(ctx, enumEntries));
}

// Thin deterministic fallback for passes that have not yet moved to framework
// constant CSE. The SIR text handoff now accepts cross-block constants via
// inline numeric operands, so new SIR canonicalization should prefer MLIR CSE
// and keep this helper as a local cleanup/backstop only.
static Attribute getSIRConstDedupKey(MLIRContext *ctx, sir::ConstOp constOp)
{
    Attribute value = constOp.getValueAttr();
    auto intAttr = dyn_cast<IntegerAttr>(value);
    if (!intAttr)
        return value;

    auto u256Type = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
    return IntegerAttr::get(u256Type, intAttr.getValue().zextOrTrunc(256));
}

static uint64_t deduplicateConstantsPerBlock(ModuleOp module)
{
    uint64_t deduplicated = 0;
    module.walk([&](Block *block)
                {
        DenseMap<Attribute, Value> consts;
        for (Operation &op : llvm::make_early_inc_range(*block))
        {
            auto constOp = dyn_cast<sir::ConstOp>(&op);
            if (!constOp)
                continue;
            Attribute key = getSIRConstDedupKey(module.getContext(), constOp);
            auto it = consts.find(key);
            if (it != consts.end())
            {
                constOp.replaceAllUsesWith(it->second);
                constOp.erase();
                ++deduplicated;
                continue;
            }
            consts.insert({key, constOp.getResult()});
        } });
    return deduplicated;
}

// Deterministic release-path framework slice: run only the SIR op
// canonicalizers whose output is already accepted in production goldens.
// Broader framework canonicalization/DCE runs later as default SIR hygiene.
template <typename... OpTys>
static LogicalResult applySelectedSIRCanonicalizationPatterns(ModuleOp module, bool &changed)
{
    RewritePatternSet patterns(module.getContext());
    (OpTys::getCanonicalizationPatterns(patterns, module.getContext()), ...);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    SmallVector<Operation *, 32> ops;
    module.walk([&](Operation *op)
                {
        if ((llvm::isa<OpTys>(op) || ...))
            ops.push_back(op); });
    if (ops.empty())
        return success();

    GreedyRewriteConfig config;
    config.enableConstantCSE(true);
    config.enableFolding(true);
    config.setMaxIterations(1);
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    return applyOpPatternsGreedily(ops, frozenPatterns, config, &changed);
}

static LogicalResult canonicalizeSIRConstantWordOps(ModuleOp module)
{
    bool changed = true;
    while (changed)
    {
        changed = false;

        bool passChanged = false;
        if (failed(applySelectedSIRCanonicalizationPatterns<
                   sir::AddOp, sir::SubOp, sir::MulOp,
                   sir::LtOp, sir::GtOp, sir::SLtOp, sir::SGtOp,
                   sir::AndOp, sir::OrOp, sir::XorOp>(
                module, passChanged)))
            return failure();
        changed |= passChanged;
    }

    return success();
}

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

            // NOTE: memref::StoreOp is side-effecting — do NOT erase based on use_empty().

            module.walk([&](mlir::memref::AllocaOp allocaOp)
                        {
                    if (allocaOp->use_empty())
                    {
                        DBG("SIRCleanupPass: removing unused alloca");
                        allocaOp->erase();
                        ++unusedAllocasRemoved;
                        recordOraMlirPassStatistic(OraMlirPassStatistic::SirUnusedAllocasRemoved);
                        changed = true;
                    } });

            module.walk([&](mlir::memref::LoadOp loadOp)
                        {
                if (loadOp->use_empty())
                {
                    DBG("SIRCleanupPass: removing unused load");
                    loadOp->erase();
                    ++unusedLoadsRemoved;
                    recordOraMlirPassStatistic(OraMlirPassStatistic::SirUnusedLoadsRemoved);
                    changed = true;
                } });

            const uint64_t dedupedConstants = deduplicateConstantsPerBlock(module);
            constantsDeduplicated += dedupedConstants;
            recordOraMlirPassStatistic(OraMlirPassStatistic::SirConstantsDeduplicated, dedupedConstants);
            changed |= dedupedConstants != 0;

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
                ++unusedPureOpsRemoved;
                recordOraMlirPassStatistic(OraMlirPassStatistic::SirUnusedPureOpsRemoved);
                changed = true; });
        }

        DBG("SIRCleanupPass: cleanup completed");
    }

private:
    Pass::Statistic unusedAllocasRemoved{this, "unused-allocas-removed", "Unused memref allocas removed"};
    Pass::Statistic unusedLoadsRemoved{this, "unused-loads-removed", "Unused memref loads removed"};
    Pass::Statistic unusedPureOpsRemoved{this, "unused-pure-ops-removed", "Unused pure operations removed"};
    Pass::Statistic constantsDeduplicated{this, "constants-deduplicated", "Duplicate SIR constants removed"};
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
        preserveEnumDiscriminants(module, ctx);
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

        MapHashCache mapHashCache;
        MemRefNamingCache memRefNamingCache;
        RewritePatternSet patterns(ctx);

        patterns.add<ConvertContractOp>(typeConverter, ctx);
        addArithmeticLoweringPatterns(patterns, typeConverter, ctx);
        patterns.add<ConvertGlobalOp>(typeConverter, ctx);
        patterns.add<ConvertFuncOp>(typeConverter, ctx);
        patterns.add<ConvertAbiEncodeOp>(typeConverter, ctx);
        patterns.add<ConvertAbiEncodeWithSelectorOp>(typeConverter, ctx);
        patterns.add<ConvertExternalCallOp>(typeConverter, ctx);
        patterns.add<ConvertAbiDecodeOp>(typeConverter, ctx);

        // ora.add/sub/mul/div/rem no longer emitted; arith.* used directly.
        patterns.add<ConvertCmpOp>(typeConverter, ctx);
        patterns.add<ConvertConstOp>(typeConverter, ctx);
        patterns.add<ConvertLengthOp>(typeConverter, ctx);
        patterns.add<ConvertByteAtOp>(typeConverter, ctx);
        patterns.add<ConvertConcatOp>(typeConverter, ctx);
        patterns.add<ConvertSliceOp>(typeConverter, ctx);
        patterns.add<ConvertKeccak256Op>(typeConverter, ctx);
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

        // Defer struct lowering until the later struct/error-payload phase.
        // This avoids forcing struct/tuple->ptr materializations before wide
        // error_union accessors have been normalized.
        patterns.add<ConvertRefinementToBaseOp>(typeConverter, ctx);
        patterns.add<ConvertBaseToRefinementOp>(typeConverter, ctx);
        patterns.add<ConvertEvmOp>(typeConverter, ctx);
        // Memref lowering happens in Phase 4; do not add memref patterns here.
        patterns.add<NormalizeAdtSLoadOp>(ctx);
        patterns.add<NormalizeAdtSStoreOp>(ctx);
        patterns.add<ConvertSLoadOp>(typeConverter, ctx);
        patterns.add<ConvertSStoreOp>(typeConverter, ctx);
        patterns.add<ConvertStorageDeriveOp>(typeConverter, ctx);
        patterns.add<ConvertStorageWordLoadOp>(typeConverter, ctx);
        patterns.add<ConvertStorageWordStoreOp>(typeConverter, ctx);
        patterns.add<ConvertStorageRangeEraseOp>(typeConverter, ctx);
        patterns.add<ConvertTLoadOp>(typeConverter, ctx);
        patterns.add<ConvertTStoreOp>(typeConverter, ctx);
        patterns.add<ConvertResourceCreateOp>(typeConverter, ctx, mapHashCache, PatternBenefit(5));
        patterns.add<ConvertResourceDestroyOp>(typeConverter, ctx, mapHashCache, PatternBenefit(5));
        patterns.add<ConvertResourceMoveOp>(typeConverter, ctx, mapHashCache, PatternBenefit(5));
        patterns.add<ConvertMapGetOp>(typeConverter, ctx, mapHashCache, PatternBenefit(5));
        patterns.add<ConvertMapStoreOp>(typeConverter, ctx, mapHashCache, PatternBenefit(5));
        patterns.add<ConvertTensorInsertOp>(typeConverter, ctx);
        patterns.add<ConvertTensorExtractOp>(typeConverter, ctx);
        patterns.add<ConvertTensorDimOp>(typeConverter, ctx);
        // Defer ora.return lowering to phase 2 so scf.if results are split first.
        patterns.add<ConvertIfOp>(typeConverter, ctx);
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
        patterns.add<ConvertErrorDeclOp>(typeConverter, ctx);
        patterns.add<ConvertErrorReturnOp>(typeConverter, ctx);
        patterns.add<ConvertUnrealizedConversionCastOp>(typeConverter, ctx);
        patterns.add<ConvertReturnOp>(typeConverter, ctx, true);
        patterns.add<ConvertCallOp>(typeConverter, ctx, true);
        patterns.add<ConvertCallTypeOp>(typeConverter, ctx);
        patterns.add<ConvertAdtTagOneToNOp>(typeConverter, ctx);
        patterns.add<ConvertAdtPayloadOneToNOp>(typeConverter, ctx);
        patterns.add<ConvertFuncTypeAttrsOp>(typeConverter, ctx);
        patterns.add<ConvertLogOp>(typeConverter, ctx);
        patterns.add<EraseOpByName>("ora.enum.decl", ctx);
        patterns.add<EraseOpByName>("ora.bitfield_decl", ctx);
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
        patterns.add<ora::ConvertLockOp>(typeConverter, ctx);
        patterns.add<ora::ConvertUnlockOp>(typeConverter, ctx);
        patterns.add<ora::ConvertTStoreGuardOp>(typeConverter, ctx);

        ConversionTarget target(*ctx);
        // Mark SIR dialect as legal
        target.addLegalDialect<mlir::BuiltinDialect>();
        target.addLegalDialect<sir::SIRDialect>();
        DBG("Marked SIR dialect as legal");
        // Ora ops are illegal by default; no Ora ops should remain after conversion
        target.addIllegalDialect<ora::OraDialect>();

        // Force storage-related tensor ops to lower when arrays/maps are enabled.
        target.addIllegalOp<mlir::tensor::InsertOp, mlir::tensor::ExtractOp, mlir::tensor::DimOp>();
        target.addIllegalOp<ora::ContractOp>();
        target.addDynamicallyLegalOp<ora::ReturnOp>(
            [&](ora::ReturnOp op)
            {
                for (Value operand : op.getOperands())
                {
                    if (euh::isPayloadlessErrorStruct(operand.getType(), op))
                        return false;
                }
                return true;
            });
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
        target.addLegalOp<ora::TupleCreateOp>();
        target.addLegalOp<ora::TupleExtractOp>();
        target.addLegalOp<ora::AdtConstructOp>();
        target.addLegalOp<mlir::UnrealizedConversionCastOp>();
        target.addLegalOp<ora::StructInstantiateOp>();
        target.addLegalOp<ora::StructInitOp>();
        target.addLegalOp<ora::StructFieldExtractOp>();
        target.addLegalOp<ora::StructFieldUpdateOp>();
        target.addLegalOp<ora::StructDeclOp>();
        // All sload/sstore must be legalized; no dynamic legality allowed.
        DBG("Marked Ora dialect as illegal");
        // Phase 1: keep cf/scf/tensor/arith legal; lower later.
        target.addLegalDialect<mlir::cf::ControlFlowDialect>();
        DBG("Marked cf dialect as legal");
        target.addLegalDialect<mlir::scf::SCFDialect>();
        DBG("Marked scf dialect as legal");
        target.addLegalDialect<mlir::tensor::TensorDialect>();
        DBG("Marked tensor dialect as legal");
        target.addLegalDialect<mlir::memref::MemRefDialect>();
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
                    // Calls are legalized in the later control-flow phase, where
                    // we can lower them with callee-aware result/ABI handling.
                    for (Value operand : callOp.getOperands())
                    {
                        if (euh::isPayloadlessErrorStruct(operand.getType(), callOp))
                            return false;
                    }
                    for (Type resultType : callOp.getResultTypes())
                    {
                        if (euh::isPayloadlessErrorStruct(resultType, callOp))
                            return false;
                    }
                    return true;
                }
                return true;
            });

        target.addIllegalOp<ora::AddOp, ora::AddWrappingOp, ora::SubWrappingOp, ora::MulWrappingOp, ora::ShlWrappingOp, ora::ShrWrappingOp, ora::SubOp, ora::MulOp, ora::DivOp, ora::RemOp, ora::MapGetOp, ora::MapStoreOp, ora::StorageDeriveOp, ora::StorageWordLoadOp, ora::StorageWordStoreOp, ora::StorageRangeEraseOp, ora::CreateOp, ora::DestroyOp, ora::MoveOp>();
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

        // Preserve error decl metadata as module attributes before any conversion
        // erases ora::ErrorDeclOp (sir::ErrorDeclOp lacks SymbolOpInterface).
        {
            SmallVector<NamedAttribute> errEntries;
            SmallVector<NamedAttribute> errSelectorEntries;
            SmallVector<NamedAttribute> errParamCountEntries;
            module.walk([&](ora::ErrorDeclOp decl)
                        {
                auto id = decl->getAttrOfType<mlir::IntegerAttr>("ora.error_id");
                auto sym = decl->getAttrOfType<mlir::StringAttr>("sym_name");
                auto selector = decl->getAttrOfType<mlir::StringAttr>("ora.error_selector");
                auto paramTypes = decl->getAttrOfType<mlir::ArrayAttr>("ora.param_types");
                if (sym && id)
                    errEntries.push_back(NamedAttribute(sym, id));
                if (sym && selector)
                    errSelectorEntries.push_back(NamedAttribute(sym, selector));
                if (sym && paramTypes && !paramTypes.empty())
                    errParamCountEntries.push_back(NamedAttribute(
                        sym,
                        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                               static_cast<int64_t>(paramTypes.size())))); });
            if (!errEntries.empty())
                module->setAttr("sir.error_ids", DictionaryAttr::get(ctx, errEntries));
            if (!errSelectorEntries.empty())
                module->setAttr("sir.error_selectors", DictionaryAttr::get(ctx, errSelectorEntries));
            if (!errParamCountEntries.empty())
                module->setAttr("sir.error_param_counts", DictionaryAttr::get(ctx, errParamCountEntries));
        }

        // Phase 0 only: normalize error_union ops into explicit packing/unpacking.
        {
            RewritePatternSet phase0Patterns(ctx);
            phase0Patterns.add<NormalizeErrorOkOp>(ctx);
            phase0Patterns.add<NormalizeErrorErrOp>(ctx);
            phase0Patterns.add<NormalizeErrorUnionCastOp>(ctx);
            phase0Patterns.add<NormalizeAdtConstructOp>(ctx);
            phase0Patterns.add<NormalizeAdtTagOp>(ctx);
            phase0Patterns.add<NormalizeAdtPayloadOp>(ctx);
            phase0Patterns.add<NormalizeScfYieldOp>(ctx);
            phase0Patterns.add<NormalizeOraYieldOp>(ctx);
            ora::TupleCreateOp::getCanonicalizationPatterns(phase0Patterns, ctx);
            ora::TupleExtractOp::getCanonicalizationPatterns(phase0Patterns, ctx);
            ora::StructInstantiateOp::getCanonicalizationPatterns(phase0Patterns, ctx);
            ora::StructInitOp::getCanonicalizationPatterns(phase0Patterns, ctx);
            ora::StructFieldExtractOp::getCanonicalizationPatterns(phase0Patterns, ctx);
            ora::StructFieldUpdateOp::getCanonicalizationPatterns(phase0Patterns, ctx);
            GreedyRewriteConfig phase0Config;
            phase0Config.setMaxIterations(64);
            if (failed(applyPatternsGreedily(module, std::move(phase0Patterns), phase0Config)))
            {
                module.emitError("[OraToSIR] Phase 0: error-union normalization failed");
                signalPassFailure();
                return;
            }
            if (failed(normalizeResidualAdtExtractOps(module)))
            {
                module.emitError("[OraToSIR] Phase 0: ADT extract normalization failed");
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
        if (failed(applyFullConversionWithDiagnostics(
                module, target, std::move(patterns),
                "[OraToSIR] Phase 1: main conversion failed (illegal ops remain)",
                "Phase1 conversion", "Before Phase1 conversion", "After Phase1 conversion",
                "Phase1 failure state")))
        {
            return signalPassFailure();
        }

        DBG("Conversion completed successfully!");

        // ---------------------------------------------------------------
        // Remaining lowering is intentionally multi-phase:
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
            addArithExtensionCastPatterns(phase2Patterns, typeConverter, ctx);

            ConversionTarget phase2Target(*ctx);
            addOraToSirBaseLegalDialectsWithSir(phase2Target);
            phase2Target.addLegalOp<mlir::UnrealizedConversionCastOp>();
            phase2Target.addLegalOp<ora::ReturnOp>();
            // Defer ora.error.is_error lowering to phase 2b. Some wide error-union
            // forms are normalized there after additional rewrites.
            phase2Target.addLegalOp<ora::ErrorIsErrorOp>();
            // Defer scalar error accessors to phase 2b together with CFG lowering.
            phase2Target.addLegalOp<ora::ErrorUnwrapOp>();
            phase2Target.addLegalOp<ora::ErrorGetErrorOp>();
            phase2Target.addLegalOp<ora::ErrorOkOp>();
            phase2Target.addLegalOp<ora::ErrorErrOp>();
            phase2Target.addLegalOp<ora::IfOp>();
            phase2Target.addIllegalOp<ora::TryStmtOp>();
            addOraRegionControlLegalOps(phase2Target);

            if (failed(applyFullConversionWithDiagnostics(
                    module, phase2Target, std::move(phase2Patterns),
                    "[OraToSIR] Phase 2: error-union lowering failed",
                    "Phase2 conversion", "Before Phase2 conversion", "After Phase2 conversion")))
            {
                return signalPassFailure();
            }
        }

        // Phase 2b: re-run try_stmt/error lowering + scf.if → CFG + ora.conditional_return + returns.
        {
            RewritePatternSet phase2bPatterns(ctx);
            phase2bPatterns.add<ConvertTryStmtOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertErrorOkOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertErrorErrOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertErrorUnwrapOp>(typeConverter, ctx);
            phase2bPatterns.add<ConvertErrorGetErrorOp>(typeConverter, ctx);
            addArithExtensionCastPatterns(phase2bPatterns, typeConverter, ctx);
            phase2bPatterns.add<ConvertScfIfOp>(typeConverter, ctx,
                                                /*lowerReturnsInMergeBlock=*/false, PatternBenefit(10));
            phase2bPatterns.add<ConvertIfOp>(typeConverter, ctx);

            ConversionTarget phase2bTarget(*ctx);
            addOraToSirBaseLegalDialectsWithSir(phase2bTarget);
            phase2bTarget.addLegalOp<mlir::UnrealizedConversionCastOp>();
            phase2bTarget.addIllegalOp<mlir::scf::IfOp>();
            phase2bTarget.addIllegalOp<ora::TryStmtOp>();
            phase2bTarget.addIllegalOp<ora::ErrorOkOp>();
            phase2bTarget.addIllegalOp<ora::ErrorErrOp>();
            phase2bTarget.addLegalOp<ora::ErrorIsErrorOp>();
            phase2bTarget.addLegalOp<ora::ErrorUnwrapOp>();
            phase2bTarget.addLegalOp<ora::ErrorGetErrorOp>();
            phase2bTarget.addIllegalOp<ora::IfOp>();
            // ora.return stays legal — lowered in Phase 3a/3b.
            phase2bTarget.addLegalOp<ora::ReturnOp>();
            addOraRegionControlLegalOps(phase2bTarget);

            if (failed(applyFullConversionWithDiagnostics(
                    module, phase2bTarget, std::move(phase2bPatterns),
                    "[OraToSIR] Phase 2b: scf.if/error-union/return lowering failed",
                    "Phase2b conversion", "Before Phase2b conversion", "After Phase2b conversion")))
            {
                return signalPassFailure();
            }

            // Drop any dead blocks introduced by try_stmt inlining.
            {
                mlir::IRRewriter cleanupRewriter(ctx);
                (void)mlir::eraseUnreachableBlocks(cleanupRewriter, module.getOperation()->getRegions());
            }
        }

        // Enable struct lowering before final return lowering so aggregate
        // returns can convert to ptr carriers during phase 3/3b.
        typeConverter.setEnableStructLowering(true);
        // Enable memref lowering before final return lowering so fixed-array
        // returns can convert to ptr carriers during phase 3/3b.
        typeConverter.setEnableMemRefLowering(true);

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
            module.walk([&](ora::ReturnOp)
                        { hasRemainingReturns = true; });
            if (hasRemainingReturns)
            {
                RewritePatternSet phase3bPatterns(ctx);
                phase3bPatterns.add<ConvertReturnOp>(typeConverter, ctx);
                phase3bPatterns.add<ConvertScfIfOp>(typeConverter, ctx,
                                                    /*lowerReturnsInMergeBlock=*/false, PatternBenefit(10));

                ConversionTarget phase3bTarget(*ctx);
                addOraToSirBaseLegalDialectsWithSir(phase3bTarget);
                phase3bTarget.addLegalOp<mlir::UnrealizedConversionCastOp>();
                phase3bTarget.addIllegalOp<mlir::scf::IfOp>();
                phase3bTarget.addIllegalOp<ora::ReturnOp>();
                phase3bTarget.addLegalOp<ora::IfOp>();
                phase3bTarget.addLegalOp<ora::TryStmtOp>();
                addOraRegionControlLegalOps(phase3bTarget);

                if (failed(applyFullConversionWithDiagnostics(
                        module, phase3bTarget, std::move(phase3bPatterns),
                        "[OraToSIR] Phase 3b: final return lowering failed",
                        "Phase3b conversion", "Before Phase3b conversion")))
                {
                    return signalPassFailure();
                }
            }
            logModuleOps(module, "After Phase3 conversion");
        }

        // Phase 4: lower scf.for, scf.while, memref ops (stack temps) to SIR.
        {
            RewritePatternSet phase3PrePatterns(ctx);
            phase3PrePatterns.add<NormalizeNarrowErrorUnionMemRefLoadOp>(ctx);
            phase3PrePatterns.add<NormalizeNarrowErrorUnionMemRefStoreOp>(ctx);
            if (failed(applyPatternsGreedily(module, std::move(phase3PrePatterns))))
            {
                dumpModuleOnFailure(module, "Phase3 pre-normalization");
                module.emitError("[OraToSIR] Phase 4: error-union memref normalization failed");
                signalPassFailure();
                return;
            }

            RewritePatternSet phase3Patterns(ctx);
            phase3Patterns.add<ConvertScfForOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertScfWhileOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertIfOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertErrorReturnOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertErrorOkOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertErrorErrOp>(typeConverter, ctx);
            phase3Patterns.add<ConvertMemRefAllocOp>(typeConverter, ctx, memRefNamingCache);
            phase3Patterns.add<ConvertMemRefLoadOp>(typeConverter, ctx, memRefNamingCache);
            phase3Patterns.add<ConvertMemRefStoreOp>(typeConverter, ctx, memRefNamingCache);
            phase3Patterns.add<ConvertMemRefDimOp>(typeConverter, ctx);

            ConversionTarget phase3Target(*ctx);
            addLegalDialects<mlir::BuiltinDialect, sir::SIRDialect, mlir::func::FuncDialect,
                             mlir::cf::ControlFlowDialect, mlir::arith::ArithDialect,
                             mlir::scf::SCFDialect, mlir::tensor::TensorDialect>(phase3Target);
            phase3Target.addLegalOp<mlir::UnrealizedConversionCastOp>();
            phase3Target.addIllegalDialect<mlir::memref::MemRefDialect>();
            phase3Target.addIllegalOp<mlir::scf::ForOp>();
            phase3Target.addIllegalOp<mlir::scf::WhileOp>();
            phase3Target.addIllegalOp<ora::ReturnOp>();
            phase3Target.addIllegalOp<ora::IfOp>();
            phase3Target.addIllegalOp<ora::ErrorReturnOp>();
            phase3Target.addIllegalOp<ora::ErrorOkOp>();
            phase3Target.addIllegalOp<ora::ErrorErrOp>();
            phase3Target.addLegalOp<ora::TryStmtOp>();
            addOraRegionControlLegalOps(phase3Target);

            ConversionConfig phase3Config;
            // Avoid ptr<->memref materializations; memref ops must be fully rewritten.
            phase3Config.buildMaterializations = false;
            if (failed(applyFullConversionWithDiagnostics(
                    module, phase3Target, std::move(phase3Patterns),
                    "[OraToSIR] Phase 4: scf.for/memref lowering failed",
                    "Phase3 conversion", "Before Phase3 conversion", "After Phase3 conversion",
                    StringRef(), &phase3Config)))
            {
                return signalPassFailure();
            }
        }

        // Phase 5: lower remaining Ora control flow + structs + cleanup.
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
            phase4Patterns.add<NormalizeErrorIsErrorOp>(ctx, PatternBenefit(2));
            phase4Patterns.add<NormalizeErrorUnwrapOp>(ctx, PatternBenefit(2));
            phase4Patterns.add<NormalizeErrorGetErrorOp>(ctx, PatternBenefit(2));
            phase4Patterns.add<ConvertErrorIsErrorOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertErrorUnwrapOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertErrorGetErrorOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertUnrealizedConversionCastOp>(typeConverter, ctx);
            phase4Patterns.add<StripNormalizedErrorUnionCastOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertCfBrOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertCfCondBrOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertCfAssertOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertLengthOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertByteAtOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertConcatOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertSliceOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertKeccak256Op>(typeConverter, ctx);
            addArithmeticLoweringPatterns(phase4Patterns, typeConverter, ctx);
            phase4Patterns.add<ConvertAddrToI160Op>(typeConverter, ctx);
            phase4Patterns.add<ConvertI160ToAddrOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertTensorInsertOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertTensorExtractOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertTensorDimOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertBaseToRefinementOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertRefinementToBaseOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructInitOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructInstantiateOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertTupleCreateOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertTupleExtractOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertAdtConstructOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructFieldExtractOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructFieldUpdateOp>(typeConverter, ctx);
            phase4Patterns.add<ConvertStructDeclOp>(typeConverter, ctx);
            phase4Patterns.add<StripStructMaterializeOp>(typeConverter, ctx);
            phase4Patterns.add<StripAddressMaterializeOp>(typeConverter, ctx);
            phase4Patterns.add<StripBytesMaterializeOp>(typeConverter, ctx);

            ConversionTarget phase4Target(*ctx);
            addLegalDialects<mlir::BuiltinDialect, sir::SIRDialect, mlir::func::FuncDialect,
                             mlir::scf::SCFDialect>(phase4Target);
            phase4Target.addIllegalDialect<mlir::cf::ControlFlowDialect>();
            phase4Target.addIllegalDialect<mlir::arith::ArithDialect>();
            phase4Target.addIllegalDialect<mlir::tensor::TensorDialect>();
            phase4Target.addIllegalDialect<mlir::memref::MemRefDialect>();
            phase4Target.addLegalOp<mlir::UnrealizedConversionCastOp>();
            phase4Target.addIllegalOp<mlir::func::CallOp>();
            phase4Target.addIllegalOp<ora::ReturnOp>();
            phase4Target.addIllegalOp<ora::ErrorIsErrorOp>();
            phase4Target.addIllegalOp<ora::ErrorUnwrapOp>();
            phase4Target.addIllegalOp<ora::ErrorGetErrorOp>();
            phase4Target.addDynamicallyLegalOp<mlir::func::FuncOp>(
                [&](mlir::func::FuncOp funcOp)
                {
                    auto isSIRSignatureType = [](Type type) {
                        if (llvm::isa<sir::U256Type, sir::PtrType, ora::StructType, ora::TupleType,
                                      mlir::MemRefType, mlir::UnrankedMemRefType>(type))
                            return true;
                        if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
                            return intType.getWidth() <= 256;
                        return false;
                    };
                    auto fnType = funcOp.getFunctionType();
                    for (Type inputType : fnType.getInputs())
                    {
                        if (!isSIRSignatureType(inputType))
                            return false;
                    }
                    for (Type resultType : fnType.getResults())
                    {
                        if (!isSIRSignatureType(resultType))
                            return false;
                    }

                    // Also require all block arguments to be legal, otherwise
                    // SIR text legalizer will reject the function body.
                    for (Block &block : funcOp.getBody())
                    {
                        for (BlockArgument arg : block.getArguments())
                        {
                            if (!isSIRSignatureType(arg.getType()))
                                return false;
                        }
                    }
                    return true;
                });
            phase4Target.addIllegalOp<ora::IfOp>();
            phase4Target.addIllegalOp<ora::YieldOp>();
            phase4Target.addIllegalOp<ora::ContinueOp>();
            phase4Target.addIllegalOp<ora::TryStmtOp>();
            phase4Target.addIllegalOp<ora::SwitchOp>();
            phase4Target.addIllegalOp<ora::AddrToI160Op>();
            phase4Target.addIllegalOp<ora::I160ToAddrOp>();
            phase4Target.addIllegalOp<ora::StructInitOp>();
            phase4Target.addIllegalOp<ora::StructInstantiateOp>();
            phase4Target.addIllegalOp<ora::AdtConstructOp>();
            phase4Target.addIllegalOp<ora::StructFieldExtractOp>();
            phase4Target.addIllegalOp<ora::StructFieldUpdateOp>();
            phase4Target.addIllegalOp<ora::StructDeclOp>();
            phase4Target.addIllegalOp<ora::BaseToRefinementOp>();
            phase4Target.addIllegalOp<ora::RefinementToBaseOp>();
            phase4Target.addIllegalOp<ora::AbiEncodeOp>();
            phase4Target.addIllegalOp<ora::AbiEncodeWithSelectorOp>();
            phase4Target.addIllegalOp<ora::ExternalCallOp>();
            phase4Target.addIllegalOp<ora::AbiDecodeOp>();
            phase4Target.addIllegalOp<ora::Keccak256Op>();
            phase4Target.addLegalDialect<ora::OraDialect>();

            // Debug: report any unrealized casts still present before Phase 4.
            if (failed(applyFullConversionWithDiagnostics(
                    module, phase4Target, std::move(phase4Patterns),
                    "[OraToSIR] Phase 5: final control-flow/struct lowering failed",
                    "Phase4 conversion", "Before Phase4 conversion", "After Phase4 conversion")))
            {
                return signalPassFailure();
            }

            SmallVector<mlir::UnrealizedConversionCastOp, 16> residualCasts;
            module.walk([&](mlir::UnrealizedConversionCastOp op) { residualCasts.push_back(op); });
            llvm::SmallPtrSet<Operation *, 16> erasedResidualCasts;
            for (auto castOp : residualCasts)
            {
                if (erasedResidualCasts.contains(castOp.getOperation()))
                    continue;
                mlir::IRRewriter b(ctx);
                b.setInsertionPoint(castOp);
                auto eraseResidualCast = [&](mlir::UnrealizedConversionCastOp op) {
                    erasedResidualCasts.insert(op.getOperation());
                    b.eraseOp(op);
                };
                auto loc = castOp.getLoc();
                auto u256Ty = sir::U256Type::get(ctx);
                auto isNarrowErr = [&](ora::ErrorUnionType errType) {
                    auto successType = errType.getSuccessType();
                    return llvm::isa<mlir::ora::IntegerType, mlir::IntegerType, mlir::NoneType, mlir::ora::AddressType, mlir::ora::NonZeroAddressType>(successType);
                };
                auto asU256 = [&](Value value) -> Value {
                    if (llvm::isa<sir::U256Type>(value.getType()))
                        return value;
                    if (auto bitcast = value.getDefiningOp<sir::BitcastOp>())
                    {
                        Value input = bitcast.getInput();
                        if (llvm::isa<sir::U256Type>(input.getType()))
                            return input;
                    }
                    return b.create<sir::BitcastOp>(loc, u256Ty, value);
                };
                auto asInteger = [&](Type resultType, Value input) -> Value {
                    if (auto bitcast = input.getDefiningOp<sir::BitcastOp>())
                    {
                        Value original = bitcast.getInput();
                        if (original.getType() == resultType)
                            return original;
                    }
                    return b.create<sir::BitcastOp>(loc, resultType, input);
                };

                if (llvm::all_of(castOp.getResults(), [](Value result) { return result.use_empty(); }))
                {
                    eraseResidualCast(castOp);
                    continue;
                }

                if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1)
                {
                    Value input = castOp.getOperand(0);
                    Type resultType = castOp.getResult(0).getType();

                    if (ora::hasMaterializationKind(castOp, mat_kind::kPtrView) &&
                        llvm::isa<sir::PtrType>(resultType))
                    {
                        if (Value storageRoot = getStorageMemRefViewRootSlot(input))
                        {
                            SmallVector<sir::LoadOp, 4> loads;
                            bool allUsersAreLoads = !castOp.getResult(0).use_empty();
                            for (Operation *user : castOp.getResult(0).getUsers())
                            {
                                auto load = dyn_cast<sir::LoadOp>(user);
                                if (!load || load.getPtr() != castOp.getResult(0))
                                {
                                    allUsersAreLoads = false;
                                    break;
                                }
                                loads.push_back(load);
                            }

                            if (allUsersAreLoads && !loads.empty())
                            {
                                for (sir::LoadOp load : loads)
                                {
                                    b.setInsertionPoint(load);
                                    Value replacement = b.create<sir::SLoadOp>(
                                        load.getLoc(), u256Ty, storageRoot);
                                    load.getResult().replaceAllUsesWith(replacement);
                                    b.eraseOp(load);
                                }
                                eraseResidualCast(castOp);
                                continue;
                            }
                        }
                    }

                    if (input.getType() == resultType)
                    {
                        castOp.getResult(0).replaceAllUsesWith(input);
                        eraseResidualCast(castOp);
                        continue;
                    }

                    if (llvm::isa<mlir::IntegerType>(resultType) && llvm::isa<sir::U256Type>(input.getType()))
                    {
                        SmallVector<mlir::UnrealizedConversionCastOp, 4> backCasts;
                        bool allUsersAreBackCasts = !castOp.getResult(0).use_empty();
                        for (Operation *user : castOp.getResult(0).getUsers())
                        {
                            auto backCast = dyn_cast<mlir::UnrealizedConversionCastOp>(user);
                            if (!backCast || backCast.getNumOperands() != 1 || backCast.getNumResults() != 1 ||
                                backCast.getOperand(0) != castOp.getResult(0) ||
                                !llvm::isa<sir::U256Type>(backCast.getResult(0).getType()))
                            {
                                allUsersAreBackCasts = false;
                                break;
                            }
                            backCasts.push_back(backCast);
                        }
                        if (allUsersAreBackCasts && !backCasts.empty())
                        {
                            for (auto backCast : backCasts)
                            {
                                backCast.getResult(0).replaceAllUsesWith(input);
                                eraseResidualCast(backCast);
                            }
                            eraseResidualCast(castOp);
                            continue;
                        }
                    }

                    if (llvm::isa<sir::U256Type>(resultType) && llvm::isa<mlir::IntegerType>(input.getType()))
                    {
                        Value repl = asU256(input);
                        castOp.getResult(0).replaceAllUsesWith(repl);
                        eraseResidualCast(castOp);
                        continue;
                    }

                    if (llvm::isa<mlir::IntegerType>(resultType) && llvm::isa<sir::U256Type>(input.getType()))
                    {
                        Value repl = asInteger(resultType, input);
                        castOp.getResult(0).replaceAllUsesWith(repl);
                        eraseResidualCast(castOp);
                        continue;
                    }

                    if (llvm::isa<sir::PtrType>(resultType) &&
                        llvm::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(input.getType()))
                    {
                        Value repl = b.create<sir::BitcastOp>(loc, resultType, input);
                        castOp.getResult(0).replaceAllUsesWith(repl);
                        eraseResidualCast(castOp);
                        continue;
                    }

                    if (llvm::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(resultType) &&
                        llvm::isa<sir::PtrType>(input.getType()))
                    {
                        castOp.getResult(0).replaceAllUsesWith(input);
                        eraseResidualCast(castOp);
                        continue;
                    }

                    if (llvm::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(resultType) &&
                        llvm::isa<sir::U256Type>(input.getType()))
                    {
                        Value repl = b.create<sir::BitcastOp>(
                            loc, sir::PtrType::get(ctx, /*addrSpace*/ 1), input);
                        castOp.getResult(0).replaceAllUsesWith(repl);
                        eraseResidualCast(castOp);
                        continue;
                    }

                    if (auto errType = llvm::dyn_cast<ora::ErrorUnionType>(resultType))
                    {
                        if (isNarrowErr(errType) && llvm::isa<mlir::IntegerType>(input.getType()))
                        {
                            Value repl = asU256(input);
                            castOp.getResult(0).replaceAllUsesWith(repl);
                            eraseResidualCast(castOp);
                            continue;
                        }
                    }
                }

                if (castOp.getNumOperands() == 2 && castOp.getNumResults() == 1)
                {
                    Type resultType = castOp.getResult(0).getType();
                    if (auto errType = llvm::dyn_cast<ora::ErrorUnionType>(resultType))
                    {
                        if (isNarrowErr(errType))
                        {
                            Value tag = asU256(castOp.getOperand(0));
                            Value payload = asU256(castOp.getOperand(1));
                            Value one = constU256(b, loc, 1);
                            Value shifted = b.create<sir::ShlOp>(loc, u256Ty, one, payload);
                            Value packed = b.create<sir::OrOp>(loc, u256Ty, shifted, tag);
                            castOp.getResult(0).replaceAllUsesWith(packed);
                            eraseResidualCast(castOp);
                            continue;
                        }
                    }
                }
            }

            RewritePatternSet castCleanupPatterns(ctx);
            castCleanupPatterns.add<NormalizeErrorIsErrorOp>(ctx);
            castCleanupPatterns.add<NormalizeErrorUnwrapOp>(ctx);
            castCleanupPatterns.add<NormalizeErrorGetErrorOp>(ctx);
            castCleanupPatterns.add<ConvertUnrealizedConversionCastOp>(typeConverter, ctx);
            castCleanupPatterns.add<StripNormalizedErrorUnionCastOp>(typeConverter, ctx);
            castCleanupPatterns.add<ConvertArithShlIOp>(typeConverter, ctx);
            castCleanupPatterns.add<ConvertArithShrUIOp>(typeConverter, ctx);
            castCleanupPatterns.add<ConvertArithShrSIOp>(typeConverter, ctx);
            castCleanupPatterns.add<ConvertArithIndexCastUIOp>(typeConverter, ctx);
            castCleanupPatterns.add<ConvertArithIndexCastOp>(typeConverter, ctx);

            ConversionTarget castCleanupTarget(*ctx);
            addLegalDialects<mlir::BuiltinDialect, sir::SIRDialect, mlir::func::FuncDialect,
                             mlir::scf::SCFDialect, ora::OraDialect>(castCleanupTarget);
            castCleanupTarget.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
                [](mlir::UnrealizedConversionCastOp op)
                {
                    return ora::hasMaterializationKind(op, mat_kind::kNormalizedErrorUnion) ||
                           ora::hasMaterializationKind(op, mat_kind::kNormalizedAdt) ||
                           ora::hasMaterializationKind(op, mat_kind::kAdtHandleView) ||
                           ora::hasMaterializationKind(op, mat_kind::kNoneForward) ||
                           ora::hasMaterializationKind(op, mat_kind::kPtrView) ||
                           ora::hasMaterializationKind(op, mat_kind::kAddressForward) ||
                           ora::hasMaterializationKind(op, mat_kind::kWideErrorUnionJoin) ||
                           ora::hasMaterializationKind(op, mat_kind::kWideErrorUnionSplit);
                });

            if (failed(applyFullConversionWithDiagnostics(
                    module, castCleanupTarget, std::move(castCleanupPatterns),
                    "[OraToSIR] Phase 5: unrealized cast cleanup failed",
                    "Phase4 cast cleanup")))
            {
                return signalPassFailure();
            }

            {
                RewritePatternSet lateTuplePatterns(ctx);
                lateTuplePatterns.add<LateLowerTupleExtractOp>(&typeConverter, ctx);
                if (failed(applyPatternsGreedily(module, std::move(lateTuplePatterns))))
                {
                    module.emitError("[OraToSIR] Phase 5: late tuple extract lowering failed");
                    signalPassFailure();
                    return;
                }
            }

            // Cleanup: collapse only trivially-safe surviving unrealized casts.
            //   - same-type 1:1 cast → forward operand and erase.
            //   - dead normalized_error_union / normalized_adt with all u256
            //     operands → users have been rewired earlier; just erase the
            //     unused carrier.
            // Any other 1:1 cast must be handled by a kind-specific conversion
            // pattern or fail the later residual-cast scans. Forwarding arbitrary
            // operands here would silently bless an unmodeled representation
            // change.
            auto isSameTypeIdentityCast = [](mlir::UnrealizedConversionCastOp op) {
                return op.getNumOperands() == 1 &&
                       op.getNumResults() == 1 &&
                       op.getOperand(0).getType() == op.getResult(0).getType();
            };
            auto isTypedU256CarrierView = [](Type type) {
                return llvm::isa<mlir::IntegerType,
                                 ora::IntegerType,
                                 ora::AddressType,
                                 ora::NonZeroAddressType,
                                 ora::MinValueType,
                                 ora::MaxValueType,
                                 ora::InRangeType,
                                 ora::ScaledType,
                                 ora::ExactType,
                                 ora::ErrorUnionType>(type);
            };
            auto isDeadNormalizedU256Pack = [](mlir::UnrealizedConversionCastOp op) {
                if (!ora::hasMaterializationKind(op, mat_kind::kNormalizedErrorUnion) &&
                    !ora::hasMaterializationKind(op, mat_kind::kNormalizedAdt))
                    return false;
                if (op.getNumResults() != 1)
                    return false;
                if (!op.getResult(0).use_empty())
                    return false;
                return llvm::all_of(op.getOperands(),
                                    [](Value v) { return llvm::isa<sir::U256Type>(v.getType()); });
            };
            SmallVector<mlir::UnrealizedConversionCastOp, 32> castsToDrop;
            module.walk([&](mlir::UnrealizedConversionCastOp op) {
                if (isSameTypeIdentityCast(op) || isDeadNormalizedU256Pack(op))
                    castsToDrop.push_back(op);
            });
            for (auto castOp : castsToDrop)
            {
                if (isSameTypeIdentityCast(castOp))
                    castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
                castOp.erase();
            }

            // Typed carrier views sometimes survive until their only consumers
            // bitcast them straight back to u256. This is a representational
            // no-op, but only in that exact shape. If the typed value feeds any
            // real operation, leave the cast in place so the residual scans fail
            // instead of silently forwarding it.
            SmallVector<mlir::UnrealizedConversionCastOp, 16> carrierViewsToDrop;
            module.walk([&](mlir::UnrealizedConversionCastOp op) {
                if (op.getNumOperands() != 1 || op.getNumResults() != 1)
                    return;
                if (!llvm::isa<sir::U256Type>(op.getOperand(0).getType()) ||
                    !isTypedU256CarrierView(op.getResult(0).getType()))
                    return;
                bool allUsersAreU256Bitcasts = !op.getResult(0).use_empty();
                for (Operation *user : op.getResult(0).getUsers())
                {
                    auto bitcast = dyn_cast<sir::BitcastOp>(user);
                    if (!bitcast || bitcast.getInput() != op.getResult(0) ||
                        !llvm::isa<sir::U256Type>(bitcast.getResult().getType()))
                    {
                        allUsersAreU256Bitcasts = false;
                        break;
                    }
                }
                if (allUsersAreU256Bitcasts)
                    carrierViewsToDrop.push_back(op);
            });
            for (auto castOp : carrierViewsToDrop)
            {
                SmallVector<sir::BitcastOp, 4> bitcasts;
                for (Operation *user : castOp.getResult(0).getUsers())
                    bitcasts.push_back(cast<sir::BitcastOp>(user));
                for (auto bitcast : bitcasts)
                {
                    bitcast.getResult().replaceAllUsesWith(castOp.getOperand(0));
                    bitcast.erase();
                }
                castOp.erase();
            }

            // Cleanup: strip residual refinement bridges and refinement-typed
            // sir.bitcast results created by source materializations.
            auto isOraRefinement = [](Type t) {
                return llvm::isa<ora::MinValueType, ora::MaxValueType, ora::InRangeType,
                                 ora::ScaledType, ora::ExactType, ora::NonZeroAddressType>(t);
            };
            SmallVector<Operation *, 16> refinementOps;
            module.walk([&](Operation *op) {
                if (isa<ora::BaseToRefinementOp, ora::RefinementToBaseOp>(op))
                    refinementOps.push_back(op);
                else if (auto bc = dyn_cast<sir::BitcastOp>(op); bc && isOraRefinement(bc.getResult().getType()))
                    refinementOps.push_back(op);
            });
            for (auto *op : refinementOps)
            {
                op->getResult(0).replaceAllUsesWith(op->getOperand(0));
                op->erase();
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
                module.walk([&](sir::ICallOp op)
                            {
                    if (auto callee = op.getCalleeAttr())
                    {
                        StringRef name = callee.getValue();
                        if (errorIds.count(name))
                            errorIcalls.push_back(op);
                    } });

                for (auto op : errorIcalls)
                {
                    int64_t id = errorIds[op.getCalleeAttr().getValue()];
                    OpBuilder b(op);
                    auto u256 = sir::U256Type::get(ctx);
                    Value idConst = constU256(b, op.getLoc(), static_cast<uint64_t>(id));
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

            // Late tuple/aggregate rewrites can expose precise bridge casts.
            // The explicit final residual cleanup below handles those, then
            // verifyNoUnexpectedUnrealizedCasts enforces the fail-closed barrier.

            // Normalize malformed blocks before any final printing/validation so we
            // fail cleanly instead of reaching MLIR internals with invalid CFG.
            bool hadMalformedTerminatorBlocks = false;
            module.walk([&](mlir::func::FuncOp funcOp)
                        { hadMalformedTerminatorBlocks = normalizeFuncTerminators(funcOp) || hadMalformedTerminatorBlocks; });
            if (hadMalformedTerminatorBlocks)
            {
                module.emitError("[OraToSIR] malformed CFG: missing terminator or trailing ops after terminator");
                signalPassFailure();
                return;
            }

            // Avoid in-pass full module dump here: if IR is structurally damaged,
            // pretty-print traversal itself can crash before we report a clean
            // diagnostic. The CLI still prints SIR MLIR after successful conversion.
            if (mlir::ora::isDebugEnabled())
            {
                llvm::errs() << "[OraToSIR] Post-Phase4: internal module dump skipped\n";
                llvm::errs().flush();
            }

        }

        {
            bool hasFinalErrorOps = false;
            module.walk([&](Operation *op)
                        {
                            if (isa<ora::ErrorIsErrorOp, ora::ErrorUnwrapOp>(op))
                                hasFinalErrorOps = true;
                        });
            if (hasFinalErrorOps)
            {
                RewritePatternSet finalErrorCleanup(ctx);
                finalErrorCleanup.add<NormalizeErrorIsErrorOp>(ctx);
                finalErrorCleanup.add<NormalizeErrorUnwrapOp>(ctx);
                if (failed(applyPatternsGreedily(module, std::move(finalErrorCleanup))))
                {
                    module.emitError("[OraToSIR] final cleanup: residual error-union normalization failed");
                    signalPassFailure();
                    return;
                }
            }

            SmallVector<mlir::UnrealizedConversionCastOp, 8> deadNormalizedFinalCasts;
            module.walk([&](mlir::UnrealizedConversionCastOp op)
                        {
                            if (!ora::hasMaterializationKind(op, mat_kind::kNormalizedErrorUnion) &&
                                !ora::hasMaterializationKind(op, mat_kind::kNormalizedAdt))
                                return;
                            if (llvm::all_of(op.getResults(), [](Value result) { return result.use_empty(); }))
                                deadNormalizedFinalCasts.push_back(op);
                        });
            for (auto op : deadNormalizedFinalCasts)
                op.erase();

            RewritePatternSet finalResidualPatterns(ctx);
            finalResidualPatterns.add<ConvertUnrealizedConversionCastOp>(typeConverter, ctx);
            finalResidualPatterns.add<ConvertArithConstantOp>(typeConverter, ctx);
            finalResidualPatterns.add<ConvertArithAndIOp>(typeConverter, ctx);
            finalResidualPatterns.add<ConvertArithCmpIOp>(typeConverter, ctx);
            finalResidualPatterns.add<ConvertAddrToI160Op>(typeConverter, ctx);
            finalResidualPatterns.add<ConvertI160ToAddrOp>(typeConverter, ctx);
            finalResidualPatterns.add<ConvertTupleCreateOp>(typeConverter, ctx);
            finalResidualPatterns.add<ConvertTupleExtractOp>(typeConverter, ctx);

            ConversionTarget finalResidualTarget(*ctx);
            addLegalDialects<mlir::BuiltinDialect, sir::SIRDialect, mlir::func::FuncDialect,
                             mlir::cf::ControlFlowDialect>(finalResidualTarget);
            finalResidualTarget.addIllegalDialect<mlir::arith::ArithDialect>();
            finalResidualTarget.addIllegalOp<mlir::UnrealizedConversionCastOp>();
            finalResidualTarget.addIllegalOp<ora::AddrToI160Op>();
            finalResidualTarget.addIllegalOp<ora::I160ToAddrOp>();
            finalResidualTarget.addIllegalOp<ora::TupleCreateOp>();
            finalResidualTarget.addIllegalOp<ora::TupleExtractOp>();
            finalResidualTarget.addLegalDialect<ora::OraDialect>();

            if (failed(applyFullConversionWithDiagnostics(
                    module, finalResidualTarget, std::move(finalResidualPatterns),
                    "[OraToSIR] final cleanup: residual arith/cast lowering failed")))
            {
                return signalPassFailure();
            }

            // Final tuple-create conversion can leave ptr_view/source
            // materialization casts after their aggregate user is rewritten.
            // Collapse live ptr views to their SIR pointer carrier, then sweep
            // dead materializations before the post-conversion illegal-op scan.
            SmallVector<mlir::UnrealizedConversionCastOp, 8> finalPtrViews;
            module.walk([&](mlir::UnrealizedConversionCastOp op)
                        {
                            if (ora::hasMaterializationKind(op, mat_kind::kPtrView) &&
                                op.getNumOperands() == 1 &&
                                op.getNumResults() == 1 &&
                                llvm::isa<sir::PtrType>(op.getOperand(0).getType()))
                            {
                                finalPtrViews.push_back(op);
                            }
                        });
            for (auto op : finalPtrViews)
            {
                op.getResult(0).replaceAllUsesWith(op.getOperand(0));
                op.erase();
            }

            SmallVector<mlir::UnrealizedConversionCastOp, 8> deadFinalCasts;
            module.walk([&](mlir::UnrealizedConversionCastOp op)
                        {
                            if (llvm::all_of(op.getResults(), [](Value result) { return result.use_empty(); }))
                                deadFinalCasts.push_back(op);
                        });
            for (auto op : deadFinalCasts)
                op.erase();

            if (failed(verifyNoUnexpectedUnrealizedCasts(module)))
            {
                signalPassFailure();
                return;
            }
        }

        bool hasSurvivingMaterializationCasts = false;
        module.walk([&](mlir::UnrealizedConversionCastOp)
                    { hasSurvivingMaterializationCasts = true; });
        if (!module->hasAttr(kPhase0SkipManualBitcastFoldAttr) &&
            !hasSurvivingMaterializationCasts)
        {
            foldExplicitIntegerCarrierRoundTripBitcasts(module);
        }
        else if (mlir::ora::isDebugEnabled() && hasSurvivingMaterializationCasts)
        {
            llvm::errs() << "[OraToSIR] Phase0: skipped manual sir.bitcast fold with live materialization casts\n";
            llvm::errs().flush();
        }
        else if (mlir::ora::isDebugEnabled())
        {
            llvm::errs() << "[OraToSIR] Phase0: skipped manual sir.bitcast fold\n";
            llvm::errs().flush();
        }

        // Guard: fail if any lowering-phase dialect ops remain after all lowering phases.
        if (mlir::ora::isDebugEnabled())
        {
            llvm::errs() << "[OraToSIR] Post-Phase4: illegal-op scan start\n";
            llvm::errs().flush();
        }
        bool illegalFound = false;
        module.walk([&](Operation *op)
                    {
            if (op->getDialect())
            {
                StringRef ns = op->getDialect()->getNamespace();
                if (ns == "ora")
                {
                    llvm::errs() << "[OraToSIR] ERROR: Residual Ora op remains: " << op->getName()
                                 << " at " << op->getLoc() << "\n";
                    illegalFound = true;
                    return;
                }
                if (ns == "cf" || ns == "scf" || ns == "tensor" || ns == "arith" || ns == "memref")
                {
                    llvm::errs() << "[OraToSIR] ERROR: Residual lowering dialect op remains: "
                                 << op->getName() << " at " << op->getLoc() << "\n";
                    illegalFound = true;
                    return;
                }
                if (ns == "func" && op->getName().getStringRef() != "func.func")
                {
                    llvm::errs() << "[OraToSIR] ERROR: Residual func dialect op remains: "
                                 << op->getName() << " at " << op->getLoc() << "\n";
                    illegalFound = true;
                    return;
                }
                if (ns == "builtin" && op->getName().getStringRef() != "builtin.module")
                {
                    llvm::errs() << "[OraToSIR] ERROR: Residual builtin op remains: "
                                 << op->getName() << " at " << op->getLoc() << "\n";
                    illegalFound = true;
                }
            } });
        if (illegalFound)
        {
            module.emitError("[OraToSIR] post-conversion: residual lowering ops remain after all phases");
            signalPassFailure();
            return;
        }
        if (mlir::ora::isDebugEnabled())
        {
            llvm::errs() << "[OraToSIR] Post-Phase4: illegal-op scan done\n";
            llvm::errs().flush();
        }

        if (mlir::ora::isDebugEnabled())
        {
            llvm::errs() << "[OraToSIR] Post-Phase4: terminator scan start\n";
            llvm::errs().flush();
        }
        bool missingTerminator = false;
        module.walk([&](mlir::func::FuncOp funcOp)
                    {
            for (Block &block : funcOp.getBody())
            {
                if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>())
                {
                    llvm::errs() << "[OraToSIR] ERROR: Missing terminator in function "
                                 << funcOp.getName() << " at " << funcOp.getLoc() << "\n";
                    missingTerminator = true;
                }
            } });
        if (missingTerminator)
        {
            module.emitError("[OraToSIR] post-conversion: blocks missing terminators");
            signalPassFailure();
            return;
        }
        if (mlir::ora::isDebugEnabled())
        {
            llvm::errs() << "[OraToSIR] Post-Phase4: terminator scan done\n";
            llvm::errs().flush();
        }

        // NOTE: broad final greedy peephole batches are intentionally disabled
        // here; they caused crashes in converted loop CFGs. Keep local cleanup
        // narrow, and move generic folds to dialect canonicalization.

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

                if (failed(canonicalizeSIRConstantWordOps(module)))
                {
                    module.emitError("[SIROptimizationPass] SIR constant word canonicalization failed");
                    signalPassFailure();
                    return;
                }
            }

            StringRef getArgument() const override { return "sir-optimize"; }
            StringRef getDescription() const override { return "Optimize SIR operations"; }
        };

        std::unique_ptr<Pass> createSIROptimizationPass()
        {
            return std::make_unique<SIROptimizationPass>();
        }

        static void addLocationPreservingCanonicalizer(OpPassManager &funcPM)
        {
            GreedyRewriteConfig config;
            config.enableConstantCSE(false);
            funcPM.addPass(mlir::createCanonicalizerPass(config));
        }

        namespace
        {
            constexpr llvm::StringLiteral kOraVisibilityAttr = "ora.visibility";
            constexpr llvm::StringLiteral kOraInitAttr = "ora.init";
            constexpr llvm::StringLiteral kOraSymbolRootAttr = "ora.symbol_root";
            constexpr llvm::StringLiteral kOraDebugRootAttr = "ora.debug_root";
            constexpr llvm::StringLiteral kOraSymbolDCEBeforeAttr = "ora.symbol_dce.before_functions";
            constexpr llvm::StringLiteral kOraSymbolDCETempVisibilityAttr = "ora.symbol_dce.temp_visibility";

            static uint64_t countNestedFunctionOps(ModuleOp module)
            {
                uint64_t count = 0;
                module.walk([&](mlir::func::FuncOp) {
                    ++count;
                });
                return count;
            }

            static bool boolLikeAttrIsTrue(Operation *op, llvm::StringRef name)
            {
                if (auto attr = op->getAttrOfType<BoolAttr>(name))
                    return attr.getValue();
                return op->getAttrOfType<UnitAttr>(name) != nullptr;
            }

            static bool isOraSymbolRoot(func::FuncOp func)
            {
                Operation *op = func.getOperation();
                if (boolLikeAttrIsTrue(op, kOraInitAttr) ||
                    boolLikeAttrIsTrue(op, kOraSymbolRootAttr) ||
                    boolLikeAttrIsTrue(op, kOraDebugRootAttr))
                    return true;

                // Be conservative for handwritten/debug MLIR. Only functions
                // Ora explicitly marks private are eligible for SymbolDCE.
                auto visibility = op->getAttrOfType<StringAttr>(kOraVisibilityAttr);
                if (!visibility)
                    return true;
                return visibility.getValue() != "private";
            }
        } // namespace

        template <typename DerivedT>
        class FunctionPipelineModulePass : public PassWrapper<DerivedT, OperationPass<ModuleOp>>
        {
        protected:
            static uint64_t countNestedFunctions(ModuleOp module)
            {
                return countNestedFunctionOps(module);
            }

            LogicalResult runNestedFunctionPipeline(
                ModuleOp module,
                StringRef errorMessage,
                llvm::function_ref<void(OpPassManager &)> configure)
            {
                LogicalResult result = success();
                module.walk([&](mlir::func::FuncOp funcOp)
                            {
                    if (failed(result))
                        return;

                    OpPassManager funcPM("func.func");
                    configure(funcPM);

                    if (failed(this->runPipeline(funcPM, funcOp)))
                    {
                        funcOp.emitError(errorMessage);
                        result = failure();
                    } });
                return result;
            }
        };

        class OraSymbolVisibilityPass : public PassWrapper<OraSymbolVisibilityPass, OperationPass<ModuleOp>>
        {
        public:
            void runOnOperation() override
            {
                ModuleOp module = getOperation();
                MLIRContext *context = module.getContext();
                module->setAttr(
                    kOraSymbolDCEBeforeAttr,
                    IntegerAttr::get(mlir::IntegerType::get(context, 64), countNestedFunctionOps(module)));

                module.walk([&](func::FuncOp func)
                            {
                    Operation *op = func.getOperation();
                    if (op->hasAttr(SymbolTable::getVisibilityAttrName()))
                        return;

                    SymbolTable::setSymbolVisibility(
                        op,
                        isOraSymbolRoot(func)
                            ? SymbolTable::Visibility::Public
                            : SymbolTable::Visibility::Private);
                    op->setAttr(kOraSymbolDCETempVisibilityAttr, UnitAttr::get(context)); });
            }

            StringRef getArgument() const override { return "ora-symbol-visibility"; }
            StringRef getDescription() const override { return "Map Ora function/root metadata to temporary MLIR symbol visibility"; }
        };

        std::unique_ptr<Pass> createOraSymbolVisibilityPass()
        {
            return std::make_unique<OraSymbolVisibilityPass>();
        }

        class OraSymbolDCECleanupPass : public PassWrapper<OraSymbolDCECleanupPass, OperationPass<ModuleOp>>
        {
        public:
            void runOnOperation() override
            {
                ModuleOp module = getOperation();
                uint64_t before = 0;
                if (auto attr = module->getAttrOfType<IntegerAttr>(kOraSymbolDCEBeforeAttr))
                    before = attr.getValue().getZExtValue();

                const uint64_t after = countNestedFunctionOps(module);
                if (before > after)
                {
                    const uint64_t removed = before - after;
                    symbolsRemoved += removed;
                    recordOraMlirPassStatistic(OraMlirPassStatistic::OraSymbolsDCEd, removed);
                }

                module->removeAttr(kOraSymbolDCEBeforeAttr);
                module.walk([&](func::FuncOp func)
                            {
                    Operation *op = func.getOperation();
                    if (!op->hasAttr(kOraSymbolDCETempVisibilityAttr))
                        return;
                    op->removeAttr(SymbolTable::getVisibilityAttrName());
                    op->removeAttr(kOraSymbolDCETempVisibilityAttr); });
            }

            StringRef getArgument() const override { return "ora-symbol-dce-cleanup"; }
            StringRef getDescription() const override { return "Record framework SymbolDCE results and remove temporary Ora visibility metadata"; }

        private:
            Pass::Statistic symbolsRemoved{this, "symbols-removed", "Ora functions removed by framework SymbolDCE"};
        };

        std::unique_ptr<Pass> createOraSymbolDCECleanupPass()
        {
            return std::make_unique<OraSymbolDCECleanupPass>();
        }

        // Default post-conversion framework hygiene. This lets MLIR
        // canonicalization and DCE exercise SIR dialect hooks without
        // reintroducing the old broad bespoke peephole batch.
        class SIRFrameworkCanonicalizerPass : public FunctionPipelineModulePass<SIRFrameworkCanonicalizerPass>
        {
        public:
            void runOnOperation() override
            {
                const uint64_t functionCount = countNestedFunctions(getOperation());
                functionsProcessed += functionCount;
                recordOraMlirPassStatistic(OraMlirPassStatistic::SirFrameworkFunctionsProcessed, functionCount);
                if (failed(runNestedFunctionPipeline(
                        getOperation(),
                        "[SIRFrameworkCanonicalizer] canonicalization failed",
                        [](OpPassManager &funcPM)
                        {
                            addLocationPreservingCanonicalizer(funcPM);
                            funcPM.addPass(mlir::createRemoveDeadValuesPass());
                        })))
                    signalPassFailure();
            }

        private:
            Pass::Statistic functionsProcessed{this, "functions-processed", "Nested functions processed by the SIR framework canonicalizer"};
        };

        std::unique_ptr<Pass> createSIRFrameworkCanonicalizerPass()
        {
            return std::make_unique<SIRFrameworkCanonicalizerPass>();
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

                        // Look up the function from the call site so nested contract-local
                        // function symbols are resolved as well as module-level helpers.
                        auto funcOp = SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(callOp.getOperation(), symbolRef);
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
                            ++callsInlined;
                            recordOraMlirPassStatistic(OraMlirPassStatistic::OraCallsInlined);
                            DBG("    Successfully inlined: " << funcOp.getName());
                        }
                        else
                        {
                            DBG("    Failed to inline: " << funcOp.getName());
                        } });

                }

                bool hasFailedSourceInline = false;
                module.walk([&](mlir::func::CallOp callOp)
                            {
                    auto callee = callOp.getCallableForCallee();
                    if (!callee)
                        return;

                    auto symbolRef = llvm::dyn_cast<SymbolRefAttr>(callee);
                    if (!symbolRef)
                        return;

                    auto funcOp = SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(callOp.getOperation(), symbolRef);
                    if (!funcOp)
                        return;

                    auto sourceInlineAttr = funcOp->getAttrOfType<BoolAttr>("ora.source_inline");
                    if (!sourceInlineAttr || !sourceInlineAttr.getValue())
                        return;

                    callOp.emitError("failed to inline function marked 'inline'")
                        << ": unsupported source-inline shape in '" << funcOp.getName()
                        << "' (" << describeUnsupportedInlineShape(funcOp) << ")";
                    ++sourceInlineFailures;
                    recordOraMlirPassStatistic(OraMlirPassStatistic::OraSourceInlineFailures);
                    hasFailedSourceInline = true; });

                if (hasFailedSourceInline)
                    signalPassFailure();

                DBG("Ora inlining pass completed");
            }

        private:
            static bool opContainsNestedReturn(Operation &op)
            {
                bool foundUnsupported = false;
                for (Region &region : op.getRegions())
                {
                    region.walk([&](Operation *nestedOp)
                                {
                            if (isa<ora::ReturnOp, mlir::func::ReturnOp>(nestedOp))
                            {
                                foundUnsupported = true;
                                return WalkResult::interrupt();
                            }
                            return WalkResult::advance(); });
                    if (foundUnsupported)
                        break;
                }
                return foundUnsupported;
            }

            static bool containsNestedReturn(mlir::func::FuncOp funcOp)
            {
                auto &funcBody = funcOp.getBody();
                if (funcBody.empty())
                    return false;
                Block *entryBlock = &funcBody.front();
                for (Operation &op : entryBlock->getOperations())
                {
                    if (opContainsNestedReturn(op))
                        return true;
                }
                return false;
            }

            // Inline a function call by cloning the function body
            // NOTE: only handles single-block functions. Multi-block inlining
            // requires MLIR's InlinerInterface (not yet wired up).
            static bool returnOperands(Operation &op, SmallVectorImpl<Value> &operands)
            {
                if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op))
                {
                    operands.append(returnOp.getOperands().begin(), returnOp.getOperands().end());
                    return true;
                }
                if (auto returnOp = dyn_cast<ora::ReturnOp>(op))
                {
                    operands.append(returnOp.getOperands().begin(), returnOp.getOperands().end());
                    return true;
                }
                return false;
            }

            static bool mapReturnOperands(
                ValueRange operands,
                IRMapping &mapping,
                SmallVectorImpl<Value> &mappedOperands)
            {
                mappedOperands.clear();
                for (Value operand : operands)
                    mappedOperands.push_back(mapping.lookupOrDefault(operand));
                return true;
            }

            static bool replaceCallResultsFromReturnOperands(
                mlir::func::CallOp callOp,
                ValueRange operands,
                IRMapping &mapping)
            {
                SmallVector<Value> returnValues;
                for (auto operand : operands)
                    returnValues.push_back(mapping.lookupOrDefault(operand));
                if (returnValues.size() != callOp.getNumResults())
                    return false;
                for (unsigned i = 0; i < returnValues.size(); ++i)
                    callOp.getResult(i).replaceAllUsesWith(returnValues[i]);
                return true;
            }

            static bool isEmptyYieldOnlyElseRegion(ora::IfOp ifOp)
            {
                Region &elseRegion = ifOp.getElseRegion();
                if (!elseRegion.hasOneBlock())
                    return false;
                Block &elseBlock = elseRegion.front();
                if (elseBlock.getOperations().size() != 1)
                    return false;
                auto yieldOp = dyn_cast<ora::YieldOp>(elseBlock.getTerminator());
                return yieldOp && yieldOp.getNumOperands() == 0;
            }

            static bool cloneOpsUntilReturn(
                Operation *first,
                Operation *stopBefore,
                OpBuilder &builder,
                IRMapping &mapping,
                TypeRange resultTypes,
                SmallVectorImpl<Value> &mappedReturnOperands)
            {
                mappedReturnOperands.clear();
                for (Operation *op = first; op && op != stopBefore; op = op->getNextNode())
                {
                    SmallVector<Value> rawReturnOperands;
                    if (returnOperands(*op, rawReturnOperands))
                        return mapReturnOperands(rawReturnOperands, mapping, mappedReturnOperands);

                    if (op->hasTrait<mlir::OpTrait::IsTerminator>())
                        return false;

                    if (auto ifOp = dyn_cast<ora::IfOp>(op))
                    {
                        if (!isEmptyYieldOnlyElseRegion(ifOp))
                            return false;
                        if (!ifOp.getThenRegion().hasOneBlock())
                            return false;

                        Value mappedCondition = mapping.lookupOrDefault(ifOp.getCondition());
                        auto scfIf = builder.create<mlir::scf::IfOp>(
                            ifOp.getLoc(),
                            resultTypes,
                            mappedCondition,
                            /*withElseRegion=*/true);

                        IRMapping thenMapping = mapping;
                        OpBuilder thenBuilder = scfIf.getThenBodyBuilder();
                        SmallVector<Value> thenReturnOperands;
                        Block &thenBlock = ifOp.getThenRegion().front();
                        Operation *thenFirst = thenBlock.empty() ? nullptr : &thenBlock.front();
                        if (!cloneOpsUntilReturn(
                                thenFirst,
                                nullptr,
                                thenBuilder,
                                thenMapping,
                                resultTypes,
                                thenReturnOperands))
                        {
                            scfIf.erase();
                            return false;
                        }
                        if (thenReturnOperands.size() != resultTypes.size())
                        {
                            scfIf.erase();
                            return false;
                        }
                        if (!resultTypes.empty())
                            thenBuilder.create<mlir::scf::YieldOp>(ifOp.getLoc(), thenReturnOperands);

                        IRMapping elseMapping = mapping;
                        OpBuilder elseBuilder = scfIf.getElseBodyBuilder();
                        SmallVector<Value> elseReturnOperands;
                        if (!cloneOpsUntilReturn(
                                ifOp->getNextNode(),
                                stopBefore,
                                elseBuilder,
                                elseMapping,
                                resultTypes,
                                elseReturnOperands))
                        {
                            scfIf.erase();
                            return false;
                        }
                        if (elseReturnOperands.size() != resultTypes.size())
                        {
                            scfIf.erase();
                            return false;
                        }
                        if (!resultTypes.empty())
                            elseBuilder.create<mlir::scf::YieldOp>(ifOp.getLoc(), elseReturnOperands);

                        mappedReturnOperands.append(scfIf.getResults().begin(), scfIf.getResults().end());
                        return true;
                    }

                    // Source inline expansion is semantic, not a best-effort
                    // clone pass. Region-bearing control flow is safe to clone
                    // only while returns stay outside the nested region.
                    if (opContainsNestedReturn(*op))
                        return false;

                    builder.clone(*op, mapping);
                }
                return false;
            }

            static llvm::StringRef describeUnsupportedInlineShape(mlir::func::FuncOp funcOp)
            {
                auto &funcBody = funcOp.getBody();
                if (funcBody.empty())
                    return "empty inline body";
                if (funcBody.getBlocks().size() > 1)
                    return "multi-block inline body";

                Block &entryBlock = funcBody.front();
                unsigned conditionalReturnCount = 0;
                bool hasRegionOpBeforeFirstConditionalReturn = false;
                bool sawConditionalReturn = false;
                for (Operation &op : entryBlock.getOperations())
                {
                    if (isa<ora::IfOp>(op))
                    {
                        ++conditionalReturnCount;
                        sawConditionalReturn = true;
                        continue;
                    }
                    if (!sawConditionalReturn && op.getNumRegions() > 0)
                        hasRegionOpBeforeFirstConditionalReturn = true;
                }

                const bool returnsErrorUnion = funcOp->hasAttr("ora.returns_error_union");
                if (returnsErrorUnion && conditionalReturnCount > 1 && hasRegionOpBeforeFirstConditionalReturn)
                    return "fallible helper with multiple early error returns and region-bearing checked-condition prelude";
                if (returnsErrorUnion && conditionalReturnCount > 1)
                    return "fallible helper with multiple early error returns";
                if (returnsErrorUnion && hasRegionOpBeforeFirstConditionalReturn)
                    return "fallible helper with region-bearing checked-condition prelude";
                if (returnsErrorUnion && conditionalReturnCount == 1)
                    return "fallible helper used through error-union propagation";
                if (conditionalReturnCount > 1)
                    return "multiple early returns";
                if (hasRegionOpBeforeFirstConditionalReturn)
                    return "region-bearing operations before early return";
                if (containsNestedReturn(funcOp))
                    return "nested return/control-flow shape";
                return "shape not accepted by current source-inline inliner";
            }

            bool inlineEarlyReturnIfCall(mlir::func::CallOp callOp, mlir::func::FuncOp funcOp)
            {
                auto &funcBody = funcOp.getBody();
                if (funcBody.empty() || funcBody.getBlocks().size() > 1)
                    return false;
                Block *entryBlock = &funcBody.front();

                IRMapping mapping;
                for (unsigned i = 0; i < callOp.getNumOperands(); ++i)
                {
                    if (i < entryBlock->getNumArguments())
                        mapping.map(entryBlock->getArgument(i), callOp.getOperand(i));
                }

                OpBuilder builder(callOp);
                SmallVector<Value> returnOperands;
                if (!cloneOpsUntilReturn(&entryBlock->front(), nullptr, builder, mapping, callOp.getResultTypes(), returnOperands))
                    return false;
                if (returnOperands.size() != callOp.getNumResults())
                    return false;
                for (unsigned i = 0; i < returnOperands.size(); ++i)
                    callOp.getResult(i).replaceAllUsesWith(returnOperands[i]);
                callOp.erase();
                return true;
            }

            bool inlineCall(mlir::func::CallOp callOp, mlir::func::FuncOp funcOp)
            {
                auto &funcBody = funcOp.getBody();
                if (funcBody.empty())
                    return false;
                Block *entryBlock = &funcBody.front();
                if (entryBlock->empty() || funcBody.getBlocks().size() > 1)
                    return false;
                if (containsNestedReturn(funcOp))
                    return inlineEarlyReturnIfCall(callOp, funcOp);

                IRMapping mapping;
                for (unsigned i = 0; i < callOp.getNumOperands(); ++i)
                {
                    if (i < entryBlock->getNumArguments())
                        mapping.map(entryBlock->getArgument(i), callOp.getOperand(i));
                }

                OpBuilder builder(callOp);
                for (auto &op : entryBlock->getOperations())
                {
                    if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op))
                    {
                        if (!replaceCallResultsFromReturnOperands(callOp, returnOp.getOperands(), mapping))
                            return false;
                        break;
                    }
                    if (auto returnOp = dyn_cast<ora::ReturnOp>(op))
                    {
                        if (!replaceCallResultsFromReturnOperands(callOp, returnOp.getOperands(), mapping))
                            return false;
                        break;
                    }
                    builder.clone(op, mapping);
                }
                callOp.erase();
                return true;
            }

            Pass::Statistic callsInlined{this, "calls-inlined", "Function calls inlined"};
            Pass::Statistic sourceInlineFailures{this, "source-inline-failures", "Required source-inline calls left unexpanded"};
        };

        std::unique_ptr<Pass> createOraInliningPass()
        {
            return std::make_unique<OraInliningPass>();
        }

        //===----------------------------------------------------------------------===//
        // Ora Function Canonicalizer Pass
        //===----------------------------------------------------------------------===//

        class OraFunctionCanonicalizerPass : public FunctionPipelineModulePass<OraFunctionCanonicalizerPass>
        {
        public:
            void runOnOperation() override
            {
                const uint64_t functionCount = countNestedFunctions(getOperation());
                functionsProcessed += functionCount;
                recordOraMlirPassStatistic(OraMlirPassStatistic::OraFunctionsCanonicalized, functionCount);
                if (failed(runNestedFunctionPipeline(
                        getOperation(),
                        "[OraFunctionCanonicalizer] canonicalization failed",
                        [](OpPassManager &funcPM)
                        {
                            addLocationPreservingCanonicalizer(funcPM);
                        })))
                    signalPassFailure();
            }

            StringRef getArgument() const override { return "ora-function-canonicalize"; }
            StringRef getDescription() const override { return "Run canonicalization on nested Ora MLIR functions"; }

        private:
            Pass::Statistic functionsProcessed{this, "functions-processed", "Nested functions processed by the Ora canonicalization pipeline"};
        };

        std::unique_ptr<Pass> createOraFunctionCanonicalizerPass()
        {
            return std::make_unique<OraFunctionCanonicalizerPass>();
        }

        //===----------------------------------------------------------------------===//
        // Ora Function CSE Pass
        //===----------------------------------------------------------------------===//

        class OraFunctionCSEPass : public FunctionPipelineModulePass<OraFunctionCSEPass>
        {
        public:
            void runOnOperation() override
            {
                const uint64_t functionCount = countNestedFunctions(getOperation());
                functionsProcessed += functionCount;
                recordOraMlirPassStatistic(OraMlirPassStatistic::OraFunctionsCSEProcessed, functionCount);
                if (failed(runNestedFunctionPipeline(
                        getOperation(),
                        "[OraFunctionCSE] CSE failed",
                        [](OpPassManager &funcPM)
                        {
                            funcPM.addPass(mlir::createCSEPass());
                        })))
                    signalPassFailure();
            }

            StringRef getArgument() const override { return "ora-function-cse"; }
            StringRef getDescription() const override { return "Run MLIR CSE on nested Ora MLIR functions"; }

        private:
            Pass::Statistic functionsProcessed{this, "functions-processed", "Nested functions processed by the Ora CSE pipeline"};
        };

        std::unique_ptr<Pass> createOraFunctionCSEPass()
        {
            return std::make_unique<OraFunctionCSEPass>();
        }

        //===----------------------------------------------------------------------===//
        // Ora Storage-Read CSE Pass
        //===----------------------------------------------------------------------===//

        static void walkBlocksInRegion(Region &region, llvm::function_ref<void(Block &)> callback)
        {
            for (Block &block : region)
            {
                callback(block);
                for (Operation &op : block)
                    for (Region &nested : op.getRegions())
                        walkBlocksInRegion(nested, callback);
            }
        }

        static bool hasMemoryWriteOrUnknownEffect(Operation *op)
        {
            if (isa<SLoadOp>(op))
                return false;

            // A nested region can hide storage writes behind control flow. Keep
            // this pass block-local and restart after any region operation.
            if (op->getNumRegions() != 0)
                return true;

            SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
            if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op))
            {
                effectInterface.getEffects(effects);
                for (auto effect : effects)
                {
                    if (isa<MemoryEffects::Write, MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect()))
                        return true;
                }
                return false;
            }

            return !isMemoryEffectFree(op);
        }

        class OraStorageReadCSEPass : public PassWrapper<OraStorageReadCSEPass, OperationPass<ModuleOp>>
        {
        public:
            void runOnOperation() override
            {
                getOperation().walk([&](mlir::func::FuncOp funcOp)
                                    {
                    for (Region &region : funcOp->getRegions())
                    {
                        walkBlocksInRegion(region, [&](Block &block)
                                           {
                            DenseMap<Attribute, Value> availableLoads;
                            for (Operation &op : llvm::make_early_inc_range(block))
                            {
                                if (auto loadOp = dyn_cast<SLoadOp>(&op))
                                {
                                    Attribute global = loadOp.getGlobalAttr();
                                    auto existing = availableLoads.find(global);
                                    if (existing != availableLoads.end() &&
                                        existing->second.getType() == loadOp.getResult().getType())
                                    {
                                        loadOp.getResult().replaceAllUsesWith(existing->second);
                                        loadOp.erase();
                                        ++storageReadsReused;
                                        recordOraMlirPassStatistic(OraMlirPassStatistic::OraStorageReadsReused);
                                        continue;
                                    }

                                    availableLoads[global] = loadOp.getResult();
                                    continue;
                                }

                                if (auto storeOp = dyn_cast<SStoreOp>(&op))
                                {
                                    availableLoads.erase(storeOp.getGlobalAttr());
                                    continue;
                                }

                                if (hasMemoryWriteOrUnknownEffect(&op))
                                    availableLoads.clear();
                            } });
                    } });
            }

            StringRef getArgument() const override { return "ora-storage-read-cse"; }
            StringRef getDescription() const override { return "Reuse repeated Ora storage loads inside safe block-local regions"; }

        private:
            Pass::Statistic storageReadsReused{this, "storage-reads-reused", "Repeated Ora storage reads replaced with prior loads"};
        };

        std::unique_ptr<Pass> createOraStorageReadCSEPass()
        {
            return std::make_unique<OraStorageReadCSEPass>();
        }

    } // namespace ora
} // namespace mlir
