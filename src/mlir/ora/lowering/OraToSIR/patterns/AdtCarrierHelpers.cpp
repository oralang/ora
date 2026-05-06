#include "patterns/AdtCarrierHelpers.h"

#include "OraDialect.h"
#include "OraMaterializationKinds.h"
#include "OraToSIRTypeConverter.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace
{
    // Local copy of ensureU256 — the per-pattern translation units each carry
    // their own static version. Folding those into a shared helper is a
    // separate cleanup; this keeps C1 scoped to the ADT-helper dedup.
    Value ensureU256Local(PatternRewriter &rewriter, Location loc, Value value)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        if (llvm::isa<sir::U256Type>(value.getType()))
            return value;
        return rewriter.create<sir::BitcastOp>(loc, u256Type, value);
    }
} // namespace

namespace mlir
{
    namespace ora
    {
        namespace adt_helpers
        {

            Value materializeAdtHandle(OpBuilder &builder,
                                       Location loc,
                                       Value tag,
                                       Value payload)
            {
                auto *ctx = builder.getContext();
                auto u256Type = sir::U256Type::get(ctx);
                auto ptrType = sir::PtrType::get(ctx, /*addrSpace=*/1);
                auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

                Value tagU256 = llvm::isa<sir::U256Type>(tag.getType())
                                    ? tag
                                    : builder.create<sir::BitcastOp>(loc, u256Type, tag).getResult();
                Value payloadU256 = llvm::isa<sir::U256Type>(payload.getType())
                                        ? payload
                                        : builder.create<sir::BitcastOp>(loc, u256Type, payload).getResult();

                Value sizeVal = builder.create<sir::ConstOp>(
                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, kAdtCarrierSize));
                Value basePtr = builder.create<sir::MallocOp>(loc, ptrType, sizeVal);
                builder.create<sir::StoreOp>(loc, basePtr, tagU256);
                Value payloadOffset = builder.create<sir::ConstOp>(
                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, kAdtPayloadOffset));
                Value payloadPtr = builder.create<sir::AddPtrOp>(loc, ptrType, basePtr, payloadOffset);
                builder.create<sir::StoreOp>(loc, payloadPtr, payloadU256);
                return basePtr;
            }

            std::pair<Value, Value> loadAdtPartsFromHandle(OpBuilder &builder,
                                                           Location loc,
                                                           Value handle)
            {
                auto *ctx = builder.getContext();
                auto u256Type = sir::U256Type::get(ctx);
                auto ptrType = sir::PtrType::get(ctx, /*addrSpace=*/1);
                auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

                Value basePtr = handle;
                if (llvm::isa<sir::U256Type>(basePtr.getType()))
                    basePtr = builder.create<sir::BitcastOp>(loc, ptrType, basePtr);

                Value tag = builder.create<sir::LoadOp>(loc, u256Type, basePtr);
                Value payloadOffset = builder.create<sir::ConstOp>(
                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, kAdtPayloadOffset));
                Value payloadPtr = builder.create<sir::AddPtrOp>(loc, ptrType, basePtr, payloadOffset);
                Value payload = builder.create<sir::LoadOp>(loc, u256Type, payloadPtr);
                return {tag, payload};
            }

            bool usesAggregateAdtPayloadHandle(Type type)
            {
                return llvm::isa<ora::TupleType, ora::StructType, ora::AnonymousStructType,
                                 ora::StringType, ora::BytesType,
                                 mlir::MemRefType, mlir::UnrankedMemRefType>(type);
            }

            FailureOr<unsigned> getAdtVariantIndex(ora::AdtType type, StringRef variantName)
            {
                for (auto [index, name] : llvm::enumerate(type.getVariantNames()))
                    if (name == variantName)
                        return static_cast<unsigned>(index);
                return failure();
            }

            FailureOr<std::pair<Value, Value>>
            getNormalizedAdtPartsFromOperands(PatternRewriter &rewriter,
                                              Location loc,
                                              ArrayRef<Value> operands)
            {
                if (operands.size() != 2)
                    return failure();
                return std::make_pair(
                    ensureU256Local(rewriter, loc, operands[0]),
                    ensureU256Local(rewriter, loc, operands[1]));
            }

            FailureOr<std::pair<Value, Value>>
            getNormalizedAdtPartsFromValue(PatternRewriter &rewriter,
                                           Location loc,
                                           Value value)
            {
                auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
                if (!cast || !ora::hasMaterializationKind(cast, ora::mat_kind::kNormalizedAdt))
                    return failure();
                if (cast.getNumOperands() != 2)
                    return failure();
                return std::make_pair(
                    ensureU256Local(rewriter, loc, cast.getOperand(0)),
                    ensureU256Local(rewriter, loc, cast.getOperand(1)));
            }

            LogicalResult decodeAdtPayloadFromCarrier(Operation *op,
                                                      ConversionPatternRewriter &rewriter,
                                                      Location loc,
                                                      Type payloadType,
                                                      Type loweredType,
                                                      Value &payload)
            {
                if (usesAggregateAdtPayloadHandle(payloadType))
                {
                    if (!llvm::isa<sir::PtrType>(loweredType))
                        return rewriter.notifyMatchFailure(
                            op, "aggregate ADT payload requires compiler-runtime handle lowering");
                    payload = rewriter.create<sir::BitcastOp>(
                        loc, loweredType, ensureU256Local(rewriter, loc, payload));
                    return success();
                }

                if (payload.getType() == loweredType)
                    return success();
                if (llvm::isa<sir::U256Type>(loweredType))
                {
                    payload = ensureU256Local(rewriter, loc, payload);
                    return success();
                }
                if (llvm::isa<mlir::IntegerType>(loweredType))
                {
                    payload = rewriter.create<sir::BitcastOp>(
                        loc, loweredType, ensureU256Local(rewriter, loc, payload));
                    return success();
                }
                return rewriter.notifyMatchFailure(op, "unsupported lowered ADT payload result type");
            }

            LogicalResult convertAdtTagCommon(ora::AdtTagOp op,
                                              ArrayRef<Value> operands,
                                              ConversionPatternRewriter &rewriter)
            {
                auto parts = getNormalizedAdtPartsFromOperands(rewriter, op.getLoc(), operands);
                if (failed(parts))
                    parts = getNormalizedAdtPartsFromValue(rewriter, op.getLoc(), op.getValue());
                if (failed(parts))
                    return rewriter.notifyMatchFailure(op, "expected normalized ADT carrier");

                rewriter.replaceOp(op, parts->first);
                return success();
            }

            LogicalResult convertAdtPayloadCommon(ora::AdtPayloadOp op,
                                                  ArrayRef<Value> operands,
                                                  const TypeConverter *typeConverter,
                                                  ConversionPatternRewriter &rewriter)
            {
                auto adtType = llvm::dyn_cast<ora::AdtType>(op.getValue().getType());
                if (!adtType)
                    return rewriter.notifyMatchFailure(op, "expected !ora.adt payload source");

                auto variantIndex = getAdtVariantIndex(adtType, op.getVariantName());
                if (failed(variantIndex))
                    return rewriter.notifyMatchFailure(op, "unknown ADT variant");

                Type payloadType = adtType.getPayloadTypes()[*variantIndex];
                if (llvm::isa<mlir::NoneType>(payloadType))
                    return rewriter.notifyMatchFailure(op, "unit ADT variant has no payload");

                auto parts = getNormalizedAdtPartsFromOperands(rewriter, op.getLoc(), operands);
                if (failed(parts))
                    parts = getNormalizedAdtPartsFromValue(rewriter, op.getLoc(), op.getValue());
                if (failed(parts))
                    return rewriter.notifyMatchFailure(op, "expected normalized ADT carrier");

                Value payload = parts->second;
                Type loweredType = op.getResult().getType();
                if (typeConverter)
                    if (Type converted = typeConverter->convertType(loweredType))
                        loweredType = converted;

                if (failed(decodeAdtPayloadFromCarrier(op, rewriter, op.getLoc(), payloadType, loweredType, payload)))
                    return failure();

                rewriter.replaceOp(op, payload);
                return success();
            }

        } // namespace adt_helpers
    } // namespace ora
} // namespace mlir
