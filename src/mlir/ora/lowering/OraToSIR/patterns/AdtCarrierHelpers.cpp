#include "patterns/AdtCarrierHelpers.h"
#include "patterns/LoweringHelpers.h"

#include "OraDialect.h"
#include "OraMaterializationKinds.h"
#include "OraToSIRTypeConverter.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using mlir::ora::lowering::coerceToU256;
using mlir::ora::lowering::constU256;

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

            Value adtStoragePayloadSlot(OpBuilder &builder,
                                        Location loc,
                                        Value baseSlot)
            {
                auto *ctx = builder.getContext();
                auto u256Type = sir::U256Type::get(ctx);
                auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

                Value offset = builder.create<sir::ConstOp>(
                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, kAdtStoragePayloadSlotOffset));
                return builder.create<sir::AddOp>(loc, u256Type, baseSlot, offset);
            }

            std::pair<Value, Value> loadAdtPartsFromStorageRoot(OpBuilder &builder,
                                                                Location loc,
                                                                Value baseSlot)
            {
                auto *ctx = builder.getContext();
                auto u256Type = sir::U256Type::get(ctx);

                Value payloadSlot = adtStoragePayloadSlot(builder, loc, baseSlot);
                Value tag = builder.create<sir::SLoadOp>(loc, u256Type, baseSlot);
                Value payload = builder.create<sir::SLoadOp>(loc, u256Type, payloadSlot);
                return {tag, payload};
            }

            void storeAdtPartsToStorageRoot(OpBuilder &builder,
                                            Location loc,
                                            Value baseSlot,
                                            Value tag,
                                            Value payload)
            {
                Value payloadSlot = adtStoragePayloadSlot(builder, loc, baseSlot);
                builder.create<sir::SStoreOp>(loc, baseSlot, tag);
                builder.create<sir::SStoreOp>(loc, payloadSlot, payload);
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

            static FailureOr<Value> materializeAdtPayloadCarrier(
                PatternRewriter &rewriter,
                Location loc,
                Type payloadType,
                Value payloadValue,
                AdtConstructCarrierOptions options)
            {
                if (!usesAggregateAdtPayloadHandle(payloadType))
                {
                    if (options.requireExistingScalarCarrier)
                    {
                        Value existing = lowering::existingU256Value(payloadValue);
                        if (!existing)
                            return failure();
                        return existing;
                    }
                    return coerceToU256(rewriter, loc, payloadValue);
                }

                auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace=*/1);
                if (auto ptr = ora::materializePtrCarrierFromOraValue(rewriter, loc, ptrType, payloadValue))
                    return coerceToU256(rewriter, loc, *ptr);

                if (options.acceptExistingAggregateCarrier &&
                    llvm::isa<sir::PtrType, sir::U256Type>(payloadValue.getType()))
                {
                    return coerceToU256(rewriter, loc, payloadValue);
                }

                if (options.acceptSingleOperandCarrierCast)
                {
                    if (auto cast = payloadValue.getDefiningOp<mlir::UnrealizedConversionCastOp>())
                    {
                        if (cast.getNumOperands() == 1 &&
                            llvm::isa<sir::PtrType, sir::U256Type>(cast.getOperand(0).getType()))
                        {
                            return coerceToU256(rewriter, loc, cast.getOperand(0));
                        }
                    }
                }

                return failure();
            }

            FailureOr<std::pair<Value, Value>>
            materializeAdtConstructParts(PatternRewriter &rewriter,
                                         Location loc,
                                         ora::AdtType adtType,
                                         StringRef variantName,
                                         ValueRange payloadValues,
                                         AdtConstructCarrierOptions options)
            {
                auto variantIndex = getAdtVariantIndex(adtType, variantName);
                if (failed(variantIndex))
                    return failure();

                Value tag = constU256(rewriter, loc, *variantIndex);
                Value payload = constU256(rewriter, loc, 0);

                if (!payloadValues.empty())
                {
                    if (payloadValues.size() != 1)
                        return failure();

                    auto carrier = materializeAdtPayloadCarrier(
                        rewriter, loc, adtType.getPayloadTypes()[*variantIndex],
                        payloadValues.front(), options);
                    if (failed(carrier))
                        return failure();
                    payload = *carrier;
                }

                return std::make_pair(tag, payload);
            }

            FailureOr<std::pair<Value, Value>>
            materializeAdtConstructParts(PatternRewriter &rewriter,
                                         Location loc,
                                         ora::AdtConstructOp constructOp,
                                         AdtConstructCarrierOptions options)
            {
                auto adtType = llvm::dyn_cast<ora::AdtType>(constructOp.getResult().getType());
                if (!adtType)
                    return failure();
                return materializeAdtConstructParts(
                    rewriter, loc, adtType, constructOp.getVariantName(),
                    constructOp.getPayloadValues(), options);
            }

            FailureOr<std::pair<Value, Value>>
            getAdtPartsFromConstructOrNormalized(PatternRewriter &rewriter,
                                                 Location loc,
                                                 Value value,
                                                 AdtConstructCarrierOptions options)
            {
                if (auto constructOp = value.getDefiningOp<ora::AdtConstructOp>())
                    return materializeAdtConstructParts(rewriter, loc, constructOp, options);
                return getNormalizedAdtPartsFromValue(rewriter, loc, value);
            }

            FailureOr<std::pair<Value, Value>>
            getNormalizedAdtPartsFromOperands(PatternRewriter &rewriter,
                                              Location loc,
                                              ArrayRef<Value> operands)
            {
                if (operands.size() != 2)
                    return failure();
                return std::make_pair(
                    coerceToU256(rewriter, loc, operands[0]),
                    coerceToU256(rewriter, loc, operands[1]));
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
                    coerceToU256(rewriter, loc, cast.getOperand(0)),
                    coerceToU256(rewriter, loc, cast.getOperand(1)));
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
                        loc, loweredType, coerceToU256(rewriter, loc, payload));
                    return success();
                }

                if (payload.getType() == loweredType)
                    return success();
                if (llvm::isa<sir::U256Type>(loweredType))
                {
                    payload = coerceToU256(rewriter, loc, payload);
                    return success();
                }
                if (llvm::isa<mlir::IntegerType>(loweredType))
                {
                    payload = rewriter.create<sir::BitcastOp>(
                        loc, loweredType, coerceToU256(rewriter, loc, payload));
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
