#include "patterns/ControlFlow.h"
#include "patterns/AbiLoweringCommon.h"
#include "patterns/ErrorUnionCarrierHelpers.h"
#include "OraMaterializationKinds.h"
#include "OraToSIRTypeConverter.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;
using namespace mlir::ora::abi_lowering;
using mlir::ora::lowering::coerceToU256;
using mlir::ora::lowering::createPtrViewMaterializationCast;
namespace euh = mlir::ora::error_union_helpers;

namespace
{
static LogicalResult getErrorUnionEncodingTypes(const TypeConverter *typeConverter,
                                                Type resultType,
                                                Operation *op,
                                                SmallVector<Type> &convertedTypes)
{
    if (!typeConverter)
        return failure();
    if (failed(typeConverter->convertType(resultType, convertedTypes)))
        convertedTypes.clear();

    if (auto errType = llvm::dyn_cast<ora::ErrorUnionType>(resultType))
    {
        auto *ctx = resultType.getContext();
        if (!ctx)
            return failure();
        auto u256 = sir::U256Type::get(ctx);
        const bool narrow = euh::isNarrowErrorUnion(errType);
        if (narrow && !euh::hasForceWideErrorUnionAttr(op))
        {
            if (convertedTypes.size() != 1 || !isa<sir::U256Type>(convertedTypes.front()))
            {
                convertedTypes.clear();
                convertedTypes.push_back(u256);
            }
        }
        else
        {
            if (convertedTypes.size() != 2)
            {
                convertedTypes.clear();
                convertedTypes.push_back(u256);
                convertedTypes.push_back(euh::getWideErrorUnionCarrierType(ctx, errType.getSuccessType()));
            }
        }
    }

    if (auto adtType = llvm::dyn_cast<ora::AdtType>(resultType))
    {
        auto *ctx = resultType.getContext();
        if (!ctx)
            return failure();
        auto u256 = sir::U256Type::get(ctx);
        if (convertedTypes.size() != 2)
        {
            convertedTypes.clear();
            convertedTypes.push_back(u256);
            convertedTypes.push_back(u256);
        }
    }

    if (convertedTypes.empty())
        return failure();
    return success();
}

    static bool isSupportedNarrowScalarAbiNodeForType(const AbiLayoutNode &node, Type successType)
    {
        successType = abiDecodeUnwrapRefinementType(successType);
        if (isStaticBoolAbiNode(node))
        {
            if (llvm::isa<ora::BoolType>(successType))
                return true;
            auto intType = llvm::dyn_cast<mlir::IntegerType>(successType);
            return intType && intType.getWidth() == 1;
        }
        if (node.kind != AbiLayoutKind::Static)
            return false;
        if (node.staticKind == AbiStaticKind::Address)
            return node.width == 160 && llvm::isa<ora::AddressType>(successType);
        if (node.staticKind == AbiStaticKind::Uint && node.width > 0 && node.width < 256)
        {
            if (auto enumWidth = enumReprBitWidth(successType))
                return *enumWidth == node.width;
            if (auto oraIntType = llvm::dyn_cast<ora::IntegerType>(successType))
                return !oraIntType.getIsSigned() && oraIntType.getWidth() == node.width;
            auto intType = llvm::dyn_cast<mlir::IntegerType>(successType);
            return intType && intType.getWidth() == node.width;
        }
        if (node.staticKind == AbiStaticKind::Int && node.width > 0 && node.width < 256 && node.width % 8 == 0)
        {
            if (auto oraIntType = llvm::dyn_cast<ora::IntegerType>(successType))
                return oraIntType.getIsSigned() && oraIntType.getWidth() == node.width;
            auto intType = llvm::dyn_cast<mlir::IntegerType>(successType);
            return intType && intType.getWidth() == node.width;
        }
        return false;
    }

    static bool isSupportedFixedBytesAbiNodeForType(const AbiLayoutNode &node, Type successType)
    {
        return isStaticFixedBytesAbiNode(node) && successTypeIsU256Backed(successType);
    }

    using AbiDecodeError = lowering::AbiDecodeError;

    static Value abiDecodeErrorValue(
        PatternRewriter &rewriter,
        Location loc,
        AbiDecodeError err)
    {
        // Must match src/abi/comptime_decoder.zig DecodeError enum order.
        return constU256(rewriter, loc, static_cast<uint64_t>(err));
    }

    static Value abiDecodeRefinementSatisfied(
        PatternRewriter &rewriter,
        Location loc,
        Type type,
        Value canonicalWord,
        bool isSignedBase)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value valid;

        auto combine = [&](Value next) -> Value {
            if (!next)
                return valid;
            if (!valid)
                return next;
            return rewriter.create<sir::AndOp>(loc, u256Type, valid, next);
        };

        if (auto minType = llvm::dyn_cast<ora::MinValueType>(type))
        {
            valid = combine(abiDecodeRefinementSatisfied(rewriter, loc, minType.getBaseType(), canonicalWord, isSignedBase));
            Value bound = constU256(rewriter, loc, abiDecodeBoundAPInt(
                                                     minType.getMinHighHigh(),
                                                     minType.getMinHighLow(),
                                                     minType.getMinLowHigh(),
                                                     minType.getMinLowLow()));
            return combine(abiDecodeWordGte(rewriter, loc, canonicalWord, bound, isSignedBase));
        }
        if (auto maxType = llvm::dyn_cast<ora::MaxValueType>(type))
        {
            valid = combine(abiDecodeRefinementSatisfied(rewriter, loc, maxType.getBaseType(), canonicalWord, isSignedBase));
            Value bound = constU256(rewriter, loc, abiDecodeBoundAPInt(
                                                     maxType.getMaxHighHigh(),
                                                     maxType.getMaxHighLow(),
                                                     maxType.getMaxLowHigh(),
                                                     maxType.getMaxLowLow()));
            return combine(abiDecodeWordLte(rewriter, loc, canonicalWord, bound, isSignedBase));
        }
        if (auto rangeType = llvm::dyn_cast<ora::InRangeType>(type))
        {
            valid = combine(abiDecodeRefinementSatisfied(rewriter, loc, rangeType.getBaseType(), canonicalWord, isSignedBase));
            Value minBound = constU256(rewriter, loc, abiDecodeBoundAPInt(
                                                        rangeType.getMinHighHigh(),
                                                        rangeType.getMinHighLow(),
                                                        rangeType.getMinLowHigh(),
                                                        rangeType.getMinLowLow()));
            Value maxBound = constU256(rewriter, loc, abiDecodeBoundAPInt(
                                                        rangeType.getMaxHighHigh(),
                                                        rangeType.getMaxHighLow(),
                                                        rangeType.getMaxLowHigh(),
                                                        rangeType.getMaxLowLow()));
            Value gte = abiDecodeWordGte(rewriter, loc, canonicalWord, minBound, isSignedBase);
            Value lte = abiDecodeWordLte(rewriter, loc, canonicalWord, maxBound, isSignedBase);
            return combine(rewriter.create<sir::AndOp>(loc, u256Type, gte, lte));
        }
        if (llvm::isa<ora::NonZeroAddressType>(type))
        {
            Value zero = constU256(rewriter, loc, 0);
            Value isZero = rewriter.create<sir::EqOp>(loc, u256Type, canonicalWord, zero);
            return rewriter.create<sir::IsZeroOp>(loc, u256Type, isZero);
        }
        return {};
    }

    static Value packNarrowAbiDecodeResult(
        PatternRewriter &rewriter,
        Location loc,
        Value payload,
        bool isError)
    {
        // Narrow Result<T, AbiDecodeError> uses the same packed error-union
        // carrier as NormalizeErrorOkOp/NormalizeErrorErrOp: bit 0 is the tag
        // (0 = Ok, 1 = Err), bits 1+ carry the u256-normalized payload.
        // Current callers pass u256 payload words; ensureU256 preserves that
        // precondition and documents the intended future scalar entry point.
        Value normalizedPayload = coerceToU256(rewriter, loc, payload);
        if (isError)
        {
            Value tag = ora::error_union_helpers::narrowErrTagConst(rewriter, loc);
            return ora::error_union_helpers::packNarrowCarrierWithShift(
                rewriter, loc, tag, normalizedPayload, tag);
        }

        Value tag = ora::error_union_helpers::narrowOkTagConst(rewriter, loc);
        return ora::error_union_helpers::packNarrowCarrier(rewriter, loc, tag, normalizedPayload);
    }

    static Value packSelectedAbiDecodeResult(
        PatternRewriter &rewriter,
        Location loc,
        Value valid,
        Value payload,
        Value errorPayload)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value ok = packNarrowAbiDecodeResult(rewriter, loc, payload, /*isError=*/false);
        Value err = packNarrowAbiDecodeResult(rewriter, loc, errorPayload, /*isError=*/true);
        return rewriter.create<sir::SelectOp>(loc, u256Type, valid, ok, err);
    }

    struct ScalarAbiWordDecode
    {
        Value payload;
        Value canonicalWord;
        Value valid;
        AbiDecodeError error;
    };

    static ScalarAbiWordDecode decodeScalarWord(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Value word,
        Value one,
        bool permissive)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value payload = word;
        Value canonicalWord = word;
        Value valid = one;
        AbiDecodeError err = AbiDecodeError::NonCanonicalPadding;

        if (isStaticU256AbiNode(node) || isStaticI256AbiNode(node))
        {
            return ScalarAbiWordDecode{payload, canonicalWord, valid, err};
        }

        switch (node.staticKind)
        {
        case AbiStaticKind::Bool:
            valid = permissive ? one : boolAbiWordIsCanonical(rewriter, loc, word);
            payload = permissive ? boolAbiWordPermissivePayload(rewriter, loc, word) : word;
            canonicalWord = word;
            err = AbiDecodeError::InvalidBoolValue;
            break;
        case AbiStaticKind::Uint:
            payload = maskLowBits(rewriter, loc, word, node.width);
            canonicalWord = payload;
            valid = permissive ? one : rewriter.create<sir::EqOp>(loc, u256Type, word, payload).getResult();
            err = AbiDecodeError::NonCanonicalPadding;
            break;
        case AbiStaticKind::Int:
        {
            Value byteIndex = constU256(rewriter, loc, (node.width / 8) - 1);
            Value expected = rewriter.create<sir::SignExtendOp>(loc, u256Type, byteIndex, word).getResult();
            payload = maskLowBits(rewriter, loc, word, node.width);
            canonicalWord = expected;
            valid = permissive ? one : rewriter.create<sir::EqOp>(loc, u256Type, word, expected).getResult();
            err = AbiDecodeError::NonCanonicalPadding;
            break;
        }
        case AbiStaticKind::Address:
            payload = maskLowBits(rewriter, loc, word, 160);
            canonicalWord = payload;
            valid = permissive ? one : rewriter.create<sir::EqOp>(loc, u256Type, word, payload).getResult();
            err = AbiDecodeError::InvalidAddress;
            break;
        case AbiStaticKind::FixedBytes:
        {
            err = AbiDecodeError::InvalidFixedBytes;
            if (node.width == 32)
            {
                payload = word;
                canonicalWord = word;
                valid = one;
                break;
            }
            const uint64_t shiftBits = static_cast<uint64_t>(32 - node.width) * 8ULL;
            Value shift = constU256(rewriter, loc, shiftBits);
            payload = rewriter.create<sir::ShrOp>(loc, u256Type, shift, word).getResult();
            canonicalWord = rewriter.create<sir::ShlOp>(loc, u256Type, shift, payload).getResult();
            valid = permissive ? one : rewriter.create<sir::EqOp>(loc, u256Type, word, canonicalWord).getResult();
            break;
        }
        }

        return ScalarAbiWordDecode{payload, canonicalWord, valid, err};
    }

    static Value lowerNarrowScalarMemoryResult(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Type successType,
        Operation *contextOp,
        Value payloadPtr,
        uint64_t byteOffset,
        bool permissive)
    {
        if (!isSupportedNarrowScalarAbiNodeForType(node, successType))
            return {};

        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value one = constU256(rewriter, loc, 1);
        Value word = decodeAbiU256FromMemory(rewriter, loc, payloadPtr, byteOffset);
        ScalarAbiWordDecode decoded = decodeScalarWord(rewriter, loc, node, word, one, permissive);

        if (std::optional<uint64_t> enumCount = enumVariantCountForType(successType, contextOp))
        {
            // Enum decode is uintN decode plus an ordinal range check. This
            // mirrors the comptime decoder's current positional-variant model.
            Value rangeValid = rewriter.create<sir::LtOp>(loc, u256Type, decoded.payload, constU256(rewriter, loc, *enumCount));
            Value allValid = rewriter.create<sir::AndOp>(loc, u256Type, decoded.valid, rangeValid);
            Value paddingError = abiDecodeErrorValue(rewriter, loc, decoded.error);
            Value rangeError = abiDecodeErrorValue(rewriter, loc, AbiDecodeError::EnumOutOfRange);
            Value errorValue = rewriter.create<sir::SelectOp>(loc, u256Type, decoded.valid, rangeError, paddingError);
            return packSelectedAbiDecodeResult(rewriter, loc, allValid, decoded.payload, errorValue);
        }
        if (isEnumSuccessType(successType))
            return {};

        Value refinementValid = abiDecodeRefinementSatisfied(rewriter, loc, successType, decoded.canonicalWord, node.staticKind == AbiStaticKind::Int);
        Value allValid = refinementValid ? rewriter.create<sir::AndOp>(loc, u256Type, decoded.valid, refinementValid).getResult() : decoded.valid;
        Value abiErrorValue = abiDecodeErrorValue(rewriter, loc, decoded.error);
        Value refinementErrorValue = abiDecodeErrorValue(rewriter, loc, AbiDecodeError::RefinementViolation);
        Value errorValue = refinementValid
                               ? rewriter.create<sir::SelectOp>(loc, u256Type, decoded.valid, refinementErrorValue, abiErrorValue).getResult()
                               : abiErrorValue;
        return packSelectedAbiDecodeResult(rewriter, loc, allValid, decoded.payload, errorValue);
    }

    struct WideAbiDecodeResult
    {
        Value tag;
        Value payload;
    };

    static WideAbiDecodeResult selectWideAbiDecodeResult(
        PatternRewriter &rewriter,
        Location loc,
        Value valid,
        Value payload,
        AbiDecodeError invalidError,
        bool permissive)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value zero = constU256(rewriter, loc, 0);
        Value one = constU256(rewriter, loc, 1);
        Value errorValue = abiDecodeErrorValue(rewriter, loc, invalidError);
        return WideAbiDecodeResult{
            permissive ? zero : rewriter.create<sir::SelectOp>(loc, u256Type, valid, zero, one).getResult(),
            permissive ? payload : rewriter.create<sir::SelectOp>(loc, u256Type, valid, payload, errorValue).getResult(),
        };
    }

    struct AbiDecodeResultSink
    {
        ConversionPatternRewriter &rewriter;
        ora::AbiDecodeOp op;
        Location loc;
        ArrayRef<Type> convertedTypes;
        Block *mergeBlock;

        bool isNarrow() const { return convertedTypes.size() == 1; }
        bool isWide() const { return convertedTypes.size() == 2; }

        Type payloadType() const
        {
            return isWide() ? convertedTypes[1] : Type();
        }

        Value adaptPayload(Value payload) const
        {
            Type carrier = payloadType();
            if (carrier && payload.getType() != carrier)
                return rewriter.create<sir::BitcastOp>(loc, carrier, payload);
            return payload;
        }

        LogicalResult branch(ValueRange values) const
        {
            rewriter.create<sir::BrOp>(loc, values, mergeBlock);
            return success();
        }

        LogicalResult branchNarrow(Value packed) const
        {
            if (!isNarrow())
                return failure();
            return branch(ValueRange{packed});
        }

        LogicalResult branchWide(WideAbiDecodeResult result) const
        {
            if (!isWide())
                return failure();
            Value payload = adaptPayload(result.payload);
            return branch(ValueRange{result.tag, payload});
        }

        template <typename MaterializeWidePayload>
        LogicalResult branchErrorPayload(Value errorPayload, MaterializeWidePayload materializeWidePayload) const
        {
            if (isNarrow())
            {
                Value packed = packNarrowAbiDecodeResult(rewriter, loc, errorPayload, /*isError=*/true);
                return branchNarrow(packed);
            }

            Value tag = constU256(rewriter, loc, 1);
            return branchWide(WideAbiDecodeResult{tag, materializeWidePayload(errorPayload)});
        }

        template <typename MaterializeWidePayload>
        LogicalResult branchErrorKind(AbiDecodeError err, MaterializeWidePayload materializeWidePayload) const
        {
            return branchErrorPayload(abiDecodeErrorValue(rewriter, loc, err), materializeWidePayload);
        }

        LogicalResult branchOkPayload(Value payload) const
        {
            return branchWide(WideAbiDecodeResult{constU256(rewriter, loc, 0), payload});
        }

        LogicalResult finish() const
        {
            rewriter.setInsertionPointToStart(mergeBlock);
            SmallVector<SmallVector<Value>> replacementGroups;
            SmallVector<Value> group;
            group.reserve(convertedTypes.size());
            for (BlockArgument arg : mergeBlock->getArguments())
                group.push_back(arg);
            replacementGroups.push_back(std::move(group));
            rewriter.replaceOpWithMultiple(op, replacementGroups);
            return success();
        }
    };

    static WideAbiDecodeResult splitPackedAbiDecodeResult(
        PatternRewriter &rewriter,
        Location loc,
        Value packed)
    {
        auto [tag, payload] = ora::error_union_helpers::splitNarrowPackedCarrier(
            rewriter, loc, packed);
        return WideAbiDecodeResult{tag, payload};
    }

    static Value staticDecodeRequiredLengthConst(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &root)
    {
        // Static-only: headSlotBytes is the full required length only when the
        // layout has no dynamic tails. N3b3 dynamic decode must compute this
        // from runtime offsets and length words instead.
        return constU256(rewriter, loc, root.headSlotBytes());
    }

    static Value applyStaticOversizeCheckToNarrowResult(
        PatternRewriter &rewriter,
        Location loc,
        Value decoded,
        Value length,
        Value required)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value tooLong = rewriter.create<sir::GtOp>(loc, u256Type, length, required);
        Value tag = ora::error_union_helpers::narrowPackedCarrierTag(
            rewriter, loc, decoded);
        Value decodedOk = rewriter.create<sir::IsZeroOp>(loc, u256Type, tag);
        Value oversizeApplies = rewriter.create<sir::AndOp>(loc, u256Type, tooLong, decodedOk);

        Value oversize = packNarrowAbiDecodeResult(
            rewriter,
            loc,
            abiDecodeErrorValue(rewriter, loc, AbiDecodeError::OversizeBuffer),
            /*isError=*/true);
        return rewriter.create<sir::SelectOp>(loc, u256Type, oversizeApplies, oversize, decoded);
    }

    static WideAbiDecodeResult applyStaticOversizeCheckToWideResult(
        PatternRewriter &rewriter,
        Location loc,
        WideAbiDecodeResult decoded,
        Value length,
        Value required,
        Type payloadCarrierType)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value zero = constU256(rewriter, loc, 0);
        Value one = constU256(rewriter, loc, 1);
        Value tooLong = rewriter.create<sir::GtOp>(loc, u256Type, length, required);
        Value decodedOk = rewriter.create<sir::EqOp>(loc, u256Type, decoded.tag, zero);
        Value oversizeApplies = rewriter.create<sir::AndOp>(loc, u256Type, tooLong, decodedOk);

        Value payloadBits = coerceToU256(rewriter, loc, decoded.payload);
        Value oversize = abiDecodeErrorValue(rewriter, loc, AbiDecodeError::OversizeBuffer);

        Value tag = rewriter.create<sir::SelectOp>(loc, u256Type, oversizeApplies, one, decoded.tag);
        Value payload = rewriter.create<sir::SelectOp>(loc, u256Type, oversizeApplies, oversize, payloadBits);
        if (payloadCarrierType && payload.getType() != payloadCarrierType)
            payload = rewriter.create<sir::BitcastOp>(loc, payloadCarrierType, payload);
        return WideAbiDecodeResult{tag, payload};
    }

    static std::optional<WideAbiDecodeResult> lowerFixedBytesMemoryResult(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Type successType,
        bool useWideCarrier,
        Value payloadPtr,
        uint64_t byteOffset,
        bool permissive)
    {
        if (!useWideCarrier || !isSupportedFixedBytesAbiNodeForType(node, successType))
            return std::nullopt;

        Value word = decodeAbiU256FromMemory(rewriter, loc, payloadPtr, byteOffset);
        if (node.width == 32)
        {
            return WideAbiDecodeResult{
                constU256(rewriter, loc, 0),
                word,
            };
        }

        // Canonical bytesN encoding stores the N-byte payload in bytes [0..N-1]
        // and requires bytes [N..32] to be zero. Shift down to extract the
        // payload, then back up; a matching round-trip proves padding is zero.
        FixedBytesWordDecode decoded = decodeFixedBytesAbiWord(rewriter, loc, node.width, word);
        return selectWideAbiDecodeResult(
            rewriter,
            loc,
            decoded.valid,
            decoded.payload,
            AbiDecodeError::InvalidFixedBytes,
            permissive);
    }

    static std::optional<WideAbiDecodeResult> lowerU256BackedNarrowUintMemoryResult(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Type successType,
        bool useWideCarrier,
        Value payloadPtr,
        uint64_t byteOffset,
        bool permissive)
    {
        if (!useWideCarrier ||
            node.kind != AbiLayoutKind::Static ||
            node.staticKind != AbiStaticKind::Uint ||
            node.width == 0 ||
            node.width >= 256 ||
            !successTypeIsU256Backed(successType))
        {
            return std::nullopt;
        }

        // Bitfields currently lower through u256-backed MLIR values while the
        // context-aware ABI layout uses the declared integer width. Validate
        // the declared ABI width, then return the packed word through the wide
        // Result carrier.
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value word = decodeAbiU256FromMemory(rewriter, loc, payloadPtr, byteOffset);
        Value payload = maskLowBits(rewriter, loc, word, node.width);
        Value valid = rewriter.create<sir::EqOp>(loc, u256Type, word, payload);
        return selectWideAbiDecodeResult(
            rewriter,
            loc,
            valid,
            payload,
            AbiDecodeError::NonCanonicalPadding,
            permissive);
    }

    struct StaticMemoryDecodePart
    {
        Value value;
        Value valid;
        Value error;
    };

    static Value adaptDecodedAbiPayloadToType(
        PatternRewriter &rewriter,
        Location loc,
        Value payload,
        Type targetType)
    {
        if (!targetType || payload.getType() == targetType)
            return payload;
        return ora::createMaterializationCast(rewriter, loc, targetType, payload, mat_kind::kPayloadForward);
    }

    static std::optional<StaticMemoryDecodePart> lowerStaticMemoryDecodePart(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Type targetType,
        Operation *contextOp,
        Value payloadPtr,
        uint64_t byteOffset,
        bool permissive)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value zero = constU256(rewriter, loc, 0);
        Value one = constU256(rewriter, loc, 1);

        if (node.kind == AbiLayoutKind::Tuple)
        {
            auto tupleType = llvm::dyn_cast<ora::TupleType>(abiDecodeUnwrapRefinementType(targetType));
            if (!tupleType || tupleType.getElementTypes().size() != node.children.size())
                return std::nullopt;

            SmallVector<Value> elements;
            elements.reserve(node.children.size());
            Value allValid = one;
            Value firstError = zero;
            uint64_t cursor = byteOffset;
            for (auto [index, child] : llvm::enumerate(node.children))
            {
                auto decoded = lowerStaticMemoryDecodePart(
                    rewriter,
                    loc,
                    *child,
                    tupleType.getElementTypes()[index],
                    contextOp,
                    payloadPtr,
                    cursor,
                    permissive);
                if (!decoded)
                    return std::nullopt;

                elements.push_back(decoded->value);
                Value childInvalidError = rewriter.create<sir::SelectOp>(loc, u256Type, decoded->valid, firstError, decoded->error);
                firstError = rewriter.create<sir::SelectOp>(loc, u256Type, allValid, childInvalidError, firstError);
                allValid = rewriter.create<sir::AndOp>(loc, u256Type, allValid, decoded->valid);
                // Static tuple children are encoded inline; nested static tuples
                // occupy the recursive sum of their child head slots.
                cursor += child->headSlotBytes();
            }

            Value tuple = rewriter.create<ora::TupleCreateOp>(loc, tupleType, elements).getResult();
            return StaticMemoryDecodePart{
                adaptDecodedAbiPayloadToType(rewriter, loc, tuple, targetType),
                allValid,
                firstError,
            };
        }

        if (node.kind != AbiLayoutKind::Static)
            return std::nullopt;

        Value word = decodeAbiU256FromMemory(rewriter, loc, payloadPtr, byteOffset);

        if (!isStaticU256AbiNode(node) && !isStaticI256AbiNode(node))
        {
            switch (node.staticKind)
            {
            case AbiStaticKind::Bool:
                if (!isSupportedNarrowScalarAbiNodeForType(node, targetType))
                    return std::nullopt;
                break;
            case AbiStaticKind::Uint:
                if (!(isSupportedNarrowScalarAbiNodeForType(node, targetType) || successTypeIsU256Backed(targetType)))
                    return std::nullopt;
                break;
            case AbiStaticKind::Int:
                if (!isSupportedNarrowScalarAbiNodeForType(node, targetType))
                    return std::nullopt;
                break;
            case AbiStaticKind::Address:
                if (!isSupportedNarrowScalarAbiNodeForType(node, targetType))
                    return std::nullopt;
                break;
            case AbiStaticKind::FixedBytes:
                if (!isSupportedFixedBytesAbiNodeForType(node, targetType))
                    return std::nullopt;
                break;
            }
        }

        ScalarAbiWordDecode decoded = decodeScalarWord(rewriter, loc, node, word, one, permissive);
        Value errorValue = abiDecodeErrorValue(rewriter, loc, decoded.error);
        if (std::optional<uint64_t> enumCount = enumVariantCountForType(targetType, contextOp))
        {
            Value rangeValid = rewriter.create<sir::LtOp>(loc, u256Type, decoded.payload, constU256(rewriter, loc, *enumCount));
            Value paddingError = errorValue;
            Value rangeError = abiDecodeErrorValue(rewriter, loc, AbiDecodeError::EnumOutOfRange);
            errorValue = rewriter.create<sir::SelectOp>(loc, u256Type, decoded.valid, rangeError, paddingError);
            decoded.valid = rewriter.create<sir::AndOp>(loc, u256Type, decoded.valid, rangeValid);
        }
        else if (isEnumSuccessType(targetType))
        {
            return std::nullopt;
        }

        if (Value refinementValid = abiDecodeRefinementSatisfied(rewriter, loc, targetType, decoded.canonicalWord, node.staticKind == AbiStaticKind::Int))
        {
            Value refinementError = abiDecodeErrorValue(rewriter, loc, AbiDecodeError::RefinementViolation);
            errorValue = rewriter.create<sir::SelectOp>(loc, u256Type, decoded.valid, refinementError, errorValue);
            decoded.valid = rewriter.create<sir::AndOp>(loc, u256Type, decoded.valid, refinementValid);
        }

        return StaticMemoryDecodePart{
            adaptDecodedAbiPayloadToType(rewriter, loc, decoded.payload, targetType),
            decoded.valid,
            errorValue,
        };
    }

    static std::optional<Value> lowerReturndataSuccessfulStaticScalarValue(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Type successType,
        Value returndataPtr)
    {
        // External-call returndata decode still maps malformed returns through
        // the legacy error-union path. This helper only projects successful
        // single-word scalar returns according to the parsed ABI layout.
        if (node.kind != AbiLayoutKind::Static)
            return std::nullopt;

        Value word = decodeAbiU256FromMemory(rewriter, loc, returndataPtr, /*byteOffset=*/0);
        if (isStaticU256AbiNode(node) || isStaticI256AbiNode(node))
            return word;

        if (isStaticFixedBytesAbiNode(node) && successTypeIsU256Backed(successType))
        {
            if (node.width == 32)
                return word;
            // Returndata stores bytesN in ABI shape (left-aligned in the
            // 32-byte word). Ora's fixed-bytes value payload is u256-backed,
            // so extract the high N bytes before handing the value to users.
            Value shift = constU256(rewriter, loc, static_cast<uint64_t>(32 - node.width) * 8ULL);
            return rewriter.create<sir::ShrOp>(loc, sir::U256Type::get(rewriter.getContext()), shift, word).getResult();
        }

        if (isEnumSuccessType(successType) || !isSupportedNarrowScalarAbiNodeForType(node, successType))
            return std::nullopt;

        // Success-only returndata projection intentionally masks without
        // validation. Canonical checks must be added when this path can route
        // malformed returndata into the external-call error union.
        switch (node.staticKind)
        {
        case AbiStaticKind::Bool:
            return maskLowBits(rewriter, loc, word, 1);
        case AbiStaticKind::Uint:
            return maskLowBits(rewriter, loc, word, node.width);
        case AbiStaticKind::Int:
            return maskLowBits(rewriter, loc, word, node.width);
        case AbiStaticKind::Address:
            return maskLowBits(rewriter, loc, word, 160);
        case AbiStaticKind::FixedBytes:
            llvm_unreachable("fixed bytes handled before narrow scalar dispatch");
        }
        return std::nullopt;
    }

    template <typename BranchDynamicErr, typename ComputeRequiredBytes, typename EmitAfterBounds, typename Finish>
    static LogicalResult emitStrictDynamicDecodeBounds(
        PatternRewriter &rewriter,
        Location loc,
        Region *parentRegion,
        Block *insertBeforeBlock,
        Block *decodeBlock,
        Value basePtr,
        Value length,
        Type ptrType,
        Type u256Type,
        uint64_t offsetWordByteOffset,
        uint64_t expectedOffset,
        uint64_t minBytes,
        uint64_t capValue,
        AbiDecodeError capError,
        AbiDecodeError tailTooShortError,
        BranchDynamicErr branchDynamicErr,
        ComputeRequiredBytes computeRequiredBytes,
        EmitAfterBounds emitAfterBounds,
        Finish finish,
        bool permissive)
    {
        auto offsetOverflowBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto offsetCheckBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto invalidOffsetBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto minLengthBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto lengthTruncatedBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto lengthBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto lengthOverflowBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto capCheckBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto capExceededBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto sizeBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto tailTooShortBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());

        rewriter.setInsertionPointToEnd(decodeBlock);
        Value offsetWord = decodeAbiU256FromMemory(rewriter, loc, basePtr, offsetWordByteOffset);
        Value offsetExceedsUsize = rewriter.create<sir::GtOp>(
            loc,
            u256Type,
            offsetWord,
            constU256(rewriter, loc, llvm::APInt::getLowBitsSet(256, 64)));
        rewriter.create<sir::CondBrOp>(loc, offsetExceedsUsize, ValueRange{}, ValueRange{}, offsetOverflowBlock, offsetCheckBlock);

        rewriter.setInsertionPointToStart(offsetOverflowBlock);
        if (failed(branchDynamicErr(AbiDecodeError::InvalidOffset, constU256(rewriter, loc, 0))))
            return failure();

        rewriter.setInsertionPointToStart(offsetCheckBlock);
        Value offsetOk = permissive
                             ? constU256(rewriter, loc, 1)
                             : rewriter.create<sir::EqOp>(loc, u256Type, offsetWord, constU256(rewriter, loc, expectedOffset)).getResult();
        rewriter.create<sir::CondBrOp>(loc, offsetOk, ValueRange{}, ValueRange{}, minLengthBlock, invalidOffsetBlock);

        rewriter.setInsertionPointToStart(invalidOffsetBlock);
        if (failed(branchDynamicErr(AbiDecodeError::NonCanonicalEncoding, constU256(rewriter, loc, 0))))
            return failure();

        rewriter.setInsertionPointToStart(minLengthBlock);
        Value dynamicHeadEnd = addU256(rewriter, loc, offsetWord, constU256(rewriter, loc, 32));
        Value strictLengthMissing = rewriter.create<sir::LtOp>(loc, u256Type, length, constU256(rewriter, loc, minBytes));
        Value offsetPastEnd = rewriter.create<sir::GtOp>(loc, u256Type, offsetWord, length);
        Value permissiveLengthMissing = rewriter.create<sir::OrOp>(
            loc,
            u256Type,
            offsetPastEnd,
            rewriter.create<sir::LtOp>(loc, u256Type, length, dynamicHeadEnd));
        Value lengthWordMissing = permissive ? permissiveLengthMissing : strictLengthMissing;
        rewriter.create<sir::CondBrOp>(loc, lengthWordMissing, ValueRange{}, ValueRange{}, lengthTruncatedBlock, lengthBlock);

        rewriter.setInsertionPointToStart(lengthTruncatedBlock);
        if (failed(branchDynamicErr(AbiDecodeError::TruncatedBuffer, constU256(rewriter, loc, 0))))
            return failure();

        rewriter.setInsertionPointToStart(lengthBlock);
        Value tailOffset = permissive ? offsetWord : constU256(rewriter, loc, expectedOffset);
        Value tailPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, tailOffset);
        Value dynamicLen = rewriter.create<sir::LoadOp>(loc, u256Type, tailPtr);
        Value exceedsUsize = rewriter.create<sir::GtOp>(
            loc,
            u256Type,
            dynamicLen,
            constU256(rewriter, loc, llvm::APInt::getLowBitsSet(256, 64)));
        rewriter.create<sir::CondBrOp>(loc, exceedsUsize, ValueRange{}, ValueRange{}, lengthOverflowBlock, capCheckBlock);

        rewriter.setInsertionPointToStart(lengthOverflowBlock);
        if (failed(branchDynamicErr(AbiDecodeError::LengthOverflow, constU256(rewriter, loc, 0))))
            return failure();

        rewriter.setInsertionPointToStart(capCheckBlock);
        Value capExceeded = rewriter.create<sir::GtOp>(loc, u256Type, dynamicLen, constU256(rewriter, loc, capValue));
        rewriter.create<sir::CondBrOp>(loc, capExceeded, ValueRange{}, ValueRange{}, capExceededBlock, sizeBlock);

        rewriter.setInsertionPointToStart(capExceededBlock);
        if (failed(branchDynamicErr(capError, constU256(rewriter, loc, 0))))
            return failure();

        rewriter.setInsertionPointToStart(sizeBlock);
        Value requiredBaseBytes = permissive ? dynamicHeadEnd : constU256(rewriter, loc, minBytes);
        Value requiredDynamic = computeRequiredBytes(tailPtr, dynamicLen, requiredBaseBytes);
        Value tailMissing = rewriter.create<sir::LtOp>(loc, u256Type, length, requiredDynamic);
        Block *afterBoundsBlock = emitAfterBounds(tailPtr, dynamicLen, requiredDynamic);
        if (!afterBoundsBlock)
            return failure();
        rewriter.setInsertionPointToEnd(sizeBlock);
        rewriter.create<sir::CondBrOp>(
            loc,
            tailMissing,
            ValueRange{},
            ValueRange{},
            tailTooShortBlock,
            afterBoundsBlock);

        rewriter.setInsertionPointToStart(tailTooShortBlock);
        if (failed(branchDynamicErr(tailTooShortError, requiredDynamic)))
            return failure();

        return finish();
    }

    struct AbiValidationLoopBlocks
    {
        Block *entry;
        Block *done;
    };

    template <typename ElementValidAt>
    static AbiValidationLoopBlocks emitAbiAccumulatorValidationLoop(
        PatternRewriter &rewriter,
        Location loc,
        Region *parentRegion,
        Block *insertBeforeBlock,
        Type u256Type,
        Value count,
        ElementValidAt elementValidAt)
    {
        auto initBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto condBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        condBlock->addArgument(u256Type, loc);
        condBlock->addArgument(u256Type, loc);
        auto bodyBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        bodyBlock->addArgument(u256Type, loc);
        bodyBlock->addArgument(u256Type, loc);
        auto doneBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        doneBlock->addArgument(u256Type, loc);

        rewriter.setInsertionPointToStart(initBlock);
        rewriter.create<sir::BrOp>(loc, ValueRange{constU256(rewriter, loc, 0), constU256(rewriter, loc, 1)}, condBlock);

        rewriter.setInsertionPointToStart(condBlock);
        Value iv = condBlock->getArgument(0);
        Value allValid = condBlock->getArgument(1);
        Value hasElement = rewriter.create<sir::LtOp>(loc, u256Type, iv, count);
        Value shouldContinue = rewriter.create<sir::AndOp>(loc, u256Type, hasElement, allValid);
        rewriter.create<sir::CondBrOp>(loc, shouldContinue, ValueRange{iv, allValid}, ValueRange{allValid}, bodyBlock, doneBlock);

        rewriter.setInsertionPointToStart(bodyBlock);
        Value bodyIv = bodyBlock->getArgument(0);
        Value bodyValid = bodyBlock->getArgument(1);
        Value itemValid = elementValidAt(bodyIv);
        Value nextValid = rewriter.create<sir::AndOp>(loc, u256Type, bodyValid, itemValid);
        Value nextIv = addU256(rewriter, loc, bodyIv, constU256(rewriter, loc, 1));
        rewriter.create<sir::BrOp>(loc, ValueRange{nextIv, nextValid}, condBlock);

        return AbiValidationLoopBlocks{initBlock, doneBlock};
    }

    template <typename ElementCanonical>
    static AbiValidationLoopBlocks emitWordArrayElementValidationLoop(
        PatternRewriter &rewriter,
        Location loc,
        Region *parentRegion,
        Block *insertBeforeBlock,
        Type ptrType,
        Type u256Type,
        Value tailPtr,
        Value elementLen,
        ElementCanonical elementCanonical)
    {
        Value contentPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, tailPtr, constU256(rewriter, loc, 32));
        return emitAbiAccumulatorValidationLoop(
            rewriter,
            loc,
            parentRegion,
            insertBeforeBlock,
            u256Type,
            elementLen,
            [&](Value bodyIv) -> Value {
                Value elementByteOffset = mulU256(rewriter, loc, bodyIv, constU256(rewriter, loc, 32));
                Value elementPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, contentPtr, elementByteOffset);
                Value elementWord = rewriter.create<sir::LoadOp>(loc, u256Type, elementPtr);
                return elementCanonical(elementWord);
            });
    }

    static AbiValidationLoopBlocks emitFixedBytesArrayValidationLoop(
        PatternRewriter &rewriter,
        Location loc,
        Region *parentRegion,
        Block *insertBeforeBlock,
        Type ptrType,
        Type u256Type,
        Value tailPtr,
        Value elementLen,
        const AbiLayoutNode &elementNode)
    {
        Value sourceContentPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, tailPtr, constU256(rewriter, loc, 32));
        return emitAbiAccumulatorValidationLoop(
            rewriter,
            loc,
            parentRegion,
            insertBeforeBlock,
            u256Type,
            elementLen,
            [&](Value bodyIv) -> Value {
                Value elementByteOffset = mulU256(rewriter, loc, bodyIv, constU256(rewriter, loc, 32));
                Value sourceElementPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, sourceContentPtr, elementByteOffset);
                Value elementWord = rewriter.create<sir::LoadOp>(loc, u256Type, sourceElementPtr);
                FixedBytesWordDecode element = decodeFixedBytesAbiWord(rewriter, loc, elementNode.width, elementWord);
                return element.valid;
            });
    }

    static AbiValidationLoopBlocks emitDynamicBytesPaddingValidationLoop(
        PatternRewriter &rewriter,
        Location loc,
        Region *parentRegion,
        Block *insertBeforeBlock,
        Type ptrType,
        Type u256Type,
        Value tailPtr,
        Value dynamicLen,
        Value padded)
    {
        Value contentPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, tailPtr, constU256(rewriter, loc, 32));
        Value padStart = rewriter.create<sir::AddPtrOp>(loc, ptrType, contentPtr, dynamicLen);
        Value padCount = subU256(rewriter, loc, padded, dynamicLen);
        // ABI canonical padding is byte-addressed, so validation is a
        // per-padding-byte loop. This is correct but gas-linear in the
        // padding length; D2 tracks broader decode hot-path costs.
        return emitAbiAccumulatorValidationLoop(
            rewriter,
            loc,
            parentRegion,
            insertBeforeBlock,
            u256Type,
            padCount,
            [&](Value bodyIv) -> Value {
                Value bytePtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, padStart, bodyIv);
                Value padByte = rewriter.create<sir::Load8Op>(loc, u256Type, bytePtr, constU256(rewriter, loc, 0));
                return rewriter.create<sir::EqOp>(loc, u256Type, padByte, constU256(rewriter, loc, 0));
            });
    }

    struct FixedBytesArrayCopyLoop
    {
        Block *entry;
        Block *done;
        Value resultPtr;
    };

    static FixedBytesArrayCopyLoop emitFixedBytesArrayDecodeCopyLoop(
        PatternRewriter &rewriter,
        Location loc,
        Region *parentRegion,
        Block *insertBeforeBlock,
        Type ptrType,
        Type u256Type,
        Value sourceContentPtr,
        Value elementLen,
        const AbiLayoutNode &elementNode)
    {
        auto copyInitBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        auto copyCondBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        copyCondBlock->addArgument(u256Type, loc);
        auto copyBodyBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());
        copyBodyBlock->addArgument(u256Type, loc);
        auto copyDoneBlock = rewriter.createBlock(parentRegion, insertBeforeBlock->getIterator());

        rewriter.setInsertionPointToStart(copyInitBlock);
        Value elementBytes = mulU256(rewriter, loc, elementLen, constU256(rewriter, loc, 32));
        Value sliceBytes = addU256(rewriter, loc, constU256(rewriter, loc, 32), elementBytes);
        Value resultPtr = rewriter.create<sir::MallocOp>(loc, ptrType, sliceBytes);
        rewriter.create<sir::StoreOp>(loc, resultPtr, elementLen);
        Value resultContentPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, resultPtr, constU256(rewriter, loc, 32));
        rewriter.create<sir::BrOp>(loc, ValueRange{constU256(rewriter, loc, 0)}, copyCondBlock);

        rewriter.setInsertionPointToStart(copyCondBlock);
        Value copyIv = copyCondBlock->getArgument(0);
        Value hasElement = rewriter.create<sir::LtOp>(loc, u256Type, copyIv, elementLen);
        rewriter.create<sir::CondBrOp>(loc, hasElement, ValueRange{copyIv}, ValueRange{}, copyBodyBlock, copyDoneBlock);

        rewriter.setInsertionPointToStart(copyBodyBlock);
        Value bodyIv = copyBodyBlock->getArgument(0);
        Value elementByteOffset = mulU256(rewriter, loc, bodyIv, constU256(rewriter, loc, 32));
        Value sourceElementPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, sourceContentPtr, elementByteOffset);
        Value elementWord = rewriter.create<sir::LoadOp>(loc, u256Type, sourceElementPtr);
        FixedBytesWordDecode element = decodeFixedBytesAbiWord(rewriter, loc, elementNode.width, elementWord);
        Value resultElementPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, resultContentPtr, elementByteOffset);
        rewriter.create<sir::StoreOp>(loc, resultElementPtr, element.payload);
        Value nextIv = addU256(rewriter, loc, bodyIv, constU256(rewriter, loc, 1));
        rewriter.create<sir::BrOp>(loc, ValueRange{nextIv}, copyCondBlock);

        return FixedBytesArrayCopyLoop{copyInitBlock, copyDoneBlock, resultPtr};
    }

}

// -----------------------------------------------------------------------------
// Lower ora.abi_decode - scalar/dynamic/aggregate v1 decode from return buffer
// -----------------------------------------------------------------------------
LogicalResult ConvertAbiDecodeOp::matchAndRewrite(
    ora::AbiDecodeOp op,
    typename ora::AbiDecodeOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
        return rewriter.notifyMatchFailure(op, "missing type converter");

    auto returnTypesAttr = op->getAttrOfType<mlir::ArrayAttr>("return_types");
    if (!returnTypesAttr || returnTypesAttr.size() != 1)
        return rewriter.notifyMatchFailure(op, "only single-result ABI decode is supported");
    auto layoutAttr = op->getAttrOfType<mlir::StringAttr>("layout");
    if (!layoutAttr)
        return rewriter.notifyMatchFailure(op, "missing ABI decode layout attr");
    auto sourceAttr = op->getAttrOfType<mlir::StringAttr>("source");
    auto failureModeAttr = op->getAttrOfType<mlir::StringAttr>("failure_mode");
    if (!sourceAttr || !failureModeAttr)
        return rewriter.notifyMatchFailure(op, "missing ABI decode source or failure mode");
    auto decodeModeAttr = op->getAttrOfType<mlir::StringAttr>("decode_mode");
    const bool permissive = decodeModeAttr && decodeModeAttr.getValue() == "permissive";
    AbiLayoutNode root;
    if (!parseAbiLayout(layoutAttr.getValue(), root, AbiLayoutSyntax::LayoutDsl))
        return rewriter.notifyMatchFailure(op, "unsupported or malformed ABI decode layout attr");

    if (sourceAttr.getValue() == "memory" && failureModeAttr.getValue() == "result")
    {
        auto resultType = llvm::dyn_cast<ora::ErrorUnionType>(op.getResult().getType());
        if (!resultType)
            return rewriter.notifyMatchFailure(op, "memory/result ABI decode requires error-union result type");
        if (root.kind != AbiLayoutKind::Tuple)
            return rewriter.notifyMatchFailure(op, "memory/result ABI decode currently supports tuple-shaped layouts only");
        Type successType = resultType.getSuccessType();
        if (root.children.empty())
        {
            if (!llvm::isa<mlir::NoneType>(successType))
                return rewriter.notifyMatchFailure(op, "memory/result empty ABI decode requires void success type");
            auto u256Type = sir::U256Type::get(op.getContext());
            Value bytesPtr = abiMemoryBytesPtr(rewriter, op.getLoc(), adaptor.getReturndata());
            Value length = rewriter.create<sir::LoadOp>(op.getLoc(), u256Type, bytesPtr);
            Value valid = permissive
                              ? constU256(rewriter, op.getLoc(), 1)
                              : rewriter.create<sir::EqOp>(op.getLoc(), u256Type, length, constU256(rewriter, op.getLoc(), 0)).getResult();
            Value ok = packNarrowAbiDecodeResult(rewriter, op.getLoc(), constU256(rewriter, op.getLoc(), 0), /*isError=*/false);
            Value err = packNarrowAbiDecodeResult(rewriter, op.getLoc(), abiDecodeErrorValue(rewriter, op.getLoc(), AbiDecodeError::OversizeBuffer), /*isError=*/true);
            rewriter.replaceOp(op, rewriter.create<sir::SelectOp>(op.getLoc(), u256Type, valid, ok, err).getResult());
            return success();
        }
        auto u256Type = sir::U256Type::get(op.getContext());
        Value bytesPtr = abiMemoryBytesPtr(rewriter, op.getLoc(), adaptor.getReturndata());
        Value length = rewriter.create<sir::LoadOp>(op.getLoc(), u256Type, bytesPtr);

        SmallVector<Type> convertedResultTypes;
        if (failed(getErrorUnionEncodingTypes(typeConverter, op.getResult().getType(), op.getOperation(), convertedResultTypes)) ||
            convertedResultTypes.empty())
        {
            return rewriter.notifyMatchFailure(op, "failed to convert memory/result ABI decode result type");
        }
        if (convertedResultTypes.size() != 1 && convertedResultTypes.size() != 2)
            return rewriter.notifyMatchFailure(op, "unsupported memory/result ABI decode carrier shape");

        auto ptrType = sir::PtrType::get(op.getContext(), /*addrSpace*/ 1);
        Value required = staticDecodeRequiredLengthConst(rewriter, op.getLoc(), root);
        Value tooShort = rewriter.create<sir::LtOp>(op.getLoc(), u256Type, length, required);

        Block *parentBlock = op->getBlock();
        Region *parentRegion = parentBlock->getParent();
        auto mergeBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
        for (Type type : convertedResultTypes)
            mergeBlock->addArgument(type, op.getLoc());
        auto shortBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
        auto decodeBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
        AbiDecodeResultSink resultSink{rewriter, op, op.getLoc(), convertedResultTypes, mergeBlock};
        auto materializeMemoryResultErrorPayload = [&](Value errorPayload) -> Value {
            if (convertedResultTypes.size() == 2 && llvm::isa<sir::PtrType>(convertedResultTypes[1]) &&
                !llvm::isa<sir::PtrType>(errorPayload.getType()))
            {
                Value ptr = rewriter.create<sir::MallocOp>(op.getLoc(), convertedResultTypes[1], constU256(rewriter, op.getLoc(), 32));
                rewriter.create<sir::StoreOp>(op.getLoc(), ptr, coerceToU256(rewriter, op.getLoc(), errorPayload));
                return ptr;
            }
            if (convertedResultTypes.size() == 2 && errorPayload.getType() != convertedResultTypes[1])
                return rewriter.create<sir::BitcastOp>(op.getLoc(), convertedResultTypes[1], errorPayload);
            return errorPayload;
        };

        rewriter.setInsertionPointToEnd(parentBlock);
        rewriter.create<sir::CondBrOp>(op.getLoc(), tooShort, ValueRange{}, ValueRange{}, shortBlock, decodeBlock);

        rewriter.setInsertionPointToStart(shortBlock);
        if (resultSink.isNarrow())
        {
            Value truncated = packNarrowAbiDecodeResult(
                rewriter,
                op.getLoc(),
                abiDecodeErrorValue(rewriter, op.getLoc(), AbiDecodeError::TruncatedBuffer),
                /*isError=*/true);
            if (failed(resultSink.branchNarrow(truncated)))
                return failure();
        }
        else if (resultSink.isWide())
        {
            if (failed(resultSink.branchErrorKind(AbiDecodeError::TruncatedBuffer, materializeMemoryResultErrorPayload)))
                return failure();
        }

        rewriter.setInsertionPointToStart(decodeBlock);
        Value payloadPtr = rewriter.create<sir::AddPtrOp>(op.getLoc(), ptrType, bytesPtr, constU256(rewriter, op.getLoc(), 32));

        auto branchNarrow = [&](Value value) -> LogicalResult {
            if (resultSink.isNarrow())
            {
                Value checked = permissive
                                    ? value
                                    : applyStaticOversizeCheckToNarrowResult(
                                          rewriter,
                                          op.getLoc(),
                                          value,
                                          length,
                                          required);
                if (failed(resultSink.branchNarrow(checked)))
                    return failure();
                return resultSink.finish();
            }
            if (!resultSink.isWide())
                return rewriter.notifyMatchFailure(op, "narrow ABI decode produced an unsupported carrier");

            WideAbiDecodeResult split = splitPackedAbiDecodeResult(rewriter, op.getLoc(), value);
            WideAbiDecodeResult checked = permissive
                                              ? split
                                              : applyStaticOversizeCheckToWideResult(
                                                    rewriter,
                                                    op.getLoc(),
                                                    split,
                                                    length,
                                                    required,
                                                    convertedResultTypes[1]);
            if (failed(resultSink.branchWide(checked)))
                return failure();
            return resultSink.finish();
        };

        auto emitWideResultBranch = [&](WideAbiDecodeResult value, Type payloadCarrierType, Value requiredBytes) -> LogicalResult {
            if (convertedResultTypes.size() != 2)
                return rewriter.notifyMatchFailure(op, "wide ABI decode produced a non-wide carrier");
            if (llvm::isa<sir::PtrType>(payloadCarrierType))
            {
                auto sourceInsertion = rewriter.saveInsertionPoint();
                auto successBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
                auto errorBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
                rewriter.restoreInsertionPoint(sourceInsertion);
                Value isOk = rewriter.create<sir::EqOp>(op.getLoc(), u256Type, value.tag, constU256(rewriter, op.getLoc(), 0));
                rewriter.create<sir::CondBrOp>(op.getLoc(), isOk, ValueRange{}, ValueRange{}, successBlock, errorBlock);

                rewriter.setInsertionPointToStart(errorBlock);
                if (failed(resultSink.branchErrorPayload(coerceToU256(rewriter, op.getLoc(), value.payload), materializeMemoryResultErrorPayload)))
                    return failure();

                rewriter.setInsertionPointToStart(successBlock);
                if (permissive)
                    return resultSink.branchWide(WideAbiDecodeResult{constU256(rewriter, op.getLoc(), 0), value.payload});

                auto successInsertion = rewriter.saveInsertionPoint();
                auto oversizeBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
                auto okBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
                rewriter.restoreInsertionPoint(successInsertion);
                Value tooLong = rewriter.create<sir::GtOp>(op.getLoc(), u256Type, length, requiredBytes);
                rewriter.create<sir::CondBrOp>(op.getLoc(), tooLong, ValueRange{}, ValueRange{}, oversizeBlock, okBlock);

                rewriter.setInsertionPointToStart(oversizeBlock);
                if (failed(resultSink.branchErrorKind(AbiDecodeError::OversizeBuffer, materializeMemoryResultErrorPayload)))
                    return failure();

                rewriter.setInsertionPointToStart(okBlock);
                return resultSink.branchWide(WideAbiDecodeResult{constU256(rewriter, op.getLoc(), 0), value.payload});
            }
            auto checked = permissive
                               ? value
                               : applyStaticOversizeCheckToWideResult(
                                     rewriter,
                                     op.getLoc(),
                                     value,
                                     length,
                                     requiredBytes,
                                     payloadCarrierType);
            return resultSink.branchWide(checked);
        };

        auto branchWide = [&](WideAbiDecodeResult value, Type payloadCarrierType) -> LogicalResult {
            if (failed(emitWideResultBranch(value, payloadCarrierType, required)))
                return failure();
            return resultSink.finish();
        };

        auto emitWideValidationResultBranch = [&](Value valid,
                                                  AbiDecodeError invalidError,
                                                  Value requiredDynamic,
                                                  Type payloadCarrierType,
                                                  auto successPayloadBits) -> LogicalResult {
            if (llvm::isa<sir::PtrType>(payloadCarrierType))
            {
                auto sourceInsertion = rewriter.saveInsertionPoint();
                auto okBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
                auto invalidBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
                rewriter.restoreInsertionPoint(sourceInsertion);
                rewriter.create<sir::CondBrOp>(op.getLoc(), valid, ValueRange{}, ValueRange{}, okBlock, invalidBlock);

                rewriter.setInsertionPointToStart(invalidBlock);
                if (failed(resultSink.branchErrorKind(invalidError, materializeMemoryResultErrorPayload)))
                    return failure();

                rewriter.setInsertionPointToStart(okBlock);
                return emitWideResultBranch(
                    WideAbiDecodeResult{constU256(rewriter, op.getLoc(), 0), successPayloadBits()},
                    payloadCarrierType,
                    requiredDynamic);
            }

            auto sourceInsertion = rewriter.saveInsertionPoint();
            auto okBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            auto invalidBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            auto resultBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            resultBlock->addArgument(u256Type, op.getLoc());
            resultBlock->addArgument(u256Type, op.getLoc());
            rewriter.restoreInsertionPoint(sourceInsertion);
            rewriter.create<sir::CondBrOp>(op.getLoc(), valid, ValueRange{}, ValueRange{}, okBlock, invalidBlock);

            rewriter.setInsertionPointToStart(invalidBlock);
            Value errorValue = abiDecodeErrorValue(rewriter, op.getLoc(), invalidError);
            rewriter.create<sir::BrOp>(op.getLoc(), ValueRange{constU256(rewriter, op.getLoc(), 1), errorValue}, resultBlock);

            rewriter.setInsertionPointToStart(okBlock);
            Value payloadBits = coerceToU256(rewriter, op.getLoc(), successPayloadBits());
            rewriter.create<sir::BrOp>(op.getLoc(), ValueRange{constU256(rewriter, op.getLoc(), 0), payloadBits}, resultBlock);

            rewriter.setInsertionPointToStart(resultBlock);
            return emitWideResultBranch(
                WideAbiDecodeResult{resultBlock->getArgument(0), resultBlock->getArgument(1)},
                payloadCarrierType,
                requiredDynamic);
        };

        auto buildMixedDynamicTupleCarrier = [&](Value firstWord, Value tailPtr) -> Value {
            Value tupleSize = constU256(rewriter, op.getLoc(), 64);
            Value tuplePtr = rewriter.create<sir::MallocOp>(op.getLoc(), ptrType, tupleSize);
            rewriter.create<sir::StoreOp>(op.getLoc(), tuplePtr, firstWord);
            Value tailSlot = rewriter.create<sir::AddPtrOp>(op.getLoc(), ptrType, tuplePtr, constU256(rewriter, op.getLoc(), 32));
            rewriter.create<sir::StoreOp>(op.getLoc(), tailSlot, coerceToU256(rewriter, op.getLoc(), tailPtr));
            return tuplePtr;
        };

        auto computeWordArrayRequiredBytes = [&]() {
            return [&](Value, Value elementLen, Value requiredBaseBytes) -> Value {
                Value elementBytes = mulU256(rewriter, op.getLoc(), elementLen, constU256(rewriter, op.getLoc(), 32));
                return addU256(rewriter, op.getLoc(), requiredBaseBytes, elementBytes);
            };
        };

        auto emitValidatedWordArrayTail = [&](Value tailPtr,
                                              Value elementLen,
                                              Value requiredDynamic,
                                              Type payloadCarrierType,
                                              AbiDecodeError invalidElementError,
                                              auto elementCanonical,
                                              auto successPayloadBits) -> Block * {
            AbiValidationLoopBlocks validation = emitWordArrayElementValidationLoop(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                ptrType,
                u256Type,
                tailPtr,
                elementLen,
                elementCanonical);

            rewriter.setInsertionPointToStart(validation.done);
            Value elementsValid = validation.done->getArgument(0);
            if (failed(emitWideValidationResultBranch(
                    elementsValid,
                    invalidElementError,
                    requiredDynamic,
                    payloadCarrierType,
                    [&]() -> Value { return successPayloadBits(tailPtr); })))
                return nullptr;

            return validation.entry;
        };

        auto emitDecodedFixedBytesArrayTail = [&](Value tailPtr,
                                                  Value elementLen,
                                                  Value requiredDynamic,
                                                  Type payloadCarrierType,
                                                  const AbiLayoutNode &elementNode,
                                                  auto successPayloadBits) -> Block * {
            Value sourceContentPtr = rewriter.create<sir::AddPtrOp>(op.getLoc(), ptrType, tailPtr, constU256(rewriter, op.getLoc(), 32));

            AbiValidationLoopBlocks validation = emitFixedBytesArrayValidationLoop(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                ptrType,
                u256Type,
                tailPtr,
                elementLen,
                elementNode);

            FixedBytesArrayCopyLoop copy = emitFixedBytesArrayDecodeCopyLoop(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                ptrType,
                u256Type,
                sourceContentPtr,
                elementLen,
                elementNode);
            auto invalidBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());

            rewriter.setInsertionPointToStart(validation.done);
            Value elementsValid = validation.done->getArgument(0);
            rewriter.create<sir::CondBrOp>(op.getLoc(), elementsValid, ValueRange{}, ValueRange{}, copy.entry, invalidBlock);

            rewriter.setInsertionPointToStart(invalidBlock);
            Value errorPayload = abiDecodeErrorValue(rewriter, op.getLoc(), AbiDecodeError::InvalidFixedBytes);
            if (!llvm::isa<sir::PtrType>(payloadCarrierType) && errorPayload.getType() != payloadCarrierType)
                errorPayload = rewriter.create<sir::BitcastOp>(op.getLoc(), payloadCarrierType, errorPayload);
            if (failed(emitWideResultBranch(
                    WideAbiDecodeResult{constU256(rewriter, op.getLoc(), 1), errorPayload},
                    payloadCarrierType,
                    requiredDynamic)))
                return nullptr;

            rewriter.setInsertionPointToStart(copy.done);
            Value payloadBits = successPayloadBits(copy.resultPtr);
            if (!llvm::isa<sir::PtrType>(payloadCarrierType) && payloadBits.getType() != payloadCarrierType)
                payloadBits = rewriter.create<sir::BitcastOp>(op.getLoc(), payloadCarrierType, payloadBits);
            if (failed(emitWideResultBranch(
                    WideAbiDecodeResult{constU256(rewriter, op.getLoc(), 0), payloadBits},
                    payloadCarrierType,
                    requiredDynamic)))
                return nullptr;

            return validation.entry;
        };

        auto emitPaddingValidatedBytesTail = [&](Value tailPtr,
                                                 Value dynamicLen,
                                                 Value padded,
                                                 Value requiredDynamic,
                                                 Type payloadCarrierType,
                                                 auto successPayloadBits) -> Block * {
            if (permissive)
            {
                auto okBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
                rewriter.setInsertionPointToStart(okBlock);
                Value payloadBits = successPayloadBits(tailPtr);
                if (!llvm::isa<sir::PtrType>(payloadCarrierType))
                    payloadBits = coerceToU256(rewriter, op.getLoc(), payloadBits);
                if (failed(emitWideResultBranch(
                        WideAbiDecodeResult{constU256(rewriter, op.getLoc(), 0), payloadBits},
                        payloadCarrierType,
                        requiredDynamic)))
                    return nullptr;
                return okBlock;
            }

            AbiValidationLoopBlocks validation = emitDynamicBytesPaddingValidationLoop(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                ptrType,
                u256Type,
                tailPtr,
                dynamicLen,
                padded);

            rewriter.setInsertionPointToStart(validation.done);
            Value paddingValid = validation.done->getArgument(0);
            if (failed(emitWideValidationResultBranch(
                    paddingValid,
                    AbiDecodeError::NonCanonicalEncoding,
                    requiredDynamic,
                    payloadCarrierType,
                    [&]() -> Value { return successPayloadBits(tailPtr); })))
                return nullptr;

            return validation.entry;
        };

        auto emitDynamicDecodeBounds = [&](
                                           uint64_t offsetWordByteOffset,
                                           uint64_t expectedOffset,
                                           uint64_t minBytes,
                                           uint64_t capValue,
                                           AbiDecodeError capError,
                                           AbiDecodeError tailTooShortError,
                                           auto computeRequiredBytes,
                                           auto emitAfterBounds,
                                           auto branchDynamicErr) -> LogicalResult {
            // `payloadPtr` is defined in decodeBlock immediately after the
            // static short-buffer guard. The shared bounds helper appends all
            // dynamic offset/length reads to decodeBlock so every payload load
            // is dominated by that guard.
            return emitStrictDynamicDecodeBounds(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                decodeBlock,
                payloadPtr,
                length,
                ptrType,
                u256Type,
                offsetWordByteOffset,
                expectedOffset,
                minBytes,
                capValue,
                capError,
                tailTooShortError,
                branchDynamicErr,
                computeRequiredBytes,
                emitAfterBounds,
                [&]() -> LogicalResult { return resultSink.finish(); },
                permissive);
        };

        // Dynamic memory/result decode is driven by the current LayoutNode and
        // its corresponding semantic type. Single returns and admitted
        // `(u256, dynamic)` tuples call the same node walker; the caller only
        // supplies how a validated tail becomes the final Result payload.
        auto emitDynamicMemoryNode = [&](
                                         const AbiLayoutNode &node,
                                         Type nodeSuccessType,
                                         Type carrierSuccessType,
                                         uint64_t offsetWordByteOffset,
                                         uint64_t expectedOffset,
                                         uint64_t minBytes,
                                         auto successPayloadBits) -> std::optional<LogicalResult> {
            if (node.kind != AbiLayoutKind::DynamicBytes &&
                node.kind != AbiLayoutKind::DynamicArray)
            {
                return std::nullopt;
            }
            if (!euh::shouldUseWideErrorUnionCarrier(resultType, op.getOperation()) || convertedResultTypes.size() != 2)
                return failure();

            Type payloadCarrierType = euh::getWideErrorUnionCarrierType(op.getContext(), carrierSuccessType);
            auto branchDynamicErr = [&, payloadCarrierType](AbiDecodeError err, Value requiredBytes) -> LogicalResult {
                return emitWideResultBranch(
                    WideAbiDecodeResult{
                        constU256(rewriter, op.getLoc(), 1),
                        abiDecodeErrorValue(rewriter, op.getLoc(), err),
                    },
                    payloadCarrierType,
                    requiredBytes);
            };

            if (node.kind == AbiLayoutKind::DynamicBytes && isDynamicBytesSuccessType(nodeSuccessType))
            {
                Value padded;
                auto computeRequiredBytes = [&](Value, Value dynamicLen, Value requiredBaseBytes) -> Value {
                    padded = ceil32(rewriter, op.getLoc(), dynamicLen);
                    return addU256(rewriter, op.getLoc(), requiredBaseBytes, padded);
                };
                auto emitAfterBounds = [&](Value tailPtr, Value dynamicLen, Value requiredDynamic) -> Block * {
                    return emitPaddingValidatedBytesTail(
                        tailPtr,
                        dynamicLen,
                        padded,
                        requiredDynamic,
                        payloadCarrierType,
                        successPayloadBits);
                };
                return emitDynamicDecodeBounds(
                    offsetWordByteOffset,
                    expectedOffset,
                    minBytes,
                    /*capValue=*/1024 * 1024,
                    AbiDecodeError::StringLengthExceeded,
                    AbiDecodeError::TruncatedBuffer,
                    computeRequiredBytes,
                    emitAfterBounds,
                    branchDynamicErr);
            }

            auto emitDynamicArrayDecodeBounds = [&](auto emitAfterBounds) -> LogicalResult {
                auto computeRequiredBytes = computeWordArrayRequiredBytes();
                return emitDynamicDecodeBounds(
                    offsetWordByteOffset,
                    expectedOffset,
                    minBytes,
                    /*capValue=*/32768,
                    AbiDecodeError::ArrayLengthExceeded,
                    AbiDecodeError::TruncatedBuffer,
                    computeRequiredBytes,
                    emitAfterBounds,
                    branchDynamicErr);
            };

            if (isDynamicU256ArrayAbiNode(node) && isDynamicU256MemRefSuccessType(nodeSuccessType))
            {
                auto emitAfterBounds = [&](Value tailPtr, Value, Value requiredDynamic) -> Block * {
                    auto okBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
                    rewriter.setInsertionPointToStart(okBlock);
                    Value payload = successPayloadBits(tailPtr);
                    if (!llvm::isa<sir::PtrType>(payloadCarrierType))
                        payload = coerceToU256(rewriter, op.getLoc(), payload);
                    if (!llvm::isa<sir::PtrType>(payloadCarrierType) && payload.getType() != payloadCarrierType)
                        payload = rewriter.create<sir::BitcastOp>(op.getLoc(), payloadCarrierType, payload);
                    if (failed(emitWideResultBranch(WideAbiDecodeResult{constU256(rewriter, op.getLoc(), 0), payload}, payloadCarrierType, requiredDynamic)))
                        return nullptr;
                    return okBlock;
                };
                return emitDynamicArrayDecodeBounds(emitAfterBounds);
            }

            if (isDynamicAddressArrayAbiNode(node) && isDynamicAddressMemRefSuccessType(nodeSuccessType))
            {
                auto emitAfterBounds = [&](Value tailPtr, Value elementLen, Value requiredDynamic) -> Block * {
                    return emitValidatedWordArrayTail(
                        tailPtr,
                        elementLen,
                        requiredDynamic,
                        payloadCarrierType,
                        AbiDecodeError::InvalidAddress,
                        [&](Value elementWord) -> Value {
                            Value elementPayload = maskLowBits(rewriter, op.getLoc(), elementWord, 160);
                            return rewriter.create<sir::EqOp>(op.getLoc(), u256Type, elementWord, elementPayload);
                        },
                        successPayloadBits);
                };
                return emitDynamicArrayDecodeBounds(emitAfterBounds);
            }

            if (isDynamicBoolArrayAbiNode(node) && isDynamicBoolMemRefSuccessType(nodeSuccessType))
            {
                auto emitAfterBounds = [&](Value tailPtr, Value elementLen, Value requiredDynamic) -> Block * {
                    return emitValidatedWordArrayTail(
                        tailPtr,
                        elementLen,
                        requiredDynamic,
                        payloadCarrierType,
                        AbiDecodeError::InvalidBoolValue,
                        [&](Value elementWord) -> Value {
                            return boolAbiWordIsCanonical(rewriter, op.getLoc(), elementWord);
                        },
                        successPayloadBits);
                };
                return emitDynamicArrayDecodeBounds(emitAfterBounds);
            }

            if (isDynamicFixedBytesArrayAbiNode(node) && isDynamicFixedBytesMemRefSuccessType(nodeSuccessType))
            {
                const AbiLayoutNode &elementNode = *node.children.front();
                auto emitAfterBounds = [&](Value tailPtr, Value elementLen, Value requiredDynamic) -> Block * {
                    return emitDecodedFixedBytesArrayTail(
                        tailPtr,
                        elementLen,
                        requiredDynamic,
                        payloadCarrierType,
                        elementNode,
                        successPayloadBits);
                };
                return emitDynamicArrayDecodeBounds(emitAfterBounds);
            }

            return std::nullopt;
        };

        // A single ABI return value decodes to the value itself. A source-level
        // tuple target decodes to an Ora tuple, matching comptime
        // topLevelWrapsSingleArgument behavior.
        if (root.children.size() == 1)
        {
            if (auto dynamicResult = emitDynamicMemoryNode(
                    *root.children.front(),
                    successType,
                    successType,
                    /*offsetWordByteOffset=*/0,
                    /*expectedOffset=*/32,
                    /*minBytes=*/64,
                    [&](Value validTailPtr) -> Value {
                        return validTailPtr;
                    }))
            {
                if (failed(*dynamicResult))
                    return rewriter.notifyMatchFailure(op, "dynamic ABI decode requires a supported wide error-union carrier");
                return *dynamicResult;
            }
            if (root.children.front()->kind == AbiLayoutKind::DynamicArray)
                return rewriter.notifyMatchFailure(op, "dynamic array ABI decode requires a supported static-element slice target");

            if (Value narrowResult = lowerNarrowScalarMemoryResult(
                    rewriter,
                    op.getLoc(),
                    *root.children.front(),
                    successType,
                    op.getOperation(),
                    payloadPtr,
                    /*byteOffset=*/0,
                    permissive))
            {
                return branchNarrow(narrowResult);
            }
            if (isEnumSuccessType(successType))
                return rewriter.notifyMatchFailure(op, "enum ABI decode requires enum variant metadata");
            if (auto u256BackedNarrowResult = lowerU256BackedNarrowUintMemoryResult(
                    rewriter,
                    op.getLoc(),
                    *root.children.front(),
                    successType,
                    euh::shouldUseWideErrorUnionCarrier(resultType, op.getOperation()),
                    payloadPtr,
                    /*byteOffset=*/0,
                    permissive))
            {
                return branchWide(*u256BackedNarrowResult, u256BackedNarrowResult->payload.getType());
            }
            if (auto fixedBytesResult = lowerFixedBytesMemoryResult(
                    rewriter,
                    op.getLoc(),
                    *root.children.front(),
                    successType,
                    euh::shouldUseWideErrorUnionCarrier(resultType, op.getOperation()),
                    payloadPtr,
                    /*byteOffset=*/0,
                    permissive))
            {
                return branchWide(*fixedBytesResult, fixedBytesResult->payload.getType());
            }
            if (isStaticFixedBytesAbiNode(*root.children.front()))
                return rewriter.notifyMatchFailure(op, "fixed-bytes ABI decode requires u256-backed success type and wide error-union carrier");
            if (!isStaticU256AbiNode(*root.children.front()) && !isStaticI256AbiNode(*root.children.front()))
                return rewriter.notifyMatchFailure(op, "memory/result scalar ABI decode requires a supported static scalar target");
            Value decoded = decodeAbiU256FromMemory(rewriter, op.getLoc(), payloadPtr, /*byteOffset=*/0);
            if (!euh::shouldUseWideErrorUnionCarrier(resultType, op.getOperation()))
                return rewriter.notifyMatchFailure(op, "full-word ABI decode requires wide error-union carrier");
            Type payloadCarrierType = euh::getWideErrorUnionCarrierType(op.getContext(), successType);
            if (Value refinementValid = abiDecodeRefinementSatisfied(
                    rewriter,
                    op.getLoc(),
                    successType,
                    decoded,
                root.children.front()->staticKind == AbiStaticKind::Int))
            {
                Value zero = constU256(rewriter, op.getLoc(), 0);
                Value one = constU256(rewriter, op.getLoc(), 1);
                Value errorValue = abiDecodeErrorValue(rewriter, op.getLoc(), AbiDecodeError::RefinementViolation);
                return branchWide(
                    WideAbiDecodeResult{
                        rewriter.create<sir::SelectOp>(op.getLoc(), u256Type, refinementValid, zero, one),
                        rewriter.create<sir::SelectOp>(op.getLoc(), u256Type, refinementValid, decoded, errorValue),
                    },
                    payloadCarrierType);
            }
            if (Type convertedSuccessType = typeConverter->convertType(successType))
                if (decoded.getType() != convertedSuccessType)
                    decoded = rewriter.create<sir::BitcastOp>(op.getLoc(), convertedSuccessType, decoded);
            return branchWide(WideAbiDecodeResult{constU256(rewriter, op.getLoc(), 0), decoded}, payloadCarrierType);
        }
        else
        {
            Type unwrappedSuccessType = abiDecodeUnwrapRefinementType(successType);
            if (root.children.size() == 2 && isStaticU256AbiNode(*root.children[0]))
            {
                auto tupleType = llvm::dyn_cast<ora::TupleType>(unwrappedSuccessType);
                if (tupleType && tupleType.getElementTypes().size() == 2 &&
                    successTypeIsU256Backed(tupleType.getElementTypes()[0]))
                {
                    if (auto dynamicResult = emitDynamicMemoryNode(
                            *root.children[1],
                            tupleType.getElementTypes()[1],
                            successType,
                            /*offsetWordByteOffset=*/32,
                            /*expectedOffset=*/64,
                            /*minBytes=*/96,
                            [&](Value validTailPtr) -> Value {
                                Value firstWord = decodeAbiU256FromMemory(rewriter, op.getLoc(), payloadPtr, /*byteOffset=*/0);
                                return buildMixedDynamicTupleCarrier(firstWord, validTailPtr);
                            }))
                    {
                        if (failed(*dynamicResult))
                            return rewriter.notifyMatchFailure(op, "mixed dynamic tuple ABI decode requires a supported wide error-union carrier");
                        return *dynamicResult;
                    }
                }
            }

            auto decodedTuple = lowerStaticMemoryDecodePart(
                rewriter,
                op.getLoc(),
                root,
                successType,
                op.getOperation(),
                payloadPtr,
                /*byteOffset=*/0,
                permissive);
            if (!decodedTuple)
                return rewriter.notifyMatchFailure(op, "memory/result tuple ABI decode requires a supported static tuple target");
            if (!euh::shouldUseWideErrorUnionCarrier(resultType, op.getOperation()))
                return rewriter.notifyMatchFailure(op, "memory/result tuple ABI decode requires wide error-union carrier");

            auto u256Type = sir::U256Type::get(op.getContext());
            Value zero = constU256(rewriter, op.getLoc(), 0);
            Value one = constU256(rewriter, op.getLoc(), 1);
            Type payloadCarrierType = euh::getWideErrorUnionCarrierType(op.getContext(), successType);
            Value okPayload = decodedTuple->value;
            if (auto materialized = ora::materializePtrCarrierFromOraValue(rewriter, op.getLoc(), payloadCarrierType, okPayload))
                okPayload = *materialized;
            Value okPayloadBits = coerceToU256(rewriter, op.getLoc(), okPayload);
            // Wide Result consumers must inspect the tag first. The selected
            // Err payload bits are an AbiDecodeError ordinal, not a success value.
            Value payloadBits = rewriter.create<sir::SelectOp>(op.getLoc(), u256Type, decodedTuple->valid, okPayloadBits, decodedTuple->error);
            Value payload = payloadBits;
            if (payloadCarrierType && payload.getType() != payloadCarrierType)
                payload = rewriter.create<sir::BitcastOp>(op.getLoc(), payloadCarrierType, payload);
            Value tag = rewriter.create<sir::SelectOp>(op.getLoc(), u256Type, decodedTuple->valid, zero, one);
            return branchWide(WideAbiDecodeResult{tag, payload}, payloadCarrierType);
        }

        return rewriter.notifyMatchFailure(op, "unreachable memory/result ABI decode branch");
    }

    // The dialect verifier accepts all planned decode sources and failure
    // modes. The legacy lowering below still implements only external-call
    // returndata decode mapped into the existing error-union path.
    if (sourceAttr.getValue() != "returndata")
        return rewriter.notifyMatchFailure(op, "only returndata ABI decode is supported");
    if (failureModeAttr.getValue() != "error_union")
        return rewriter.notifyMatchFailure(op, "only error-union ABI decode failure mode is supported");

    auto *ctx = op.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    Value returndataPtr = rewriter.create<sir::BitcastOp>(op.getLoc(), ptrType, adaptor.getReturndata());

    if (op->getAttrOfType<mlir::StringAttr>("returndata_failure_error"))
    {
        // N3c strict external-return decode: validate strict ABI shape before
        // exposing a returned value. Decode failures preserve the same
        // AbiDecodeError ordinal produced by user-facing memory/result decode;
        // the surrounding external-call error-union surface carries that
        // category for diagnostics.
        auto resultType = llvm::dyn_cast<ora::ErrorUnionType>(op.getResult().getType());
        if (!resultType)
            return rewriter.notifyMatchFailure(op, "strict returndata ABI decode requires error-union result type");
        if (root.kind != AbiLayoutKind::Tuple || root.children.empty())
            return rewriter.notifyMatchFailure(op, "strict returndata ABI decode currently requires a non-empty tuple layout");

        SmallVector<Type> convertedResultTypes;
        if (failed(getErrorUnionEncodingTypes(typeConverter, op.getResult().getType(), op.getOperation(), convertedResultTypes)) ||
            convertedResultTypes.empty())
        {
            return rewriter.notifyMatchFailure(op, "failed to convert strict returndata ABI decode result type");
        }
        if (convertedResultTypes.size() != 1 && convertedResultTypes.size() != 2)
            return rewriter.notifyMatchFailure(op, "unsupported strict returndata ABI decode carrier shape");

        Value length = rewriter.create<sir::ReturnDataSizeOp>(op.getLoc(), u256Type);

        Block *parentBlock = op->getBlock();
        Region *parentRegion = parentBlock->getParent();
        auto mergeBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
        for (Type type : convertedResultTypes)
            mergeBlock->addArgument(type, op.getLoc());
        auto errorBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
        auto decodeBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
        AbiDecodeResultSink resultSink{rewriter, op, op.getLoc(), convertedResultTypes, mergeBlock};

        auto materializeReturndataErrorPayload = [&](Value errorPayload) -> Value {
            Value payload = errorPayload;
            if (llvm::isa<sir::PtrType>(convertedResultTypes[1]) && !llvm::isa<sir::PtrType>(payload.getType()))
            {
                Value size = constU256(rewriter, op.getLoc(), 32);
                Value ptr = rewriter.create<sir::MallocOp>(op.getLoc(), convertedResultTypes[1], size);
                rewriter.create<sir::StoreOp>(op.getLoc(), ptr, coerceToU256(rewriter, op.getLoc(), payload));
                payload = ptr;
            }
            else if (payload.getType() != convertedResultTypes[1])
            {
                payload = rewriter.create<sir::BitcastOp>(op.getLoc(), convertedResultTypes[1], payload);
            }
            return payload;
        };

        Type successType = resultType.getSuccessType();
        auto branchDecodeOk = [&](Block *block, Value payload) -> LogicalResult {
            rewriter.setInsertionPointToEnd(block);
            if (!resultSink.isWide())
                return failure();
            return resultSink.branchOkPayload(payload);
        };

        auto buildMixedReturndataTupleCarrier = [&](Value firstWord, Value tailPtr) -> Value {
            Value tupleSize = constU256(rewriter, op.getLoc(), 64);
            Value tuplePtr = rewriter.create<sir::MallocOp>(op.getLoc(), ptrType, tupleSize);
            rewriter.create<sir::StoreOp>(op.getLoc(), tuplePtr, firstWord);
            Value tailSlot = rewriter.create<sir::AddPtrOp>(op.getLoc(), ptrType, tuplePtr, constU256(rewriter, op.getLoc(), 32));
            rewriter.create<sir::StoreOp>(op.getLoc(), tailSlot, coerceToU256(rewriter, op.getLoc(), tailPtr));
            return coerceToU256(rewriter, op.getLoc(), tuplePtr);
        };

        auto computeReturndataWordArrayRequiredBytes = [&]() {
            return [&](Value, Value elementLen, Value requiredBaseBytes) -> Value {
                Value elementBytes = mulU256(rewriter, op.getLoc(), elementLen, constU256(rewriter, op.getLoc(), 32));
                return addU256(rewriter, op.getLoc(), requiredBaseBytes, elementBytes);
            };
        };

        auto emitReturndataOkAfterOversize = [&](Value requiredDynamic, auto successPayloadBits) -> Block * {
            auto oversizeCheckBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            auto oversizeBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            auto okBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());

            rewriter.setInsertionPointToStart(oversizeCheckBlock);
            Value tooLong = rewriter.create<sir::GtOp>(op.getLoc(), u256Type, length, requiredDynamic);
            rewriter.create<sir::CondBrOp>(op.getLoc(), tooLong, ValueRange{}, ValueRange{}, oversizeBlock, okBlock);

            rewriter.setInsertionPointToStart(oversizeBlock);
            if (failed(resultSink.branchErrorKind(AbiDecodeError::OversizeBuffer, materializeReturndataErrorPayload)))
                return nullptr;

            rewriter.setInsertionPointToStart(okBlock);
            if (failed(branchDecodeOk(okBlock, successPayloadBits())))
                return nullptr;

            return oversizeCheckBlock;
        };

        auto emitReturndataPaddingValidatedTail = [&](Value tailPtr,
                                                      Value dynamicLen,
                                                      Value padded,
                                                      Value requiredDynamic,
                                                      auto successPayloadBits) -> Block * {
            auto validation = emitDynamicBytesPaddingValidationLoop(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                ptrType,
                u256Type,
                tailPtr,
                dynamicLen,
                padded);

            auto paddingErrorBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            Block *okAfterOversize = emitReturndataOkAfterOversize(requiredDynamic, [&]() -> Value {
                return successPayloadBits(tailPtr);
            });
            if (!okAfterOversize)
                return nullptr;

            rewriter.setInsertionPointToStart(validation.done);
            Value paddingValid = validation.done->getArgument(0);
            rewriter.create<sir::CondBrOp>(op.getLoc(), paddingValid, ValueRange{}, ValueRange{}, okAfterOversize, paddingErrorBlock);

            rewriter.setInsertionPointToStart(paddingErrorBlock);
            if (failed(resultSink.branchErrorKind(AbiDecodeError::NonCanonicalEncoding, materializeReturndataErrorPayload)))
                return nullptr;

            return validation.entry;
        };

        auto emitReturndataValidatedWordArrayTail = [&](Value tailPtr,
                                                        Value elementLen,
                                                        Value requiredDynamic,
                                                        AbiDecodeError invalidElementError,
                                                        auto elementCanonical,
                                                        auto successPayloadBits) -> Block * {
            auto validation = emitWordArrayElementValidationLoop(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                ptrType,
                u256Type,
                tailPtr,
                elementLen,
                elementCanonical);

            auto invalidBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            Block *okAfterOversize = emitReturndataOkAfterOversize(requiredDynamic, [&]() -> Value {
                return successPayloadBits(tailPtr);
            });
            if (!okAfterOversize)
                return nullptr;

            rewriter.setInsertionPointToStart(validation.done);
            Value elementsValid = validation.done->getArgument(0);
            rewriter.create<sir::CondBrOp>(op.getLoc(), elementsValid, ValueRange{}, ValueRange{}, okAfterOversize, invalidBlock);

            rewriter.setInsertionPointToStart(invalidBlock);
            if (failed(resultSink.branchErrorKind(invalidElementError, materializeReturndataErrorPayload)))
                return nullptr;

            return validation.entry;
        };

        auto emitReturndataDecodedFixedBytesArrayTail = [&](Value tailPtr,
                                                            Value elementLen,
                                                            Value requiredDynamic,
                                                            const AbiLayoutNode &elementNode,
                                                            auto successPayloadBits) -> Block * {
            Value sourceContentPtr = rewriter.create<sir::AddPtrOp>(op.getLoc(), ptrType, tailPtr, constU256(rewriter, op.getLoc(), 32));
            auto validation = emitFixedBytesArrayValidationLoop(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                ptrType,
                u256Type,
                tailPtr,
                elementLen,
                elementNode);

            auto invalidBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            auto oversizeCheckBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            auto oversizeBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            FixedBytesArrayCopyLoop copy = emitFixedBytesArrayDecodeCopyLoop(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                ptrType,
                u256Type,
                sourceContentPtr,
                elementLen,
                elementNode);

            rewriter.setInsertionPointToStart(validation.done);
            Value elementsValid = validation.done->getArgument(0);
            rewriter.create<sir::CondBrOp>(op.getLoc(), elementsValid, ValueRange{}, ValueRange{}, oversizeCheckBlock, invalidBlock);

            rewriter.setInsertionPointToStart(invalidBlock);
            if (failed(resultSink.branchErrorKind(AbiDecodeError::InvalidFixedBytes, materializeReturndataErrorPayload)))
                return nullptr;

            rewriter.setInsertionPointToStart(oversizeCheckBlock);
            Value tooLong = rewriter.create<sir::GtOp>(op.getLoc(), u256Type, length, requiredDynamic);
            rewriter.create<sir::CondBrOp>(op.getLoc(), tooLong, ValueRange{}, ValueRange{}, oversizeBlock, copy.entry);

            rewriter.setInsertionPointToStart(oversizeBlock);
            if (failed(resultSink.branchErrorKind(AbiDecodeError::OversizeBuffer, materializeReturndataErrorPayload)))
                return nullptr;

            rewriter.setInsertionPointToStart(copy.done);
            if (failed(branchDecodeOk(copy.done, successPayloadBits(copy.resultPtr))))
                return nullptr;

            return validation.entry;
        };

        auto emitStrictReturndataDynamicBounds = [&](uint64_t offsetWordByteOffset,
                                                     uint64_t expectedOffset,
                                                     uint64_t minBytes,
                                                     uint64_t capValue,
                                                     AbiDecodeError capError,
                                                     auto computeRequiredBytes,
                                                     auto emitAfterBounds) -> LogicalResult {
            rewriter.setInsertionPointToEnd(parentBlock);
            Value headRequired = constU256(rewriter, op.getLoc(), root.headSlotBytes());
            Value headTooShort = rewriter.create<sir::LtOp>(op.getLoc(), u256Type, length, headRequired);
            rewriter.create<sir::CondBrOp>(op.getLoc(), headTooShort, ValueRange{}, ValueRange{}, errorBlock, decodeBlock);
            rewriter.setInsertionPointToStart(errorBlock);
            if (failed(resultSink.branchErrorKind(AbiDecodeError::TruncatedBuffer, materializeReturndataErrorPayload)))
                return failure();

            auto branchDynamicErr = [&](AbiDecodeError err, Value) -> LogicalResult {
                return resultSink.branchErrorKind(err, materializeReturndataErrorPayload);
            };

            return emitStrictDynamicDecodeBounds(
                rewriter,
                op.getLoc(),
                parentRegion,
                mergeBlock,
                decodeBlock,
                returndataPtr,
                length,
                ptrType,
                u256Type,
                offsetWordByteOffset,
                expectedOffset,
                minBytes,
                capValue,
                capError,
                AbiDecodeError::TruncatedBuffer,
                branchDynamicErr,
                computeRequiredBytes,
                emitAfterBounds,
                [&]() -> LogicalResult { return resultSink.finish(); },
                /*permissive=*/false);
        };

        auto emitStrictReturndataDynamicNode = [&](const AbiLayoutNode &node,
                                                   Type nodeSuccessType,
                                                   uint64_t offsetWordByteOffset,
                                                   uint64_t expectedOffset,
                                                   uint64_t minBytes,
                                                   auto successPayloadBits) -> std::optional<LogicalResult> {
            if (node.kind != AbiLayoutKind::DynamicBytes &&
                node.kind != AbiLayoutKind::DynamicArray)
            {
                return std::nullopt;
            }
            if (convertedResultTypes.size() != 2)
                return failure();

            if (node.kind == AbiLayoutKind::DynamicBytes && isDynamicBytesSuccessType(nodeSuccessType))
            {
                Value padded;
                auto computeRequiredBytes = [&](Value, Value dynamicLen, Value requiredBaseBytes) -> Value {
                    padded = ceil32(rewriter, op.getLoc(), dynamicLen);
                    return addU256(rewriter, op.getLoc(), requiredBaseBytes, padded);
                };
                auto emitAfterBounds = [&](Value tailPtr, Value dynamicLen, Value requiredDynamic) -> Block * {
                    return emitReturndataPaddingValidatedTail(
                        tailPtr,
                        dynamicLen,
                        padded,
                        requiredDynamic,
                        successPayloadBits);
                };
                return emitStrictReturndataDynamicBounds(
                    offsetWordByteOffset,
                    expectedOffset,
                    minBytes,
                    /*capValue=*/1024 * 1024,
                    AbiDecodeError::StringLengthExceeded,
                    computeRequiredBytes,
                    emitAfterBounds);
            }

            auto emitDynamicArrayDecodeBounds = [&](auto emitAfterBounds) -> LogicalResult {
                auto computeRequiredBytes = computeReturndataWordArrayRequiredBytes();
                return emitStrictReturndataDynamicBounds(
                    offsetWordByteOffset,
                    expectedOffset,
                    minBytes,
                    /*capValue=*/32768,
                    AbiDecodeError::ArrayLengthExceeded,
                    computeRequiredBytes,
                    emitAfterBounds);
            };

            if (isDynamicU256ArrayAbiNode(node) && isDynamicU256MemRefSuccessType(nodeSuccessType))
            {
                auto emitAfterBounds = [&](Value tailPtr, Value, Value requiredDynamic) -> Block * {
                    return emitReturndataOkAfterOversize(requiredDynamic, [&]() -> Value {
                        return successPayloadBits(tailPtr);
                    });
                };
                return emitDynamicArrayDecodeBounds(emitAfterBounds);
            }

            if (isDynamicAddressArrayAbiNode(node) && isDynamicAddressMemRefSuccessType(nodeSuccessType))
            {
                auto emitAfterBounds = [&](Value tailPtr, Value elementLen, Value requiredDynamic) -> Block * {
                    return emitReturndataValidatedWordArrayTail(
                        tailPtr,
                        elementLen,
                        requiredDynamic,
                        AbiDecodeError::InvalidAddress,
                        [&](Value elementWord) -> Value {
                            Value elementPayload = maskLowBits(rewriter, op.getLoc(), elementWord, 160);
                            return rewriter.create<sir::EqOp>(op.getLoc(), u256Type, elementWord, elementPayload);
                        },
                        successPayloadBits);
                };
                return emitDynamicArrayDecodeBounds(emitAfterBounds);
            }

            if (isDynamicBoolArrayAbiNode(node) && isDynamicBoolMemRefSuccessType(nodeSuccessType))
            {
                auto emitAfterBounds = [&](Value tailPtr, Value elementLen, Value requiredDynamic) -> Block * {
                    return emitReturndataValidatedWordArrayTail(
                        tailPtr,
                        elementLen,
                        requiredDynamic,
                        AbiDecodeError::InvalidBoolValue,
                        [&](Value elementWord) -> Value {
                            return boolAbiWordIsCanonical(rewriter, op.getLoc(), elementWord);
                        },
                        successPayloadBits);
                };
                return emitDynamicArrayDecodeBounds(emitAfterBounds);
            }

            if (isDynamicFixedBytesArrayAbiNode(node) && isDynamicFixedBytesMemRefSuccessType(nodeSuccessType))
            {
                const AbiLayoutNode &elementNode = *node.children.front();
                auto emitAfterBounds = [&](Value tailPtr, Value elementLen, Value requiredDynamic) -> Block * {
                    return emitReturndataDecodedFixedBytesArrayTail(
                        tailPtr,
                        elementLen,
                        requiredDynamic,
                        elementNode,
                        successPayloadBits);
                };
                return emitDynamicArrayDecodeBounds(emitAfterBounds);
            }

            return std::nullopt;
        };

        if (root.children.size() == 1)
        {
            if (auto dynamicResult = emitStrictReturndataDynamicNode(
                    *root.children.front(),
                    successType,
                    /*offsetWordByteOffset=*/0,
                    /*expectedOffset=*/32,
                    /*minBytes=*/64,
                    [&](Value validTailPtr) -> Value {
                        return validTailPtr;
                    }))
            {
                if (failed(*dynamicResult))
                    return rewriter.notifyMatchFailure(op, "strict dynamic returndata decode requires a supported wide error-union carrier");
                return *dynamicResult;
            }
        }
        else if (root.children.size() == 2 && isStaticU256AbiNode(*root.children[0]))
        {
            Type unwrappedSuccessType = abiDecodeUnwrapRefinementType(successType);
            auto tupleType = llvm::dyn_cast<ora::TupleType>(unwrappedSuccessType);
            if (tupleType && tupleType.getElementTypes().size() == 2 &&
                successTypeIsU256Backed(tupleType.getElementTypes()[0]))
            {
                if (auto dynamicResult = emitStrictReturndataDynamicNode(
                        *root.children[1],
                        tupleType.getElementTypes()[1],
                        /*offsetWordByteOffset=*/32,
                        /*expectedOffset=*/64,
                        /*minBytes=*/96,
                        [&](Value validTailPtr) -> Value {
                            Value firstWord = decodeAbiU256FromMemory(rewriter, op.getLoc(), returndataPtr, /*byteOffset=*/0);
                            return buildMixedReturndataTupleCarrier(firstWord, validTailPtr);
                        }))
                {
                    if (failed(*dynamicResult))
                        return rewriter.notifyMatchFailure(op, "strict mixed dynamic returndata decode requires a supported wide error-union carrier");
                    return *dynamicResult;
                }
            }
        }

        rewriter.setInsertionPointToEnd(parentBlock);
        Value required = staticDecodeRequiredLengthConst(rewriter, op.getLoc(), root);
        Value tooShort = rewriter.create<sir::LtOp>(op.getLoc(), u256Type, length, required);
        Value tooLong = rewriter.create<sir::GtOp>(op.getLoc(), u256Type, length, required);
        Value badLength = rewriter.create<sir::OrOp>(op.getLoc(), u256Type, tooShort, tooLong);
        Value shortError = abiDecodeErrorValue(rewriter, op.getLoc(), AbiDecodeError::TruncatedBuffer);
        Value oversizeError = abiDecodeErrorValue(rewriter, op.getLoc(), AbiDecodeError::OversizeBuffer);
        Value lengthError = rewriter.create<sir::SelectOp>(op.getLoc(), u256Type, tooShort, shortError, oversizeError);

        rewriter.create<sir::CondBrOp>(op.getLoc(), badLength, ValueRange{}, ValueRange{}, errorBlock, decodeBlock);
        rewriter.setInsertionPointToStart(errorBlock);
        if (failed(resultSink.branchErrorPayload(lengthError, materializeReturndataErrorPayload)))
            return failure();

        rewriter.setInsertionPointToStart(decodeBlock);
        std::optional<StaticMemoryDecodePart> decoded;
        if (root.children.size() == 1)
        {
            decoded = lowerStaticMemoryDecodePart(
                rewriter,
                op.getLoc(),
                *root.children.front(),
                successType,
                op.getOperation(),
                returndataPtr,
                /*byteOffset=*/0,
                /*permissive=*/false);
        }
        else
        {
            decoded = lowerStaticMemoryDecodePart(
                rewriter,
                op.getLoc(),
                root,
                successType,
                op.getOperation(),
                returndataPtr,
                /*byteOffset=*/0,
                /*permissive=*/false);
        }
        if (!decoded)
            return rewriter.notifyMatchFailure(op, "strict returndata ABI decode requires a supported static return target");

        if (resultSink.isNarrow())
        {
            Value selected = packSelectedAbiDecodeResult(
                rewriter,
                op.getLoc(),
                decoded->valid,
                decoded->value,
                decoded->error);
            if (failed(resultSink.branchNarrow(selected)))
                return failure();
        }
        else
        {
            Value zero = constU256(rewriter, op.getLoc(), 0);
            Value one = constU256(rewriter, op.getLoc(), 1);
            Type payloadCarrierType = convertedResultTypes[1];
            Value okPayload = decoded->value;
            if (auto materialized = ora::materializePtrCarrierFromOraValue(rewriter, op.getLoc(), payloadCarrierType, okPayload))
                okPayload = *materialized;
            Value okPayloadBits = coerceToU256(rewriter, op.getLoc(), okPayload);
            Value errPayloadBits = coerceToU256(rewriter, op.getLoc(), decoded->error);
            Value tag = rewriter.create<sir::SelectOp>(op.getLoc(), u256Type, decoded->valid, zero, one);
            Value payload = rewriter.create<sir::SelectOp>(op.getLoc(), u256Type, decoded->valid, okPayloadBits, errPayloadBits);
            if (failed(resultSink.branchWide(WideAbiDecodeResult{tag, payload})))
                return failure();
        }

        return resultSink.finish();
    }

    Type origType = op.getResult().getType();
    if (llvm::isa<ora::StringType, ora::BytesType>(origType))
    {
        Type convertedType = typeConverter->convertType(origType);
        if (!convertedType)
            return rewriter.notifyMatchFailure(op, "failed to convert dynamic ABI return type");

        Value offset = rewriter.create<sir::LoadOp>(op.getLoc(), u256Type, returndataPtr);
        Value dataPtr = rewriter.create<sir::AddPtrOp>(op.getLoc(), ptrType, returndataPtr, offset);
        if (convertedType != ptrType)
            rewriter.replaceOp(op, rewriter.create<sir::BitcastOp>(op.getLoc(), convertedType, dataPtr));
        else
            rewriter.replaceOp(op, dataPtr);
        return success();
    }

    auto replaceReturndataPtrView = [&](Type viewType) -> LogicalResult {
        Type convertedType = typeConverter->convertType(viewType);
        if (!convertedType)
            return rewriter.notifyMatchFailure(op, "failed to convert ABI return ptr-view type");

        Value decodedPtr = returndataPtr;
        if (convertedType != ptrType)
            decodedPtr = rewriter.create<sir::BitcastOp>(op.getLoc(), convertedType, decodedPtr);

        rewriter.replaceOp(op, createPtrViewMaterializationCast(rewriter, op.getLoc(), viewType, decodedPtr));
        return success();
    };

    if (llvm::isa<ora::StructType>(origType))
        return replaceReturndataPtrView(origType);

    if (llvm::isa<ora::TupleType>(origType))
        return replaceReturndataPtrView(origType);

    if (llvm::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(origType))
        return replaceReturndataPtrView(origType);

    if (root.kind == AbiLayoutKind::Tuple && root.children.size() == 1)
    {
        if (std::optional<Value> decoded = lowerReturndataSuccessfulStaticScalarValue(
                rewriter,
                op.getLoc(),
                *root.children.front(),
                origType,
                returndataPtr))
        {
            Type convertedType = typeConverter->convertType(op.getResult().getType());
            if (!convertedType)
                return rewriter.notifyMatchFailure(op, "failed to convert ABI decode result type");
            Value replacement = *decoded;
            if (convertedType != replacement.getType())
                replacement = rewriter.create<sir::BitcastOp>(op.getLoc(), convertedType, replacement);

            rewriter.replaceOp(op, replacement);
            return success();
        }
    }

    Value loaded = rewriter.create<sir::LoadOp>(op.getLoc(), u256Type, returndataPtr);

    Type convertedType = typeConverter->convertType(op.getResult().getType());
    if (!convertedType)
        return rewriter.notifyMatchFailure(op, "failed to convert ABI decode result type");
    if (convertedType != loaded.getType())
        loaded = rewriter.create<sir::BitcastOp>(op.getLoc(), convertedType, loaded);

    rewriter.replaceOp(op, loaded);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.contract by splicing its body into the parent module
// -----------------------------------------------------------------------------
LogicalResult ConvertContractOp::matchAndRewrite(
    ora::ContractOp op,
    typename ora::ContractOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    Region *parent = op->getParentRegion();
    if (!parent || parent->empty())
        return rewriter.notifyMatchFailure(op, "missing parent region");

    Block &contractBlock = op.getBody().front();

    for (auto it = contractBlock.begin(); it != contractBlock.end();)
    {
        Operation *inner = &*it++;
        if (llvm::isa<ora::YieldOp>(inner))
        {
            rewriter.eraseOp(inner);
            continue;
        }
        rewriter.moveOpBefore(inner, op);
    }

    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.error.decl to sir.error.decl metadata
// -----------------------------------------------------------------------------
LogicalResult ConvertErrorDeclOp::matchAndRewrite(
    ora::ErrorDeclOp op,
    typename ora::ErrorDeclOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
        return rewriter.notifyMatchFailure(op, "missing type converter");

    SmallVector<NamedAttribute> attrs;
    attrs.reserve(op->getAttrs().size());
    for (auto attr : op->getAttrs())
    {
        StringRef name = attr.getName();
        if (name == "ora.error_decl")
        {
            continue;
        }
        if (name == "ora.error_id")
        {
            attrs.push_back(rewriter.getNamedAttr("sir.error_id", attr.getValue()));
            continue;
        }
        if (name == "ora.param_names")
        {
            attrs.push_back(rewriter.getNamedAttr("sir.param_names", attr.getValue()));
            continue;
        }
        if (name == "ora.error_selector")
        {
            attrs.push_back(rewriter.getNamedAttr("sir.error_selector", attr.getValue()));
            continue;
        }
        if (name == "ora.param_types")
        {
            auto arr = llvm::dyn_cast<ArrayAttr>(attr.getValue());
            if (!arr)
                return rewriter.notifyMatchFailure(op, "ora.param_types is not ArrayAttr");
            SmallVector<Attribute> converted;
            converted.reserve(arr.size());
            for (auto elem : arr)
            {
                auto typeAttr = llvm::dyn_cast<TypeAttr>(elem);
                if (!typeAttr)
                    return rewriter.notifyMatchFailure(op, "ora.param_types element is not TypeAttr");
                Type origType = typeAttr.getValue();
                Type convertedType = typeConverter->convertType(origType);
                if (!convertedType)
                    return rewriter.notifyMatchFailure(op, "unable to convert error param type");
                if (convertedType == origType)
                {
                    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(origType))
                    {
                        if (intType.getWidth() == 256)
                            convertedType = sir::U256Type::get(op.getContext());
                    }
                }
                converted.push_back(TypeAttr::get(convertedType));
            }
            attrs.push_back(rewriter.getNamedAttr("sir.param_types", rewriter.getArrayAttr(converted)));
            continue;
        }
        if (name.starts_with("ora."))
            continue;
        attrs.push_back(attr);
    }

    OperationState state(op.getLoc(), sir::ErrorDeclOp::getOperationName());
    state.addAttributes(attrs);
    rewriter.create(state);
    rewriter.eraseOp(op);
    return success();
}
