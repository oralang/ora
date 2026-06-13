#pragma once

#include "OraMaterializationKinds.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

namespace mlir::ora::lowering
{
    // Constructor dynamic-tail decode uses bounded temporary memory before the
    // runtime heap is initialized; keep the free pointer above that region.
    inline constexpr uint64_t kConstructorDecodeScratchFenceBytes = 64 * 1024;

    static_assert(kConstructorDecodeScratchFenceBytes % 32 == 0,
                  "constructor decode scratch fence must be word-aligned");

    enum class AbiDecodeError : uint64_t
    {
        TruncatedBuffer = 0,
        OversizeBuffer = 1,
        BufferSizeExceeded = 2,
        NonCanonicalPadding = 3,
        InvalidBoolValue = 4,
        InvalidAddress = 5,
        InvalidFixedBytes = 6,
        EnumOutOfRange = 7,
        DepthLimitExceeded = 8,
        ArrayLengthExceeded = 9,
        RefinementViolation = 10,
        NonCanonicalEncoding = 11,
        InvalidOffset = 12,
        LengthOverflow = 13,
        StringLengthExceeded = 14,
    };

    inline Value constU256(OpBuilder &rewriter, Location loc, const llvm::APInt &value)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        auto u256IntType = mlir::IntegerType::get(rewriter.getContext(), 256, mlir::IntegerType::Unsigned);
        return rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(u256IntType, value.zextOrTrunc(256)));
    }

    inline Value constU256(OpBuilder &rewriter, Location loc, mlir::IntegerAttr value)
    {
        return constU256(rewriter, loc, value.getValue());
    }

    inline Value constU256(OpBuilder &rewriter, Location loc, uint64_t value)
    {
        return constU256(rewriter, loc, llvm::APInt(256, value));
    }

    inline Value createPtrViewMaterializationCast(OpBuilder &rewriter, Location loc, Type resultType, Value input)
    {
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, TypeRange{resultType}, ValueRange{input});
        cast->setAttr(kOraMaterializationKindAttr, rewriter.getStringAttr(mat_kind::kPtrView));
        return cast.getResult(0);
    }

    inline Value maskLowBits(OpBuilder &rewriter, Location loc, Value value, unsigned bits)
    {
        if (bits >= 256)
            return value;
        llvm::APInt mask = llvm::APInt::getLowBitsSet(256, bits);
        Value maskValue = constU256(rewriter, loc, mask);
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        return rewriter.create<sir::AndOp>(loc, u256Type, value, maskValue).getResult();
    }

    inline Value boolAbiWordIsCanonical(OpBuilder &rewriter, Location loc, Value word)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value zero = constU256(rewriter, loc, 0);
        Value one = constU256(rewriter, loc, 1);
        Value isZero = rewriter.create<sir::EqOp>(loc, u256Type, word, zero);
        Value isOne = rewriter.create<sir::EqOp>(loc, u256Type, word, one);
        return rewriter.create<sir::OrOp>(loc, u256Type, isZero, isOne);
    }

    inline Value boolAbiWordPermissivePayload(OpBuilder &rewriter, Location loc, Value word)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value zero = constU256(rewriter, loc, 0);
        Value one = constU256(rewriter, loc, 1);
        Value isZero = rewriter.create<sir::IsZeroOp>(loc, u256Type, word);
        return rewriter.create<sir::SelectOp>(loc, u256Type, isZero, zero, one);
    }

    struct FixedBytesWordDecode
    {
        Value payload;
        Value valid;
    };

    inline FixedBytesWordDecode decodeFixedBytesAbiWord(OpBuilder &rewriter, Location loc, uint64_t width, Value word)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        if (width == 32)
        {
            return FixedBytesWordDecode{
                word,
                constU256(rewriter, loc, 1),
            };
        }

        const uint64_t shiftBits = static_cast<uint64_t>(32 - width) * 8ULL;
        Value shift = constU256(rewriter, loc, shiftBits);
        Value payload = rewriter.create<sir::ShrOp>(loc, u256Type, shift, word).getResult();
        Value expected = rewriter.create<sir::ShlOp>(loc, u256Type, shift, payload).getResult();
        Value valid = rewriter.create<sir::EqOp>(loc, u256Type, word, expected);
        return FixedBytesWordDecode{payload, valid};
    }

    inline llvm::APInt abiDecodeBoundAPInt(uint64_t highHigh, uint64_t highLow, uint64_t lowHigh, uint64_t lowLow)
    {
        // Ora refinement bounds are stored as four 64-bit limbs so the full
        // u256 bound is preserved independent of comparison signedness.
        llvm::APInt value(256, highHigh);
        value = value.shl(64);
        value |= llvm::APInt(256, highLow);
        value = value.shl(64);
        value |= llvm::APInt(256, lowHigh);
        value = value.shl(64);
        value |= llvm::APInt(256, lowLow);
        return value;
    }

    inline Value abiDecodeWordGte(OpBuilder &rewriter, Location loc, Value word, Value bound, bool isSigned)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value lt = isSigned
                       ? rewriter.create<sir::SLtOp>(loc, u256Type, word, bound).getResult()
                       : rewriter.create<sir::LtOp>(loc, u256Type, word, bound).getResult();
        return rewriter.create<sir::IsZeroOp>(loc, u256Type, lt);
    }

    inline Value abiDecodeWordLte(OpBuilder &rewriter, Location loc, Value word, Value bound, bool isSigned)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value gt = isSigned
                       ? rewriter.create<sir::SGtOp>(loc, u256Type, word, bound).getResult()
                       : rewriter.create<sir::GtOp>(loc, u256Type, word, bound).getResult();
        return rewriter.create<sir::IsZeroOp>(loc, u256Type, gt);
    }

    inline Value existingU256Value(Value value)
    {
        if (llvm::isa<sir::U256Type>(value.getType()))
            return value;
        if (auto bitcast = value.getDefiningOp<sir::BitcastOp>())
        {
            Value input = bitcast.getInput();
            if (llvm::isa<sir::U256Type>(input.getType()))
                return input;
        }
        if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (cast.getNumOperands() == 1)
            {
                Value input = cast.getOperand(0);
                if (llvm::isa<sir::U256Type>(input.getType()))
                    return input;
            }
        }
        return Value();
    }

    // Explicit carrier relabel. Use only at sites that have already established
    // the value is representable as a u256 carrier.
    inline Value coerceToU256(OpBuilder &rewriter, Location loc, Value value)
    {
        if (Value existing = existingU256Value(value))
            return existing;
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        return rewriter.create<sir::BitcastOp>(loc, u256Type, value);
    }

    inline Value ensureU256(PatternRewriter &rewriter, Location loc, Operation *anchor, Value value, llvm::StringRef context)
    {
        (void)loc;
        if (Value existing = existingU256Value(value))
            return existing;
        if (anchor)
            (void)rewriter.notifyMatchFailure(anchor, llvm::Twine(context) + " requires u256 value");
        return Value();
    }

    inline Value ceil32(OpBuilder &rewriter, Location loc, Value length)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value word = constU256(rewriter, loc, 32);
        Value addend = constU256(rewriter, loc, 31);
        Value rounded = rewriter.create<sir::AddOp>(loc, u256Type, length, addend);
        Value words = rewriter.create<sir::DivOp>(loc, u256Type, rounded, word);
        return rewriter.create<sir::MulOp>(loc, u256Type, words, word);
    }
}
