#pragma once

#include "patterns/LoweringHelpers.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include <memory>
#include <optional>

namespace mlir::ora::abi_lowering
{
    using mlir::ora::lowering::boolAbiWordIsCanonical;
    using mlir::ora::lowering::boolAbiWordPermissivePayload;
    using mlir::ora::lowering::abiDecodeBoundAPInt;
    using mlir::ora::lowering::abiDecodeWordGte;
    using mlir::ora::lowering::abiDecodeWordLte;
    using mlir::ora::lowering::ceil32;
    using mlir::ora::lowering::constU256;
    using mlir::ora::lowering::decodeFixedBytesAbiWord;
    using mlir::ora::lowering::ensureU256;
    using mlir::ora::lowering::FixedBytesWordDecode;
    using mlir::ora::lowering::maskLowBits;

    enum class AbiStaticKind
    {
        Uint,
        Int,
        Bool,
        Address,
        FixedBytes,
    };

    struct AbiStaticLeaf
    {
        AbiStaticKind kind;
        unsigned width;
        unsigned operandIndex;
        unsigned aggregateIndex;
        bool loadFromAggregate;
    };

    enum class AbiLayoutKind
    {
        Static,
        DynamicBytes,
        DynamicArray,
        Tuple,
        FixedArray,
    };

    struct AbiLayoutNode
    {
        AbiLayoutKind kind = AbiLayoutKind::Tuple;
        AbiStaticKind staticKind = AbiStaticKind::Uint;
        unsigned width = 0;
        unsigned operandIndex = 0;
        unsigned aggregateIndex = 0;
        bool loadFromAggregate = false;
        unsigned arrayLen = 0;
        SmallVector<std::unique_ptr<AbiLayoutNode>, 4> children;

        bool isDynamic() const
        {
            switch (kind)
            {
            case AbiLayoutKind::DynamicBytes:
            case AbiLayoutKind::DynamicArray:
                return true;
            case AbiLayoutKind::Tuple:
                for (const auto &child : children)
                    if (child->isDynamic())
                        return true;
                return false;
            case AbiLayoutKind::FixedArray:
                return !children.empty() && children.front()->isDynamic();
            case AbiLayoutKind::Static:
                return false;
            }
            return false;
        }

        uint64_t headBytes() const
        {
            switch (kind)
            {
            case AbiLayoutKind::Static:
                return 32;
            case AbiLayoutKind::Tuple:
            {
                uint64_t total = 0;
                for (const auto &child : children)
                    total += child->headSlotBytes();
                return total;
            }
            case AbiLayoutKind::FixedArray:
                return children.empty() ? 0 : static_cast<uint64_t>(arrayLen) * children.front()->headBytes();
            case AbiLayoutKind::DynamicBytes:
            case AbiLayoutKind::DynamicArray:
                return 32;
            }
            return 0;
        }

        uint64_t headSlotBytes() const
        {
            return isDynamic() ? 32 : headBytes();
        }

        void collectStaticLeaves(SmallVectorImpl<AbiStaticLeaf> &leaves) const
        {
            switch (kind)
            {
            case AbiLayoutKind::Static:
                leaves.push_back({staticKind, width, operandIndex, aggregateIndex, loadFromAggregate});
                return;
            case AbiLayoutKind::Tuple:
                for (const auto &child : children)
                    child->collectStaticLeaves(leaves);
                return;
            case AbiLayoutKind::FixedArray:
                if (children.empty())
                    return;
                for (unsigned i = 0; i < arrayLen; ++i)
                {
                    SmallVector<AbiStaticLeaf, 4> elementLeaves;
                    children.front()->collectStaticLeaves(elementLeaves);
                    const unsigned elementWords = static_cast<unsigned>(children.front()->headBytes() / 32);
                    for (AbiStaticLeaf leaf : elementLeaves)
                    {
                        leaf.aggregateIndex += i * elementWords;
                        leaves.push_back(leaf);
                    }
                }
                return;
            case AbiLayoutKind::DynamicBytes:
            case AbiLayoutKind::DynamicArray:
                return;
            }
        }
    };

    class AbiLayoutDslParser
    {
    public:
        explicit AbiLayoutDslParser(StringRef text) : text(text) {}

        bool parse(SmallVectorImpl<AbiStaticLeaf> &leaves)
        {
            AbiLayoutNode root;
            if (!parse(root))
                return false;
            root.collectStaticLeaves(leaves);
            return true;
        }

        bool parse(AbiLayoutNode &root)
        {
            if (!parseRoot(root))
                return false;
            skipSpaces();
            return pos == text.size();
        }

        unsigned getOperandCount() const { return operandCount; }

    private:
        StringRef text;
        size_t pos = 0;
        unsigned operandCount = 0;

        void skipSpaces()
        {
            while (pos < text.size() && llvm::isSpace(text[pos]))
                ++pos;
        }

        bool consume(StringRef token)
        {
            skipSpaces();
            if (!text.substr(pos).starts_with(token))
                return false;
            pos += token.size();
            return true;
        }

        bool parseUnsigned(unsigned &out)
        {
            skipSpaces();
            if (pos >= text.size() || !llvm::isDigit(text[pos]))
                return false;
            unsigned value = 0;
            while (pos < text.size() && llvm::isDigit(text[pos]))
            {
                value = value * 10 + static_cast<unsigned>(text[pos] - '0');
                ++pos;
            }
            out = value;
            return true;
        }

        bool parseRoot(AbiLayoutNode &root)
        {
            skipSpaces();
            if (!text.substr(pos).starts_with("tuple("))
            {
                // Production Zig emission wraps parameter lists in tuple(...).
                // This bare form is accepted only for defensive/manual layouts.
                unsigned aggregateIndex = 0;
                operandCount = 1;
                return parseNode(root, /*operandIndex=*/0, /*loadFromAggregate=*/false, aggregateIndex, /*assignSelfAggregateSlot=*/false);
            }

            if (!consume("tuple("))
                return false;
            root.kind = AbiLayoutKind::Tuple;
            skipSpaces();
            if (consume(")"))
                return true;
            while (true)
            {
                const unsigned operandIndex = operandCount++;
                unsigned aggregateIndex = 0;
                auto child = std::make_unique<AbiLayoutNode>();
                if (!parseNode(*child, operandIndex, /*loadFromAggregate=*/false, aggregateIndex, /*assignSelfAggregateSlot=*/false))
                    return false;
                root.children.push_back(std::move(child));
                skipSpaces();
                if (consume(")"))
                    return true;
                if (!consume(","))
                    return false;
            }
        }

        bool parseNode(
            AbiLayoutNode &node,
            unsigned operandIndex,
            bool loadFromAggregate,
            unsigned &aggregateIndex,
            bool assignSelfAggregateSlot)
        {
            skipSpaces();
            if (consume("static("))
            {
                if (!parseStatic(node, operandIndex, loadFromAggregate, aggregateIndex))
                    return false;
                return consume(")");
            }
            if (consume("dynamic("))
            {
                skipSpaces();
                if (consume("string") || consume("bytes"))
                {
                    node.kind = AbiLayoutKind::DynamicBytes;
                    node.operandIndex = operandIndex;
                    node.aggregateIndex = aggregateIndex++;
                    node.loadFromAggregate = loadFromAggregate;
                    return consume(")");
                }
                return false;
            }
            if (consume("tuple("))
            {
                node.kind = AbiLayoutKind::Tuple;
                node.operandIndex = operandIndex;
                if (assignSelfAggregateSlot)
                {
                    node.aggregateIndex = aggregateIndex++;
                    node.loadFromAggregate = loadFromAggregate;
                }
                skipSpaces();
                if (consume(")"))
                    return true;
                unsigned childAggregateIndex = 0;
                while (true)
                {
                    auto child = std::make_unique<AbiLayoutNode>();
                    const bool childIsAggregate =
                        text.substr(pos).starts_with("tuple(") ||
                        (text.substr(pos).starts_with("array(") &&
                         !text.substr(pos).starts_with("array(dynamic"));
                    if (!parseNode(*child, operandIndex, /*loadFromAggregate=*/true, childAggregateIndex, childIsAggregate))
                        return false;
                    node.children.push_back(std::move(child));
                    skipSpaces();
                    if (consume(")"))
                        return true;
                    if (!consume(","))
                        return false;
                }
            }
            if (consume("array("))
            {
                skipSpaces();
                if (consume("dynamic"))
                {
                    if (!consume(","))
                        return false;
                    node.kind = AbiLayoutKind::DynamicArray;
                    node.operandIndex = operandIndex;
                    node.aggregateIndex = aggregateIndex++;
                    node.loadFromAggregate = loadFromAggregate;
                    auto element = std::make_unique<AbiLayoutNode>();
                    unsigned elementAggregateIndex = 0;
                    if (!parseNode(*element, operandIndex, /*loadFromAggregate=*/true, elementAggregateIndex, /*assignSelfAggregateSlot=*/false) || !consume(")"))
                        return false;
                    node.children.push_back(std::move(element));
                    return true;
                }
                unsigned len = 0;
                if (!parseUnsigned(len) || !consume(","))
                    return false;
                node.kind = AbiLayoutKind::FixedArray;
                node.arrayLen = len;
                node.operandIndex = operandIndex;
                if (assignSelfAggregateSlot)
                {
                    node.aggregateIndex = aggregateIndex++;
                    node.loadFromAggregate = loadFromAggregate;
                }
                auto element = std::make_unique<AbiLayoutNode>();
                unsigned elementAggregateIndex = 0;
                if (!parseNode(*element, operandIndex, /*loadFromAggregate=*/true, elementAggregateIndex, /*assignSelfAggregateSlot=*/false) || !consume(")"))
                    return false;
                if (!assignSelfAggregateSlot)
                    aggregateIndex += len * (element->headSlotBytes() / 32);
                node.children.push_back(std::move(element));
                return true;
            }
            return false;
        }

        bool parseStatic(
            AbiLayoutNode &node,
            unsigned operandIndex,
            bool loadFromAggregate,
            unsigned &aggregateIndex)
        {
            node.kind = AbiLayoutKind::Static;
            node.operandIndex = operandIndex;
            node.aggregateIndex = aggregateIndex++;
            node.loadFromAggregate = loadFromAggregate;
            skipSpaces();
            if (consume("bool"))
            {
                node.staticKind = AbiStaticKind::Bool;
                node.width = 1;
                return true;
            }
            if (consume("address"))
            {
                node.staticKind = AbiStaticKind::Address;
                node.width = 160;
                return true;
            }
            if (consume("uint"))
            {
                unsigned bits = 0;
                if (!parseUnsigned(bits))
                    return false;
                node.staticKind = AbiStaticKind::Uint;
                node.width = bits;
                return true;
            }
            if (consume("int"))
            {
                unsigned bits = 0;
                if (!parseUnsigned(bits))
                    return false;
                node.staticKind = AbiStaticKind::Int;
                node.width = bits;
                return true;
            }
            if (consume("bytes"))
            {
                unsigned len = 0;
                if (!parseUnsigned(len))
                    return false;
                node.staticKind = AbiStaticKind::FixedBytes;
                node.width = len;
                return true;
            }
            return false;
        }
    };

    inline Value abiAggregateSlotValue(PatternRewriter &rewriter, Location loc, Value aggregate, unsigned slotIndex)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
        if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(aggregate.getType());
            memrefType && memrefType.hasStaticShape() && memrefType.getRank() == 1)
        {
            // Static rank-1 memrefs match Ora's [T; N] fixed-array lowering.
            // Multi-dimensional or dynamic-shape arrays need separate handling.
            Value index = rewriter.create<mlir::arith::ConstantIndexOp>(loc, slotIndex);
            return rewriter.create<mlir::memref::LoadOp>(loc, aggregate, index).getResult();
        }

        Value basePtr = aggregate;
        if (llvm::isa<sir::U256Type>(basePtr.getType()) ||
            llvm::isa<mlir::IntegerType>(basePtr.getType()))
        {
            basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, ensureU256(rewriter, loc, basePtr));
        }
        else if (!llvm::isa<sir::PtrType>(basePtr.getType()))
        {
            if (auto castOp = basePtr.getDefiningOp<mlir::UnrealizedConversionCastOp>())
            {
                if (castOp.getNumOperands() == 1)
                {
                    Value src = castOp.getOperand(0);
                    if (llvm::isa<sir::PtrType>(src.getType()))
                    {
                        basePtr = src;
                    }
                    else if (llvm::isa<sir::U256Type>(src.getType()))
                    {
                        basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, src);
                    }
                }
            }
        }

        Value slotPtr = basePtr;
        if (slotIndex != 0)
        {
            Value offset = constU256(rewriter, loc, static_cast<uint64_t>(slotIndex) * 32ULL);
            slotPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, offset);
        }
        return rewriter.create<sir::LoadOp>(loc, u256Type, slotPtr);
    }

    inline Value materializeAbiStaticWord(PatternRewriter &rewriter, Location loc, AbiStaticLeaf leaf, Value operand)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
        if (leaf.kind == AbiStaticKind::FixedBytes && llvm::isa<sir::PtrType>(operand.getType()))
        {
            Value dataOffset = constU256(rewriter, loc, 32);
            Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, operand, dataOffset);
            // Fixed bytes literals can lower as a bytes-style pointer
            // (length word followed by right-padded data). In that case the
            // data word is already in ABI fixed-bytes shape. If another
            // pointer layout starts representing bytesN values, this branch
            // must distinguish it explicitly instead of assuming the bytes
            // literal layout.
            return rewriter.create<sir::LoadOp>(loc, u256Type, dataPtr);
        }
        Value value = ensureU256(rewriter, loc, operand);
        switch (leaf.kind)
        {
        case AbiStaticKind::Uint:
            return maskLowBits(rewriter, loc, value, leaf.width);
        case AbiStaticKind::Int:
            if (leaf.width < 256 && leaf.width > 0 && leaf.width % 8 == 0)
            {
                Value byteIndex = constU256(rewriter, loc, (leaf.width / 8) - 1);
                return rewriter.create<sir::SignExtendOp>(loc, u256Type, byteIndex, value).getResult();
            }
            return value;
        case AbiStaticKind::Bool:
            return maskLowBits(rewriter, loc, value, 1);
        case AbiStaticKind::Address:
            return maskLowBits(rewriter, loc, value, 160);
        case AbiStaticKind::FixedBytes:
            value = maskLowBits(rewriter, loc, value, leaf.width * 8);
            if (leaf.width < 32)
            {
                Value shift = constU256(rewriter, loc, (32 - leaf.width) * 8);
                return rewriter.create<sir::ShlOp>(loc, u256Type, shift, value).getResult();
            }
            return value;
        }
        return value;
    }

    inline Value addU256(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        return rewriter.create<sir::AddOp>(loc, u256Type, lhs, rhs);
    }

    inline Value subU256(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        return rewriter.create<sir::SubOp>(loc, u256Type, lhs, rhs);
    }

    inline Value mulU256(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        return rewriter.create<sir::MulOp>(loc, u256Type, lhs, rhs);
    }

    inline Value divU256(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        return rewriter.create<sir::DivOp>(loc, u256Type, lhs, rhs);
    }

    inline Value abiMemoryBytesPtr(PatternRewriter &rewriter, Location loc, Value bytes)
    {
        // Runtime @abiDecode currently lowers only memory-backed `bytes` values.
        // Ora's memory bytes representation stores a length word followed by
        // payload bytes in address space 1.
        auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
        Value bytesPtr = bytes;
        if (!llvm::isa<sir::PtrType>(bytesPtr.getType()))
            bytesPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, bytesPtr);
        return bytesPtr;
    }

    inline Value decodeAbiU256FromMemory(
        PatternRewriter &rewriter,
        Location loc,
        Value payloadPtr,
        uint64_t byteOffset)
    {
        // Raw ABI word load. Full-word scalar and all-u256 tuple callers can
        // use this directly; narrow callers must validate canonical padding
        // before constructing a Result.
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
        Value slotPtr = payloadPtr;
        if (byteOffset != 0)
            slotPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, payloadPtr, constU256(rewriter, loc, byteOffset));
        return rewriter.create<sir::LoadOp>(loc, u256Type, slotPtr);
    }

    inline bool isStaticU256AbiNode(const AbiLayoutNode &node)
    {
        return node.kind == AbiLayoutKind::Static &&
               node.staticKind == AbiStaticKind::Uint &&
               node.width == 256;
    }

    inline bool isStaticI256AbiNode(const AbiLayoutNode &node)
    {
        return node.kind == AbiLayoutKind::Static &&
               node.staticKind == AbiStaticKind::Int &&
               node.width == 256;
    }

    inline bool isStaticBoolAbiNode(const AbiLayoutNode &node)
    {
        return node.kind == AbiLayoutKind::Static &&
               node.staticKind == AbiStaticKind::Bool &&
               node.width == 1;
    }

    inline bool isStaticFixedBytesAbiNode(const AbiLayoutNode &node)
    {
        return node.kind == AbiLayoutKind::Static &&
               node.staticKind == AbiStaticKind::FixedBytes &&
               node.width > 0 &&
               node.width <= 32;
    }

    inline Type abiDecodeUnwrapRefinementType(Type type)
    {
        if (auto minType = llvm::dyn_cast<ora::MinValueType>(type))
            return abiDecodeUnwrapRefinementType(minType.getBaseType());
        if (auto maxType = llvm::dyn_cast<ora::MaxValueType>(type))
            return abiDecodeUnwrapRefinementType(maxType.getBaseType());
        if (auto rangeType = llvm::dyn_cast<ora::InRangeType>(type))
            return abiDecodeUnwrapRefinementType(rangeType.getBaseType());
        if (auto scaledType = llvm::dyn_cast<ora::ScaledType>(type))
            return abiDecodeUnwrapRefinementType(scaledType.getBaseType());
        if (auto exactType = llvm::dyn_cast<ora::ExactType>(type))
            return abiDecodeUnwrapRefinementType(exactType.getBaseType());
        if (llvm::isa<ora::NonZeroAddressType>(type))
            return ora::AddressType::get(type.getContext());
        return type;
    }

    inline bool successTypeIsU256Backed(Type successType)
    {
        successType = abiDecodeUnwrapRefinementType(successType);
        if (llvm::isa<sir::U256Type>(successType))
            return true;
        if (auto oraIntType = llvm::dyn_cast<ora::IntegerType>(successType))
            return !oraIntType.getIsSigned() && oraIntType.getWidth() == 256;
        auto intType = llvm::dyn_cast<mlir::IntegerType>(successType);
        return intType && intType.getWidth() == 256;
    }

    inline std::optional<unsigned> enumReprBitWidth(Type successType)
    {
        successType = abiDecodeUnwrapRefinementType(successType);
        auto enumType = llvm::dyn_cast<ora::EnumType>(successType);
        if (!enumType)
            return std::nullopt;
        Type reprType = enumType.getReprType();
        if (auto oraIntType = llvm::dyn_cast<ora::IntegerType>(reprType))
            return oraIntType.getWidth();
        if (auto intType = llvm::dyn_cast<mlir::IntegerType>(reprType))
            return intType.getWidth();
        return std::nullopt;
    }

    inline std::optional<uint64_t> enumVariantCountForType(Type successType, Operation *op)
    {
        successType = abiDecodeUnwrapRefinementType(successType);
        if (!op)
            return std::nullopt;
        if (auto countAttr = op->getAttrOfType<mlir::IntegerAttr>("enum_variant_count"))
            return countAttr.getValue().getZExtValue();

        auto enumType = llvm::dyn_cast<ora::EnumType>(successType);
        if (!enumType)
            return std::nullopt;

        const std::string prefix = enumType.getName().str() + ".";
        if (auto module = op->getParentOfType<mlir::ModuleOp>())
        {
            if (auto enumDict = module->getAttrOfType<DictionaryAttr>("sir.enum_values"))
            {
                uint64_t count = 0;
                for (NamedAttribute entry : enumDict)
                {
                    if (entry.getName().strref().starts_with(prefix))
                        ++count;
                }
                if (count > 0)
                    return count;
            }

            std::optional<uint64_t> count;
            module.walk([&](ora::EnumDeclOp decl) {
                if (count || decl.getName() != enumType.getName())
                    return;
                if (auto variantNames = decl->getAttrOfType<mlir::ArrayAttr>("ora.variant_names"))
                    count = variantNames.size();
            });
            return count;
        }

        return std::nullopt;
    }

    inline bool isEnumSuccessType(Type successType)
    {
        return llvm::isa<ora::EnumType>(abiDecodeUnwrapRefinementType(successType));
    }

    inline bool isDynamicBytesSuccessType(Type successType)
    {
        successType = abiDecodeUnwrapRefinementType(successType);
        return llvm::isa<ora::StringType, ora::BytesType>(successType);
    }

    inline bool isDynamicU256MemRefSuccessType(Type successType)
    {
        successType = abiDecodeUnwrapRefinementType(successType);
        auto memrefType = llvm::dyn_cast<mlir::MemRefType>(successType);
        if (!memrefType || memrefType.getRank() != 1 || memrefType.hasStaticShape())
            return false;
        Type elementType = memrefType.getElementType();
        if (auto oraIntType = llvm::dyn_cast<ora::IntegerType>(elementType))
            return !oraIntType.getIsSigned() && oraIntType.getWidth() == 256;
        auto intType = llvm::dyn_cast<mlir::IntegerType>(elementType);
        return intType && intType.getWidth() == 256;
    }

    inline bool isDynamicAddressMemRefSuccessType(Type successType)
    {
        successType = abiDecodeUnwrapRefinementType(successType);
        auto memrefType = llvm::dyn_cast<mlir::MemRefType>(successType);
        if (!memrefType || memrefType.getRank() != 1 || memrefType.hasStaticShape())
            return false;
        return llvm::isa<ora::AddressType>(abiDecodeUnwrapRefinementType(memrefType.getElementType()));
    }

    inline bool isDynamicBoolMemRefSuccessType(Type successType)
    {
        successType = abiDecodeUnwrapRefinementType(successType);
        auto memrefType = llvm::dyn_cast<mlir::MemRefType>(successType);
        if (!memrefType || memrefType.getRank() != 1 || memrefType.hasStaticShape())
            return false;
        Type elementType = abiDecodeUnwrapRefinementType(memrefType.getElementType());
        if (llvm::isa<ora::BoolType>(elementType))
            return true;
        auto intType = llvm::dyn_cast<mlir::IntegerType>(elementType);
        return intType && intType.getWidth() == 1;
    }

    inline bool isDynamicFixedBytesMemRefSuccessType(Type successType)
    {
        successType = abiDecodeUnwrapRefinementType(successType);
        auto memrefType = llvm::dyn_cast<mlir::MemRefType>(successType);
        if (!memrefType || memrefType.getRank() != 1 || memrefType.hasStaticShape())
            return false;
        return successTypeIsU256Backed(memrefType.getElementType());
    }

    inline bool isDynamicArrayOfAbiNode(const AbiLayoutNode &node, bool (*elementPred)(const AbiLayoutNode &))
    {
        if (node.kind != AbiLayoutKind::DynamicArray || node.children.size() != 1)
            return false;
        return elementPred(*node.children.front());
    }

    inline bool isDynamicU256ArrayAbiNode(const AbiLayoutNode &node)
    {
        return isDynamicArrayOfAbiNode(node, isStaticU256AbiNode);
    }

    inline bool isStaticAddressAbiNode(const AbiLayoutNode &node)
    {
        return node.kind == AbiLayoutKind::Static &&
               node.staticKind == AbiStaticKind::Address &&
               node.width == 160;
    }

    inline bool isDynamicAddressArrayAbiNode(const AbiLayoutNode &node)
    {
        return isDynamicArrayOfAbiNode(node, isStaticAddressAbiNode);
    }

    inline bool isDynamicBoolArrayAbiNode(const AbiLayoutNode &node)
    {
        return isDynamicArrayOfAbiNode(node, isStaticBoolAbiNode);
    }

    inline bool isDynamicFixedBytesArrayAbiNode(const AbiLayoutNode &node)
    {
        return isDynamicArrayOfAbiNode(node, isStaticFixedBytesAbiNode);
    }

}
