#include "patterns/ControlFlow.h"
#include "patterns/AbiLoweringCommon.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;
using namespace mlir::ora::abi_lowering;

namespace
{
    static Value operandForAbiNode(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        ValueRange operands)
    {
        if (node.operandIndex >= operands.size())
            return {};
        Value operand = operands[node.operandIndex];
        if (node.loadFromAggregate)
            operand = abiAggregateSlotValue(rewriter, loc, operand, node.aggregateIndex);
        return operand;
    }

    static Value ptrForDynamicBytesOperand(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        ValueRange operands)
    {
        auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
        Value operand = operandForAbiNode(rewriter, loc, node, operands);
        if (!operand)
            return {};
        if (llvm::isa<sir::PtrType>(operand.getType()))
            return operand;
        return rewriter.create<sir::BitcastOp>(loc, ptrType, ensureU256(rewriter, loc, operand));
    }

    static bool isPointerBackedAggregateNode(const AbiLayoutNode &node)
    {
        return node.loadFromAggregate &&
               (node.kind == AbiLayoutKind::Tuple || node.kind == AbiLayoutKind::FixedArray);
    }

    static Value ptrForAggregateNodeOperand(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        ValueRange operands)
    {
        auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
        Value operand = operandForAbiNode(rewriter, loc, node, operands);
        if (!operand)
            return {};
        if (llvm::isa<sir::PtrType>(operand.getType()))
            return operand;
        return rewriter.create<sir::BitcastOp>(loc, ptrType, ensureU256(rewriter, loc, operand));
    }

    static std::optional<uint64_t> staticMemRefElementCount(Operation *ownerOp, const AbiLayoutNode &node)
    {
        if (!ownerOp || node.loadFromAggregate || node.operandIndex >= ownerOp->getNumOperands())
            return std::nullopt;
        auto memrefType = llvm::dyn_cast<mlir::MemRefType>(ownerOp->getOperand(node.operandIndex).getType());
        if (!memrefType || !memrefType.hasStaticShape() || memrefType.getRank() != 1)
            return std::nullopt;
        return static_cast<uint64_t>(memrefType.getDimSize(0));
    }

    static bool staticMemRefElementUsesPointerLayout(Operation *ownerOp, const AbiLayoutNode &node)
    {
        if (!ownerOp || node.loadFromAggregate || node.operandIndex >= ownerOp->getNumOperands())
            return false;
        auto memrefType = llvm::dyn_cast<mlir::MemRefType>(ownerOp->getOperand(node.operandIndex).getType());
        if (!memrefType || !memrefType.hasStaticShape() || memrefType.getRank() != 1)
            return false;
        // Ora lowers memref<N x T> with pointer-backed storage when T is an
        // aggregate that does not fit inline. Update this list if any other
        // element kinds adopt pointer-backed storage.
        return llvm::isa<ora::TupleType, ora::StructType, ora::AnonymousStructType,
                         ora::StringType, ora::BytesType,
                         mlir::MemRefType, mlir::UnrankedMemRefType>(memrefType.getElementType());
    }

    struct DynamicArraySource
    {
        Value ptr;
        Value length;
        bool sourceHasInlineLength;
        bool staticElementsUsePointerLayout;
        std::optional<uint64_t> staticElementCount;
    };

    static std::optional<DynamicArraySource> dynamicArraySource(
        PatternRewriter &rewriter,
        Location loc,
        Operation *ownerOp,
        const AbiLayoutNode &node,
        ValueRange operands)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value source = ptrForDynamicBytesOperand(rewriter, loc, node, operands);
        if (!source)
            return std::nullopt;

        if (std::optional<uint64_t> staticCount = staticMemRefElementCount(ownerOp, node))
        {
            // Source casts from fixed arrays to slices materialize runtime
            // slices before reaching the encoder. This path remains for
            // static pointer-slot memrefs used by isolated layout/materializer
            // tests and aggregate storage that is already fixed-size in IR.
            return DynamicArraySource{
                source,
                constU256(rewriter, loc, *staticCount),
                false,
                staticMemRefElementUsesPointerLayout(ownerOp, node),
                *staticCount,
            };
        }

        bool dynamicSourceUsesPointerLayout = false;
        if (!node.children.empty())
        {
            const AbiLayoutNode &element = *node.children.front();
            dynamicSourceUsesPointerLayout =
                element.isDynamic() ||
                element.kind == AbiLayoutKind::Tuple ||
                element.kind == AbiLayoutKind::FixedArray;
        }

        return DynamicArraySource{
            source,
            rewriter.create<sir::LoadOp>(loc, u256Type, source),
            true,
            dynamicSourceUsesPointerLayout,
            std::nullopt,
        };
    }

    static Value addPtrOffset(PatternRewriter &rewriter, Location loc, Value basePtr, Value offset);

    static Value dynamicArrayElementPointer(
        PatternRewriter &rewriter,
        Location loc,
        const DynamicArraySource &source,
        Value index)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
        Value word = constU256(rewriter, loc, 32);
        Value offset = mulU256(rewriter, loc, index, word);
        if (source.sourceHasInlineLength)
            offset = addU256(rewriter, loc, word, offset);
        Value slot = addPtrOffset(rewriter, loc, source.ptr, offset);
        Value ptrWord = rewriter.create<sir::LoadOp>(loc, u256Type, slot);
        return rewriter.create<sir::BitcastOp>(loc, ptrType, ptrWord);
    }

    static Value aggregateDynamicChildPointer(
        PatternRewriter &rewriter,
        Location loc,
        Value aggregatePtr,
        const AbiLayoutNode &child)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
        Value fieldOffset = constU256(rewriter, loc, static_cast<uint64_t>(child.aggregateIndex) * 32ULL);
        Value fieldSlot = addPtrOffset(rewriter, loc, aggregatePtr, fieldOffset);
        Value ptrWord = rewriter.create<sir::LoadOp>(loc, u256Type, fieldSlot);
        return rewriter.create<sir::BitcastOp>(loc, ptrType, ptrWord);
    }

    static Value dynamicBytesTailSizeFromPtr(PatternRewriter &rewriter, Location loc, Value source)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value length = rewriter.create<sir::LoadOp>(loc, u256Type, source);
        return addU256(rewriter, loc, constU256(rewriter, loc, 32), ceil32(rewriter, loc, length));
    }

    static Value abiEncodedSizeFromPointer(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Value sourcePtr);

    static Value dynamicArrayRuntimeTailSizeFromSource(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &element,
        const DynamicArraySource &source,
        Value initialTotal)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value zero = constU256(rewriter, loc, 0);
        Value one = constU256(rewriter, loc, 1);

        Block *parentBlock = rewriter.getInsertionBlock();
        Region *parentRegion = parentBlock->getParent();
        auto afterBlock = rewriter.splitBlock(parentBlock, rewriter.getInsertionPoint());
        afterBlock->addArgument(u256Type, loc);
        auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type, u256Type}, {loc, loc});
        auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type, u256Type}, {loc, loc});

        rewriter.setInsertionPointToEnd(parentBlock);
        rewriter.create<sir::BrOp>(loc, ValueRange{zero, initialTotal}, condBlock);

        rewriter.setInsertionPointToStart(condBlock);
        Value iv = condBlock->getArgument(0);
        Value total = condBlock->getArgument(1);
        Value lt = rewriter.create<sir::LtOp>(loc, u256Type, iv, source.length);
        rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv, total}, ValueRange{total}, bodyBlock, afterBlock);

        rewriter.setInsertionPointToStart(bodyBlock);
        Value bodyIv = bodyBlock->getArgument(0);
        Value bodyTotal = bodyBlock->getArgument(1);
        Value elementPtr = dynamicArrayElementPointer(rewriter, loc, source, bodyIv);
        Value childSize = abiEncodedSizeFromPointer(rewriter, loc, element, elementPtr);
        if (!childSize)
            return {};
        Value nextTotal = addU256(rewriter, loc, bodyTotal, childSize);
        Value next = addU256(rewriter, loc, bodyIv, one);
        rewriter.create<sir::BrOp>(loc, ValueRange{next, nextTotal}, condBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        return afterBlock->getArgument(0);
    }

    static Value dynamicArrayTailSizeFromSource(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        const DynamicArraySource &source)
    {
        if (node.children.empty())
            return {};
        const AbiLayoutNode &element = *node.children.front();
        Value total = addU256(
            rewriter,
            loc,
            constU256(rewriter, loc, 32),
            mulU256(rewriter, loc, source.length, constU256(rewriter, loc, element.headSlotBytes())));
        if (!element.isDynamic())
            return total;
        if (!source.staticElementCount)
            return dynamicArrayRuntimeTailSizeFromSource(rewriter, loc, element, source, total);
        for (uint64_t index = 0; index < *source.staticElementCount; ++index)
        {
            Value elementPtr = dynamicArrayElementPointer(rewriter, loc, source, constU256(rewriter, loc, index));
            Value childSize = abiEncodedSizeFromPointer(rewriter, loc, element, elementPtr);
            if (!childSize)
                return {};
            total = addU256(rewriter, loc, total, childSize);
        }
        return total;
    }

    static Value fixedArrayDynamicSizeFromPointer(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Value sourcePtr)
    {
        if (node.children.empty())
            return {};
        const AbiLayoutNode &element = *node.children.front();
        Value total = constU256(rewriter, loc, static_cast<uint64_t>(node.arrayLen) * element.headSlotBytes());
        if (!element.isDynamic())
            return total;
        DynamicArraySource source{sourcePtr, constU256(rewriter, loc, node.arrayLen), false, true, node.arrayLen};
        for (uint64_t index = 0; index < node.arrayLen; ++index)
        {
            Value elementPtr = dynamicArrayElementPointer(rewriter, loc, source, constU256(rewriter, loc, index));
            Value childSize = abiEncodedSizeFromPointer(rewriter, loc, element, elementPtr);
            if (!childSize)
                return {};
            total = addU256(rewriter, loc, total, childSize);
        }
        return total;
    }

    static Value abiEncodedSizeFromPointer(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Value sourcePtr)
    {
        switch (node.kind)
        {
        case AbiLayoutKind::DynamicBytes:
            return dynamicBytesTailSizeFromPtr(rewriter, loc, sourcePtr);
        case AbiLayoutKind::DynamicArray:
        {
            auto u256Type = sir::U256Type::get(rewriter.getContext());
            DynamicArraySource source{sourcePtr, rewriter.create<sir::LoadOp>(loc, u256Type, sourcePtr), true, node.children.empty() ? false : node.children.front()->isDynamic(), std::nullopt};
            return dynamicArrayTailSizeFromSource(rewriter, loc, node, source);
        }
        case AbiLayoutKind::Tuple:
        {
            Value total = constU256(rewriter, loc, node.headBytes());
            for (const auto &child : node.children)
            {
                if (!child->isDynamic())
                    continue;
                Value childPtr = aggregateDynamicChildPointer(rewriter, loc, sourcePtr, *child);
                Value childSize = abiEncodedSizeFromPointer(rewriter, loc, *child, childPtr);
                if (!childSize)
                    return {};
                total = addU256(rewriter, loc, total, childSize);
            }
            return total;
        }
        case AbiLayoutKind::FixedArray:
            return fixedArrayDynamicSizeFromPointer(rewriter, loc, node, sourcePtr);
        case AbiLayoutKind::Static:
            return constU256(rewriter, loc, 32);
        }
        return {};
    }

    static Value dynamicBytesTailSize(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        ValueRange operands)
    {
        Value source = ptrForDynamicBytesOperand(rewriter, loc, node, operands);
        if (!source)
            return {};
        return dynamicBytesTailSizeFromPtr(rewriter, loc, source);
    }

    static Value dynamicArrayTailSize(
        PatternRewriter &rewriter,
        Location loc,
        Operation *ownerOp,
        const AbiLayoutNode &node,
        ValueRange operands)
    {
        if (node.children.empty())
            return {};
        std::optional<DynamicArraySource> source = dynamicArraySource(rewriter, loc, ownerOp, node, operands);
        if (!source)
            return {};
        return dynamicArrayTailSizeFromSource(rewriter, loc, node, *source);
    }

    using AbiSizeCache = llvm::DenseMap<const AbiLayoutNode *, Value>;

    static Value abiEncodedSize(
        PatternRewriter &rewriter,
        Location loc,
        Operation *ownerOp,
        const AbiLayoutNode &node,
        ValueRange operands,
        AbiSizeCache &cache)
    {
        auto cached = cache.find(&node);
        if (cached != cache.end())
            return cached->second;

        Value result;
        switch (node.kind)
        {
        case AbiLayoutKind::Static:
            result = constU256(rewriter, loc, node.headBytes());
            break;
        case AbiLayoutKind::FixedArray:
            if (node.isDynamic())
            {
                Value source = ptrForDynamicBytesOperand(rewriter, loc, node, operands);
                if (!source)
                    return {};
                result = fixedArrayDynamicSizeFromPointer(rewriter, loc, node, source);
            }
            else
            {
                result = constU256(rewriter, loc, node.headBytes());
            }
            break;
        case AbiLayoutKind::DynamicBytes:
            result = dynamicBytesTailSize(rewriter, loc, node, operands);
            break;
        case AbiLayoutKind::DynamicArray:
            result = dynamicArrayTailSize(rewriter, loc, ownerOp, node, operands);
            break;
        case AbiLayoutKind::Tuple:
        {
            Value total = constU256(rewriter, loc, node.headBytes());
            for (const auto &child : node.children)
            {
                if (!child->isDynamic())
                    continue;
                Value childSize;
                if (isPointerBackedAggregateNode(*child))
                {
                    Value childPtr = ptrForAggregateNodeOperand(rewriter, loc, *child, operands);
                    if (!childPtr)
                        return {};
                    childSize = abiEncodedSizeFromPointer(rewriter, loc, *child, childPtr);
                }
                else
                {
                    childSize = abiEncodedSize(rewriter, loc, ownerOp, *child, operands, cache);
                }
                if (!childSize)
                    return {};
                total = addU256(rewriter, loc, total, childSize);
            }
            result = total;
            break;
        }
        }
        if (!result)
            return {};
        cache.try_emplace(&node, result);
        return result;
    }

    static Value addPtrOffset(PatternRewriter &rewriter, Location loc, Value basePtr, Value offset)
    {
        auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1);
        return rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, offset);
    }

    static void storeAbiWordAt(PatternRewriter &rewriter, Location loc, Value basePtr, Value byteOffset, Value value)
    {
        Value slotPtr = addPtrOffset(rewriter, loc, basePtr, byteOffset);
        rewriter.create<sir::StoreOp>(loc, slotPtr, value);
    }

    static LogicalResult emitAbiStaticNode(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        ValueRange operands,
        Value basePtr,
        Value byteOffset)
    {
        if (node.kind == AbiLayoutKind::Static)
        {
            AbiStaticLeaf leaf{node.staticKind, node.width, node.operandIndex, node.aggregateIndex, node.loadFromAggregate};
            Value operand = operandForAbiNode(rewriter, loc, node, operands);
            if (!operand)
                return failure();
            Value abiValue = materializeAbiStaticWord(rewriter, loc, leaf, operand);
            storeAbiWordAt(rewriter, loc, basePtr, byteOffset, abiValue);
            return success();
        }
        if (node.kind == AbiLayoutKind::Tuple && !node.isDynamic())
        {
            uint64_t cursor = 0;
            for (const auto &child : node.children)
            {
                if (failed(emitAbiStaticNode(rewriter, loc, *child, operands, basePtr, addU256(rewriter, loc, byteOffset, constU256(rewriter, loc, cursor)))))
                    return failure();
                cursor += child->headBytes();
            }
            return success();
        }
        if (node.kind == AbiLayoutKind::FixedArray && !node.isDynamic())
        {
            SmallVector<AbiStaticLeaf, 8> leaves;
            node.collectStaticLeaves(leaves);
            for (auto [index, leaf] : llvm::enumerate(leaves))
            {
                if (leaf.operandIndex >= operands.size())
                    return failure();
                Value operand = operands[leaf.operandIndex];
                if (leaf.loadFromAggregate)
                    operand = abiAggregateSlotValue(rewriter, loc, operand, leaf.aggregateIndex);
                Value slotOffset = addU256(rewriter, loc, byteOffset, constU256(rewriter, loc, 32 * index));
                storeAbiWordAt(rewriter, loc, basePtr, slotOffset, materializeAbiStaticWord(rewriter, loc, leaf, operand));
            }
            return success();
        }
        return failure();
    }

    static LogicalResult emitAbiEncoding(
        PatternRewriter &rewriter,
        Location loc,
        Operation *anchorOp,
        const AbiLayoutNode &node,
        ValueRange operands,
        Value basePtr,
        Value baseOffset,
        AbiSizeCache &sizeCache);

    static LogicalResult emitDynamicBytesTailFromPtr(
        PatternRewriter &rewriter,
        Location loc,
        Value source,
        Value basePtr,
        Value baseOffset);

    static LogicalResult emitDynamicBytesTail(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        ValueRange operands,
        Value basePtr,
        Value baseOffset)
    {
        Value source = ptrForDynamicBytesOperand(rewriter, loc, node, operands);
        if (!source)
            return failure();
        return emitDynamicBytesTailFromPtr(rewriter, loc, source, basePtr, baseOffset);
    }

    static LogicalResult emitDynamicBytesTailFromPtr(
        PatternRewriter &rewriter,
        Location loc,
        Value source,
        Value basePtr,
        Value baseOffset)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value length = rewriter.create<sir::LoadOp>(loc, u256Type, source);
        storeAbiWordAt(rewriter, loc, basePtr, baseOffset, length);

        Value word = constU256(rewriter, loc, 32);
        Value sourcePayload = addPtrOffset(rewriter, loc, source, word);
        Value destPayload = addPtrOffset(rewriter, loc, basePtr, addU256(rewriter, loc, baseOffset, word));
        // Source string/bytes values use Ora's dynamic bytes layout:
        // [len: u256][bytes...]. The destination allocation is fresh EVM
        // memory, so copying the logical length leaves ABI padding zeroed.
        rewriter.create<sir::MCopyOp>(loc, destPayload, sourcePayload, length);
        return success();
    }

    static LogicalResult emitAbiEncodingFromPointer(
        PatternRewriter &rewriter,
        Location loc,
        Operation *anchorOp,
        const AbiLayoutNode &node,
        Value sourcePtr,
        Value basePtr,
        Value baseOffset);

    static LogicalResult emitAbiStaticNodeFromPointer(
        PatternRewriter &rewriter,
        Location loc,
        const AbiLayoutNode &node,
        Value sourcePtr,
        Value basePtr,
        Value byteOffset)
    {
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        if (node.kind == AbiLayoutKind::Static)
        {
            Value fieldOffset = constU256(rewriter, loc, static_cast<uint64_t>(node.aggregateIndex) * 32ULL);
            Value fieldPtr = addPtrOffset(rewriter, loc, sourcePtr, fieldOffset);
            Value rawValue = rewriter.create<sir::LoadOp>(loc, u256Type, fieldPtr);
            AbiStaticLeaf leaf{node.staticKind, node.width, 0, 0, false};
            storeAbiWordAt(rewriter, loc, basePtr, byteOffset, materializeAbiStaticWord(rewriter, loc, leaf, rawValue));
            return success();
        }
        if (node.kind == AbiLayoutKind::Tuple && !node.isDynamic())
        {
            uint64_t cursor = 0;
            for (const auto &child : node.children)
            {
                if (failed(emitAbiStaticNodeFromPointer(rewriter, loc, *child, sourcePtr, basePtr, addU256(rewriter, loc, byteOffset, constU256(rewriter, loc, cursor)))))
                    return failure();
                cursor += child->headBytes();
            }
            return success();
        }
        return failure();
    }

    static LogicalResult emitAbiTupleEncodingFromPointer(
        PatternRewriter &rewriter,
        Location loc,
        Operation *anchorOp,
        const AbiLayoutNode &node,
        Value sourcePtr,
        Value basePtr,
        Value baseOffset)
    {
        Value tailOffset = constU256(rewriter, loc, node.headBytes());
        uint64_t headCursor = 0;
        for (const auto &child : node.children)
        {
            Value headSlotOffset = addU256(rewriter, loc, baseOffset, constU256(rewriter, loc, headCursor));
            if (child->isDynamic())
            {
                storeAbiWordAt(rewriter, loc, basePtr, headSlotOffset, tailOffset);
                Value childPtr = aggregateDynamicChildPointer(rewriter, loc, sourcePtr, *child);
                Value childBaseOffset = addU256(rewriter, loc, baseOffset, tailOffset);
                if (failed(emitAbiEncodingFromPointer(rewriter, loc, anchorOp, *child, childPtr, basePtr, childBaseOffset)))
                    return failure();
                Value childSize = abiEncodedSizeFromPointer(rewriter, loc, *child, childPtr);
                if (!childSize)
                    return failure();
                tailOffset = addU256(rewriter, loc, tailOffset, childSize);
                headCursor += 32;
            }
            else
            {
                if (failed(emitAbiStaticNodeFromPointer(rewriter, loc, *child, sourcePtr, basePtr, headSlotOffset)))
                    return failure();
                headCursor += child->headBytes();
            }
        }
        return success();
    }

    static LogicalResult emitFixedArrayEncodingFromPointer(
        PatternRewriter &rewriter,
        Location loc,
        Operation *anchorOp,
        const AbiLayoutNode &node,
        Value sourcePtr,
        Value basePtr,
        Value baseOffset)
    {
        if (node.children.empty())
            return failure();
        const AbiLayoutNode &element = *node.children.front();
        if (!element.isDynamic())
            return emitAbiStaticNodeFromPointer(rewriter, loc, node, sourcePtr, basePtr, baseOffset);

        DynamicArraySource source{sourcePtr, constU256(rewriter, loc, node.arrayLen), false, true, node.arrayLen};
        Value tailOffset = constU256(rewriter, loc, static_cast<uint64_t>(node.arrayLen) * element.headSlotBytes());
        for (uint64_t index = 0; index < node.arrayLen; ++index)
        {
            Value headSlotOffset = addU256(rewriter, loc, baseOffset, constU256(rewriter, loc, index * element.headSlotBytes()));
            storeAbiWordAt(rewriter, loc, basePtr, headSlotOffset, tailOffset);
            Value elementPtr = dynamicArrayElementPointer(rewriter, loc, source, constU256(rewriter, loc, index));
            Value elementBaseOffset = addU256(rewriter, loc, baseOffset, tailOffset);
            if (failed(emitAbiEncodingFromPointer(rewriter, loc, anchorOp, element, elementPtr, basePtr, elementBaseOffset)))
                return failure();
            Value elementSize = abiEncodedSizeFromPointer(rewriter, loc, element, elementPtr);
            if (!elementSize)
                return failure();
            tailOffset = addU256(rewriter, loc, tailOffset, elementSize);
        }
        return success();
    }

    static LogicalResult emitDynamicArrayTailFromSource(
        PatternRewriter &rewriter,
        Location loc,
        Operation *anchorOp,
        const AbiLayoutNode &node,
        const DynamicArraySource &source,
        Value basePtr,
        Value baseOffset)
    {
        if (node.children.empty())
            return failure();
        storeAbiWordAt(rewriter, loc, basePtr, baseOffset, source.length);
        const AbiLayoutNode &element = *node.children.front();

        if (element.isDynamic())
        {
            if (!source.staticElementCount)
            {
                auto u256Type = sir::U256Type::get(rewriter.getContext());
                Value zero = constU256(rewriter, loc, 0);
                Value one = constU256(rewriter, loc, 1);
                Value word = constU256(rewriter, loc, 32);
                Value initialTailOffset = mulU256(rewriter, loc, source.length, constU256(rewriter, loc, element.headSlotBytes()));

                Block *parentBlock = rewriter.getInsertionBlock();
                Region *parentRegion = parentBlock->getParent();
                auto afterBlock = rewriter.splitBlock(parentBlock, rewriter.getInsertionPoint());
                auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type, u256Type}, {loc, loc});
                auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type, u256Type}, {loc, loc});

                rewriter.setInsertionPointToEnd(parentBlock);
                rewriter.create<sir::BrOp>(loc, ValueRange{zero, initialTailOffset}, condBlock);

                rewriter.setInsertionPointToStart(condBlock);
                Value iv = condBlock->getArgument(0);
                Value tailOffsetArg = condBlock->getArgument(1);
                Value lt = rewriter.create<sir::LtOp>(loc, u256Type, iv, source.length);
                rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv, tailOffsetArg}, ValueRange{}, bodyBlock, afterBlock);

                rewriter.setInsertionPointToStart(bodyBlock);
                Value bodyIv = bodyBlock->getArgument(0);
                Value bodyTailOffset = bodyBlock->getArgument(1);
                Value headSlotOffset = addU256(
                    rewriter,
                    loc,
                    baseOffset,
                    addU256(rewriter, loc, word, mulU256(rewriter, loc, bodyIv, constU256(rewriter, loc, element.headSlotBytes()))));
                storeAbiWordAt(rewriter, loc, basePtr, headSlotOffset, bodyTailOffset);

                Value elementPtr = dynamicArrayElementPointer(rewriter, loc, source, bodyIv);
                Value elementBaseOffset = addU256(rewriter, loc, baseOffset, addU256(rewriter, loc, word, bodyTailOffset));
                if (failed(emitAbiEncodingFromPointer(rewriter, loc, anchorOp, element, elementPtr, basePtr, elementBaseOffset)))
                    return failure();
                Value elementSize = abiEncodedSizeFromPointer(rewriter, loc, element, elementPtr);
                if (!elementSize)
                    return failure();
                Value nextTailOffset = addU256(rewriter, loc, bodyTailOffset, elementSize);
                Value next = addU256(rewriter, loc, bodyIv, one);
                rewriter.create<sir::BrOp>(loc, ValueRange{next, nextTailOffset}, condBlock);

                rewriter.setInsertionPointToStart(afterBlock);
                return success();
            }
            Value tailOffset = constU256(rewriter, loc, (*source.staticElementCount) * element.headSlotBytes());
            for (uint64_t index = 0; index < *source.staticElementCount; ++index)
            {
                Value headSlotOffset = addU256(rewriter, loc, baseOffset, addU256(rewriter, loc, constU256(rewriter, loc, 32), constU256(rewriter, loc, index * element.headSlotBytes())));
                storeAbiWordAt(rewriter, loc, basePtr, headSlotOffset, tailOffset);
                Value elementPtr = dynamicArrayElementPointer(rewriter, loc, source, constU256(rewriter, loc, index));
                Value elementBaseOffset = addU256(rewriter, loc, baseOffset, addU256(rewriter, loc, constU256(rewriter, loc, 32), tailOffset));
                if (failed(emitAbiEncodingFromPointer(rewriter, loc, anchorOp, element, elementPtr, basePtr, elementBaseOffset)))
                    return failure();
                Value elementSize = abiEncodedSizeFromPointer(rewriter, loc, element, elementPtr);
                if (!elementSize)
                    return failure();
                tailOffset = addU256(rewriter, loc, tailOffset, elementSize);
            }
            return success();
        }

        SmallVector<AbiStaticLeaf, 8> leaves;
        element.collectStaticLeaves(leaves);
        if (leaves.empty())
            return failure();

        auto u256Type = sir::U256Type::get(rewriter.getContext());
        Value zero = constU256(rewriter, loc, 0);
        Value one = constU256(rewriter, loc, 1);
        Value word = constU256(rewriter, loc, 32);
        const uint64_t elementWords = element.headBytes() / 32;

        Block *parentBlock = anchorOp->getBlock();
        Region *parentRegion = parentBlock->getParent();
        auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(anchorOp));
        auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});
        auto bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {u256Type}, {loc});

        rewriter.setInsertionPointToEnd(parentBlock);
        rewriter.create<sir::BrOp>(loc, ValueRange{zero}, condBlock);

        rewriter.setInsertionPointToStart(condBlock);
        Value iv = condBlock->getArgument(0);
        Value lt = rewriter.create<sir::LtOp>(loc, u256Type, iv, source.length);
        rewriter.create<sir::CondBrOp>(loc, lt, ValueRange{iv}, ValueRange{}, bodyBlock, afterBlock);

        rewriter.setInsertionPointToStart(bodyBlock);
        // bodyIv is the same loop counter passed through the true edge; it is
        // a distinct block argument because each SIR block owns its arguments.
        Value bodyIv = bodyBlock->getArgument(0);
        Value elementWordBase = bodyIv;
        if (elementWords != 1)
            elementWordBase = mulU256(rewriter, loc, bodyIv, constU256(rewriter, loc, elementWords));

        for (auto [leafIndex, leaf] : llvm::enumerate(leaves))
        {
            Value wordIndex = elementWordBase;
            if (leafIndex != 0)
                wordIndex = addU256(rewriter, loc, elementWordBase, constU256(rewriter, loc, static_cast<uint64_t>(leafIndex)));
            Value byteIndex = mulU256(rewriter, loc, wordIndex, word);

            Value rawValue;
            if (source.staticElementsUsePointerLayout)
            {
                // Pointer-backed memref slots are u256-sized in Ora's MLIR
                // lowering. If pointer width changes, the slot stride and
                // load width must change with it.
                Value elementPointerOffset = mulU256(rewriter, loc, bodyIv, word);
                if (source.sourceHasInlineLength)
                    elementPointerOffset = addU256(rewriter, loc, word, elementPointerOffset);
                Value elementPointerSlot = addPtrOffset(rewriter, loc, source.ptr, elementPointerOffset);
                Value elementPtrWord = rewriter.create<sir::LoadOp>(loc, u256Type, elementPointerSlot);
                Value elementPtr = rewriter.create<sir::BitcastOp>(loc, sir::PtrType::get(rewriter.getContext(), /*addrSpace*/ 1), elementPtrWord);
                Value fieldOffset = constU256(rewriter, loc, static_cast<uint64_t>(leaf.aggregateIndex) * 32ULL);
                Value fieldPtr = addPtrOffset(rewriter, loc, elementPtr, fieldOffset);
                rawValue = rewriter.create<sir::LoadOp>(loc, u256Type, fieldPtr);
            }
            else
            {
                Value sourceOffset = source.sourceHasInlineLength ? addU256(rewriter, loc, word, byteIndex) : byteIndex;
                Value sourcePtr = addPtrOffset(rewriter, loc, source.ptr, sourceOffset);
                rawValue = rewriter.create<sir::LoadOp>(loc, u256Type, sourcePtr);
            }

            // The loop already loaded the element word, so only encoding kind
            // and width are meaningful here; operand-source fields are ignored.
            AbiStaticLeaf elementLeaf{leaf.kind, leaf.width, 0, 0, false};
            Value destOffset = addU256(rewriter, loc, baseOffset, addU256(rewriter, loc, word, byteIndex));
            storeAbiWordAt(rewriter, loc, basePtr, destOffset, materializeAbiStaticWord(rewriter, loc, elementLeaf, rawValue));
        }

        Value next = addU256(rewriter, loc, bodyIv, one);
        rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        return success();
    }

    static LogicalResult emitAbiEncodingFromPointer(
        PatternRewriter &rewriter,
        Location loc,
        Operation *anchorOp,
        const AbiLayoutNode &node,
        Value sourcePtr,
        Value basePtr,
        Value baseOffset)
    {
        switch (node.kind)
        {
        case AbiLayoutKind::DynamicBytes:
            return emitDynamicBytesTailFromPtr(rewriter, loc, sourcePtr, basePtr, baseOffset);
        case AbiLayoutKind::DynamicArray:
        {
            auto u256Type = sir::U256Type::get(rewriter.getContext());
            DynamicArraySource source{sourcePtr, rewriter.create<sir::LoadOp>(loc, u256Type, sourcePtr), true, false, std::nullopt};
            return emitDynamicArrayTailFromSource(rewriter, loc, anchorOp, node, source, basePtr, baseOffset);
        }
        case AbiLayoutKind::Tuple:
            return emitAbiTupleEncodingFromPointer(rewriter, loc, anchorOp, node, sourcePtr, basePtr, baseOffset);
        case AbiLayoutKind::FixedArray:
            return emitFixedArrayEncodingFromPointer(rewriter, loc, anchorOp, node, sourcePtr, basePtr, baseOffset);
        case AbiLayoutKind::Static:
            return emitAbiStaticNodeFromPointer(rewriter, loc, node, sourcePtr, basePtr, baseOffset);
        }
        return failure();
    }

    static LogicalResult emitDynamicArrayTail(
        PatternRewriter &rewriter,
        Location loc,
        Operation *anchorOp,
        const AbiLayoutNode &node,
        ValueRange operands,
        Value basePtr,
        Value baseOffset)
    {
        if (node.children.empty())
            return failure();

        std::optional<DynamicArraySource> source = dynamicArraySource(rewriter, loc, anchorOp, node, operands);
        if (!source)
            return failure();
        return emitDynamicArrayTailFromSource(rewriter, loc, anchorOp, node, *source, basePtr, baseOffset);
    }

    static LogicalResult emitAbiTupleEncoding(
        PatternRewriter &rewriter,
        Location loc,
        Operation *anchorOp,
        const AbiLayoutNode &node,
        ValueRange operands,
        Value basePtr,
        Value baseOffset,
        AbiSizeCache &sizeCache)
    {
        Value tailOffset = constU256(rewriter, loc, node.headBytes());
        uint64_t headCursor = 0;
        for (const auto &child : node.children)
        {
            Value headSlotOffset = addU256(rewriter, loc, baseOffset, constU256(rewriter, loc, headCursor));
            if (child->isDynamic())
            {
                storeAbiWordAt(rewriter, loc, basePtr, headSlotOffset, tailOffset);
                Value childBaseOffset = addU256(rewriter, loc, baseOffset, tailOffset);
                Value childSize;
                if (isPointerBackedAggregateNode(*child))
                {
                    Value childPtr = ptrForAggregateNodeOperand(rewriter, loc, *child, operands);
                    if (!childPtr)
                        return failure();
                    if (failed(emitAbiEncodingFromPointer(rewriter, loc, anchorOp, *child, childPtr, basePtr, childBaseOffset)))
                        return failure();
                    childSize = abiEncodedSizeFromPointer(rewriter, loc, *child, childPtr);
                }
                else
                {
                    if (failed(emitAbiEncoding(rewriter, loc, anchorOp, *child, operands, basePtr, childBaseOffset, sizeCache)))
                        return failure();
                    childSize = abiEncodedSize(rewriter, loc, anchorOp, *child, operands, sizeCache);
                }
                if (!childSize)
                    return failure();
                tailOffset = addU256(rewriter, loc, tailOffset, childSize);
                headCursor += 32;
            }
            else
            {
                if (isPointerBackedAggregateNode(*child))
                {
                    Value childPtr = ptrForAggregateNodeOperand(rewriter, loc, *child, operands);
                    if (!childPtr)
                        return failure();
                    if (failed(emitAbiStaticNodeFromPointer(rewriter, loc, *child, childPtr, basePtr, headSlotOffset)))
                        return failure();
                }
                else
                {
                    if (failed(emitAbiStaticNode(rewriter, loc, *child, operands, basePtr, headSlotOffset)))
                        return failure();
                }
                headCursor += child->headBytes();
            }
        }
        return success();
    }

    static LogicalResult emitAbiEncoding(
        PatternRewriter &rewriter,
        Location loc,
        Operation *anchorOp,
        const AbiLayoutNode &node,
        ValueRange operands,
        Value basePtr,
        Value baseOffset,
        AbiSizeCache &sizeCache)
    {
        switch (node.kind)
        {
        case AbiLayoutKind::DynamicBytes:
            return emitDynamicBytesTail(rewriter, loc, node, operands, basePtr, baseOffset);
        case AbiLayoutKind::DynamicArray:
            return emitDynamicArrayTail(rewriter, loc, anchorOp, node, operands, basePtr, baseOffset);
        case AbiLayoutKind::Tuple:
            return emitAbiTupleEncoding(rewriter, loc, anchorOp, node, operands, basePtr, baseOffset, sizeCache);
        case AbiLayoutKind::Static:
            return emitAbiStaticNode(rewriter, loc, node, operands, basePtr, baseOffset);
        case AbiLayoutKind::FixedArray:
            if (node.isDynamic())
            {
                Value source = ptrForDynamicBytesOperand(rewriter, loc, node, operands);
                if (!source)
                    return failure();
                return emitFixedArrayEncodingFromPointer(rewriter, loc, anchorOp, node, source, basePtr, baseOffset);
            }
            return emitAbiStaticNode(rewriter, loc, node, operands, basePtr, baseOffset);
        }
        return failure();
    }

    template <typename OpT, typename AdaptorT>
    static LogicalResult lowerAbiEncode(
        OpT op,
        AdaptorT adaptor,
        ConversionPatternRewriter &rewriter,
        std::optional<mlir::IntegerAttr> selectorAttr)
    {
        auto *ctx = op.getContext();
        auto u256Type = sir::U256Type::get(ctx);
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

        auto layoutAttr = op->template getAttrOfType<mlir::StringAttr>("layout");
        if (!layoutAttr)
            return rewriter.notifyMatchFailure(op, "missing ABI layout attr");

        AbiLayoutNode root;
        AbiLayoutDslParser parser(layoutAttr.getValue());
        if (!parser.parse(root))
            return rewriter.notifyMatchFailure(op, "unsupported or malformed ABI layout attr");
        if (parser.getOperandCount() != adaptor.getOperands().size())
            return rewriter.notifyMatchFailure(op, "ABI layout operand count does not match converted operands");

        const uint64_t selectorBytes = selectorAttr.has_value() ? 4 : 0;
        AbiSizeCache sizeCache;
        Value payloadSize = abiEncodedSize(rewriter, op.getLoc(), op.getOperation(), root, adaptor.getOperands(), sizeCache);
        if (!payloadSize)
            return rewriter.notifyMatchFailure(op, "unable to compute ABI payload size");
        Value totalSize = selectorBytes == 0
                              ? payloadSize
                              : addU256(rewriter, op.getLoc(), constU256(rewriter, op.getLoc(), selectorBytes), payloadSize);
        Value basePtr = rewriter.create<sir::MallocOp>(op.getLoc(), ptrType, totalSize);

        if (selectorAttr)
        {
            llvm::APInt selectorWord = selectorAttr->getValue().zextOrTrunc(256);
            selectorWord = selectorWord.shl(224);
            Value selectorValue = constU256(rewriter, op.getLoc(), selectorWord);
            // The selector occupies the high 4 bytes of this word. The first
            // argument store starts at byte 4 and intentionally overwrites the
            // zero padding from this selector word.
            rewriter.create<sir::StoreOp>(op.getLoc(), basePtr, selectorValue);
        }

        if (failed(emitAbiEncoding(rewriter, op.getLoc(), op.getOperation(), root, adaptor.getOperands(), basePtr, constU256(rewriter, op.getLoc(), selectorBytes), sizeCache)))
            return rewriter.notifyMatchFailure(op, "unable to materialize ABI layout");

        Value result = rewriter.create<sir::BitcastOp>(op.getLoc(), u256Type, basePtr);
        rewriter.replaceOp(op, result);
        return success();
    }
}

// -----------------------------------------------------------------------------
// Lower ora.abi_encode - allocate payload buffer and store ABI args
// -----------------------------------------------------------------------------
LogicalResult ConvertAbiEncodeOp::matchAndRewrite(
    ora::AbiEncodeOp op,
    typename ora::AbiEncodeOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerAbiEncode(op, adaptor, rewriter, std::nullopt);
}

// -----------------------------------------------------------------------------
// Lower ora.abi_encode_with_selector - selector ++ ABI payload
// -----------------------------------------------------------------------------
LogicalResult ConvertAbiEncodeWithSelectorOp::matchAndRewrite(
    ora::AbiEncodeWithSelectorOp op,
    typename ora::AbiEncodeWithSelectorOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto selectorAttr = op->getAttrOfType<mlir::IntegerAttr>("selector");
    if (!selectorAttr)
        return rewriter.notifyMatchFailure(op, "missing selector attr");
    return lowerAbiEncode(op, adaptor, rewriter, selectorAttr);
}

// -----------------------------------------------------------------------------
// Lower ora.external_call - scalar/dynamic v1 call/staticcall boundary
// -----------------------------------------------------------------------------
LogicalResult ConvertExternalCallOp::matchAndRewrite(
    ora::ExternalCallOp op,
    typename ora::ExternalCallOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *ctx = op.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    auto callKind = op->getAttrOfType<mlir::StringAttr>("call_kind");
    if (!callKind)
        return rewriter.notifyMatchFailure(op, "missing call_kind attr");

    auto encodedDef = op.getCalldata().getDefiningOp<ora::AbiEncodeWithSelectorOp>();
    if (!encodedDef)
        return rewriter.notifyMatchFailure(op, "expected calldata from ora.abi_encode_with_selector");
    auto encodedArgTypes = encodedDef->getAttrOfType<mlir::ArrayAttr>("arg_types");
    if (!encodedArgTypes)
        return rewriter.notifyMatchFailure(op, "missing arg_types on ora.abi_encode_with_selector");
    auto selectorAttr = encodedDef->getAttrOfType<mlir::IntegerAttr>("selector");
    auto encodedLayout = encodedDef->getAttrOfType<mlir::StringAttr>("layout");
    if (!encodedLayout)
        return rewriter.notifyMatchFailure(op, "missing layout on ora.abi_encode_with_selector");
    AbiLayoutNode calldataRoot;
    AbiLayoutDslParser calldataParser(encodedLayout.getValue());
    if (!calldataParser.parse(calldataRoot))
        return rewriter.notifyMatchFailure(op, "unsupported ABI calldata layout");

    Value calldataPayloadLen = nullptr;
    if (calldataRoot.isDynamic())
    {
        SmallVector<Value, 4> encodedOperands;
        for (Value operand : encodedDef.getOperands())
        {
            Value remapped = rewriter.getRemappedValue(operand);
            if (!remapped)
                return rewriter.notifyMatchFailure(op, "unable to remap ABI encode operand for calldata length");
            encodedOperands.push_back(remapped);
        }
        AbiSizeCache sizeCache;
        calldataPayloadLen = abiEncodedSize(rewriter, op.getLoc(), encodedDef.getOperation(), calldataRoot, encodedOperands, sizeCache);
        if (!calldataPayloadLen)
            return rewriter.notifyMatchFailure(op, "unable to compute dynamic ABI calldata length");
    }
    else
    {
        calldataPayloadLen = constU256(rewriter, op.getLoc(), calldataRoot.headBytes());
    }
    Value calldataLen = addU256(rewriter, op.getLoc(), constU256(rewriter, op.getLoc(), 4), calldataPayloadLen);
    Value scratchReturnLen = rewriter.create<sir::ConstOp>(
        op.getLoc(),
        u256Type,
        mlir::IntegerAttr::get(ui64Type, 32));
    Value scratchReturnPtr = rewriter.create<sir::MallocOp>(op.getLoc(), ptrType, scratchReturnLen);
    Value calldataPtr = rewriter.create<sir::BitcastOp>(op.getLoc(), ptrType, adaptor.getCalldata());
    Value gas = ensureU256(rewriter, op.getLoc(), adaptor.getGas());
    Value target = ensureU256(rewriter, op.getLoc(), adaptor.getTarget());

    Operation *callOp = nullptr;
    if (callKind.getValue() == "staticcall")
    {
        callOp = rewriter.create<sir::StaticCallOp>(
            op.getLoc(),
            u256Type,
            gas,
            target,
            calldataPtr,
            calldataLen,
            scratchReturnPtr,
            scratchReturnLen);
    }
    else if (callKind.getValue() == "call")
    {
        Value zeroValue = rewriter.create<sir::ConstOp>(
            op.getLoc(),
            u256Type,
            mlir::IntegerAttr::get(ui64Type, 0));
        callOp = rewriter.create<sir::CallOp>(
            op.getLoc(),
            u256Type,
            gas,
            target,
            zeroValue,
            calldataPtr,
            calldataLen,
            scratchReturnPtr,
            scratchReturnLen);
    }
    else
    {
        return rewriter.notifyMatchFailure(op, "unsupported extern call kind");
    }

    if (!callOp)
        return rewriter.notifyMatchFailure(op, "failed to create SIR call op");

    callOp->setAttr("ora.trait_name", op->getAttr("trait_name"));
    callOp->setAttr("ora.method_name", op->getAttr("method_name"));
    callOp->setAttr("ora.call_kind", callKind);
    if (selectorAttr)
        callOp->setAttr("ora.selector", selectorAttr);

    Value callSuccess = callOp->getResult(0);
    Value fullReturnLen = rewriter.create<sir::ReturnDataSizeOp>(op.getLoc(), u256Type);
    Value fullReturnPtr = rewriter.create<sir::MallocOp>(op.getLoc(), ptrType, fullReturnLen);
    Value zeroOffset = rewriter.create<sir::ConstOp>(
        op.getLoc(),
        u256Type,
        mlir::IntegerAttr::get(ui64Type, 0));
    rewriter.create<sir::ReturnDataCopyOp>(op.getLoc(), fullReturnPtr, zeroOffset, fullReturnLen);
    Value fullReturnPtrU256 = rewriter.create<sir::BitcastOp>(op.getLoc(), u256Type, fullReturnPtr);
    rewriter.replaceOp(op, ValueRange{callSuccess, fullReturnPtrU256});
    return success();
}

