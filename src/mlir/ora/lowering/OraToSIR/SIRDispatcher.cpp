//===- SIRDispatcher.cpp - Build SIR Dispatcher ---------------------===//
//
// Creates a Solidity-style calldata dispatcher for public functions.
//
//===----------------------------------------------------------------------===//

#include "OraToSIR.h"
#include "patterns/AdtCarrierLayout.h"
#include "patterns/AbiLoweringCommon.h"
#include "patterns/ErrorUnionCarrierHelpers.h"
#include "patterns/LoweringHelpers.h"

#include "SIR/SIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <string>

using namespace mlir;

namespace mlir
{
    namespace ora
    {
        namespace
        {
            using abi_lowering::AbiLayoutKind;
            using abi_lowering::AbiLayoutNode;
            using abi_lowering::AbiStaticKind;
            using abi_lowering::canonicalAbiLayoutHeadSlots;
            using abi_lowering::canonicalAbiLayoutIsDynamic;
            using abi_lowering::canonicalAbiLayoutIsTupleLike;
            using abi_lowering::canonicalAbiLayoutStaticElementWordCount;
            using abi_lowering::canonicalAbiLayoutSupportsDynamicArray;
            using abi_lowering::isDynamicAddressArrayAbiNode;
            using abi_lowering::isDynamicBoolArrayAbiNode;
            using abi_lowering::isDynamicFixedBytesArrayAbiNode;
            using abi_lowering::isDynamicU256ArrayAbiNode;
            using abi_lowering::isStaticBoolAbiNode;
            using abi_lowering::AbiLayoutSyntax;
            using abi_lowering::parseAbiLayout;

            static std::optional<uint32_t> extractTaggedStmtId(Location loc, StringRef prefix)
            {
                if (loc == nullptr)
                    return std::nullopt;
                if (auto nameLoc = dyn_cast<NameLoc>(loc))
                {
                    StringRef name = nameLoc.getName().getValue();
                    if (name.starts_with(prefix))
                    {
                        uint32_t stmtId = 0;
                        if (!name.drop_front(prefix.size()).getAsInteger(10, stmtId))
                            return stmtId;
                    }
                    return extractTaggedStmtId(nameLoc.getChildLoc(), prefix);
                }
                if (auto callSite = dyn_cast<CallSiteLoc>(loc))
                {
                    if (auto callee = extractTaggedStmtId(callSite.getCallee(), prefix))
                        return callee;
                    return extractTaggedStmtId(callSite.getCaller(), prefix);
                }
                if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
                {
                    for (Location child : fusedLoc.getLocations())
                    {
                        if (auto stmtId = extractTaggedStmtId(child, prefix))
                            return stmtId;
                    }
                }
                return std::nullopt;
            }

            static Location stripProvenanceTags(Location loc)
            {
                if (loc == nullptr)
                    return loc;
                if (auto nameLoc = dyn_cast<NameLoc>(loc))
                {
                    StringRef name = nameLoc.getName().getValue();
                    if (name.starts_with("ora.stmt.") || name.starts_with("ora.origin_stmt.") || name.starts_with("ora.synthetic."))
                        return stripProvenanceTags(nameLoc.getChildLoc());
                    Location child = stripProvenanceTags(nameLoc.getChildLoc());
                    return NameLoc::get(nameLoc.getName(), child);
                }
                if (auto callSite = dyn_cast<CallSiteLoc>(loc))
                {
                    Location callee = stripProvenanceTags(callSite.getCallee());
                    Location caller = stripProvenanceTags(callSite.getCaller());
                    return CallSiteLoc::get(callee, caller);
                }
                if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
                {
                    SmallVector<Location, 4> children;
                    for (Location child : fusedLoc.getLocations())
                        children.push_back(stripProvenanceTags(child));
                    return FusedLoc::get(loc.getContext(), children, fusedLoc.getMetadata());
                }
                return loc;
            }

            static std::optional<uint32_t> parseErrorSelector(StringRef selector)
            {
                if (!selector.starts_with("0x") || selector.size() != 10)
                    return std::nullopt;

                uint32_t value = 0;
                for (char c : selector.drop_front(2))
                {
                    value <<= 4;
                    if (c >= '0' && c <= '9')
                        value |= (c - '0');
                    else if (c >= 'a' && c <= 'f')
                        value |= (c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F')
                        value |= (c - 'A' + 10);
                    else
                        return std::nullopt;
                }
                return value;
            }

            static Attribute lookupDictionaryAttr(DictionaryAttr dict, StringRef name)
            {
                if (!dict)
                    return {};
                for (NamedAttribute entry : dict)
                    if (entry.getName() == name)
                        return entry.getValue();
                return {};
            }

            static Location makeSyntheticOriginOnlyLoc(Location loc, StringRef syntheticKind)
            {
                MLIRContext *ctx = loc.getContext();
                Location base = stripProvenanceTags(loc);
                if (auto originStmt = extractTaggedStmtId(loc, "ora.origin_stmt."))
                {
                    std::string originTag = "ora.origin_stmt." + std::to_string(*originStmt);
                    base = NameLoc::get(StringAttr::get(ctx, originTag), base);
                }
                std::string syntheticTag = "ora.synthetic." + syntheticKind.str();
                return NameLoc::get(StringAttr::get(ctx, syntheticTag), base);
            }

            static Location findFunctionProvenanceLoc(func::FuncOp func)
            {
                for (Block &block : func.getBlocks())
                {
                    for (Operation &op : block.getOperations())
                    {
                        Location opLoc = op.getLoc();
                        if (extractTaggedStmtId(opLoc, "ora.origin_stmt.") || extractTaggedStmtId(opLoc, "ora.stmt."))
                            return opLoc;
                    }
                }
                return func.getLoc();
            }

            static uint64_t computeNamedMemoryReserveBytes(ModuleOp module)
            {
                if (!module)
                    return 0;
                auto slotsAttr = module->getAttrOfType<DictionaryAttr>("ora.global_slots");
                if (!slotsAttr || slotsAttr.empty())
                    return 0;

                uint64_t maxSlot = 0;
                bool found = false;
                for (NamedAttribute attr : slotsAttr)
                {
                    auto slotAttr = llvm::dyn_cast<IntegerAttr>(attr.getValue());
                    if (!slotAttr)
                        continue;
                    maxSlot = std::max<uint64_t>(maxSlot, slotAttr.getValue().getZExtValue());
                    found = true;
                }
                if (!found)
                    return 0;
                return (maxSlot + 1) * 32;
            }

            struct CalldataStaticDecode
            {
                Value payload;
                Value canonicalWord;
                SmallVector<std::pair<Value, lowering::AbiDecodeError>, 2> checks;
            };

            struct StrictDynamicCalldataValue
            {
                Value payload;
                Value nextExpectedOffset;
            };

            struct StrictDynamicBoundsBase
            {
                Value payloadBase;
                Value lengthOffset;
            };

            struct StrictDynamicBounds
            {
                Value wordSize;
                Value payloadBase;
                Value dynamicLen;
                Value padded;
                Value total;
                Value nextExpectedOffset;
            };

            enum class StrictDynamicCalldataKind
            {
                BytesLike,
                U256Array,
                AddressArray,
                BoolArray,
                FixedBytesArray,
            };

            static uint64_t strictDynamicCalldataCap(StrictDynamicCalldataKind kind)
            {
                return kind == StrictDynamicCalldataKind::BytesLike ? 1024 * 1024 : 32768;
            }

            static bool isFullWordStaticAggregateAbiLayout(const AbiLayoutNode &node)
            {
                if (node.kind == AbiLayoutKind::Static)
                {
                    return (node.staticKind == AbiStaticKind::Uint || node.staticKind == AbiStaticKind::Int) &&
                           node.width == 256;
                }
                if (node.kind == AbiLayoutKind::FixedArray)
                {
                    return node.children.size() == 1 &&
                           isFullWordStaticAggregateAbiLayout(*node.children.front());
                }
                if (canonicalAbiLayoutIsTupleLike(node))
                {
                    for (const auto &child : node.children)
                        if (!isFullWordStaticAggregateAbiLayout(*child))
                            return false;
                    return true;
                }
                return false;
            }

            static std::optional<uint64_t> fullWordStaticArrayElementWords(const AbiLayoutNode &node)
            {
                if (!canonicalAbiLayoutSupportsDynamicArray(node) || node.children.size() != 1)
                    return std::nullopt;
                const AbiLayoutNode &element = *node.children.front();
                if (!isFullWordStaticAggregateAbiLayout(element))
                    return std::nullopt;
                int64_t elementWords = canonicalAbiLayoutStaticElementWordCount(element);
                if (elementWords <= 1)
                    return std::nullopt;
                return static_cast<uint64_t>(elementWords);
            }

            static std::optional<uint64_t> dynamicTupleArrayElementHeadWords(const AbiLayoutNode &node)
            {
                if (node.kind != AbiLayoutKind::DynamicArray || node.children.size() != 1)
                    return std::nullopt;
                const AbiLayoutNode &element = *node.children.front();
                if (!canonicalAbiLayoutIsTupleLike(element) || !canonicalAbiLayoutIsDynamic(element))
                    return std::nullopt;

                for (const auto &child : element.children)
                    if (canonicalAbiLayoutIsDynamic(*child))
                    {
                        if (!isDynamicU256ArrayAbiNode(*child))
                            return std::nullopt;
                    }
                    else if (!isFullWordStaticAggregateAbiLayout(*child))
                    {
                        return std::nullopt;
                    }

                int64_t headWords = canonicalAbiLayoutHeadSlots(element);
                if (headWords <= 1)
                    return std::nullopt;
                return static_cast<uint64_t>(headWords);
            }

            static lowering::AbiDecodeError strictDynamicCalldataCapError(StrictDynamicCalldataKind kind)
            {
                return kind == StrictDynamicCalldataKind::BytesLike
                           ? lowering::AbiDecodeError::StringLengthExceeded
                           : lowering::AbiDecodeError::ArrayLengthExceeded;
            }

            static bool strictDynamicCalldataValidatesWordElements(StrictDynamicCalldataKind kind)
            {
                return kind == StrictDynamicCalldataKind::AddressArray ||
                       kind == StrictDynamicCalldataKind::BoolArray ||
                       kind == StrictDynamicCalldataKind::FixedBytesArray;
            }

            static FailureOr<lowering::AbiDecodeError> strictDynamicCalldataInvalidElementError(StrictDynamicCalldataKind kind)
            {
                switch (kind)
                {
                case StrictDynamicCalldataKind::AddressArray:
                    return lowering::AbiDecodeError::InvalidAddress;
                case StrictDynamicCalldataKind::BoolArray:
                    return lowering::AbiDecodeError::InvalidBoolValue;
                case StrictDynamicCalldataKind::FixedBytesArray:
                    return lowering::AbiDecodeError::InvalidFixedBytes;
                case StrictDynamicCalldataKind::BytesLike:
                case StrictDynamicCalldataKind::U256Array:
                    return failure();
                }
                return failure();
            }

            template <typename AddBlock, typename GetRevertBlock, typename GetBufferSize, typename MaterializeBase, typename ReadDynamicLen>
            static StrictDynamicBounds emitStrictDynamicBoundsPrefix(
                OpBuilder &builder,
                Location loc,
                Type u256Type,
                Value offsetWord,
                Value expectedOffset,
                StrictDynamicCalldataKind kind,
                AddBlock addBlock,
                GetRevertBlock getRevertBlock,
                GetBufferSize getBufferSize,
                MaterializeBase materializeBase,
                ReadDynamicLen readDynamicLen,
                bool permissive = false,
                uint64_t arrayElementWords = 1)
            {
                Block *offsetOkBlock = addBlock();
                Block *lengthBlock = addBlock();
                Block *capBlock = addBlock();
                Block *sizeBlock = addBlock();

                if (permissive)
                {
                    builder.create<sir::BrOp>(loc, ValueRange{}, offsetOkBlock);
                }
                else
                {
                    Value offsetOk = builder.create<sir::EqOp>(loc, u256Type, offsetWord, expectedOffset);
                    builder.create<sir::CondBrOp>(
                        loc,
                        offsetOk,
                        ValueRange{},
                        ValueRange{},
                        offsetOkBlock,
                        getRevertBlock(lowering::AbiDecodeError::NonCanonicalEncoding));
                }

                builder.setInsertionPointToEnd(offsetOkBlock);
                Value wordSize = lowering::constU256(builder, loc, 32);
                StrictDynamicBoundsBase base = materializeBase(wordSize);
                Value bufferSize = getBufferSize();
                Value lengthEnd = builder.create<sir::AddOp>(loc, u256Type, base.lengthOffset, wordSize);
                Value lengthMissing = builder.create<sir::LtOp>(loc, u256Type, bufferSize, lengthEnd);
                Value lengthPresent = builder.create<sir::IsZeroOp>(loc, u256Type, lengthMissing);
                builder.create<sir::CondBrOp>(
                    loc,
                    lengthPresent,
                    ValueRange{},
                    ValueRange{},
                    lengthBlock,
                    getRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                builder.setInsertionPointToEnd(lengthBlock);
                Value dynamicLen = readDynamicLen(base.payloadBase);
                Value exceedsUsize = builder.create<sir::GtOp>(
                    loc,
                    u256Type,
                    dynamicLen,
                    lowering::constU256(builder, loc, llvm::APInt::getLowBitsSet(256, 64)));
                Value lenFits = builder.create<sir::IsZeroOp>(loc, u256Type, exceedsUsize);
                builder.create<sir::CondBrOp>(
                    loc,
                    lenFits,
                    ValueRange{},
                    ValueRange{},
                    capBlock,
                    getRevertBlock(lowering::AbiDecodeError::LengthOverflow));

                builder.setInsertionPointToEnd(capBlock);
                const uint64_t capValue = strictDynamicCalldataCap(kind);
                const lowering::AbiDecodeError capError = strictDynamicCalldataCapError(kind);
                Value tooLong = builder.create<sir::GtOp>(
                    loc,
                    u256Type,
                    dynamicLen,
                    lowering::constU256(builder, loc, capValue));
                Value withinCap = builder.create<sir::IsZeroOp>(loc, u256Type, tooLong);
                builder.create<sir::CondBrOp>(
                    loc,
                    withinCap,
                    ValueRange{},
                    ValueRange{},
                    sizeBlock,
                    getRevertBlock(capError));

                builder.setInsertionPointToEnd(sizeBlock);
                Value padded = nullptr;
                Value total = nullptr;
                if (kind == StrictDynamicCalldataKind::BytesLike)
                {
                    padded = lowering::ceil32(builder, loc, dynamicLen);
                    total = builder.create<sir::AddOp>(loc, u256Type, padded, wordSize);
                }
                else
                {
                    Value elementStrideBytes = wordSize;
                    if (arrayElementWords > 1)
                        elementStrideBytes = lowering::constU256(builder, loc, arrayElementWords * 32ULL);
                    Value elementBytes = builder.create<sir::MulOp>(loc, u256Type, dynamicLen, elementStrideBytes);
                    total = builder.create<sir::AddOp>(loc, u256Type, elementBytes, wordSize);
                }

                return StrictDynamicBounds{
                    wordSize,
                    base.payloadBase,
                    dynamicLen,
                    padded,
                    total,
                    builder.create<sir::AddOp>(loc, u256Type, offsetWord, total),
                };
            }

            struct DynamicTupleChild
            {
                int64_t headByteOffset = 0;
                StrictDynamicCalldataKind kind = StrictDynamicCalldataKind::BytesLike;
                unsigned fixedBytesWidth = 0;
            };

            struct CalldataRefinementSpec
            {
                bool isSigned = false;
                bool hasMin = false;
                bool hasMax = false;
                bool nonZeroAddress = false;
                llvm::APInt min = llvm::APInt(256, 0);
                llvm::APInt max = llvm::APInt(256, 0);
            };

            static bool isPtrWordResultRepair(Type from, Type to)
            {
                return (isa<sir::PtrType>(from) && isa<sir::U256Type>(to)) ||
                       (isa<sir::U256Type>(from) && isa<sir::PtrType>(to));
            }

            static bool parseRefinementBoundToken(StringRef token, StringRef prefix, bool &isSigned, llvm::APInt &bound)
            {
                if (!token.starts_with(prefix))
                    return false;
                SmallVector<StringRef, 6> parts;
                token.split(parts, ':');
                if (parts.size() != 6)
                    return false;
                if (parts[1] == "s")
                    isSigned = true;
                else if (parts[1] == "u")
                    isSigned = false;
                else
                    return false;

                uint64_t limbs[4] = {};
                for (size_t i = 0; i < 4; ++i)
                {
                    if (parts[i + 2].getAsInteger(10, limbs[i]))
                        return false;
                }
                bound = lowering::abiDecodeBoundAPInt(limbs[0], limbs[1], limbs[2], limbs[3]);
                return true;
            }

            static bool parseCalldataRefinementSpec(StringRef text, CalldataRefinementSpec &out)
            {
                out = CalldataRefinementSpec{};
                if (text.empty())
                    return true;

                SmallVector<StringRef, 8> tokens;
                text.split(tokens, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
                for (StringRef token : tokens)
                {
                    if (token == "nonzero_address")
                    {
                        out.nonZeroAddress = true;
                        continue;
                    }
                    bool isSigned = false;
                    llvm::APInt bound(256, 0);
                    if (parseRefinementBoundToken(token, "min:", isSigned, bound))
                    {
                        out.isSigned = isSigned;
                        out.hasMin = true;
                        out.min = bound;
                        continue;
                    }
                    if (parseRefinementBoundToken(token, "max:", isSigned, bound))
                    {
                        out.isSigned = isSigned;
                        out.hasMax = true;
                        out.max = bound;
                        continue;
                    }
                    return false;
                }
                return true;
            }

            static Value calldataRefinementSatisfied(OpBuilder &builder, Location loc, const CalldataRefinementSpec &spec, Value canonicalWord)
            {
                auto u256Type = sir::U256Type::get(builder.getContext());
                Value valid;
                auto combine = [&](Value next) -> Value {
                    if (!next)
                        return valid;
                    if (!valid)
                        return next;
                    return builder.create<sir::AndOp>(loc, u256Type, valid, next);
                };

                if (spec.hasMin)
                    valid = combine(lowering::abiDecodeWordGte(builder, loc, canonicalWord, lowering::constU256(builder, loc, spec.min), spec.isSigned));
                if (spec.hasMax)
                    valid = combine(lowering::abiDecodeWordLte(builder, loc, canonicalWord, lowering::constU256(builder, loc, spec.max), spec.isSigned));
                if (spec.nonZeroAddress)
                {
                    Value zero = lowering::constU256(builder, loc, 0);
                    Value isZero = builder.create<sir::EqOp>(loc, u256Type, canonicalWord, zero);
                    valid = combine(builder.create<sir::IsZeroOp>(loc, u256Type, isZero));
                }
                return valid;
            }

            static int64_t dispatcherHeadSlotsForLayout(const AbiLayoutNode &layout)
            {
                return canonicalAbiLayoutIsDynamic(layout) ? 1 : canonicalAbiLayoutHeadSlots(layout);
            }

            static bool isStaticFixedArrayLayout(const AbiLayoutNode &layout)
            {
                return layout.kind == AbiLayoutKind::FixedArray &&
                       !canonicalAbiLayoutIsDynamic(layout) &&
                       canonicalAbiLayoutHeadSlots(layout) > 0;
            }

            static std::optional<CalldataStaticDecode> decodeStaticCalldataWord(OpBuilder &builder, Location loc, MLIRContext *ctx, const AbiLayoutNode &layout, Value word, uint64_t enumVariantCount = 0, bool needsRefinementCheck = false, bool permissive = false)
            {
                if (layout.kind != AbiLayoutKind::Static)
                    return std::nullopt;

                auto u256Type = sir::U256Type::get(ctx);
                auto makeDecode = [&](Value payload, Value canonicalWord, Value valid, lowering::AbiDecodeError error) {
                    CalldataStaticDecode decoded;
                    decoded.payload = payload;
                    decoded.canonicalWord = canonicalWord;
                    if (!permissive)
                        decoded.checks.push_back({valid, error});
                    return decoded;
                };
                auto addEnumRangeCheck = [&](CalldataStaticDecode decoded) {
                    if (enumVariantCount == 0)
                        return decoded;
                    Value count = lowering::constU256(builder, loc, enumVariantCount);
                    Value inRange = builder.create<sir::LtOp>(loc, u256Type, decoded.payload, count);
                    decoded.checks.push_back({inRange, lowering::AbiDecodeError::EnumOutOfRange});
                    return decoded;
                };
                switch (layout.staticKind)
                {
                case AbiStaticKind::Bool:
                {
                    Value payload = permissive ? lowering::boolAbiWordPermissivePayload(builder, loc, word) : word;
                    return makeDecode(payload, payload, lowering::boolAbiWordIsCanonical(builder, loc, word), lowering::AbiDecodeError::InvalidBoolValue);
                }
                case AbiStaticKind::Address:
                {
                    Value payload = lowering::maskLowBits(builder, loc, word, 160);
                    return makeDecode(payload, payload, builder.create<sir::EqOp>(loc, u256Type, word, payload), lowering::AbiDecodeError::InvalidAddress);
                }
                case AbiStaticKind::Uint:
                {
                    if (layout.width >= 256)
                    {
                        if (enumVariantCount == 0 && !needsRefinementCheck)
                            return std::nullopt;
                        CalldataStaticDecode decoded;
                        decoded.payload = word;
                        decoded.canonicalWord = word;
                        return addEnumRangeCheck(decoded);
                    }
                    Value payload = lowering::maskLowBits(builder, loc, word, layout.width);
                    return addEnumRangeCheck(makeDecode(payload, payload, builder.create<sir::EqOp>(loc, u256Type, word, payload), lowering::AbiDecodeError::NonCanonicalPadding));
                }
                case AbiStaticKind::Int:
                {
                    if (layout.width >= 256)
                    {
                        if (enumVariantCount == 0 && !needsRefinementCheck)
                            return std::nullopt;
                        CalldataStaticDecode decoded;
                        decoded.payload = word;
                        decoded.canonicalWord = word;
                        return addEnumRangeCheck(decoded);
                    }
                    if (layout.width == 0 || layout.width % 8 != 0)
                        return std::nullopt;
                    Value byteIndex = lowering::constU256(builder, loc, (layout.width / 8) - 1);
                    Value expected = builder.create<sir::SignExtendOp>(loc, u256Type, byteIndex, word);
                    return addEnumRangeCheck(makeDecode(expected, expected, builder.create<sir::EqOp>(loc, u256Type, word, expected), lowering::AbiDecodeError::NonCanonicalPadding));
                }
                case AbiStaticKind::FixedBytes:
                {
                    if (layout.width >= 32 || layout.width == 0)
                        return std::nullopt;
                    uint64_t shiftBits = static_cast<uint64_t>(32 - layout.width) * 8ULL;
                    Value shift = lowering::constU256(builder, loc, shiftBits);
                    Value payload = builder.create<sir::ShrOp>(loc, u256Type, shift, word);
                    Value expected = builder.create<sir::ShlOp>(loc, u256Type, shift, payload);
                    return makeDecode(payload, expected, builder.create<sir::EqOp>(loc, u256Type, word, expected), lowering::AbiDecodeError::InvalidFixedBytes);
                }
                }
                return std::nullopt;
            }

            static Value materializeAbiReturnStaticWord(OpBuilder &builder, Location loc, MLIRContext *ctx, const AbiLayoutNode &layout, Value payload)
            {
                if (layout.kind != AbiLayoutKind::Static)
                    return payload;

                auto u256Type = sir::U256Type::get(ctx);
                if (layout.staticKind == AbiStaticKind::Int &&
                    layout.width > 0 &&
                    layout.width < 256 &&
                    layout.width % 8 == 0)
                {
                    Value byteIndex = lowering::constU256(builder, loc, (layout.width / 8) - 1);
                    return builder.create<sir::SignExtendOp>(loc, u256Type, byteIndex, payload).getResult();
                }

                if (layout.staticKind == AbiStaticKind::FixedBytes &&
                    layout.width > 0 &&
                    layout.width < 32)
                {
                    // Ora fixed-bytes values carry their bytes in the low bits
                    // after ABI decode. The ABI word is left-aligned.
                    Value shift = lowering::constU256(builder, loc, static_cast<uint64_t>(32 - layout.width) * 8ULL);
                    return builder.create<sir::ShlOp>(loc, u256Type, shift, payload).getResult();
                }

                return payload;
            }

            static Value computeAbiEncodedSize(
                OpBuilder &builder,
                Location loc,
                MLIRContext *ctx,
                Value basePtr,
                const AbiLayoutNode &layout)
            {
                auto u256Type = sir::U256Type::get(ctx);
                auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
                Value c32 = lowering::constU256(builder, loc, 32);

                if (canonicalAbiLayoutSupportsDynamicArray(layout))
                {
                    Value length = builder.create<sir::LoadOp>(loc, u256Type, basePtr);
                    Value elemWords = lowering::constU256(builder, loc, canonicalAbiLayoutStaticElementWordCount(layout));
                    Value words = builder.create<sir::MulOp>(loc, u256Type, length, elemWords);
                    Value lenBytes = builder.create<sir::MulOp>(loc, u256Type, words, c32);
                    return builder.create<sir::AddOp>(loc, u256Type, lenBytes, c32);
                }

                if (!canonicalAbiLayoutIsTupleLike(layout))
                {
                    if (layout.kind == AbiLayoutKind::DynamicBytes)
                    {
                        Value length = builder.create<sir::LoadOp>(loc, u256Type, basePtr);
                        Value padded = lowering::ceil32(builder, loc, length);
                        return builder.create<sir::AddOp>(loc, u256Type, padded, c32);
                    }
                    if (layout.kind == AbiLayoutKind::DynamicArray)
                    {
                        Value length = builder.create<sir::LoadOp>(loc, u256Type, basePtr);
                        Value lenBytes = builder.create<sir::MulOp>(loc, u256Type, length, c32);
                        return builder.create<sir::AddOp>(loc, u256Type, lenBytes, c32);
                    }
                    return c32;
                }

                Value total = lowering::constU256(builder, loc, canonicalAbiLayoutHeadSlots(layout) * 32);
                int64_t headOffset = 0;
                for (const auto &fieldPtr : layout.children)
                {
                    const AbiLayoutNode &field = *fieldPtr;
                    if (canonicalAbiLayoutIsDynamic(field))
                    {
                        Value headOff = lowering::constU256(builder, loc, headOffset);
                        Value offPtr = builder.create<sir::AddPtrOp>(loc, ptrType, basePtr, headOff);
                        Value relOff = builder.create<sir::LoadOp>(loc, u256Type, offPtr);
                        Value childPtr = builder.create<sir::AddPtrOp>(loc, ptrType, basePtr, relOff);
                        Value childSize = computeAbiEncodedSize(builder, loc, ctx, childPtr, field);
                        total = builder.create<sir::AddOp>(loc, u256Type, total, childSize);
                        headOffset += 32;
                    }
                    else
                    {
                        headOffset += canonicalAbiLayoutHeadSlots(field) * 32;
                    }
                }
                return total;
            }

            struct AbiReturnBuffer
            {
                Value ptr;
                Value size;
            };

            static Value computePointerBackedAbiEncodedSize(
                OpBuilder &builder,
                Location loc,
                MLIRContext *ctx,
                Value sourcePtr,
                const AbiLayoutNode &layout);

            static LogicalResult emitPointerBackedAbiEncoding(
                OpBuilder &builder,
                Location loc,
                MLIRContext *ctx,
                Value sourcePtr,
                const AbiLayoutNode &layout,
                Value destPtr);

            static LogicalResult emitPointerBackedStaticAbiWords(
                OpBuilder &builder,
                Location loc,
                MLIRContext *ctx,
                const AbiLayoutNode &layout,
                Value sourcePtr,
                Value sourceOffset,
                Value destPtr,
                Value destOffset)
            {
                auto u256Type = sir::U256Type::get(ctx);
                auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

                if (layout.kind == AbiLayoutKind::Static)
                {
                    Value src = builder.create<sir::AddPtrOp>(loc, ptrType, sourcePtr, sourceOffset);
                    Value dst = builder.create<sir::AddPtrOp>(loc, ptrType, destPtr, destOffset);
                    Value raw = builder.create<sir::LoadOp>(loc, u256Type, src);
                    builder.create<sir::StoreOp>(loc, dst, materializeAbiReturnStaticWord(builder, loc, ctx, layout, raw));
                    return success();
                }

                if (canonicalAbiLayoutIsTupleLike(layout) && !canonicalAbiLayoutIsDynamic(layout))
                {
                    Value sourceCursor = sourceOffset;
                    Value destCursor = destOffset;
                    for (const auto &child : layout.children)
                    {
                        if (failed(emitPointerBackedStaticAbiWords(builder, loc, ctx, *child, sourcePtr, sourceCursor, destPtr, destCursor)))
                            return failure();
                        Value childBytes = lowering::constU256(builder, loc, canonicalAbiLayoutHeadSlots(*child) * 32);
                        sourceCursor = builder.create<sir::AddOp>(loc, u256Type, sourceCursor, childBytes);
                        destCursor = builder.create<sir::AddOp>(loc, u256Type, destCursor, childBytes);
                    }
                    return success();
                }

                if (layout.kind == AbiLayoutKind::FixedArray && !canonicalAbiLayoutIsDynamic(layout) && layout.children.size() == 1)
                {
                    const AbiLayoutNode &element = *layout.children.front();
                    int64_t elementSlots = canonicalAbiLayoutHeadSlots(element);
                    if (elementSlots <= 0)
                        return failure();
                    Value elementBytes = lowering::constU256(builder, loc, elementSlots * 32);
                    Value sourceCursor = sourceOffset;
                    Value destCursor = destOffset;
                    for (unsigned i = 0; i < layout.arrayLen; ++i)
                    {
                        if (failed(emitPointerBackedStaticAbiWords(builder, loc, ctx, element, sourcePtr, sourceCursor, destPtr, destCursor)))
                            return failure();
                        sourceCursor = builder.create<sir::AddOp>(loc, u256Type, sourceCursor, elementBytes);
                        destCursor = builder.create<sir::AddOp>(loc, u256Type, destCursor, elementBytes);
                    }
                    return success();
                }

                int64_t words = canonicalAbiLayoutHeadSlots(layout);
                if (words <= 0)
                    return failure();
                Value bytes = lowering::constU256(builder, loc, words * 32);
                Value src = builder.create<sir::AddPtrOp>(loc, ptrType, sourcePtr, sourceOffset);
                Value dst = builder.create<sir::AddPtrOp>(loc, ptrType, destPtr, destOffset);
                builder.create<sir::MCopyOp>(loc, dst, src, bytes);
                return success();
            }

            static Value computePointerBackedAbiEncodedSize(
                OpBuilder &builder,
                Location loc,
                MLIRContext *ctx,
                Value sourcePtr,
                const AbiLayoutNode &layout)
            {
                auto u256Type = sir::U256Type::get(ctx);
                auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
                Value c32 = lowering::constU256(builder, loc, 32);

                if (layout.kind == AbiLayoutKind::DynamicBytes)
                {
                    Value length = builder.create<sir::LoadOp>(loc, u256Type, sourcePtr);
                    Value padded = lowering::ceil32(builder, loc, length);
                    return builder.create<sir::AddOp>(loc, u256Type, padded, c32);
                }

                if (canonicalAbiLayoutSupportsDynamicArray(layout))
                {
                    Value length = builder.create<sir::LoadOp>(loc, u256Type, sourcePtr);
                    Value elemWords = lowering::constU256(builder, loc, canonicalAbiLayoutStaticElementWordCount(*layout.children.front()));
                    Value words = builder.create<sir::MulOp>(loc, u256Type, length, elemWords);
                    Value lenBytes = builder.create<sir::MulOp>(loc, u256Type, words, c32);
                    return builder.create<sir::AddOp>(loc, u256Type, lenBytes, c32);
                }

                if (canonicalAbiLayoutIsTupleLike(layout))
                {
                    Value total = lowering::constU256(builder, loc, canonicalAbiLayoutHeadSlots(layout) * 32);
                    Value sourceCursor = lowering::constU256(builder, loc, 0);
                    for (const auto &childPtr : layout.children)
                    {
                        const AbiLayoutNode &child = *childPtr;
                        if (canonicalAbiLayoutIsDynamic(child))
                        {
                            Value childSlot = builder.create<sir::AddPtrOp>(loc, ptrType, sourcePtr, sourceCursor);
                            Value childPtrWord = builder.create<sir::LoadOp>(loc, u256Type, childSlot);
                            Value childSource = builder.create<sir::BitcastOp>(loc, ptrType, childPtrWord);
                            Value childSize = computePointerBackedAbiEncodedSize(builder, loc, ctx, childSource, child);
                            total = builder.create<sir::AddOp>(loc, u256Type, total, childSize);
                            sourceCursor = builder.create<sir::AddOp>(loc, u256Type, sourceCursor, c32);
                        }
                        else
                        {
                            Value childBytes = lowering::constU256(builder, loc, canonicalAbiLayoutHeadSlots(child) * 32);
                            sourceCursor = builder.create<sir::AddOp>(loc, u256Type, sourceCursor, childBytes);
                        }
                    }
                    return total;
                }

                int64_t words = canonicalAbiLayoutHeadSlots(layout);
                return lowering::constU256(builder, loc, (words > 0 ? words : 1) * 32);
            }

            static LogicalResult emitPointerBackedAbiEncoding(
                OpBuilder &builder,
                Location loc,
                MLIRContext *ctx,
                Value sourcePtr,
                const AbiLayoutNode &layout,
                Value destPtr)
            {
                auto u256Type = sir::U256Type::get(ctx);
                auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
                Value c32 = lowering::constU256(builder, loc, 32);

                if (layout.kind == AbiLayoutKind::DynamicBytes)
                {
                    Value length = builder.create<sir::LoadOp>(loc, u256Type, sourcePtr);
                    Value padded = lowering::ceil32(builder, loc, length);
                    builder.create<sir::StoreOp>(loc, destPtr, length);
                    Value srcPayload = builder.create<sir::AddPtrOp>(loc, ptrType, sourcePtr, c32);
                    Value dstPayload = builder.create<sir::AddPtrOp>(loc, ptrType, destPtr, c32);
                    builder.create<sir::MCopyOp>(loc, dstPayload, srcPayload, padded);
                    return success();
                }

                if (canonicalAbiLayoutSupportsDynamicArray(layout))
                {
                    Value length = builder.create<sir::LoadOp>(loc, u256Type, sourcePtr);
                    Value elemWords = lowering::constU256(builder, loc, canonicalAbiLayoutStaticElementWordCount(*layout.children.front()));
                    Value words = builder.create<sir::MulOp>(loc, u256Type, length, elemWords);
                    Value payloadBytes = builder.create<sir::MulOp>(loc, u256Type, words, c32);
                    builder.create<sir::StoreOp>(loc, destPtr, length);
                    Value srcPayload = builder.create<sir::AddPtrOp>(loc, ptrType, sourcePtr, c32);
                    Value dstPayload = builder.create<sir::AddPtrOp>(loc, ptrType, destPtr, c32);
                    builder.create<sir::MCopyOp>(loc, dstPayload, srcPayload, payloadBytes);
                    return success();
                }

                if (canonicalAbiLayoutIsTupleLike(layout))
                {
                    Value tailOffset = lowering::constU256(builder, loc, canonicalAbiLayoutHeadSlots(layout) * 32);
                    Value sourceCursor = lowering::constU256(builder, loc, 0);
                    Value headCursor = lowering::constU256(builder, loc, 0);
                    for (const auto &childPtr : layout.children)
                    {
                        const AbiLayoutNode &child = *childPtr;
                        if (canonicalAbiLayoutIsDynamic(child))
                        {
                            Value headSlot = builder.create<sir::AddPtrOp>(loc, ptrType, destPtr, headCursor);
                            builder.create<sir::StoreOp>(loc, headSlot, tailOffset);

                            Value childSlot = builder.create<sir::AddPtrOp>(loc, ptrType, sourcePtr, sourceCursor);
                            Value childPtrWord = builder.create<sir::LoadOp>(loc, u256Type, childSlot);
                            Value childSource = builder.create<sir::BitcastOp>(loc, ptrType, childPtrWord);
                            Value childDest = builder.create<sir::AddPtrOp>(loc, ptrType, destPtr, tailOffset);
                            if (failed(emitPointerBackedAbiEncoding(builder, loc, ctx, childSource, child, childDest)))
                                return failure();
                            Value childSize = computePointerBackedAbiEncodedSize(builder, loc, ctx, childSource, child);
                            tailOffset = builder.create<sir::AddOp>(loc, u256Type, tailOffset, childSize);
                            sourceCursor = builder.create<sir::AddOp>(loc, u256Type, sourceCursor, c32);
                            headCursor = builder.create<sir::AddOp>(loc, u256Type, headCursor, c32);
                        }
                        else
                        {
                            if (failed(emitPointerBackedStaticAbiWords(builder, loc, ctx, child, sourcePtr, sourceCursor, destPtr, headCursor)))
                                return failure();
                            Value childBytes = lowering::constU256(builder, loc, canonicalAbiLayoutHeadSlots(child) * 32);
                            sourceCursor = builder.create<sir::AddOp>(loc, u256Type, sourceCursor, childBytes);
                            headCursor = builder.create<sir::AddOp>(loc, u256Type, headCursor, childBytes);
                        }
                    }
                    return success();
                }

                return emitPointerBackedStaticAbiWords(
                    builder,
                    loc,
                    ctx,
                    layout,
                    sourcePtr,
                    lowering::constU256(builder, loc, 0),
                    destPtr,
                    lowering::constU256(builder, loc, 0));
            }

            static FailureOr<AbiReturnBuffer> materializeSingleDynamicTupleAbiReturn(
                OpBuilder &builder,
                Location loc,
                MLIRContext *ctx,
                Value sourcePtrWord,
                const AbiLayoutNode &layout)
            {
                auto u256Type = sir::U256Type::get(ctx);
                auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
                Value wordSize = lowering::constU256(builder, loc, 32);
                Value sourcePtr = builder.create<sir::BitcastOp>(loc, ptrType, sourcePtrWord);
                Value tupleSize = computePointerBackedAbiEncodedSize(builder, loc, ctx, sourcePtr, layout);
                Value size = builder.create<sir::AddOp>(loc, u256Type, wordSize, tupleSize);
                Value retPtr = builder.create<sir::SAllocAnyOp>(loc, ptrType, size);
                builder.create<sir::StoreOp>(loc, retPtr, wordSize);
                Value tupleDest = builder.create<sir::AddPtrOp>(loc, ptrType, retPtr, wordSize);
                if (failed(emitPointerBackedAbiEncoding(builder, loc, ctx, sourcePtr, layout, tupleDest)))
                    return failure();
                return AbiReturnBuffer{retPtr, size};
            }

            static AbiReturnBuffer materializeSingleDynamicAbiReturn(
                OpBuilder &builder,
                Location loc,
                MLIRContext *ctx,
                Value sourcePtrWord,
                bool byteLengthPayload)
            {
                auto u256Type = sir::U256Type::get(ctx);
                auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
                Value wordSize = lowering::constU256(builder, loc, 32);
                Value twoWords = lowering::constU256(builder, loc, 64);
                Value sourcePtr = builder.create<sir::BitcastOp>(loc, ptrType, sourcePtrWord);
                Value length = builder.create<sir::LoadOp>(loc, u256Type, sourcePtr);
                Value payloadBytes = byteLengthPayload
                                         ? length
                                         : builder.create<sir::MulOp>(loc, u256Type, length, wordSize).getResult();
                Value paddedPayloadBytes = byteLengthPayload
                                               ? lowering::ceil32(builder, loc, payloadBytes)
                                               : payloadBytes;
                Value size = builder.create<sir::AddOp>(loc, u256Type, twoWords, paddedPayloadBytes);
                Value retPtr = builder.create<sir::SAllocAnyOp>(loc, ptrType, size);
                builder.create<sir::StoreOp>(loc, retPtr, wordSize);
                Value lengthSlot = builder.create<sir::AddPtrOp>(loc, ptrType, retPtr, wordSize);
                builder.create<sir::StoreOp>(loc, lengthSlot, length);
                Value sourcePayload = builder.create<sir::AddPtrOp>(loc, ptrType, sourcePtr, wordSize);
                Value destPayload = builder.create<sir::AddPtrOp>(loc, ptrType, retPtr, twoWords);
                builder.create<sir::MCopyOp>(loc, destPayload, sourcePayload, payloadBytes);
                return AbiReturnBuffer{retPtr, size};
            }

            struct ErrorInfo
            {
                uint64_t id = 0;
                uint32_t selector = 0;
                uint64_t paramCount = 0;
            };

            struct PubFuncInfo
            {
                func::FuncOp func;
                Location provenanceLoc;
                uint32_t selector = 0;
                unsigned argCount = 0;
                unsigned retCount = 0;
                bool returnsErrorUnion = false;
                bool permissiveAbiDecode = false;
                SmallVector<std::string, 8> abiParamLayouts;
                SmallVector<uint64_t, 8> abiParamEnumCounts;
                SmallVector<std::string, 8> abiParamRefinements;
                SmallVector<std::string, 8> resultInputModes;
                SmallVector<int64_t, 8> resultInputErrorIds;
                SmallVector<ErrorInfo, 8> returnErrors;
                bool hasReturnErrorMetadata = false;
                bool hasAbiReturn = false;
                int64_t abiReturnWords = -1;
                std::string abiReturnLayout;
                int64_t minHeadBytes = 0;
                SmallVector<Type, 8> inputTypes;

                PubFuncInfo(func::FuncOp func, Location provenanceLoc)
                    : func(func), provenanceLoc(provenanceLoc)
                {
                }
            };

            static Value getShiftedSelectorConst(OpBuilder &builder, Location loc, MLIRContext *, uint32_t selector)
            {
                llvm::APInt selectorWord(256, selector);
                selectorWord = selectorWord.shl(224);
                return lowering::constU256(builder, loc, selectorWord);
            }

            struct SIRDispatcherPass : public PassWrapper<SIRDispatcherPass, OperationPass<ModuleOp>>
            {
                static void setResultName(Operation *op, StringRef name)
                {
                    if (!op || name.empty())
                        return;
                    op->setAttr("sir.result_name_0", StringAttr::get(op->getContext(), name));
                }

                static void setBlockName(Block *block, StringRef name)
                {
                    if (!block || name.empty())
                        return;
                    if (block->empty())
                        return;
                    block->back().setAttr("sir.block_name", StringAttr::get(block->getParent()->getContext(), name));
                }

                static void setBlockOrder(Block *block, int64_t order)
                {
                    if (!block)
                        return;
                    if (block->empty())
                        return;
                    block->back().setAttr("sir.block_order", IntegerAttr::get(mlir::IntegerType::get(block->getParent()->getContext(), 64), order));
                }

                static Value getConst(OpBuilder &builder,
                                      Location loc,
                                      Type type,
                                      mlir::IntegerType i64Type,
                                      int64_t value,
                                      DenseMap<Block *, DenseMap<int64_t, Value>> &cache,
                                      Block *block,
                                      StringRef name = "")
                {
                    auto &blockCache = cache[block];
                    auto it = blockCache.find(value);
                    if (it != blockCache.end())
                        return it->second;
                    Value v = builder.create<sir::ConstOp>(loc, type, IntegerAttr::get(i64Type, value));
                    if (!name.empty())
                        setResultName(v.getDefiningOp(), name);
                    blockCache.try_emplace(value, v);
                    return v;
                }

                void runOnOperation() override
                {
                    ModuleOp module = getOperation();
                    MLIRContext *ctx = module.getContext();
                    OpBuilder builder(ctx);
                    Location loc = builder.getUnknownLoc();

                    // Preserve user-defined init (constructor) if present.
                    func::FuncOp userInit;
                    if (auto sym = SymbolTable::lookupSymbolIn(module, StringRef("init")))
                    {
                        if (auto f = dyn_cast<func::FuncOp>(sym))
                        {
                            userInit = f;
                            std::string baseName = "__ora_user_init";
                            std::string newName = baseName;
                            int suffix = 0;
                            while (SymbolTable::lookupSymbolIn(module, newName))
                            {
                                ++suffix;
                                newName = baseName + "_" + std::to_string(suffix);
                            }
                            userInit.setName(newName);
                            userInit->setAttr("ora.visibility", StringAttr::get(ctx, "private"));
                            userInit->removeAttr("ora.selector");
                            userInit->removeAttr("ora.init");
                        }
                    }

                    auto u256Type = sir::U256Type::get(ctx);
                    auto ptrType = sir::PtrType::get(ctx, 1);
                    auto i64Type = builder.getI64Type();

                    SmallVector<ErrorInfo, 8> abiErrors;
                    llvm::DenseSet<uint64_t> seenErrorIds;
                    auto addErrorInfo = [&](uint64_t id, uint32_t selector, uint64_t paramCount) {
                        if (!seenErrorIds.insert(id).second)
                            return;
                        abiErrors.push_back(ErrorInfo{id, selector, paramCount});
                    };
                    module.walk([&](sir::ErrorDeclOp decl) {
                        auto idAttr = decl->getAttrOfType<IntegerAttr>("sir.error_id");
                        auto selectorAttr = decl->getAttrOfType<StringAttr>("sir.error_selector");
                        auto paramTypes = decl->getAttrOfType<ArrayAttr>("sir.param_types");
                        if (!idAttr || !selectorAttr || !paramTypes)
                            return;

                        auto selector = parseErrorSelector(selectorAttr.getValue());
                        if (!selector)
                            return;
                        addErrorInfo(
                            idAttr.getValue().getZExtValue(),
                            *selector,
                            static_cast<uint64_t>(paramTypes.size()));
                    });
                    if (auto idDict = module->getAttrOfType<DictionaryAttr>("sir.error_ids"))
                    {
                        auto selectorDict = module->getAttrOfType<DictionaryAttr>("sir.error_selectors");
                        auto paramCountDict = module->getAttrOfType<DictionaryAttr>("sir.error_param_counts");
                        for (NamedAttribute idEntry : idDict)
                        {
                            auto idAttr = dyn_cast<IntegerAttr>(idEntry.getValue());
                            auto selectorAttr = dyn_cast_or_null<StringAttr>(lookupDictionaryAttr(selectorDict, idEntry.getName()));
                            if (!idAttr || !selectorAttr)
                                continue;
                            auto selector = parseErrorSelector(selectorAttr.getValue());
                            if (!selector)
                                continue;
                            uint64_t paramCount = 0;
                            if (auto paramCountAttr = dyn_cast_or_null<IntegerAttr>(lookupDictionaryAttr(paramCountDict, idEntry.getName())))
                                paramCount = paramCountAttr.getValue().getZExtValue();
                            addErrorInfo(idAttr.getValue().getZExtValue(), *selector, paramCount);
                        }
                    }

                    // Rewrite all non-entry functions: public ABI functions
                    // return (ptr,len), but private scalar helpers return the
                    // scalar word itself. Keeping private helpers ABI-shaped
                    // makes internal callers observe memory pointers as values.
                    for (func::FuncOp func : module.getOps<func::FuncOp>())
                    {
                        if (func.getName() == "init" || func.getName() == "main")
                            continue;
                        if (userInit && func == userInit)
                            continue;

                        auto vis = func->getAttrOfType<StringAttr>("ora.visibility");
                        auto ft = func.getFunctionType();
                        const bool privateScalarReturn =
                            vis && vis.getValue() == "private" &&
                            ft.getNumResults() == 1 &&
                            !isa<sir::PtrType>(ft.getResult(0));
                        const bool privateErrorUnionReturn =
                            vis && vis.getValue() == "private" &&
                            ft.getNumResults() == 2 &&
                            func->getAttrOfType<BoolAttr>("ora.returns_error_union");

                        bool hasReturn = false;
                        for (Block &block : func.getBlocks())
                        {
                            if (!block.getTerminator())
                                continue;
                            if (auto ret = dyn_cast<sir::ReturnOp>(block.getTerminator()))
                            {
                                builder.setInsertionPoint(ret);
                                if (privateErrorUnionReturn)
                                {
                                    Value ptr = ret.getPtr();
                                    Value tag = builder.create<sir::LoadOp>(ret.getLoc(), u256Type, ptr);
                                    Value c32 = lowering::constU256(builder, ret.getLoc(), 32);
                                    Value payloadPtr = builder.create<sir::AddPtrOp>(ret.getLoc(), ptrType, ptr, c32);
                                    Value payload = builder.create<sir::LoadOp>(ret.getLoc(), u256Type, payloadPtr);
                                    builder.create<sir::IRetOp>(ret.getLoc(), ValueRange{tag, payload});
                                }
                                else if (privateScalarReturn)
                                {
                                    Value scalar = builder.create<sir::LoadOp>(ret.getLoc(), u256Type, ret.getPtr());
                                    builder.create<sir::IRetOp>(ret.getLoc(), ValueRange{scalar});
                                }
                                else
                                {
                                    Value ptr = ret.getPtr();
                                    Value len = ret.getLen();
                                    Value ptr_u = builder.create<sir::BitcastOp>(ret.getLoc(), u256Type, ptr);
                                    builder.create<sir::IRetOp>(ret.getLoc(), ValueRange{ptr_u, len});
                                }
                                ret.erase();
                                hasReturn = true;
                            }
                        }
                        if (hasReturn)
                        {
                            SmallVector<Type, 4> results;
                            if (privateErrorUnionReturn)
                            {
                                results.push_back(u256Type);
                                results.push_back(u256Type);
                            }
                            else if (privateScalarReturn)
                            {
                                results.push_back(u256Type);
                            }
                            else
                            {
                                results.push_back(u256Type);
                                results.push_back(u256Type);
                            }
                            auto newType = builder.getFunctionType(ft.getInputs(), results);
                            func.setType(newType);
                        }
                    }

                    // Collect public functions and selectors (after return rewriting).
                    SmallVector<PubFuncInfo, 8> pubFuncs;
                    for (func::FuncOp func : module.getOps<func::FuncOp>())
                    {
                        auto vis = func->getAttrOfType<StringAttr>("ora.visibility");
                        if (!vis || vis.getValue() != "pub")
                            continue;
                        if (func.getName() == "init")
                            continue;
                        if (func.getName() == "main")
                            continue;
                        if (userInit && func == userInit)
                            continue;

                        auto selAttr = func->getAttrOfType<StringAttr>("ora.selector");
                        if (!selAttr)
                        {
                            func.emitError("missing ora.selector for public function");
                            signalPassFailure();
                            return;
                        }

                        StringRef selStr = selAttr.getValue();
                        if (!selStr.starts_with("0x") || selStr.size() != 10)
                        {
                            func.emitError("invalid selector format (expected 0x + 8 hex chars)");
                            signalPassFailure();
                            return;
                        }

                        uint32_t sel = 0;
                        for (char c : selStr.drop_front(2))
                        {
                            sel <<= 4;
                            if (c >= '0' && c <= '9')
                                sel |= (c - '0');
                            else if (c >= 'a' && c <= 'f')
                                sel |= (c - 'a' + 10);
                            else if (c >= 'A' && c <= 'F')
                                sel |= (c - 'A' + 10);
                            else
                            {
                                func.emitError("invalid selector hex");
                                signalPassFailure();
                                return;
                            }
                        }

                        PubFuncInfo info(func, findFunctionProvenanceLoc(func));
                        info.selector = sel;
                        info.argCount = func.getFunctionType().getNumInputs();
                        info.retCount = func.getFunctionType().getNumResults();
                        info.inputTypes.append(func.getFunctionType().getInputs().begin(),
                                               func.getFunctionType().getInputs().end());

                        if (auto abiAttr = func->getAttrOfType<ArrayAttr>("ora.abi_params"))
                        {
                            for (Attribute a : abiAttr)
                            {
                                auto sattr = dyn_cast<StringAttr>(a);
                                if (!sattr)
                                {
                                    func.emitError("ora.abi_params contains non-string attr");
                                    signalPassFailure();
                                    return;
                                }
                                AbiLayoutNode layout;
                                if (!parseAbiLayout(sattr.getValue(), layout, AbiLayoutSyntax::CanonicalAbi))
                                {
                                    func.emitError("unsupported ABI param type: " + sattr.getValue());
                                    signalPassFailure();
                                    return;
                                }
                                info.abiParamLayouts.push_back(sattr.getValue().str());
                            }
                        }
                        if (auto resultInputModesAttr = func->getAttrOfType<ArrayAttr>("ora.result_input_modes"))
                        {
                            for (Attribute a : resultInputModesAttr)
                            {
                                auto sattr = dyn_cast<StringAttr>(a);
                                if (!sattr)
                                {
                                    func.emitError("ora.result_input_modes contains non-string attr");
                                    signalPassFailure();
                                    return;
                                }
                                info.resultInputModes.push_back(sattr.getValue().str());
                            }
                        }
                        if (auto enumCountsAttr = func->getAttrOfType<ArrayAttr>("ora.abi_param_enum_counts"))
                        {
                            for (Attribute a : enumCountsAttr)
                            {
                                auto iattr = dyn_cast<IntegerAttr>(a);
                                if (!iattr)
                                {
                                    func.emitError("ora.abi_param_enum_counts contains non-integer attr");
                                    signalPassFailure();
                                    return;
                                }
                                info.abiParamEnumCounts.push_back(iattr.getValue().getZExtValue());
                            }
                        }
                        if (auto refinementsAttr = func->getAttrOfType<ArrayAttr>("ora.abi_param_refinements"))
                        {
                            for (Attribute a : refinementsAttr)
                            {
                                auto sattr = dyn_cast<StringAttr>(a);
                                if (!sattr)
                                {
                                    func.emitError("ora.abi_param_refinements contains non-string attr");
                                    signalPassFailure();
                                    return;
                                }
                                CalldataRefinementSpec parsed;
                                if (!parseCalldataRefinementSpec(sattr.getValue(), parsed))
                                {
                                    func.emitError("ora.abi_param_refinements contains malformed refinement metadata");
                                    signalPassFailure();
                                    return;
                                }
                                info.abiParamRefinements.push_back(sattr.getValue().str());
                            }
                        }
                        if (auto resultInputErrorIdsAttr = func->getAttrOfType<ArrayAttr>("ora.result_input_error_ids"))
                        {
                            for (Attribute a : resultInputErrorIdsAttr)
                            {
                                auto iattr = dyn_cast<IntegerAttr>(a);
                                if (!iattr)
                                {
                                    func.emitError("ora.result_input_error_ids contains non-integer attr");
                                    signalPassFailure();
                                    return;
                                }
                                info.resultInputErrorIds.push_back(iattr.getInt());
                            }
                        }
                        if (auto returnErrorIdsAttr = func->getAttrOfType<ArrayAttr>("ora.return_error_ids"))
                        {
                            info.hasReturnErrorMetadata = true;
                            for (Attribute a : returnErrorIdsAttr)
                            {
                                auto iattr = dyn_cast<IntegerAttr>(a);
                                if (!iattr)
                                {
                                    func.emitError("ora.return_error_ids contains non-integer attr");
                                    signalPassFailure();
                                    return;
                                }
                                uint64_t errorId = iattr.getValue().getZExtValue();
                                auto found = llvm::find_if(abiErrors, [&](const ErrorInfo &errInfo) {
                                    return errInfo.id == errorId;
                                });
                                if (found == abiErrors.end())
                                {
                                    func.emitError("ora.return_error_ids references unknown error id");
                                    signalPassFailure();
                                    return;
                                }
                                info.returnErrors.push_back(*found);
                            }
                        }

                        if (auto abiReturnAttr = func->getAttrOfType<StringAttr>("ora.abi_return"))
                        {
                            AbiLayoutNode abiReturn;
                            if (!parseAbiLayout(abiReturnAttr.getValue(), abiReturn, AbiLayoutSyntax::CanonicalAbi))
                            {
                                func.emitError("unsupported ABI return type: " + abiReturnAttr.getValue());
                                signalPassFailure();
                                return;
                            }
                            info.abiReturnLayout = abiReturnAttr.getValue().str();
                            info.hasAbiReturn = true;
                        }
                        if (auto abiReturnWordsAttr = func->getAttrOfType<IntegerAttr>("ora.abi_return_words"))
                            info.abiReturnWords = abiReturnWordsAttr.getInt();
                        if (auto abiReturnLayoutAttr = func->getAttrOfType<StringAttr>("ora.abi_return_layout"))
                        {
                            AbiLayoutNode abiReturnLayout;
                            if (!parseAbiLayout(abiReturnLayoutAttr.getValue(), abiReturnLayout, AbiLayoutSyntax::CanonicalAbi))
                            {
                                func.emitError("invalid ora.abi_return_layout");
                                signalPassFailure();
                                return;
                            }
                            info.abiReturnLayout = abiReturnLayoutAttr.getValue().str();
                        }

                        if (auto returnsErrorUnionAttr = func->getAttrOfType<BoolAttr>("ora.returns_error_union"))
                            info.returnsErrorUnion = returnsErrorUnionAttr.getValue();
                        if (info.returnsErrorUnion && !info.hasReturnErrorMetadata)
                        {
                            func.emitError("public error-union function is missing ora.return_error_ids metadata");
                            signalPassFailure();
                            return;
                        }
                        if (auto modeAttr = func->getAttrOfType<StringAttr>("ora.abi_decode_mode"))
                            info.permissiveAbiDecode = modeAttr.getValue() == "permissive";
                        if (!info.abiParamLayouts.empty() && info.resultInputModes.size() != info.abiParamLayouts.size())
                        {
                            func.emitError("ora.result_input_modes length does not match ora.abi_params");
                            signalPassFailure();
                            return;
                        }
                        if (!info.abiParamLayouts.empty() && info.abiParamEnumCounts.size() != info.abiParamLayouts.size())
                        {
                            func.emitError("ora.abi_param_enum_counts length does not match ora.abi_params");
                            signalPassFailure();
                            return;
                        }
                        if (!info.abiParamLayouts.empty() && info.abiParamRefinements.size() != info.abiParamLayouts.size())
                        {
                            func.emitError("ora.abi_param_refinements length does not match ora.abi_params");
                            signalPassFailure();
                            return;
                        }
                        if (!info.abiParamLayouts.empty() && info.resultInputErrorIds.size() != info.abiParamLayouts.size())
                        {
                            func.emitError("ora.result_input_error_ids length does not match ora.abi_params");
                            signalPassFailure();
                            return;
                        }

                        auto loweredArgCountForSourceParam = [&](size_t idx) -> unsigned {
                            if (idx < info.resultInputModes.size() &&
                                (info.resultInputModes[idx] == "wide_payloadless" || info.resultInputModes[idx] == "wide_single_error"))
                                return static_cast<unsigned>(adt_helpers::kAdtCarrierWordCount);
                            return 1;
                        };

                        unsigned expectedArgCount = info.abiParamLayouts.empty() ? info.argCount : 0;
                        if (!info.abiParamLayouts.empty())
                        {
                            for (size_t i = 0; i < info.abiParamLayouts.size(); ++i)
                                expectedArgCount += loweredArgCountForSourceParam(i);
                        }

                        if (!info.abiParamLayouts.empty() && expectedArgCount != info.argCount)
                        {
                            func.emitError("public ABI param metadata does not match lowered function argument count");
                            signalPassFailure();
                            return;
                        }

                        int64_t headSlots = 0;
                        size_t sourceParamCount = info.abiParamLayouts.empty() ? info.argCount : info.abiParamLayouts.size();
                        for (size_t i = 0; i < sourceParamCount; ++i)
                        {
                            if (info.abiParamLayouts.empty())
                            {
                                headSlots += 1;
                            }
                            else
                            {
                                AbiLayoutNode layout;
                                if (!parseAbiLayout(info.abiParamLayouts[i], layout, AbiLayoutSyntax::CanonicalAbi))
                                {
                                    func.emitError("invalid ABI param layout");
                                    signalPassFailure();
                                    return;
                                }
                                int64_t slots = dispatcherHeadSlotsForLayout(layout);
                                if (slots < 0)
                                {
                                    func.emitError("unsupported ABI type for head sizing");
                                    signalPassFailure();
                                    return;
                                }
                                headSlots += slots;
                            }
                        }
                        info.minHeadBytes = 4 + 32 * headSlots;
                        pubFuncs.push_back(info);
                    }

                    // Frequency-ordered dispatch: state-mutating functions get
                    // the cheap chain positions; provably read-only functions
                    // (ora.dispatch_class = "readonly", normally reached via
                    // gas-free eth_call) sort last. Stable sort preserves
                    // declaration order within each class, and a missing
                    // attribute never demotes. Order is semantics-neutral —
                    // it only shifts linear-chain gas.
                    std::stable_sort(pubFuncs.begin(), pubFuncs.end(),
                                     [](const PubFuncInfo &a, const PubFuncInfo &b) {
                                         auto classOf = [](const PubFuncInfo &info) {
                                             auto attr = info.func->getAttrOfType<StringAttr>("ora.dispatch_class");
                                             return (attr && attr.getValue() == "readonly") ? 1 : 0;
                                         };
                                         return classOf(a) < classOf(b);
                                     });

                    // Synthesize boilerplate init/main; user-defined versions are replaced.
                    if (auto sym = SymbolTable::lookupSymbolIn(module, StringRef("init")))
                        sym->erase();
                    if (auto sym = SymbolTable::lookupSymbolIn(module, StringRef("main")))
                        sym->erase();

                    // Align sir.icall result types with updated callee signatures.
                    module.walk([&](sir::ICallOp icall) {
                        auto calleeAttr = icall.getCalleeAttr();
                        if (!calleeAttr)
                            return;
                        Operation *sym = SymbolTable::lookupSymbolIn(module, calleeAttr);
                        if (!sym)
                        {
                            sym = SymbolTable::lookupNearestSymbolFrom(icall, calleeAttr);
                        }
                        auto calleeFunc = dyn_cast_or_null<func::FuncOp>(sym);
                        if (!calleeFunc)
                            return;

                        auto calleeType = calleeFunc.getFunctionType();
                        unsigned matchedResults = std::min(icall.getNumResults(), calleeType.getNumResults());
                        for (unsigned i = 0; i < matchedResults; ++i)
                        {
                            Type oldType = icall.getResult(i).getType();
                            Type calleeResultType = calleeType.getResult(i);
                            if (isPtrWordResultRepair(oldType, calleeResultType))
                            {
                                icall.emitError("dispatcher icall ptr/u256 result mismatch requires explicit lowering");
                                signalPassFailure();
                                return;
                            }
                        }

                        // sir.icall is word-based: args/results must be sir.u256.
                        // Avoid retyping calls to raw callee signatures that use
                        // non-word types (e.g. i256/ptr), which creates invalid IR.
                        bool calleeAllU256 = llvm::all_of(
                            calleeType.getResults(),
                            [](Type t) { return isa<sir::U256Type>(t); });
                        if (!calleeAllU256)
                            return;

                        if (calleeType.getNumResults() == icall.getNumResults())
                            return;

                        SmallVector<Type, 4> newResults;
                        newResults.append(calleeType.getResults().begin(), calleeType.getResults().end());
                        OpBuilder callBuilder(icall);
                        auto newCall = callBuilder.create<sir::ICallOp>(
                            icall.getLoc(),
                            newResults,
                            calleeAttr,
                            icall.getArgs());

                        unsigned common = std::min(icall.getNumResults(), newCall.getNumResults());
                        for (unsigned i = 0; i < common; ++i)
                        {
                            Value oldRes = icall.getResult(i);
                            Value newRes = newCall.getResult(i);
                            if (oldRes.getType() == newRes.getType())
                            {
                                oldRes.replaceAllUsesWith(newRes);
                                continue;
                            }

                            if (isPtrWordResultRepair(oldRes.getType(), newRes.getType()))
                            {
                                icall.emitError("dispatcher icall ptr/u256 result mismatch requires explicit lowering");
                                signalPassFailure();
                                return;
                            }

                            oldRes.replaceAllUsesWith(newRes);
                        }

                        if (icall.getNumResults() > newCall.getNumResults())
                        {
                            icall.emitError("dispatcher icall result count exceeds rewritten callee result count");
                            signalPassFailure();
                            return;
                        }

                        icall.erase();
                    });

                    // Build init(): run user init (if any), then copy runtime into memory and return it.
                    auto initType = builder.getFunctionType({}, {});
                    Location initLoc = userInit
                                           ? makeSyntheticOriginOnlyLoc(userInit.getLoc(), "constructor_decode")
                                           : makeSyntheticOriginOnlyLoc(loc, "constructor_decode");
                    auto initFunc = func::FuncOp::create(initLoc, "init", initType);
                    initFunc.setPrivate();
                    Block *initEntry = initFunc.addEntryBlock();
                    Block *initRevert = nullptr;
                    Block *initDecode = nullptr;
                    builder.setInsertionPointToEnd(initEntry);
                    DenseMap<Block *, DenseMap<int64_t, Value>> constCache;
                    DenseMap<uint64_t, Block *> initAbiDecodeRevertBlocks;

                    auto getInitRevert = [&]() -> Block * {
                        if (!initRevert)
                            initRevert = initFunc.addBlock();
                        return initRevert;
                    };

                    auto getInitAbiDecodeRevertBlock = [&](lowering::AbiDecodeError error) -> Block * {
                        uint64_t ordinal = static_cast<uint64_t>(error);
                        auto it = initAbiDecodeRevertBlocks.find(ordinal);
                        if (it != initAbiDecodeRevertBlocks.end())
                            return it->second;

                        OpBuilder::InsertionGuard guard(builder);
                        Block *block = initFunc.addBlock();
                        initAbiDecodeRevertBlocks.try_emplace(ordinal, block);
                        builder.setInsertionPointToEnd(block);
                        Value size = lowering::constU256(builder, initLoc, 32);
                        Value payload = lowering::constU256(builder, initLoc, ordinal);
                        Value ptr = builder.create<sir::SAllocAnyOp>(initLoc, ptrType, size);
                        builder.create<sir::StoreOp>(initLoc, ptr, payload);
                        builder.create<sir::RevertOp>(initLoc, ptr, size);
                        std::string blockName = "init_abi_decode_revert_" + std::to_string(ordinal);
                        setBlockName(block, blockName);
                        return block;
                    };

                    if (userInit)
                    {
                        auto userInitType = userInit.getFunctionType();
                        // Strip void/none return types from init (the frontend may emit them).
                        if (userInitType.getNumResults() != 0)
                        {
                            bool allNone = true;
                            for (auto rt : userInitType.getResults())
                            {
                                if (!llvm::isa<mlir::NoneType>(rt))
                                {
                                    allNone = false;
                                    break;
                                }
                            }
                            if (allNone)
                            {
                                auto newType = mlir::FunctionType::get(
                                    userInit.getContext(), userInitType.getInputs(), {});
                                userInit.setFunctionType(newType);
                                // Also strip result attributes to match the new 0-result type.
                                userInit.setAllResultAttrs(ArrayRef<DictionaryAttr>{});
                            }
                            else
                            {
                                userInit.emitError("constructor init must not return values");
                                signalPassFailure();
                                return;
                            }
                        }

                        SmallVector<std::string, 8> initAbiParamLayouts;
                        if (auto abiAttr = userInit->getAttrOfType<ArrayAttr>("ora.abi_params"))
                        {
                            for (Attribute a : abiAttr)
                            {
                                auto sattr = dyn_cast<StringAttr>(a);
                                if (!sattr)
                                {
                                    userInit.emitError("ora.abi_params contains non-string attr");
                                    signalPassFailure();
                                    return;
                                }
                                AbiLayoutNode layout;
                                if (!parseAbiLayout(sattr.getValue(), layout, AbiLayoutSyntax::CanonicalAbi))
                                {
                                    userInit.emitError("unsupported ABI param type: " + sattr.getValue());
                                    signalPassFailure();
                                    return;
                                }
                                initAbiParamLayouts.push_back(sattr.getValue().str());
                            }
                        }

                        unsigned argCount = userInitType.getNumInputs();
                        if (!initAbiParamLayouts.empty() && initAbiParamLayouts.size() != argCount)
                        {
                            userInit.emitError("ora.abi_params length does not match function argument count");
                            signalPassFailure();
                            return;
                        }
                        bool hasDynamicConstructorParam = false;
                        for (StringRef abiLayoutText : initAbiParamLayouts)
                        {
                            AbiLayoutNode layout;
                            if (!parseAbiLayout(abiLayoutText, layout, AbiLayoutSyntax::CanonicalAbi))
                            {
                                userInit.emitError("unsupported ABI param type: " + abiLayoutText);
                                signalPassFailure();
                                return;
                            }
                            if (canonicalAbiLayoutIsDynamic(layout))
                            {
                                hasDynamicConstructorParam = true;
                                break;
                            }
                        }
                        int64_t headSlots = 0;
                        for (unsigned i = 0; i < argCount; ++i)
                        {
                            if (initAbiParamLayouts.empty())
                            {
                                headSlots += 1;
                            }
                            else
                            {
                                AbiLayoutNode layout;
                                if (!parseAbiLayout(initAbiParamLayouts[i], layout, AbiLayoutSyntax::CanonicalAbi))
                                {
                                    userInit.emitError("invalid constructor ABI param layout");
                                    signalPassFailure();
                                    return;
                                }
                                int64_t slots = dispatcherHeadSlotsForLayout(layout);
                                if (slots < 0)
                                {
                                    userInit.emitError("unsupported ABI type for head sizing");
                                    signalPassFailure();
                                    return;
                                }
                                headSlots += slots;
                            }
                        }
                        Value codeSize = builder.create<sir::CodeSizeOp>(initLoc, u256Type);
                        Value initEnd = builder.create<sir::InitEndOffsetOp>(initLoc, u256Type);
                        Value codeTooShort = builder.create<sir::LtOp>(initLoc, u256Type, codeSize, initEnd);
                        Value dataLen = builder.create<sir::SubOp>(initLoc, u256Type, codeSize, initEnd);

                        int64_t minHeadBytes = 32 * headSlots;
                        if (minHeadBytes > 0)
                        {
                            Value valid_code = builder.create<sir::IsZeroOp>(initLoc, u256Type, codeTooShort);
                            Block *codeOkBlock = initFunc.addBlock();
                            builder.create<sir::CondBrOp>(initLoc, valid_code, ValueRange{}, ValueRange{}, codeOkBlock, getInitRevert());
                            builder.setInsertionPointToEnd(codeOkBlock);

                            Value minSizeVal = lowering::constU256(builder, initLoc, minHeadBytes);
                            Value dataTooShort = builder.create<sir::LtOp>(initLoc, u256Type, dataLen, minSizeVal);
                            Value valid_args = builder.create<sir::IsZeroOp>(initLoc, u256Type, dataTooShort);
                            initDecode = initFunc.addBlock();
                            builder.create<sir::CondBrOp>(initLoc, valid_args, ValueRange{}, ValueRange{}, initDecode, getInitAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));
                            builder.setInsertionPointToEnd(initDecode);
                        }
                        else
                        {
                            Value valid_args = builder.create<sir::IsZeroOp>(initLoc, u256Type, codeTooShort);
                            initDecode = initFunc.addBlock();
                            builder.create<sir::CondBrOp>(initLoc, valid_args, ValueRange{}, ValueRange{}, initDecode, getInitRevert());
                            builder.setInsertionPointToEnd(initDecode);
                        }

                        if (hasDynamicConstructorParam)
                        {
                            // Dynamic constructor decoding copies appended ABI args into
                            // heap memory before validating dynamic tails. Keep that buffer
                            // above the SIR text legalizer's fixed scratch area,
                            // otherwise branch operand spills can corrupt later tails.
                            Value initFreePtrSlot = builder.create<sir::BitcastOp>(initLoc, ptrType, lowering::constU256(builder, initLoc, 32));
                            Value initScratchFence = lowering::constU256(builder, initLoc, lowering::kConstructorDecodeScratchFenceBytes);
                            Value initCurrentFreePtr = builder.create<sir::LoadOp>(initLoc, u256Type, initFreePtrSlot);
                            Value initShouldRaiseFreePtr = builder.create<sir::LtOp>(initLoc, u256Type, initCurrentFreePtr, initScratchFence);
                            Value initFreePtr = builder.create<sir::SelectOp>(initLoc, u256Type, initShouldRaiseFreePtr, initScratchFence, initCurrentFreePtr);
                            builder.create<sir::StoreOp>(initLoc, initFreePtrSlot, initFreePtr);
                        }

                        Value dataBuf = builder.create<sir::MallocOp>(initLoc, ptrType, dataLen);
                        builder.create<sir::CodeCopyOp>(initLoc, dataBuf, initEnd, dataLen);

                        SmallVector<int64_t, 8> headOffsets;
                        int64_t headSlot = 0;
                        for (unsigned i = 0; i < argCount; ++i)
                        {
                            headOffsets.push_back(32 * headSlot);
                            int64_t slots = 1;
                            if (!initAbiParamLayouts.empty())
                            {
                                AbiLayoutNode layout;
                                if (!parseAbiLayout(initAbiParamLayouts[i], layout, AbiLayoutSyntax::CanonicalAbi))
                                {
                                    module.emitError("invalid constructor ABI param layout");
                                    signalPassFailure();
                                    return;
                                }
                                slots = dispatcherHeadSlotsForLayout(layout);
                            }
                            if (slots < 0)
                            {
                                module.emitError("unsupported ABI type for head offset sizing");
                                signalPassFailure();
                                return;
                            }
                            headSlot += slots;
                        }

                        SmallVector<Value, 8> args;
                        auto ptrType = sir::PtrType::get(ctx, 1);
                        Value nextConstructorDynamicOffset;
                        auto getNextConstructorDynamicOffset = [&]() -> Value {
                            if (!nextConstructorDynamicOffset)
                                nextConstructorDynamicOffset = lowering::constU256(builder, initLoc, static_cast<uint64_t>(headSlot) * 32ULL);
                            return nextConstructorDynamicOffset;
                        };
                        auto emitStrictConstructorBoundsPrefix = [&](Value offsetWord,
                                                                     Value expectedOffset,
                                                                     StrictDynamicCalldataKind kind,
                                                                     uint64_t arrayElementWords = 1) -> StrictDynamicBounds {
                            return emitStrictDynamicBoundsPrefix(
                                builder,
                                initLoc,
                                u256Type,
                                offsetWord,
                                expectedOffset,
                                kind,
                                [&]() -> Block * { return initFunc.addBlock(); },
                                [&](lowering::AbiDecodeError error) -> Block * { return getInitAbiDecodeRevertBlock(error); },
                                [&]() -> Value { return dataLen; },
                                [&](Value) -> StrictDynamicBoundsBase {
                                    Value dynamicPtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, dataBuf, offsetWord);
                                    return StrictDynamicBoundsBase{dynamicPtr, offsetWord};
                                },
                                [&](Value dynamicPtr) -> Value {
                                    return builder.create<sir::LoadOp>(initLoc, u256Type, dynamicPtr);
                                },
                                false,
                                arrayElementWords);
                        };
                        auto materializeStrictConstructorDynamic = [&](Value offsetWord,
                                                                       Value expectedOffset,
                                                                       StrictDynamicCalldataKind kind,
                                                                       unsigned fixedBytesWidth = 0,
                                                                       uint64_t arrayElementWords = 1) -> FailureOr<StrictDynamicCalldataValue> {
                            StrictDynamicBounds bounds = emitStrictConstructorBoundsPrefix(
                                offsetWord,
                                expectedOffset,
                                kind,
                                arrayElementWords);
                            Block *doneBlock = initFunc.addBlock();

                            Value tailEnd = builder.create<sir::AddOp>(initLoc, u256Type, offsetWord, bounds.total);
                            Value tailMissing = builder.create<sir::LtOp>(initLoc, u256Type, dataLen, tailEnd);
                            Value tailPresent = builder.create<sir::IsZeroOp>(initLoc, u256Type, tailMissing);
                            const bool validatesWordElements = strictDynamicCalldataValidatesWordElements(kind);

                            Block *wordArrayValidateCondBlock = nullptr;
                            Block *wordArrayValidateBodyBlock = nullptr;
                            Block *wordArrayValidateDoneBlock = nullptr;
                            if (validatesWordElements)
                            {
                                wordArrayValidateCondBlock = initFunc.addBlock();
                                wordArrayValidateCondBlock->addArgument(u256Type, initLoc);
                                wordArrayValidateCondBlock->addArgument(u256Type, initLoc);
                                wordArrayValidateBodyBlock = initFunc.addBlock();
                                wordArrayValidateBodyBlock->addArgument(u256Type, initLoc);
                                wordArrayValidateBodyBlock->addArgument(u256Type, initLoc);
                                wordArrayValidateDoneBlock = initFunc.addBlock();
                                wordArrayValidateDoneBlock->addArgument(u256Type, initLoc);
                            }
                            Block *bytesPadCondBlock = nullptr;
                            if (kind == StrictDynamicCalldataKind::BytesLike)
                            {
                                bytesPadCondBlock = doneBlock;
                                bytesPadCondBlock->addArgument(u256Type, initLoc);
                                bytesPadCondBlock->addArgument(u256Type, initLoc);
                            }
                            builder.create<sir::CondBrOp>(
                                initLoc,
                                tailPresent,
                                validatesWordElements
                                    ? ValueRange{
                                          lowering::constU256(builder, initLoc, 0),
                                          lowering::constU256(builder, initLoc, 1),
                                      }
                                : kind == StrictDynamicCalldataKind::BytesLike
                                    ? ValueRange{
                                          lowering::constU256(builder, initLoc, 0),
                                          lowering::constU256(builder, initLoc, 1),
                                      }
                                    : ValueRange{},
                                ValueRange{},
                                validatesWordElements
                                    ? wordArrayValidateCondBlock
                                : kind == StrictDynamicCalldataKind::BytesLike
                                    ? bytesPadCondBlock
                                    : doneBlock,
                                getInitAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                            if (validatesWordElements)
                            {
                                builder.setInsertionPointToEnd(wordArrayValidateCondBlock);
                                Value iv = wordArrayValidateCondBlock->getArgument(0);
                                Value allValid = wordArrayValidateCondBlock->getArgument(1);
                                Value hasElement = builder.create<sir::LtOp>(initLoc, u256Type, iv, bounds.dynamicLen);
                                Value continueValidation = builder.create<sir::AndOp>(initLoc, u256Type, hasElement, allValid);
                                builder.create<sir::CondBrOp>(
                                    initLoc,
                                    continueValidation,
                                    ValueRange{iv, allValid},
                                    ValueRange{allValid},
                                    wordArrayValidateBodyBlock,
                                    wordArrayValidateDoneBlock);

                                builder.setInsertionPointToEnd(wordArrayValidateBodyBlock);
                                Value bodyIv = wordArrayValidateBodyBlock->getArgument(0);
                                Value bodyAllValid = wordArrayValidateBodyBlock->getArgument(1);
                                Value elementByteOffset = builder.create<sir::MulOp>(initLoc, u256Type, bodyIv, bounds.wordSize);
                                Value elementTailOffset = builder.create<sir::AddOp>(initLoc, u256Type, elementByteOffset, bounds.wordSize);
                                Value elementPtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, bounds.payloadBase, elementTailOffset);
                                Value elementWord = builder.create<sir::LoadOp>(initLoc, u256Type, elementPtr);
                                Value elementValid = nullptr;
                                if (kind == StrictDynamicCalldataKind::AddressArray)
                                {
                                    Value elementPayload = lowering::maskLowBits(builder, initLoc, elementWord, 160);
                                    elementValid = builder.create<sir::EqOp>(initLoc, u256Type, elementWord, elementPayload);
                                }
                                else if (kind == StrictDynamicCalldataKind::BoolArray)
                                {
                                    elementValid = lowering::boolAbiWordIsCanonical(builder, initLoc, elementWord);
                                }
                                else
                                {
                                    lowering::FixedBytesWordDecode decoded = lowering::decodeFixedBytesAbiWord(builder, initLoc, fixedBytesWidth, elementWord);
                                    elementValid = decoded.valid;
                                }
                                Value nextAllValid = builder.create<sir::AndOp>(initLoc, u256Type, bodyAllValid, elementValid);
                                Value nextIv = builder.create<sir::AddOp>(initLoc, u256Type, bodyIv, lowering::constU256(builder, initLoc, 1));
                                builder.create<sir::BrOp>(initLoc, ValueRange{nextIv, nextAllValid}, wordArrayValidateCondBlock);

                                builder.setInsertionPointToEnd(wordArrayValidateDoneBlock);
                                Value validElements = wordArrayValidateDoneBlock->getArgument(0);
                                FailureOr<lowering::AbiDecodeError> invalidElementError = strictDynamicCalldataInvalidElementError(kind);
                                if (failed(invalidElementError))
                                    return failure();
                                builder.create<sir::CondBrOp>(
                                    initLoc,
                                    validElements,
                                    ValueRange{},
                                    ValueRange{},
                                    doneBlock,
                                    getInitAbiDecodeRevertBlock(*invalidElementError));
                            }

                            if (kind == StrictDynamicCalldataKind::FixedBytesArray)
                            {
                                builder.setInsertionPointToEnd(doneBlock);
                                Value resultPtr = builder.create<sir::SAllocAnyOp>(initLoc, ptrType, bounds.total);
                                builder.create<sir::StoreOp>(initLoc, resultPtr, bounds.dynamicLen);
                                Value resultContentPtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, resultPtr, bounds.wordSize);

                                Block *copyCondBlock = initFunc.addBlock();
                                copyCondBlock->addArgument(u256Type, initLoc);
                                Block *copyBodyBlock = initFunc.addBlock();
                                copyBodyBlock->addArgument(u256Type, initLoc);
                                Block *copyDoneBlock = initFunc.addBlock();

                                builder.create<sir::BrOp>(
                                    initLoc,
                                    ValueRange{lowering::constU256(builder, initLoc, 0)},
                                    copyCondBlock);

                                builder.setInsertionPointToEnd(copyCondBlock);
                                Value copyIv = copyCondBlock->getArgument(0);
                                Value hasElement = builder.create<sir::LtOp>(initLoc, u256Type, copyIv, bounds.dynamicLen);
                                builder.create<sir::CondBrOp>(initLoc, hasElement, ValueRange{copyIv}, ValueRange{}, copyBodyBlock, copyDoneBlock);

                                builder.setInsertionPointToEnd(copyBodyBlock);
                                Value bodyIv = copyBodyBlock->getArgument(0);
                                Value elementByteOffset = builder.create<sir::MulOp>(initLoc, u256Type, bodyIv, bounds.wordSize);
                                Value elementTailOffset = builder.create<sir::AddOp>(initLoc, u256Type, elementByteOffset, bounds.wordSize);
                                Value elementPtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, bounds.payloadBase, elementTailOffset);
                                Value elementWord = builder.create<sir::LoadOp>(initLoc, u256Type, elementPtr);
                                lowering::FixedBytesWordDecode decoded = lowering::decodeFixedBytesAbiWord(builder, initLoc, fixedBytesWidth, elementWord);
                                Value resultElementPtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, resultContentPtr, elementByteOffset);
                                builder.create<sir::StoreOp>(initLoc, resultElementPtr, decoded.payload);
                                Value nextIv = builder.create<sir::AddOp>(initLoc, u256Type, bodyIv, lowering::constU256(builder, initLoc, 1));
                                builder.create<sir::BrOp>(initLoc, ValueRange{nextIv}, copyCondBlock);

                                builder.setInsertionPointToEnd(copyDoneBlock);
                                return StrictDynamicCalldataValue{
                                    builder.create<sir::BitcastOp>(initLoc, u256Type, resultPtr).getResult(),
                                    bounds.nextExpectedOffset,
                                };
                            }

                            if (kind != StrictDynamicCalldataKind::BytesLike)
                            {
                                builder.setInsertionPointToEnd(doneBlock);
                                return StrictDynamicCalldataValue{
                                    builder.create<sir::BitcastOp>(initLoc, u256Type, bounds.payloadBase).getResult(),
                                    bounds.nextExpectedOffset,
                                };
                            }

                            Block *padCondBlock = bytesPadCondBlock;
                            Block *padBodyBlock = initFunc.addBlock();
                            padBodyBlock->addArgument(u256Type, initLoc);
                            padBodyBlock->addArgument(u256Type, initLoc);
                            Block *padDoneBlock = initFunc.addBlock();
                            padDoneBlock->addArgument(u256Type, initLoc);
                            Block *bytesDoneBlock = initFunc.addBlock();

                            builder.setInsertionPointToEnd(padCondBlock);
                            Value padIv = padCondBlock->getArgument(0);
                            Value padAllValid = padCondBlock->getArgument(1);
                            Value contentPtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, bounds.payloadBase, bounds.wordSize);
                            Value padStart = builder.create<sir::AddPtrOp>(initLoc, ptrType, contentPtr, bounds.dynamicLen);
                            Value padCount = builder.create<sir::SubOp>(initLoc, u256Type, bounds.padded, bounds.dynamicLen);
                            Value hasPadByte = builder.create<sir::LtOp>(initLoc, u256Type, padIv, padCount);
                            Value continuePad = builder.create<sir::AndOp>(initLoc, u256Type, hasPadByte, padAllValid);
                            builder.create<sir::CondBrOp>(
                                initLoc,
                                continuePad,
                                ValueRange{padIv, padAllValid},
                                ValueRange{padAllValid},
                                padBodyBlock,
                                padDoneBlock);

                            builder.setInsertionPointToEnd(padBodyBlock);
                            Value bodyIv = padBodyBlock->getArgument(0);
                            Value bodyAllValid = padBodyBlock->getArgument(1);
                            Value padBytePtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, padStart, bodyIv);
                            Value padByte = builder.create<sir::Load8Op>(initLoc, u256Type, padBytePtr, lowering::constU256(builder, initLoc, 0));
                            Value byteIsZero = builder.create<sir::EqOp>(initLoc, u256Type, padByte, lowering::constU256(builder, initLoc, 0));
                            Value nextAllValid = builder.create<sir::AndOp>(initLoc, u256Type, bodyAllValid, byteIsZero);
                            Value nextIv = builder.create<sir::AddOp>(initLoc, u256Type, bodyIv, lowering::constU256(builder, initLoc, 1));
                            builder.create<sir::BrOp>(initLoc, ValueRange{nextIv, nextAllValid}, padCondBlock);

                            builder.setInsertionPointToEnd(padDoneBlock);
                            Value paddingValid = padDoneBlock->getArgument(0);
                            builder.create<sir::CondBrOp>(
                                initLoc,
                                paddingValid,
                                ValueRange{},
                                ValueRange{},
                                bytesDoneBlock,
                                getInitAbiDecodeRevertBlock(lowering::AbiDecodeError::NonCanonicalEncoding));

                            builder.setInsertionPointToEnd(bytesDoneBlock);
                            return StrictDynamicCalldataValue{
                                builder.create<sir::BitcastOp>(initLoc, u256Type, bounds.payloadBase).getResult(),
                                bounds.nextExpectedOffset,
                            };
                        };
                        for (unsigned idx = 0; idx < argCount; ++idx)
                        {
                            int64_t offs = headOffsets[idx];
                            Value offc = lowering::constU256(builder, initLoc, offs);
                            Value headPtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, dataBuf, offc);
                            Value head = builder.create<sir::LoadOp>(initLoc, u256Type, headPtr);
                            AbiLayoutNode abiLayout;
                            const bool hasAbiLayout = !initAbiParamLayouts.empty();
                            if (hasAbiLayout && !parseAbiLayout(initAbiParamLayouts[idx], abiLayout, AbiLayoutSyntax::CanonicalAbi))
                            {
                                module.emitError("invalid constructor ABI param layout");
                                signalPassFailure();
                                return;
                            }
                            Value argVal = head;

                            if (hasAbiLayout && isStaticFixedArrayLayout(abiLayout))
                            {
                                int64_t totalBytes = canonicalAbiLayoutHeadSlots(abiLayout) * 32;
                                Value totalVal = lowering::constU256(builder, initLoc, totalBytes);
                                Value buf = builder.create<sir::SAllocAnyOp>(initLoc, ptrType, totalVal);
                                Value src = builder.create<sir::AddPtrOp>(initLoc, ptrType, dataBuf, offc);
                                builder.create<sir::MCopyOp>(initLoc, buf, src, totalVal);
                                argVal = buf;
                            }
                            else if (hasAbiLayout && canonicalAbiLayoutIsDynamic(abiLayout))
                            {
                                if (abiLayout.kind == AbiLayoutKind::DynamicBytes)
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg =
                                        materializeStrictConstructorDynamic(head, getNextConstructorDynamicOffset(), StrictDynamicCalldataKind::BytesLike);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for constructor");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextConstructorDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (isDynamicU256ArrayAbiNode(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg =
                                        materializeStrictConstructorDynamic(head, getNextConstructorDynamicOffset(), StrictDynamicCalldataKind::U256Array);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for constructor");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextConstructorDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (isDynamicAddressArrayAbiNode(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg =
                                        materializeStrictConstructorDynamic(head, getNextConstructorDynamicOffset(), StrictDynamicCalldataKind::AddressArray);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for constructor");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextConstructorDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (isDynamicBoolArrayAbiNode(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg =
                                        materializeStrictConstructorDynamic(head, getNextConstructorDynamicOffset(), StrictDynamicCalldataKind::BoolArray);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for constructor");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextConstructorDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (isDynamicFixedBytesArrayAbiNode(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg =
                                        materializeStrictConstructorDynamic(head, getNextConstructorDynamicOffset(), StrictDynamicCalldataKind::FixedBytesArray, abiLayout.children.front()->width);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for constructor");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextConstructorDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (std::optional<uint64_t> elementWords = fullWordStaticArrayElementWords(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg =
                                        materializeStrictConstructorDynamic(head, getNextConstructorDynamicOffset(), StrictDynamicCalldataKind::U256Array, 0, *elementWords);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for constructor");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextConstructorDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else
                                {
                                    module.emitError("unsupported dynamic ABI type for constructor");
                                    signalPassFailure();
                                    return;
                                }
                            }

                            if (idx < userInitType.getInputs().size())
                            {
                                if (auto ptrTy = dyn_cast<sir::PtrType>(userInitType.getInputs()[idx]))
                                {
                                    if (!isa<sir::PtrType>(argVal.getType()))
                                        argVal = builder.create<sir::BitcastOp>(initLoc, ptrType, argVal);
                                    argVal = builder.create<sir::BitcastOp>(initLoc, u256Type, argVal);
                                }
                            }
                            // sir.icall requires all args to be !sir.u256.
                            if (isa<sir::PtrType>(argVal.getType()))
                                argVal = builder.create<sir::BitcastOp>(initLoc, u256Type, argVal);

                            args.push_back(argVal);
                        }

                        builder.create<sir::ICallOp>(initLoc,
                                                     TypeRange{},
                                                     SymbolRefAttr::get(ctx, userInit.getName()),
                                                     args);
                    }

                    Value runtimeStart = builder.create<sir::RuntimeStartOffsetOp>(initLoc, u256Type);
                    Value runtimeLen = builder.create<sir::RuntimeLengthOp>(initLoc, u256Type);
                    Value initBuf = builder.create<sir::MallocOp>(initLoc, ptrType, runtimeLen);
                    builder.create<sir::CodeCopyOp>(initLoc, initBuf, runtimeStart, runtimeLen);
                    builder.create<sir::ReturnOp>(initLoc, initBuf, runtimeLen);

                    // Revert block for constructor argument checks.
                    if (initRevert)
                    {
                        builder.setInsertionPointToEnd(initRevert);
                        Value c0_revert = getConst(builder, initLoc, u256Type, i64Type, 0, constCache, initRevert, "zero");
                        Value p0b = builder.create<sir::BitcastOp>(initLoc, ptrType, c0_revert);
                        builder.create<sir::RevertOp>(initLoc, p0b, c0_revert);
                    }
                    module.push_back(initFunc);

                    // Build dispatcher: func main() -> ()
                    auto mainType = builder.getFunctionType({}, {});
                    Location dispatcherMainLoc = makeSyntheticOriginOnlyLoc(module.getLoc(), "dispatcher_main");
                    auto mainFunc = func::FuncOp::create(dispatcherMainLoc, "main", mainType);
                    Block *entry = mainFunc.addEntryBlock();
                    Block *loadSelector = mainFunc.addBlock();
                    Block *revertError = mainFunc.addBlock();

                    // entry: callvalue check (non-payable by default) + const prelude
                    builder.setInsertionPointToEnd(entry);
                    static_cast<void>(getConst(builder, dispatcherMainLoc, u256Type, i64Type, 0, constCache, entry, "zero"));
                    static_cast<void>(getConst(builder, dispatcherMainLoc, u256Type, i64Type, 4, constCache, entry, "selector_offset"));
                    Value c32_entry = getConst(builder, dispatcherMainLoc, u256Type, i64Type, 32, constCache, entry, "word_size");
                    static_cast<void>(getConst(builder, dispatcherMainLoc, u256Type, i64Type, 35, constCache, entry, "min_cdsize_1arg"));
                    static_cast<void>(getConst(builder, dispatcherMainLoc, u256Type, i64Type, 36, constCache, entry, "arg1_offset"));
                    static_cast<void>(getConst(builder, dispatcherMainLoc, u256Type, i64Type, 67, constCache, entry, "min_cdsize_2args"));
                    static_cast<void>(getConst(builder, dispatcherMainLoc, u256Type, i64Type, 68, constCache, entry, "arg2_offset"));
                    static_cast<void>(getConst(builder, dispatcherMainLoc, u256Type, i64Type, 99, constCache, entry, "min_cdsize_3args"));
                    static_cast<void>(getConst(builder, dispatcherMainLoc, u256Type, i64Type, 224, constCache, entry, "selector_shift"));

                    // Plank initializes memory[0x20] to its static memory high-water mark.
                    // before entering main. Raise it past compiler-owned named-memory slots
                    // placed after CODESIZE so user allocations cannot collide with them.
                    Value freePtrSlot = builder.create<sir::BitcastOp>(dispatcherMainLoc, ptrType, c32_entry);
                    Value runtimeHeapBase = builder.create<sir::CodeSizeOp>(dispatcherMainLoc, u256Type);
                    if (uint64_t namedMemoryBytes = computeNamedMemoryReserveBytes(module))
                    {
                        Value reservedBytes = lowering::constU256(builder, dispatcherMainLoc, namedMemoryBytes);
                        runtimeHeapBase = builder.create<sir::AddOp>(dispatcherMainLoc, u256Type, runtimeHeapBase, reservedBytes);
                    }
                    Value currentFreePtr = builder.create<sir::LoadOp>(dispatcherMainLoc, u256Type, freePtrSlot);
                    Value shouldRaiseFreePtr = builder.create<sir::LtOp>(dispatcherMainLoc, u256Type, currentFreePtr, runtimeHeapBase);
                    Value initialFreePtr = builder.create<sir::SelectOp>(dispatcherMainLoc, u256Type, shouldRaiseFreePtr, runtimeHeapBase, currentFreePtr);
                    builder.create<sir::StoreOp>(dispatcherMainLoc, freePtrSlot, initialFreePtr);

                    Value cv = builder.create<sir::CallValueOp>(dispatcherMainLoc, u256Type);
                    setResultName(cv.getDefiningOp(), "cv");
                    Value cv_zero = builder.create<sir::IsZeroOp>(dispatcherMainLoc, u256Type, cv);
                    setResultName(cv_zero.getDefiningOp(), "cv_nonzero");
                    // CALLDATALOAD(0) zero-pads, so calldata shorter than 4 bytes can
                    // only alias a selector whose low byte(s) are zero — and only a
                    // zero-argument function lacks the per-case min-calldatasize guard.
                    // Guard the whole dispatcher iff such a function exists; every
                    // other contract keeps its current byte-stable entry sequence.
                    bool needsShortCalldataGuard = false;
                    for (auto &info : pubFuncs)
                    {
                        int64_t caseMinSize = info.minHeadBytes > 0 ? info.minHeadBytes : (4 + 32 * static_cast<int64_t>(info.argCount));
                        if (caseMinSize <= 4 && (info.selector & 0xFFu) == 0)
                        {
                            needsShortCalldataGuard = true;
                            break;
                        }
                    }
                    Value dispatchOk = cv_zero;
                    if (needsShortCalldataGuard)
                    {
                        Value cds_entry = builder.create<sir::CallDataSizeOp>(dispatcherMainLoc, u256Type);
                        setResultName(cds_entry.getDefiningOp(), "cdsize_guard");
                        Value c4_entry = getConst(builder, dispatcherMainLoc, u256Type, i64Type, 4, constCache, entry, "selector_offset");
                        Value shortCalldata = builder.create<sir::LtOp>(dispatcherMainLoc, u256Type, cds_entry, c4_entry);
                        Value cdOk = builder.create<sir::IsZeroOp>(dispatcherMainLoc, u256Type, shortCalldata);
                        setResultName(cdOk.getDefiningOp(), "cd_has_selector");
                        dispatchOk = builder.create<sir::AndOp>(dispatcherMainLoc, u256Type, cv_zero, cdOk);
                        setResultName(dispatchOk.getDefiningOp(), "dispatch_ok");
                    }
                    builder.create<sir::CondBrOp>(dispatcherMainLoc, dispatchOk, ValueRange{}, ValueRange{}, loadSelector, revertError);
                    setBlockName(entry, "main_entry");
                    setBlockOrder(entry, 0);

                    // load_selector: selector + switch
                    builder.setInsertionPointToEnd(loadSelector);
                    Value c0_ls = getConst(builder, dispatcherMainLoc, u256Type, i64Type, 0, constCache, loadSelector, "zero");
                    Value c224_ls = getConst(builder, dispatcherMainLoc, u256Type, i64Type, 224, constCache, loadSelector, "selector_shift");
                    Value word = builder.create<sir::CallDataLoadOp>(dispatcherMainLoc, u256Type, c0_ls);
                    setResultName(word.getDefiningOp(), "selector_word");
                    Value selector = builder.create<sir::ShrOp>(dispatcherMainLoc, u256Type, c224_ls, word);
                    setResultName(selector.getDefiningOp(), "selector");

                    SmallVector<Block *, 8> caseBlocks;
                    SmallVector<int64_t, 8> caseValues;
                    for (auto &info : pubFuncs)
                    {
                        Block *caseCheck = mainFunc.addBlock();
                        caseBlocks.push_back(caseCheck);
                        caseValues.push_back(static_cast<int64_t>(info.selector));
                    }

                    auto caseAttr = builder.getI64ArrayAttr(caseValues);
                    auto sw = builder.create<sir::SwitchOp>(dispatcherMainLoc, selector, caseAttr, revertError, caseBlocks);
                    sw->setAttr("sir.selector_switch", builder.getUnitAttr());
                    setBlockName(loadSelector, "load_selector");
                    setBlockOrder(loadSelector, 1);

                    DenseMap<uint64_t, Block *> abiDecodeRevertBlocks;
                    auto getAbiDecodeRevertBlock = [&](lowering::AbiDecodeError error) -> Block * {
                        uint64_t ordinal = static_cast<uint64_t>(error);
                        auto it = abiDecodeRevertBlocks.find(ordinal);
                        if (it != abiDecodeRevertBlocks.end())
                            return it->second;

                        OpBuilder::InsertionGuard guard(builder);
                        Block *block = mainFunc.addBlock();
                        abiDecodeRevertBlocks.try_emplace(ordinal, block);
                        builder.setInsertionPointToEnd(block);
                        Value size = getConst(builder, dispatcherMainLoc, u256Type, i64Type, 32, constCache, block, "word_size");
                        Value payload = lowering::constU256(builder, dispatcherMainLoc, ordinal);
                        Value ptr = builder.create<sir::SAllocAnyOp>(dispatcherMainLoc, ptrType, size);
                        builder.create<sir::StoreOp>(dispatcherMainLoc, ptr, payload);
                        builder.create<sir::RevertOp>(dispatcherMainLoc, ptr, size);
                        std::string blockName = "abi_decode_revert_" + std::to_string(ordinal);
                        setBlockName(block, blockName);
                        return block;
                    };

                    // case blocks
                    for (size_t i = 0; i < pubFuncs.size(); ++i)
                    {
                        auto &info = pubFuncs[i];
                        Location caseLoc = makeSyntheticOriginOnlyLoc(info.provenanceLoc, "dispatcher_case");
                        Location caseDecodeLoc = makeSyntheticOriginOnlyLoc(info.provenanceLoc, "dispatcher_decode");
                        Location caseReturnLoc = makeSyntheticOriginOnlyLoc(info.provenanceLoc, "dispatcher_return");
                        Location caseErrorLoc = makeSyntheticOriginOnlyLoc(info.provenanceLoc, "dispatcher_error");
                        Block *caseCheck = caseBlocks[i];
                        Block *caseBody = caseCheck;
                        builder.setInsertionPointToEnd(caseCheck);

                        int64_t minSize = info.minHeadBytes > 0 ? info.minHeadBytes : (4 + 32 * static_cast<int64_t>(info.argCount));
                        if (minSize > 4)
                        {
                            Value cdsize_case = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                            setResultName(cdsize_case.getDefiningOp(), ("cdsize_" + info.func.getName()).str());
                            int64_t minMinus = minSize - 1;
                            StringRef minName = (minMinus == 35   ? StringRef("min_cdsize_1arg")
                                                 : minMinus == 67 ? StringRef("min_cdsize_2args")
                                                 : minMinus == 99 ? StringRef("min_cdsize_3args")
                                                                  : StringRef());
                            Value minSizeVal = getConst(builder, caseDecodeLoc, u256Type, i64Type, minMinus, constCache, caseCheck, minName);
                            Value valid_args = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, minSizeVal, cdsize_case);
                            setResultName(valid_args.getDefiningOp(), ("valid_" + info.func.getName()).str());
                            caseBody = mainFunc.addBlock();
                            builder.create<sir::CondBrOp>(caseDecodeLoc, valid_args, ValueRange{}, ValueRange{}, caseBody, revertError);
                            builder.setInsertionPointToEnd(caseBody);
                        }

                        SmallVector<Value, 8> args;
                        SmallVector<int64_t, 8> headOffsets;
                        int64_t headSlot = 0;
                        size_t sourceParamCount = info.abiParamLayouts.empty() ? info.argCount : info.abiParamLayouts.size();
                        for (size_t i = 0; i < sourceParamCount; ++i)
                        {
                            headOffsets.push_back(4 + 32 * headSlot);
                            int64_t slots = 1;
                            if (!info.abiParamLayouts.empty())
                            {
                                AbiLayoutNode layout;
                                if (!parseAbiLayout(info.abiParamLayouts[i], layout, AbiLayoutSyntax::CanonicalAbi))
                                {
                                    info.func.emitError("invalid ABI param layout");
                                    signalPassFailure();
                                    return;
                                }
                                slots = dispatcherHeadSlotsForLayout(layout);
                            }
                            if (slots < 0)
                            {
                                module.emitError("unsupported ABI type for head offset sizing");
                                signalPassFailure();
                                return;
                            }
                            headSlot += slots;
                        }

                        Value nextDynamicOffset;
                        auto getNextDynamicOffset = [&]() -> Value {
                            // Keep the common static/no-arg dispatcher path byte-stable:
                            // this offset is only meaningful once a strict dynamic
                            // calldata parameter is actually being decoded.
                            if (!nextDynamicOffset)
                                nextDynamicOffset = lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(headSlot) * 32ULL);
                            return nextDynamicOffset;
                        };
                        unsigned loweredArgIndex = 0;
                        for (unsigned idx = 0; idx < sourceParamCount; ++idx)
                        {
                            int64_t offs = headOffsets[idx];
                            StringRef offName = offs == 4  ? StringRef("selector_offset")
                                              : offs == 36 ? StringRef("arg1_offset")
                                              : offs == 68 ? StringRef("arg2_offset")
                                                           : StringRef();
                            Value offc = offName.empty()
                                             ? lowering::constU256(builder, caseDecodeLoc, offs)
                                             : getConst(builder, caseDecodeLoc, u256Type, i64Type, offs, constCache, caseBody, offName);
                            Value head = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, offc);
                            StringRef argPrefix = idx == 0 ? "a_" : (idx == 1 ? "b_" : (idx == 2 ? "n_" : "arg_"));
                            setResultName(head.getDefiningOp(), (argPrefix + info.func.getName()).str());

                            AbiLayoutNode abiLayout;
                            const bool hasAbiLayout = !info.abiParamLayouts.empty();
                            if (hasAbiLayout && !parseAbiLayout(info.abiParamLayouts[idx], abiLayout, AbiLayoutSyntax::CanonicalAbi))
                            {
                                info.func.emitError("invalid ABI param layout");
                                signalPassFailure();
                                return;
                            }
                            Value argVal = head;
                            if (hasAbiLayout && abiLayout.kind == AbiLayoutKind::Static)
                            {
                                uint64_t enumVariantCount = idx < info.abiParamEnumCounts.size() ? info.abiParamEnumCounts[idx] : 0;
                                const bool needsRefinementCheck = idx < info.abiParamRefinements.size() && !info.abiParamRefinements[idx].empty();
                                if (std::optional<CalldataStaticDecode> decoded = decodeStaticCalldataWord(builder, caseDecodeLoc, ctx, abiLayout, head, enumVariantCount, needsRefinementCheck, info.permissiveAbiDecode))
                                {
                                    if (needsRefinementCheck)
                                    {
                                        CalldataRefinementSpec refinementSpec;
                                        if (!parseCalldataRefinementSpec(info.abiParamRefinements[idx], refinementSpec))
                                        {
                                            info.func.emitError("invalid public calldata refinement metadata");
                                            signalPassFailure();
                                            return;
                                        }
                                        if (Value refinementValid = calldataRefinementSatisfied(builder, caseDecodeLoc, refinementSpec, decoded->canonicalWord))
                                            decoded->checks.push_back({refinementValid, lowering::AbiDecodeError::RefinementViolation});
                                    }
                                    for (auto [valid, error] : decoded->checks)
                                    {
                                        Block *validBody = mainFunc.addBlock();
                                        builder.create<sir::CondBrOp>(
                                            caseDecodeLoc,
                                            valid,
                                            ValueRange{},
                                            ValueRange{},
                                            validBody,
                                            getAbiDecodeRevertBlock(error));
                                        builder.setInsertionPointToEnd(validBody);
                                        caseBody = validBody;
                                    }
                                    argVal = decoded->payload;
                                }
                            }
                            bool appendWideResultInput = false;
                            Value wideTag;
                            Value widePayload;
                            auto resultInputCarrierExpectsPtr = [&]() -> bool {
                                return loweredArgIndex + 1 < info.inputTypes.size() &&
                                       isa<sir::PtrType>(info.inputTypes[loweredArgIndex + 1]);
                            };
                            auto hasCurrentInputSlot = [&]() -> bool {
                                return loweredArgIndex < info.inputTypes.size();
                            };
                            auto emitStrictCalldataBoundsPrefix = [&](Value frameBaseOff,
                                                                      Value offsetWord,
                                                                      Value expectedOffset,
                                                                      StrictDynamicCalldataKind kind,
                                                                      uint64_t arrayElementWords = 1) -> StrictDynamicBounds {
                                return emitStrictDynamicBoundsPrefix(
                                    builder,
                                    caseDecodeLoc,
                                    u256Type,
                                    offsetWord,
                                    expectedOffset,
                                    kind,
                                    [&]() -> Block * { return mainFunc.addBlock(); },
                                    [&](lowering::AbiDecodeError error) -> Block * { return getAbiDecodeRevertBlock(error); },
                                    [&]() -> Value { return builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type); },
                                    [&](Value) -> StrictDynamicBoundsBase {
                                        Value absOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, frameBaseOff, offsetWord);
                                        return StrictDynamicBoundsBase{absOff, absOff};
                                    },
                                    [&](Value absOff) -> Value {
                                        return builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, absOff);
                                    },
                                    info.permissiveAbiDecode,
                                    arrayElementWords);
                            };
                            auto materializeStrictDynamicCalldataValue = [&](Value offsetWord,
                                                                             Value expectedOffset,
                                                                             StrictDynamicCalldataKind kind,
                                                                             unsigned fixedBytesWidth = 0,
                                                                             uint64_t arrayElementWords = 1) -> FailureOr<StrictDynamicCalldataValue> {
                                if (!hasCurrentInputSlot())
                                    return failure();

                                StrictDynamicBounds bounds = emitStrictCalldataBoundsPrefix(
                                    lowering::constU256(builder, caseDecodeLoc, 4),
                                    offsetWord,
                                    expectedOffset,
                                    kind,
                                    arrayElementWords);
                                Block *copyBlock = mainFunc.addBlock();
                                Value cdsize = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                                Value tailEnd = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, bounds.total);
                                Value tailMissing = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, cdsize, tailEnd);
                                Value tailPresent = builder.create<sir::IsZeroOp>(caseDecodeLoc, u256Type, tailMissing);
                                const bool validatesWordElements = strictDynamicCalldataValidatesWordElements(kind) && !info.permissiveAbiDecode;
                                Block *wordArrayValidateCondBlock = nullptr;
                                Block *wordArrayValidateBodyBlock = nullptr;
                                Block *wordArrayValidateDoneBlock = nullptr;
                                if (validatesWordElements)
                                {
                                    wordArrayValidateCondBlock = mainFunc.addBlock();
                                    wordArrayValidateCondBlock->addArgument(u256Type, caseDecodeLoc);
                                    wordArrayValidateCondBlock->addArgument(u256Type, caseDecodeLoc);
                                    wordArrayValidateBodyBlock = mainFunc.addBlock();
                                    wordArrayValidateBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    wordArrayValidateBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    wordArrayValidateDoneBlock = mainFunc.addBlock();
                                    wordArrayValidateDoneBlock->addArgument(u256Type, caseDecodeLoc);
                                }
                                builder.create<sir::CondBrOp>(
                                    caseDecodeLoc,
                                    tailPresent,
                                    validatesWordElements
                                        ? ValueRange{
                                              lowering::constU256(builder, caseDecodeLoc, 0),
                                              lowering::constU256(builder, caseDecodeLoc, 1),
                                          }
                                        : ValueRange{},
                                    ValueRange{},
                                    validatesWordElements ? wordArrayValidateCondBlock : copyBlock,
                                    getAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                                if (validatesWordElements)
                                {
                                    builder.setInsertionPointToEnd(wordArrayValidateCondBlock);
                                    Value iv = wordArrayValidateCondBlock->getArgument(0);
                                    Value allValid = wordArrayValidateCondBlock->getArgument(1);
                                    Value hasElement = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, iv, bounds.dynamicLen);
                                    Value continueValidation = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, hasElement, allValid);
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        continueValidation,
                                        ValueRange{iv, allValid},
                                        ValueRange{allValid},
                                        wordArrayValidateBodyBlock,
                                        wordArrayValidateDoneBlock);

                                    builder.setInsertionPointToEnd(wordArrayValidateBodyBlock);
                                    Value bodyIv = wordArrayValidateBodyBlock->getArgument(0);
                                    Value bodyAllValid = wordArrayValidateBodyBlock->getArgument(1);
                                    Value elementByteOffset = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bodyIv, bounds.wordSize);
                                    Value elementTailOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, elementByteOffset, bounds.wordSize);
                                    Value elementAbsOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, elementTailOffset);
                                    Value elementWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, elementAbsOffset);
                                    Value elementValid = nullptr;
                                    if (kind == StrictDynamicCalldataKind::AddressArray)
                                    {
                                        Value elementPayload = lowering::maskLowBits(builder, caseDecodeLoc, elementWord, 160);
                                        elementValid = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, elementWord, elementPayload);
                                    }
                                    else if (kind == StrictDynamicCalldataKind::BoolArray)
                                    {
                                        elementValid = lowering::boolAbiWordIsCanonical(builder, caseDecodeLoc, elementWord);
                                    }
                                    else
                                    {
                                        lowering::FixedBytesWordDecode decoded = lowering::decodeFixedBytesAbiWord(builder, caseDecodeLoc, fixedBytesWidth, elementWord);
                                        elementValid = decoded.valid;
                                    }
                                    Value nextAllValid = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, bodyAllValid, elementValid);
                                    Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                    builder.create<sir::BrOp>(caseDecodeLoc, ValueRange{nextIv, nextAllValid}, wordArrayValidateCondBlock);

                                    builder.setInsertionPointToEnd(wordArrayValidateDoneBlock);
                                    Value validElements = wordArrayValidateDoneBlock->getArgument(0);
                                    FailureOr<lowering::AbiDecodeError> invalidElementError = strictDynamicCalldataInvalidElementError(kind);
                                    if (failed(invalidElementError))
                                        return failure();
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        validElements,
                                        ValueRange{},
                                        ValueRange{},
                                        copyBlock,
                                        getAbiDecodeRevertBlock(*invalidElementError));
                                }

                                builder.setInsertionPointToEnd(copyBlock);
                                if (kind == StrictDynamicCalldataKind::FixedBytesArray ||
                                    (info.permissiveAbiDecode &&
                                     (kind == StrictDynamicCalldataKind::AddressArray ||
                                      kind == StrictDynamicCalldataKind::BoolArray)))
                                {
                                    Value resultPtr = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.total);
                                    builder.create<sir::StoreOp>(caseDecodeLoc, resultPtr, bounds.dynamicLen);
                                    Value resultContentPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, resultPtr, bounds.wordSize);

                                    Block *copyCondBlock = mainFunc.addBlock();
                                    copyCondBlock->addArgument(u256Type, caseDecodeLoc);
                                    Block *copyBodyBlock = mainFunc.addBlock();
                                    copyBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    Block *copyDoneBlock = mainFunc.addBlock();

                                    builder.create<sir::BrOp>(
                                        caseDecodeLoc,
                                        ValueRange{lowering::constU256(builder, caseDecodeLoc, 0)},
                                        copyCondBlock);

                                    builder.setInsertionPointToEnd(copyCondBlock);
                                    Value copyIv = copyCondBlock->getArgument(0);
                                    Value hasElement = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, copyIv, bounds.dynamicLen);
                                    builder.create<sir::CondBrOp>(caseDecodeLoc, hasElement, ValueRange{copyIv}, ValueRange{}, copyBodyBlock, copyDoneBlock);

                                    builder.setInsertionPointToEnd(copyBodyBlock);
                                    Value bodyIv = copyBodyBlock->getArgument(0);
                                    Value elementByteOffset = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bodyIv, bounds.wordSize);
                                    Value elementTailOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, elementByteOffset, bounds.wordSize);
                                    Value elementAbsOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, elementTailOffset);
                                    Value elementWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, elementAbsOffset);
                                    Value elementPayload = nullptr;
                                    if (kind == StrictDynamicCalldataKind::AddressArray)
                                        elementPayload = lowering::maskLowBits(builder, caseDecodeLoc, elementWord, 160);
                                    else if (kind == StrictDynamicCalldataKind::BoolArray)
                                        elementPayload = lowering::boolAbiWordPermissivePayload(builder, caseDecodeLoc, elementWord);
                                    else
                                        elementPayload = lowering::decodeFixedBytesAbiWord(builder, caseDecodeLoc, fixedBytesWidth, elementWord).payload;
                                    Value resultElementPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, resultContentPtr, elementByteOffset);
                                    builder.create<sir::StoreOp>(caseDecodeLoc, resultElementPtr, elementPayload);
                                    Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                    builder.create<sir::BrOp>(caseDecodeLoc, ValueRange{nextIv}, copyCondBlock);

                                    builder.setInsertionPointToEnd(copyDoneBlock);
                                    caseBody = copyDoneBlock;
                                    return StrictDynamicCalldataValue{
                                        builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, resultPtr).getResult(),
                                        bounds.nextExpectedOffset,
                                    };
                                }

                                Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.total);
                                builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, bounds.payloadBase, bounds.total);
                                if (kind == StrictDynamicCalldataKind::U256Array ||
                                    kind == StrictDynamicCalldataKind::AddressArray ||
                                    kind == StrictDynamicCalldataKind::BoolArray)
                                {
                                    caseBody = copyBlock;
                                    return StrictDynamicCalldataValue{
                                        builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, buf).getResult(),
                                        bounds.nextExpectedOffset,
                                    };
                                }
                                if (info.permissiveAbiDecode)
                                {
                                    caseBody = copyBlock;
                                    return StrictDynamicCalldataValue{
                                        builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, buf).getResult(),
                                        bounds.nextExpectedOffset,
                                    };
                                }

                                Block *padCondBlock = mainFunc.addBlock();
                                padCondBlock->addArgument(u256Type, caseDecodeLoc);
                                padCondBlock->addArgument(u256Type, caseDecodeLoc);
                                Block *padBodyBlock = mainFunc.addBlock();
                                padBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                padBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                Block *padDoneBlock = mainFunc.addBlock();
                                padDoneBlock->addArgument(u256Type, caseDecodeLoc);
                                Block *okBlock = mainFunc.addBlock();
                                Value contentPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, buf, bounds.wordSize);
                                Value padStart = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, contentPtr, bounds.dynamicLen);
                                Value padCount = builder.create<sir::SubOp>(caseDecodeLoc, u256Type, bounds.padded, bounds.dynamicLen);
                                builder.create<sir::BrOp>(
                                    caseDecodeLoc,
                                    ValueRange{
                                        lowering::constU256(builder, caseDecodeLoc, 0),
                                        lowering::constU256(builder, caseDecodeLoc, 1),
                                    },
                                    padCondBlock);

                                builder.setInsertionPointToEnd(padCondBlock);
                                Value padIv = padCondBlock->getArgument(0);
                                Value padAllValid = padCondBlock->getArgument(1);
                                Value hasPadByte = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, padIv, padCount);
                                Value continuePad = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, hasPadByte, padAllValid);
                                builder.create<sir::CondBrOp>(
                                    caseDecodeLoc,
                                    continuePad,
                                    ValueRange{padIv, padAllValid},
                                    ValueRange{padAllValid},
                                    padBodyBlock,
                                    padDoneBlock);

                                builder.setInsertionPointToEnd(padBodyBlock);
                                Value bodyIv = padBodyBlock->getArgument(0);
                                Value bodyAllValid = padBodyBlock->getArgument(1);
                                Value padBytePtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, padStart, bodyIv);
                                Value padByte = builder.create<sir::Load8Op>(caseDecodeLoc, u256Type, padBytePtr, lowering::constU256(builder, caseDecodeLoc, 0));
                                Value byteIsZero = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, padByte, lowering::constU256(builder, caseDecodeLoc, 0));
                                Value nextAllValid = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, bodyAllValid, byteIsZero);
                                Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                builder.create<sir::BrOp>(caseDecodeLoc, ValueRange{nextIv, nextAllValid}, padCondBlock);

                                builder.setInsertionPointToEnd(padDoneBlock);
                                Value paddingValid = padDoneBlock->getArgument(0);
                                builder.create<sir::CondBrOp>(
                                    caseDecodeLoc,
                                    paddingValid,
                                    ValueRange{},
                                    ValueRange{},
                                    okBlock,
                                    getAbiDecodeRevertBlock(lowering::AbiDecodeError::NonCanonicalEncoding));

                                builder.setInsertionPointToEnd(okBlock);
                                caseBody = okBlock;
                                return StrictDynamicCalldataValue{
                                    builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, buf).getResult(),
                                    bounds.nextExpectedOffset,
                                };
                            };
                            auto resultFieldDynamicKind = [&](const AbiLayoutNode &fieldLayout, StrictDynamicCalldataKind &kind, unsigned &fixedBytesWidth) -> bool {
                                fixedBytesWidth = 0;
                                if (!canonicalAbiLayoutIsDynamic(fieldLayout))
                                    return false;
                                if (canonicalAbiLayoutIsTupleLike(fieldLayout))
                                    return false;
                                if (fieldLayout.kind == AbiLayoutKind::DynamicBytes)
                                {
                                    kind = StrictDynamicCalldataKind::BytesLike;
                                    return true;
                                }
                                if (isDynamicU256ArrayAbiNode(fieldLayout))
                                {
                                    kind = StrictDynamicCalldataKind::U256Array;
                                    return true;
                                }
                                if (isDynamicAddressArrayAbiNode(fieldLayout))
                                {
                                    kind = StrictDynamicCalldataKind::AddressArray;
                                    return true;
                                }
                                if (isDynamicBoolArrayAbiNode(fieldLayout))
                                {
                                    kind = StrictDynamicCalldataKind::BoolArray;
                                    return true;
                                }
                                if (isDynamicFixedBytesArrayAbiNode(fieldLayout))
                                {
                                    kind = StrictDynamicCalldataKind::FixedBytesArray;
                                    fixedBytesWidth = fieldLayout.children.front()->width;
                                    return true;
                                }
                                return false;
                            };

                            auto validateNestedDynamicCalldataValue = [&](Value frameBaseOff,
                                                                          Value offsetWord,
                                                                          Value expectedOffset,
                                                                          StrictDynamicCalldataKind kind,
                                                                          unsigned fixedBytesWidth = 0) -> FailureOr<Value> {
                                StrictDynamicBounds bounds = emitStrictCalldataBoundsPrefix(
                                    frameBaseOff,
                                    offsetWord,
                                    expectedOffset,
                                    kind);
                                Block *doneBlock = mainFunc.addBlock();

                                Value cdsize = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                                Value tailEnd = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, bounds.total);
                                Value tailMissing = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, cdsize, tailEnd);
                                Value tailPresent = builder.create<sir::IsZeroOp>(caseDecodeLoc, u256Type, tailMissing);
                                const bool validatesWordElements = strictDynamicCalldataValidatesWordElements(kind) && !info.permissiveAbiDecode;
                                Block *wordArrayValidateCondBlock = nullptr;
                                Block *wordArrayValidateBodyBlock = nullptr;
                                Block *wordArrayValidateDoneBlock = nullptr;
                                if (validatesWordElements)
                                {
                                    wordArrayValidateCondBlock = mainFunc.addBlock();
                                    wordArrayValidateCondBlock->addArgument(u256Type, caseDecodeLoc);
                                    wordArrayValidateCondBlock->addArgument(u256Type, caseDecodeLoc);
                                    wordArrayValidateBodyBlock = mainFunc.addBlock();
                                    wordArrayValidateBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    wordArrayValidateBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    wordArrayValidateDoneBlock = mainFunc.addBlock();
                                    wordArrayValidateDoneBlock->addArgument(u256Type, caseDecodeLoc);
                                }
                                else if (kind == StrictDynamicCalldataKind::BytesLike)
                                {
                                    doneBlock->addArgument(u256Type, caseDecodeLoc);
                                    doneBlock->addArgument(u256Type, caseDecodeLoc);
                                }
                                builder.create<sir::CondBrOp>(
                                    caseDecodeLoc,
                                    tailPresent,
                                    validatesWordElements
                                        ? ValueRange{
                                              lowering::constU256(builder, caseDecodeLoc, 0),
                                              lowering::constU256(builder, caseDecodeLoc, 1),
                                          }
                                    : kind == StrictDynamicCalldataKind::BytesLike
                                        ? ValueRange{
                                              lowering::constU256(builder, caseDecodeLoc, 0),
                                              lowering::constU256(builder, caseDecodeLoc, 1),
                                          }
                                        : ValueRange{},
                                    ValueRange{},
                                    validatesWordElements
                                        ? wordArrayValidateCondBlock
                                    : kind == StrictDynamicCalldataKind::BytesLike
                                        ? doneBlock
                                        : doneBlock,
                                    getAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                                if (validatesWordElements)
                                {
                                    builder.setInsertionPointToEnd(wordArrayValidateCondBlock);
                                    Value iv = wordArrayValidateCondBlock->getArgument(0);
                                    Value allValid = wordArrayValidateCondBlock->getArgument(1);
                                    Value hasElement = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, iv, bounds.dynamicLen);
                                    Value continueValidation = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, hasElement, allValid);
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        continueValidation,
                                        ValueRange{iv, allValid},
                                        ValueRange{allValid},
                                        wordArrayValidateBodyBlock,
                                        wordArrayValidateDoneBlock);

                                    builder.setInsertionPointToEnd(wordArrayValidateBodyBlock);
                                    Value bodyIv = wordArrayValidateBodyBlock->getArgument(0);
                                    Value bodyAllValid = wordArrayValidateBodyBlock->getArgument(1);
                                    Value elementByteOffset = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bodyIv, bounds.wordSize);
                                    Value elementTailOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, elementByteOffset, bounds.wordSize);
                                    Value elementAbsOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, elementTailOffset);
                                    Value elementWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, elementAbsOffset);
                                    Value elementValid = nullptr;
                                    if (kind == StrictDynamicCalldataKind::AddressArray)
                                    {
                                        Value elementPayload = lowering::maskLowBits(builder, caseDecodeLoc, elementWord, 160);
                                        elementValid = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, elementWord, elementPayload);
                                    }
                                    else if (kind == StrictDynamicCalldataKind::BoolArray)
                                    {
                                        elementValid = lowering::boolAbiWordIsCanonical(builder, caseDecodeLoc, elementWord);
                                    }
                                    else
                                    {
                                        lowering::FixedBytesWordDecode decoded = lowering::decodeFixedBytesAbiWord(builder, caseDecodeLoc, fixedBytesWidth, elementWord);
                                        elementValid = decoded.valid;
                                    }
                                    Value nextAllValid = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, bodyAllValid, elementValid);
                                    Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                    builder.create<sir::BrOp>(caseDecodeLoc, ValueRange{nextIv, nextAllValid}, wordArrayValidateCondBlock);

                                    builder.setInsertionPointToEnd(wordArrayValidateDoneBlock);
                                    Value validElements = wordArrayValidateDoneBlock->getArgument(0);
                                    FailureOr<lowering::AbiDecodeError> invalidElementError = strictDynamicCalldataInvalidElementError(kind);
                                    if (failed(invalidElementError))
                                        return failure();
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        validElements,
                                        ValueRange{},
                                        ValueRange{},
                                        doneBlock,
                                        getAbiDecodeRevertBlock(*invalidElementError));
                                }

                                if (kind == StrictDynamicCalldataKind::BytesLike && info.permissiveAbiDecode)
                                {
                                    builder.setInsertionPointToEnd(doneBlock);
                                    caseBody = doneBlock;
                                    return bounds.nextExpectedOffset;
                                }

                                if (kind == StrictDynamicCalldataKind::BytesLike)
                                {
                                    Block *padCondBlock = doneBlock;
                                    Block *padBodyBlock = mainFunc.addBlock();
                                    padBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    padBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    Block *padDoneBlock = mainFunc.addBlock();
                                    padDoneBlock->addArgument(u256Type, caseDecodeLoc);
                                    Block *okBlock = mainFunc.addBlock();

                                    builder.setInsertionPointToEnd(padCondBlock);
                                    Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.total);
                                    builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, bounds.payloadBase, bounds.total);
                                    Value padIv = padCondBlock->getArgument(0);
                                    Value padAllValid = padCondBlock->getArgument(1);
                                    Value contentPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, buf, bounds.wordSize);
                                    Value padStart = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, contentPtr, bounds.dynamicLen);
                                    Value padCount = builder.create<sir::SubOp>(caseDecodeLoc, u256Type, bounds.padded, bounds.dynamicLen);
                                    Value hasPadByte = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, padIv, padCount);
                                    Value continuePad = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, hasPadByte, padAllValid);
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        continuePad,
                                        ValueRange{padIv, padAllValid},
                                        ValueRange{padAllValid},
                                        padBodyBlock,
                                        padDoneBlock);

                                    builder.setInsertionPointToEnd(padBodyBlock);
                                    Value bodyIv = padBodyBlock->getArgument(0);
                                    Value bodyAllValid = padBodyBlock->getArgument(1);
                                    Value padBytePtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, padStart, bodyIv);
                                    Value padByte = builder.create<sir::Load8Op>(caseDecodeLoc, u256Type, padBytePtr, lowering::constU256(builder, caseDecodeLoc, 0));
                                    Value byteIsZero = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, padByte, lowering::constU256(builder, caseDecodeLoc, 0));
                                    Value nextAllValid = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, bodyAllValid, byteIsZero);
                                    Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                    builder.create<sir::BrOp>(caseDecodeLoc, ValueRange{nextIv, nextAllValid}, padCondBlock);

                                    builder.setInsertionPointToEnd(padDoneBlock);
                                    Value paddingValid = padDoneBlock->getArgument(0);
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        paddingValid,
                                        ValueRange{},
                                        ValueRange{},
                                        okBlock,
                                        getAbiDecodeRevertBlock(lowering::AbiDecodeError::NonCanonicalEncoding));

                                    builder.setInsertionPointToEnd(okBlock);
                                    caseBody = okBlock;
                                    return bounds.nextExpectedOffset;
                                }

                                builder.setInsertionPointToEnd(doneBlock);
                                caseBody = doneBlock;
                                return bounds.nextExpectedOffset;
                            };

                            std::function<bool(const AbiLayoutNode &, Value, int64_t)> validateStaticCalldataFieldsAtHead =
                                [&](const AbiLayoutNode &layout, Value baseOff, int64_t headByteOffset) -> bool {
                                if (canonicalAbiLayoutIsDynamic(layout))
                                    return true;

                                if (layout.kind == AbiLayoutKind::Static)
                                {
                                    Value fieldHeadOff = builder.create<sir::AddOp>(
                                        caseDecodeLoc,
                                        u256Type,
                                        baseOff,
                                        lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(headByteOffset)));
                                    Value word = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, fieldHeadOff);
                                    std::optional<CalldataStaticDecode> decoded =
                                        decodeStaticCalldataWord(builder, caseDecodeLoc, ctx, layout, word, 0, false, info.permissiveAbiDecode);
                                    if (!decoded)
                                        return true;

                                    for (auto [valid, error] : decoded->checks)
                                    {
                                        Block *validBody = mainFunc.addBlock();
                                        builder.create<sir::CondBrOp>(
                                            caseDecodeLoc,
                                            valid,
                                            ValueRange{},
                                            ValueRange{},
                                            validBody,
                                            getAbiDecodeRevertBlock(error));
                                        builder.setInsertionPointToEnd(validBody);
                                        caseBody = validBody;
                                    }
                                    return true;
                                }

                                if (canonicalAbiLayoutIsTupleLike(layout))
                                {
                                    int64_t childHeadByteOffset = headByteOffset;
                                    for (const auto &childLayoutPtr : layout.children)
                                    {
                                        const AbiLayoutNode &childLayout = *childLayoutPtr;
                                        if (!validateStaticCalldataFieldsAtHead(childLayout, baseOff, childHeadByteOffset))
                                            return false;
                                        int64_t slots = canonicalAbiLayoutIsDynamic(childLayout) ? 1 : canonicalAbiLayoutHeadSlots(childLayout);
                                        if (slots <= 0)
                                            return false;
                                        childHeadByteOffset += slots * 32;
                                    }
                                    return true;
                                }

                                if (layout.kind == AbiLayoutKind::FixedArray && layout.children.size() == 1)
                                {
                                    const AbiLayoutNode &elementLayout = *layout.children.front();
                                    int64_t elementSlots = canonicalAbiLayoutHeadSlots(elementLayout);
                                    if (elementSlots <= 0)
                                        return false;
                                    int64_t elementHeadByteOffset = headByteOffset;
                                    for (unsigned i = 0; i < layout.arrayLen; ++i)
                                    {
                                        if (!validateStaticCalldataFieldsAtHead(elementLayout, baseOff, elementHeadByteOffset))
                                            return false;
                                        elementHeadByteOffset += elementSlots * 32;
                                    }
                                    return true;
                                }

                                return true;
                            };

                            auto materializeDynamicTupleCarrierAtHead = [&](const AbiLayoutNode &fieldLayout,
                                                                            Value tupleBaseOff,
                                                                            Value fieldHeadOff,
                                                                            Value expectedDynamicOffset) -> FailureOr<StrictDynamicCalldataValue> {
                                int64_t tupleHeadSlots = canonicalAbiLayoutHeadSlots(fieldLayout);
                                if (tupleHeadSlots <= 0)
                                    return failure();

                                Block *offsetOkBlock = mainFunc.addBlock();
                                Block *headOkBlock = mainFunc.addBlock();
                                Value offsetWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, fieldHeadOff);
                                Value offsetOk = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, offsetWord, expectedDynamicOffset);
                                builder.create<sir::CondBrOp>(
                                    caseDecodeLoc,
                                    offsetOk,
                                    ValueRange{},
                                    ValueRange{},
                                    offsetOkBlock,
                                    getAbiDecodeRevertBlock(lowering::AbiDecodeError::NonCanonicalEncoding));

                                builder.setInsertionPointToEnd(offsetOkBlock);
                                Value fieldBaseOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, tupleBaseOff, offsetWord);
                                Value cdsize = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                                Value tupleHeadBytes = lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(tupleHeadSlots) * 32ULL);
                                Value tupleHeadEnd = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, fieldBaseOff, tupleHeadBytes);
                                Value headMissing = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, cdsize, tupleHeadEnd);
                                Value headPresent = builder.create<sir::IsZeroOp>(caseDecodeLoc, u256Type, headMissing);
                                builder.create<sir::CondBrOp>(
                                    caseDecodeLoc,
                                    headPresent,
                                    ValueRange{},
                                    ValueRange{},
                                    headOkBlock,
                                    getAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                                builder.setInsertionPointToEnd(headOkBlock);
                                Value nextFieldDynamicOffset = tupleHeadBytes;
                                SmallVector<DynamicTupleChild, 2> dynamicChildren;
                                int64_t childHeadByteOffset = 0;
                                for (const auto &childLayoutPtr : fieldLayout.children)
                                {
                                    const AbiLayoutNode &childLayout = *childLayoutPtr;
                                    if (canonicalAbiLayoutIsDynamic(childLayout))
                                    {
                                        StrictDynamicCalldataKind childKind;
                                        unsigned fixedBytesWidth = 0;
                                        if (!resultFieldDynamicKind(childLayout, childKind, fixedBytesWidth))
                                            return failure();

                                        Value childHeadOff = builder.create<sir::AddOp>(
                                            caseDecodeLoc,
                                            u256Type,
                                            fieldBaseOff,
                                            lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(childHeadByteOffset)));
                                        Value childOffset = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, childHeadOff);
                                        FailureOr<Value> childNextOffset = validateNestedDynamicCalldataValue(
                                            fieldBaseOff,
                                            childOffset,
                                            nextFieldDynamicOffset,
                                            childKind,
                                            fixedBytesWidth);
                                        if (failed(childNextOffset))
                                            return failure();
                                        dynamicChildren.push_back({childHeadByteOffset, childKind, fixedBytesWidth});
                                        nextFieldDynamicOffset = *childNextOffset;
                                        childHeadByteOffset += 32;
                                    }
                                    else
                                    {
                                        int64_t slots = canonicalAbiLayoutHeadSlots(childLayout);
                                        if (slots <= 0)
                                            return failure();
                                        if (!validateStaticCalldataFieldsAtHead(childLayout, fieldBaseOff, childHeadByteOffset))
                                            return failure();
                                        childHeadByteOffset += slots * 32;
                                    }
                                }

                                Value resultPtr = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, nextFieldDynamicOffset);
                                builder.create<sir::CallDataCopyOp>(caseDecodeLoc, resultPtr, fieldBaseOff, nextFieldDynamicOffset);

                                for (const DynamicTupleChild &child : dynamicChildren)
                                {
                                    Value childHeadPtr = builder.create<sir::AddPtrOp>(
                                        caseDecodeLoc,
                                        ptrType,
                                        resultPtr,
                                        lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(child.headByteOffset)));
                                    Value childOffset = builder.create<sir::LoadOp>(caseDecodeLoc, u256Type, childHeadPtr);
                                    Value childPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, resultPtr, childOffset);
                                    Value childPayload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, childPtr);
                                    builder.create<sir::StoreOp>(caseDecodeLoc, childHeadPtr, childPayload);
                                }

                                Value nextOuterDynamicOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, offsetWord, nextFieldDynamicOffset);
                                Value payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, resultPtr);
                                return StrictDynamicCalldataValue{payload, nextOuterDynamicOffset};
                            };

                            auto materializeResultCarrier = [&](const AbiLayoutNode &fieldLayout,
                                                                Value tupleBaseOff,
                                                                int64_t fieldHeadByteOffset,
                                                                Value expectedDynamicOffset,
                                                                bool wrapDynamicPayloadInSingleFieldStruct = false) -> FailureOr<StrictDynamicCalldataValue> {
                                bool expectsPtr = resultInputCarrierExpectsPtr();
                                Value fieldHeadOff = builder.create<sir::AddOp>(
                                    caseDecodeLoc,
                                    u256Type,
                                    tupleBaseOff,
                                    getConst(builder, caseDecodeLoc, u256Type, i64Type, fieldHeadByteOffset, constCache, caseBody)
                                );

                                if (canonicalAbiLayoutIsDynamic(fieldLayout))
                                {
                                    if (canonicalAbiLayoutIsTupleLike(fieldLayout))
                                    {
                                        int64_t tupleHeadSlots = canonicalAbiLayoutHeadSlots(fieldLayout);
                                        if (tupleHeadSlots <= 0)
                                            return failure();

                                        Block *offsetOkBlock = mainFunc.addBlock();
                                        Block *headOkBlock = mainFunc.addBlock();
                                        Value offsetWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, fieldHeadOff);
                                        Value offsetOk = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, offsetWord, expectedDynamicOffset);
                                        builder.create<sir::CondBrOp>(
                                            caseDecodeLoc,
                                            offsetOk,
                                            ValueRange{},
                                            ValueRange{},
                                            offsetOkBlock,
                                            getAbiDecodeRevertBlock(lowering::AbiDecodeError::NonCanonicalEncoding));

                                        builder.setInsertionPointToEnd(offsetOkBlock);
                                        Value wordSize = lowering::constU256(builder, caseDecodeLoc, 32);
                                        Value fieldBaseOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, tupleBaseOff, offsetWord);
                                        Value cdsize = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                                        Value tupleHeadBytes = lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(tupleHeadSlots) * 32ULL);
                                        Value tupleHeadEnd = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, fieldBaseOff, tupleHeadBytes);
                                        Value headMissing = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, cdsize, tupleHeadEnd);
                                        Value headPresent = builder.create<sir::IsZeroOp>(caseDecodeLoc, u256Type, headMissing);
                                        builder.create<sir::CondBrOp>(
                                            caseDecodeLoc,
                                            headPresent,
                                            ValueRange{},
                                            ValueRange{},
                                            headOkBlock,
                                            getAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                                        builder.setInsertionPointToEnd(headOkBlock);
                                        Value nextFieldDynamicOffset = tupleHeadBytes;
                                        SmallVector<DynamicTupleChild, 2> dynamicChildren;
                                        int64_t childHeadByteOffset = 0;
                                        for (const auto &childLayoutPtr : fieldLayout.children)
                                        {
                                            const AbiLayoutNode &childLayout = *childLayoutPtr;
                                            if (canonicalAbiLayoutIsDynamic(childLayout))
                                            {
                                                StrictDynamicCalldataKind childKind;
                                                unsigned fixedBytesWidth = 0;
                                                if (!resultFieldDynamicKind(childLayout, childKind, fixedBytesWidth))
                                                    return failure();

                                                Value childHeadOff = builder.create<sir::AddOp>(
                                                    caseDecodeLoc,
                                                    u256Type,
                                                    fieldBaseOff,
                                                    lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(childHeadByteOffset)));
                                                Value childOffset = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, childHeadOff);
                                                FailureOr<Value> childNextOffset = validateNestedDynamicCalldataValue(
                                                    fieldBaseOff,
                                                    childOffset,
                                                    nextFieldDynamicOffset,
                                                    childKind,
                                                    fixedBytesWidth);
                                                if (failed(childNextOffset))
                                                    return failure();
                                                dynamicChildren.push_back({childHeadByteOffset, childKind, fixedBytesWidth});
                                                nextFieldDynamicOffset = *childNextOffset;
                                                childHeadByteOffset += 32;
                                            }
                                            else
                                            {
                                                int64_t slots = canonicalAbiLayoutHeadSlots(childLayout);
                                                if (slots <= 0)
                                                    return failure();
                                                if (!validateStaticCalldataFieldsAtHead(childLayout, fieldBaseOff, childHeadByteOffset))
                                                    return failure();
                                                childHeadByteOffset += slots * 32;
                                            }
                                        }

                                        Value resultPtr = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, nextFieldDynamicOffset);
                                        builder.create<sir::CallDataCopyOp>(caseDecodeLoc, resultPtr, fieldBaseOff, nextFieldDynamicOffset);

                                        for (const DynamicTupleChild &child : dynamicChildren)
                                        {
                                            Value childHeadPtr = builder.create<sir::AddPtrOp>(
                                                caseDecodeLoc,
                                                ptrType,
                                                resultPtr,
                                                lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(child.headByteOffset)));
                                            Value childOffset = builder.create<sir::LoadOp>(caseDecodeLoc, u256Type, childHeadPtr);
                                            Value childPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, resultPtr, childOffset);
                                            Value childPayload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, childPtr);
                                            builder.create<sir::StoreOp>(caseDecodeLoc, childHeadPtr, childPayload);
                                            if (child.kind != StrictDynamicCalldataKind::FixedBytesArray)
                                                continue;

                                            Value childLen = builder.create<sir::LoadOp>(caseDecodeLoc, u256Type, childPtr);
                                            Value childContentPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, childPtr, wordSize);

                                            Block *copyCondBlock = mainFunc.addBlock();
                                            copyCondBlock->addArgument(u256Type, caseDecodeLoc);
                                            Block *copyBodyBlock = mainFunc.addBlock();
                                            copyBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                            Block *copyDoneBlock = mainFunc.addBlock();

                                            builder.create<sir::BrOp>(
                                                caseDecodeLoc,
                                                ValueRange{lowering::constU256(builder, caseDecodeLoc, 0)},
                                                copyCondBlock);

                                            builder.setInsertionPointToEnd(copyCondBlock);
                                            Value copyIv = copyCondBlock->getArgument(0);
                                            Value hasElement = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, copyIv, childLen);
                                            builder.create<sir::CondBrOp>(caseDecodeLoc, hasElement, ValueRange{copyIv}, ValueRange{}, copyBodyBlock, copyDoneBlock);

                                            builder.setInsertionPointToEnd(copyBodyBlock);
                                            Value bodyIv = copyBodyBlock->getArgument(0);
                                            Value elementByteOffset = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bodyIv, wordSize);
                                            Value elementPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, childContentPtr, elementByteOffset);
                                            Value elementWord = builder.create<sir::LoadOp>(caseDecodeLoc, u256Type, elementPtr);
                                            lowering::FixedBytesWordDecode decoded = lowering::decodeFixedBytesAbiWord(builder, caseDecodeLoc, child.fixedBytesWidth, elementWord);
                                            builder.create<sir::StoreOp>(caseDecodeLoc, elementPtr, decoded.payload);
                                            Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                            builder.create<sir::BrOp>(caseDecodeLoc, ValueRange{nextIv}, copyCondBlock);

                                            builder.setInsertionPointToEnd(copyDoneBlock);
                                            caseBody = copyDoneBlock;
                                        }

                                        Value nextOuterDynamicOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, offsetWord, nextFieldDynamicOffset);
                                        Value payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, resultPtr);
                                        if (wrapDynamicPayloadInSingleFieldStruct)
                                        {
                                            Value wrapper = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, lowering::constU256(builder, caseDecodeLoc, 32));
                                            builder.create<sir::StoreOp>(caseDecodeLoc, wrapper, payload);
                                            payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, wrapper);
                                        }
                                        return StrictDynamicCalldataValue{payload, nextOuterDynamicOffset};
                                    }

                                    StrictDynamicCalldataKind kind;
                                    unsigned fixedBytesWidth = 0;
                                    if (!resultFieldDynamicKind(fieldLayout, kind, fixedBytesWidth))
                                        return failure();

                                    Value offsetWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, fieldHeadOff);
                                    StrictDynamicBounds bounds = emitStrictCalldataBoundsPrefix(
                                        tupleBaseOff,
                                        offsetWord,
                                        expectedDynamicOffset,
                                        kind);
                                    Block *copyBlock = mainFunc.addBlock();

                                    Value cdsize = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                                    Value tailEnd = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, bounds.total);
                                    Value tailMissing = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, cdsize, tailEnd);
                                    Value tailPresent = builder.create<sir::IsZeroOp>(caseDecodeLoc, u256Type, tailMissing);
                                    const bool validatesWordElements = strictDynamicCalldataValidatesWordElements(kind) && !info.permissiveAbiDecode;
                                    Block *wordArrayValidateCondBlock = nullptr;
                                    Block *wordArrayValidateBodyBlock = nullptr;
                                    Block *wordArrayValidateDoneBlock = nullptr;
                                    if (validatesWordElements)
                                    {
                                        wordArrayValidateCondBlock = mainFunc.addBlock();
                                        wordArrayValidateCondBlock->addArgument(u256Type, caseDecodeLoc);
                                        wordArrayValidateCondBlock->addArgument(u256Type, caseDecodeLoc);
                                        wordArrayValidateBodyBlock = mainFunc.addBlock();
                                        wordArrayValidateBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                        wordArrayValidateBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                        wordArrayValidateDoneBlock = mainFunc.addBlock();
                                        wordArrayValidateDoneBlock->addArgument(u256Type, caseDecodeLoc);
                                    }
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        tailPresent,
                                        validatesWordElements
                                            ? ValueRange{
                                                  lowering::constU256(builder, caseDecodeLoc, 0),
                                                  lowering::constU256(builder, caseDecodeLoc, 1),
                                              }
                                            : ValueRange{},
                                        ValueRange{},
                                        validatesWordElements ? wordArrayValidateCondBlock : copyBlock,
                                        getAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                                    if (validatesWordElements)
                                    {
                                        builder.setInsertionPointToEnd(wordArrayValidateCondBlock);
                                        Value iv = wordArrayValidateCondBlock->getArgument(0);
                                        Value allValid = wordArrayValidateCondBlock->getArgument(1);
                                        Value hasElement = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, iv, bounds.dynamicLen);
                                        Value continueValidation = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, hasElement, allValid);
                                        builder.create<sir::CondBrOp>(
                                            caseDecodeLoc,
                                            continueValidation,
                                            ValueRange{iv, allValid},
                                            ValueRange{allValid},
                                            wordArrayValidateBodyBlock,
                                            wordArrayValidateDoneBlock);

                                        builder.setInsertionPointToEnd(wordArrayValidateBodyBlock);
                                        Value bodyIv = wordArrayValidateBodyBlock->getArgument(0);
                                        Value bodyAllValid = wordArrayValidateBodyBlock->getArgument(1);
                                        Value elementByteOffset = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bodyIv, bounds.wordSize);
                                        Value elementTailOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, elementByteOffset, bounds.wordSize);
                                        Value elementAbsOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, elementTailOffset);
                                        Value elementWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, elementAbsOffset);
                                        Value elementValid = nullptr;
                                        if (kind == StrictDynamicCalldataKind::AddressArray)
                                        {
                                            Value elementPayload = lowering::maskLowBits(builder, caseDecodeLoc, elementWord, 160);
                                            elementValid = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, elementWord, elementPayload);
                                        }
                                        else if (kind == StrictDynamicCalldataKind::BoolArray)
                                        {
                                            elementValid = lowering::boolAbiWordIsCanonical(builder, caseDecodeLoc, elementWord);
                                        }
                                        else
                                        {
                                            lowering::FixedBytesWordDecode decoded = lowering::decodeFixedBytesAbiWord(builder, caseDecodeLoc, fixedBytesWidth, elementWord);
                                            elementValid = decoded.valid;
                                        }
                                        Value nextAllValid = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, bodyAllValid, elementValid);
                                        Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                        builder.create<sir::BrOp>(caseDecodeLoc, ValueRange{nextIv, nextAllValid}, wordArrayValidateCondBlock);

                                        builder.setInsertionPointToEnd(wordArrayValidateDoneBlock);
                                        Value validElements = wordArrayValidateDoneBlock->getArgument(0);
                                        FailureOr<lowering::AbiDecodeError> invalidElementError = strictDynamicCalldataInvalidElementError(kind);
                                        if (failed(invalidElementError))
                                            return failure();
                                        builder.create<sir::CondBrOp>(
                                            caseDecodeLoc,
                                            validElements,
                                            ValueRange{},
                                            ValueRange{},
                                            copyBlock,
                                            getAbiDecodeRevertBlock(*invalidElementError));
                                    }

                                    builder.setInsertionPointToEnd(copyBlock);
                                    if (kind == StrictDynamicCalldataKind::FixedBytesArray ||
                                        (info.permissiveAbiDecode &&
                                         (kind == StrictDynamicCalldataKind::AddressArray ||
                                          kind == StrictDynamicCalldataKind::BoolArray)))
                                    {
                                        Value resultPtr = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.total);
                                        builder.create<sir::StoreOp>(caseDecodeLoc, resultPtr, bounds.dynamicLen);
                                        Value resultContentPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, resultPtr, bounds.wordSize);

                                        Block *copyCondBlock = mainFunc.addBlock();
                                        copyCondBlock->addArgument(u256Type, caseDecodeLoc);
                                        Block *copyBodyBlock = mainFunc.addBlock();
                                        copyBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                        Block *copyDoneBlock = mainFunc.addBlock();

                                        builder.create<sir::BrOp>(
                                            caseDecodeLoc,
                                            ValueRange{lowering::constU256(builder, caseDecodeLoc, 0)},
                                            copyCondBlock);

                                        builder.setInsertionPointToEnd(copyCondBlock);
                                        Value copyIv = copyCondBlock->getArgument(0);
                                        Value hasElement = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, copyIv, bounds.dynamicLen);
                                        builder.create<sir::CondBrOp>(caseDecodeLoc, hasElement, ValueRange{copyIv}, ValueRange{}, copyBodyBlock, copyDoneBlock);

                                        builder.setInsertionPointToEnd(copyBodyBlock);
                                        Value bodyIv = copyBodyBlock->getArgument(0);
                                        Value elementByteOffset = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bodyIv, bounds.wordSize);
                                        Value elementTailOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, elementByteOffset, bounds.wordSize);
                                        Value elementAbsOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, elementTailOffset);
                                        Value elementWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, elementAbsOffset);
                                        Value elementPayload = nullptr;
                                        if (kind == StrictDynamicCalldataKind::AddressArray)
                                            elementPayload = lowering::maskLowBits(builder, caseDecodeLoc, elementWord, 160);
                                        else if (kind == StrictDynamicCalldataKind::BoolArray)
                                            elementPayload = lowering::boolAbiWordPermissivePayload(builder, caseDecodeLoc, elementWord);
                                        else
                                            elementPayload = lowering::decodeFixedBytesAbiWord(builder, caseDecodeLoc, fixedBytesWidth, elementWord).payload;
                                        Value resultElementPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, resultContentPtr, elementByteOffset);
                                        builder.create<sir::StoreOp>(caseDecodeLoc, resultElementPtr, elementPayload);
                                        Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                        builder.create<sir::BrOp>(caseDecodeLoc, ValueRange{nextIv}, copyCondBlock);

                                        builder.setInsertionPointToEnd(copyDoneBlock);
                                        caseBody = copyDoneBlock;
                                        Value payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, resultPtr);
                                        if (wrapDynamicPayloadInSingleFieldStruct)
                                        {
                                            Value wrapper = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.wordSize);
                                            builder.create<sir::StoreOp>(caseDecodeLoc, wrapper, payload);
                                            payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, wrapper);
                                        }
                                        return StrictDynamicCalldataValue{payload, bounds.nextExpectedOffset};
                                    }

                                    if (kind == StrictDynamicCalldataKind::U256Array ||
                                        kind == StrictDynamicCalldataKind::AddressArray ||
                                        kind == StrictDynamicCalldataKind::BoolArray)
                                    {
                                        Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.total);
                                        builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, bounds.payloadBase, bounds.total);
                                        caseBody = copyBlock;
                                        Value payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, buf);
                                        if (wrapDynamicPayloadInSingleFieldStruct)
                                        {
                                            Value wrapper = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.wordSize);
                                            builder.create<sir::StoreOp>(caseDecodeLoc, wrapper, payload);
                                            payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, wrapper);
                                        }
                                        return StrictDynamicCalldataValue{payload, bounds.nextExpectedOffset};
                                    }
                                    if (info.permissiveAbiDecode)
                                    {
                                        Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.total);
                                        builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, bounds.payloadBase, bounds.total);
                                        caseBody = copyBlock;
                                        Value payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, buf);
                                        if (wrapDynamicPayloadInSingleFieldStruct)
                                        {
                                            Value wrapper = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.wordSize);
                                            builder.create<sir::StoreOp>(caseDecodeLoc, wrapper, payload);
                                            payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, wrapper);
                                        }
                                        return StrictDynamicCalldataValue{payload, bounds.nextExpectedOffset};
                                    }

                                    Block *padCondBlock = mainFunc.addBlock();
                                    padCondBlock->addArgument(u256Type, caseDecodeLoc);
                                    padCondBlock->addArgument(u256Type, caseDecodeLoc);
                                    padCondBlock->addArgument(u256Type, caseDecodeLoc);
                                    padCondBlock->addArgument(u256Type, caseDecodeLoc);
                                    Block *padBodyBlock = mainFunc.addBlock();
                                    padBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    padBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    padBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    padBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                    Block *padDoneBlock = mainFunc.addBlock();
                                    padDoneBlock->addArgument(u256Type, caseDecodeLoc);
                                    Block *okBlock = mainFunc.addBlock();
                                    Value contentOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, bounds.wordSize);
                                    Value padStart = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, contentOff, bounds.dynamicLen);
                                    Value padCount = builder.create<sir::SubOp>(caseDecodeLoc, u256Type, bounds.padded, bounds.dynamicLen);
                                    builder.create<sir::BrOp>(
                                        caseDecodeLoc,
                                        ValueRange{
                                            lowering::constU256(builder, caseDecodeLoc, 0),
                                            lowering::constU256(builder, caseDecodeLoc, 1),
                                            padStart,
                                            padCount,
                                        },
                                        padCondBlock);

                                    builder.setInsertionPointToEnd(padCondBlock);
                                    Value padIv = padCondBlock->getArgument(0);
                                    Value padAllValid = padCondBlock->getArgument(1);
                                    Value condPadStart = padCondBlock->getArgument(2);
                                    Value condPadCount = padCondBlock->getArgument(3);
                                    Value hasPadByte = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, padIv, condPadCount);
                                    Value continuePad = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, hasPadByte, padAllValid);
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        continuePad,
                                        ValueRange{padIv, padAllValid, condPadStart, condPadCount},
                                        ValueRange{padAllValid},
                                        padBodyBlock,
                                        padDoneBlock);

                                    builder.setInsertionPointToEnd(padBodyBlock);
                                    Value bodyIv = padBodyBlock->getArgument(0);
                                    Value bodyAllValid = padBodyBlock->getArgument(1);
                                    Value bodyPadStart = padBodyBlock->getArgument(2);
                                    Value bodyPadCount = padBodyBlock->getArgument(3);
                                    Value padByteOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyPadStart, bodyIv);
                                    Value padWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, padByteOff);
                                    Value c248_padShift = lowering::constU256(builder, caseDecodeLoc, 248);
                                    Value padByte = builder.create<sir::ShrOp>(caseDecodeLoc, u256Type, c248_padShift, padWord);
                                    Value byteIsZero = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, padByte, lowering::constU256(builder, caseDecodeLoc, 0));
                                    Value nextAllValid = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, bodyAllValid, byteIsZero);
                                    Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                    builder.create<sir::BrOp>(caseDecodeLoc, ValueRange{nextIv, nextAllValid, bodyPadStart, bodyPadCount}, padCondBlock);

                                    builder.setInsertionPointToEnd(padDoneBlock);
                                    Value paddingValid = padDoneBlock->getArgument(0);
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        paddingValid,
                                        ValueRange{},
                                        ValueRange{},
                                        okBlock,
                                        getAbiDecodeRevertBlock(lowering::AbiDecodeError::NonCanonicalEncoding));

                                    builder.setInsertionPointToEnd(okBlock);
                                    caseBody = okBlock;
                                    Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.total);
                                    builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, bounds.payloadBase, bounds.total);
                                    Value payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, buf);
                                    if (wrapDynamicPayloadInSingleFieldStruct)
                                    {
                                        Value wrapper = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, bounds.wordSize);
                                        builder.create<sir::StoreOp>(caseDecodeLoc, wrapper, payload);
                                        payload = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, wrapper);
                                    }
                                    return StrictDynamicCalldataValue{payload, bounds.nextExpectedOffset};
                                }

                                int64_t words = canonicalAbiLayoutHeadSlots(fieldLayout);
                                if (words <= 0)
                                    return failure();
                                if (!expectsPtr && words != 1)
                                    return failure();
                                if (!expectsPtr)
                                    return StrictDynamicCalldataValue{
                                        builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, fieldHeadOff).getResult(),
                                        expectedDynamicOffset,
                                    };

                                Value totalVal = lowering::constU256(builder, caseDecodeLoc, words * 32);
                                Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, totalVal);
                                builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, fieldHeadOff, totalVal);
                                return StrictDynamicCalldataValue{
                                    builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, buf).getResult(),
                                    expectedDynamicOffset,
                                };
                            };

                            auto materializeDynamicTupleArrayCalldataValue = [&](const AbiLayoutNode &arrayLayout,
                                                                                 Value offsetWord,
                                                                                 Value expectedOffset) -> FailureOr<StrictDynamicCalldataValue> {
                                std::optional<uint64_t> elementHeadWords = dynamicTupleArrayElementHeadWords(arrayLayout);
                                if (!elementHeadWords || arrayLayout.children.size() != 1)
                                    return failure();
                                const AbiLayoutNode &elementLayout = *arrayLayout.children.front();

                                StrictDynamicBounds bounds = emitStrictCalldataBoundsPrefix(
                                    lowering::constU256(builder, caseDecodeLoc, 4),
                                    offsetWord,
                                    expectedOffset,
                                    StrictDynamicCalldataKind::U256Array);

                                Block *tableOkBlock = mainFunc.addBlock();
                                Value cdsize = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                                Value elementOffsetBase = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, bounds.wordSize);
                                Value arrayHeadBytes = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bounds.dynamicLen, bounds.wordSize);
                                Value tableEnd = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, elementOffsetBase, arrayHeadBytes);
                                Value tableMissing = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, cdsize, tableEnd);
                                Value tablePresent = builder.create<sir::IsZeroOp>(caseDecodeLoc, u256Type, tableMissing);
                                builder.create<sir::CondBrOp>(
                                    caseDecodeLoc,
                                    tablePresent,
                                    ValueRange{},
                                    ValueRange{},
                                    tableOkBlock,
                                    getAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                                builder.setInsertionPointToEnd(tableOkBlock);
                                Value elementBytes = lowering::constU256(builder, caseDecodeLoc, *elementHeadWords * 32ULL);
                                Value rowBytes = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bounds.dynamicLen, elementBytes);
                                Value resultTotal = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.wordSize, rowBytes);
                                Value resultPtr = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, resultTotal);
                                builder.create<sir::StoreOp>(caseDecodeLoc, resultPtr, bounds.dynamicLen);

                                Value firstElementOffset = arrayHeadBytes;
                                Block *loopCondBlock = mainFunc.addBlock();
                                loopCondBlock->addArgument(u256Type, caseDecodeLoc);
                                loopCondBlock->addArgument(u256Type, caseDecodeLoc);
                                Block *loopBodyBlock = mainFunc.addBlock();
                                loopBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                loopBodyBlock->addArgument(u256Type, caseDecodeLoc);
                                Block *loopDoneBlock = mainFunc.addBlock();
                                loopDoneBlock->addArgument(u256Type, caseDecodeLoc);

                                builder.create<sir::BrOp>(
                                    caseDecodeLoc,
                                    ValueRange{lowering::constU256(builder, caseDecodeLoc, 0), firstElementOffset},
                                    loopCondBlock);

                                builder.setInsertionPointToEnd(loopCondBlock);
                                Value iv = loopCondBlock->getArgument(0);
                                Value expectedElementOffset = loopCondBlock->getArgument(1);
                                Value hasElement = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, iv, bounds.dynamicLen);
                                builder.create<sir::CondBrOp>(
                                    caseDecodeLoc,
                                    hasElement,
                                    ValueRange{iv, expectedElementOffset},
                                    ValueRange{expectedElementOffset},
                                    loopBodyBlock,
                                    loopDoneBlock);

                                builder.setInsertionPointToEnd(loopBodyBlock);
                                Value bodyIv = loopBodyBlock->getArgument(0);
                                Value bodyExpectedElementOffset = loopBodyBlock->getArgument(1);
                                Value offsetIndexBytes = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bodyIv, bounds.wordSize);
                                Value offsetTableByteOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.wordSize, offsetIndexBytes);
                                Value elementHeadOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.payloadBase, offsetTableByteOffset);
                                FailureOr<StrictDynamicCalldataValue> elementCarrier = materializeDynamicTupleCarrierAtHead(
                                    elementLayout,
                                    elementOffsetBase,
                                    elementHeadOff,
                                    bodyExpectedElementOffset);
                                if (failed(elementCarrier))
                                    return failure();

                                Value elementPtr = builder.create<sir::BitcastOp>(caseDecodeLoc, ptrType, elementCarrier->payload);
                                Value rowIndexBytes = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, bodyIv, elementBytes);
                                Value rowByteOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.wordSize, rowIndexBytes);
                                Value rowPtr = builder.create<sir::AddPtrOp>(caseDecodeLoc, ptrType, resultPtr, rowByteOffset);
                                builder.create<sir::MCopyOp>(caseDecodeLoc, rowPtr, elementPtr, elementBytes);
                                Value nextIv = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bodyIv, lowering::constU256(builder, caseDecodeLoc, 1));
                                builder.create<sir::BrOp>(
                                    caseDecodeLoc,
                                    ValueRange{nextIv, elementCarrier->nextExpectedOffset},
                                    loopCondBlock);

                                builder.setInsertionPointToEnd(loopDoneBlock);
                                caseBody = loopDoneBlock;
                                Value finalElementOffset = loopDoneBlock->getArgument(0);
                                Value arrayTotal = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, bounds.wordSize, finalElementOffset);
                                Value nextOuterDynamicOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, offsetWord, arrayTotal);
                                return StrictDynamicCalldataValue{
                                    builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, resultPtr).getResult(),
                                    nextOuterDynamicOffset,
                                };
                            };

                            if (hasAbiLayout &&
                                idx < info.resultInputModes.size() &&
                                info.resultInputModes[idx] == "narrow_payloadless")
                            {
                                const AbiLayoutNode &layout = abiLayout;
                                if (!canonicalAbiLayoutIsTupleLike(layout))
                                {
                                    info.func.emitError("public Result input requires tuple ABI layout");
                                    signalPassFailure();
                                    return;
                                }
                                if (canonicalAbiLayoutIsDynamic(layout) || layout.children.size() != 2 ||
                                    !isStaticBoolAbiNode(*layout.children[0]) ||
                                    canonicalAbiLayoutHeadSlots(*layout.children[0]) != 1 ||
                                    canonicalAbiLayoutIsDynamic(*layout.children[1]) ||
                                    canonicalAbiLayoutHeadSlots(*layout.children[1]) != 1)
                                {
                                    info.func.emitError("public Result input currently requires static layout (bool,payload)");
                                    signalPassFailure();
                                    return;
                                }

                                Value one = getConst(builder, caseDecodeLoc, u256Type, i64Type, 1, constCache, caseBody, "one");
                                Value tag = error_union_helpers::maskedTagWordWithMask(builder, caseDecodeLoc, head, one);
                                Value payloadOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, offc, getConst(builder, caseDecodeLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size"));
                                Value payload = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, payloadOff);
                                Value packedOk = builder.create<sir::ShlOp>(caseDecodeLoc, u256Type, one, payload);
                                Value errId = getConst(builder, caseDecodeLoc, u256Type, i64Type, info.resultInputErrorIds[idx], constCache, caseBody);
                                Value packedErrPayload = builder.create<sir::ShlOp>(caseDecodeLoc, u256Type, one, errId);
                                Value packedErr = builder.create<sir::OrOp>(caseDecodeLoc, u256Type, packedErrPayload, one);
                                Value isError = error_union_helpers::extractedTagIsErrorWithMask(builder, caseDecodeLoc, tag, one);
                                argVal = builder.create<sir::SelectOp>(caseDecodeLoc, u256Type, isError, packedErr, packedOk);
                            }
                            else if (hasAbiLayout &&
                                     idx < info.resultInputModes.size() &&
                                     info.resultInputModes[idx] == "wide_payloadless")
                            {
                                const AbiLayoutNode &layout = abiLayout;
                                if (!canonicalAbiLayoutIsTupleLike(layout))
                                {
                                    info.func.emitError("public wide Result input requires tuple ABI layout");
                                    signalPassFailure();
                                    return;
                                }
                                if (layout.children.size() != 2 || !isStaticBoolAbiNode(*layout.children[0]) ||
                                    canonicalAbiLayoutHeadSlots(*layout.children[0]) != 1)
                                {
                                    info.func.emitError("public Result input currently requires layout (bool,payload)");
                                    signalPassFailure();
                                    return;
                                }

                                if (loweredArgIndex + 1 >= info.inputTypes.size())
                                {
                                    info.func.emitError("wide Result input metadata does not match lowered argument types");
                                    signalPassFailure();
                                    return;
                                }

                                int64_t resultHeadSlots = canonicalAbiLayoutHeadSlots(layout);
                                if (resultHeadSlots <= 0)
                                {
                                    info.func.emitError("public Result input currently requires carrier-compatible payload layout");
                                    signalPassFailure();
                                    return;
                                }
                                Value one = getConst(builder, caseDecodeLoc, u256Type, i64Type, 1, constCache, caseBody, "one");
                                Value tupleBaseOff = offc;
                                Value tagWord = head;
                                if (canonicalAbiLayoutIsDynamic(layout))
                                {
                                    Block *resultOffsetOkBlock = mainFunc.addBlock();
                                    Block *resultHeadOkBlock = mainFunc.addBlock();
                                    Value offsetOk = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, head, getNextDynamicOffset());
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        offsetOk,
                                        ValueRange{},
                                        ValueRange{},
                                        resultOffsetOkBlock,
                                        getAbiDecodeRevertBlock(lowering::AbiDecodeError::NonCanonicalEncoding));

                                    builder.setInsertionPointToEnd(resultOffsetOkBlock);
                                    tupleBaseOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, offc, head);
                                    Value cdsize = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                                    Value tupleHeadBytes = lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(resultHeadSlots) * 32ULL);
                                    Value tupleHeadEnd = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, tupleBaseOff, tupleHeadBytes);
                                    Value tupleHeadMissing = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, cdsize, tupleHeadEnd);
                                    Value tupleHeadPresent = builder.create<sir::IsZeroOp>(caseDecodeLoc, u256Type, tupleHeadMissing);
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        tupleHeadPresent,
                                        ValueRange{},
                                        ValueRange{},
                                        resultHeadOkBlock,
                                        getAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                                    builder.setInsertionPointToEnd(resultHeadOkBlock);
                                    tagWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, tupleBaseOff);
                                }
                                Value tag = error_union_helpers::maskedTagWordWithMask(builder, caseDecodeLoc, tagWord, one);
                                Value resultDynamicOffset = lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(resultHeadSlots) * 32ULL);
                                FailureOr<StrictDynamicCalldataValue> okCarrier = materializeResultCarrier(*layout.children[1], tupleBaseOff, 32, resultDynamicOffset);
                                if (failed(okCarrier))
                                {
                                    info.func.emitError("public Result input currently requires a carrier-compatible payload layout");
                                    signalPassFailure();
                                    return;
                                }
                                Value isError = error_union_helpers::extractedTagIsErrorWithMask(builder, caseDecodeLoc, tag, one);
                                Value zeroCarrier = getConst(builder, caseDecodeLoc, u256Type, i64Type, 0, constCache, caseBody, "zero");
                                wideTag = tag;
                                widePayload = builder.create<sir::SelectOp>(caseDecodeLoc, u256Type, isError, zeroCarrier, okCarrier->payload);
                                if (canonicalAbiLayoutIsDynamic(layout))
                                    nextDynamicOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, head, okCarrier->nextExpectedOffset);
                                appendWideResultInput = true;
                            }
                            else if (hasAbiLayout &&
                                     idx < info.resultInputModes.size() &&
                                     info.resultInputModes[idx] == "wide_single_error")
                            {
                                const AbiLayoutNode &layout = abiLayout;
                                if (!canonicalAbiLayoutIsTupleLike(layout))
                                {
                                    info.func.emitError("public wide Result input requires tuple ABI layout");
                                    signalPassFailure();
                                    return;
                                }
                                if (layout.children.size() != 3 || !isStaticBoolAbiNode(*layout.children[0]) ||
                                    canonicalAbiLayoutHeadSlots(*layout.children[0]) != 1)
                                {
                                    info.func.emitError("public Result input currently requires layout (bool,ok_payload,err_payload)");
                                    signalPassFailure();
                                    return;
                                }

                                if (loweredArgIndex + 1 >= info.inputTypes.size())
                                {
                                    info.func.emitError("wide Result input metadata does not match lowered argument types");
                                    signalPassFailure();
                                    return;
                                }

                                int64_t errFieldOffset = 32 + (canonicalAbiLayoutIsDynamic(*layout.children[1]) ? 32 : canonicalAbiLayoutHeadSlots(*layout.children[1]) * 32);
                                int64_t resultHeadSlots = canonicalAbiLayoutHeadSlots(layout);
                                if (resultHeadSlots <= 0 || errFieldOffset <= 0)
                                {
                                    info.func.emitError("public Result input currently requires carrier-compatible ok/error payload layouts");
                                    signalPassFailure();
                                    return;
                                }
                                Value one = getConst(builder, caseDecodeLoc, u256Type, i64Type, 1, constCache, caseBody, "one");
                                Value tupleBaseOff = offc;
                                Value tagWord = head;
                                if (canonicalAbiLayoutIsDynamic(layout))
                                {
                                    Block *resultOffsetOkBlock = mainFunc.addBlock();
                                    Block *resultHeadOkBlock = mainFunc.addBlock();
                                    Value offsetOk = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, head, getNextDynamicOffset());
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        offsetOk,
                                        ValueRange{},
                                        ValueRange{},
                                        resultOffsetOkBlock,
                                        getAbiDecodeRevertBlock(lowering::AbiDecodeError::NonCanonicalEncoding));

                                    builder.setInsertionPointToEnd(resultOffsetOkBlock);
                                    tupleBaseOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, offc, head);
                                    Value cdsize = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                                    Value tupleHeadBytes = lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(resultHeadSlots) * 32ULL);
                                    Value tupleHeadEnd = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, tupleBaseOff, tupleHeadBytes);
                                    Value tupleHeadMissing = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, cdsize, tupleHeadEnd);
                                    Value tupleHeadPresent = builder.create<sir::IsZeroOp>(caseDecodeLoc, u256Type, tupleHeadMissing);
                                    builder.create<sir::CondBrOp>(
                                        caseDecodeLoc,
                                        tupleHeadPresent,
                                        ValueRange{},
                                        ValueRange{},
                                        resultHeadOkBlock,
                                        getAbiDecodeRevertBlock(lowering::AbiDecodeError::TruncatedBuffer));

                                    builder.setInsertionPointToEnd(resultHeadOkBlock);
                                    tagWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, tupleBaseOff);
                                }
                                Value tag = error_union_helpers::maskedTagWordWithMask(builder, caseDecodeLoc, tagWord, one);
                                Value resultDynamicOffset = lowering::constU256(builder, caseDecodeLoc, static_cast<uint64_t>(resultHeadSlots) * 32ULL);
                                FailureOr<StrictDynamicCalldataValue> okPayload = materializeResultCarrier(*layout.children[1], tupleBaseOff, 32, resultDynamicOffset);
                                if (failed(okPayload))
                                {
                                    info.func.emitError("public Result input currently requires carrier-compatible ok/error payload layouts");
                                    signalPassFailure();
                                    return;
                                }
                                FailureOr<StrictDynamicCalldataValue> errPayload = materializeResultCarrier(
                                    *layout.children[2],
                                    tupleBaseOff,
                                    errFieldOffset,
                                    okPayload->nextExpectedOffset,
                                    canonicalAbiLayoutIsDynamic(*layout.children[2]) && !canonicalAbiLayoutIsTupleLike(*layout.children[2]));
                                if (failed(errPayload))
                                {
                                    info.func.emitError("public Result input currently requires carrier-compatible ok/error payload layouts");
                                    signalPassFailure();
                                    return;
                                }
                                Value isError = error_union_helpers::extractedTagIsErrorWithMask(builder, caseDecodeLoc, tag, one);
                                wideTag = tag;
                                widePayload = builder.create<sir::SelectOp>(caseDecodeLoc, u256Type, isError, errPayload->payload, okPayload->payload);
                                if (canonicalAbiLayoutIsDynamic(layout))
                                    nextDynamicOffset = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, head, errPayload->nextExpectedOffset);
                                appendWideResultInput = true;
                            }
                            else if (hasAbiLayout && isStaticFixedArrayLayout(abiLayout))
                            {
                                int64_t totalBytes = canonicalAbiLayoutHeadSlots(abiLayout) * 32;
                                Value totalVal = lowering::constU256(builder, caseDecodeLoc, totalBytes);
                                Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, totalVal);
                                builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, offc, totalVal);
                                argVal = buf;
                            }
                            else if (hasAbiLayout && canonicalAbiLayoutIsTupleLike(abiLayout))
                            {
                                if (canonicalAbiLayoutIsDynamic(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg =
                                        materializeResultCarrier(
                                            abiLayout,
                                            lowering::constU256(builder, caseDecodeLoc, 4),
                                            offs - 4,
                                            getNextDynamicOffset());
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic tuple ABI type for dispatcher");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else
                                {
                                    int64_t words = canonicalAbiLayoutHeadSlots(abiLayout);
                                    if (words <= 0)
                                    {
                                        module.emitError("unsupported tuple ABI type for dispatcher");
                                        signalPassFailure();
                                        return;
                                    }
                                    Value totalVal = lowering::constU256(builder, caseDecodeLoc, words * 32);
                                    Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, totalVal);
                                    builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, offc, totalVal);
                                    argVal = buf;
                                }
                            }
                            else if (hasAbiLayout && canonicalAbiLayoutIsDynamic(abiLayout))
                            {
                                if (abiLayout.kind == AbiLayoutKind::DynamicBytes)
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg = materializeStrictDynamicCalldataValue(head, getNextDynamicOffset(), StrictDynamicCalldataKind::BytesLike);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for dispatcher");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (isDynamicU256ArrayAbiNode(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg = materializeStrictDynamicCalldataValue(head, getNextDynamicOffset(), StrictDynamicCalldataKind::U256Array);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for dispatcher");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (isDynamicAddressArrayAbiNode(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg = materializeStrictDynamicCalldataValue(head, getNextDynamicOffset(), StrictDynamicCalldataKind::AddressArray);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for dispatcher");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (isDynamicBoolArrayAbiNode(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg = materializeStrictDynamicCalldataValue(head, getNextDynamicOffset(), StrictDynamicCalldataKind::BoolArray);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for dispatcher");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (isDynamicFixedBytesArrayAbiNode(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg = materializeStrictDynamicCalldataValue(head, getNextDynamicOffset(), StrictDynamicCalldataKind::FixedBytesArray, abiLayout.children.front()->width);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for dispatcher");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (dynamicTupleArrayElementHeadWords(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg =
                                        materializeDynamicTupleArrayCalldataValue(abiLayout, head, getNextDynamicOffset());
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for dispatcher");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else if (std::optional<uint64_t> elementWords = fullWordStaticArrayElementWords(abiLayout))
                                {
                                    FailureOr<StrictDynamicCalldataValue> strictArg =
                                        materializeStrictDynamicCalldataValue(head, getNextDynamicOffset(), StrictDynamicCalldataKind::U256Array, 0, *elementWords);
                                    if (failed(strictArg))
                                    {
                                        module.emitError("unsupported dynamic ABI type for dispatcher");
                                        signalPassFailure();
                                        return;
                                    }
                                    argVal = strictArg->payload;
                                    nextDynamicOffset = strictArg->nextExpectedOffset;
                                }
                                else
                                {
                                    module.emitError("unsupported dynamic ABI type for dispatcher");
                                    signalPassFailure();
                                    return;
                                }
                            }

                            if (appendWideResultInput)
                            {
                                args.push_back(wideTag);
                                args.push_back(widePayload);
                                loweredArgIndex += static_cast<unsigned>(adt_helpers::kAdtCarrierWordCount);
                                continue;
                            }

                            if (loweredArgIndex < info.inputTypes.size())
                            {
                                if (auto ptrTy = dyn_cast<sir::PtrType>(info.inputTypes[loweredArgIndex]))
                                {
                                    if (!isa<sir::PtrType>(argVal.getType()))
                                    {
                                        argVal = builder.create<sir::BitcastOp>(caseDecodeLoc, ptrType, argVal);
                                    }
                                    argVal = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, argVal);
                                }
                            }
                            // sir.icall requires all args to be !sir.u256.
                            if (isa<sir::PtrType>(argVal.getType()))
                                argVal = builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, argVal);
                            args.push_back(argVal);
                            loweredArgIndex += 1;
                        }

                        SmallVector<Type, 4> resultTypes;
                        for (unsigned r = 0; r < info.retCount; ++r)
                            resultTypes.push_back(u256Type);

                        auto call = builder.create<sir::ICallOp>(
                            caseLoc,
                            resultTypes,
                            SymbolRefAttr::get(ctx, info.func.getName()),
                            args);

                        AbiLayoutNode abiReturnLayout;
                        const bool hasReturnLayout = info.hasAbiReturn && !info.abiReturnLayout.empty();
                        if (hasReturnLayout && !parseAbiLayout(info.abiReturnLayout, abiReturnLayout, AbiLayoutSyntax::CanonicalAbi))
                        {
                            info.func.emitError("invalid ora.abi_return_layout");
                            signalPassFailure();
                            return;
                        }

                        if (info.returnsErrorUnion)
                        {
                            if (info.retCount != 2)
                            {
                                info.func.emitError("public error-union function must return dispatcher ptr/len pair");
                                signalPassFailure();
                                return;
                            }

                            Value ptr_u = call.getResult(0);
                            Value len = call.getResult(1);
                            (void)len;
                            Value ptr = builder.create<sir::BitcastOp>(caseErrorLoc, ptrType, ptr_u);
                            Value tag = builder.create<sir::LoadOp>(caseErrorLoc, u256Type, ptr);
                            Value c32_union = getConst(builder, caseErrorLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size");
                            Value payloadPtr = builder.create<sir::AddPtrOp>(caseErrorLoc, ptrType, ptr, c32_union);
                            Value payload = builder.create<sir::LoadOp>(caseErrorLoc, u256Type, payloadPtr);
                            Value one = getConst(builder, caseErrorLoc, u256Type, i64Type, 1, constCache, caseBody, "one");
                            Value maskedTag = error_union_helpers::maskedTagWordWithMask(builder, caseErrorLoc, tag, one);
                            Value isError = error_union_helpers::extractedTagIsErrorWithMask(builder, caseErrorLoc, maskedTag, one);
                            Value payloadScratchSize = getConst(builder, caseErrorLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size");
                            Value payloadScratchPtr = builder.create<sir::SAllocAnyOp>(caseErrorLoc, ptrType, payloadScratchSize);
                            builder.create<sir::StoreOp>(caseErrorLoc, payloadScratchPtr, payload);

                            auto loadPayloadFromScratch = [&](Block *block, Location loc) -> Value {
                                (void)block;
                                return builder.create<sir::LoadOp>(loc, u256Type, payloadScratchPtr);
                            };

                            Block *successBlock = mainFunc.addBlock();
                            Block *errorDispatchBlock = mainFunc.addBlock();
                            builder.create<sir::CondBrOp>(caseErrorLoc, isError, ValueRange{}, ValueRange{}, errorDispatchBlock, successBlock);

                            builder.setInsertionPointToEnd(successBlock);
                            Value successPayload = loadPayloadFromScratch(successBlock, caseReturnLoc);
                            Value retPtr = nullptr;
                            Value size = nullptr;
                            if (!info.hasAbiReturn)
                            {
                                size = getConst(builder, caseReturnLoc, u256Type, i64Type, 0, constCache, successBlock, "zero");
                                retPtr = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, size);
                            }
                            else if (hasReturnLayout && abiReturnLayout.kind == AbiLayoutKind::Static)
                            {
                                size = getConst(builder, caseReturnLoc, u256Type, i64Type, 32, constCache, successBlock, "word_size");
                                retPtr = builder.create<sir::SAllocAnyOp>(caseReturnLoc, ptrType, size);
                                setResultName(retPtr.getDefiningOp(), ("buf_" + info.func.getName()).str());
                                builder.create<sir::StoreOp>(caseReturnLoc, retPtr, materializeAbiReturnStaticWord(builder, caseReturnLoc, ctx, abiReturnLayout, successPayload));
                            }
                            else if (hasReturnLayout && !canonicalAbiLayoutIsDynamic(abiReturnLayout) && info.abiReturnWords > 0)
                            {
                                size = getConst(builder, caseReturnLoc, u256Type, i64Type, info.abiReturnWords * 32, constCache, successBlock);
                                retPtr = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, successPayload);
                            }
                            else if (hasReturnLayout)
                            {
                                retPtr = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, successPayload);
                                size = computeAbiEncodedSize(builder, caseReturnLoc, ctx, retPtr, abiReturnLayout);
                            }
                            else
                            {
                                info.func.emitError("public error-union dispatcher currently supports scalar, static tuple/struct, bytes/string, and static-base dynamic array ABI success payloads");
                                signalPassFailure();
                                return;
                            }
                            builder.create<sir::ReturnOp>(caseReturnLoc, retPtr, size);

                            builder.setInsertionPointToEnd(errorDispatchBlock);
                            Block *nextErrorBlock = revertError;
                            for (const ErrorInfo &errInfo : llvm::reverse(info.returnErrors))
                            {
                                Block *compareBlock = mainFunc.addBlock();
                                builder.setInsertionPointToEnd(compareBlock);
                                Value comparePayload = loadPayloadFromScratch(compareBlock, caseErrorLoc);

                                Value errId = getConst(builder, caseErrorLoc, u256Type, i64Type, static_cast<int64_t>(errInfo.id), constCache, compareBlock);
                                Value payloadAggPtr = builder.create<sir::BitcastOp>(caseErrorLoc, ptrType, comparePayload);
                                Value compareValue = builder.create<sir::LoadOp>(caseErrorLoc, u256Type, payloadAggPtr);
                                Value matches = builder.create<sir::EqOp>(caseErrorLoc, u256Type, compareValue, errId);
                                Block *emitBlock = mainFunc.addBlock();
                                builder.create<sir::CondBrOp>(caseErrorLoc, matches, ValueRange{}, ValueRange{}, emitBlock, nextErrorBlock);

                                builder.setInsertionPointToEnd(emitBlock);
                                Value emitPayload = loadPayloadFromScratch(emitBlock, caseErrorLoc);
                                if (errInfo.paramCount == 0)
                                {
                                    Value size4 = getConst(builder, caseErrorLoc, u256Type, i64Type, 4, constCache, emitBlock);
                                    Value revertPtr = builder.create<sir::SAllocAnyOp>(caseErrorLoc, ptrType, size4);
                                    builder.create<sir::StoreOp>(caseErrorLoc, revertPtr, getShiftedSelectorConst(builder, caseErrorLoc, ctx, errInfo.selector));
                                    builder.create<sir::RevertOp>(caseErrorLoc, revertPtr, size4);
                                }
                                else
                                {
                                    const int64_t totalBytes = 4 + static_cast<int64_t>(errInfo.paramCount) * 32;
                                    Value revertSize = getConst(builder, caseErrorLoc, u256Type, i64Type, totalBytes, constCache, emitBlock);
                                    Value revertPtr = builder.create<sir::SAllocAnyOp>(caseErrorLoc, ptrType, revertSize);
                                    builder.create<sir::StoreOp>(caseErrorLoc, revertPtr, getShiftedSelectorConst(builder, caseErrorLoc, ctx, errInfo.selector));

                                    Value payloadAggPtr = builder.create<sir::BitcastOp>(caseErrorLoc, ptrType, emitPayload);
                                    for (uint64_t index = 0; index < errInfo.paramCount; ++index)
                                    {
                                        Value srcOffset = getConst(builder, caseErrorLoc, u256Type, i64Type, static_cast<int64_t>((index + 1) * 32), constCache, emitBlock);
                                        Value srcPtr = builder.create<sir::AddPtrOp>(caseErrorLoc, ptrType, payloadAggPtr, srcOffset);
                                        Value fieldWord = builder.create<sir::LoadOp>(caseErrorLoc, u256Type, srcPtr);

                                        Value dstOffset = getConst(builder, caseErrorLoc, u256Type, i64Type, static_cast<int64_t>(4 + index * 32), constCache, emitBlock);
                                        Value dstPtr = builder.create<sir::AddPtrOp>(caseErrorLoc, ptrType, revertPtr, dstOffset);
                                        builder.create<sir::StoreOp>(caseErrorLoc, dstPtr, fieldWord);
                                    }

                                    builder.create<sir::RevertOp>(caseErrorLoc, revertPtr, revertSize);
                                }

                                nextErrorBlock = compareBlock;
                            }

                            builder.setInsertionPointToEnd(errorDispatchBlock);
                            builder.create<sir::BrOp>(caseErrorLoc, ValueRange{}, nextErrorBlock);
                        }
                        else if (info.retCount == 2)
                        {
                            Value ptr_u = call.getResult(0);
                            Value len = call.getResult(1);
                            if (hasReturnLayout &&
                                canonicalAbiLayoutIsTupleLike(abiReturnLayout) &&
                                canonicalAbiLayoutIsDynamic(abiReturnLayout))
                            {
                                (void)len;
                                FailureOr<AbiReturnBuffer> encoded = materializeSingleDynamicTupleAbiReturn(
                                    builder,
                                    caseReturnLoc,
                                    ctx,
                                    ptr_u,
                                    abiReturnLayout);
                                if (failed(encoded))
                                {
                                    info.func.emitError("unsupported dynamic tuple ABI return layout");
                                    signalPassFailure();
                                    return;
                                }
                                builder.create<sir::ReturnOp>(caseReturnLoc, encoded->ptr, encoded->size);
                            }
                            else if (hasReturnLayout &&
                                (abiReturnLayout.kind == AbiLayoutKind::DynamicBytes ||
                                 canonicalAbiLayoutSupportsDynamicArray(abiReturnLayout)))
                            {
                                (void)len;
                                AbiReturnBuffer encoded = materializeSingleDynamicAbiReturn(
                                    builder,
                                    caseReturnLoc,
                                    ctx,
                                    ptr_u,
                                    abiReturnLayout.kind == AbiLayoutKind::DynamicBytes);
                                builder.create<sir::ReturnOp>(caseReturnLoc, encoded.ptr, encoded.size);
                            }
                            else
                            {
                                Value ptr = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, ptr_u);
                                builder.create<sir::ReturnOp>(caseReturnLoc, ptr, len);
                            }
                        }
                        else if (info.retCount == 1)
                        {
                            Value val = call.getResult(0);
                            Value c32_ret = getConst(builder, caseReturnLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size");
                            Value size = c32_ret;
                            Value ptr = builder.create<sir::SAllocAnyOp>(caseReturnLoc, ptrType, size);
                            setResultName(ptr.getDefiningOp(), ("buf_" + info.func.getName()).str());
                            Value stored = hasReturnLayout ? materializeAbiReturnStaticWord(builder, caseReturnLoc, ctx, abiReturnLayout, val) : val;
                            builder.create<sir::StoreOp>(caseReturnLoc, ptr, stored);
                            builder.create<sir::ReturnOp>(caseReturnLoc, ptr, size);
                        }
                        else
                        {
                            Value z = lowering::constU256(builder, caseReturnLoc, 0);
                            Value pz = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, z);
                            builder.create<sir::ReturnOp>(caseReturnLoc, pz, z);
                        }

                        setBlockName(caseCheck, (info.func.getName() + "_").str());
                        setBlockOrder(caseCheck, 3 + static_cast<int64_t>(i) * 2);
                        if (caseBody != caseCheck)
                        {
                            setBlockName(caseBody, (info.func.getName() + "_exec").str());
                            setBlockOrder(caseBody, 3 + static_cast<int64_t>(i) * 2 + 1);
                        }
                    }

                    // Revert helper block (placed last).
                    builder.setInsertionPointToEnd(revertError);
                    Value c0_revert_main = getConst(builder, dispatcherMainLoc, u256Type, i64Type, 0, constCache, revertError, "zero");
                    Value p0b_main = builder.create<sir::BitcastOp>(dispatcherMainLoc, ptrType, c0_revert_main);
                    builder.create<sir::RevertOp>(dispatcherMainLoc, p0b_main, c0_revert_main);
                    setBlockName(revertError, "revert_error");
                    setBlockOrder(revertError, 2);
                    module.push_back(mainFunc);
                }
            };
        } // namespace

        std::unique_ptr<Pass> createSIRDispatcherPass()
        {
            return std::make_unique<SIRDispatcherPass>();
        }

    } // namespace ora
} // namespace mlir
