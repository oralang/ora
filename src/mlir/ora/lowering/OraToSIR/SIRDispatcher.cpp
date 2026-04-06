//===- SIRDispatcher.cpp - Build SIR Dispatcher ---------------------===//
//
// Creates a Solidity-style calldata dispatcher for public functions.
//
//===----------------------------------------------------------------------===//

#include "OraToSIR.h"

#include "SIR/SIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

using namespace mlir;

namespace mlir
{
    namespace ora
    {
        namespace
        {
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

            enum class AbiBase
            {
                Uint,
                Int,
                Bool,
                Address,
                BytesDyn,
                String,
                Unknown,
                Tuple
            };

            struct AbiType
            {
                AbiBase base = AbiBase::Unknown;
                SmallVector<int64_t, 2> dims; // -1 for dynamic []

                bool isArray() const { return !dims.empty(); }
                bool baseIsDynamic() const { return base == AbiBase::BytesDyn || base == AbiBase::String || base == AbiBase::Tuple; }
                bool isDynamic() const
                {
                    if (baseIsDynamic())
                        return true;
                    for (int64_t d : dims)
                        if (d < 0)
                            return true;
                    return false;
                }
                bool isStaticBase() const
                {
                    return base == AbiBase::Uint || base == AbiBase::Int || base == AbiBase::Bool || base == AbiBase::Address;
                }
                bool supportsStaticArray() const
                {
                    return isStaticBase() && isArray() && dims.size() == 1 && dims.front() >= 0;
                }
                bool supportsDynamicArray() const
                {
                    return isStaticBase() && isArray() && dims.size() == 1 && dims.front() < 0;
                }
                int64_t headSlots() const
                {
                    if (isDynamic())
                        return 1;
                    if (!isArray())
                        return 1;
                    if (!supportsStaticArray())
                        return -1;
                    return dims.front();
                }
            };

            static bool parseAbiType(StringRef s, AbiType &out)
            {
                out = AbiType();
                if (s.empty())
                    return false;
                if (s.front() == '(' || s == "tuple")
                {
                    out.base = AbiBase::Tuple;
                    return true;
                }

                size_t baseEnd = s.find('[');
                StringRef base = (baseEnd == StringRef::npos) ? s : s.take_front(baseEnd);
                if (base.starts_with("uint"))
                    out.base = AbiBase::Uint;
                else if (base.starts_with("int"))
                    out.base = AbiBase::Int;
                else if (base == "bool")
                    out.base = AbiBase::Bool;
                else if (base == "address")
                    out.base = AbiBase::Address;
                else if (base == "bytes")
                    out.base = AbiBase::BytesDyn;
                else if (base == "string")
                    out.base = AbiBase::String;
                else
                    out.base = AbiBase::Unknown;

                size_t pos = baseEnd;
                while (pos != StringRef::npos && pos < s.size() && s[pos] == '[')
                {
                    size_t close = s.find(']', pos);
                    if (close == StringRef::npos)
                        return false;
                    StringRef inner = s.slice(pos + 1, close);
                    if (inner.empty())
                    {
                        out.dims.push_back(-1);
                    }
                    else
                    {
                        int64_t val = 0;
                        if (inner.getAsInteger(10, val))
                            return false;
                        out.dims.push_back(val);
                    }
                    pos = close + 1;
                }
                return out.base != AbiBase::Unknown;
            }

            static uint64_t computeDebugNamedMemoryReserveBytes(ModuleOp module)
            {
                if (!module || !module->hasAttr("ora.debug_info"))
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

            struct AbiLayout
            {
                AbiType abi;
                std::vector<AbiLayout> fields;

                bool isTupleLike() const { return !fields.empty(); }
                bool isDynamic() const
                {
                    if (isTupleLike())
                    {
                        return llvm::any_of(fields, [](const AbiLayout &field) { return field.isDynamic(); });
                    }
                    return abi.isDynamic();
                }
                int64_t headSlots() const
                {
                    if (isTupleLike())
                    {
                        int64_t total = 0;
                        for (const AbiLayout &field : fields)
                        {
                            int64_t slots = field.isDynamic() ? 1 : field.headSlots();
                            if (slots < 0)
                                return -1;
                            total += slots;
                        }
                        return total;
                    }
                    return abi.headSlots();
                }
            };

            static bool parseAbiLayout(StringRef text, size_t &pos, AbiLayout &out)
            {
                while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
                    ++pos;
                if (pos >= text.size())
                    return false;

                if (text[pos] == '(')
                {
                    ++pos;
                    out = AbiLayout{};
                    out.abi.base = AbiBase::Tuple;
                    while (pos < text.size() && text[pos] != ')')
                    {
                        AbiLayout child;
                        if (!parseAbiLayout(text, pos, child))
                            return false;
                        out.fields.push_back(std::move(child));
                        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
                            ++pos;
                        if (pos < text.size() && text[pos] == ',')
                            ++pos;
                    }
                    if (pos >= text.size() || text[pos] != ')')
                        return false;
                    ++pos;
                }
                else
                {
                    size_t start = pos;
                    while (pos < text.size() && text[pos] != ',' && text[pos] != ')' && text[pos] != '[')
                        ++pos;
                    if (start == pos)
                        return false;
                    AbiType abi;
                    if (!parseAbiType(text.substr(start, pos - start), abi))
                        return false;
                    out = AbiLayout{};
                    out.abi = abi;
                }

                while (pos < text.size() && text[pos] == '[')
                {
                    size_t close = text.find(']', pos);
                    if (close == StringRef::npos)
                        return false;
                    StringRef inner = text.slice(pos + 1, close);
                    if (inner.empty())
                    {
                        out.abi.dims.push_back(-1);
                    }
                    else
                    {
                        int64_t val = 0;
                        if (inner.getAsInteger(10, val))
                            return false;
                        out.abi.dims.push_back(val);
                    }
                    pos = close + 1;
                }
                return true;
            }

            static bool parseAbiLayout(StringRef text, AbiLayout &out)
            {
                size_t pos = 0;
                if (!parseAbiLayout(text, pos, out))
                    return false;
                while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
                    ++pos;
                return pos == text.size();
            }

            static Value computePaddedBytes(OpBuilder &builder, Location loc, MLIRContext *ctx, Value len)
            {
                auto u256Type = sir::U256Type::get(ctx);
                Value c31 = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(IntegerType::get(ctx, 64, IntegerType::Unsigned), 31));
                Value c5 = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(IntegerType::get(ctx, 64, IntegerType::Unsigned), 5));
                Value lenPlus = builder.create<sir::AddOp>(loc, u256Type, len, c31);
                Value shifted = builder.create<sir::ShrOp>(loc, u256Type, c5, lenPlus);
                return builder.create<sir::ShlOp>(loc, u256Type, c5, shifted);
            }

            static Value computeAbiEncodedSize(
                OpBuilder &builder,
                Location loc,
                MLIRContext *ctx,
                Value basePtr,
                const AbiLayout &layout)
            {
                auto u256Type = sir::U256Type::get(ctx);
                auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
                auto i64Type = IntegerType::get(ctx, 64, IntegerType::Unsigned);
                Value c32 = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(i64Type, 32));

                if (!layout.isTupleLike())
                {
                    if (layout.abi.base == AbiBase::BytesDyn || layout.abi.base == AbiBase::String)
                    {
                        Value length = builder.create<sir::LoadOp>(loc, u256Type, basePtr);
                        Value padded = computePaddedBytes(builder, loc, ctx, length);
                        return builder.create<sir::AddOp>(loc, u256Type, padded, c32);
                    }
                    if (layout.abi.supportsDynamicArray())
                    {
                        Value length = builder.create<sir::LoadOp>(loc, u256Type, basePtr);
                        Value lenBytes = builder.create<sir::MulOp>(loc, u256Type, length, c32);
                        return builder.create<sir::AddOp>(loc, u256Type, lenBytes, c32);
                    }
                    return c32;
                }

                Value total = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(i64Type, layout.headSlots() * 32));
                int64_t headOffset = 0;
                for (const AbiLayout &field : layout.fields)
                {
                    if (field.isDynamic())
                    {
                        Value headOff = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(i64Type, headOffset));
                        Value offPtr = builder.create<sir::AddPtrOp>(loc, ptrType, basePtr, headOff);
                        Value relOff = builder.create<sir::LoadOp>(loc, u256Type, offPtr);
                        Value childPtr = builder.create<sir::AddPtrOp>(loc, ptrType, basePtr, relOff);
                        Value childSize = computeAbiEncodedSize(builder, loc, ctx, childPtr, field);
                        total = builder.create<sir::AddOp>(loc, u256Type, total, childSize);
                        headOffset += 32;
                    }
                    else
                    {
                        headOffset += field.headSlots() * 32;
                    }
                }
                return total;
            }

            struct PubFuncInfo
            {
                func::FuncOp func;
                Location provenanceLoc;
                uint32_t selector = 0;
                unsigned argCount = 0;
                unsigned retCount = 0;
                bool returnsErrorUnion = false;
                SmallVector<AbiType, 8> abiParams;
                SmallVector<std::string, 8> abiParamLayouts;
                SmallVector<std::string, 8> resultInputModes;
                SmallVector<int64_t, 8> resultInputErrorIds;
                AbiType abiReturn;
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

            struct ErrorInfo
            {
                uint64_t id = 0;
                uint32_t selector = 0;
                uint64_t paramCount = 0;
            };

            static Value getShiftedSelectorConst(OpBuilder &builder, Location loc, MLIRContext *ctx, uint32_t selector)
            {
                auto u256Type = sir::U256Type::get(ctx);
                auto u256IntType = IntegerType::get(ctx, 256, IntegerType::Unsigned);
                llvm::APInt selectorWord(256, selector);
                selectorWord = selectorWord.shl(224);
                return builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(u256IntType, selectorWord));
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
                    block->back().setAttr("sir.block_order", IntegerAttr::get(IntegerType::get(block->getParent()->getContext(), 64), order));
                }

                static Value getConst(OpBuilder &builder,
                                      Location loc,
                                      Type type,
                                      IntegerType i64Type,
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
                    module.walk([&](sir::ErrorDeclOp decl) {
                        auto idAttr = decl->getAttrOfType<IntegerAttr>("sir.error_id");
                        auto selectorAttr = decl->getAttrOfType<StringAttr>("sir.error_selector");
                        auto paramTypes = decl->getAttrOfType<ArrayAttr>("sir.param_types");
                        if (!idAttr || !selectorAttr || !paramTypes)
                            return;

                        StringRef selStr = selectorAttr.getValue();
                        if (!selStr.starts_with("0x") || selStr.size() != 10)
                            return;

                        uint32_t selector = 0;
                        for (char c : selStr.drop_front(2))
                        {
                            selector <<= 4;
                            if (c >= '0' && c <= '9')
                                selector |= (c - '0');
                            else if (c >= 'a' && c <= 'f')
                                selector |= (c - 'a' + 10);
                            else if (c >= 'A' && c <= 'F')
                                selector |= (c - 'A' + 10);
                            else
                                return;
                        }

                        abiErrors.push_back(ErrorInfo{
                            idAttr.getValue().getZExtValue(),
                            selector,
                            static_cast<uint64_t>(paramTypes.size()),
                        });
                    });

                    // Rewrite all non-entry functions: sir.return -> sir.iret (ptr as u256).
                    for (func::FuncOp func : module.getOps<func::FuncOp>())
                    {
                        if (func.getName() == "init" || func.getName() == "main")
                            continue;
                        if (userInit && func == userInit)
                            continue;

                        bool hasReturn = false;
                        for (Block &block : func.getBlocks())
                        {
                            if (!block.getTerminator())
                                continue;
                            if (auto ret = dyn_cast<sir::ReturnOp>(block.getTerminator()))
                            {
                                builder.setInsertionPoint(ret);
                                Value ptr = ret.getPtr();
                                Value len = ret.getLen();
                                Value ptr_u = builder.create<sir::BitcastOp>(loc, u256Type, ptr);
                                builder.create<sir::IRetOp>(loc, ValueRange{ptr_u, len});
                                ret.erase();
                                hasReturn = true;
                            }
                        }
                        if (hasReturn)
                        {
                            auto ft = func.getFunctionType();
                            SmallVector<Type, 4> results;
                            results.push_back(u256Type);
                            results.push_back(u256Type);
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
                                AbiType abi;
                                if (!parseAbiType(sattr.getValue(), abi))
                                {
                                    func.emitError("unsupported ABI param type: " + sattr.getValue());
                                    signalPassFailure();
                                    return;
                                }
                                info.abiParams.push_back(abi);
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

                        if (auto abiReturnAttr = func->getAttrOfType<StringAttr>("ora.abi_return"))
                        {
                            AbiType abiReturn;
                            if (!parseAbiType(abiReturnAttr.getValue(), abiReturn))
                            {
                                func.emitError("unsupported ABI return type: " + abiReturnAttr.getValue());
                                signalPassFailure();
                                return;
                            }
                            info.abiReturn = abiReturn;
                            info.hasAbiReturn = true;
                        }
                        if (auto abiReturnWordsAttr = func->getAttrOfType<IntegerAttr>("ora.abi_return_words"))
                            info.abiReturnWords = abiReturnWordsAttr.getInt();
                        if (auto abiReturnLayoutAttr = func->getAttrOfType<StringAttr>("ora.abi_return_layout"))
                            info.abiReturnLayout = abiReturnLayoutAttr.getValue().str();

                        if (auto returnsErrorUnionAttr = func->getAttrOfType<BoolAttr>("ora.returns_error_union"))
                            info.returnsErrorUnion = returnsErrorUnionAttr.getValue();
                        if (!info.abiParams.empty() && info.resultInputModes.size() != info.abiParams.size())
                        {
                            func.emitError("ora.result_input_modes length does not match ora.abi_params");
                            signalPassFailure();
                            return;
                        }
                        if (!info.abiParams.empty() && info.resultInputErrorIds.size() != info.abiParams.size())
                        {
                            func.emitError("ora.result_input_error_ids length does not match ora.abi_params");
                            signalPassFailure();
                            return;
                        }

                        auto loweredArgCountForSourceParam = [&](size_t idx) -> unsigned {
                            if (idx < info.resultInputModes.size() &&
                                (info.resultInputModes[idx] == "wide_payloadless" || info.resultInputModes[idx] == "wide_single_error"))
                                return 2;
                            return 1;
                        };

                        unsigned expectedArgCount = info.abiParams.empty() ? info.argCount : 0;
                        if (!info.abiParams.empty())
                        {
                            for (size_t i = 0; i < info.abiParams.size(); ++i)
                                expectedArgCount += loweredArgCountForSourceParam(i);
                        }

                        if (!info.abiParams.empty() && expectedArgCount != info.argCount)
                        {
                            func.emitError("public ABI param metadata does not match lowered function argument count");
                            signalPassFailure();
                            return;
                        }

                        int64_t headSlots = 0;
                        size_t sourceParamCount = info.abiParams.empty() ? info.argCount : info.abiParams.size();
                        for (size_t i = 0; i < sourceParamCount; ++i)
                        {
                            AbiType abi = info.abiParams.empty() ? AbiType{} : info.abiParams[i];
                            if (info.abiParams.empty())
                            {
                                headSlots += 1;
                            }
                            else
                            {
                                int64_t slots = abi.headSlots();
                                if (abi.base == AbiBase::Tuple && i < info.abiParamLayouts.size())
                                {
                                    AbiLayout layout;
                                    if (!parseAbiLayout(info.abiParamLayouts[i], layout))
                                    {
                                        func.emitError("invalid tuple ABI param layout");
                                        signalPassFailure();
                                        return;
                                    }
                                    slots = layout.isDynamic() ? 1 : layout.headSlots();
                                }
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

                            if (isa<sir::PtrType>(oldRes.getType()) && isa<sir::U256Type>(newRes.getType()))
                            {
                                auto bc = callBuilder.create<sir::BitcastOp>(icall.getLoc(),
                                                                             oldRes.getType(),
                                                                             newRes);
                                oldRes.replaceAllUsesWith(bc.getResult());
                                continue;
                            }
                            if (isa<sir::U256Type>(oldRes.getType()) && isa<sir::PtrType>(newRes.getType()))
                            {
                                auto bc = callBuilder.create<sir::BitcastOp>(icall.getLoc(),
                                                                             oldRes.getType(),
                                                                             newRes);
                                oldRes.replaceAllUsesWith(bc.getResult());
                                continue;
                            }

                            oldRes.replaceAllUsesWith(newRes);
                        }

                        // If old call had fewer results, the prefix mapping above is sufficient.
                        if (icall.getNumResults() > newCall.getNumResults())
                        {
                            auto u256 = sir::U256Type::get(module.getContext());
                            auto zero = callBuilder.create<sir::ConstOp>(icall.getLoc(), u256,
                                                                         IntegerAttr::get(callBuilder.getI64Type(), 0));
                            for (unsigned i = common; i < icall.getNumResults(); ++i)
                            {
                                Value oldRes = icall.getResult(i);
                                if (isa<sir::PtrType>(oldRes.getType()))
                                {
                                    auto bc = callBuilder.create<sir::BitcastOp>(icall.getLoc(),
                                                                                 oldRes.getType(),
                                                                                 zero);
                                    oldRes.replaceAllUsesWith(bc.getResult());
                                }
                                else
                                {
                                    oldRes.replaceAllUsesWith(zero);
                                }
                            }
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

                    auto getInitRevert = [&]() -> Block * {
                        if (!initRevert)
                            initRevert = initFunc.addBlock();
                        return initRevert;
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
                                auto newType = FunctionType::get(
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

                        SmallVector<AbiType, 8> initAbiParams;
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
                                AbiType abi;
                                if (!parseAbiType(sattr.getValue(), abi))
                                {
                                    userInit.emitError("unsupported ABI param type: " + sattr.getValue());
                                    signalPassFailure();
                                    return;
                                }
                                initAbiParams.push_back(abi);
                            }
                        }

                        unsigned argCount = userInitType.getNumInputs();
                        if (!initAbiParams.empty() && initAbiParams.size() != argCount)
                        {
                            userInit.emitError("ora.abi_params length does not match function argument count");
                            signalPassFailure();
                            return;
                        }

                        int64_t headSlots = 0;
                        for (unsigned i = 0; i < argCount; ++i)
                        {
                            if (initAbiParams.empty())
                            {
                                headSlots += 1;
                            }
                            else
                            {
                                int64_t slots = initAbiParams[i].headSlots();
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
                            Value minSizeVal = builder.create<sir::ConstOp>(initLoc, u256Type, IntegerAttr::get(i64Type, minHeadBytes));
                            Value dataTooShort = builder.create<sir::LtOp>(initLoc, u256Type, dataLen, minSizeVal);
                            Value anyTooShort = builder.create<sir::OrOp>(initLoc, u256Type, codeTooShort, dataTooShort);
                            Value valid_args = builder.create<sir::IsZeroOp>(initLoc, u256Type, anyTooShort);
                            initDecode = initFunc.addBlock();
                            builder.create<sir::CondBrOp>(initLoc, valid_args, ValueRange{}, ValueRange{}, initDecode, getInitRevert());
                            builder.setInsertionPointToEnd(initDecode);
                        }
                        else
                        {
                            Value valid_args = builder.create<sir::IsZeroOp>(initLoc, u256Type, codeTooShort);
                            initDecode = initFunc.addBlock();
                            builder.create<sir::CondBrOp>(initLoc, valid_args, ValueRange{}, ValueRange{}, initDecode, getInitRevert());
                            builder.setInsertionPointToEnd(initDecode);
                        }

                        Value dataBuf = builder.create<sir::MallocOp>(initLoc, ptrType, dataLen);
                        builder.create<sir::CodeCopyOp>(initLoc, dataBuf, initEnd, dataLen);

                        SmallVector<int64_t, 8> headOffsets;
                        int64_t headSlot = 0;
                        for (unsigned i = 0; i < argCount; ++i)
                        {
                            headOffsets.push_back(32 * headSlot);
                            int64_t slots = initAbiParams.empty() ? 1 : initAbiParams[i].headSlots();
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
                        for (unsigned idx = 0; idx < argCount; ++idx)
                        {
                            int64_t offs = headOffsets[idx];
                            Value offc = builder.create<sir::ConstOp>(initLoc, u256Type, IntegerAttr::get(i64Type, offs));
                            Value headPtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, dataBuf, offc);
                            Value head = builder.create<sir::LoadOp>(initLoc, u256Type, headPtr);
                            AbiType abi = initAbiParams.empty() ? AbiType{} : initAbiParams[idx];
                            Value argVal = head;

                            if (!initAbiParams.empty() && abi.isArray() && abi.supportsStaticArray())
                            {
                                int64_t elemCount = abi.dims.front();
                                int64_t totalBytes = elemCount * 32;
                                Value totalVal = builder.create<sir::ConstOp>(initLoc, u256Type, IntegerAttr::get(i64Type, totalBytes));
                                Value buf = builder.create<sir::SAllocAnyOp>(initLoc, ptrType, totalVal);
                                Value src = builder.create<sir::AddPtrOp>(initLoc, ptrType, dataBuf, offc);
                                builder.create<sir::MCopyOp>(initLoc, buf, src, totalVal);
                                argVal = buf;
                            }
                            else if (!initAbiParams.empty() && abi.isDynamic())
                            {
                                if (abi.base == AbiBase::BytesDyn || abi.base == AbiBase::String || abi.supportsDynamicArray())
                                {
                                    Value c32_dyn = getConst(builder, initLoc, u256Type, i64Type, 32, constCache, builder.getInsertionBlock(), "word_size");
                                    Value absOff = head;
                                    Value absPtr = builder.create<sir::AddPtrOp>(initLoc, ptrType, dataBuf, absOff);
                                    Value len = builder.create<sir::LoadOp>(initLoc, u256Type, absPtr);

                                    Value total = nullptr;
                                    if (abi.baseIsDynamic())
                                    {
                                        Value c31 = getConst(builder, initLoc, u256Type, i64Type, 31, constCache, builder.getInsertionBlock(), "pad_31");
                                        Value c5 = getConst(builder, initLoc, u256Type, i64Type, 5, constCache, builder.getInsertionBlock(), "shift_5");
                                        Value lenPlus = builder.create<sir::AddOp>(initLoc, u256Type, len, c31);
                                        Value shifted = builder.create<sir::ShrOp>(initLoc, u256Type, c5, lenPlus);
                                        Value padded = builder.create<sir::ShlOp>(initLoc, u256Type, c5, shifted);
                                        total = builder.create<sir::AddOp>(initLoc, u256Type, padded, c32_dyn);
                                    }
                                    else
                                    {
                                        Value lenBytes = builder.create<sir::MulOp>(initLoc, u256Type, len, c32_dyn);
                                        total = builder.create<sir::AddOp>(initLoc, u256Type, lenBytes, c32_dyn);
                                    }

                                    Value end = builder.create<sir::AddOp>(initLoc, u256Type, absOff, total);
                                    Value tooShortDyn = builder.create<sir::LtOp>(initLoc, u256Type, dataLen, end);
                                    Value valid_dyn = builder.create<sir::IsZeroOp>(initLoc, u256Type, tooShortDyn);
                                    Block *dynBody = initFunc.addBlock();
                                    builder.create<sir::CondBrOp>(initLoc, valid_dyn, ValueRange{}, ValueRange{}, dynBody, getInitRevert());
                                    builder.setInsertionPointToEnd(dynBody);

                                    Value buf = builder.create<sir::SAllocAnyOp>(initLoc, ptrType, total);
                                    Value src = builder.create<sir::AddPtrOp>(initLoc, ptrType, dataBuf, absOff);
                                    builder.create<sir::MCopyOp>(initLoc, buf, src, total);
                                    argVal = buf;
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

                    // Initialize runtime malloc free-pointer slot (memory[0x20]).
                    // Without this, malloc may start at address 0 and clobber scratch state.
                    Value freePtrSlot = builder.create<sir::BitcastOp>(dispatcherMainLoc, ptrType, c32_entry);
                    Value heapBase = builder.create<sir::CodeSizeOp>(dispatcherMainLoc, u256Type);
                    Value initialFreePtr = heapBase;
                    if (uint64_t debugNamedMemoryBytes = computeDebugNamedMemoryReserveBytes(module))
                    {
                        Value reservedBytes = builder.create<sir::ConstOp>(
                            dispatcherMainLoc,
                            u256Type,
                            IntegerAttr::get(i64Type, debugNamedMemoryBytes));
                        initialFreePtr = builder.create<sir::AddOp>(dispatcherMainLoc, u256Type, heapBase, reservedBytes);
                    }
                    builder.create<sir::StoreOp>(dispatcherMainLoc, freePtrSlot, initialFreePtr);

                    Value cv = builder.create<sir::CallValueOp>(dispatcherMainLoc, u256Type);
                    setResultName(cv.getDefiningOp(), "cv");
                    Value cv_zero = builder.create<sir::IsZeroOp>(dispatcherMainLoc, u256Type, cv);
                    setResultName(cv_zero.getDefiningOp(), "cv_nonzero");
                    builder.create<sir::CondBrOp>(dispatcherMainLoc, cv_zero, ValueRange{}, ValueRange{}, loadSelector, revertError);
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
                        size_t sourceParamCount = info.abiParams.empty() ? info.argCount : info.abiParams.size();
                        for (size_t i = 0; i < sourceParamCount; ++i)
                        {
                            headOffsets.push_back(4 + 32 * headSlot);
                            AbiType abi = info.abiParams.empty() ? AbiType{} : info.abiParams[i];
                            int64_t slots = info.abiParams.empty() ? 1 : abi.headSlots();
                            if (slots < 0)
                            {
                                module.emitError("unsupported ABI type for head offset sizing");
                                signalPassFailure();
                                return;
                            }
                            headSlot += slots;
                        }

                        unsigned loweredArgIndex = 0;
                        for (unsigned idx = 0; idx < sourceParamCount; ++idx)
                        {
                            int64_t offs = headOffsets[idx];
                            StringRef offName = offs == 4  ? StringRef("selector_offset")
                                              : offs == 36 ? StringRef("arg1_offset")
                                              : offs == 68 ? StringRef("arg2_offset")
                                                           : StringRef();
                            Value offc = offName.empty()
                                             ? builder.create<sir::ConstOp>(caseDecodeLoc, u256Type, IntegerAttr::get(i64Type, offs))
                                             : getConst(builder, caseDecodeLoc, u256Type, i64Type, offs, constCache, caseBody, offName);
                            Value head = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, offc);
                            StringRef argPrefix = idx == 0 ? "a_" : (idx == 1 ? "b_" : (idx == 2 ? "n_" : "arg_"));
                            setResultName(head.getDefiningOp(), (argPrefix + info.func.getName()).str());

                            AbiType abi = info.abiParams.empty() ? AbiType{} : info.abiParams[idx];
                            Value argVal = head;
                            bool appendWideResultInput = false;
                            Value wideTag;
                            Value widePayload;
                            auto resultInputCarrierExpectsPtr = [&]() -> bool {
                                return loweredArgIndex + 1 < info.inputTypes.size() &&
                                       isa<sir::PtrType>(info.inputTypes[loweredArgIndex + 1]);
                            };
                            auto materializeDynamicAbiValue = [&](const AbiLayout &fieldLayout, Value absOff, bool expectsPtr) -> FailureOr<Value> {
                                if (fieldLayout.isTupleLike())
                                    return failure();
                                if (!expectsPtr)
                                    return failure();
                                if (!(fieldLayout.abi.base == AbiBase::BytesDyn || fieldLayout.abi.base == AbiBase::String || fieldLayout.abi.supportsDynamicArray()))
                                    return failure();

                                Value len = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, absOff);
                                Value total = nullptr;
                                if (fieldLayout.abi.baseIsDynamic())
                                {
                                    Value c31 = getConst(builder, caseDecodeLoc, u256Type, i64Type, 31, constCache, caseBody, "pad_31");
                                    Value c5 = getConst(builder, caseDecodeLoc, u256Type, i64Type, 5, constCache, caseBody, "shift_5");
                                    Value lenPlus = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, len, c31);
                                    Value shifted = builder.create<sir::ShrOp>(caseDecodeLoc, u256Type, c5, lenPlus);
                                    Value padded = builder.create<sir::ShlOp>(caseDecodeLoc, u256Type, c5, shifted);
                                    total = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, padded, getConst(builder, caseDecodeLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size"));
                                }
                                else
                                {
                                    Value lenBytes = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, len, getConst(builder, caseDecodeLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size"));
                                    total = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, lenBytes, getConst(builder, caseDecodeLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size"));
                                }
                                Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, total);
                                builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, absOff, total);
                                return builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, buf).getResult();
                            };
                            auto materializeResultCarrier = [&](const AbiLayout &fieldLayout, Value tupleBaseOff, int64_t fieldHeadByteOffset) -> FailureOr<Value> {
                                bool expectsPtr = resultInputCarrierExpectsPtr();
                                Value fieldHeadOff = builder.create<sir::AddOp>(
                                    caseDecodeLoc,
                                    u256Type,
                                    tupleBaseOff,
                                    getConst(builder, caseDecodeLoc, u256Type, i64Type, fieldHeadByteOffset, constCache, caseBody)
                                );

                                if (fieldLayout.isDynamic())
                                {
                                    Value relOff = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, fieldHeadOff);
                                    Value absOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, tupleBaseOff, relOff);
                                    return materializeDynamicAbiValue(fieldLayout, absOff, expectsPtr);
                                }

                                int64_t words = fieldLayout.headSlots();
                                if (words <= 0)
                                    return failure();
                                if (!expectsPtr && words != 1)
                                    return failure();
                                if (!expectsPtr)
                                    return builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, fieldHeadOff).getResult();

                                Value totalVal = builder.create<sir::ConstOp>(caseDecodeLoc, u256Type, IntegerAttr::get(i64Type, words * 32));
                                Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, totalVal);
                                builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, fieldHeadOff, totalVal);
                                return builder.create<sir::BitcastOp>(caseDecodeLoc, u256Type, buf).getResult();
                            };

                            if (!info.abiParams.empty() &&
                                idx < info.resultInputModes.size() &&
                                info.resultInputModes[idx] == "narrow_payloadless")
                            {
                                if (abi.base != AbiBase::Tuple || idx >= info.abiParamLayouts.size())
                                {
                                    info.func.emitError("public Result input requires tuple ABI layout");
                                    signalPassFailure();
                                    return;
                                }
                                AbiLayout layout;
                                if (!parseAbiLayout(info.abiParamLayouts[idx], layout))
                                {
                                    info.func.emitError("invalid Result input ABI layout");
                                    signalPassFailure();
                                    return;
                                }
                                if (layout.isDynamic() || layout.fields.size() != 2 || layout.fields[0].abi.base != AbiBase::Bool ||
                                    layout.fields[0].headSlots() != 1 || layout.fields[1].isDynamic() || layout.fields[1].headSlots() != 1)
                                {
                                    info.func.emitError("public Result input currently requires static layout (bool,payload)");
                                    signalPassFailure();
                                    return;
                                }

                                Value one = getConst(builder, caseDecodeLoc, u256Type, i64Type, 1, constCache, caseBody, "one");
                                Value tag = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, head, one);
                                Value payloadOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, offc, getConst(builder, caseDecodeLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size"));
                                Value payload = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, payloadOff);
                                Value packedOk = builder.create<sir::ShlOp>(caseDecodeLoc, u256Type, one, payload);
                                Value errId = getConst(builder, caseDecodeLoc, u256Type, i64Type, info.resultInputErrorIds[idx], constCache, caseBody);
                                Value packedErrPayload = builder.create<sir::ShlOp>(caseDecodeLoc, u256Type, one, errId);
                                Value packedErr = builder.create<sir::OrOp>(caseDecodeLoc, u256Type, packedErrPayload, one);
                                Value isError = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, tag, one);
                                argVal = builder.create<sir::SelectOp>(caseDecodeLoc, u256Type, isError, packedErr, packedOk);
                            }
                            else if (!info.abiParams.empty() &&
                                     idx < info.resultInputModes.size() &&
                                     info.resultInputModes[idx] == "wide_payloadless")
                            {
                                if (abi.base != AbiBase::Tuple || idx >= info.abiParamLayouts.size())
                                {
                                    info.func.emitError("public wide Result input requires tuple ABI layout");
                                    signalPassFailure();
                                    return;
                                }
                                AbiLayout layout;
                                if (!parseAbiLayout(info.abiParamLayouts[idx], layout))
                                {
                                    info.func.emitError("invalid wide Result input ABI layout");
                                    signalPassFailure();
                                    return;
                                }
                                if (layout.fields.size() != 2 || layout.fields[0].abi.base != AbiBase::Bool ||
                                    layout.fields[0].headSlots() != 1)
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

                                Value one = getConst(builder, caseDecodeLoc, u256Type, i64Type, 1, constCache, caseBody, "one");
                                Value tupleBaseOff = offc;
                                Value tagWord = head;
                                if (layout.isDynamic())
                                {
                                    tupleBaseOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, offc, head);
                                    tagWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, tupleBaseOff);
                                }
                                Value tag = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, tagWord, one);
                                FailureOr<Value> okCarrier = materializeResultCarrier(layout.fields[1], tupleBaseOff, 32);
                                if (failed(okCarrier))
                                {
                                    info.func.emitError("public Result input currently requires a carrier-compatible payload layout");
                                    signalPassFailure();
                                    return;
                                }
                                Value isError = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, tag, one);
                                Value zeroCarrier = getConst(builder, caseDecodeLoc, u256Type, i64Type, 0, constCache, caseBody, "zero");
                                wideTag = tag;
                                widePayload = builder.create<sir::SelectOp>(caseDecodeLoc, u256Type, isError, zeroCarrier, *okCarrier);
                                appendWideResultInput = true;
                            }
                            else if (!info.abiParams.empty() &&
                                     idx < info.resultInputModes.size() &&
                                     info.resultInputModes[idx] == "wide_single_error")
                            {
                                if (abi.base != AbiBase::Tuple || idx >= info.abiParamLayouts.size())
                                {
                                    info.func.emitError("public wide Result input requires tuple ABI layout");
                                    signalPassFailure();
                                    return;
                                }
                                AbiLayout layout;
                                if (!parseAbiLayout(info.abiParamLayouts[idx], layout))
                                {
                                    info.func.emitError("invalid wide Result input ABI layout");
                                    signalPassFailure();
                                    return;
                                }
                                if (layout.fields.size() != 3 || layout.fields[0].abi.base != AbiBase::Bool ||
                                    layout.fields[0].headSlots() != 1)
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

                                Value one = getConst(builder, caseDecodeLoc, u256Type, i64Type, 1, constCache, caseBody, "one");
                                Value tupleBaseOff = offc;
                                Value tagWord = head;
                                if (layout.isDynamic())
                                {
                                    tupleBaseOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, offc, head);
                                    tagWord = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, tupleBaseOff);
                                }
                                Value tag = builder.create<sir::AndOp>(caseDecodeLoc, u256Type, tagWord, one);
                                int64_t errFieldOffset = 32 + (layout.fields[1].isDynamic() ? 32 : layout.fields[1].headSlots() * 32);
                                FailureOr<Value> okPayload = materializeResultCarrier(layout.fields[1], tupleBaseOff, 32);
                                FailureOr<Value> errPayload = materializeResultCarrier(layout.fields[2], tupleBaseOff, errFieldOffset);
                                if (failed(okPayload) || failed(errPayload))
                                {
                                    info.func.emitError("public Result input currently requires carrier-compatible ok/error payload layouts");
                                    signalPassFailure();
                                    return;
                                }
                                Value isError = builder.create<sir::EqOp>(caseDecodeLoc, u256Type, tag, one);
                                wideTag = tag;
                                widePayload = builder.create<sir::SelectOp>(caseDecodeLoc, u256Type, isError, *errPayload, *okPayload);
                                appendWideResultInput = true;
                            }
                            else if (!info.abiParams.empty() && abi.isArray() && abi.supportsStaticArray())
                            {
                                int64_t elemCount = abi.dims.front();
                                int64_t totalBytes = elemCount * 32;
                                Value totalVal = builder.create<sir::ConstOp>(caseDecodeLoc, u256Type, IntegerAttr::get(i64Type, totalBytes));
                                Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, totalVal);
                                builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, offc, totalVal);
                                argVal = buf;
                            }
                            else if (!info.abiParams.empty() && abi.base == AbiBase::Tuple && idx < info.abiParamLayouts.size())
                            {
                                AbiLayout layout;
                                if (!parseAbiLayout(info.abiParamLayouts[idx], layout))
                                {
                                    info.func.emitError("invalid tuple ABI param layout");
                                    signalPassFailure();
                                    return;
                                }
                                if (layout.isDynamic())
                                {
                                    module.emitError("unsupported dynamic ABI type for dispatcher");
                                    signalPassFailure();
                                    return;
                                }
                                int64_t words = layout.headSlots();
                                if (words <= 0)
                                {
                                    module.emitError("unsupported tuple ABI type for dispatcher");
                                    signalPassFailure();
                                    return;
                                }
                                Value totalVal = builder.create<sir::ConstOp>(caseDecodeLoc, u256Type, IntegerAttr::get(i64Type, words * 32));
                                Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, totalVal);
                                builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, offc, totalVal);
                                argVal = buf;
                            }
                            else if (!info.abiParams.empty() && abi.isDynamic())
                            {
                                if (abi.base == AbiBase::BytesDyn || abi.base == AbiBase::String || abi.supportsDynamicArray())
                                {
                                    Value c4_dyn = getConst(builder, caseDecodeLoc, u256Type, i64Type, 4, constCache, caseBody, "selector_offset");
                                    Value c32_dyn = getConst(builder, caseDecodeLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size");
                                    Value absOff = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, head, c4_dyn);
                                    Value len = builder.create<sir::CallDataLoadOp>(caseDecodeLoc, u256Type, absOff);

                                    Value total = nullptr;
                                    if (abi.baseIsDynamic())
                                    {
                                        Value c31 = getConst(builder, caseDecodeLoc, u256Type, i64Type, 31, constCache, caseBody, "pad_31");
                                        Value c5 = getConst(builder, caseDecodeLoc, u256Type, i64Type, 5, constCache, caseBody, "shift_5");
                                        Value lenPlus = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, len, c31);
                                        Value shifted = builder.create<sir::ShrOp>(caseDecodeLoc, u256Type, c5, lenPlus);
                                        Value padded = builder.create<sir::ShlOp>(caseDecodeLoc, u256Type, c5, shifted);
                                        total = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, padded, c32_dyn);
                                    }
                                    else
                                    {
                                        Value lenBytes = builder.create<sir::MulOp>(caseDecodeLoc, u256Type, len, c32_dyn);
                                        total = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, lenBytes, c32_dyn);
                                    }

                                    Value cdsize_case = builder.create<sir::CallDataSizeOp>(caseDecodeLoc, u256Type);
                                    Value end = builder.create<sir::AddOp>(caseDecodeLoc, u256Type, absOff, total);
                                    // Check cdsize >= end (i.e., !(cdsize < end))
                                    Value tooShort = builder.create<sir::LtOp>(caseDecodeLoc, u256Type, cdsize_case, end);
                                    Value valid_dyn = builder.create<sir::IsZeroOp>(caseDecodeLoc, u256Type, tooShort);
                                    Block *dynBody = mainFunc.addBlock();
                                    builder.create<sir::CondBrOp>(caseDecodeLoc, valid_dyn, ValueRange{}, ValueRange{}, dynBody, revertError);
                                    builder.setInsertionPointToEnd(dynBody);

                                    Value buf = builder.create<sir::SAllocAnyOp>(caseDecodeLoc, ptrType, total);
                                    builder.create<sir::CallDataCopyOp>(caseDecodeLoc, buf, absOff, total);
                                    argVal = buf;
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
                                loweredArgIndex += 2;
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

                        if (info.returnsErrorUnion)
                        {
                            if (info.retCount != 2)
                            {
                                info.func.emitError("public error-union function must return dispatcher ptr/len pair");
                                signalPassFailure();
                                return;
                            }
                            if (!info.hasAbiReturn)
                            {
                                info.func.emitError("public error-union dispatcher missing ABI return metadata");
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
                            Value maskedTag = builder.create<sir::AndOp>(caseErrorLoc, u256Type, tag, one);
                            Value isError = builder.create<sir::EqOp>(caseErrorLoc, u256Type, maskedTag, one);

                            Block *successBlock = mainFunc.addBlock();
                            Block *errorDispatchBlock = mainFunc.addBlock();
                            builder.create<sir::CondBrOp>(caseErrorLoc, isError, ValueRange{}, ValueRange{}, errorDispatchBlock, successBlock);

                            builder.setInsertionPointToEnd(successBlock);
                            Value retPtr = nullptr;
                            Value size = nullptr;
                            if (info.abiReturn.isStaticBase() && !info.abiReturn.isArray() && !info.abiReturn.baseIsDynamic())
                            {
                                size = getConst(builder, caseReturnLoc, u256Type, i64Type, 32, constCache, successBlock, "word_size");
                                retPtr = builder.create<sir::SAllocAnyOp>(caseReturnLoc, ptrType, size);
                                setResultName(retPtr.getDefiningOp(), ("buf_" + info.func.getName()).str());
                                builder.create<sir::StoreOp>(caseReturnLoc, retPtr, payload);
                            }
                            else if (info.abiReturn.base == AbiBase::Tuple && info.abiReturnWords > 0)
                            {
                                size = getConst(builder, caseReturnLoc, u256Type, i64Type, info.abiReturnWords * 32, constCache, successBlock);
                                retPtr = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, payload);
                            }
                            else if (!info.abiReturnLayout.empty())
                            {
                                AbiLayout layout;
                                if (!parseAbiLayout(info.abiReturnLayout, layout))
                                {
                                    info.func.emitError("invalid ora.abi_return_layout");
                                    signalPassFailure();
                                    return;
                                }
                                retPtr = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, payload);
                                size = computeAbiEncodedSize(builder, caseReturnLoc, ctx, retPtr, layout);
                            }
                            else if (info.abiReturn.base == AbiBase::BytesDyn || info.abiReturn.base == AbiBase::String)
                            {
                                retPtr = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, payload);
                                Value length = builder.create<sir::LoadOp>(caseReturnLoc, u256Type, retPtr);
                                Value padded = computePaddedBytes(builder, caseReturnLoc, ctx, length);
                                Value wordSize = getConst(builder, caseReturnLoc, u256Type, i64Type, 32, constCache, successBlock, "word_size");
                                size = builder.create<sir::AddOp>(caseReturnLoc, u256Type, padded, wordSize);
                            }
                            else if (info.abiReturn.supportsDynamicArray())
                            {
                                retPtr = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, payload);
                                Value length = builder.create<sir::LoadOp>(caseReturnLoc, u256Type, retPtr);
                                Value wordSize = getConst(builder, caseReturnLoc, u256Type, i64Type, 32, constCache, successBlock, "word_size");
                                Value lenBytes = builder.create<sir::MulOp>(caseReturnLoc, u256Type, length, wordSize);
                                size = builder.create<sir::AddOp>(caseReturnLoc, u256Type, lenBytes, wordSize);
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
                            for (const ErrorInfo &errInfo : llvm::reverse(abiErrors))
                            {
                                Block *compareBlock = mainFunc.addBlock();
                                builder.setInsertionPointToEnd(compareBlock);

                                Value errId = getConst(builder, caseErrorLoc, u256Type, i64Type, static_cast<int64_t>(errInfo.id), constCache, compareBlock);
                                Value compareValue = payload;
                                if (errInfo.paramCount != 0)
                                {
                                    Value payloadAggPtr = builder.create<sir::BitcastOp>(caseErrorLoc, ptrType, payload);
                                    compareValue = builder.create<sir::LoadOp>(caseErrorLoc, u256Type, payloadAggPtr);
                                }
                                Value matches = builder.create<sir::EqOp>(caseErrorLoc, u256Type, compareValue, errId);
                                Block *emitBlock = mainFunc.addBlock();
                                builder.create<sir::CondBrOp>(caseErrorLoc, matches, ValueRange{}, ValueRange{}, emitBlock, nextErrorBlock);

                                builder.setInsertionPointToEnd(emitBlock);
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

                                    Value payloadAggPtr = builder.create<sir::BitcastOp>(caseErrorLoc, ptrType, payload);
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
                            Value ptr = builder.create<sir::BitcastOp>(caseReturnLoc, ptrType, ptr_u);
                            builder.create<sir::ReturnOp>(caseReturnLoc, ptr, len);
                        }
                        else if (info.retCount == 1)
                        {
                            Value val = call.getResult(0);
                            Value c32_ret = getConst(builder, caseReturnLoc, u256Type, i64Type, 32, constCache, caseBody, "word_size");
                            Value size = c32_ret;
                            Value ptr = builder.create<sir::SAllocAnyOp>(caseReturnLoc, ptrType, size);
                            setResultName(ptr.getDefiningOp(), ("buf_" + info.func.getName()).str());
                            builder.create<sir::StoreOp>(caseReturnLoc, ptr, val);
                            builder.create<sir::ReturnOp>(caseReturnLoc, ptr, size);
                        }
                        else
                        {
                            Value z = builder.create<sir::ConstOp>(caseReturnLoc, u256Type, IntegerAttr::get(i64Type, 0));
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
