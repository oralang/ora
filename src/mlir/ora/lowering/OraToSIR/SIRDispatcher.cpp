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

#include <string>

using namespace mlir;

namespace mlir
{
    namespace ora
    {
        namespace
        {
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
                if (s.front() == '(')
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

            struct PubFuncInfo
            {
                func::FuncOp func;
                uint32_t selector = 0;
                unsigned argCount = 0;
                unsigned retCount = 0;
                SmallVector<AbiType, 8> abiParams;
                int64_t minHeadBytes = 0;
                SmallVector<Type, 8> inputTypes;
            };

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

                        PubFuncInfo info;
                        info.func = func;
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
                            }
                        }

                        if (!info.abiParams.empty() && info.abiParams.size() != info.argCount)
                        {
                            func.emitError("ora.abi_params length does not match function argument count");
                            signalPassFailure();
                            return;
                        }

                        int64_t headSlots = 0;
                        for (size_t i = 0; i < info.argCount; ++i)
                        {
                            AbiType abi = info.abiParams.empty() ? AbiType{} : info.abiParams[i];
                            if (info.abiParams.empty())
                            {
                                headSlots += 1;
                            }
                            else
                            {
                                int64_t slots = abi.headSlots();
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

                    if (pubFuncs.empty() && !userInit)
                        return;

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
                    auto initFunc = func::FuncOp::create(loc, "init", initType);
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
                        Value codeSize = builder.create<sir::CodeSizeOp>(loc, u256Type);
                        Value initEnd = builder.create<sir::InitEndOffsetOp>(loc, u256Type);
                        Value codeTooShort = builder.create<sir::LtOp>(loc, u256Type, codeSize, initEnd);
                        Value dataLen = builder.create<sir::SubOp>(loc, u256Type, codeSize, initEnd);

                        int64_t minHeadBytes = 32 * headSlots;
                        if (minHeadBytes > 0)
                        {
                            Value minSizeVal = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(i64Type, minHeadBytes));
                            Value dataTooShort = builder.create<sir::LtOp>(loc, u256Type, dataLen, minSizeVal);
                            Value anyTooShort = builder.create<sir::OrOp>(loc, u256Type, codeTooShort, dataTooShort);
                            Value valid_args = builder.create<sir::IsZeroOp>(loc, u256Type, anyTooShort);
                            initDecode = initFunc.addBlock();
                            builder.create<sir::CondBrOp>(loc, valid_args, ValueRange{}, ValueRange{}, initDecode, getInitRevert());
                            builder.setInsertionPointToEnd(initDecode);
                        }
                        else
                        {
                            Value valid_args = builder.create<sir::IsZeroOp>(loc, u256Type, codeTooShort);
                            initDecode = initFunc.addBlock();
                            builder.create<sir::CondBrOp>(loc, valid_args, ValueRange{}, ValueRange{}, initDecode, getInitRevert());
                            builder.setInsertionPointToEnd(initDecode);
                        }

                        Value dataBuf = builder.create<sir::MallocOp>(loc, ptrType, dataLen);
                        builder.create<sir::CodeCopyOp>(loc, dataBuf, initEnd, dataLen);

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
                            Value offc = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(i64Type, offs));
                            Value headPtr = builder.create<sir::AddPtrOp>(loc, ptrType, dataBuf, offc);
                            Value head = builder.create<sir::LoadOp>(loc, u256Type, headPtr);
                            AbiType abi = initAbiParams.empty() ? AbiType{} : initAbiParams[idx];
                            Value argVal = head;

                            if (!initAbiParams.empty() && abi.isArray() && abi.supportsStaticArray())
                            {
                                int64_t elemCount = abi.dims.front();
                                int64_t totalBytes = elemCount * 32;
                                Value totalVal = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(i64Type, totalBytes));
                                Value buf = builder.create<sir::SAllocAnyOp>(loc, ptrType, totalVal);
                                Value src = builder.create<sir::AddPtrOp>(loc, ptrType, dataBuf, offc);
                                builder.create<sir::MCopyOp>(loc, buf, src, totalVal);
                                argVal = buf;
                            }
                            else if (!initAbiParams.empty() && abi.isDynamic())
                            {
                                if (abi.base == AbiBase::BytesDyn || abi.base == AbiBase::String || abi.supportsDynamicArray())
                                {
                                    Value c32_dyn = getConst(builder, loc, u256Type, i64Type, 32, constCache, builder.getInsertionBlock(), "word_size");
                                    Value absOff = head;
                                    Value absPtr = builder.create<sir::AddPtrOp>(loc, ptrType, dataBuf, absOff);
                                    Value len = builder.create<sir::LoadOp>(loc, u256Type, absPtr);

                                    Value total = nullptr;
                                    if (abi.baseIsDynamic())
                                    {
                                        Value c31 = getConst(builder, loc, u256Type, i64Type, 31, constCache, builder.getInsertionBlock(), "pad_31");
                                        Value c5 = getConst(builder, loc, u256Type, i64Type, 5, constCache, builder.getInsertionBlock(), "shift_5");
                                        Value lenPlus = builder.create<sir::AddOp>(loc, u256Type, len, c31);
                                        Value shifted = builder.create<sir::ShrOp>(loc, u256Type, c5, lenPlus);
                                        Value padded = builder.create<sir::ShlOp>(loc, u256Type, c5, shifted);
                                        total = builder.create<sir::AddOp>(loc, u256Type, padded, c32_dyn);
                                    }
                                    else
                                    {
                                        Value lenBytes = builder.create<sir::MulOp>(loc, u256Type, len, c32_dyn);
                                        total = builder.create<sir::AddOp>(loc, u256Type, lenBytes, c32_dyn);
                                    }

                                    Value end = builder.create<sir::AddOp>(loc, u256Type, absOff, total);
                                    Value tooShortDyn = builder.create<sir::LtOp>(loc, u256Type, dataLen, end);
                                    Value valid_dyn = builder.create<sir::IsZeroOp>(loc, u256Type, tooShortDyn);
                                    Block *dynBody = initFunc.addBlock();
                                    builder.create<sir::CondBrOp>(loc, valid_dyn, ValueRange{}, ValueRange{}, dynBody, getInitRevert());
                                    builder.setInsertionPointToEnd(dynBody);

                                    Value buf = builder.create<sir::SAllocAnyOp>(loc, ptrType, total);
                                    Value src = builder.create<sir::AddPtrOp>(loc, ptrType, dataBuf, absOff);
                                    builder.create<sir::MCopyOp>(loc, buf, src, total);
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
                                        argVal = builder.create<sir::BitcastOp>(loc, ptrType, argVal);
                                    argVal = builder.create<sir::BitcastOp>(loc, u256Type, argVal);
                                }
                            }
                            // sir.icall requires all args to be !sir.u256.
                            if (isa<sir::PtrType>(argVal.getType()))
                                argVal = builder.create<sir::BitcastOp>(loc, u256Type, argVal);

                            args.push_back(argVal);
                        }

                        builder.create<sir::ICallOp>(loc,
                                                     TypeRange{},
                                                     SymbolRefAttr::get(ctx, userInit.getName()),
                                                     args);
                    }

                    Value runtimeStart = builder.create<sir::RuntimeStartOffsetOp>(loc, u256Type);
                    Value runtimeLen = builder.create<sir::RuntimeLengthOp>(loc, u256Type);
                    Value initBuf = builder.create<sir::MallocOp>(loc, ptrType, runtimeLen);
                    builder.create<sir::CodeCopyOp>(loc, initBuf, runtimeStart, runtimeLen);
                    builder.create<sir::ReturnOp>(loc, initBuf, runtimeLen);

                    // Revert block for constructor argument checks.
                    if (initRevert)
                    {
                        builder.setInsertionPointToEnd(initRevert);
                        Value c0_revert = getConst(builder, loc, u256Type, i64Type, 0, constCache, initRevert, "zero");
                        Value p0b = builder.create<sir::BitcastOp>(loc, ptrType, c0_revert);
                        builder.create<sir::RevertOp>(loc, p0b, c0_revert);
                    }
                    module.push_back(initFunc);

                    // Build dispatcher: func main() -> ()
                    auto mainType = builder.getFunctionType({}, {});
                    auto mainFunc = func::FuncOp::create(loc, "main", mainType);
                    Block *entry = mainFunc.addEntryBlock();
                    Block *loadSelector = mainFunc.addBlock();
                    Block *revertError = mainFunc.addBlock();

                    // entry: callvalue check (non-payable by default) + const prelude
                    builder.setInsertionPointToEnd(entry);
                    static_cast<void>(getConst(builder, loc, u256Type, i64Type, 0, constCache, entry, "zero"));
                    static_cast<void>(getConst(builder, loc, u256Type, i64Type, 4, constCache, entry, "selector_offset"));
                    static_cast<void>(getConst(builder, loc, u256Type, i64Type, 32, constCache, entry, "word_size"));
                    static_cast<void>(getConst(builder, loc, u256Type, i64Type, 35, constCache, entry, "min_cdsize_1arg"));
                    static_cast<void>(getConst(builder, loc, u256Type, i64Type, 36, constCache, entry, "arg1_offset"));
                    static_cast<void>(getConst(builder, loc, u256Type, i64Type, 67, constCache, entry, "min_cdsize_2args"));
                    static_cast<void>(getConst(builder, loc, u256Type, i64Type, 68, constCache, entry, "arg2_offset"));
                    static_cast<void>(getConst(builder, loc, u256Type, i64Type, 99, constCache, entry, "min_cdsize_3args"));
                    static_cast<void>(getConst(builder, loc, u256Type, i64Type, 224, constCache, entry, "selector_shift"));

                    Value cv = builder.create<sir::CallValueOp>(loc, u256Type);
                    setResultName(cv.getDefiningOp(), "cv");
                    Value cv_zero = builder.create<sir::IsZeroOp>(loc, u256Type, cv);
                    setResultName(cv_zero.getDefiningOp(), "cv_nonzero");
                    builder.create<sir::CondBrOp>(loc, cv_zero, ValueRange{}, ValueRange{}, loadSelector, revertError);
                    setBlockName(entry, "main_entry");
                    setBlockOrder(entry, 0);

                    // load_selector: selector + switch
                    builder.setInsertionPointToEnd(loadSelector);
                    Value c0_ls = getConst(builder, loc, u256Type, i64Type, 0, constCache, loadSelector, "zero");
                    Value c224_ls = getConst(builder, loc, u256Type, i64Type, 224, constCache, loadSelector, "selector_shift");
                    Value word = builder.create<sir::CallDataLoadOp>(loc, u256Type, c0_ls);
                    setResultName(word.getDefiningOp(), "selector_word");
                    Value selector = builder.create<sir::ShrOp>(loc, u256Type, c224_ls, word);
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
                    auto sw = builder.create<sir::SwitchOp>(loc, selector, caseAttr, revertError, caseBlocks);
                    sw->setAttr("sir.selector_switch", builder.getUnitAttr());
                    setBlockName(loadSelector, "load_selector");
                    setBlockOrder(loadSelector, 1);

                    // case blocks
                    for (size_t i = 0; i < pubFuncs.size(); ++i)
                    {
                        auto &info = pubFuncs[i];
                        Block *caseCheck = caseBlocks[i];
                        Block *caseBody = caseCheck;
                        builder.setInsertionPointToEnd(caseCheck);

                        int64_t minSize = info.minHeadBytes > 0 ? info.minHeadBytes : (4 + 32 * static_cast<int64_t>(info.argCount));
                        if (minSize > 4)
                        {
                            Value cdsize_case = builder.create<sir::CallDataSizeOp>(loc, u256Type);
                            setResultName(cdsize_case.getDefiningOp(), ("cdsize_" + info.func.getName()).str());
                            int64_t minMinus = minSize - 1;
                            StringRef minName = (minMinus == 35   ? StringRef("min_cdsize_1arg")
                                                 : minMinus == 67 ? StringRef("min_cdsize_2args")
                                                 : minMinus == 99 ? StringRef("min_cdsize_3args")
                                                                  : StringRef());
                            Value minSizeVal = getConst(builder, loc, u256Type, i64Type, minMinus, constCache, caseCheck, minName);
                            Value valid_args = builder.create<sir::LtOp>(loc, u256Type, minSizeVal, cdsize_case);
                            setResultName(valid_args.getDefiningOp(), ("valid_" + info.func.getName()).str());
                            caseBody = mainFunc.addBlock();
                            builder.create<sir::CondBrOp>(loc, valid_args, ValueRange{}, ValueRange{}, caseBody, revertError);
                            builder.setInsertionPointToEnd(caseBody);
                        }

                        SmallVector<Value, 8> args;
                        SmallVector<int64_t, 8> headOffsets;
                        int64_t headSlot = 0;
                        for (unsigned i = 0; i < info.argCount; ++i)
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

                        for (unsigned idx = 0; idx < info.argCount; ++idx)
                        {
                            int64_t offs = headOffsets[idx];
                            StringRef offName = offs == 4  ? StringRef("selector_offset")
                                              : offs == 36 ? StringRef("arg1_offset")
                                              : offs == 68 ? StringRef("arg2_offset")
                                                           : StringRef();
                            Value offc = offName.empty()
                                             ? builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(i64Type, offs))
                                             : getConst(builder, loc, u256Type, i64Type, offs, constCache, caseBody, offName);
                            Value head = builder.create<sir::CallDataLoadOp>(loc, u256Type, offc);
                            StringRef argPrefix = idx == 0 ? "a_" : (idx == 1 ? "b_" : (idx == 2 ? "n_" : "arg_"));
                            setResultName(head.getDefiningOp(), (argPrefix + info.func.getName()).str());

                            AbiType abi = info.abiParams.empty() ? AbiType{} : info.abiParams[idx];
                            Value argVal = head;

                            if (!info.abiParams.empty() && abi.isArray() && abi.supportsStaticArray())
                            {
                                int64_t elemCount = abi.dims.front();
                                int64_t totalBytes = elemCount * 32;
                                Value totalVal = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(i64Type, totalBytes));
                                Value buf = builder.create<sir::SAllocAnyOp>(loc, ptrType, totalVal);
                                builder.create<sir::CallDataCopyOp>(loc, buf, offc, totalVal);
                                argVal = buf;
                            }
                            else if (!info.abiParams.empty() && abi.isDynamic())
                            {
                                if (abi.base == AbiBase::BytesDyn || abi.base == AbiBase::String || abi.supportsDynamicArray())
                                {
                                    Value c4_dyn = getConst(builder, loc, u256Type, i64Type, 4, constCache, caseBody, "selector_offset");
                                    Value c32_dyn = getConst(builder, loc, u256Type, i64Type, 32, constCache, caseBody, "word_size");
                                    Value absOff = builder.create<sir::AddOp>(loc, u256Type, head, c4_dyn);
                                    Value len = builder.create<sir::CallDataLoadOp>(loc, u256Type, absOff);

                                    Value total = nullptr;
                                    if (abi.baseIsDynamic())
                                    {
                                        Value c31 = getConst(builder, loc, u256Type, i64Type, 31, constCache, caseBody, "pad_31");
                                        Value c5 = getConst(builder, loc, u256Type, i64Type, 5, constCache, caseBody, "shift_5");
                                        Value lenPlus = builder.create<sir::AddOp>(loc, u256Type, len, c31);
                                        Value shifted = builder.create<sir::ShrOp>(loc, u256Type, c5, lenPlus);
                                        Value padded = builder.create<sir::ShlOp>(loc, u256Type, c5, shifted);
                                        total = builder.create<sir::AddOp>(loc, u256Type, padded, c32_dyn);
                                    }
                                    else
                                    {
                                        Value lenBytes = builder.create<sir::MulOp>(loc, u256Type, len, c32_dyn);
                                        total = builder.create<sir::AddOp>(loc, u256Type, lenBytes, c32_dyn);
                                    }

                                    Value cdsize_case = builder.create<sir::CallDataSizeOp>(loc, u256Type);
                                    Value end = builder.create<sir::AddOp>(loc, u256Type, absOff, total);
                                    // Check cdsize >= end (i.e., !(cdsize < end))
                                    Value tooShort = builder.create<sir::LtOp>(loc, u256Type, cdsize_case, end);
                                    Value valid_dyn = builder.create<sir::IsZeroOp>(loc, u256Type, tooShort);
                                    Block *dynBody = mainFunc.addBlock();
                                    builder.create<sir::CondBrOp>(loc, valid_dyn, ValueRange{}, ValueRange{}, dynBody, revertError);
                                    builder.setInsertionPointToEnd(dynBody);

                                    Value buf = builder.create<sir::SAllocAnyOp>(loc, ptrType, total);
                                    builder.create<sir::CallDataCopyOp>(loc, buf, absOff, total);
                                    argVal = buf;
                                }
                                else
                                {
                                    module.emitError("unsupported dynamic ABI type for dispatcher");
                                    signalPassFailure();
                                    return;
                                }
                            }

                            if (idx < info.inputTypes.size())
                            {
                                if (auto ptrTy = dyn_cast<sir::PtrType>(info.inputTypes[idx]))
                                {
                                    if (!isa<sir::PtrType>(argVal.getType()))
                                    {
                                        argVal = builder.create<sir::BitcastOp>(loc, ptrType, argVal);
                                    }
                                    argVal = builder.create<sir::BitcastOp>(loc, u256Type, argVal);
                                }
                            }
                            // sir.icall requires all args to be !sir.u256.
                            if (isa<sir::PtrType>(argVal.getType()))
                                argVal = builder.create<sir::BitcastOp>(loc, u256Type, argVal);
                            args.push_back(argVal);
                        }

                        SmallVector<Type, 4> resultTypes;
                        for (unsigned r = 0; r < info.retCount; ++r)
                            resultTypes.push_back(u256Type);

                        auto call = builder.create<sir::ICallOp>(
                            loc,
                            resultTypes,
                            SymbolRefAttr::get(ctx, info.func.getName()),
                            args);

                        if (info.retCount == 2)
                        {
                            Value ptr_u = call.getResult(0);
                            Value len = call.getResult(1);
                            Value ptr = builder.create<sir::BitcastOp>(loc, ptrType, ptr_u);
                            builder.create<sir::ReturnOp>(loc, ptr, len);
                        }
                        else if (info.retCount == 1)
                        {
                            Value val = call.getResult(0);
                            Value c32_ret = getConst(builder, loc, u256Type, i64Type, 32, constCache, caseBody, "word_size");
                            Value size = c32_ret;
                            Value ptr = builder.create<sir::SAllocAnyOp>(loc, ptrType, size);
                            setResultName(ptr.getDefiningOp(), ("buf_" + info.func.getName()).str());
                            builder.create<sir::StoreOp>(loc, ptr, val);
                            builder.create<sir::ReturnOp>(loc, ptr, size);
                        }
                        else
                        {
                            Value z = builder.create<sir::ConstOp>(loc, u256Type, IntegerAttr::get(i64Type, 0));
                            Value pz = builder.create<sir::BitcastOp>(loc, ptrType, z);
                            builder.create<sir::ReturnOp>(loc, pz, z);
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
                    Value c0_revert_main = getConst(builder, loc, u256Type, i64Type, 0, constCache, revertError, "zero");
                    Value p0b_main = builder.create<sir::BitcastOp>(loc, ptrType, c0_revert_main);
                    builder.create<sir::RevertOp>(loc, p0b_main, c0_revert_main);
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
