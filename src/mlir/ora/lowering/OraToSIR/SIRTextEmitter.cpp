//===- SIRTextEmitter.cpp - SIR Text Emitter -------------------------===//
//
// Emits Sensei SIR text from SIR MLIR. Assumes SIR text legality has been
// validated by the SIRTextLegalizer pass.
//
//===----------------------------------------------------------------------===//

#include "SIRTextEmitter.h"

#include "SIR/SIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <unordered_set>
#include <unordered_map>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Casting.h"
#include <cstdio>
#include <functional>
#include <optional>

using namespace mlir;

namespace
{
    struct NameTable
    {
        llvm::DenseMap<Value, std::string> valueNames;
        llvm::DenseMap<Block *, std::string> blockNames;
        std::unordered_set<std::string> usedNames;
        std::unordered_set<std::string> emittedDefs;
        std::unordered_map<std::string, Value> constNames;
        unsigned nextValueId = 0;

        std::string allocateName(StringRef base)
        {
            if (base.empty())
                return "";
            std::string name = base.str();
            if (usedNames.find(name) == usedNames.end())
            {
                usedNames.insert(name);
                return name;
            }
            unsigned i = 0;
            while (true)
            {
                std::string candidate = name + "_" + std::to_string(i++);
                if (usedNames.find(candidate) == usedNames.end())
                {
                    usedNames.insert(candidate);
                    return candidate;
                }
            }
        }

        std::string nameFor(Value v)
        {
            auto it = valueNames.find(v);
            if (it != valueNames.end())
                return it->second;

            if (Operation *def = v.getDefiningOp())
            {
                if (auto res = dyn_cast<OpResult>(v))
                {
                    unsigned idx = res.getResultNumber();
                    std::string attrName = "sir.result_name_" + std::to_string(idx);
                    if (auto nameAttr = def->getAttrOfType<StringAttr>(attrName))
                    {
                        std::string n = nameAttr.getValue().str();
                        if (!n.empty())
                        {
                            // Allow consts to reuse explicit names across blocks
                            // only if they share the same value attribute.
                            if (auto cOp = dyn_cast<sir::ConstOp>(def))
                            {
                                auto existing = constNames.find(n);
                                if (existing == constNames.end())
                                {
                                    constNames[n] = v;
                                    valueNames[v] = n;
                                    usedNames.insert(n);
                                    return n;
                                }
                                // Same name already registered — reuse only if same value.
                                if (auto prevConst = existing->second.getDefiningOp<sir::ConstOp>())
                                {
                                    if (prevConst.getValueAttr() == cOp.getValueAttr())
                                    {
                                        valueNames[v] = n;
                                        return n;
                                    }
                                }
                                // Different value — fall through to allocateName() below.
                            }
                            std::string unique = allocateName(n);
                            if (!unique.empty())
                            {
                                valueNames[v] = unique;
                                return unique;
                            }
                        }
                    }
                }
            }

            std::string name = allocateName("v" + std::to_string(nextValueId++));
            valueNames[v] = name;
            return name;
        }
    };

    std::string formatInt(const APInt &v)
    {
        llvm::SmallString<32> s;
        // Sensei SIR text parser only accepts unsigned decimal/hex literals.
        // Emit all APInt values as unsigned hex (two's-complement for negatives).
        v.toString(s, 16, false, false);
        return ("0x" + s.str()).str();
    }

    std::string formatSelector(const APInt &v)
    {
        llvm::SmallString<32> s;
        v.toString(s, 16, false, false);
        if (s.size() < 8)
        {
            llvm::SmallString<32> padded;
            padded.append(8 - s.size(), '0');
            padded.append(s);
            s = padded;
        }
        return ("0x" + s.str()).str();
    }

    std::string formatConstKey(const APInt &v)
    {
        llvm::SmallString<32> s;
        v.toString(s, 16, v.isNegative(), false);
        return (std::to_string(v.getBitWidth()) + ":" + s.str()).str();
    }

    std::string formatStringAsHex(StringRef s)
    {
        static const char *hex = "0123456789abcdef";
        std::string out;
        out.reserve(2 + s.size() * 2);
        out += "0x";
        for (unsigned char c : s)
        {
            out += hex[c >> 4];
            out += hex[c & 0x0f];
        }
        return out;
    }

    std::string sanitizeGlobalName(StringRef s)
    {
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s)
        {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')
            {
                out.push_back(static_cast<char>(c));
            }
            else
            {
                out.push_back('_');
            }
        }
        return out;
    }

    std::string mnemonicFor(Operation &op)
    {
        StringRef name = op.getName().getStringRef();
        if (name.starts_with("sir."))
            return name.drop_front(4).str();
        return name.str();
    }

    std::optional<FileLineColLoc> extractFileLineColLoc(Location loc)
    {
        if (loc == nullptr)
            return std::nullopt;
        if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
            return fileLoc;
        if (auto nameLoc = dyn_cast<NameLoc>(loc))
            return extractFileLineColLoc(nameLoc.getChildLoc());
        if (auto callSite = dyn_cast<CallSiteLoc>(loc))
        {
            if (auto calleeLoc = extractFileLineColLoc(callSite.getCallee()))
                return calleeLoc;
            return extractFileLineColLoc(callSite.getCaller());
        }
        if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
        {
            for (Location child : fusedLoc.getLocations())
            {
                if (auto childLoc = extractFileLineColLoc(child))
                    return childLoc;
            }
        }
        return std::nullopt;
    }

    std::optional<uint32_t> extractStmtIdFromLoc(Location loc)
    {
        if (loc == nullptr)
            return std::nullopt;
        if (auto nameLoc = dyn_cast<NameLoc>(loc))
        {
            StringRef name = nameLoc.getName().getValue();
            constexpr StringLiteral prefix("ora.stmt.");
            if (name.starts_with(prefix))
            {
                uint32_t stmtId = 0;
                if (!name.drop_front(prefix.size()).getAsInteger(10, stmtId))
                    return stmtId;
            }
            return extractStmtIdFromLoc(nameLoc.getChildLoc());
        }
        if (auto callSite = dyn_cast<CallSiteLoc>(loc))
        {
            if (auto calleeStmt = extractStmtIdFromLoc(callSite.getCallee()))
                return calleeStmt;
            return extractStmtIdFromLoc(callSite.getCaller());
        }
        if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
        {
            for (Location child : fusedLoc.getLocations())
            {
                if (auto stmtId = extractStmtIdFromLoc(child))
                    return stmtId;
            }
        }
        return std::nullopt;
    }

    std::optional<uint32_t> extractOriginStmtIdFromLoc(Location loc)
    {
        if (loc == nullptr)
            return std::nullopt;
        if (auto nameLoc = dyn_cast<NameLoc>(loc))
        {
            StringRef name = nameLoc.getName().getValue();
            constexpr StringLiteral prefix("ora.origin_stmt.");
            if (name.starts_with(prefix))
            {
                uint32_t stmtId = 0;
                if (!name.drop_front(prefix.size()).getAsInteger(10, stmtId))
                    return stmtId;
            }
            return extractOriginStmtIdFromLoc(nameLoc.getChildLoc());
        }
        if (auto callSite = dyn_cast<CallSiteLoc>(loc))
        {
            if (auto calleeStmt = extractOriginStmtIdFromLoc(callSite.getCallee()))
                return calleeStmt;
            return extractOriginStmtIdFromLoc(callSite.getCaller());
        }
        if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
        {
            for (Location child : fusedLoc.getLocations())
            {
                if (auto stmtId = extractOriginStmtIdFromLoc(child))
                    return stmtId;
            }
        }
        return std::nullopt;
    }

    bool hasSyntheticLocTag(Location loc)
    {
        if (loc == nullptr)
            return false;
        if (auto nameLoc = dyn_cast<NameLoc>(loc))
        {
            StringRef name = nameLoc.getName().getValue();
            constexpr StringLiteral prefix("ora.synthetic.");
            if (name.starts_with(prefix))
                return true;
            return hasSyntheticLocTag(nameLoc.getChildLoc());
        }
        if (auto callSite = dyn_cast<CallSiteLoc>(loc))
        {
            return hasSyntheticLocTag(callSite.getCallee()) || hasSyntheticLocTag(callSite.getCaller());
        }
        if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
        {
            for (Location child : fusedLoc.getLocations())
            {
                if (hasSyntheticLocTag(child))
                    return true;
            }
        }
        return false;
    }

    std::optional<std::pair<uint32_t, uint32_t>> extractSyntheticLocTag(Location loc)
    {
        SmallVector<std::pair<uint32_t, uint32_t>, 4> tags;
        std::function<void(Location)> collect = [&](Location current) {
            if (current == nullptr)
                return;
            if (auto nameLoc = dyn_cast<NameLoc>(current))
            {
                collect(nameLoc.getChildLoc());
                StringRef name = nameLoc.getName().getValue();
                constexpr StringLiteral prefix("ora.synthetic.");
                if (name.starts_with(prefix))
                {
                    StringRef rest = name.drop_front(prefix.size());
                    auto parts = rest.split('.');
                    uint32_t syntheticIndex = 0;
                    uint32_t syntheticCount = 0;
                    if (!parts.first.empty() && !parts.second.empty() &&
                        !parts.first.getAsInteger(10, syntheticIndex) &&
                        !parts.second.getAsInteger(10, syntheticCount))
                    {
                        tags.emplace_back(syntheticIndex, syntheticCount);
                    }
                }
                return;
            }
            if (auto callSite = dyn_cast<CallSiteLoc>(current))
            {
                collect(callSite.getCallee());
                collect(callSite.getCaller());
                return;
            }
            if (auto fusedLoc = dyn_cast<FusedLoc>(current))
            {
                for (Location child : fusedLoc.getLocations())
                    collect(child);
            }
        };
        collect(loc);
        if (tags.empty())
            return std::nullopt;
        return tags.back();
    }

    std::optional<std::string> extractSyntheticLocPath(Location loc)
    {
        if (loc == nullptr)
            return std::nullopt;
        SmallVector<std::pair<uint32_t, uint32_t>, 4> tags;
        std::function<void(Location)> collect = [&](Location current) {
            if (current == nullptr)
                return;
            if (auto nameLoc = dyn_cast<NameLoc>(current))
            {
                collect(nameLoc.getChildLoc());
                StringRef name = nameLoc.getName().getValue();
                constexpr StringLiteral prefix("ora.synthetic.");
                if (name.starts_with(prefix))
                {
                    StringRef rest = name.drop_front(prefix.size());
                    auto parts = rest.split('.');
                    uint32_t syntheticIndex = 0;
                    uint32_t syntheticCount = 0;
                    if (!parts.first.empty() && !parts.second.empty() &&
                        !parts.first.getAsInteger(10, syntheticIndex) &&
                        !parts.second.getAsInteger(10, syntheticCount))
                    {
                        tags.emplace_back(syntheticIndex, syntheticCount);
                    }
                }
                return;
            }
            if (auto callSite = dyn_cast<CallSiteLoc>(current))
            {
                collect(callSite.getCallee());
                collect(callSite.getCaller());
                return;
            }
            if (auto fusedLoc = dyn_cast<FusedLoc>(current))
            {
                for (Location child : fusedLoc.getLocations())
                    collect(child);
            }
        };
        collect(loc);
        if (tags.empty())
            return std::nullopt;
        std::string path;
        llvm::raw_string_ostream os(path);
        for (size_t i = 0; i < tags.size(); ++i)
        {
            if (i != 0)
                os << "/";
            os << tags[i].first << "." << tags[i].second;
        }
        os.flush();
        return path;
    }

    template <typename RangeT>
    void emitValueList(raw_ostream &os, NameTable &names, RangeT values)
    {
        size_t i = 0;
        for (auto v : values)
        {
            if (i++)
                os << " ";
            os << names.nameFor(v);
        }
    }

    void emitOperandList(raw_ostream &os, NameTable &names, OperandRange operands)
    {
        for (size_t i = 0; i < operands.size(); ++i)
        {
            if (i)
                os << " ";
            os << names.nameFor(operands[i]);
        }
    }

    void emitTerminator(raw_ostream &os, NameTable &names, Block &block)
    {
        Operation &term = block.back();

        if (auto br = dyn_cast<sir::BrOp>(term))
        {
            os << "=> @" << names.blockNames[br.getDest()];
            return;
        }
        if (auto br = dyn_cast<sir::CondBrOp>(term))
        {
            os << "=> " << names.nameFor(br.getCond()) << " ? @"
               << names.blockNames[br.getTrueDest()] << " : @"
               << names.blockNames[br.getFalseDest()];
            return;
        }
        if (auto sw = dyn_cast<sir::SwitchOp>(term))
        {
            const bool isSelectorSwitch = sw->hasAttr("sir.selector_switch");
            os << "switch " << names.nameFor(sw.getSelector()) << " {\n";
            auto caseVals = sw.getCaseValues();
            auto caseDests = sw.getCaseDests();
            for (size_t i = 0; i < caseVals.size(); ++i)
            {
                auto intAttr = dyn_cast<IntegerAttr>(caseVals[i]);
                if (!intAttr)
                {
                    os << "        " << "0x0" << " => @"
                       << names.blockNames[caseDests[i]] << "\n";
                    continue;
                }
                const auto &ap = intAttr.getValue();
                os << "        " << (isSelectorSwitch ? formatSelector(ap) : formatInt(ap)) << " => @"
                   << names.blockNames[caseDests[i]] << "\n";
            }
            os << "        default => @" << names.blockNames[sw.getDefaultDest()] << "\n";
            os << "    }";
            return;
        }
        if (auto ret = dyn_cast<sir::ReturnOp>(term))
        {
            os << "return " << names.nameFor(ret.getPtr()) << " " << names.nameFor(ret.getLen());
            return;
        }
        if (isa<sir::StopOp>(term))
        {
            os << "stop";
            return;
        }
        if (auto rev = dyn_cast<sir::RevertOp>(term))
        {
            os << "revert " << names.nameFor(rev.getPtr()) << " " << names.nameFor(rev.getLen());
            return;
        }
        if (isa<sir::InvalidOp>(term))
        {
            os << "invalid";
            return;
        }
        if (auto sd = dyn_cast<sir::SelfDestructOp>(term))
        {
            os << "selfdestruct " << names.nameFor(sd.getBeneficiary());
            return;
        }
        if (isa<sir::IRetOp>(term))
        {
            os << "iret";
            return;
        }

        os << "// unsupported terminator: " << term.getName().getStringRef();
    }

    void emitOperation(raw_ostream &os, NameTable &names, Operation &op)
    {
        // Skip terminators here; they are emitted by emitTerminator.
        if (op.hasTrait<OpTrait::IsTerminator>())
            return;

        if (auto cst = dyn_cast<sir::ConstOp>(op))
        {
            if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
            {
                std::string key = formatConstKey(intAttr.getValue());
                if (auto it = names.constNames.find(key); it != names.constNames.end())
                {
                    Value defVal = it->second;
                    if (auto defOp = defVal.getDefiningOp())
                    {
                        if (defOp->getBlock() == op.getBlock() && defOp->isBeforeInBlock(&op))
                        {
                            names.valueNames[cst.getResult()] = names.nameFor(defVal);
                            return;
                        }
                    }
                }
            }
        }

        bool isConst = isa<sir::ConstOp>(op);
        if (isConst && op.getNumResults() == 1)
        {
            std::string name = names.nameFor(op.getResult(0));
            if (names.emittedDefs.find(name) != names.emittedDefs.end())
                return;
            names.emittedDefs.insert(name);
        }

        if (auto bitcast = dyn_cast<sir::BitcastOp>(op))
        {
            // Bitcast has no SIR text equivalent; alias the result to the operand.
            Operation *bitcastOp = bitcast.getOperation();
            if (bitcastOp->getNumOperands() == 1 && bitcastOp->getNumResults() == 1)
            {
                Value src = bitcast.getOperand();
                Value dst = bitcast.getResult();
                names.valueNames[dst] = names.nameFor(src);
            }
            return;
        }
        if (isa<sir::ErrorDeclOp>(op))
            return;

        // select(cond, a, b) → or(and(sub(0,cond), a), and(not(sub(0,cond)), b))
        // Expanded inline because sensei has no "select" mnemonic.
        if (auto sel = dyn_cast<sir::SelectOp>(op))
        {
            const char *ind = "        ";
            std::string result = names.nameFor(sel.getRes());
            std::string condN = names.nameFor(sel.getCond());
            std::string trueN = names.nameFor(sel.getTrueValue());
            std::string falseN = names.nameFor(sel.getFalseValue());
            std::string mask = names.allocateName("sel_mask");
            std::string t = names.allocateName("sel_t");
            std::string inv = names.allocateName("sel_inv");
            std::string f = names.allocateName("sel_f");
            // Reuse existing zero const or emit one.
            std::string zero;
            std::string zeroKey = formatConstKey(APInt(256, 0));
            auto zIt = names.constNames.find(zeroKey);
            if (zIt != names.constNames.end())
                zero = names.nameFor(zIt->second);
            else
            {
                zero = names.allocateName("sel_zero");
                os << zero << " = const 0x0\n" << ind;
            }
            // mask = 0 - cond  (0→0, 1→0xFF..FF)
            os << mask << " = sub " << zero << " " << condN << "\n" << ind;
            os << t << " = and " << mask << " " << trueN << "\n" << ind;
            os << inv << " = not " << mask << "\n" << ind;
            os << f << " = and " << inv << " " << falseN << "\n" << ind;
            os << result << " = or " << t << " " << f;
            return;
        }

        // Results
        if (op.getNumResults() > 0)
        {
            emitValueList(os, names, op.getResults());
            os << " = ";
        }

        std::string mnemonic = mnemonicFor(op);
        if (isa<sir::AddPtrOp>(op))
            mnemonic = "add";
        else if (isa<sir::LoadOp>(op))
            mnemonic = "mload256";
        else if (isa<sir::StoreOp>(op))
            mnemonic = "mstore256";
        else if (isa<sir::Load8Op>(op))
            mnemonic = "mload8";
        else if (isa<sir::Store8Op>(op))
            mnemonic = "mstore8";
        else if (auto sallocany = dyn_cast<sir::SAllocAnyOp>(op))
        {
            if (!sallocany.getSize().getDefiningOp<sir::ConstOp>())
                mnemonic = "mallocany";
        }
        else if (auto cst = dyn_cast<sir::ConstOp>(op))
        {
            // Sensei uses "const" for u32 values, "large_const" for larger.
            if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
            {
                if (intAttr.getValue().getActiveBits() > 32)
                    mnemonic = "large_const";
            }
        }

        os << mnemonic;

        if (auto cst = dyn_cast<sir::ConstOp>(op))
        {
            if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
            {
                std::string key = formatConstKey(intAttr.getValue());
                if (auto it = names.constNames.find(key); it != names.constNames.end())
                {
                    Value defVal = it->second;
                    if (auto defOp = defVal.getDefiningOp())
                    {
                        if (defOp->getBlock() == op.getBlock() && defOp->isBeforeInBlock(&op))
                        {
                            names.valueNames[cst.getResult()] = names.nameFor(defVal);
                            return;
                        }
                    }
                }
                names.constNames[key] = cst.getResult();
                os << " " << formatInt(intAttr.getValue());
            }
            return;
        }
        if (auto sc = dyn_cast<sir::StringConstantOp>(op))
        {
            os << " " << formatStringAsHex(sc.getValue());
            return;
        }
        // NOTE: ErrorDeclOp is handled by early-return above (line ~310).
        if (auto ic = dyn_cast<sir::ICallOp>(op))
        {
            os << " @" << sanitizeGlobalName(ic.getCalleeAttr().getValue());
            if (!ic.getArgs().empty())
            {
                os << " ";
                emitValueList(os, names, ic.getArgs());
            }
            return;
        }

        if (auto salloc = dyn_cast<sir::SAllocOp>(op))
        {
            if (auto cst = salloc.getSize().getDefiningOp<sir::ConstOp>())
            {
                if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
                {
                    os << " " << formatInt(intAttr.getValue());
                    return;
                }
            }
        }
        if (auto sallocany = dyn_cast<sir::SAllocAnyOp>(op))
        {
            if (auto cst = sallocany.getSize().getDefiningOp<sir::ConstOp>())
            {
                if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
                {
                    os << " " << formatInt(intAttr.getValue());
                    return;
                }
            }
        }

        if (!op.getOperands().empty())
        {
            os << " ";
            emitOperandList(os, names, op.getOperands());
        }
    }

    static Value peelBitcast(Value v)
    {
        while (auto *def = v.getDefiningOp())
        {
            if (auto bitcast = dyn_cast<sir::BitcastOp>(def))
            {
                if (def->getNumOperands() == 1)
                {
                    v = bitcast.getOperand();
                    continue;
                }
            }
            break;
        }
        return v;
    }

    SmallVector<Value, 8> computeBlockOutputs(Block &block)
    {
        Operation *term = block.getTerminator();
        if (!term)
            return {};
        if (auto br = dyn_cast<sir::BrOp>(term))
        {
            SmallVector<Value, 8> out;
            out.reserve(br.getDestOperands().size());
            for (Value v : br.getDestOperands())
                out.push_back(peelBitcast(v));
            return out;
        }
        if (auto br = dyn_cast<sir::CondBrOp>(term))
        {
            SmallVector<Value, 8> out;
            out.reserve(br.getTrueOperands().size());
            for (Value v : br.getTrueOperands())
                out.push_back(peelBitcast(v));
            return out;
        }
        if (auto iret = dyn_cast<sir::IRetOp>(term))
        {
            SmallVector<Value, 8> out;
            out.reserve(iret.getValues().size());
            for (Value v : iret.getValues())
                out.push_back(peelBitcast(v));
            return out;
        }
        return {};
    }

    static SmallVector<func::FuncOp, 8> collectSortedFuncs(ModuleOp module)
    {
        SmallVector<func::FuncOp, 8> funcs;
        for (func::FuncOp func : module.getOps<func::FuncOp>())
            funcs.push_back(func);

        llvm::stable_sort(funcs, [&](func::FuncOp a, func::FuncOp b) {
            auto rank = [](func::FuncOp f) -> int {
                if (f.getName() == "init")
                    return 0;
                if (f.getName() == "main")
                    return 2;
                return 1;
            };
            int ra = rank(a);
            int rb = rank(b);
            if (ra != rb)
                return ra < rb;
            return false;
        });

        return funcs;
    }

    static SmallVector<Block *, 16> collectOrderedBlocks(func::FuncOp func, NameTable &names)
    {
        bool first = true;
        unsigned bbId = 0;
        SmallVector<Block *, 16> blocks;
        for (Block &block : func.getBlocks())
        {
            std::string blockName;
            if (!block.empty())
            {
                if (auto attr = block.back().getAttrOfType<StringAttr>("sir.block_name"))
                    blockName = attr.getValue().str();
            }

            if (blockName.empty())
            {
                if (first)
                {
                    blockName = "entry";
                    first = false;
                }
                else
                {
                    blockName = "bb" + std::to_string(bbId++);
                }
            }

            blockName = names.allocateName(blockName);
            names.blockNames[&block] = blockName;
            blocks.push_back(&block);
        }

        bool hasOrder = false;
        llvm::SmallVector<std::pair<int64_t, Block *>, 16> ordered;
        ordered.reserve(blocks.size());
        for (size_t i = 0; i < blocks.size(); ++i)
        {
            Block *block = blocks[i];
            int64_t order = static_cast<int64_t>(i);
            if (!block->empty())
            {
                if (auto attr = block->back().getAttrOfType<IntegerAttr>("sir.block_order"))
                {
                    order = attr.getInt();
                    hasOrder = true;
                }
            }
            ordered.push_back({order, block});
        }
        if (hasOrder)
        {
            llvm::stable_sort(ordered, [](const auto &a, const auto &b) { return a.first < b.first; });
            blocks.clear();
            for (auto &pair : ordered)
                blocks.push_back(pair.second);
        }

        return blocks;
    }

    static void preassignNamesForFunc(func::FuncOp func, NameTable &names)
    {
        for (Block &block : func.getBlocks())
        {
            for (BlockArgument arg : block.getArguments())
                names.nameFor(arg);

            for (Operation &op : block)
            {
                if (auto cst = dyn_cast<sir::ConstOp>(op))
                {
                    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
                    {
                        std::string key = formatConstKey(intAttr.getValue());
                        auto it = names.constNames.find(key);
                        if (it != names.constNames.end())
                        {
                            Value defVal = it->second;
                            if (auto defOp = defVal.getDefiningOp())
                            {
                                if (defOp->getBlock() == &block && defOp->isBeforeInBlock(&op))
                                {
                                    names.valueNames[cst.getResult()] = names.nameFor(defVal);
                                    continue;
                                }
                            }
                        }
                        names.nameFor(cst.getResult());
                        names.constNames[key] = cst.getResult();
                        continue;
                    }
                }

                if (auto bitcast = dyn_cast<sir::BitcastOp>(op))
                {
                    if (bitcast->getNumOperands() == 1 && bitcast->getNumResults() == 1)
                    {
                        names.valueNames[bitcast.getResult()] = names.nameFor(bitcast.getOperand());
                        continue;
                    }
                }

                for (Value res : op.getResults())
                    names.nameFor(res);
            }
        }
    }

    struct OpLineKey
    {
        Operation *op;
        uint32_t syntheticIndex;

        bool operator==(const OpLineKey &other) const
        {
            return op == other.op && syntheticIndex == other.syntheticIndex;
        }
    };

    struct OpLineKeyHash
    {
        std::size_t operator()(const OpLineKey &key) const
        {
            std::size_t h1 = std::hash<void *>{}(key.op);
            std::size_t h2 = std::hash<uint32_t>{}(key.syntheticIndex);
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };

    static bool isBlankSirLine(StringRef line)
    {
        for (char c : line)
        {
            if (c != ' ' && c != '\t' && c != '\r')
                return false;
        }
        return true;
    }
}

namespace mlir
{
    namespace ora
    {
        std::string emitSIRText(ModuleOp module)
        {
            std::string out;
            llvm::raw_string_ostream os(out);

            SmallVector<func::FuncOp, 8> funcs = collectSortedFuncs(module);

            for (size_t funcIndex = 0; funcIndex < funcs.size(); ++funcIndex)
            {
                func::FuncOp func = funcs[funcIndex];
                NameTable names;

                SmallVector<Block *, 16> blocks = collectOrderedBlocks(func, names);
                preassignNamesForFunc(func, names);

                os << "fn " << sanitizeGlobalName(func.getName()) << ":\n";
                for (Block *blockPtr : blocks)
                {
                    Block &block = *blockPtr;
                    SmallVector<Value, 8> outputs = computeBlockOutputs(block);

                    os << "    " << names.blockNames[&block];
                    if (!block.getArguments().empty())
                    {
                        os << " ";
                        emitValueList(os, names, block.getArguments());
                    }
                    if (!outputs.empty())
                    {
                        os << " -> ";
                        emitValueList(os, names, outputs);
                    }
                    os << " {\n";

                    for (Operation &op : block)
                    {
                        if (op.hasTrait<OpTrait::IsTerminator>())
                            continue;
                        std::string opText;
                        llvm::raw_string_ostream opOs(opText);
                        emitOperation(opOs, names, op);
                        opOs.flush();
                        if (opText.empty())
                            continue;
                        os << "        " << opText << "\n";
                    }

                    os << "        ";
                    emitTerminator(os, names, block);
                    os << "\n";
                    os << "    }\n";
                }
                if (funcIndex + 1 < funcs.size())
                    os << "\n";
            }

            os.flush();
            return out;
        }

        static void emitJsonString(raw_ostream &os, StringRef text)
        {
            os << "\"";
            for (char c : text)
            {
                switch (c)
                {
                case '"':
                    os << "\\\"";
                    break;
                case '\\':
                    os << "\\\\";
                    break;
                case '\n':
                    os << "\\n";
                    break;
                case '\r':
                    os << "\\r";
                    break;
                case '\t':
                    os << "\\t";
                    break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20)
                    {
                        char buf[7];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        os << buf;
                    }
                    else
                    {
                        os << c;
                    }
                    break;
                }
            }
            os << "\"";
        }

        // Helper: emit a single JSON location entry
        static void emitLocEntry(raw_ostream &os, uint32_t idx, FileLineColLoc fileLoc, bool &firstEntry)
        {
            if (!firstEntry)
                os << ",";
            os << "{\"idx\":" << idx << ",\"file\":";
            emitJsonString(os, fileLoc.getFilename());
            os << ",\"line\":" << fileLoc.getLine()
               << ",\"col\":" << fileLoc.getColumn() << "}";
            firstEntry = false;
        }

        template <typename CallbackT>
        static void visitSerializedSIROps(ModuleOp module, CallbackT callback)
        {
            uint32_t opIndex = 0;
            std::unordered_map<std::string, func::FuncOp> funcByName;
            SmallVector<func::FuncOp, 8> funcs;
            for (func::FuncOp func : module.getOps<func::FuncOp>())
            {
                funcs.push_back(func);
                funcByName.emplace(func.getName().str(), func);
            }

            std::unordered_map<Operation *, std::unique_ptr<NameTable>> nameTables;
            auto getNamesForFunc = [&](func::FuncOp func) -> NameTable & {
                auto it = nameTables.find(func.getOperation());
                if (it != nameTables.end())
                    return *it->second;

                auto names = std::make_unique<NameTable>();
                bool first = true;
                unsigned bbId = 0;
                for (Block &block : func.getBlocks())
                {
                    std::string blockName;
                    if (!block.empty())
                    {
                        if (auto attr = block.back().getAttrOfType<StringAttr>("sir.block_name"))
                            blockName = attr.getValue().str();
                    }
                    if (blockName.empty())
                    {
                        if (first)
                        {
                            blockName = "entry";
                            first = false;
                        }
                        else
                        {
                            blockName = "bb" + std::to_string(bbId++);
                        }
                    }
                    blockName = names->allocateName(blockName);
                    names->blockNames[&block] = blockName;
                }

                for (Block &block : func.getBlocks())
                {
                    for (BlockArgument arg : block.getArguments())
                        names->nameFor(arg);
                    for (Operation &op : block)
                    {
                        if (auto cst = dyn_cast<sir::ConstOp>(op))
                        {
                            if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
                            {
                                std::string key = formatConstKey(intAttr.getValue());
                                auto itConst = names->constNames.find(key);
                                if (itConst != names->constNames.end())
                                {
                                    Value defVal = itConst->second;
                                    if (auto defOp = defVal.getDefiningOp())
                                    {
                                        if (defOp->getBlock() == &block && defOp->isBeforeInBlock(&op))
                                        {
                                            names->valueNames[cst.getResult()] = names->nameFor(defVal);
                                            continue;
                                        }
                                    }
                                }
                                names->nameFor(cst.getResult());
                                names->constNames[key] = cst.getResult();
                                continue;
                            }
                        }
                        if (auto bitcast = dyn_cast<sir::BitcastOp>(op))
                        {
                            if (bitcast->getNumOperands() == 1 && bitcast->getNumResults() == 1)
                            {
                                names->valueNames[bitcast.getResult()] = names->nameFor(bitcast.getOperand());
                                continue;
                            }
                        }
                        for (Value res : op.getResults())
                            names->nameFor(res);
                    }
                }

                auto *ptr = names.get();
                nameTables.emplace(func.getOperation(), std::move(names));
                return *ptr;
            };

            auto emitSerializedOp = [&](func::FuncOp func, Block &block, NameTable &names, Operation &op) {
                if (op.hasTrait<OpTrait::IsTerminator>())
                    return;

                if (auto cst = dyn_cast<sir::ConstOp>(op))
                {
                    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
                    {
                        std::string key = formatConstKey(intAttr.getValue());
                        auto it = names.constNames.find(key);
                        if (it != names.constNames.end())
                        {
                            Value defVal = it->second;
                            if (auto defOp = defVal.getDefiningOp())
                            {
                                if (defOp->getBlock() == op.getBlock() && defOp->isBeforeInBlock(&op))
                                    return;
                            }
                        }
                    }
                }

                if (isa<sir::ConstOp>(op) && op.getNumResults() == 1)
                {
                    std::string name = names.nameFor(op.getResult(0));
                    if (names.emittedDefs.find(name) != names.emittedDefs.end())
                        return;
                    names.emittedDefs.insert(name);
                }

                if (isa<sir::BitcastOp>(op))
                    return;
                if (isa<sir::ErrorDeclOp>(op))
                    return;

                if (isa<sir::SelectOp>(op))
                {
                    std::string zeroKey = formatConstKey(APInt(256, 0));
                    bool needsZero = (names.constNames.find(zeroKey) == names.constNames.end());
                    uint32_t expandCount = needsZero ? 6 : 5;
                    for (uint32_t expandIdx = 0; expandIdx < expandCount; ++expandIdx)
                    {
                        callback(opIndex, op, func, block, names, true, expandIdx, expandCount);
                        opIndex++;
                    }
                    return;
                }

                callback(opIndex, op, func, block, names, false, 0, 1);
                opIndex++;
            };

            auto appendSuccessors = [&](Operation &term, SmallVectorImpl<std::pair<func::FuncOp, Block *>> &stack, func::FuncOp func) {
                if (auto br = dyn_cast<sir::BrOp>(term))
                {
                    stack.push_back({func, br.getDest()});
                    return;
                }
                if (auto br = dyn_cast<sir::CondBrOp>(term))
                {
                    stack.push_back({func, br.getFalseDest()});
                    stack.push_back({func, br.getTrueDest()});
                    return;
                }
                if (auto sw = dyn_cast<sir::SwitchOp>(term))
                {
                    for (Block *dest : sw.getCaseDests())
                        stack.push_back({func, dest});
                    stack.push_back({func, sw.getDefaultDest()});
                }
            };

            llvm::DenseSet<Block *> translatedBlocks;
            auto traverseRoot = [&](func::FuncOp root) {
                if (!root)
                    return;
                SmallVector<std::pair<func::FuncOp, Block *>, 16> stack;
                stack.push_back({root, &root.front()});

                while (!stack.empty())
                {
                    auto item = stack.pop_back_val();
                    func::FuncOp func = item.first;
                    Block *blockPtr = item.second;
                    if (!translatedBlocks.insert(blockPtr).second)
                        continue;

                    NameTable &names = getNamesForFunc(func);
                    Block &block = *blockPtr;
                    SmallVector<func::FuncOp, 4> callees;

                    for (Operation &op : block)
                    {
                        if (op.hasTrait<OpTrait::IsTerminator>())
                            continue;
                        emitSerializedOp(func, block, names, op);
                        if (auto icall = dyn_cast<sir::ICallOp>(op))
                        {
                            auto it = funcByName.find(icall.getCalleeAttr().getValue().str());
                            if (it != funcByName.end())
                                callees.push_back(it->second);
                        }
                    }

                    if (!block.empty())
                    {
                        Operation &term = block.back();
                        callback(opIndex, term, func, block, names, false, 0, 1);
                        opIndex++;

                        for (func::FuncOp callee : callees)
                            stack.push_back({callee, &callee.front()});
                        appendSuccessors(term, stack, func);
                    }
                }
            };

            auto initIt = funcByName.find("init");
            if (initIt != funcByName.end())
                traverseRoot(initIt->second);
            auto mainIt = funcByName.find("main");
            if (mainIt != funcByName.end())
                traverseRoot(mainIt->second);
        }

        std::string extractSIRLocations(ModuleOp module)
        {
            // Extract source locations from SIR MLIR ops as JSON.
            // CRITICAL: The op indexing must exactly match the serialized backend
            // execution order used by Sensei source-map emission. Skipped ops
            // (deduped consts, bitcasts, ErrorDeclOp) must be skipped here too,
            // and expanded ops (SelectOp -> 5-6 SIR ops) must produce matching
            // synthetic indices. The separate line map ties these backend indices
            // back to textual .sir lines.

            std::string out;
            llvm::raw_string_ostream os(out);
            os << "[";
            bool firstEntry = true;

            visitSerializedSIROps(module, [&](uint32_t opIndex, Operation &op, func::FuncOp, Block &, NameTable &, bool, uint32_t, uint32_t) {
                if (auto fileLoc = extractFileLineColLoc(op.getLoc()))
                    emitLocEntry(os, opIndex, *fileLoc, firstEntry);
            });

            os << "]";
            os.flush();
            return out;
        }

        std::string extractSIRDebugInfo(ModuleOp module)
        {
            struct DebugRecord
            {
                uint32_t idx;
                std::string op;
                std::string function;
                std::string block;
                std::optional<std::string> file;
                std::optional<uint32_t> line;
                std::optional<uint32_t> col;
                SmallVector<std::string, 4> resultNames;
                bool isTerminator = false;
                bool isSynthetic = false;
                std::optional<uint32_t> syntheticIndex;
                std::optional<uint32_t> syntheticCount;
                std::optional<std::string> syntheticPath;
                std::optional<uint32_t> statementId;
                std::optional<uint32_t> originStatementId;
                std::optional<uint32_t> executionRegionId;
                std::optional<uint32_t> statementRunIndex;
                bool isHoisted = false;
                bool isDuplicated = false;
            };

            SmallVector<DebugRecord, 128> records;
            records.reserve(256);

            visitSerializedSIROps(module, [&](uint32_t opIndex, Operation &op, func::FuncOp func, Block &block, NameTable &names, bool synthetic, uint32_t syntheticIndex, uint32_t syntheticCount) {
                DebugRecord record;
                record.idx = opIndex;
                record.op = op.getName().getStringRef().str();
                record.function = func.getName().str();
                record.block = names.blockNames[&block];
                if (auto fileLoc = extractFileLineColLoc(op.getLoc()))
                {
                    record.file = fileLoc->getFilename().str();
                    record.line = fileLoc->getLine();
                    record.col = fileLoc->getColumn();
                }
                if (auto stmtId = extractStmtIdFromLoc(op.getLoc()))
                    record.statementId = *stmtId;
                if (auto originStmtId = extractOriginStmtIdFromLoc(op.getLoc()))
                    record.originStatementId = *originStmtId;
                else if (record.statementId)
                    record.originStatementId = *record.statementId;
                record.resultNames.reserve(op.getNumResults());
                for (unsigned i = 0; i < op.getNumResults(); ++i)
                    record.resultNames.push_back(names.nameFor(op.getResult(i)));
                record.isTerminator = op.hasTrait<OpTrait::IsTerminator>();
                const auto locSyntheticTag = extractSyntheticLocTag(op.getLoc());
                const auto locSyntheticPath = extractSyntheticLocPath(op.getLoc());
                record.isSynthetic = synthetic || locSyntheticTag.has_value() || hasSyntheticLocTag(op.getLoc());
                if (synthetic)
                {
                    record.syntheticIndex = syntheticIndex;
                    record.syntheticCount = syntheticCount;
                }
                else if (locSyntheticTag)
                {
                    record.syntheticIndex = locSyntheticTag->first;
                    record.syntheticCount = locSyntheticTag->second;
                }
                if (locSyntheticPath)
                    record.syntheticPath = *locSyntheticPath;
                records.push_back(std::move(record));
            });

            auto provenanceStmtFor = [](const DebugRecord &record) -> std::optional<uint32_t> {
                if (record.statementId)
                    return record.statementId;
                return record.originStatementId;
            };

            std::unordered_map<std::string, uint32_t> prevStmtByFunction;
            std::unordered_set<std::string> prevStmtKnown;
            std::unordered_map<std::string, uint32_t> stmtRunCounts;
            for (const DebugRecord &record : records)
            {
                auto provenanceStmt = provenanceStmtFor(record);
                if (!provenanceStmt)
                    continue;
                const std::string &function = record.function;
                const uint32_t stmtId = *provenanceStmt;
                auto prevIt = prevStmtByFunction.find(function);
                if (prevIt == prevStmtByFunction.end() || !prevStmtKnown.count(function) || prevIt->second != stmtId)
                {
                    std::string key = function + "#" + std::to_string(stmtId);
                    stmtRunCounts[key] += 1;
                    prevStmtByFunction[function] = stmtId;
                    prevStmtKnown.insert(function);
                }
            }

            std::unordered_map<std::string, uint32_t> maxSeenStmtByFunction;
            prevStmtByFunction.clear();
            prevStmtKnown.clear();
            std::unordered_map<std::string, uint32_t> regionCountByFunction;
            std::unordered_map<std::string, uint32_t> currentRegionByFunction;
            std::unordered_map<std::string, uint32_t> currentRunByFunction;
            std::unordered_map<std::string, uint32_t> seenRunsByStmt;
            for (DebugRecord &record : records)
            {
                auto provenanceStmt = provenanceStmtFor(record);
                if (!provenanceStmt)
                    continue;
                const std::string &function = record.function;
                const uint32_t stmtId = *provenanceStmt;
                std::string key = function + "#" + std::to_string(stmtId);
                auto prevIt = prevStmtByFunction.find(function);
                if (prevIt == prevStmtByFunction.end() || !prevStmtKnown.count(function) || prevIt->second != stmtId)
                {
                    regionCountByFunction[function] += 1;
                    currentRegionByFunction[function] = regionCountByFunction[function];
                    seenRunsByStmt[key] += 1;
                    currentRunByFunction[function] = seenRunsByStmt[key];
                    prevStmtByFunction[function] = stmtId;
                    prevStmtKnown.insert(function);
                }
                record.executionRegionId = currentRegionByFunction[function];
                auto currentRunIt = currentRunByFunction.find(function);
                if (currentRunIt != currentRunByFunction.end())
                    record.statementRunIndex = currentRunIt->second;
                auto maxIt = maxSeenStmtByFunction.find(function);
                if (maxIt != maxSeenStmtByFunction.end() && stmtId < maxIt->second)
                {
                    record.isHoisted = true;
                }
                else
                {
                    maxSeenStmtByFunction[function] = stmtId;
                }
                auto runIt = stmtRunCounts.find(key);
                if (runIt != stmtRunCounts.end() && runIt->second > 1)
                    record.isDuplicated = true;
            }

            std::string out;
            llvm::raw_string_ostream os(out);
            os << "{\"version\":1,\"ops\":[";
            bool firstEntry = true;

            for (const DebugRecord &record : records)
            {
                if (!firstEntry)
                    os << ",";

                os << "{\"idx\":" << record.idx << ",\"op\":";
                emitJsonString(os, record.op);
                os << ",\"function\":";
                emitJsonString(os, record.function);
                os << ",\"block\":";
                emitJsonString(os, record.block);
                os << ",\"file\":";
                if (record.file)
                {
                    emitJsonString(os, *record.file);
                    os << ",\"line\":" << *record.line
                       << ",\"col\":" << *record.col;
                }
                else
                {
                    os << "null,\"line\":null,\"col\":null";
                }
                os << ",\"statement_id\":";
                if (record.statementId)
                    os << *record.statementId;
                else
                    os << "null";
                os << ",\"origin_statement_id\":";
                if (record.originStatementId)
                    os << *record.originStatementId;
                else
                    os << "null";
                if (record.executionRegionId)
                    os << ",\"execution_region_id\":" << *record.executionRegionId;
                if (record.statementRunIndex)
                    os << ",\"statement_run_index\":" << *record.statementRunIndex;
                os << ",\"result_names\":[";
                for (unsigned i = 0; i < record.resultNames.size(); ++i)
                {
                    if (i != 0)
                        os << ",";
                    emitJsonString(os, record.resultNames[i]);
                }
                os << "],\"is_terminator\":" << (record.isTerminator ? "true" : "false")
                   << ",\"is_synthetic\":" << (record.isSynthetic ? "true" : "false");
                if (record.syntheticIndex && record.syntheticCount)
                {
                    os << ",\"synthetic_index\":" << *record.syntheticIndex
                       << ",\"synthetic_count\":" << *record.syntheticCount;
                }
                if (record.syntheticPath)
                {
                    os << ",\"synthetic_path\":";
                    emitJsonString(os, *record.syntheticPath);
                }
                if (record.isHoisted)
                    os << ",\"is_hoisted\":true";
                if (record.isDuplicated)
                    os << ",\"is_duplicated\":true";
                os << "}";
                firstEntry = false;
            }

            os << "]}";
            os.flush();
            return out;
        }

        std::string extractSIRLineMap(ModuleOp module)
        {
            std::unordered_map<OpLineKey, uint32_t, OpLineKeyHash> textLineByOp;
            SmallVector<func::FuncOp, 8> funcs = collectSortedFuncs(module);
            uint32_t currentLine = 1;

            for (size_t funcIndex = 0; funcIndex < funcs.size(); ++funcIndex)
            {
                func::FuncOp func = funcs[funcIndex];
                NameTable names;
                SmallVector<Block *, 16> blocks = collectOrderedBlocks(func, names);
                preassignNamesForFunc(func, names);

                currentLine += 1; // fn header
                for (Block *blockPtr : blocks)
                {
                    Block &block = *blockPtr;
                    currentLine += 1; // block header

                    for (Operation &op : block)
                    {
                        if (op.hasTrait<OpTrait::IsTerminator>())
                            continue;

                        if (auto cst = dyn_cast<sir::ConstOp>(op))
                        {
                            if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
                            {
                                std::string key = formatConstKey(intAttr.getValue());
                                auto it = names.constNames.find(key);
                                if (it != names.constNames.end())
                                {
                                    Value defVal = it->second;
                                    if (auto defOp = defVal.getDefiningOp())
                                    {
                                        if (defOp->getBlock() == op.getBlock() && defOp->isBeforeInBlock(&op))
                                            continue;
                                    }
                                }
                            }
                        }

                        if (isa<sir::ConstOp>(op) && op.getNumResults() == 1)
                        {
                            std::string name = names.nameFor(op.getResult(0));
                            if (names.emittedDefs.find(name) != names.emittedDefs.end())
                                continue;
                            names.emittedDefs.insert(name);
                        }

                        if (isa<sir::BitcastOp>(op))
                            continue;
                        if (isa<sir::ErrorDeclOp>(op))
                            continue;

                        if (isa<sir::SelectOp>(op))
                        {
                            std::string zeroKey = formatConstKey(APInt(256, 0));
                            bool needsZero = (names.constNames.find(zeroKey) == names.constNames.end());
                            uint32_t expandCount = needsZero ? 6 : 5;
                            for (uint32_t expandIdx = 0; expandIdx < expandCount; ++expandIdx)
                            {
                                textLineByOp[{&op, expandIdx}] = currentLine;
                                currentLine += 1;
                            }
                            continue;
                        }

                        textLineByOp[{&op, 0}] = currentLine;
                        currentLine += 1;
                    }

                    if (Operation *term = block.getTerminator())
                        textLineByOp[{term, 0}] = currentLine;
                    currentLine += 1; // terminator
                    currentLine += 1; // closing brace
                }

                if (funcIndex + 1 < funcs.size())
                    currentLine += 1; // blank line between functions
            }

            const std::string sirText = emitSIRText(module);
            std::vector<StringRef> sirLines;
            sirLines.reserve(static_cast<size_t>(std::count(sirText.begin(), sirText.end(), '\n')) + 1);
            size_t lineStart = 0;
            for (size_t i = 0; i < sirText.size(); ++i)
            {
                if (sirText[i] != '\n')
                    continue;
                sirLines.emplace_back(sirText.data() + lineStart, i - lineStart);
                lineStart = i + 1;
            }
            if (lineStart <= sirText.size())
                sirLines.emplace_back(sirText.data() + lineStart, sirText.size() - lineStart);

            std::string out;
            llvm::raw_string_ostream os(out);
            os << "[";
            bool firstEntry = true;

            visitSerializedSIROps(module, [&](uint32_t opIndex, Operation &op, func::FuncOp, Block &, NameTable &, bool synthetic, uint32_t syntheticIndex, uint32_t) {
                auto it = textLineByOp.find({&op, synthetic ? syntheticIndex : 0});
                if (it == textLineByOp.end())
                    return;
                uint32_t line = it->second;
                while (line >= 1 and line <= sirLines.size() and isBlankSirLine(sirLines[line - 1]))
                    line += 1;
                if (line < 1 or line > sirLines.size())
                    line = it->second;
                if (!firstEntry)
                    os << ",";
                os << "{\"idx\":" << opIndex << ",\"line\":" << line << "}";
                firstEntry = false;
            });

            os << "]";
            os.flush();
            return out;
        }
    } // namespace ora
} // namespace mlir
