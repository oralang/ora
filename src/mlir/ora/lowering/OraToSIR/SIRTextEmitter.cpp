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
#include <unordered_set>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Casting.h"

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

    std::string mnemonicFor(Operation &op)
    {
        StringRef name = op.getName().getStringRef();
        if (name.starts_with("sir."))
            return name.drop_front(4).str();
        return name.str();
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
            os << " @" << ic.getCalleeAttr().getValue();
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
}

namespace mlir
{
    namespace ora
    {
        std::string emitSIRText(ModuleOp module)
        {
            std::string out;
            llvm::raw_string_ostream os(out);

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

            for (size_t funcIndex = 0; funcIndex < funcs.size(); ++funcIndex)
            {
                func::FuncOp func = funcs[funcIndex];
                NameTable names;

                // Assign block names in order. First block is "entry" unless overridden.
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
                    // Route through allocateName for collision detection.
                    blockName = names.allocateName(blockName);
                    names.blockNames[&block] = blockName;
                    blocks.push_back(&block);
                }

                // If block order attributes are present, sort accordingly (used by dispatcher).
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

                // Pre-assign names for all values (block args, op results).
                // Also perform const dedup and bitcast aliasing so that
                // block output headers use the final names.
                for (Block &block : func.getBlocks())
                {
                    for (BlockArgument arg : block.getArguments())
                    {
                        names.nameFor(arg);
                    }
                    for (Operation &op : block)
                    {
                        // Const dedup: alias duplicate consts to the first definition.
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
                                // First occurrence in this block — assign name and register.
                                names.nameFor(cst.getResult());
                                names.constNames[key] = cst.getResult();
                                continue;
                            }
                        }
                        // Bitcast aliasing: alias result to operand.
                        if (auto bitcast = dyn_cast<sir::BitcastOp>(op))
                        {
                            if (bitcast->getNumOperands() == 1 && bitcast->getNumResults() == 1)
                            {
                                names.valueNames[bitcast.getResult()] = names.nameFor(bitcast.getOperand());
                                continue;
                            }
                        }
                        for (Value res : op.getResults())
                        {
                            names.nameFor(res);
                        }
                    }
                }

                os << "fn " << func.getName() << ":\n";
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
                        os << "        ";
                        emitOperation(os, names, op);
                        os << "\n";
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
    } // namespace ora
} // namespace mlir
