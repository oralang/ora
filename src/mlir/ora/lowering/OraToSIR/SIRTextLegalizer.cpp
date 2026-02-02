//===- SIRTextLegalizer.cpp - SIR Text Legalizer Pass ----------------===//
//
// Validates SIR MLIR against constraints required by the Sensei text format.
// This pass should be run after Ora -> SIR conversion and before text emission.
//
//===----------------------------------------------------------------------===//

#include "OraToSIR.h"

#include "SIR/SIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace mlir
{
    namespace ora
    {

        namespace
        {
            struct SIRTextLegalizerPass : public PassWrapper<SIRTextLegalizerPass, OperationPass<ModuleOp>>
            {
                void runOnOperation() override
                {
                    ModuleOp module = getOperation();
                    bool failed_any = false;

                    auto report = [&](Operation *op, const Twine &msg) {
                        op->emitError() << msg;
                        failed_any = true;
                    };

                    auto isAllowedDialect = [](Dialect *dialect) {
                        if (!dialect)
                            return false;
                        StringRef ns = dialect->getNamespace();
                        return ns == "builtin" || ns == "func" || ns == "sir";
                    };

                    module.walk([&](Operation *op) {
                        if (!isAllowedDialect(op->getDialect()))
                        {
                            report(op, Twine("illegal dialect for SIR text: ") +
                                           (op->getDialect() ? op->getDialect()->getNamespace() : "null"));
                        }
                    });

                    auto isSIRValueType = [](Type t) {
                        if (isa<sir::U256Type, sir::PtrType>(t))
                            return true;
                        if (auto intTy = dyn_cast<IntegerType>(t))
                        {
                            return intTy.getWidth() <= 256;
                        }
                        return false;
                    };

                    auto isAllowedSIRTextOp = [](Operation &op) {
                        static const llvm::StringSet<> allowed = []() {
                            llvm::StringSet<> s;
                            // Constants & intrinsics.
                            s.insert("sir.const");
                            s.insert("sir.icall");
                            s.insert("sir.noop");
                            s.insert("sir.runtime_start_offset");
                            s.insert("sir.init_end_offset");
                            s.insert("sir.runtime_length");

                            // Arithmetic.
                            s.insert("sir.add");
                            s.insert("sir.mul");
                            s.insert("sir.sub");
                            s.insert("sir.div");
                            s.insert("sir.sdiv");
                            s.insert("sir.mod");
                            s.insert("sir.smod");
                            s.insert("sir.addmod");
                            s.insert("sir.mulmod");
                            s.insert("sir.exp");
                            s.insert("sir.signextend");

                            // Comparisons / bitwise.
                            s.insert("sir.lt");
                            s.insert("sir.gt");
                            s.insert("sir.slt");
                            s.insert("sir.sgt");
                            s.insert("sir.eq");
                            s.insert("sir.iszero");
                            s.insert("sir.and");
                            s.insert("sir.or");
                            s.insert("sir.xor");
                            s.insert("sir.not");
                            s.insert("sir.byte");
                            s.insert("sir.shl");
                            s.insert("sir.shr");
                            s.insert("sir.sar");

                            // Crypto.
                            s.insert("sir.keccak256");

                            // Environment.
                            s.insert("sir.address");
                            s.insert("sir.balance");
                            s.insert("sir.origin");
                            s.insert("sir.caller");
                            s.insert("sir.callvalue");
                            s.insert("sir.calldataload");
                            s.insert("sir.calldatasize");
                            s.insert("sir.calldatacopy");
                            s.insert("sir.codesize");
                            s.insert("sir.codecopy");
                            s.insert("sir.gasprice");
                            s.insert("sir.extcodesize");
                            s.insert("sir.extcodecopy");
                            s.insert("sir.returndatasize");
                            s.insert("sir.returndatacopy");
                            s.insert("sir.extcodehash");

                            // Block information.
                            s.insert("sir.blockhash");
                            s.insert("sir.coinbase");
                            s.insert("sir.timestamp");
                            s.insert("sir.number");
                            s.insert("sir.difficulty");
                            s.insert("sir.gaslimit");
                            s.insert("sir.chainid");
                            s.insert("sir.selfbalance");
                            s.insert("sir.basefee");
                            s.insert("sir.blobhash");
                            s.insert("sir.blobbasefee");

                            // State.
                            s.insert("sir.sload");
                            s.insert("sir.sstore");
                            s.insert("sir.tload");
                            s.insert("sir.tstore");
                            s.insert("sir.gas");

                            // Logging.
                            s.insert("sir.log0");
                            s.insert("sir.log1");
                            s.insert("sir.log2");
                            s.insert("sir.log3");
                            s.insert("sir.log4");

                            // System.
                            s.insert("sir.create");
                            s.insert("sir.create2");
                            s.insert("sir.call");
                            s.insert("sir.callcode");
                            s.insert("sir.delegatecall");
                            s.insert("sir.staticcall");
                            s.insert("sir.return");
                            s.insert("sir.stop");
                            s.insert("sir.revert");
                            s.insert("sir.invalid");
                            s.insert("sir.selfdestruct");

                            // Memory.
                            s.insert("sir.malloc");
                            s.insert("sir.mallocany");
                            s.insert("sir.freeptr");
                            s.insert("sir.salloc");
                            s.insert("sir.sallocany");
                            s.insert("sir.mcopy");

                            // Control flow.
                            s.insert("sir.br");
                            s.insert("sir.cond_br");
                            s.insert("sir.switch");
                            s.insert("sir.iret");

                            // Lowered/aliasable ops.
                            s.insert("sir.addptr");
                            s.insert("sir.load");
                            s.insert("sir.store");
                            s.insert("sir.bitcast");
                            s.insert("sir.error.decl");

                            return s;
                        }();

                        return allowed.contains(op.getName().getStringRef());
                    };

                    auto verifyBranchOperands = [&](Operation *op,
                                                   Block *src,
                                                   Block *dest,
                                                   OperandRange operands,
                                                   ArrayRef<Value> expected) {
                        if (operands.size() != expected.size())
                        {
                            report(op, "branch operand count does not match other outgoing edges");
                        }
                        else
                        {
                            for (size_t i = 0; i < operands.size(); ++i)
                            {
                                if (operands[i] != expected[i])
                                {
                                    report(op, "branch operands differ between outgoing edges (SIR text requires uniform outputs)");
                                    break;
                                }
                            }
                        }

                        if (dest && dest->getNumArguments() != operands.size())
                        {
                            report(op, "successor block argument count does not match branch operand count");
                        }
                    };

                    for (func::FuncOp func : module.getOps<func::FuncOp>())
                    {
                        // Function signature must be SIR types only.
                        auto funcType = func.getFunctionType();
                        for (Type t : funcType.getInputs())
                        {
                            if (!isSIRValueType(t))
                            {
                                report(func.getOperation(), "function argument type is not a SIR type");
                            }
                        }
                        for (Type t : funcType.getResults())
                        {
                            if (!isSIRValueType(t))
                            {
                                report(func.getOperation(), "function result type is not a SIR type");
                            }
                        }

                        for (Block &block : func.getBlocks())
                        {
                            if (block.empty())
                            {
                                func.emitError() << "empty block in function: " << func.getName();
                                failed_any = true;
                                continue;
                            }

                            Operation &terminator = block.back();
                            if (!terminator.hasTrait<OpTrait::IsTerminator>())
                            {
                                report(&terminator, "block missing terminator for SIR text");
                            }

                            // Block argument types must be SIR types.
                            for (BlockArgument arg : block.getArguments())
                            {
                                if (!isSIRValueType(arg.getType()))
                                {
                                    report(&terminator, "block argument type is not a SIR type");
                                }
                            }

                            if (auto br = dyn_cast<sir::BrOp>(terminator))
                            {
                                SmallVector<Value, 8> expected(br.getDestOperands().begin(), br.getDestOperands().end());
                                verifyBranchOperands(&terminator, &block, br.getDest(), br.getDestOperands(), expected);
                            }
                            else if (auto br = dyn_cast<sir::CondBrOp>(terminator))
                            {
                                SmallVector<Value, 8> expected(br.getTrueOperands().begin(), br.getTrueOperands().end());
                                verifyBranchOperands(&terminator, &block, br.getTrueDest(), br.getTrueOperands(), expected);
                                verifyBranchOperands(&terminator, &block, br.getFalseDest(), br.getFalseOperands(), expected);
                            }
                            else if (auto sw = dyn_cast<sir::SwitchOp>(terminator))
                            {
                                auto caseVals = sw.getCaseValues();
                                if (caseVals.size() != sw.getCaseDests().size())
                                {
                                    report(&terminator, "switch caseValues count does not match caseDests count");
                                }
                                if (sw.getDefaultDest()->getNumArguments() != 0)
                                {
                                    report(&terminator, "switch default destination cannot take block arguments");
                                }
                                for (Block *dest : sw.getCaseDests())
                                {
                                    if (dest->getNumArguments() != 0)
                                    {
                                        report(&terminator, "switch case destination cannot take block arguments");
                                    }
                                }
                            }
                            else if (auto iret = dyn_cast<sir::IRetOp>(terminator))
                            {
                                (void)iret;
                            }
                            else if (isa<sir::ReturnOp, sir::StopOp, sir::RevertOp, sir::InvalidOp, sir::SelfDestructOp>(terminator))
                            {
                                // Valid SIR terminators with no additional text constraints here.
                            }
                            else
                            {
                                report(&terminator, "unsupported terminator for SIR text");
                            }

                            // For each predecessor, ensure operand count matches block arguments.
                            for (Block *pred : block.getPredecessors())
                            {
                                Operation *predTerm = pred->getTerminator();
                                if (!predTerm)
                                    continue;
                                size_t incomingSize = 0;
                                bool found = false;
                                if (auto br = dyn_cast<sir::BrOp>(predTerm))
                                {
                                    if (br.getDest() == &block)
                                    {
                                        incomingSize = br.getDestOperands().size();
                                        found = true;
                                    }
                                }
                                else if (auto br = dyn_cast<sir::CondBrOp>(predTerm))
                                {
                                    if (br.getTrueDest() == &block)
                                    {
                                        incomingSize = br.getTrueOperands().size();
                                        found = true;
                                    }
                                    else if (br.getFalseDest() == &block)
                                    {
                                        incomingSize = br.getFalseOperands().size();
                                        found = true;
                                    }
                                }
                                else if (auto sw = dyn_cast<sir::SwitchOp>(predTerm))
                                {
                                    if (sw.getDefaultDest() == &block)
                                    {
                                        incomingSize = 0;
                                        found = true;
                                    }
                                    else
                                    {
                                        for (Block *dest : sw.getCaseDests())
                                        {
                                            if (dest == &block)
                                            {
                                                incomingSize = 0;
                                                found = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (found && incomingSize != block.getNumArguments())
                                {
                                    report(predTerm, "incoming operand count does not match destination block arguments");
                                }
                            }

                            // Disallow non-SIR ops inside function bodies.
                            for (Operation &op : block)
                            {
                                Dialect *dialect = op.getDialect();
                                if (!dialect || dialect->getNamespace() != "sir")
                                {
                                    report(&op, "non-SIR operation in SIR function body");
                                    break;
                                }

                                if (isa<sir::StringConstantOp>(op))
                                {
                                    report(&op, "string.constant must be lowered to data segments before SIR text emission");
                                    break;
                                }
                                if (isa<sir::Load8Op, sir::Store8Op>(op))
                                {
                                    report(&op, "load8/store8 are not directly representable in SIR text; lower to addptr + mload/mstore");
                                    break;
                                }
                                if (!isAllowedSIRTextOp(op))
                                {
                                    report(&op, Twine("SIR op is not supported by SIR text emitter: ") +
                                                   op.getName().getStringRef());
                                    break;
                                }

                                // Validate operand/result types for SIR ops (except bitcast).
                                if (isa<sir::BitcastOp>(op))
                                    continue;

                                for (Value operand : op.getOperands())
                                {
                                    Type t = operand.getType();
                                    if (!isSIRValueType(t))
                                    {
                                        report(&op, "operand type is not a SIR type");
                                        break;
                                    }
                                }
                                for (Type t : op.getResultTypes())
                                {
                                    if (!isSIRValueType(t))
                                    {
                                        report(&op, "result type is not a SIR type");
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    module.walk([&](sir::ICallOp op) {
                        if (auto callee = op.getCalleeAttr())
                        {
                            Operation *sym = SymbolTable::lookupNearestSymbolFrom(op, callee);
                            if (!sym)
                            {
                                report(op.getOperation(), "icall callee symbol not found");
                                return;
                            }
                            if (auto func = dyn_cast<func::FuncOp>(sym))
                            {
                                auto funcType = func.getFunctionType();
                                if (funcType.getNumInputs() != op.getArgs().size())
                                {
                                    report(op.getOperation(), "icall argument count does not match callee function inputs");
                                }
                                if (funcType.getNumResults() != op.getResults().size())
                                {
                                    // Repair mismatched icall result counts by rebuilding the call.
                                    OpBuilder b(op);
                                    SmallVector<Type, 4> newResults;
                                    newResults.append(funcType.getResults().begin(), funcType.getResults().end());
                                    auto newCall = b.create<sir::ICallOp>(op.getLoc(), newResults, op.getCalleeAttr(), op.getArgs());
                                    unsigned common = std::min(op.getNumResults(), newCall.getNumResults());
                                    for (unsigned i = 0; i < common; ++i)
                                    {
                                        Value oldRes = op.getResult(i);
                                        Value newRes = newCall.getResult(i);
                                        if (oldRes.getType() == newRes.getType())
                                        {
                                            oldRes.replaceAllUsesWith(newRes);
                                        }
                                        else if (isa<sir::PtrType>(oldRes.getType()) && isa<sir::U256Type>(newRes.getType()))
                                        {
                                            auto bc = b.create<sir::BitcastOp>(op.getLoc(), oldRes.getType(), newRes);
                                            oldRes.replaceAllUsesWith(bc.getResult());
                                        }
                                        else if (isa<sir::U256Type>(oldRes.getType()) && isa<sir::PtrType>(newRes.getType()))
                                        {
                                            auto bc = b.create<sir::BitcastOp>(op.getLoc(), oldRes.getType(), newRes);
                                            oldRes.replaceAllUsesWith(bc.getResult());
                                        }
                                        else
                                        {
                                            oldRes.replaceAllUsesWith(newRes);
                                        }
                                    }
                                    if (op.getNumResults() > newCall.getNumResults())
                                    {
                                        auto u256 = sir::U256Type::get(op.getContext());
                                        auto zero = b.create<sir::ConstOp>(op.getLoc(), u256,
                                                                           IntegerAttr::get(b.getI64Type(), 0));
                                        for (unsigned i = common; i < op.getNumResults(); ++i)
                                        {
                                            Value oldRes = op.getResult(i);
                                            if (isa<sir::PtrType>(oldRes.getType()))
                                            {
                                                auto bc = b.create<sir::BitcastOp>(op.getLoc(), oldRes.getType(), zero);
                                                oldRes.replaceAllUsesWith(bc.getResult());
                                            }
                                            else
                                            {
                                                oldRes.replaceAllUsesWith(zero);
                                            }
                                        }
                                    }
                                    op.erase();
                                }
                            }
                            else
                            {
                                report(op.getOperation(), "icall callee is not a func.func");
                            }
                        }
                    });

                    if (failed_any)
                        signalPassFailure();
                }
            };
        } // namespace

        std::unique_ptr<Pass> createSIRTextLegalizerPass()
        {
            return std::make_unique<SIRTextLegalizerPass>();
        }

    } // namespace ora
} // namespace mlir
