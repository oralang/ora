//===- SIROps.td ------------------------------------------------*- tablegen -*-===//

#ifndef SIR_OPS
#define SIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "SIRDialect.td"
include "SIRTypes.td"

// --------------------------------------------------------------------------
// Base op definition for all SIR ops
// --------------------------------------------------------------------------

// Base class for SIR operations (default - may have side effects)
class SIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<SIR_Dialect, mnemonic, traits>;

// Base class for pure SIR operations (no side effects - can be removed by DCE)
// Pure = AlwaysSpeculatable + NoMemoryEffect (for canonicalization)
// Explicitly include NoMemoryEffect for DCE (even though Pure includes it)
// This ensures DCE can see the MemoryEffects<[]> interface
class SIR_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<SIR_Dialect, mnemonic, !listconcat(traits, [Pure, NoMemoryEffect])>;

// Common `lhs, rhs -> res` word operation shape.
class SIR_BinaryWordOp<string mnemonic, string summaryText> :
    SIR_PureOp<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

class SIR_FoldableBinaryWordOp<string mnemonic, string summaryText,
                               bit canonicalizer = 0> :
    SIR_BinaryWordOp<mnemonic, summaryText> {
  let hasFolder = 1;
  let hasCanonicalizer = canonicalizer;
}

class SIR_NamedFoldableBinaryWordOp<string mnemonic, string summaryText> :
    SIR_PureOp<mnemonic, [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

class SIR_FoldableUnaryWordOp<string mnemonic, string summaryText> :
    SIR_PureOp<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$x);
  let results   = (outs SIR_U256:$res);
  let hasFolder = 1;
  let assemblyFormat =
    "$x `:` type($x) attr-dict `:` type($res)";
}

class SIR_FoldableShiftWordOp<string mnemonic, string summaryText> :
    SIR_PureOp<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$shift, SIR_U256:$x);
  let results   = (outs SIR_U256:$res);
  let hasFolder = 1;
  let assemblyFormat =
    "$shift `:` type($shift) `,` $x `:` type($x) attr-dict `:` type($res)";
}

class SIR_ModularWordOp<string mnemonic, string summaryText> :
    SIR_PureOp<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$a, SIR_U256:$b, SIR_U256:$n);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$a `:` type($a) `,` $b `:` type($b) `,` $n `:` type($n) "
    "attr-dict `:` type($res)";
}

class SIR_NamedBinaryWordOp<string mnemonic, string summaryText,
                            dag operands, string format,
                            bit folder = 0> :
    SIR_PureOp<mnemonic> {
  let summary = summaryText;
  let arguments = operands;
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = format;
  let hasFolder = folder;
}

// Common nullary EVM-word operation shape. Kept as SIR_Op, not SIR_PureOp:
// environment reads are not modeled as pure arithmetic.
class SIR_NullaryWordOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

class SIR_NullaryPtrOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins);
  let results   = (outs SIR_Ptr:$ptr);
  let assemblyFormat = "attr-dict `:` type($ptr)";
}

class SIR_NullaryOffsetOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins);
  let results   = (outs SIR_U256:$offset);
  let assemblyFormat = "attr-dict `:` type($offset)";
}

class SIR_NullaryLenOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins);
  let results   = (outs SIR_U256:$len);
  let assemblyFormat = "attr-dict `:` type($len)";
}

class SIR_SizeToPtrOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$size);
  let results   = (outs SIR_Ptr:$ptr);
  let assemblyFormat =
    "$size `:` type($size) attr-dict `:` type($ptr)";
}

class SIR_SlotWordWriteOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$slot, SIR_U256:$val);
  let assemblyFormat =
    "$slot `:` type($slot) `,` $val `:` type($val) attr-dict";
}

class SIR_SlotOnlyOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$slot);
  let assemblyFormat = "$slot `:` type($slot) attr-dict";
}

class SIR_AddressQueryOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$addr);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$addr `:` type($addr) attr-dict `:` type($res)";
}

class SIR_U256ToU256Op<string mnemonic, string summaryText,
                       dag operands, string format> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = operands;
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = format;
}

class SIR_AttrOnlyOp<string mnemonic, string summaryText,
                     list<Trait> traits = []> :
    SIR_Op<mnemonic, traits> {
  let summary = summaryText;
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

class SIR_CopyToPtrOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_Ptr:$dst,
                       SIR_U256:$offset,
                       SIR_U256:$len);
  let assemblyFormat =
    "$dst `:` type($dst) `,` $offset `:` type($offset) `,` $len `:` type($len) "
    "attr-dict";
}

class SIR_ValueCallOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$gas,
                       SIR_U256:$addr,
                       SIR_U256:$value,
                       SIR_Ptr:$argsPtr,
                       SIR_U256:$argsLen,
                       SIR_Ptr:$retPtr,
                       SIR_U256:$retLen);
  let results   = (outs SIR_U256:$success);
  let assemblyFormat =
    "$gas `:` type($gas) `,` $addr `:` type($addr) `,` $value `:` type($value) "
    "`,` $argsPtr `:` type($argsPtr) `,` $argsLen `:` type($argsLen) `,` "
    "$retPtr `:` type($retPtr) `,` $retLen `:` type($retLen) "
    "attr-dict `:` type($success)";
}

class SIR_NoValueCallOp<string mnemonic, string summaryText> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins SIR_U256:$gas,
                       SIR_U256:$addr,
                       SIR_Ptr:$argsPtr,
                       SIR_U256:$argsLen,
                       SIR_Ptr:$retPtr,
                       SIR_U256:$retLen);
  let results   = (outs SIR_U256:$success);
  let assemblyFormat =
    "$gas `:` type($gas) `,` $addr `:` type($addr) `,` "
    "$argsPtr `:` type($argsPtr) `,` $argsLen `:` type($argsLen) `,` "
    "$retPtr `:` type($retPtr) `,` $retLen `:` type($retLen) "
    "attr-dict `:` type($success)";
}

class SIR_PtrLenOp<string mnemonic, string summaryText,
                   list<Trait> traits = []> :
    SIR_Op<mnemonic, traits> {
  let summary = summaryText;
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$len);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $len `:` type($len) attr-dict";
}

class SIR_PtrLenTerminatorOp<string mnemonic, string summaryText> :
    SIR_PtrLenOp<mnemonic, summaryText, [Terminator]>;

class SIR_LogTopicOp<string mnemonic, string summaryText,
                     dag operands, string format> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = operands;
  let assemblyFormat = format;
}

class SIR_CreateLikeOp<string mnemonic, string summaryText,
                       dag operands, string format> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = operands;
  let results   = (outs SIR_U256:$addr);
  let assemblyFormat = format;
}

class SIR_MemoryReadOp<string mnemonic, string summaryText,
                       dag operands, string format> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = operands;
  let results   = (outs SIR_U256:$val);
  let assemblyFormat = format;
}

class SIR_MemoryWriteOp<string mnemonic, string summaryText,
                        dag operands, string format> :
    SIR_Op<mnemonic> {
  let summary = summaryText;
  let arguments = operands;
  let assemblyFormat = format;
}

class SIR_NoResultFormattedOp<string mnemonic, string summaryText,
                              dag operands, string format,
                              list<Trait> traits = []> :
    SIR_Op<mnemonic, traits> {
  let summary = summaryText;
  let arguments = operands;
  let assemblyFormat = format;
}

// ==========================================================================
// Constants
// ==========================================================================
def ConstOp : SIR_PureOp<"const", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Constant 256-bit value";
  let arguments = (ins APIntAttr:$value);
  let results   = (outs SIR_U256:$res);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;

  let builders = [
    // Convenience builder for small constants (fits in uint64_t).
    OpBuilder<(ins "uint64_t":$value), [{
      auto u256Type = ::mlir::IntegerType::get($_builder.getContext(), 256);
      $_state.addAttribute("value", ::mlir::IntegerAttr::get(u256Type, ::llvm::APInt(256, value)));
      $_state.addTypes(sir::U256Type::get($_builder.getContext()));
    }]>,
    // Full-width builder for arbitrary 256-bit constants.
    OpBuilder<(ins "const ::llvm::APInt &":$value), [{
      auto u256Type = ::mlir::IntegerType::get($_builder.getContext(), 256);
      ::llvm::APInt val = value.getBitWidth() == 256 ? value : value.zext(256);
      $_state.addAttribute("value", ::mlir::IntegerAttr::get(u256Type, val));
      $_state.addTypes(sir::U256Type::get($_builder.getContext()));
    }]>,
    // Builder accepting an IntegerAttr directly.
    OpBuilder<(ins "::mlir::IntegerAttr":$value), [{
      $_state.addAttribute("value", value);
      $_state.addTypes(sir::U256Type::get($_builder.getContext()));
    }]>
  ];
}

def StringConstantOp : SIR_Op<"string.constant"> {
  let summary = "String constant";
  let arguments = (ins StrAttr:$value);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "$value attr-dict `:` type($res)";
}

// ==========================================================================
// Metadata
// ==========================================================================

def ErrorDeclOp : SIR_NoResultFormattedOp<"error.decl", "Error declaration metadata",
    (ins StrAttr:$sym_name), "attr-dict", [NoMemoryEffect]>;

// ==========================================================================
// Arithmetic Operations (binary, word-wise)
// ==========================================================================

def AddOp : SIR_NamedFoldableBinaryWordOp<"add", "Add two 256-bit values">;

def SubOp : SIR_FoldableBinaryWordOp<"sub", "Subtract two 256-bit values", 1>;

def MulOp : SIR_NamedFoldableBinaryWordOp<"mul", "Multiply two 256-bit values">;

def DivOp : SIR_BinaryWordOp<"div", "Unsigned division">;

def SDivOp : SIR_BinaryWordOp<"sdiv", "Signed division">;

def ModOp : SIR_BinaryWordOp<"mod", "Unsigned modulo">;

def SModOp : SIR_BinaryWordOp<"smod", "Signed modulo">;

def AddModOp : SIR_ModularWordOp<"addmod", "Modular addition">;

def MulModOp : SIR_ModularWordOp<"mulmod", "Modular multiplication">;

def ExpOp : SIR_NamedBinaryWordOp<"exp", "Exponentiation",
    (ins SIR_U256:$base, SIR_U256:$exp),
    "$base `:` type($base) `,` $exp `:` type($exp) attr-dict `:` type($res)">;

def SignExtendOp : SIR_NamedBinaryWordOp<"signextend", "Sign extend",
    (ins SIR_U256:$b, SIR_U256:$x),
    "$b `:` type($b) `,` $x `:` type($x) attr-dict `:` type($res)">;

// ==========================================================================
// Comparison Operations
// ==========================================================================

def LtOp : SIR_FoldableBinaryWordOp<"lt", "Unsigned less-than", 1>;

def GtOp : SIR_FoldableBinaryWordOp<"gt", "Unsigned greater-than", 1>;

def SLtOp : SIR_FoldableBinaryWordOp<"slt", "Signed less-than comparison", 1>;

def SGtOp : SIR_FoldableBinaryWordOp<"sgt", "Signed greater-than comparison", 1>;

def EqOp : SIR_FoldableBinaryWordOp<"eq", "Equality comparison">;

def IsZeroOp : SIR_FoldableUnaryWordOp<"iszero", "Zero check">;

def SelectOp : SIR_PureOp<"select"> {
  let summary = "Select between two values";
  let arguments = (ins SIR_U256:$cond, SIR_U256:$true_value, SIR_U256:$false_value);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$cond `:` type($cond) `,` $true_value `:` type($true_value) `,` "
    "$false_value `:` type($false_value) attr-dict `:` type($res)";
}

// ==========================================================================
// Bitwise Operations
// ==========================================================================

def AndOp : SIR_FoldableBinaryWordOp<"and", "Bitwise AND", 1>;

def OrOp : SIR_FoldableBinaryWordOp<"or", "Bitwise OR", 1>;

def XorOp : SIR_FoldableBinaryWordOp<"xor", "Bitwise XOR", 1>;

def NotOp : SIR_FoldableUnaryWordOp<"not", "Bitwise NOT">;

def ByteOp : SIR_NamedBinaryWordOp<"byte", "BYTE opcode",
    (ins SIR_U256:$i, SIR_U256:$x),
    "$i `:` type($i) `,` $x `:` type($x) attr-dict `:` type($res)", 1>;

def ShlOp : SIR_FoldableShiftWordOp<"shl", "Logical shift left">;

def ShrOp : SIR_FoldableShiftWordOp<"shr", "Logical shift right">;

def SarOp : SIR_FoldableShiftWordOp<"sar", "Arithmetic shift right">;

// ==========================================================================
// Pointer Arithmetic
// ==========================================================================

def AddPtrOp : SIR_PureOp<"addptr", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Pointer arithmetic";
  let arguments = (ins SIR_Ptr:$base, SIR_U256:$offset);
  let results   = (outs SIR_Ptr:$res);
  let hasCustomAssemblyFormat = 1;
}

// ==========================================================================
// Generic Memory Loads & Stores (word-based)
// ==========================================================================

def LoadOp : SIR_MemoryReadOp<"load", "Load a 256-bit value from a pointer",
    (ins SIR_Ptr:$ptr),
    "$ptr `:` type($ptr) attr-dict `:` type($val)">;

def StoreOp : SIR_MemoryWriteOp<"store", "Store a 256-bit value into a pointer",
    (ins SIR_Ptr:$ptr, SIR_U256:$val),
    "$ptr `:` type($ptr) `,` $val `:` type($val) attr-dict">;

def Load8Op : SIR_MemoryReadOp<"load8", "Load a byte from memory",
    (ins SIR_Ptr:$ptr, SIR_U256:$offset),
    "$ptr `:` type($ptr) `,` $offset `:` type($offset) "
    "attr-dict `:` type($val)">;

def Store8Op : SIR_MemoryWriteOp<"store8", "Store a byte to memory",
    (ins SIR_Ptr:$ptr, SIR_U256:$offset, SIR_U256:$val),
    "$ptr `:` type($ptr) `,` $offset `:` type($offset) `,` "
    "$val `:` type($val) attr-dict">;

// ==========================================================================
// CallData
// ==========================================================================

def CallDataLoadOp : SIR_U256ToU256Op<"calldataload", "Load from calldata",
    (ins SIR_U256:$offset),
    "$offset `:` type($offset) attr-dict `:` type($res)">;

def CallDataSizeOp : SIR_NullaryWordOp<"calldatasize", "Calldata size">;

def CallDataCopyOp : SIR_CopyToPtrOp<"calldatacopy", "Copy from calldata to memory">;

// ==========================================================================
// Memory Allocation & Management
// ==========================================================================

def MallocOp : SIR_Op<"malloc", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Allocate memory";
  let arguments = (ins SIR_U256:$size);
  let results   = (outs SIR_Ptr:$ptr);
  let hasCustomAssemblyFormat = 1;
}

def MallocAnyOp : SIR_SizeToPtrOp<"mallocany", "Allocate memory (unbounded)">;

def SAllocOp : SIR_SizeToPtrOp<"salloc", "Static allocation">;

def SAllocAnyOp : SIR_SizeToPtrOp<"sallocany", "Static allocation (dynamic size)">;

def FreePtrOp : SIR_NullaryPtrOp<"freeptr", "Get or advance free memory pointer">;

def MCopyOp : SIR_NoResultFormattedOp<"mcopy", "Copy memory region",
    (ins SIR_Ptr:$dst, SIR_Ptr:$src, SIR_U256:$len),
    "$dst `:` type($dst) `,` $src `:` type($src) `,` "
    "$len `:` type($len) attr-dict">;

// ==========================================================================
// Crypto
// ==========================================================================

def KeccakOp : SIR_PureOp<"keccak256", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Compute keccak256";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$len);
  let results   = (outs SIR_U256:$hash);
  let hasCustomAssemblyFormat = 1;
}

// ==========================================================================
// Storage & Transient Storage & Gas
// ==========================================================================

def SStoreOp : SIR_SlotWordWriteOp<"sstore", "Write storage slot">;

def SLoadOp : SIR_Op<"sload", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Read storage slot";
  let arguments = (ins SIR_U256:$slot);
  let results   = (outs SIR_U256:$res);
  let hasCustomAssemblyFormat = 1;
}

def TStoreOp : SIR_SlotWordWriteOp<"tstore", "Write transient storage slot">;

def TLoadOp : SIR_U256ToU256Op<"tload", "Read transient storage slot",
    (ins SIR_U256:$slot),
    "$slot `:` type($slot) attr-dict `:` type($res)">;

def TStoreLockOp : SIR_SlotOnlyOp<"tstore.lock", "Add TStore slot to locked set (tx-scoped)">;

def TStoreUnlockOp : SIR_SlotOnlyOp<"tstore.unlock", "Remove TStore slot from locked set">;

def TStoreGuardOp : SIR_SlotOnlyOp<"tstore.guard", "Revert if TStore slot is in locked set">;

def GasOp : SIR_NullaryWordOp<"gas", "Remaining gas">;

// ==========================================================================
// Environment & Block Info
// ==========================================================================

def AddressOp : SIR_NullaryWordOp<"address", "Current address">;

def BalanceOp : SIR_AddressQueryOp<"balance", "Account balance">;

def OriginOp : SIR_NullaryWordOp<"origin", "Transaction origin">;

def CallerOp : SIR_NullaryWordOp<"caller", "Message caller">;

def CallValueOp : SIR_NullaryWordOp<"callvalue", "Call value">;

def CodeSizeOp : SIR_NullaryWordOp<"codesize", "Current code size">;

def CodeCopyOp : SIR_CopyToPtrOp<"codecopy", "Copy from code to memory">;

def GasPriceOp : SIR_NullaryWordOp<"gasprice", "Gas price">;

def ExtCodeSizeOp : SIR_AddressQueryOp<"extcodesize", "External account code size">;

def ExtCodeCopyOp : SIR_NoResultFormattedOp<"extcodecopy",
    "Copy external account code to memory",
    (ins SIR_U256:$addr, SIR_Ptr:$dst, SIR_U256:$offset, SIR_U256:$len),
    "$addr `:` type($addr) `,` $dst `:` type($dst) `,` "
    "$offset `:` type($offset) `,` $len `:` type($len) attr-dict">;

def ReturnDataSizeOp : SIR_NullaryWordOp<"returndatasize", "returndatasize">;

def ReturnDataCopyOp : SIR_CopyToPtrOp<"returndatacopy", "Copy returndata to memory">;

def ExtCodeHashOp : SIR_AddressQueryOp<"extcodehash", "External account code hash">;

// Block info: blockhash, coinbase, timestamp, number, difficulty, gaslimit,
// chainid, selfbalance, basefee, blobhash, blobbasefee.

def BlockHashOp : SIR_U256ToU256Op<"blockhash", "BLOCKHASH opcode",
    (ins SIR_U256:$blockNumber),
    "$blockNumber `:` type($blockNumber) attr-dict `:` type($res)">;

def CoinbaseOp : SIR_NullaryWordOp<"coinbase", "COINBASE">;

def TimestampOp : SIR_NullaryWordOp<"timestamp", "TIMESTAMP">;

def NumberOp : SIR_NullaryWordOp<"number", "NUMBER">;

def DifficultyOp : SIR_NullaryWordOp<"difficulty", "DIFFICULTY / PREVRANDAO">;

def GasLimitOp : SIR_NullaryWordOp<"gaslimit", "GASLIMIT">;

def ChainIdOp : SIR_NullaryWordOp<"chainid", "CHAINID">;

def SelfBalanceOp : SIR_NullaryWordOp<"selfbalance", "SELFBALANCE">;

def BaseFeeOp : SIR_NullaryWordOp<"basefee", "BASEFEE">;

def BlobHashOp : SIR_U256ToU256Op<"blobhash", "BLOBHASH",
    (ins SIR_U256:$index),
    "$index `:` type($index) attr-dict `:` type($res)">;

def BlobBaseFeeOp : SIR_NullaryWordOp<"blobbasefee", "BLOBBASEFEE">;

// ==========================================================================
// Logging
// ==========================================================================

def Log0Op : SIR_PtrLenOp<"log0", "LOG0">;

def Log1Op : SIR_LogTopicOp<"log1", "LOG1",
    (ins SIR_Ptr:$ptr, SIR_U256:$len, SIR_U256:$topic0),
    "$ptr `:` type($ptr) `,` $len `:` type($len) `,` "
    "$topic0 `:` type($topic0) attr-dict">;

def Log2Op : SIR_LogTopicOp<"log2", "LOG2",
    (ins SIR_Ptr:$ptr, SIR_U256:$len, SIR_U256:$topic0, SIR_U256:$topic1),
    "$ptr `:` type($ptr) `,` $len `:` type($len) `,` "
    "$topic0 `:` type($topic0) `,` $topic1 `:` type($topic1) attr-dict">;

def Log3Op : SIR_LogTopicOp<"log3", "LOG3",
    (ins SIR_Ptr:$ptr, SIR_U256:$len, SIR_U256:$topic0, SIR_U256:$topic1,
         SIR_U256:$topic2),
    "$ptr `:` type($ptr) `,` $len `:` type($len) `,` "
    "$topic0 `:` type($topic0) `,` $topic1 `:` type($topic1) `,` "
    "$topic2 `:` type($topic2) attr-dict">;

def Log4Op : SIR_LogTopicOp<"log4", "LOG4",
    (ins SIR_Ptr:$ptr, SIR_U256:$len, SIR_U256:$topic0, SIR_U256:$topic1,
         SIR_U256:$topic2, SIR_U256:$topic3),
    "$ptr `:` type($ptr) `,` $len `:` type($len) `,` "
    "$topic0 `:` type($topic0) `,` $topic1 `:` type($topic1) `,` "
    "$topic2 `:` type($topic2) `,` $topic3 `:` type($topic3) attr-dict">;

// ==========================================================================
// System / Call / Create / Termination
// ==========================================================================

def CreateOp : SIR_CreateLikeOp<"create", "CREATE",
    (ins SIR_U256:$value, SIR_Ptr:$ptr, SIR_U256:$len),
    "$value `:` type($value) `,` $ptr `:` type($ptr) `,` "
    "$len `:` type($len) attr-dict `:` type($addr)">;

def Create2Op : SIR_CreateLikeOp<"create2", "CREATE2",
    (ins SIR_U256:$value, SIR_Ptr:$ptr, SIR_U256:$len, SIR_U256:$salt),
    "$value `:` type($value) `,` $ptr `:` type($ptr) `,` "
    "$len `:` type($len) `,` $salt `:` type($salt) "
    "attr-dict `:` type($addr)">;

// Approximate signature for calls: gas, addr, value, argsPtr, argsLen,
// retPtr, retLen -> success (0/1).

def CallOp : SIR_ValueCallOp<"call", "CALL">;

def CallCodeOp : SIR_ValueCallOp<"callcode", "CALLCODE">;

def DelegateCallOp : SIR_NoValueCallOp<"delegatecall", "DELEGATECALL">;

def StaticCallOp : SIR_NoValueCallOp<"staticcall", "STATICCALL">;

// Return & termination ops.
// You already have a ReturnOp, we keep it and add stop/revert/invalid/selfdestruct.

// CFG ops.

def BrOp : SIR_Op<"br", [Terminator]> {
  let summary = "Unconditional branch";
  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);
  let hasCanonicalizer = 1;
  let assemblyFormat =
    "$dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict";
}

def CondBrOp : SIR_Op<"cond_br", [Terminator, AttrSizedOperandSegments]> {
  let summary = "Conditional branch";
  let arguments = (ins SIR_U256:$cond,
                       Variadic<AnyType>:$trueOperands,
                       Variadic<AnyType>:$falseOperands);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let hasCanonicalizer = 1;
  let assemblyFormat =
    "$cond `:` type($cond) `,` $trueDest (`(` $trueOperands^ `:` type($trueOperands) `)`)? `,` "
    "$falseDest (`(` $falseOperands^ `:` type($falseOperands) `)`)? attr-dict";
}

def SwitchOp : SIR_Op<"switch", [Terminator]> {
  let summary = "Switch on selector";
  let arguments = (ins SIR_U256:$selector, I64ArrayAttr:$caseValues);
  let successors = (successor AnySuccessor:$defaultDest, VariadicSuccessor<AnySuccessor>:$caseDests);
  let assemblyFormat =
    "$selector `:` type($selector) `,` `[` $caseValues `]` `,` $caseDests `,` $defaultDest attr-dict";
}

def ReturnOp : SIR_PtrLenTerminatorOp<"return", "Return a pointer + size">;

def StopOp : SIR_AttrOnlyOp<"stop", "STOP", [Terminator]>;

def RevertOp : SIR_PtrLenTerminatorOp<"revert", "REVERT">;

def InvalidOp : SIR_AttrOnlyOp<"invalid", "INVALID", [Terminator]>;

def SelfDestructOp : SIR_NoResultFormattedOp<"selfdestruct", "SELFDESTRUCT",
    (ins SIR_U256:$beneficiary),
    "$beneficiary `:` type($beneficiary) attr-dict", [Terminator]>;

// ==========================================================================
// Intrinsics: runtime_start_offset, init_end_offset, runtime_length, icall, noop
// ==========================================================================

def RuntimeStartOffsetOp :
    SIR_NullaryOffsetOp<"runtime_start_offset", "runtime_start_offset()">;

def InitEndOffsetOp :
    SIR_NullaryOffsetOp<"init_end_offset", "init_end_offset()">;

def RuntimeLengthOp :
    SIR_NullaryLenOp<"runtime_length", "runtime_length()">;

// icall(fn, ...) -> ...
// For now we model it with a symbol ref and variadic operands / results.

def ICallOp : SIR_Op<"icall"> {
  let summary = "Internal function call";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<SIR_U256Type>:$args);
  let results   = (outs Variadic<SIR_U256Type>:$results);
  let assemblyFormat =
    "$callee `(` $args `)` attr-dict `:` type($results)";
}

def NoOpOp : SIR_AttrOnlyOp<"noop", "No-op">;

// ==========================================================================
// Internal Return
// ==========================================================================

def IRetOp : SIR_NoResultFormattedOp<"iret", "Internal function return",
    (ins Variadic<SIR_U256Type>:$values), "$values attr-dict", [Terminator]>;

// ==========================================================================
// Type Conversion / Bitcast (Materialization)
// ==========================================================================

def BitcastOp : SIR_PureOp<"bitcast"> {
  let summary = "Bitcast between compatible types";
  let description = [{
    Bitcast operation for type conversion. Used for materialization during
    dialect conversion to bridge between compatible types (e.g., u256 <-> ora.int<256>).
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let hasFolder = 1;
  let assemblyFormat =
    "$input `:` type($input) attr-dict `:` type($output)";
}

#endif // SIR_OPS
