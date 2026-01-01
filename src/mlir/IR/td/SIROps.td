//===- SIROps.td ------------------------------------------------*- tablegen -*-===//

#ifndef SIR_OPS
#define SIR_OPS

include "mlir/IR/OpBase.td"
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

// ==========================================================================
// Constants
// ==========================================================================
def ConstOp : SIR_PureOp<"const", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Constant 256-bit value";
  let arguments = (ins UI64Attr:$value);
  let results   = (outs SIR_U256:$res);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "uint64_t":$value), [{
      auto ui64Type = IntegerType::get($_builder.getContext(), 64,
                                       IntegerType::Unsigned);
      $_state.addAttribute("value", IntegerAttr::get(ui64Type, value));
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
// Arithmetic Operations (binary, word-wise)
// ==========================================================================

def AddOp : SIR_PureOp<"add", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Add two 256-bit values";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

def SubOp : SIR_PureOp<"sub"> {
  let summary = "Subtract two 256-bit values";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def MulOp : SIR_PureOp<"mul", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Multiply two 256-bit values";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

def DivOp : SIR_PureOp<"div"> {
  let summary = "Unsigned division";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def SDivOp : SIR_PureOp<"sdiv"> {
  let summary = "Signed division";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def ModOp : SIR_PureOp<"mod"> {
  let summary = "Unsigned modulo";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def SModOp : SIR_PureOp<"smod"> {
  let summary = "Signed modulo";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def AddModOp : SIR_PureOp<"addmod"> {
  let summary = "Modular addition";
  let arguments = (ins SIR_U256:$a, SIR_U256:$b, SIR_U256:$n);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$a `:` type($a) `,` $b `:` type($b) `,` $n `:` type($n) "
    "attr-dict `:` type($res)";
}

def MulModOp : SIR_PureOp<"mulmod"> {
  let summary = "Modular multiplication";
  let arguments = (ins SIR_U256:$a, SIR_U256:$b, SIR_U256:$n);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$a `:` type($a) `,` $b `:` type($b) `,` $n `:` type($n) "
    "attr-dict `:` type($res)";
}

def ExpOp : SIR_PureOp<"exp"> {
  let summary = "Exponentiation";
  let arguments = (ins SIR_U256:$base, SIR_U256:$exp);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$base `:` type($base) `,` $exp `:` type($exp) attr-dict `:` type($res)";
}

def SignExtendOp : SIR_Op<"signextend"> {
  let summary = "Sign extend";
  let arguments = (ins SIR_U256:$b, SIR_U256:$x);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$b `:` type($b) `,` $x `:` type($x) attr-dict `:` type($res)";
}

// ==========================================================================
// Comparison Operations
// ==========================================================================

def LtOp : SIR_Op<"lt"> {
  let summary = "Unsigned less-than";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res); // 0 or 1
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def GtOp : SIR_Op<"gt"> {
  let summary = "Unsigned greater-than";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def SLtOp : SIR_Op<"slt"> {
  let summary = "Signed less-than comparison";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def SGtOp : SIR_Op<"sgt"> {
  let summary = "Signed greater-than comparison";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def EqOp : SIR_Op<"eq"> {
  let summary = "Equality comparison";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res); // 0 or 1
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def IsZeroOp : SIR_Op<"iszero"> {
  let summary = "Zero check";
  let arguments = (ins SIR_U256:$x);
  let results   = (outs SIR_U256:$res); // 0 or 1
  let assemblyFormat =
    "$x `:` type($x) attr-dict `:` type($res)";
}

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

def AndOp : SIR_Op<"and"> {
  let summary = "Bitwise AND";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def OrOp : SIR_Op<"or"> {
  let summary = "Bitwise OR";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def XorOp : SIR_Op<"xor"> {
  let summary = "Bitwise XOR";
  let arguments = (ins SIR_U256:$lhs, SIR_U256:$rhs);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `:` type($res)";
}

def NotOp : SIR_Op<"not"> {
  let summary = "Bitwise NOT";
  let arguments = (ins SIR_U256:$x);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$x `:` type($x) attr-dict `:` type($res)";
}

def ByteOp : SIR_Op<"byte"> {
  let summary = "BYTE opcode";
  let arguments = (ins SIR_U256:$i, SIR_U256:$x);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$i `:` type($i) `,` $x `:` type($x) attr-dict `:` type($res)";
}

def ShlOp : SIR_Op<"shl"> {
  let summary = "Logical shift left";
  let arguments = (ins SIR_U256:$shift, SIR_U256:$x);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$shift `:` type($shift) `,` $x `:` type($x) attr-dict `:` type($res)";
}

def ShrOp : SIR_Op<"shr"> {
  let summary = "Logical shift right";
  let arguments = (ins SIR_U256:$shift, SIR_U256:$x);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$shift `:` type($shift) `,` $x `:` type($x) attr-dict `:` type($res)";
}

def SarOp : SIR_Op<"sar"> {
  let summary = "Arithmetic shift right";
  let arguments = (ins SIR_U256:$shift, SIR_U256:$x);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$shift `:` type($shift) `,` $x `:` type($x) attr-dict `:` type($res)";
}

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

def LoadOp : SIR_Op<"load"> {
  let summary = "Load a 256-bit value from a pointer";
  let arguments = (ins SIR_Ptr:$ptr);
  let results   = (outs SIR_U256:$val);
  let assemblyFormat =
    "$ptr `:` type($ptr) attr-dict `:` type($val)";
}

def StoreOp : SIR_Op<"store"> {
  let summary = "Store a 256-bit value into a pointer";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$val);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $val `:` type($val) attr-dict";
}

def Load8Op : SIR_Op<"load8"> {
  let summary = "Load a byte from memory";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$offset);
  let results   = (outs SIR_U256:$val);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $offset `:` type($offset) attr-dict `:` type($val)";
}

def Store8Op : SIR_Op<"store8"> {
  let summary = "Store a byte to memory";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$offset, SIR_U256:$val);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $offset `:` type($offset) `,` $val `:` type($val) attr-dict";
}

// ==========================================================================
// CallData
// ==========================================================================

def CallDataLoadOp : SIR_Op<"calldataload"> {
  let summary = "Load from calldata";
  let arguments = (ins SIR_U256:$offset);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$offset `:` type($offset) attr-dict `:` type($res)";
}

def CallDataSizeOp : SIR_Op<"calldatasize"> {
  let summary = "Calldata size";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def CallDataCopyOp : SIR_Op<"calldatacopy"> {
  let summary = "Copy from calldata to memory";
  let arguments = (ins SIR_Ptr:$dst,
                       SIR_U256:$offset,
                       SIR_U256:$len);
  let assemblyFormat =
    "$dst `:` type($dst) `,` $offset `:` type($offset) `,` $len `:` type($len) "
    "attr-dict";
}

// ==========================================================================
// Memory Allocation & Management
// ==========================================================================

def MallocOp : SIR_Op<"malloc", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Allocate memory";
  let arguments = (ins SIR_U256:$size);
  let results   = (outs SIR_Ptr:$ptr);
  let hasCustomAssemblyFormat = 1;
}

def MallocAnyOp : SIR_Op<"mallocany"> {
  let summary = "Allocate memory (unbounded)";
  let arguments = (ins SIR_U256:$size);
  let results   = (outs SIR_Ptr:$ptr);
  let assemblyFormat =
    "$size `:` type($size) attr-dict `:` type($ptr)";
}

def SAllocOp : SIR_Op<"salloc"> {
  let summary = "Static allocation";
  let arguments = (ins SIR_U256:$size);
  let results   = (outs SIR_Ptr:$ptr);
  let assemblyFormat =
    "$size `:` type($size) attr-dict `:` type($ptr)";
}

def SAllocAnyOp : SIR_Op<"sallocany"> {
  let summary = "Static allocation (dynamic size)";
  let arguments = (ins SIR_U256:$size);
  let results   = (outs SIR_Ptr:$ptr);
  let assemblyFormat =
    "$size `:` type($size) attr-dict `:` type($ptr)";
}

def FreePtrOp : SIR_Op<"freeptr"> {
  let summary = "Get or advance free memory pointer";
  let arguments = (ins);
  let results   = (outs SIR_Ptr:$ptr);
  let assemblyFormat = "attr-dict `:` type($ptr)";
}

def MCopyOp : SIR_Op<"mcopy"> {
  let summary = "Copy memory region";
  let arguments = (ins SIR_Ptr:$dst,
                       SIR_Ptr:$src,
                       SIR_U256:$len);
  let assemblyFormat =
    "$dst `:` type($dst) `,` $src `:` type($src) `,` $len `:` type($len) "
    "attr-dict";
}

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

def SStoreOp : SIR_Op<"sstore"> {
  let summary = "Write storage slot";
  let arguments = (ins SIR_U256:$slot, SIR_U256:$val);
  let assemblyFormat =
    "$slot `:` type($slot) `,` $val `:` type($val) attr-dict";
}

def SLoadOp : SIR_Op<"sload", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Read storage slot";
  let arguments = (ins SIR_U256:$slot);
  let results   = (outs SIR_U256:$res);
  let hasCustomAssemblyFormat = 1;
}

def TStoreOp : SIR_Op<"tstore"> {
  let summary = "Write transient storage slot";
  let arguments = (ins SIR_U256:$slot, SIR_U256:$val);
  let assemblyFormat =
    "$slot `:` type($slot) `,` $val `:` type($val) attr-dict";
}

def TLoadOp : SIR_Op<"tload"> {
  let summary = "Read transient storage slot";
  let arguments = (ins SIR_U256:$slot);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$slot `:` type($slot) attr-dict `:` type($res)";
}

def GasOp : SIR_Op<"gas"> {
  let summary = "Remaining gas";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

// ==========================================================================
// Environment & Block Info
// ==========================================================================

def AddressOp : SIR_Op<"address"> {
  let summary = "Current address";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def BalanceOp : SIR_Op<"balance"> {
  let summary = "Account balance";
  let arguments = (ins SIR_U256:$addr);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$addr `:` type($addr) attr-dict `:` type($res)";
}

def OriginOp : SIR_Op<"origin"> {
  let summary = "Transaction origin";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def CallerOp : SIR_Op<"caller"> {
  let summary = "Message caller";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def CallValueOp : SIR_Op<"callvalue"> {
  let summary = "Call value";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def CodeSizeOp : SIR_Op<"codesize"> {
  let summary = "Current code size";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def CodeCopyOp : SIR_Op<"codecopy"> {
  let summary = "Copy from code to memory";
  let arguments = (ins SIR_Ptr:$dst,
                       SIR_U256:$offset,
                       SIR_U256:$len);
  let assemblyFormat =
    "$dst `:` type($dst) `,` $offset `:` type($offset) `,` $len `:` type($len) "
    "attr-dict";
}

def GasPriceOp : SIR_Op<"gasprice"> {
  let summary = "Gas price";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def ExtCodeSizeOp : SIR_Op<"extcodesize"> {
  let summary = "External account code size";
  let arguments = (ins SIR_U256:$addr);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$addr `:` type($addr) attr-dict `:` type($res)";
}

def ExtCodeCopyOp : SIR_Op<"extcodecopy"> {
  let summary = "Copy external account code to memory";
  let arguments = (ins SIR_U256:$addr,
                       SIR_Ptr:$dst,
                       SIR_U256:$offset,
                       SIR_U256:$len);
  let assemblyFormat =
    "$addr `:` type($addr) `,` $dst `:` type($dst) `,` "
    "$offset `:` type($offset) `,` $len `:` type($len) attr-dict";
}

def ReturnDataSizeOp : SIR_Op<"returndatasize"> {
  let summary = "returndatasize";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def ReturnDataCopyOp : SIR_Op<"returndatacopy"> {
  let summary = "Copy returndata to memory";
  let arguments = (ins SIR_Ptr:$dst,
                       SIR_U256:$offset,
                       SIR_U256:$len);
  let assemblyFormat =
    "$dst `:` type($dst) `,` $offset `:` type($offset) `,` $len `:` type($len) "
    "attr-dict";
}

def ExtCodeHashOp : SIR_Op<"extcodehash"> {
  let summary = "External account code hash";
  let arguments = (ins SIR_U256:$addr);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$addr `:` type($addr) attr-dict `:` type($res)";
}

// Block info: blockhash, coinbase, timestamp, number, difficulty, gaslimit,
// chainid, selfbalance, basefee, blobhash, blobbasefee.

def BlockHashOp : SIR_Op<"blockhash"> {
  let summary = "BLOCKHASH opcode";
  let arguments = (ins SIR_U256:$blockNumber);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$blockNumber `:` type($blockNumber) attr-dict `:` type($res)";
}

def CoinbaseOp : SIR_Op<"coinbase"> {
  let summary = "COINBASE";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def TimestampOp : SIR_Op<"timestamp"> {
  let summary = "TIMESTAMP";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def NumberOp : SIR_Op<"number"> {
  let summary = "NUMBER";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def DifficultyOp : SIR_Op<"difficulty"> {
  let summary = "DIFFICULTY / PREVRANDAO";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def GasLimitOp : SIR_Op<"gaslimit"> {
  let summary = "GASLIMIT";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def ChainIdOp : SIR_Op<"chainid"> {
  let summary = "CHAINID";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def SelfBalanceOp : SIR_Op<"selfbalance"> {
  let summary = "SELFBALANCE";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def BaseFeeOp : SIR_Op<"basefee"> {
  let summary = "BASEFEE";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def BlobHashOp : SIR_Op<"blobhash"> {
  let summary = "BLOBHASH";
  let arguments = (ins SIR_U256:$index);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat =
    "$index `:` type($index) attr-dict `:` type($res)";
}

def BlobBaseFeeOp : SIR_Op<"blobbasefee"> {
  let summary = "BLOBBASEFEE";
  let arguments = (ins);
  let results   = (outs SIR_U256:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

// ==========================================================================
// Logging
// ==========================================================================

def Log0Op : SIR_Op<"log0"> {
  let summary = "LOG0";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$len);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $len `:` type($len) attr-dict";
}

def Log1Op : SIR_Op<"log1"> {
  let summary = "LOG1";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$len,
                       SIR_U256:$topic0);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $len `:` type($len) `,` $topic0 `:` type($topic0) "
    "attr-dict";
}

def Log2Op : SIR_Op<"log2"> {
  let summary = "LOG2";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$len,
                       SIR_U256:$topic0, SIR_U256:$topic1);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $len `:` type($len) `,` "
    "$topic0 `:` type($topic0) `,` $topic1 `:` type($topic1) attr-dict";
}

def Log3Op : SIR_Op<"log3"> {
  let summary = "LOG3";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$len,
                       SIR_U256:$topic0, SIR_U256:$topic1,
                       SIR_U256:$topic2);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $len `:` type($len) `,` "
    "$topic0 `:` type($topic0) `,` $topic1 `:` type($topic1) `,` "
    "$topic2 `:` type($topic2) attr-dict";
}

def Log4Op : SIR_Op<"log4"> {
  let summary = "LOG4";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$len,
                       SIR_U256:$topic0, SIR_U256:$topic1,
                       SIR_U256:$topic2, SIR_U256:$topic3);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $len `:` type($len) `,` "
    "$topic0 `:` type($topic0) `,` $topic1 `:` type($topic1) `,` "
    "$topic2 `:` type($topic2) `,` $topic3 `:` type($topic3) attr-dict";
}

// ==========================================================================
// System / Call / Create / Termination
// ==========================================================================

def CreateOp : SIR_Op<"create"> {
  let summary = "CREATE";
  let arguments = (ins SIR_U256:$value,
                       SIR_Ptr:$ptr,
                       SIR_U256:$len);
  let results   = (outs SIR_U256:$addr);
  let assemblyFormat =
    "$value `:` type($value) `,` $ptr `:` type($ptr) `,` $len `:` type($len) "
    "attr-dict `:` type($addr)";
}

def Create2Op : SIR_Op<"create2"> {
  let summary = "CREATE2";
  let arguments = (ins SIR_U256:$value,
                       SIR_Ptr:$ptr,
                       SIR_U256:$len,
                       SIR_U256:$salt);
  let results   = (outs SIR_U256:$addr);
  let assemblyFormat =
    "$value `:` type($value) `,` $ptr `:` type($ptr) `,` "
    "$len `:` type($len) `,` $salt `:` type($salt) "
    "attr-dict `:` type($addr)";
}

// Approximate signature for calls: gas, addr, value, argsPtr, argsLen,
// retPtr, retLen -> success (0/1).

def CallOp : SIR_Op<"call"> {
  let summary = "CALL";
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

def CallCodeOp : SIR_Op<"callcode"> {
  let summary = "CALLCODE";
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

def DelegateCallOp : SIR_Op<"delegatecall"> {
  let summary = "DELEGATECALL";
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

def StaticCallOp : SIR_Op<"staticcall"> {
  let summary = "STATICCALL";
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

// Return & termination ops.
// You already have a ReturnOp, we keep it and add stop/revert/invalid/selfdestruct.

def ReturnOp : SIR_Op<"return", [Terminator]> {
  let summary = "Return a pointer + size";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$len);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $len `:` type($len) attr-dict";
}

def StopOp : SIR_Op<"stop"> {
  let summary = "STOP";
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def RevertOp : SIR_Op<"revert"> {
  let summary = "REVERT";
  let arguments = (ins SIR_Ptr:$ptr, SIR_U256:$len);
  let assemblyFormat =
    "$ptr `:` type($ptr) `,` $len `:` type($len) attr-dict";
}

def InvalidOp : SIR_Op<"invalid"> {
  let summary = "INVALID";
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def SelfDestructOp : SIR_Op<"selfdestruct"> {
  let summary = "SELFDESTRUCT";
  let arguments = (ins SIR_U256:$beneficiary);
  let assemblyFormat =
    "$beneficiary `:` type($beneficiary) attr-dict";
}

// ==========================================================================
// Intrinsics: runtime_start_offset, init_end_offset, runtime_length, icall, noop
// ==========================================================================

def RuntimeStartOffsetOp : SIR_Op<"runtime_start_offset"> {
  let summary = "runtime_start_offset()";
  let arguments = (ins);
  let results   = (outs SIR_U256:$offset);
  let assemblyFormat = "attr-dict `:` type($offset)";
}

def InitEndOffsetOp : SIR_Op<"init_end_offset"> {
  let summary = "init_end_offset()";
  let arguments = (ins);
  let results   = (outs SIR_U256:$offset);
  let assemblyFormat = "attr-dict `:` type($offset)";
}

def RuntimeLengthOp : SIR_Op<"runtime_length"> {
  let summary = "runtime_length()";
  let arguments = (ins);
  let results   = (outs SIR_U256:$len);
  let assemblyFormat = "attr-dict `:` type($len)";
}

// icall(fn, ...) -> ...
// For now we model it with a symbol ref and variadic operands / results.

def ICallOp : SIR_Op<"icall"> {
  let summary = "Internal function call";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<SIR_U256Type>:$args);
  let results   = (outs Variadic<SIR_U256Type>:$results);
  let assemblyFormat =
    "$callee `(` $args `)` attr-dict `:` type($results)";
}

def NoOpOp : SIR_Op<"noop"> {
  let summary = "No-op";
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

// ==========================================================================
// Internal Return
// ==========================================================================

def IRetOp : SIR_Op<"iret"> {
  let summary = "Internal function return";
  let arguments = (ins Variadic<SIR_U256Type>:$values);
  let assemblyFormat =
    "$values attr-dict";
}

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
  let assemblyFormat =
    "$input `:` type($input) attr-dict `:` type($output)";
}

#endif // SIR_OPS
