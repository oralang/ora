//===- EthIROps.td ------------------------------------------------*- tablegen -*-===//

#ifndef ETHIR_OPS
#define ETHIR_OPS

include "EthIRDialect.td"
include "mlir/IR/OpBase.td"
include "EthIRTypes.td"

// --------------------------------------------------------------------------
// 256-bit integer type constraint
// --------------------------------------------------------------------------
//
// This is a **type constraint**, not a type alias.
// It checks that the MLIR type is an IntegerType(256).
//
def U256 : TypeConstraint<
  CPred<"::llvm::isa<::mlir::IntegerType>($_self) && "
        "::llvm::cast<::mlir::IntegerType>($_self).getWidth() == 256">,
  "u256">;

// --------------------------------------------------------------------------
// Base op definition for all EthIR ops
// --------------------------------------------------------------------------

class EthIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<EthIR_Dialect, mnemonic, traits>;

// ==========================================================================
// Constants
// ==========================================================================
def ConstOp : EthIR_Op<"const"> {
  let summary = "Constant 256-bit value";
  let arguments = (ins I64Attr:$value);
  let results   = (outs U256:$res);
  let assemblyFormat = "$value attr-dict `:` type($res)";
}

// ==========================================================================
// Pointer Arithmetic
// ==========================================================================
def AddPtrOp : EthIR_Op<"addptr"> {
  let summary = "Pointer arithmetic";
  let arguments = (ins EthIR_PtrType:$base, U256:$offset);
  let results   = (outs EthIR_PtrType:$res);
  let assemblyFormat = "$base `:` type($base) `,` $offset `:` type($offset) attr-dict `:` type($res)";
}

// ==========================================================================
// Memory Loads & Stores
// ==========================================================================
def LoadOp : EthIR_Op<"load"> {
  let summary = "Load a 256-bit value from a pointer";
  let arguments = (ins EthIR_PtrType:$ptr);
  let results   = (outs U256:$val);
  let assemblyFormat = "$ptr `:` type($ptr) attr-dict `:` type($val)";
}

def StoreOp : EthIR_Op<"store"> {
  let summary = "Store a 256-bit value into a pointer";
  let arguments = (ins EthIR_PtrType:$ptr, U256:$val);
  let assemblyFormat = "$ptr `:` type($ptr) `,` $val `:` type($val) attr-dict";
}

// ==========================================================================
// CallData
// ==========================================================================
def CallDataLoadOp : EthIR_Op<"calldataload"> {
  let summary = "Load from calldata";
  let arguments = (ins U256:$offset);
  let results   = (outs U256:$res);
  let assemblyFormat = "$offset `:` type($offset) attr-dict `:` type($res)";
}

// ==========================================================================
// Memory Allocation
// ==========================================================================
def MallocOp : EthIR_Op<"malloc"> {
  let summary = "Allocate memory";
  let arguments = (ins U256:$size);
  let results   = (outs EthIR_PtrType:$ptr);

  // Same reason as AddPtrOp: custom type = must print explicitly.
  let assemblyFormat = "$size `:` type($size) attr-dict `:` type($ptr)";
}

// ==========================================================================
// Crypto
// ==========================================================================
def KeccakOp : EthIR_Op<"keccak256"> {
  let summary = "Compute keccak256";
  let arguments = (ins EthIR_PtrType:$ptr, U256:$len);
  let results   = (outs U256:$hash);
  let assemblyFormat = "$ptr `:` type($ptr) `,` $len `:` type($len) attr-dict `:` type($hash)";
}

// ==========================================================================
// Storage
// ==========================================================================
def SStoreOp : EthIR_Op<"sstore"> {
  let summary = "Write storage slot";
  let arguments = (ins U256:$slot, U256:$val);
  let assemblyFormat = "$slot `:` type($slot) `,` $val `:` type($val) attr-dict";
}

def SLoadOp : EthIR_Op<"sload"> {
  let summary = "Read storage slot";
  let arguments = (ins U256:$slot);
  let results   = (outs U256:$res);
  let assemblyFormat = "$slot `:` type($slot) attr-dict `:` type($res)";
}

// ==========================================================================
// Return
// ==========================================================================
def ReturnOp : EthIR_Op<"return"> {
  let summary = "Return a pointer + size";
  let arguments = (ins EthIR_PtrType:$ptr, U256:$len);
  let assemblyFormat = "$ptr `:` type($ptr) `,` $len `:` type($len) attr-dict";
}

#endif // ETHIR_OPS