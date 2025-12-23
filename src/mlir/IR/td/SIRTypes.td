//===- SIRTypes.td - SIR dialect types ---------------------------*- tablegen -*-===//
//
// Defines !sir.u256 and !sir.ptr<AS> types.
//
//----------------------------------------------------------------------------===//

#ifndef SIR_TYPES
#define SIR_TYPES

include "mlir/IR/OpBase.td"
include "SIRDialect.td"
include "mlir/IR/AttrTypeBase.td"

// SIR_U256Type -> !sir.u256
def SIR_U256Type : TypeDef<SIR_Dialect, "U256"> {
  let mnemonic = "u256";
  let summary = "256-bit EVM word (unsigned)";
  let description = [{
    Canonical value type for SIR: a single 256-bit EVM word (logically
    unsigned). This is the type of storage values, stack values, arithmetic
    operands, etc.
  }];
}

// SIR_PtrType -> !sir.ptr<AS>
def SIR_PtrType : TypeDef<SIR_Dialect, "Ptr"> {
  let mnemonic = "ptr";
  let summary = "Pointer into a specific address space";
  let description = [{
    Pointer into an EVM address space, parameterized by an address space
    integer:
      0 = calldata (byte-addressed)
      1 = memory   (byte-addressed)
      2 = storage  (word-addressed)
      3 = transient (tstore-like, if used)
  }];

  // Address space parameter as an unsigned integer (32-bit)
  let parameters = (ins "unsigned":$addrSpace);

  // Printed as: !sir.ptr<1>
  let assemblyFormat = "`<` $addrSpace `>`";
}

// Type constraints to use in ops (SIR_U256, SIR_Ptr)
def SIR_U256 : TypeConstraint<
  CPred<"::llvm::isa<::sir::U256Type>($_self)">,
  "sir.u256">;

def SIR_Ptr : TypeConstraint<
  CPred<"::llvm::isa<::sir::PtrType>($_self)">,
  "sir.ptr">;

#endif // SIR_TYPES

