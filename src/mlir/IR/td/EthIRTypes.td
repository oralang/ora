//===- EthIRTypes.td -------------------------------------------------------*- tablegen -*-===//

#ifndef ETHIR_TYPES
#define ETHIR_TYPES

include "EthIRDialect.td"
include "mlir/IR/AttrTypeBase.td"

def EthIR_PtrType : TypeDef<EthIR_Dialect, "Ptr"> {
  let summary = "Pointer into a specific Ethereum address space";
  let description = [{
    Pointer type representing an address-space region:
    0 = calldata
    1 = memory
    2 = storage
    3 = transient
  }];

  let parameters = (ins "unsigned":$addrSpace);

  let mnemonic = "ptr";
  let assemblyFormat = "`<` $addrSpace `>`";
}

#endif // ETHIR_TYPES

