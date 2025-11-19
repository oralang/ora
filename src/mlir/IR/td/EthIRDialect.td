//===- EthIRDialect.td - EthIR dialect definition ----------------*- tablegen -*-===//

#ifndef ETHIR_DIALECT
#define ETHIR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"

def EthIR_Dialect : Dialect {
  let name = "ethir";
  let cppNamespace = "::ethir";
  let summary = "Ethereum Lower-Level IR Dialect";
  let description = [{
    A low-level IR for Ethereum byte-oriented operations:
    - explicit address spaces (calldata, memory, storage, transient)
    - untyped 256-bit arithmetic
    - simple control flow
    - direct lowering to EVM bytecode
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // ETHIR_DIALECT
