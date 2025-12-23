//===- SIRDialect.td - Sensei IR dialect definition -------------*- tablegen -*-===//
//
// Defines the SIR MLIR dialect.
//
//----------------------------------------------------------------------------===//

#ifndef SIR_DIALECT
#define SIR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"

def SIR_Dialect : Dialect {
  let name = "sir";
  let cppNamespace = "::sir";
  let summary = "Sensei IR (SIR) dialect for low-level EVM-like operations";
  let description = [{
    SIR is a low-level, SSA-based IR targeting the EVM. This MLIR dialect is a
    typed representation of SIR, with:
      - !sir.u256 as the canonical 256-bit EVM word type
      - !sir.ptr<AS> as an address-space-tagged pointer type
    It is intended as an intermediate stage between high-level Ora IR and the
    textual SIR format / EVM bytecode.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // SIR_DIALECT

