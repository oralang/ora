//===- OraTypes.td - Ora dialect types ------------------*- tablegen -*-===//
//
// This file defines the minimal types for the Ora MLIR dialect.
// Currently focuses on basic types that are actually used.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_TYPES
#define ORA_TYPES

include "OraDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Ora Types - Minimal Set
//===----------------------------------------------------------------------===//

// Note: For now, we use standard MLIR types (i256, i1, etc.) 
// These type definitions are placeholders for future expansion

class Ora_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ora_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // ORA_TYPES