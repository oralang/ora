//===- OraAttributes.td - Ora dialect attributes --------*- tablegen -*-===//
//
// This file defines the minimal attributes for the Ora MLIR dialect.
// Currently focuses on attributes that are actually used in the 4 core operations.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_ATTRIBUTES
#define ORA_ATTRIBUTES

include "OraDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// Ora Attributes - Minimal Set
//===----------------------------------------------------------------------===//

class Ora_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Ora_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Note: Currently using standard MLIR attributes:
// - SymbolNameAttr for contract and global names
// - TypeAttr for global variable types  
// - AnyAttr for global initial values
// - StrAttr for global name references in sload/sstore
//
// Custom attributes can be added here when needed.

#endif // ORA_ATTRIBUTES