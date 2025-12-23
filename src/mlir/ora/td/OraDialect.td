//===- OraDialect.td - Ora dialect definition ------------*- tablegen -*-===//
//
// This file defines the minimal Ora MLIR dialect.
// Currently supports the 4 core operations that are actually implemented.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_DIALECT
#define ORA_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Ora dialect definition
//===----------------------------------------------------------------------===//

def Ora_Dialect : Dialect {
  let name = "ora";
  let summary = "Ora smart contract language dialect";
  let description = [{
    The Ora dialect provides operations for representing Ora smart contract
    language constructs in MLIR. Supports comprehensive Ora language features:
    
    - Contract declarations (ora.contract)
    - Storage operations (ora.global, ora.sload, ora.sstore)
    - Memory operations (ora.mload, ora.mstore)
    - Transient storage (ora.tload, ora.tstore)
    - Constants (ora.string.constant, ora.address.constant, ora.hex.constant, ora.binary.constant)
    - Arithmetic operations (ora.power)
    - Destructuring (ora.destructure)
    - Enum operations (ora.enum.decl, ora.enum_constant)
    - Struct operations (ora.struct.decl, ora.struct_field_store, ora.struct_instantiate, ora.struct_init)
    - Map operations (ora.map_get, ora.map_store)
    - Formal verification (ora.requires, ora.ensures, ora.invariant, ora.old)
    - Control flow (ora.if, ora.while, ora.for, ora.switch, ora.break, ora.continue, ora.return)
    - Financial operations (ora.move)
    - Event logging (ora.log)
    - Locking (ora.lock, ora.unlock)
    - Error handling (ora.try)
    - Variable declarations (ora.const, ora.immutable)
    
    This dialect provides complete coverage of the Ora language grammar.
  }];
  
  let cppNamespace = "::mlir::ora";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base classes for Ora operations
//===----------------------------------------------------------------------===//

// Base class for all Ora operations
class Ora_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ora_Dialect, mnemonic, traits>;

// Base class for pure Ora operations (no side effects)
// Pure = AlwaysSpeculatable + NoMemoryEffect (for canonicalization)
// Explicitly include NoMemoryEffect for DCE (even though Pure includes it)
// This ensures DCE can see the MemoryEffects<[]> interface
class Ora_PureOp<string mnemonic, list<Trait> traits = []> :
    Ora_Op<mnemonic, !listconcat(traits, [Pure, NoMemoryEffect])>;

//===----------------------------------------------------------------------===//
// Ora Types
//===----------------------------------------------------------------------===//

// Note: Type definitions are now in OraTypes.td
// IntegerType and all other types are defined there

#endif // ORA_DIALECT