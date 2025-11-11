//===- OraOpsMinimal.td - Minimal Ora dialect operations -----*- tablegen -*-===//
//
// This file defines the minimal operations for the Ora MLIR dialect.
// Only includes the core operations that are actually implemented and used.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_OPS_MINIMAL
#define ORA_OPS_MINIMAL

include "OraDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Ora Operations - Core 4 Operations Only
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Contract Operations
//===----------------------------------------------------------------------===//

def Ora_ContractOp : Ora_Op<"contract"> {
  let summary = "Ora contract declaration";
  let description = [{
    Represents an Ora smart contract with its associated functions and storage.
    Contains global storage variables and function definitions.
    
    Example:
    ```mlir
    ora.contract @SimpleContract {
      ora.global "counter" : i256 = 0 : i256
      ora.global "status" : i1 = false
      func.func @init() { ... }
    }
    ```
  }];
  
  let arguments = (ins StrAttr:$sym_name);
  let regions = (region AnyRegion:$body);
  
  // Custom parser/printer needed for @ symbol prefix
  // let assemblyFormat = "$sym_name region($body) attr-dict";
  
  let extraClassDeclaration = [{
    /// Get the body region
    ::mlir::Region &getBodyRegion() { return getBody(); }
  }];
}

//===----------------------------------------------------------------------===//
// Storage Operations
//===----------------------------------------------------------------------===//

def Ora_GlobalOp : Ora_Op<"global"> {
  let summary = "Global storage variable declaration";
  let description = [{
    Declares a global storage variable with an initial value.
    These variables are stored in the contract's persistent storage.
    
    Example:
    ```mlir
    ora.global "counter" : i256 = 0 : i256
    ora.global "status" : i1 = false
    ```
  }];
  
  let arguments = (ins 
    StrAttr:$sym_name,
    TypeAttr:$type,
    AnyAttr:$init
  );
  
  // Custom parser/printer needed for type-attr syntax
  // let assemblyFormat = "$sym_name `:` type-attr($type) `=` $init attr-dict";
  
  let extraClassDeclaration = [{
    /// Get the type of the global variable
    ::mlir::Type getGlobalType() { return getTypeAttr().getValue(); }
    
    /// Get the initial value
    ::mlir::Attribute getInitialValue() { return getInit(); }
  }];
}

def Ora_SLoadOp : Ora_Op<"sload"> {
  let summary = "Load value from storage";
  let description = [{
    Loads a value from the specified global storage variable.
    The global name must reference a previously declared ora.global.
    
    Example:
    ```mlir
    %0 = ora.sload "counter" : () -> i256
    %1 = ora.sload "status" : () -> i1
    ```
  }];
  
  let arguments = (ins StrAttr:$global);
  let results = (outs AnyType:$result);
  
  // Custom parser/printer for cleaner syntax
  // let assemblyFormat = "$global attr-dict `:` type($result)";
  
  let extraClassDeclaration = [{
    /// Get the name of the global variable being loaded
    ::llvm::StringRef getGlobalName() { return getGlobal(); }
  }];
}

def Ora_SStoreOp : Ora_Op<"sstore"> {
  let summary = "Store value to storage";
  let description = [{
    Stores a value to the specified global storage variable.
    The global name must reference a previously declared ora.global.
    
    Example:
    ```mlir
    ora.sstore %value, "counter" : (i256) -> ()
    ora.sstore %status, "status" : (i1) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$value, StrAttr:$global);
  
  // Custom parser/printer for cleaner syntax
  // let assemblyFormat = "$value `,` $global attr-dict `:` type($value)";
  
  let extraClassDeclaration = [{
    /// Get the value being stored
    ::mlir::Value getStoredValue() { return getValue(); }
    
    /// Get the name of the global variable being stored to
    ::llvm::StringRef getGlobalName() { return getGlobal(); }
  }];
}

#endif // ORA_OPS_MINIMAL
