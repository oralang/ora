//===- OraOps.td - Ora dialect operations ---------------*- tablegen -*-===//
//
// This file defines the core operations for the Ora MLIR dialect.
// Currently supports the 4 operations that are actually implemented.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_OPS
#define ORA_OPS

include "OraDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Ora Operations - Core 4 Operations
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
  
  let extraClassDeclaration = [{
    /// Get the value being stored
    ::mlir::Value getStoredValue() { return getValue(); }
    
    /// Get the name of the global variable being stored to
    ::llvm::StringRef getGlobalName() { return getGlobal(); }
  }];
}

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

def Ora_StringConstantOp : Ora_PureOp<"string.constant"> {
  let summary = "String constant";
  let description = [{
    Creates a string constant value.
    
    Example:
    ```mlir
    %str = ora.string.constant "hello world" : !ora.string
    ```
  }];
  
  let arguments = (ins StrAttr:$value);
  let results = (outs AnyType:$result);
}


def Ora_HexConstantOp : Ora_PureOp<"hex.constant"> {
  let summary = "Hexadecimal constant";
  let description = [{
    Creates a hexadecimal constant value.
    
    Example:
    ```mlir
    %hex = ora.hex.constant "0x1a2b" : !ora.hex
    ```
  }];
  
  let arguments = (ins StrAttr:$value);
  let results = (outs AnyType:$result);
}


//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def Ora_PowerOp : Ora_PureOp<"power"> {
  let summary = "Power operation";
  let description = [{
    Computes base raised to the power of exponent.
    
    Example:
    ```mlir
    %result = ora.power %base, %exp : (i256, i256) -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$base, AnyType:$exponent);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Variable Declaration Operations
//===----------------------------------------------------------------------===//

def Ora_ConstOp : Ora_Op<"const"> {
  let summary = "Constant variable declaration";
  let description = [{
    Declares a constant variable with an immutable value.
    
    Example:
    ```mlir
    %const = ora.const "PI" : f64 = 3.14159 : f64
    ```
  }];
  
  let arguments = (ins StrAttr:$name, AnyAttr:$value);
  let results = (outs AnyType:$result);
}

def Ora_ImmutableOp : Ora_Op<"immutable"> {
  let summary = "Immutable variable declaration";
  let description = [{
    Declares an immutable variable that can be set once.
    
    Example:
    ```mlir
    %immut = ora.immutable "config" : i256 = %init_value : i256
    ```
  }];
  
  let arguments = (ins StrAttr:$name, AnyType:$value);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

def Ora_IfOp : Ora_Op<"if"> {
  let summary = "Conditional operation";
  let description = [{
    Executes one of two regions based on a condition.
    
    Example:
    ```mlir
    %result = ora.if %condition : i1 -> i256 {
      %then_val = arith.constant 1 : i256
      ora.yield %then_val : i256
    } else {
      %else_val = arith.constant 0 : i256
      ora.yield %else_val : i256
    }
    ```
  }];
  
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<2>:$body);
}

def Ora_YieldOp : Ora_Op<"yield"> {
  let summary = "Yield operation";
  let description = [{
    Yields values from a region.
    
    Example:
    ```mlir
    ora.yield %value : i256
    ```
  }];
  
  let arguments = (ins Variadic<AnyType>:$operands);
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def Ora_MLoadOp : Ora_Op<"mload"> {
  let summary = "Load from memory";
  let description = [{
    Loads a value from memory region.
    
    Example:
    ```mlir
    %value = ora.mload "variable" : () -> i256
    ```
  }];
  
  let arguments = (ins StrAttr:$variable);
  let results = (outs AnyType:$result);
}

def Ora_MStoreOp : Ora_Op<"mstore"> {
  let summary = "Store to memory";
  let description = [{
    Stores a value to memory region.
    
    Example:
    ```mlir
    ora.mstore %value, "variable" : (i256) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$value, StrAttr:$variable);
}

def Ora_TLoadOp : Ora_Op<"tload"> {
  let summary = "Load from transient storage";
  let description = [{
    Loads a value from transient storage.
    
    Example:
    ```mlir
    %value = ora.tload "key" : () -> i256
    ```
  }];
  
  let arguments = (ins StrAttr:$key);
  let results = (outs AnyType:$result);
}

def Ora_TStoreOp : Ora_Op<"tstore"> {
  let summary = "Store to transient storage";
  let description = [{
    Stores a value to transient storage.
    
    Example:
    ```mlir
    ora.tstore %value, "key" : (i256) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$value, StrAttr:$key);
}

//===----------------------------------------------------------------------===//
// Type Conversion Operations
//===----------------------------------------------------------------------===//

def Ora_CastOp : Ora_PureOp<"cast"> {
  let summary = "Type conversion operation";
  let description = [{
    Converts a value from one type to another.
    
    Example:
    ```mlir
    %result = ora.cast %input : i256 -> f64
    ```
  }];
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Destructuring Operations
//===----------------------------------------------------------------------===//

def Ora_DestructureOp : Ora_PureOp<"destructure"> {
  let summary = "Destructuring operation";
  let description = [{
    Destructures a value into its components using pattern matching.
    
    Example:
    ```mlir
    %result = ora.destructure %value : i256 -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$value, StrAttr:$pattern_type);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Enum Operations
//===----------------------------------------------------------------------===//

def Ora_EnumDeclOp : Ora_Op<"enum.decl"> {
  let summary = "Enum declaration";
  let description = [{
    Declares an enum type with its variants.
    
    Example:
    ```mlir
    ora.enum.decl "Status" : i8 {
      Active = 0,
      Inactive = 1
    }
    ```
  }];
  
  let arguments = (ins StrAttr:$name, TypeAttr:$repr_type);
  let regions = (region AnyRegion:$variants);
}

def Ora_EnumConstantOp : Ora_PureOp<"enum_constant"> {
  let summary = "Enum constant";
  let description = [{
    Creates an enum constant value.
    
    Example:
    ```mlir
    %status = ora.enum_constant "Status.Active" : !ora.enum<Status, i8>
    ```
  }];
  
  let arguments = (ins StrAttr:$enum_name, StrAttr:$variant_name);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Struct Operations
//===----------------------------------------------------------------------===//

def Ora_StructDeclOp : Ora_Op<"struct.decl"> {
  let summary = "Struct declaration";
  let description = [{
    Declares a struct type with its fields.
    
    Example:
    ```mlir
    ora.struct.decl "Point" {
      x: i256,
      y: i256
    }
    ```
  }];
  
  let arguments = (ins StrAttr:$name);
  let regions = (region AnyRegion:$fields);
}

def Ora_StructFieldStoreOp : Ora_Op<"struct_field_store"> {
  let summary = "Struct field store";
  let description = [{
    Stores a value to a struct field.
    
    Example:
    ```mlir
    ora.struct_field_store %point, "x", %value : (!ora.struct<Point>, i256) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$struct, StrAttr:$field_name, AnyType:$value);
}

def Ora_StructInstantiateOp : Ora_PureOp<"struct_instantiate"> {
  let summary = "Struct instantiation";
  let description = [{
    Creates a new struct instance.
    
    Example:
    ```mlir
    %point = ora.struct_instantiate "Point" (%x, %y) : (i256, i256) -> !ora.struct<Point>
    ```
  }];
  
  let arguments = (ins StrAttr:$struct_name, Variadic<AnyType>:$field_values);
  let results = (outs AnyType:$result);
}

def Ora_StructInitOp : Ora_PureOp<"struct_init"> {
  let summary = "Struct initialization";
  let description = [{
    Initializes a struct with field values.
    
    Example:
    ```mlir
    %point = ora.struct_init { x: %x, y: %y } : !ora.struct<Point>
    ```
  }];
  
  let arguments = (ins Variadic<AnyType>:$field_values);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Map Operations
//===----------------------------------------------------------------------===//

def Ora_MapGetOp : Ora_PureOp<"map_get"> {
  let summary = "Map get operation";
  let description = [{
    Gets a value from a map using a key.
    
    Example:
    ```mlir
    %value = ora.map_get %map, %key : (!ora.map<i256, i256>, i256) -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$map, AnyType:$key);
  let results = (outs AnyType:$result);
}

def Ora_MapStoreOp : Ora_Op<"map_store"> {
  let summary = "Map store operation";
  let description = [{
    Stores a value in a map using a key.
    
    Example:
    ```mlir
    ora.map_store %map, %key, %value : (!ora.map<i256, i256>, i256, i256) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$map, AnyType:$key, AnyType:$value);
}

//===----------------------------------------------------------------------===//
// Formal Verification Operations
//===----------------------------------------------------------------------===//

def Ora_RequiresOp : Ora_Op<"requires"> {
  let summary = "Precondition assertion";
  let description = [{
    Asserts a precondition that must be true at function entry.
    
    Example:
    ```mlir
    ora.requires %condition : i1
    ```
  }];
  
  let arguments = (ins I1:$condition);
}

def Ora_EnsuresOp : Ora_Op<"ensures"> {
  let summary = "Postcondition assertion";
  let description = [{
    Asserts a postcondition that must be true at function exit.
    
    Example:
    ```mlir
    ora.ensures %condition : i1
    ```
  }];
  
  let arguments = (ins I1:$condition);
}

def Ora_InvariantOp : Ora_Op<"invariant"> {
  let summary = "Invariant assertion";
  let description = [{
    Asserts an invariant that must be true at specific points.
    
    Example:
    ```mlir
    ora.invariant %condition : i1
    ```
  }];
  
  let arguments = (ins I1:$condition);
}

def Ora_OldOp : Ora_PureOp<"old"> {
  let summary = "Old value reference";
  let description = [{
    References the old value of a variable for formal verification.
    
    Example:
    ```mlir
    %old_val = ora.old %current_val : i256 -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Financial Operations
//===----------------------------------------------------------------------===//

def Ora_MoveOp : Ora_Op<"move"> {
  let summary = "Move operation for financial transfers";
  let description = [{
    Moves an amount from one balance to another.
    
    Example:
    ```mlir
    ora.move %amount from %source to %destination : (i256, i256, i256) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$amount, AnyType:$source, AnyType:$destination);
}

//===----------------------------------------------------------------------===//
// Event Operations
//===----------------------------------------------------------------------===//

def Ora_LogOp : Ora_Op<"log"> {
  let summary = "Event logging operation";
  let description = [{
    Emits an event log with the specified parameters.
    
    Example:
    ```mlir
    ora.log "Transfer" (%from, %to, %amount) : (address, address, i256) -> ()
    ```
  }];
  
  let arguments = (ins StrAttr:$event_name, Variadic<AnyType>:$parameters);
}

//===----------------------------------------------------------------------===//
// Locking Operations
//===----------------------------------------------------------------------===//

def Ora_LockOp : Ora_Op<"lock"> {
  let summary = "Lock operation";
  let description = [{
    Acquires a lock on the specified resource.
    
    Example:
    ```mlir
    ora.lock %resource : i256 -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$resource);
}

def Ora_UnlockOp : Ora_Op<"unlock"> {
  let summary = "Unlock operation";
  let description = [{
    Releases a lock on the specified resource.
    
    Example:
    ```mlir
    ora.unlock %resource : i256 -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$resource);
}

//===----------------------------------------------------------------------===//
// Error Handling Operations
//===----------------------------------------------------------------------===//

def Ora_TryOp : Ora_Op<"try_catch"> {
  let summary = "Try-catch operation";
  let description = [{
    Attempts to execute an operation with error handling.
    
    Example:
    ```mlir
    %result = ora.try_catch %try_operation : i256 -> i256 catch {
      %error_val = arith.constant 0 : i256
      ora.yield %error_val : i256
    }
    ```
  }];
  
  let arguments = (ins AnyType:$try_operation);
  let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$catch_region);
}

//===----------------------------------------------------------------------===//
// Loop Operations
//===----------------------------------------------------------------------===//

def Ora_WhileOp : Ora_Op<"while"> {
  let summary = "While loop operation";
  let description = [{
    Executes a region while a condition is true.
    
    Example:
    ```mlir
    ora.while %condition : i1 {
      // loop body
      ora.yield
    }
    ```
  }];
  
  let arguments = (ins I1:$condition);
  let regions = (region AnyRegion:$body);
}

def Ora_ForOp : Ora_Op<"for"> {
  let summary = "For loop operation";
  let description = [{
    Iterates over a collection with key-value pairs.
    
    Example:
    ```mlir
    ora.for %collection : !ora.map<i256, i256> |%key, %value| {
      // loop body
      ora.yield
    }
    ```
  }];
  
  let arguments = (ins AnyType:$collection);
  let regions = (region AnyRegion:$body);
}

def Ora_BreakOp : Ora_Op<"break"> {
  let summary = "Break operation";
  let description = [{
    Breaks out of a loop, optionally with a label and return value.
    
    Example:
    ```mlir
    ora.break "outer" %value : i256
    ```
  }];
  
  let arguments = (ins OptionalAttr<StrAttr>:$label, Variadic<AnyType>:$value);
}

def Ora_ContinueOp : Ora_Op<"continue"> {
  let summary = "Continue operation";
  let description = [{
    Continues to the next iteration of a loop, optionally with a label.
    
    Example:
    ```mlir
    ora.continue "outer"
    ```
  }];
  
  let arguments = (ins OptionalAttr<StrAttr>:$label);
}

//===----------------------------------------------------------------------===//
// Return Operations
//===----------------------------------------------------------------------===//

def Ora_ReturnOp : Ora_Op<"return"> {
  let summary = "Return operation";
  let description = [{
    Returns from a function with optional return values.
    
    Example:
    ```mlir
    ora.return %value : i256
    ```
  }];
  
  let arguments = (ins Variadic<AnyType>:$operands);
}

//===----------------------------------------------------------------------===//
// Switch Operations
//===----------------------------------------------------------------------===//

def Ora_SwitchOp : Ora_Op<"switch"> {
  let summary = "Switch operation";
  let description = [{
    Executes one of multiple regions based on a value.
    
    Example:
    ```mlir
    %result = ora.switch %value : i256 -> i256 {
      case 0 => {
        %val0 = arith.constant 0 : i256
        ora.yield %val0 : i256
      }
      case 1 => {
        %val1 = arith.constant 1 : i256
        ora.yield %val1 : i256
      }
      else => {
        %val_default = arith.constant -1 : i256
        ora.yield %val_default : i256
      }
    }
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$cases);
}

#endif // ORA_OPS