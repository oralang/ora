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
include "mlir/IR/Traits.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
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
  
  // Contract body contains declarations, not executable code, so no terminator needed
  // Contract contains symbols (func.func), so it needs SymbolTable trait
  let traits = [SingleBlock, NoTerminator, SymbolTable];
  
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = [{
    /// Get the body region
    ::mlir::Region &getBodyRegion() { return getBody(); }
  }];
}

//===----------------------------------------------------------------------===//
// Type Conversion Operations
//===----------------------------------------------------------------------===//

def Ora_AddrToI160Op : Ora_PureOp<"addr.to.i160"> {
  let summary = "Convert Ora address type to i160";
  let description = [{
    Converts an Ora address type (!ora.address) to the built-in i160 type.
    This is needed for address comparisons since arith.cmpi only accepts built-in integer types.
    
    Example:
    ```mlir
    %a160 = ora.addr.to.i160 %addr : !ora.address -> i160
    ```
  }];
  
  let arguments = (ins AnyType:$addr);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($result)";
  
  let extraClassDeclaration = [{
    /// Get the input address value
    ::mlir::Value getAddress() { return getAddr(); }
  }];
}

def Ora_I160ToAddrOp : Ora_PureOp<"i160.to.addr"> {
  let summary = "Convert i160 to Ora address type";
  let description = [{
    Converts a built-in i160 type to an Ora address type (!ora.address).
    This is needed when creating address constants from integer values.
    
    Example:
    ```mlir
    %addr = ora.i160.to.addr %i160 : i160 -> !ora.address
    ```
  }];
  
  let arguments = (ins AnyType:$i160);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$i160 attr-dict `:` type($i160) `->` type($result)";
}

def Ora_RefinementToBaseOp : Ora_PureOp<"refinement_to_base"> {
  let summary = "Convert refinement type to base type";
  let description = [{
    Converts a refinement type (e.g., !ora.min_value<i256, ...>) to its base type (i256).
    This is a pure operation with no side effects - refinement types have the same
    representation as their base types, this is just a type-level conversion.
    
    Example:
    ```mlir
    %base = ora.refinement_to_base %refined : !ora.min_value<i256, 0, 0, 0, 1000> -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
  
  let extraClassDeclaration = [{
    // Builder that infers result type from input refinement type
    static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                      ::mlir::Value value);
  }];
}

def Ora_BaseToRefinementOp : Ora_PureOp<"base_to_refinement"> {
  let summary = "Convert base type to refinement type";
  let description = [{
    Converts a base type value (e.g., i256) to a refinement type
    (e.g., !ora.in_range<i256, ...>). This is a type-level conversion
    with no runtime representation change.

    Example:
    ```mlir
    %refined = ora.base_to_refinement %base : i256 -> !ora.in_range<i256, 0, 0, 0, 0, 0, 0, 0, 100>
    ```
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
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

  let traits = [Symbol, MemoryEffects<[MemWrite]>];
  
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = [{
    /// Get the type of the global variable
    ::mlir::Type getGlobalType() { return getTypeAttr().getValue(); }
    
    /// Get the initial value
    ::mlir::Attribute getInitialValue() { return getInit(); }
  }];
}

def Ora_SLoadOp : Ora_Op<"sload",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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
  
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = [{
    /// Get the name of the global variable being loaded
    ::llvm::StringRef getGlobalName() { return getGlobal(); }
  }];
}

def Ora_SStoreOp : Ora_Op<"sstore",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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
  
  let assemblyFormat = "$value `,` $global `:` type($value) attr-dict";
  let extraClassDeclaration = [{
    /// Get the value being stored
    ::mlir::Value getStoredValue() { return getValue(); }
    
    /// Get the name of the global variable being stored to
    ::llvm::StringRef getGlobalName() { return getGlobal(); }
  }];
}

def Ora_MemoryGlobalOp : Ora_Op<"memory.global"> {
  let summary = "Memory global variable declaration";
  let description = [{
    Declares a contract-level memory variable used by Ora lowering.

    Example:
    ```mlir
    "ora.memory.global"() {sym_name = "tmp", type = i256}
    ```
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$type
  );

  let traits = [Symbol, MemoryEffects<[MemWrite]>];
}

def Ora_TStoreGlobalOp : Ora_Op<"tstore.global"> {
  let summary = "Transient storage global variable declaration";
  let description = [{
    Declares a contract-level transient storage variable used by Ora lowering.

    Example:
    ```mlir
    "ora.tstore.global"() {sym_name = "tmp", type = i256}
    ```
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$type
  );

  let traits = [Symbol, MemoryEffects<[MemWrite]>];
}

//===----------------------------------------------------------------------===//
// Test Operation (Simple Custom Printer Test)
//===----------------------------------------------------------------------===//

def Ora_TestOp : Ora_Op<"test"> {
  let summary = "Simple test operation for custom printer";
  let description = [{
    A minimal operation to test custom printer functionality.
    
    Example:
    ```mlir
    ora.test TEST_FORMAT
    ```
  }];
  
  // No arguments, no results - simplest possible operation
  // Test with assemblyFormat (declarative format)
  let assemblyFormat = "`TEST_FORMAT` attr-dict";
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
  
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Ora_BytesConstantOp : Ora_PureOp<"bytes.constant"> {
  let summary = "Bytes constant";
  let description = [{
    Creates a bytes constant value.
    
    Example:
    ```mlir
    %bytes = ora.bytes.constant "0xdeadbeef" : !ora.bytes
    ```
  }];
  
  let arguments = (ins StrAttr:$value);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$value attr-dict `:` type($result)";
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
  
  let assemblyFormat = "$value attr-dict `:` type($result)";
}


//===----------------------------------------------------------------------===//
// Range Operations
//===----------------------------------------------------------------------===//

def Ora_RangeOp : Ora_PureOp<"range"> {
  let summary = "Range operation";
  let description = [{
    Creates a range value from start to end, with optional inclusive flag.
    Used for range expressions like (0..20) or (0...20).
    
    Example:
    ```mlir
    %range = ora.range %start, %end {inclusive = false} : (i256, i256) -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$start, AnyType:$end, OptionalAttr<BoolAttr>:$inclusive);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$start `,` $end attr-dict `:` type($start) `,` type($end) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def Ora_AddOp : Ora_PureOp<"add", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let hasCanonicalizer = 1;
  let summary = "Add operation";
  let description = [{
    Adds two Ora integer values with proper unsigned semantics and overflow handling.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.add %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
  
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  
}

def Ora_SubOp : Ora_PureOp<"sub", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let hasCanonicalizer = 1;
  let summary = "Subtract operation";
  let description = [{
    Subtracts two Ora integer values with proper unsigned semantics and overflow handling.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.sub %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
  
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Ora_MulOp : Ora_PureOp<"mul", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let hasCanonicalizer = 1;
  let summary = "Multiply operation";
  let description = [{
    Multiplies two Ora integer values with proper unsigned semantics and overflow handling.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.mul %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
  
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Ora_DivOp : Ora_PureOp<"div", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let hasCanonicalizer = 1;
  let summary = "Divide operation";
  let description = [{
    Divides two Ora integer values with proper unsigned semantics.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.div %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
  
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Ora_RemOp : Ora_PureOp<"rem"> {
  let summary = "Remainder operation";
  let description = [{
    Computes the remainder of dividing two Ora integer values.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.rem %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
  
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Ora_CmpOp : Ora_PureOp<"cmp"> {
  let summary = "Compare operation";
  let description = [{
    Compares two Ora integer values with proper unsigned semantics.
    Accepts !ora.int<width, signed> types and returns i1 (boolean).
    
    Example:
    ```mlir
    %result = ora.cmp "eq", %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> i1
    ```
  }];
  
  let arguments = (ins StrAttr:$predicate, AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

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
  
  let assemblyFormat = "$base `,` $exponent attr-dict `:` type($base) `,` type($exponent) `->` type($result)";
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
  
  let assemblyFormat = "$name `:` type($result) `=` $value attr-dict";
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
  
  let assemblyFormat = "$name `:` type($result) `=` $value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// EVM Builtin Operations
//===----------------------------------------------------------------------===//

def Ora_EvmOriginOp : Ora_PureOp<"evm.origin"> {
  let summary = "EVM origin builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmCallerOp : Ora_PureOp<"evm.caller"> {
  let summary = "EVM caller builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmGaspriceOp : Ora_PureOp<"evm.gasprice"> {
  let summary = "EVM gasprice builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmCallvalueOp : Ora_PureOp<"evm.callvalue"> {
  let summary = "EVM callvalue builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmGasOp : Ora_PureOp<"evm.gas"> {
  let summary = "EVM gas builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmTimestampOp : Ora_PureOp<"evm.timestamp"> {
  let summary = "EVM timestamp builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmNumberOp : Ora_PureOp<"evm.number"> {
  let summary = "EVM block number builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmCoinbaseOp : Ora_PureOp<"evm.coinbase"> {
  let summary = "EVM coinbase builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmDifficultyOp : Ora_PureOp<"evm.difficulty"> {
  let summary = "EVM difficulty/prevrandao builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmPrevrandaoOp : Ora_PureOp<"evm.prevrandao"> {
  let summary = "EVM prevrandao builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmGaslimitOp : Ora_PureOp<"evm.gaslimit"> {
  let summary = "EVM gaslimit builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmChainidOp : Ora_PureOp<"evm.chainid"> {
  let summary = "EVM chainid builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmBasefeeOp : Ora_PureOp<"evm.basefee"> {
  let summary = "EVM basefee builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
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
    %result = ora.if %condition then {
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
  let regions = (region AnyRegion:$thenRegion,
                        AnyRegion:$elseRegion);
  
  // Custom parser/printer for prettier syntax with "then"/"else" keywords
  let hasCustomAssemblyFormat = 1;
}

def Ora_IsolatedIfOp : Ora_Op<"isolated_if", [IsolatedFromAbove]> {
  let summary = "Isolated conditional operation";
  let description = [{
    Executes one of two regions based on a condition. This variant is isolated
    from the parent scope, so values from the parent region must be passed
    explicitly.

    Example:
    ```mlir
    ora.isolated_if %condition then {
      ora.yield
    } else {
      ora.yield
    }
    ```
  }];

  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$thenRegion,
                        AnyRegion:$elseRegion);

  let hasCustomAssemblyFormat = 1;
}

def Ora_YieldOp : Ora_Op<"yield", [Terminator]> {
  let summary = "Yield operation";
  let description = [{
    Yields values from a region.
    
    Example:
    ```mlir
    ora.yield %value : i256
    ```
  }];
  
  let arguments = (ins Variadic<AnyType>:$operands);
  
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def Ora_MLoadOp : Ora_Op<"mload",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
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
  
  let assemblyFormat = "$variable attr-dict `:` type($result)";
}

def Ora_MStoreOp : Ora_Op<"mstore",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store to memory";
  let description = [{
    Stores a value to memory region.
    
    Example:
    ```mlir
    ora.mstore %value, "variable" : (i256) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$value, StrAttr:$variable);
  
  let assemblyFormat = "$value `,` $variable attr-dict `:` type($value)";
}

def Ora_MLoad8Op : Ora_Op<"mload8",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load a single byte from memory";
  let description = [{
    Loads a single byte from memory at (base + offset).
    
    Example:
    ```mlir
    %byte = ora.mload8 %base, %offset : (ptr, i256) -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$base, AnyType:$offset);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$base `,` $offset attr-dict `:` type($base) `,` type($offset) `->` type($result)";
}

def Ora_MStore8Op : Ora_Op<"mstore8",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store a single byte to memory";
  let description = [{
    Stores a single byte to memory at (base + offset).
    
    Example:
    ```mlir
    ora.mstore8 %base, %offset, %value : (ptr, i256, i256) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$base, AnyType:$offset, AnyType:$value);

  let assemblyFormat = "$base `,` $offset `,` $value attr-dict `:` type($base) `,` type($offset) `,` type($value)";
}

def Ora_TLoadOp : Ora_Op<"tload",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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
  
  let assemblyFormat = "$key attr-dict `:` type($result)";
}

def Ora_TStoreOp : Ora_Op<"tstore",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store to transient storage";
  let description = [{
    Stores a value to transient storage.
    
    Example:
    ```mlir
    ora.tstore %value, "key" : (i256) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$value, StrAttr:$key);
  
  let assemblyFormat = "$value `,` $key attr-dict `:` type($value)";
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
  
  // Custom parser/printer needed - type must come before operand
  // let assemblyFormat = "$value `as` $pattern_type attr-dict `:` type($value) `->` type($result)";
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

  let traits = [Symbol, MemoryEffects<[MemWrite]>];
  
  // Custom parser/printer needed for type-attr syntax
  // let assemblyFormat = "$name `:` type-attr($repr_type) region($variants) attr-dict";
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
  
  // Custom parser/printer needed for '.' separator
  // let assemblyFormat = "$enum_name `.` $variant_name attr-dict `:` type($result)";
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

  let traits = [Symbol, MemoryEffects<[MemWrite]>];
  
  // Regions need custom parser/printer when combined with attributes
  // let assemblyFormat = "$name region($fields) attr-dict";
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
  
  let arguments = (ins AnyType:$struct_value, StrAttr:$field_name, AnyType:$value);
  
  let assemblyFormat = "$struct_value `,` $field_name `,` $value attr-dict `:` type($struct_value) `,` type($value)";
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
  
  // Variadic operands with AnyType need custom parser/printer
  // let assemblyFormat = "$struct_name `(` $field_values `)` attr-dict `:` type($field_values) `->` type($result)";
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
  
  let assemblyFormat = "`{` $field_values `}` attr-dict `:` type($field_values) `->` type($result)";
}

// Extract field from struct (pure operation)
def Ora_StructFieldExtractOp : Ora_PureOp<"struct_field_extract"> {
  let summary = "Extract struct field";
  let description = [{
    Extracts a field value from a struct.
    
    Example:
    ```mlir
    %x = ora.struct_field_extract %point, "x" : (!ora.struct<Point>) -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$struct_value, StrAttr:$field_name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$struct_value `,` $field_name attr-dict `:` type($struct_value) `->` type($result)";
}

// Struct type constraint for operations
def Ora_StructType : TypeConstraint<
  CPred<"::llvm::isa<ora::StructType>($_self)">,
  "ora.struct">;

// Update struct field (pure operation - returns new struct)
// Result type must match input struct type (enforced in C++ builder)
def Ora_StructFieldUpdateOp : Ora_PureOp<"struct_field_update"> {
  let summary = "Update struct field";
  let description = [{
    Creates a new struct with an updated field value. The result type is always
    the same as the input struct type (inferred from struct_value in C++).
    
    Example:
    ```mlir
    %new_point = ora.struct_field_update %point, "x", %new_x : (!ora.struct<Point>, i256) -> !ora.struct<Point>
    ```
  }];
  
  let arguments = (ins Ora_StructType:$struct_value, StrAttr:$field_name, AnyType:$value);
  let results = (outs Ora_StructType:$result);
  let assemblyFormat = "$struct_value `,` $field_name `,` $value attr-dict `:` type($struct_value) `,` type($value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Map Operations
//===----------------------------------------------------------------------===//

def Ora_MapGetOp : Ora_Op<"map_get",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
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
  
  let assemblyFormat = "$map `,` $key attr-dict `:` type($map) `,` type($key) `->` type($result)";
}

def Ora_MapStoreOp : Ora_Op<"map_store",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Map store operation";
  let description = [{
    Stores a value in a map using a key.
    
    Example:
    ```mlir
    ora.map_store %map, %key, %value : (!ora.map<i256, i256>, i256, i256) -> ()
    ```
  }];
  
  let arguments = (ins AnyType:$map, AnyType:$key, AnyType:$value);
  
  let assemblyFormat = "$map `,` $key `,` $value attr-dict `:` type($map) `,` type($key) `,` type($value)";
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
  
  let assemblyFormat = "$condition attr-dict `:` type($condition)";
  let traits = [MemoryEffects<[MemWrite]>];
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
  
  let assemblyFormat = "$condition attr-dict `:` type($condition)";
  let traits = [MemoryEffects<[MemWrite]>];
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
  
  let assemblyFormat = "$condition attr-dict `:` type($condition)";
  let traits = [MemoryEffects<[MemWrite]>];
}

def Ora_AssertOp : Ora_Op<"assert"> {
  let summary = "Assertion operation";
  let description = [{
    Asserts a condition that must be true. Can be runtime or ghost (specification-only).
    
    Example:
    ```mlir
    ora.assert %condition : i1
    ```
  }];
  
  let arguments = (ins I1:$condition, OptionalAttr<StrAttr>:$message);
  
  let assemblyFormat = "$condition (`,` $message^)? attr-dict `:` type($condition)";
  let traits = [MemoryEffects<[MemWrite]>];
}

def Ora_RefinementGuardOp : Ora_Op<"refinement_guard"> {
  let summary = "Refinement guard operation";
  let description = [{
    Asserts a refinement constraint that may be discharged by SMT or emitted as
    a runtime guard.

    Example:
    ```mlir
    ora.refinement_guard %condition {ora.guard_id = "guard:foo"} : i1
    ```
  }];

  let arguments = (ins I1:$condition, OptionalAttr<StrAttr>:$message);

  let assemblyFormat = "$condition (`,` $message^)? attr-dict `:` type($condition)";
  let traits = [MemoryEffects<[MemWrite]>];
}

def Ora_DecreasesOp : Ora_Op<"decreases"> {
  let summary = "Loop termination measure";
  let description = [{
    Specifies a termination measure that must decrease on each loop iteration.
    Used for formal verification to prove loop termination.
    
    Example:
    ```mlir
    ora.decreases %measure : i256
    ```
  }];
  
  let arguments = (ins AnyType:$measure);
  
  let assemblyFormat = "$measure attr-dict `:` type($measure)";
  let traits = [MemoryEffects<[MemWrite]>];
}

def Ora_IncreasesOp : Ora_Op<"increases"> {
  let summary = "Loop progress measure";
  let description = [{
    Specifies a progress measure that must increase on each loop iteration.
    Used for formal verification to prove loop progress.
    
    Example:
    ```mlir
    ora.increases %measure : i256
    ```
  }];
  
  let arguments = (ins AnyType:$measure);
  
  let assemblyFormat = "$measure attr-dict `:` type($measure)";
  let traits = [MemoryEffects<[MemWrite]>];
}

def Ora_AssumeOp : Ora_Op<"assume"> {
  let summary = "Assume operation";
  let description = [{
    Assumes a condition for formal verification. The condition is assumed to be true
    for subsequent verification steps.
    
    Example:
    ```mlir
    ora.assume %condition : i1
    ```
  }];
  
  let arguments = (ins I1:$condition);
  
  let assemblyFormat = "$condition attr-dict `:` type($condition)";
  let traits = [MemoryEffects<[MemWrite]>];
}

def Ora_HavocOp : Ora_Op<"havoc"> {
  let summary = "Havoc operation";
  let description = [{
    Models unknown state for a variable. The variable can take any value after havoc.
    Used for formal verification to model non-deterministic behavior.
    
    Example:
    ```mlir
    ora.havoc "variable_name"
    ```
  }];
  
  let arguments = (ins StrAttr:$variable_name);
  
  let assemblyFormat = "$variable_name attr-dict";
  let traits = [MemoryEffects<[MemWrite]>];
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
  
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def Ora_QuantifiedOp : Ora_PureOp<"quantified"> {
  let summary = "Quantified expression (forall/exists)";
  let description = [{
    Represents a quantified expression for formal verification.
    Supports forall and exists quantifiers with bound variables and optional conditions.
    
    Example:
    ```mlir
    %result = ora.quantified "forall", "i", "i256", %body : i1 -> i1
    ```
  }];
  
  let arguments = (ins StrAttr:$quantifier, StrAttr:$variable, StrAttr:$variable_type, Optional<I1>:$condition, I1:$body);
  let results = (outs I1:$result);
  
  let assemblyFormat = "$quantifier `,` $variable `,` $variable_type (`,` $condition^)? `,` $body attr-dict `:` type($body) `->` type($result)";
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
  
  // let assemblyFormat = "$amount `from` $source `to` $destination attr-dict `:` type($amount) `,` type($source) `,` type($destination)";
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
  
  // let assemblyFormat = "$event_name `(` $parameters `)` attr-dict `:` type($parameters)";

  let traits = [MemoryEffects<[MemWrite]>];
}

def Ora_LogDeclOp : Ora_Op<"log.decl"> {
  let summary = "Event log declaration";
  let description = [{
    Declares an event log signature within a contract.

    Example:
    ```mlir
    "ora.log.decl"() {sym_name = "Transfer", ora.field_names = ["from"], ora.field_types = [!ora.address]}
    ```
  }];

  let arguments = (ins StrAttr:$sym_name);
  let traits = [Symbol, MemoryEffects<[MemWrite]>];
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
  
  let assemblyFormat = "$resource attr-dict `:` type($resource)";
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
  
  let assemblyFormat = "$resource attr-dict `:` type($resource)";
}

//===----------------------------------------------------------------------===//
// Error Handling Operations
//===----------------------------------------------------------------------===//

def Ora_ErrorDeclOp : Ora_Op<"error.decl"> {
  let summary = "Error declaration";
  let description = [{
    Declares an error variant within a contract.

    Example:
    ```mlir
    "ora.error.decl"() {sym_name = "E1", ora.error_id = 1 : i32, ora.error_decl = true}
    ```
  }];

  let arguments = (ins StrAttr:$sym_name);
  let traits = [Symbol, MemoryEffects<[MemWrite]>];
}

def Ora_ErrorOkOp : Ora_PureOp<"error.ok"> {
  let summary = "Construct a successful error union value";
  let description = [{
    Wraps a success value into an !ora.error_union<T>.
    
    Example:
    ```mlir
    %ok = ora.error.ok %value : i256 -> !ora.error_union<i256>
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
  
  let extraClassDeclaration = [{
    ::mlir::Value getSuccessValue() { return getValue(); }
  }];
}

def Ora_ErrorErrOp : Ora_PureOp<"error.err"> {
  let summary = "Construct an error error-union value";
  let description = [{
    Wraps an error payload into an !ora.error_union<T>.
    The result type specifies the success type parameter.
    
    Example:
    ```mlir
    %err = ora.error.err %code : i256 -> !ora.error_union<i256>
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
  
  let extraClassDeclaration = [{
    ::mlir::Value getErrorValue() { return getValue(); }
  }];
}

def Ora_ErrorIsErrorOp : Ora_PureOp<"error.is_error"> {
  let summary = "Check whether an error union holds an error";
  let description = [{
    Returns true if the error union contains an error payload.
    
    Example:
    ```mlir
    %is_err = ora.error.is_error %value : !ora.error_union<i256> -> i1
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs I1:$result);
  
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def Ora_ErrorUnwrapOp : Ora_PureOp<"error.unwrap"> {
  let summary = "Unwrap a success value from an error union";
  let description = [{
    Extracts the success value from an !ora.error_union<T>.
    Behavior is undefined if the value contains an error payload.
    
    Example:
    ```mlir
    %value = ora.error.unwrap %result : !ora.error_union<i256> -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def Ora_ErrorGetErrorOp : Ora_PureOp<"error.get_error"> {
  let summary = "Extract the error payload from an error union";
  let description = [{
    Extracts the error payload from an !ora.error_union<T>.
    Behavior is undefined if the value contains a success payload.
    
    Example:
    ```mlir
    %code = ora.error.get_error %result : !ora.error_union<i256> -> i256
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

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
  let regions = (region AnyRegion:$try_region,
                        AnyRegion:$catch_region);
  let traits = [RecursiveMemoryEffects];
  
  // let assemblyFormat = "$try_operation attr-dict `:` type($try_operation) `->` type($result) `catch` region($catch_region)";
}

def Ora_TryStmtOp : Ora_Op<"try_stmt"> {
  let summary = "Try-catch statement";
  let description = [{
    Executes a try region and, on error, executes a catch region.
    Regions may yield a value when used in return-oriented lowering.

    Example:
    ```mlir
    %result = ora.try_stmt : i256 {
      %v = arith.constant 1 : i256
      ora.yield %v : i256
    } catch {
      %v = arith.constant 0 : i256
      ora.yield %v : i256
    }
    ```
  }];

  let arguments = (ins);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$try_region,
                        AnyRegion:$catch_region);
  let traits = [RecursiveMemoryEffects];
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
  
  let hasCustomAssemblyFormat = 1;
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
  
  // Custom parser/printer needed for '|' separator syntax
  // let assemblyFormat = "$collection `|` `%key` `,` `%value` `|` region($body) attr-dict";
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
  
  // Optional variadic operands need custom parser/printer
  // let assemblyFormat = "($label^)? ($value^ `:` type($value))? attr-dict";
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
  
  let assemblyFormat = "($label^)? attr-dict";
}

//===----------------------------------------------------------------------===//
// Return Operations
//===----------------------------------------------------------------------===//

def Ora_ReturnOp : Ora_Op<"return", [Terminator]> {
  let summary = "Return operation";
  let description = [{
    Returns from a function with optional return values.
    
    Example:
    ```mlir
    ora.return %value : i256
    ```
  }];
  
  let arguments = (ins Variadic<AnyType>:$operands);
  
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// Switch Operations
//===----------------------------------------------------------------------===//

def Ora_SwitchOp : Ora_Op<"switch", [RecursiveMemoryEffects]> {
  let summary = "Switch operation";
  let description = [{
    Executes one of multiple regions based on a value.
    Supports literal case values, range patterns, and default/else case.
    
    Example:
    ```mlir
    %result = ora.switch %value : i256 -> i256 {
      case 0 => {
        %val0 = arith.constant 0 : i256
        ora.yield %val0 : i256
      }
      case 1...10 => {
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
  
  // Attributes for case patterns
  // case_values: Array of literal case values (for literal patterns)
  // range_starts: Array of range start values (for range patterns)
  // range_ends: Array of range end values (for range patterns)
  // case_kinds: Array indicating case type: 0=literal, 1=range, 2=else
  // default_case_index: Index of the default/else case (-1 if none)
  let extraClassDeclaration = [{
    // Accessor methods for case pattern attributes
    ::mlir::DenseI64ArrayAttr getCaseValuesAttr() {
      return (*this)->getAttrOfType<::mlir::DenseI64ArrayAttr>("case_values");
    }
    ::mlir::DenseI64ArrayAttr getRangeStartsAttr() {
      return (*this)->getAttrOfType<::mlir::DenseI64ArrayAttr>("range_starts");
    }
    ::mlir::DenseI64ArrayAttr getRangeEndsAttr() {
      return (*this)->getAttrOfType<::mlir::DenseI64ArrayAttr>("range_ends");
    }
    ::mlir::DenseI64ArrayAttr getCaseKindsAttr() {
      return (*this)->getAttrOfType<::mlir::DenseI64ArrayAttr>("case_kinds");
    }
    ::mlir::IntegerAttr getDefaultCaseIndexAttr() {
      return (*this)->getAttrOfType<::mlir::IntegerAttr>("default_case_index");
    }
  }];
  
  let hasCustomAssemblyFormat = 1;
}

def Ora_SwitchExprOp : Ora_Op<"switch_expr", [RecursiveMemoryEffects]> {
  let summary = "Switch expression";
  let description = [{
    Switch expression that returns a value based on pattern matching.
    Supports labeled continue with value replacement.
    
    Example:
    ```mlir
    %result = ora.switch_expr %value : i256 -> i256 {
      case 0 => {
        ora.yield %val0 : i256
      }
      case 1 => {
        continue :label %new_value : i256
      }
      else => {
        ora.yield %val_default : i256
      }
    }
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$cases);
  
  let extraClassDeclaration = [{
    ::mlir::DenseI64ArrayAttr getCaseValuesAttr() {
      return (*this)->getAttrOfType<::mlir::DenseI64ArrayAttr>("case_values");
    }
    ::mlir::DenseI64ArrayAttr getRangeStartsAttr() {
      return (*this)->getAttrOfType<::mlir::DenseI64ArrayAttr>("range_starts");
    }
    ::mlir::DenseI64ArrayAttr getRangeEndsAttr() {
      return (*this)->getAttrOfType<::mlir::DenseI64ArrayAttr>("range_ends");
    }
    ::mlir::DenseI64ArrayAttr getCaseKindsAttr() {
      return (*this)->getAttrOfType<::mlir::DenseI64ArrayAttr>("case_kinds");
    }
    ::mlir::IntegerAttr getDefaultCaseIndexAttr() {
      return (*this)->getAttrOfType<::mlir::IntegerAttr>("default_case_index");
    }
  }];
  
  let hasCustomAssemblyFormat = 1;
}

#endif // ORA_OPS
