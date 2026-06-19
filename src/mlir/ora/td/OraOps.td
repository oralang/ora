//===- OraOps.td - Ora dialect operations ---------------*- tablegen -*-===//
//
// This file defines the operations for the Ora MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_OPS
#define ORA_OPS

include "OraDialect.td"
include "OraTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Ora operation type constraints
//===----------------------------------------------------------------------===//

def Ora_IntegerValueType : TypeConstraint<
  CPred<"::llvm::isa<ora::IntegerType>($_self)">,
  "ora.int">;

def Ora_IntegerLikeValueType : TypeConstraint<
  CPred<"::llvm::isa<ora::IntegerType, ::mlir::IntegerType>($_self)">,
  "ora.int or builtin integer">;

def Ora_BoolValueType : TypeConstraint<
  CPred<"::llvm::isa<ora::BoolType>($_self)">,
  "ora.bool">;

def Ora_AddressValueType : TypeConstraint<
  CPred<"::llvm::isa<ora::AddressType>($_self)">,
  "ora.address">;

def Ora_I160ValueType : TypeConstraint<
  CPred<"::llvm::isa<::mlir::IntegerType>($_self) && ::llvm::cast<::mlir::IntegerType>($_self).getWidth() == 160">,
  "builtin i160">;

def Ora_RefinementValueType : TypeConstraint<
  CPred<"::llvm::isa<ora::MinValueType, ora::MaxValueType, ora::InRangeType, ora::ScaledType, ora::ExactType, ora::NonZeroAddressType>($_self)">,
  "Ora refinement type">;

def Ora_RefinementBaseValueType : TypeConstraint<
  CPred<"::llvm::isa<ora::IntegerType, ::mlir::IntegerType, ora::AddressType>($_self)">,
  "ora.int, builtin integer, or ora.address">;

def Ora_ComparableValueType : TypeConstraint<
  CPred<"::llvm::isa<ora::IntegerType, ::mlir::IntegerType, ora::BoolType, ora::AddressType, ora::StringType, ora::BytesType>($_self)">,
  "ora.int, builtin integer, ora.bool, ora.address, ora.string, or ora.bytes">;

// Common `lhs, rhs -> result` integer operation shape.
class Ora_BinaryIntegerOp<string mnemonic, string summaryText,
                          TypeConstraint operandConstraint,
                          list<Trait> traits = []> :
    Ora_PureOp<mnemonic, !listconcat([AllTypesMatch<["lhs", "rhs", "result"]>], traits)> {
  let hasCanonicalizer = 1;
  let summary = summaryText;
  let arguments = (ins operandConstraint:$lhs, operandConstraint:$rhs);
  let results = (outs operandConstraint:$result);
  let assemblyFormat =
    "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

class Ora_CheckedBinaryIntegerOp<string mnemonic, string summaryText> :
    Ora_BinaryIntegerOp<mnemonic, summaryText, Ora_IntegerValueType,
                        [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>;

class Ora_VerifiedBinaryIntegerOp<string mnemonic, string summaryText> :
    Ora_BinaryIntegerOp<mnemonic, summaryText, Ora_IntegerValueType> {
  let hasVerifier = 1;
}

class Ora_StringAttrConstantOp<string mnemonic, string summaryText> :
    Ora_PureOp<mnemonic> {
  let summary = summaryText;
  let arguments = (ins StrAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

class Ora_UnaryValueOp<string mnemonic, string summaryText,
                       TypeConstraint operandConstraint,
                       TypeConstraint resultConstraint,
                       bit canonicalizer = 0,
                       bit verifier = 0> :
    Ora_PureOp<mnemonic> {
  let summary = summaryText;
  let arguments = (ins operandConstraint:$value);
  let results = (outs resultConstraint:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
  let hasCanonicalizer = canonicalizer;
  let hasVerifier = verifier;
}

class Ora_TypedSymbolDeclOp<string mnemonic, string summaryText> :
    Ora_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins StrAttr:$sym_name, TypeAttr:$type);
  let traits = [Symbol, MemoryEffects<[MemWrite]>];
}

class Ora_SymbolDeclOp<string mnemonic, string summaryText> :
    Ora_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins StrAttr:$sym_name);
  let traits = [Symbol, MemoryEffects<[MemWrite]>];
}

class Ora_EvmPureNullaryOp<string mnemonic, string summaryText> :
    Ora_PureOp<mnemonic> {
  let summary = summaryText;
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

class Ora_ConditionMarkerOp<string mnemonic, string summaryText> :
    Ora_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins I1:$condition);
  let assemblyFormat = "$condition attr-dict `:` type($condition)";
  let traits = [MemoryEffects<[MemWrite]>];
}

class Ora_MessageConditionMarkerOp<string mnemonic, string summaryText> :
    Ora_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins I1:$condition, OptionalAttr<StrAttr>:$message);
  let assemblyFormat = "$condition (`,` $message^)? attr-dict `:` type($condition)";
  let traits = [MemoryEffects<[MemWrite]>];
}

class Ora_MeasureMarkerOp<string mnemonic, string summaryText> :
    Ora_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins AnyType:$measure);
  let assemblyFormat = "$measure attr-dict `:` type($measure)";
  let traits = [MemoryEffects<[MemWrite]>];
}

class Ora_LockStateOp<string mnemonic, string summaryText> :
    Ora_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins AnyType:$resource, OptionalAttr<StrAttr>:$key);
  let assemblyFormat = "$resource attr-dict `:` type($resource)";
}

class Ora_VariadicOperandTerminatorOp<string mnemonic, string summaryText,
                                      bit verifier = 0> :
    Ora_Op<mnemonic, [Terminator]> {
  let summary = summaryText;
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let hasVerifier = verifier;
}

class Ora_ResultFormattedOp<string mnemonic, string summaryText,
                            dag operands, string format,
                            list<Trait> traits = []> :
    Ora_Op<mnemonic, traits> {
  let summary = summaryText;
  let arguments = operands;
  let results = (outs AnyType:$result);
  let assemblyFormat = format;
}

class Ora_NoResultFormattedOp<string mnemonic, string summaryText,
                              dag operands, string format,
                              list<Trait> traits = []> :
    Ora_Op<mnemonic, traits> {
  let summary = summaryText;
  let arguments = operands;
  let assemblyFormat = format;
}

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
  
  let arguments = (ins Ora_AddressValueType:$addr);
  let results = (outs Ora_I160ValueType:$result);
  
  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($result)";
  let hasFolder = 1;
  
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
  
  let arguments = (ins Ora_I160ValueType:$i160);
  let results = (outs Ora_AddressValueType:$result);
  
  let assemblyFormat = "$i160 attr-dict `:` type($i160) `->` type($result)";
  let hasFolder = 1;
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
  
  let arguments = (ins Ora_RefinementValueType:$value);
  let results = (outs Ora_RefinementBaseValueType:$result);
  
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
  
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

  let arguments = (ins Ora_RefinementBaseValueType:$value);
  let results = (outs Ora_RefinementValueType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
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
  let hasVerifier = 1;
  
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
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Get the value being stored
    ::mlir::Value getStoredValue() { return getValue(); }
    
    /// Get the name of the global variable being stored to
    ::llvm::StringRef getGlobalName() { return getGlobal(); }
  }];
}

def Ora_StorageDeriveOp : Ora_PureOp<"storage.derive"> {
  let summary = "Derive an opaque computed-storage slot";
  let description = [{
    Derives a compiler-owned computed-storage slot from a compile-time namespace
    hash and zero or more scalar runtime keys. The namespace string is retained
    for diagnostics, source maps, and layout manifests; lowering hashes a
    domain-separated preimage containing a fixed Ora computed-storage tag, the
    key count, the namespace_hash attribute, and the keys.

    Example:
    ```mlir
    %slot = ora.storage.derive "vault.position"(%owner : !ora.address)
      {namespace_hash = 123 : i256} : (!ora.address) -> !ora.int<256, false>
    ```
  }];

  let arguments = (ins StrAttr:$namespace_name, AnyAttr:$namespace_hash, Variadic<AnyType>:$keys);
  let results = (outs AnyType:$slot);

  let assemblyFormat = "$namespace_name `(` $keys `)` attr-dict `:` functional-type($keys, results)";
  let hasVerifier = 1;
}

def Ora_StorageWordLoadOp : Ora_Op<"storage.word_load",
    [MemoryEffects<[MemRead]>]> {
  let summary = "Load one word from a computed-storage capability";
  let description = [{
    Loads a single word from an opaque computed-storage slot plus a word offset.
    The source language can construct the slot only through compiler-owned
    derivation primitives, not integer casts.

    Example:
    ```mlir
    %word = ora.storage.word_load %slot, %offset : i256, i256 -> i256
    ```
  }];

  let arguments = (ins AnyType:$slot, AnyType:$offset);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$slot `,` $offset attr-dict `:` type($slot) `,` type($offset) `->` type($result)";
  let hasVerifier = 1;
}

def Ora_StorageWordStoreOp : Ora_Op<"storage.word_store",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Store one word through a computed-storage capability";
  let description = [{
    Stores a single word to an opaque computed-storage slot plus a word offset.
    Source programs can construct the slot only through compiler-owned
    derivation primitives, and this op lowers through the normal SIR storage
    path rather than exposing raw EVM assembly.

    Example:
    ```mlir
    ora.storage.word_store %slot, %offset, %value : i256, i256, i256
    ```
  }];

  let arguments = (ins AnyType:$slot, AnyType:$offset, AnyType:$value);

  let assemblyFormat = "$slot `,` $offset `,` $value attr-dict `:` type($slot) `,` type($offset) `,` type($value)";
  let hasVerifier = 1;
}

def Ora_StorageRangeEraseOp : Ora_Op<"storage.range_erase",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Erase a bounded computed-storage word range";
  let description = [{
    Erases a bounded computed-storage word range by zeroing every physical word
    in `[slot, slot + word_count)`. The range is represented in source as an
    opaque StorageRange capability, so user code cannot forge it with integer
    casts. The lowering contract is intentionally stricter than the value
    operands: `word_count` must be present as a compile-time attribute, and the
    converter must reject the operation if only a runtime length is available.
    Physical slots are computed as `slot + offset`; source and HIR require
    computed-storage ranges to be statically bounded, and OraToSIR rejects any
    malformed unbounded range before bytecode emission.

    Example:
    ```mlir
    ora.storage.range_erase %slot {word_count = 4 : i64} : i256
    ```
  }];

  let arguments = (ins AnyType:$slot, I64Attr:$word_count);

  let assemblyFormat = "$slot attr-dict `:` type($slot)";
  let hasVerifier = 1;
}

def Ora_MemoryGlobalOp : Ora_TypedSymbolDeclOp<"memory.global",
    "Memory global variable declaration"> {
  let summary = "Memory global variable declaration";
  let description = [{
    Declares a contract-level memory variable used by Ora lowering.

    Example:
    ```mlir
    "ora.memory.global"() {sym_name = "tmp", type = i256}
    ```
  }];
}

def Ora_TStoreGlobalOp : Ora_TypedSymbolDeclOp<"tstore.global",
    "Transient storage global variable declaration"> {
  let summary = "Transient storage global variable declaration";
  let description = [{
    Declares a contract-level transient storage variable used by Ora lowering.

    Example:
    ```mlir
    "ora.tstore.global"() {sym_name = "tmp", type = i256}
    ```
  }];
}

def Ora_BitfieldDeclOp : Ora_SymbolDeclOp<"bitfield_decl",
    "Bitfield declaration"> {
  let summary = "Bitfield declaration";
  let description = [{
    Declares a bitfield type with optional layout metadata used by Ora lowering.

    Example:
    ```mlir
    "ora.bitfield_decl"() {sym_name = "Flags", ora.bitfield = "Flags", ora.bitfield_layout = "u1:enabled,u7:mode"}
    ```
  }];
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

def Ora_StringConstantOp : Ora_StringAttrConstantOp<"string.constant", "String constant"> {
  let summary = "String constant";
  let description = [{
    Creates a string constant value.
    
    Example:
    ```mlir
    %str = ora.string.constant "hello world" : !ora.string
    ```
  }];
}

def Ora_BytesConstantOp : Ora_StringAttrConstantOp<"bytes.constant", "Bytes constant"> {
  let summary = "Bytes constant";
  let description = [{
    Creates a bytes constant value.
    
    Example:
    ```mlir
    %bytes = ora.bytes.constant "0xdeadbeef" : !ora.bytes
    ```
  }];
}

def Ora_LengthOp : Ora_UnaryValueOp<"length",
    "Length of a dynamic string or bytes value", AnyType, AnyType> {
  let summary = "Length of a dynamic string or bytes value";
  let description = [{
    Reads the runtime length header of an Ora `string` or `bytes` value.

    Example:
    ```mlir
    %len = ora.length %value : !ora.bytes -> i256
    ```
  }];
}

def Ora_ByteAtOp : Ora_PureOp<"byte_at"> {
  let summary = "Byte access for dynamic string or bytes";
  let description = [{
    Reads the byte at the given runtime index from an Ora `string` or `bytes`
    value. The result is a raw integer byte value.

    Example:
    ```mlir
    %byte = ora.byte_at %value, %index : (!ora.bytes, i256) -> i256
    ```
  }];

  let arguments = (ins AnyType:$value, AnyType:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value `,` $index attr-dict `:` type($value) `,` type($index) `->` type($result)";
}

def Ora_ConcatOp : Ora_PureOp<"concat"> {
  let summary = "Concatenate dynamic string or bytes values";
  let description = [{
    Builds a new dynamic `string` or `bytes` value by concatenating two runtime
    values with the same dynamic byte layout.

    Example:
    ```mlir
    %joined = ora.concat %lhs, %rhs : !ora.bytes, !ora.bytes -> !ora.bytes
    ```
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Ora_SliceOp : Ora_PureOp<"slice"> {
  let summary = "Slice a dynamic string or bytes value";
  let description = [{
    Builds a new dynamic `string` or `bytes` value by copying `length` bytes
    from `value` starting at byte offset `start`.

    Example:
    ```mlir
    %part = ora.slice %value, %start, %length : !ora.bytes, i256, i256 -> !ora.bytes
    ```
  }];

  let arguments = (ins AnyType:$value, AnyType:$start, AnyType:$length);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value `,` $start `,` $length attr-dict `:` type($value) `,` type($start) `,` type($length) `->` type($result)";
}

def Ora_Keccak256Op : Ora_UnaryValueOp<"keccak256",
    "Runtime keccak256 over dynamic string or bytes", AnyType, AnyType> {
  let summary = "Runtime keccak256 over dynamic string or bytes";
  let description = [{
    Computes the EVM keccak256 hash of the runtime bytes contained in an Ora
    `string` or `bytes` value.

    Example:
    ```mlir
    %hash = ora.keccak256 %data : !ora.bytes -> i256
    ```
  }];
}


def Ora_HexConstantOp : Ora_StringAttrConstantOp<"hex.constant", "Hexadecimal constant"> {
  let summary = "Hexadecimal constant";
  let description = [{
    Creates a hexadecimal constant value.
    
    Example:
    ```mlir
    %hex = ora.hex.constant "0x1a2b" : !ora.hex
    ```
  }];
}


//===----------------------------------------------------------------------===//
// Range Operations
//===----------------------------------------------------------------------===//

def Ora_RangeOp : Ora_PureOp<"range", [AllTypesMatch<["start", "end", "result"]>]> {
  let summary = "Range operation";
  let description = [{
    Creates a range value from start to end, with optional inclusive flag.
    Used for range expressions like (0..20) or (0...20).
    
    Example:
    ```mlir
    %range = ora.range %start, %end {inclusive = false} : (i256, i256) -> i256
    ```
  }];
  
  let arguments = (ins Ora_IntegerValueType:$start, Ora_IntegerValueType:$end, OptionalAttr<BoolAttr>:$inclusive);
  let results = (outs Ora_IntegerValueType:$result);
  
  let assemblyFormat = "$start `,` $end attr-dict `:` type($start) `,` type($end) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def Ora_AddOp : Ora_BinaryIntegerOp<"add", "Add operation",
    Ora_IntegerLikeValueType,
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Add operation";
  let description = [{
    Adds two Ora integer values with proper unsigned semantics and overflow handling.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.add %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
}

def Ora_AddWrappingOp : Ora_BinaryIntegerOp<"add_wrapping",
    "Wrapping add operation", Ora_IntegerLikeValueType> {
  let summary = "Wrapping add operation";
  let description = [{
    Adds two Ora integer values with modular wrapping semantics.
    Accepts !ora.int<width, signed> types and preserves the type in the result.

    Example:
    ```mlir
    %result = ora.add_wrapping %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
}

def Ora_SubWrappingOp : Ora_BinaryIntegerOp<"sub_wrapping",
    "Wrapping subtract operation", Ora_IntegerLikeValueType> {
  let summary = "Wrapping subtract operation";
  let description = [{Subtracts two Ora integer values with modular wrapping semantics.}];
}

def Ora_MulWrappingOp : Ora_BinaryIntegerOp<"mul_wrapping",
    "Wrapping multiply operation", Ora_IntegerLikeValueType> {
  let summary = "Wrapping multiply operation";
  let description = [{Multiplies two Ora integer values with modular wrapping semantics.}];
}

def Ora_ShlWrappingOp : Ora_BinaryIntegerOp<"shl_wrapping",
    "Wrapping shift-left operation", Ora_IntegerLikeValueType> {
  let summary = "Wrapping shift-left operation";
  let description = [{Shifts an Ora integer left with modular wrapping semantics.}];
}

def Ora_ShrWrappingOp : Ora_BinaryIntegerOp<"shr_wrapping",
    "Wrapping shift-right operation", Ora_IntegerLikeValueType> {
  let summary = "Wrapping shift-right operation";
  let description = [{Shifts an Ora integer right with wrapping semantics, preserving signedness from the Ora type.}];
}

def Ora_SubOp : Ora_CheckedBinaryIntegerOp<"sub", "Subtract operation"> {
  let summary = "Subtract operation";
  let description = [{
    Subtracts two Ora integer values with proper unsigned semantics and overflow handling.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.sub %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
}

def Ora_MulOp : Ora_CheckedBinaryIntegerOp<"mul", "Multiply operation"> {
  let summary = "Multiply operation";
  let description = [{
    Multiplies two Ora integer values with proper unsigned semantics and overflow handling.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.mul %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
}

def Ora_DivOp : Ora_CheckedBinaryIntegerOp<"div", "Divide operation"> {
  let summary = "Divide operation";
  let description = [{
    Divides two Ora integer values with proper unsigned semantics.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.div %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
  let hasVerifier = 1;
}

def Ora_RemOp : Ora_VerifiedBinaryIntegerOp<"rem", "Remainder operation"> {
  let summary = "Remainder operation";
  let description = [{
    Computes the remainder of dividing two Ora integer values.
    Accepts !ora.int<width, signed> types and preserves the type in the result.
    
    Example:
    ```mlir
    %result = ora.rem %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
    ```
  }];
}

def Ora_CmpOp : Ora_PureOp<"cmp", [SameTypeOperands]> {
  let summary = "Compare operation";
  let description = [{
    Compares two Ora integer values with proper unsigned semantics.
    Accepts !ora.int<width, signed> types and returns i1 (boolean).
    
    Example:
    ```mlir
    %result = ora.cmp "eq", %lhs, %rhs : (!ora.int<256, false>, !ora.int<256, false>) -> i1
    ```
  }];
  
  let arguments = (ins StrAttr:$predicate, Ora_ComparableValueType:$lhs, Ora_ComparableValueType:$rhs);
  let results = (outs I1:$result);
  
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasCanonicalizer = 1;
}

def Ora_PowerOp : Ora_PureOp<"power", [AllTypesMatch<["base", "exponent", "result"]>]> {
  let hasCanonicalizer = 1;
  let summary = "Power operation";
  let description = [{
    Computes base raised to the power of exponent.
    
    Example:
    ```mlir
    %result = ora.power %base, %exp : (i256, i256) -> i256
    ```
  }];
  
  let arguments = (ins Ora_IntegerLikeValueType:$base, Ora_IntegerLikeValueType:$exponent);
  let results = (outs Ora_IntegerLikeValueType:$result);
  
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

def Ora_EvmOriginOp : Ora_EvmPureNullaryOp<"evm.origin", "EVM origin builtin">;

def Ora_EvmCallerOp : Ora_EvmPureNullaryOp<"evm.caller", "EVM caller builtin">;

def Ora_EvmGaspriceOp :
    Ora_EvmPureNullaryOp<"evm.gasprice", "EVM gasprice builtin">;

def Ora_EvmCallvalueOp :
    Ora_EvmPureNullaryOp<"evm.callvalue", "EVM callvalue builtin">;

def Ora_EvmGasOp : Ora_Op<"evm.gas", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "EVM gas builtin";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Ora_EvmTimestampOp :
    Ora_EvmPureNullaryOp<"evm.timestamp", "EVM timestamp builtin">;

def Ora_EvmNumberOp :
    Ora_EvmPureNullaryOp<"evm.number", "EVM block number builtin">;

def Ora_EvmCoinbaseOp :
    Ora_EvmPureNullaryOp<"evm.coinbase", "EVM coinbase builtin">;

def Ora_EvmDifficultyOp :
    Ora_EvmPureNullaryOp<"evm.difficulty", "EVM difficulty/prevrandao builtin">;

def Ora_EvmPrevrandaoOp :
    Ora_EvmPureNullaryOp<"evm.prevrandao", "EVM prevrandao builtin">;

def Ora_EvmGaslimitOp :
    Ora_EvmPureNullaryOp<"evm.gaslimit", "EVM gaslimit builtin">;

def Ora_EvmChainidOp :
    Ora_EvmPureNullaryOp<"evm.chainid", "EVM chainid builtin">;

def Ora_EvmBasefeeOp :
    Ora_EvmPureNullaryOp<"evm.basefee", "EVM basefee builtin">;

def Ora_FunctionRefOp : Ora_PureOp<"function_ref"> {
  let summary = "Reference a function symbol";
  let description = [{
    Materializes a first-class reference to an Ora function symbol.

    Example:
    ```mlir
    %f = ora.function_ref @helper : !ora.function<(i256) -> (i256)>
    ```
  }];

  let arguments = (ins StrAttr:$sym_name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$sym_name attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

def Ora_IfOp : Ora_Op<"conditional_return"> {
  let summary = "Conditional early return";
  let description = [{
    Conditional early return escape hatch for cases that cannot be expressed
    with scf.if because the then-region must terminate the enclosing function.
    This is not a general-purpose if op: it is reserved for early-return control
    flow. The then-region must terminate with func.return or ora.return, and the
    else-region must contain only an empty ora.yield terminator.
    
    Example:
    ```mlir
    ora.conditional_return %condition then {
      %err = ora.error.err %code : i256 -> !ora.error_union<i256>
      func.return %err : !ora.error_union<i256>
    } else {
      ora.yield
    }
    ```
  }];
  
  let arguments = (ins I1:$condition);
  let results = (outs);
  let regions = (region AnyRegion:$thenRegion,
                        AnyRegion:$elseRegion);
  
  // Custom parser/printer for prettier syntax with "then"/"else" keywords
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Ora_YieldOp : Ora_VariadicOperandTerminatorOp<"yield", "Yield operation"> {
  let summary = "Yield operation";
  let description = [{
    Yields values from a region.
    
    Example:
    ```mlir
    ora.yield %value : i256
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def Ora_MLoadOp : Ora_ResultFormattedOp<"mload", "Load from memory",
    (ins StrAttr:$variable), "$variable attr-dict `:` type($result)",
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
}

def Ora_MStoreOp : Ora_NoResultFormattedOp<"mstore", "Store to memory",
    (ins AnyType:$value, StrAttr:$variable),
    "$value `,` $variable attr-dict `:` type($value)",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store to memory";
  let description = [{
    Stores a value to memory region.
    
    Example:
    ```mlir
    ora.mstore %value, "variable" : (i256) -> ()
    ```
  }];
}

def Ora_MLoad8Op : Ora_ResultFormattedOp<"mload8", "Load a single byte from memory",
    (ins AnyType:$base, AnyType:$offset),
    "$base `,` $offset attr-dict `:` type($base) `,` type($offset) `->` type($result)",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load a single byte from memory";
  let description = [{
    Loads a single byte from memory at (base + offset).
    
    Example:
    ```mlir
    %byte = ora.mload8 %base, %offset : (ptr, i256) -> i256
    ```
  }];
}

def Ora_MStore8Op : Ora_NoResultFormattedOp<"mstore8", "Store a single byte to memory",
    (ins AnyType:$base, AnyType:$offset, AnyType:$value),
    "$base `,` $offset `,` $value attr-dict `:` type($base) `,` type($offset) `,` type($value)",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store a single byte to memory";
  let description = [{
    Stores a single byte to memory at (base + offset).
    
    Example:
    ```mlir
    ora.mstore8 %base, %offset, %value : (ptr, i256, i256) -> ()
    ```
  }];
}

def Ora_TLoadOp : Ora_ResultFormattedOp<"tload", "Load from transient storage",
    (ins StrAttr:$key), "$key attr-dict `:` type($result)",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load from transient storage";
  let description = [{
    Loads a value from transient storage.
    
    Example:
    ```mlir
    %value = ora.tload "key" : () -> i256
    ```
  }];
}

def Ora_TStoreOp : Ora_NoResultFormattedOp<"tstore", "Store to transient storage",
    (ins AnyType:$value, StrAttr:$key),
    "$value `,` $key attr-dict `:` type($value)",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store to transient storage";
  let description = [{
    Stores a value to transient storage.
    
    Example:
    ```mlir
    ora.tstore %value, "key" : (i256) -> ()
    ```
  }];
}

def Ora_TStoreGuardOp : Ora_Op<"tstore.guard"> {
  let summary = "Runtime guard: revert if TStore slot key is locked";
  let description = [{
    Emitted before a TStore write when the slot key is in the function's
    locked set. For indexed paths, `resource` carries the runtime index key.
    Lowers to a runtime check; reverts if the slot is locked.
  }];
  let arguments = (ins AnyType:$resource, StrAttr:$key);
  let assemblyFormat = "$resource `,` $key attr-dict `:` type($resource)";
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
// ADT Operations
//===----------------------------------------------------------------------===//

def Ora_AdtValueType : TypeConstraint<
  CPred<"::llvm::isa<ora::AdtType>($_self)">,
  "ora.adt">;

def Ora_AdtConstructOp : Ora_PureOp<"adt.construct"> {
  let summary = "Construct an ADT variant";
  let description = [{
    Constructs a payload-carrying enum/ADT value by variant name.

    Example:
    ```mlir
    %event = ora.adt.construct "Value"(%payload) : (!ora.int<256, false>) -> !ora.adt<"Event", ("Value", !ora.int<256, false>)>
    ```
  }];

  let arguments = (ins StrAttr:$variant_name, Variadic<AnyType>:$payload_values);
  let results = (outs Ora_AdtValueType:$result);

  let assemblyFormat = "$variant_name `(` $payload_values `)` attr-dict `:` functional-type($payload_values, results)";
  let hasVerifier = 1;
}

def Ora_AdtTagOp : Ora_UnaryValueOp<"adt.tag",
    "Extract an ADT variant tag", Ora_AdtValueType, AnyType, 1, 1> {
  let summary = "Extract an ADT variant tag";
  let description = [{
    Extracts the integer tag/discriminant from an ADT value.
  }];
}

def Ora_AdtPayloadOp : Ora_PureOp<"adt.payload"> {
  let summary = "Extract an ADT variant payload";
  let description = [{
    Extracts a payload value from an ADT value for the named variant. For Phase
    2b each variant has a single payload type; multi-field source payloads are
    represented by tuple or anonymous-struct payload types.
  }];

  let arguments = (ins Ora_AdtValueType:$value, StrAttr:$variant_name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value `,` $variant_name attr-dict `:` type($value) `->` type($result)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Ora_AdtMatchArmOp : Ora_Op<"adt.match_arm"> {
  let summary = "ADT match arm marker";
  let description = [{
    Represents one lowered ADT match arm for the named variant. Phase 3 will
    produce this from source-level structural matches; Phase 2b introduces the
    verifiable IR surface.
  }];

  let arguments = (ins Ora_AdtValueType:$value, StrAttr:$variant_name);
  let regions = (region AnyRegion:$body);

  let hasVerifier = 1;
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
  
  let arguments = (ins StrAttr:$sym_name);
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
  
  let hasCanonicalizer = 1;

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
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Ora_TupleCreateOp : Ora_PureOp<"tuple_create"> {
  let summary = "Tuple construction";
  let description = [{
    Creates a tuple value from element operands.

    Example:
    ```mlir
    %pair = ora.tuple_create (%lhs, %rhs) : (i256, i1) -> !ora.tuple<i256, i1>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $elements `)` attr-dict `:` type($elements) `->` type($result)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Ora_TupleExtractOp : Ora_PureOp<"tuple_extract"> {
  let summary = "Tuple element extraction";
  let description = [{
    Extracts an element from a tuple by constant index.

    Example:
    ```mlir
    %lhs = ora.tuple_extract %pair[0] : !ora.tuple<i256, i1> -> i256
    ```
  }];

  let arguments = (ins AnyType:$tuple_value, I64Attr:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$tuple_value `[` $index `]` attr-dict `:` type($tuple_value) `->` type($result)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
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
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

// Struct type constraint for operations
def Ora_StructValueType : TypeConstraint<
  CPred<"::llvm::isa<ora::StructType, ora::AnonymousStructType>($_self)">,
  "ora.struct or ora.struct_anon">;

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
  
  let arguments = (ins Ora_StructValueType:$struct_value, StrAttr:$field_name, AnyType:$value);
  let results = (outs Ora_StructValueType:$result);
  let assemblyFormat = "$struct_value `,` $field_name `,` $value attr-dict `:` type($struct_value) `,` type($value) `->` type($result)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Formal Verification Operations
//===----------------------------------------------------------------------===//

def Ora_RequiresOp : Ora_ConditionMarkerOp<"requires",
    "Precondition assertion"> {
  let summary = "Precondition assertion";
  let description = [{
    Asserts a precondition that must be true at function entry.
    
    Example:
    ```mlir
    ora.requires %condition : i1
    ```
  }];
}

def Ora_EnsuresOp : Ora_ConditionMarkerOp<"ensures",
    "Postcondition assertion"> {
  let summary = "Postcondition assertion";
  let description = [{
    Asserts a postcondition that must be true at function exit.
    
    Example:
    ```mlir
    ora.ensures %condition : i1
    ```
  }];
}

def Ora_InvariantOp : Ora_ConditionMarkerOp<"invariant",
    "Invariant assertion"> {
  let summary = "Invariant assertion";
  let description = [{
    Asserts an invariant that must be true at specific points.
    
    Example:
    ```mlir
    ora.invariant %condition : i1
    ```
  }];
}

def Ora_AssertOp : Ora_MessageConditionMarkerOp<"assert",
    "Assertion operation"> {
  let summary = "Assertion operation";
  let description = [{
    Asserts a condition that must be true. Can be runtime or ghost (specification-only).
    
    Example:
    ```mlir
    ora.assert %condition : i1
    ```
  }];
}

def Ora_RefinementGuardOp : Ora_MessageConditionMarkerOp<"refinement_guard",
    "Refinement guard operation"> {
  let summary = "Refinement guard operation";
  let description = [{
    Asserts a refinement constraint that may be discharged by SMT or emitted as
    a runtime guard.

    Example:
    ```mlir
    ora.refinement_guard %condition {ora.guard_id = "guard:foo"} : i1
    ```
  }];
}

def Ora_DecreasesOp : Ora_MeasureMarkerOp<"decreases",
    "Loop termination measure"> {
  let summary = "Loop termination measure";
  let description = [{
    Specifies a termination measure that must decrease on each loop iteration.
    Used for formal verification to prove loop termination.
    
    Example:
    ```mlir
    ora.decreases %measure : i256
    ```
  }];
}

def Ora_IncreasesOp : Ora_MeasureMarkerOp<"increases",
    "Loop progress measure"> {
  let summary = "Loop progress measure";
  let description = [{
    Specifies a progress measure that must increase on each loop iteration.
    Used for formal verification to prove loop progress.
    
    Example:
    ```mlir
    ora.increases %measure : i256
    ```
  }];
}

def Ora_AssumeOp : Ora_ConditionMarkerOp<"assume", "Assume operation"> {
  let summary = "Assume operation";
  let description = [{
    Assumes a condition for formal verification. The condition is assumed to be true
    for subsequent verification steps.
    
    Example:
    ```mlir
    ora.assume %condition : i1
    ```
  }];
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

def Ora_OldOp : Ora_UnaryValueOp<"old", "Old value reference", AnyType, AnyType> {
  let summary = "Old value reference";
  let description = [{
    References the old value of a variable for formal verification.
    
    Example:
    ```mlir
    %old_val = ora.old %current_val : i256 -> i256
    ```
  }];
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
// Resource Operations
//===----------------------------------------------------------------------===//

class Ora_ResourceBoundaryOp<string mnemonic, string summaryText> :
    Ora_Op<mnemonic, [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = summaryText;
  let arguments = (ins
    Variadic<AnyType>:$place,
    AnyType:$amount,
    StrAttr:$domain,
    TypeAttr:$carrier_type
  );
  let hasVerifier = 1;
}

def Ora_MoveOp : Ora_Op<"move", [AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Move quantity between resource places";
  let description = [{
    Moves an amount between two Resource<T> storage places in the same resource
    domain. The source_place and destination_place segments are place paths, not
    copied values. The first value in each segment is the map root; subsequent
    values are the evaluated map keys.

    Example:
    ```mlir
    "ora.move"(%balances, %from, %balances, %to, %amount)
      <{operand_segment_sizes = array<i32: 2, 2, 1>,
        domain = "TokenUnit",
        carrier_type = !ora.int<256, false>}>
      : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address,
         !ora.map<!ora.address, !ora.int<256, false>>, !ora.address,
         !ora.int<256, false>) -> ()
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$source_place,
    Variadic<AnyType>:$destination_place,
    AnyType:$amount,
    StrAttr:$domain,
    TypeAttr:$carrier_type
  );
  let hasVerifier = 1;
}

def Ora_CreateOp : Ora_ResourceBoundaryOp<"create", "Create quantity in a resource place"> {
  let description = [{
    Creates an amount in a Resource<T> storage place as an explicit domain
    boundary delta. The place segment is a place path, not a copied value.
  }];
}

def Ora_DestroyOp : Ora_ResourceBoundaryOp<"destroy", "Destroy quantity from a resource place"> {
  let description = [{
    Destroys an amount from a Resource<T> storage place as an explicit domain
    boundary delta. The place segment is a place path, not a copied value.
  }];
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

def Ora_LogDeclOp : Ora_SymbolDeclOp<"log.decl", "Event log declaration"> {
  let summary = "Event log declaration";
  let description = [{
    Declares an event log signature within a contract.

    Example:
    ```mlir
    "ora.log.decl"() {sym_name = "Transfer", ora.field_names = ["from"], ora.field_types = [!ora.address]}
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Locking Operations
//===----------------------------------------------------------------------===//

def Ora_LockOp : Ora_LockStateOp<"lock", "Lock operation"> {
  let summary = "Lock operation";
  let description = [{
    Acquires a lock on the specified TStore slot (key used for OraToSIR slot lookup).
  }];
}

def Ora_UnlockOp : Ora_LockStateOp<"unlock", "Unlock operation"> {
  let summary = "Unlock operation";
  let description = [{
    Releases a lock on the specified TStore slot (key used for OraToSIR slot lookup).
  }];
}

//===----------------------------------------------------------------------===//
// Error Handling Operations
//===----------------------------------------------------------------------===//

def Ora_ErrorDeclOp : Ora_SymbolDeclOp<"error.decl", "Error declaration"> {
  let summary = "Error declaration";
  let description = [{
    Declares an error variant within a contract.

    Example:
    ```mlir
    "ora.error.decl"() {sym_name = "E1", ora.error_id = 1 : i32, ora.error_decl = true}
    ```
  }];
}

def Ora_ErrorReturnOp : Ora_PureOp<"error.return"> {
  let summary = "Construct a nominal error value";
  let description = [{
    Materializes a named error constructor expression with zero or more payload operands.

    Example:
    ```mlir
    %err = ora.error.return "Failure"(%code : i256) : !ora.struct<"Failure">
    ```
  }];

  let arguments = (ins StrAttr:$sym_name, Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$sym_name (`(` $operands^ `:` type($operands) `)`)? attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Ora_ErrorOkOp : Ora_UnaryValueOp<"error.ok",
    "Construct a successful error union value", AnyType, AnyType> {
  let summary = "Construct a successful error union value";
  let description = [{
    Wraps a success value into an !ora.error_union<T>.
    
    Example:
    ```mlir
    %ok = ora.error.ok %value : i256 -> !ora.error_union<i256>
    ```
  }];
  
  let extraClassDeclaration = [{
    ::mlir::Value getSuccessValue() { return getValue(); }
  }];
}

def Ora_ErrorErrOp : Ora_UnaryValueOp<"error.err",
    "Construct an error error-union value", AnyType, AnyType> {
  let summary = "Construct an error error-union value";
  let description = [{
    Wraps an error payload into an !ora.error_union<T>.
    The result type specifies the success type parameter.
    
    Example:
    ```mlir
    %err = ora.error.err %code : i256 -> !ora.error_union<i256>
    ```
  }];
  
  let extraClassDeclaration = [{
    ::mlir::Value getErrorValue() { return getValue(); }
  }];
}

def Ora_ErrorIsErrorOp : Ora_UnaryValueOp<"error.is_error",
    "Check whether an error union holds an error", AnyType, I1, 1> {
  let summary = "Check whether an error union holds an error";
  let description = [{
    Returns true if the error union contains an error payload.
    
    Example:
    ```mlir
    %is_err = ora.error.is_error %value : !ora.error_union<i256> -> i1
    ```
  }];
}

def Ora_ErrorUnwrapOp : Ora_UnaryValueOp<"error.unwrap",
    "Unwrap a success value from an error union", AnyType, AnyType, 1, 1> {
  let summary = "Unwrap a success value from an error union";
  let description = [{
    Extracts the success value from an !ora.error_union<T>.
    Behavior is undefined if the value contains an error payload.
    
    Example:
    ```mlir
    %value = ora.error.unwrap %result : !ora.error_union<i256> -> i256
    ```
  }];
}

def Ora_ErrorGetErrorOp : Ora_UnaryValueOp<"error.get_error",
    "Extract the error payload from an error union", AnyType, AnyType, 1> {
  let summary = "Extract the error payload from an error union";
  let description = [{
    Extracts the error payload from an !ora.error_union<T>.
    Behavior is undefined if the value contains a success payload.
    
    Example:
    ```mlir
    %code = ora.error.get_error %result : !ora.error_union<i256> -> i256
    ```
  }];
}

//===----------------------------------------------------------------------===//
// External Call Operations
//===----------------------------------------------------------------------===//

def Ora_AbiEncodeOp : Ora_Op<"abi_encode"> {
  let summary = "ABI-encode payload arguments";
  let description = [{
    Encodes arguments into ABI payload format without a function selector.
  }];

  let arguments = (ins AnyAttr:$layout, Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Ora_AbiEncodeWithSelectorOp : Ora_Op<"abi_encode_with_selector"> {
  let summary = "ABI-encode selector-prefixed external call arguments";
  let description = [{
    Encodes arguments into external-call calldata format: selector ++ payload.
  }];

  let arguments = (ins AnyAttr:$selector, AnyAttr:$arg_types, AnyAttr:$layout, Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Ora_ExternalCallOp : Ora_Op<"external_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "External CALL/STATICCALL security boundary";
  let description = [{
    Represents an external call with explicit call kind, target, gas, and calldata.
  }];

  let arguments = (ins StrAttr:$call_kind, StrAttr:$trait_name, StrAttr:$method_name,
                       AnyType:$target, AnyType:$gas, AnyType:$calldata);
  let results = (outs I1:$success, AnyType:$returndata);

  let assemblyFormat = "$target `,` $gas `,` $calldata attr-dict `:` type($target) `,` type($gas) `,` type($calldata) `->` type($success) `,` type($returndata)";
  let hasVerifier = 1;
}

def Ora_AbiDecodeOp : Ora_Op<"abi_decode"> {
  let summary = "ABI-decode external call returndata";
  let description = [{
    Decodes returndata into a typed value.
  }];

  let arguments = (ins AnyAttr:$return_types, AnyType:$returndata);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$returndata attr-dict `:` type($returndata) `->` type($result)";
  let hasVerifier = 1;
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

def Ora_ReturnOp : Ora_VariadicOperandTerminatorOp<"return",
    "Return operation", 1> {
  let summary = "Return operation";
  let description = [{
    Returns from a function with optional return values.
    
    Example:
    ```mlir
    ora.return %value : i256
    ```
  }];
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
}

#endif // ORA_OPS
