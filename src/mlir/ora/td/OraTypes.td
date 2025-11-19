//===- OraTypes.td - Ora dialect types ------------------*- tablegen -*-===//
//
// This file defines all types for the Ora MLIR dialect.
// Includes primitive types, complex types, and refinement types.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_TYPES
#define ORA_TYPES

include "OraDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class for Ora types
//===----------------------------------------------------------------------===//

class Ora_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ora_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Integer Type (moved from OraDialect.td)
//===----------------------------------------------------------------------===//

// Ora integer type - parameterized by width and signedness
// Uses declarative assemblyFormat as per MLIR docs
def Ora_IntegerType : Ora_Type<"Integer", "int"> {
  let summary = "Ora integer type with width and signedness";
  let description = [{
    Ora integer type with configurable bit width and signedness.
    
    Syntax:
    ```
    !ora.int<256, true>   // 256-bit signed integer
    !ora.int<256, false> // 256-bit unsigned integer
    !ora.int<8, true>     // 8-bit signed integer
    !ora.int<8, false>    // 8-bit unsigned integer
    ```
  }];
  
  let parameters = (ins "unsigned":$width, "bool":$isSigned);
  let assemblyFormat = "`<` $width `,` $isSigned `>`";
  let builders = [TypeBuilder<(ins "unsigned":$width, "bool":$isSigned)>];
  
  // No extra methods needed - our Ora types are MLIR types and work directly
  // with MLIR infrastructure
}

//===----------------------------------------------------------------------===//
// Primitive Types
//===----------------------------------------------------------------------===//

// Boolean type - 1 bit
def Ora_BoolType : Ora_Type<"Bool", "bool"> {
  let summary = "Ora boolean type";
  let description = [{
    Boolean type representing true or false values.
    Syntax: `!ora.bool`
  }];
}

// Address type - 160 bits (20 bytes) for Ethereum addresses
def Ora_AddressType : Ora_Type<"Address", "address"> {
  let summary = "Ethereum address type";
  let description = [{
    Address type representing Ethereum addresses (160 bits).
    Syntax: `!ora.address`
  }];
}

// String type - dynamic string
def Ora_StringType : Ora_Type<"String", "string"> {
  let summary = "Ora string type";
  let description = [{
    Dynamic string type for text data.
    Syntax: `!ora.string`
  }];
}

// Bytes type - dynamic byte array
def Ora_BytesType : Ora_Type<"Bytes", "bytes"> {
  let summary = "Ora bytes type";
  let description = [{
    Dynamic byte array type.
    Syntax: `!ora.bytes`
  }];
}

//===----------------------------------------------------------------------===//
// Complex Types
//===----------------------------------------------------------------------===//

// Fixed-size array type [T; N]
def Ora_ArrayType : Ora_Type<"Array", "array"> {
  let summary = "Fixed-size array type";
  let description = [{
    Fixed-size array type with element type and length.
    Syntax: `!ora.array<element_type, length>`
  }];
  
  let parameters = (ins "Type":$elementType, "uint64_t":$length);
  let assemblyFormat = "`<` $elementType `,` $length `>`";
}

// Dynamic slice type []T
def Ora_SliceType : Ora_Type<"Slice", "slice"> {
  let summary = "Dynamic slice type";
  let description = [{
    Dynamic slice type representing a variable-length array.
    Syntax: `!ora.slice<element_type>`
  }];
  
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

// Map type mapping K -> V
def Ora_MapType : Ora_Type<"Map", "map"> {
  let summary = "Storage mapping type";
  let description = [{
    Storage mapping type with key and value types.
    Uses keccak256 hashing for storage layout.
    Syntax: `!ora.map<key_type, value_type>`
  }];
  
  let parameters = (ins "Type":$keyType, "Type":$valueType);
  let assemblyFormat = "`<` $keyType `,` $valueType `>`";
}

// Double map type mapping (K1, K2) -> V
def Ora_DoubleMapType : Ora_Type<"DoubleMap", "doublemap"> {
  let summary = "Nested storage mapping type";
  let description = [{
    Nested storage mapping type with two keys and a value type.
    Syntax: `!ora.doublemap<key1_type, key2_type, value_type>`
  }];
  
  let parameters = (ins "Type":$key1Type, "Type":$key2Type, "Type":$valueType);
  let assemblyFormat = "`<` $key1Type `,` $key2Type `,` $valueType `>`";
}

// Tuple type (T1, T2, ..., Tn)
def Ora_TupleType : Ora_Type<"Tuple", "tuple"> {
  let summary = "Tuple type";
  let description = [{
    Tuple type containing multiple element types.
    Syntax: `!ora.tuple<type1, type2, ...>`
  }];
  
  let parameters = (ins "ArrayRef<Type>":$elementTypes);
  let hasCustomAssemblyFormat = 1;
}

// Struct type - named struct
def Ora_StructType : Ora_Type<"Struct", "struct"> {
  let summary = "Named struct type";
  let description = [{
    Named struct type with a struct name.
    Syntax: `!ora.struct<"struct_name">`
  }];
  
  let parameters = (ins "StringRef":$name);
  let hasCustomAssemblyFormat = 1;
}

// Enum type - named enum with representation type
def Ora_EnumType : Ora_Type<"Enum", "enum"> {
  let summary = "Named enum type";
  let description = [{
    Named enum type with a name and underlying integer representation.
    Syntax: `!ora.enum<"enum_name", repr_type>`
  }];
  
  let parameters = (ins "StringRef":$name, "Type":$reprType);
  let hasCustomAssemblyFormat = 1;
}

// Contract type - named contract
def Ora_ContractType : Ora_Type<"Contract", "contract"> {
  let summary = "Contract type";
  let description = [{
    Contract type representing a contract instance.
    Syntax: `!ora.contract<"contract_name">`
  }];
  
  let parameters = (ins "StringRef":$name);
  let hasCustomAssemblyFormat = 1;
}

// Function type - function signature
def Ora_FunctionType : Ora_Type<"Function", "function"> {
  let summary = "Function type";
  let description = [{
    Function type with parameter types and optional return type.
    Syntax: `!ora.function<param_types, return_type>`
  }];
  
  let parameters = (ins "ArrayRef<Type>":$paramTypes, "Type":$returnType);
  let hasCustomAssemblyFormat = 1;
}

// Union type - sum type T1 | T2 | ...
def Ora_UnionType : Ora_Type<"Union", "union"> {
  let summary = "Union type";
  let description = [{
    Union type representing a sum type.
    Syntax: `!ora.union<type1, type2, ...>`
  }];
  
  let parameters = (ins "ArrayRef<Type>":$elementTypes);
  let hasCustomAssemblyFormat = 1;
}

// Error union type - result type !T
def Ora_ErrorUnionType : Ora_Type<"ErrorUnion", "error_union"> {
  let summary = "Error union type";
  let description = [{
    Error union type representing a result that can be an error.
    Syntax: `!ora.error_union<success_type>`
  }];
  
  let parameters = (ins "Type":$successType);
  let assemblyFormat = "`<` $successType `>`";
}

//===----------------------------------------------------------------------===//
// Refinement Types
//===----------------------------------------------------------------------===//

// Min value refinement - integer with minimum value constraint
def Ora_MinValueType : Ora_Type<"MinValue", "min_value"> {
  let summary = "Integer type with minimum value constraint";
  let description = [{
    Refinement type constraining an integer to have a minimum value.
    Syntax: `!ora.min_value<base_type, min>`
  }];
  
  let parameters = (ins "Type":$baseType, "uint64_t":$min);
  let assemblyFormat = "`<` $baseType `,` $min `>`";
}

// Max value refinement - integer with maximum value constraint
def Ora_MaxValueType : Ora_Type<"MaxValue", "max_value"> {
  let summary = "Integer type with maximum value constraint";
  let description = [{
    Refinement type constraining an integer to have a maximum value.
    Syntax: `!ora.max_value<base_type, max>`
  }];
  
  let parameters = (ins "Type":$baseType, "uint64_t":$max);
  let assemblyFormat = "`<` $baseType `,` $max `>`";
}

// In range refinement - integer with range constraint
def Ora_InRangeType : Ora_Type<"InRange", "in_range"> {
  let summary = "Integer type with range constraint";
  let description = [{
    Refinement type constraining an integer to be within a range.
    Syntax: `!ora.in_range<base_type, min, max>`
  }];
  
  let parameters = (ins "Type":$baseType, "uint64_t":$min, "uint64_t":$max);
  let assemblyFormat = "`<` $baseType `,` $min `,` $max `>`";
}

// Scaled refinement - integer with decimal scaling
def Ora_ScaledType : Ora_Type<"Scaled", "scaled"> {
  let summary = "Integer type with decimal scaling";
  let description = [{
    Refinement type for integers with decimal scaling (e.g., for fixed-point arithmetic).
    Syntax: `!ora.scaled<base_type, decimals>`
  }];
  
  let parameters = (ins "Type":$baseType, "uint32_t":$decimals);
  let assemblyFormat = "`<` $baseType `,` $decimals `>`";
}

// Exact refinement - integer that must participate in exact division
def Ora_ExactType : Ora_Type<"Exact", "exact"> {
  let summary = "Integer type requiring exact division";
  let description = [{
    Refinement type for integers that must participate in exact division operations.
    Syntax: `!ora.exact<base_type>`
  }];
  
  let parameters = (ins "Type":$baseType);
  let assemblyFormat = "`<` $baseType `>`";
}

// Non-zero address - address that cannot be zero
def Ora_NonZeroAddressType : Ora_Type<"NonZeroAddress", "non_zero_address"> {
  let summary = "Non-zero address type";
  let description = [{
    Refinement type for addresses that cannot be the zero address.
    Syntax: `!ora.non_zero_address`
  }];
}

#endif // ORA_TYPES