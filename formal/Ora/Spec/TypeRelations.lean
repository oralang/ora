/-
Spec-side type relation cases for `check-formal-type-relations-sync`.

The compiler snapshot is data-only: label, left spelling, right spelling, result.
This file owns the typed Lean cases and projects them to the same wire shape by
running `Ty.beq`, `Ty.assignable`, `Located.beq`, and `Located.assignable`.

Scope: this is an intentionally small curated corpus over the relation surface
that is currently modeled. It is not an exhaustive serialization of all Ora
types, and it deliberately avoids deferred refinement-subtyping/error-union
subtyping rows until those relations are modeled in Lean.
-/

import Ora.Types.TypeEq
import Ora.Types.Assignable

namespace Ora.Spec.TypeRelations

open Ora.Types

abbrev RelationRow := String × String × String × Bool

def u8Ty : Ty := .prim u8
def u16Ty : Ty := .prim u16
def u256Ty : Ty := .prim u256
def i8Ty : Ty := .prim i8
def boolTy : Ty := .prim .bool
def addressTy : Ty := .prim .address

def bytes4Ty : Ty := .prim (.fixedBytes ⟨4, by decide, by decide⟩)
def bytes32Ty : Ty := .prim (.fixedBytes ⟨32, by decide, by decide⟩)

def pointTy : Ty := .struct_ "Point"
def otherPointTy : Ty := .struct_ "OtherPoint"

def tokenResourceU8 : Ty := .resourceDomain "TokenUnit" u8Ty
def tokenResourceU16 : Ty := .resourceDomain "TokenUnit" u16Ty
def tokenPlaceU8 : Ty := .resourcePlace tokenResourceU8
def tokenPlaceU16 : Ty := .resourcePlace tokenResourceU16

def tupleU8Bool : Ty := .tuple [u8Ty, boolTy]
def tupleU16Bool : Ty := .tuple [u16Ty, boolTy]

def arrayU8_3 : Ty := .array u8Ty (some 3)
def arrayU16_3 : Ty := .array u16Ty (some 3)
def sliceU8 : Ty := .slice u8Ty
def sliceU16 : Ty := .slice u16Ty
def mapAddressU8 : Ty := .map addressTy u8Ty
def mapAddressU16 : Ty := .map addressTy u16Ty

def minOne : Ty := .refinement "MinValue" u256Ty [.integer "1"]
def minHundred : Ty := .refinement "MinValue" u256Ty [.integer "100"]

def fnFoo : Ty := .function (some "foo") [u8Ty] [boolTy]
def fnBar : Ty := .function (some "bar") [u8Ty] [boolTy]

structure TypeRelationCase where
  label : String
  lhsName : String
  rhsName : String
  lhs : Ty
  rhs : Ty

structure LocatedRelationCase where
  label : String
  lhsName : String
  rhsName : String
  lhs : Located
  rhs : Located

def typeEqlCases : List TypeRelationCase :=
  [ { label := "primitive_u256_self", lhsName := "u256", rhsName := "u256", lhs := u256Ty, rhs := u256Ty },
    { label := "primitive_u8_vs_u256", lhsName := "u8", rhsName := "u256", lhs := u8Ty, rhs := u256Ty },
    { label := "fixed_bytes_same", lhsName := "bytes4", rhsName := "bytes4", lhs := bytes4Ty, rhs := bytes4Ty },
    { label := "fixed_bytes_different", lhsName := "bytes4", rhsName := "bytes32", lhs := bytes4Ty, rhs := bytes32Ty },
    { label := "tuple_same", lhsName := "(u8,bool)", rhsName := "(u8,bool)", lhs := tupleU8Bool, rhs := tupleU8Bool },
    { label := "tuple_element_width_diff", lhsName := "(u8,bool)", rhsName := "(u16,bool)", lhs := tupleU8Bool, rhs := tupleU16Bool },
    { label := "array_same", lhsName := "[3]u8", rhsName := "[3]u8", lhs := arrayU8_3, rhs := arrayU8_3 },
    { label := "slice_element_diff", lhsName := "slice[u8]", rhsName := "slice[u16]", lhs := sliceU8, rhs := sliceU16 },
    { label := "map_same", lhsName := "map[address,u8]", rhsName := "map[address,u8]", lhs := mapAddressU8, rhs := mapAddressU8 },
    { label := "nominal_struct_same", lhsName := "Point", rhsName := "Point", lhs := pointTy, rhs := pointTy },
    { label := "nominal_struct_different", lhsName := "Point", rhsName := "OtherPoint", lhs := pointTy, rhs := otherPointTy },
    { label := "refinement_args_differ", lhsName := "MinValue<u256,1>", rhsName := "MinValue<u256,100>", lhs := minOne, rhs := minHundred },
    { label := "function_same_name", lhsName := "foo(u8)->bool", rhsName := "foo(u8)->bool", lhs := fnFoo, rhs := fnFoo },
    { label := "function_different_name", lhsName := "foo(u8)->bool", rhsName := "bar(u8)->bool", lhs := fnFoo, rhs := fnBar },
    { label := "resource_domain_same", lhsName := "resource TokenUnit<u8>", rhsName := "resource TokenUnit<u8>", lhs := tokenResourceU8, rhs := tokenResourceU8 },
    { label := "resource_domain_carrier_diff", lhsName := "resource TokenUnit<u8>", rhsName := "resource TokenUnit<u16>", lhs := tokenResourceU8, rhs := tokenResourceU16 } ]

def assignableCases : List TypeRelationCase :=
  [ { label := "u8_to_u256", lhsName := "u256", rhsName := "u8", lhs := u256Ty, rhs := u8Ty },
    { label := "u256_to_u8_rejected", lhsName := "u8", rhsName := "u256", lhs := u8Ty, rhs := u256Ty },
    { label := "signedness_mismatch", lhsName := "u8", rhsName := "i8", lhs := u8Ty, rhs := i8Ty },
    { label := "tuple_widening", lhsName := "(u16,bool)", rhsName := "(u8,bool)", lhs := tupleU16Bool, rhs := tupleU8Bool },
    { label := "tuple_narrowing_rejected", lhsName := "(u8,bool)", rhsName := "(u16,bool)", lhs := tupleU8Bool, rhs := tupleU16Bool },
    { label := "array_widening", lhsName := "[3]u16", rhsName := "[3]u8", lhs := arrayU16_3, rhs := arrayU8_3 },
    { label := "slice_widening", lhsName := "slice[u16]", rhsName := "slice[u8]", lhs := sliceU16, rhs := sliceU8 },
    { label := "map_value_widening", lhsName := "map[address,u16]", rhsName := "map[address,u8]", lhs := mapAddressU16, rhs := mapAddressU8 },
    { label := "nominal_struct_same", lhsName := "Point", rhsName := "Point", lhs := pointTy, rhs := pointTy },
    { label := "nominal_struct_different", lhsName := "Point", rhsName := "OtherPoint", lhs := pointTy, rhs := otherPointTy },
    { label := "function_same_name_identity", lhsName := "foo(u8)->bool", rhsName := "foo(u8)->bool", lhs := fnFoo, rhs := fnFoo },
    { label := "function_different_name_rejected", lhsName := "foo(u8)->bool", rhsName := "bar(u8)->bool", lhs := fnFoo, rhs := fnBar },
    { label := "resource_domain_same_identity", lhsName := "resource TokenUnit<u8>", rhsName := "resource TokenUnit<u8>", lhs := tokenResourceU8, rhs := tokenResourceU8 },
    { label := "resource_domain_widening_rejected", lhsName := "resource TokenUnit<u16>", rhsName := "resource TokenUnit<u8>", lhs := tokenResourceU16, rhs := tokenResourceU8 },
    { label := "resource_place_same_identity", lhsName := "Resource<TokenUnit<u8>>", rhsName := "Resource<TokenUnit<u8>>", lhs := tokenPlaceU8, rhs := tokenPlaceU8 },
    { label := "resource_place_widening_rejected", lhsName := "Resource<TokenUnit<u16>>", rhsName := "Resource<TokenUnit<u8>>", lhs := tokenPlaceU16, rhs := tokenPlaceU8 } ]

def storageU256Local : Located := { ty := u256Ty, region := .storage, provenance := .local }
def storageU256StorageProvenance : Located := { ty := u256Ty, region := .storage, provenance := .storage }
def memoryU256 : Located := { ty := u256Ty, region := .memory, provenance := .local }
def storageU8 : Located := { ty := u8Ty, region := .storage, provenance := .local }
def calldataU8 : Located := { ty := u8Ty, region := .calldata, provenance := .local }
def storageBool : Located := { ty := boolTy, region := .storage, provenance := .local }
def memoryBool : Located := { ty := boolTy, region := .memory, provenance := .local }
def memoryAddress : Located := { ty := addressTy, region := .memory, provenance := .local }
def calldataU256 : Located := { ty := u256Ty, region := .calldata, provenance := .local }

def locatedEqlCases : List LocatedRelationCase :=
  [ { label := "same_type_region_provenance", lhsName := "u256@storage/local", rhsName := "u256@storage/local", lhs := storageU256Local, rhs := storageU256Local },
    { label := "different_region", lhsName := "u256@storage/local", rhsName := "u256@memory/local", lhs := storageU256Local, rhs := memoryU256 },
    { label := "different_type", lhsName := "u8@storage/local", rhsName := "u256@storage/local", lhs := storageU8, rhs := storageU256Local },
    { label := "different_provenance", lhsName := "u256@storage/local", rhsName := "u256@storage/storage", lhs := storageU256Local, rhs := storageU256StorageProvenance } ]

def locatedAssignableCases : List LocatedRelationCase :=
  [ { label := "calldata_u8_to_memory_u256", lhsName := "u8@calldata/local", rhsName := "u256@memory/local", lhs := calldataU8, rhs := memoryU256 },
    { label := "memory_u256_to_calldata_u256_rejected", lhsName := "u256@memory/local", rhsName := "u256@calldata/local", lhs := memoryU256, rhs := calldataU256 },
    { label := "storage_bool_to_memory_bool", lhsName := "bool@storage/local", rhsName := "bool@memory/local", lhs := storageBool, rhs := memoryBool },
    { label := "storage_bool_to_memory_address_rejected", lhsName := "bool@storage/local", rhsName := "address@memory/local", lhs := storageBool, rhs := memoryAddress },
    { label := "storage_u8_to_storage_u256", lhsName := "u8@storage/local", rhsName := "u256@storage/local", lhs := storageU8, rhs := storageU256Local },
    { label := "provenance_ignored_by_assignability", lhsName := "u256@storage/storage", rhsName := "u256@storage/local", lhs := storageU256StorageProvenance, rhs := storageU256Local } ]

def typeRow (c : TypeRelationCase) (result : Ty → Ty → Bool) : RelationRow :=
  (c.label, c.lhsName, c.rhsName, result c.lhs c.rhs)

def locatedRow (c : LocatedRelationCase) (result : Located → Located → Bool) : RelationRow :=
  (c.label, c.lhsName, c.rhsName, result c.lhs c.rhs)

def expectedTypeEqlRows : List RelationRow :=
  typeEqlCases.map fun c => typeRow c Ty.beq

def expectedTypesAssignableRows : List RelationRow :=
  assignableCases.map fun c => typeRow c Ty.assignable

def expectedLocatedTypeEqlRows : List RelationRow :=
  locatedEqlCases.map fun c => locatedRow c Located.beq

def expectedLocatedAssignableRows : List RelationRow :=
  locatedAssignableCases.map fun c => locatedRow c Located.assignable

end Ora.Spec.TypeRelations
