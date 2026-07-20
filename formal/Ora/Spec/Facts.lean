/-
Spec-side FACT INTERFACE for `check-formal-sync`.

This file projects the trusted Lean spec (`Ora.Types.*`) into typed facts that
the generated compiler snapshot must match.

Important:
We avoid proving only arrays of strings. Strings are allowed at the boundary
for compiler spellings, but the sync layer should compare typed rows whenever
possible.

Trusted:
  Ora.Types.*
  Ora.Spec.Facts

Generated:
  Ora.Generated.CompilerSnapshot

Goal:
  prove generated data-only facts equal projections from expected typed facts.
-/

import Ora.Types.Prim
import Ora.Types.Region
import Ora.Types.Refinement

namespace Ora.Spec

open Ora.Types

/-! ## Integer registrations -/

def uintBits : UIntWidth → Nat
  | .w8 => 8
  | .w16 => 16
  | .w32 => 32
  | .w64 => 64
  | .w128 => 128
  | .w160 => 160
  | .w256 => 256

def sintBits : SIntWidth → Nat
  | .w8 => 8
  | .w16 => 16
  | .w32 => 32
  | .w64 => 64
  | .w128 => 128
  | .w256 => 256

inductive IntegerShape where
  | unsigned : UIntWidth → IntegerShape
  | signed : SIntWidth → IntegerShape
  deriving Repr, DecidableEq

def IntegerShape.bits : IntegerShape → Nat
  | .unsigned width => uintBits width
  | .signed width => sintBits width

def IntegerShape.isSigned : IntegerShape → Bool
  | .unsigned _ => false
  | .signed _ => true

def IntegerShape.primTy : IntegerShape → PrimTy
  | .unsigned width => .int (.uint width)
  | .signed width => .int (.sint width)

/-! ## Builtin primitive spellings -/

/--
A typed compiler spelling for a primitive type.

This is still partly boundary-facing because compiler names are strings,
but the semantic identity is `PrimTy`, not just the string.
-/
structure BuiltinTypeFact where
  name : String
  ty   : PrimTy
  compilerTypeId : Nat
  deriving Repr

def expectedCompilerTypeIdU256 : Nat := 6
def expectedCompilerTypeIdI256 : Nat := 12
def expectedCompilerTypeIdBool : Nat := 13

structure IntegerRegistrationFact where
  name : String
  shape : IntegerShape
  compilerTypeId : Nat
  deriving Repr, DecidableEq

def expectedIntegerRegistrationFacts : List IntegerRegistrationFact :=
  [ { name := "u8", shape := .unsigned .w8, compilerTypeId := 1 },
    { name := "u16", shape := .unsigned .w16, compilerTypeId := 2 },
    { name := "u32", shape := .unsigned .w32, compilerTypeId := 3 },
    { name := "u64", shape := .unsigned .w64, compilerTypeId := 4 },
    { name := "u128", shape := .unsigned .w128, compilerTypeId := 5 },
    { name := "u160", shape := .unsigned .w160, compilerTypeId := 18 },
    { name := "u256", shape := .unsigned .w256,
      compilerTypeId := expectedCompilerTypeIdU256 },
    { name := "i8", shape := .signed .w8, compilerTypeId := 7 },
    { name := "i16", shape := .signed .w16, compilerTypeId := 8 },
    { name := "i32", shape := .signed .w32, compilerTypeId := 9 },
    { name := "i64", shape := .signed .w64, compilerTypeId := 10 },
    { name := "i128", shape := .signed .w128, compilerTypeId := 11 },
    { name := "i256", shape := .signed .w256,
      compilerTypeId := expectedCompilerTypeIdI256 } ]

def IntegerRegistrationFact.toBuiltinTypeFact
    (fact : IntegerRegistrationFact) : BuiltinTypeFact :=
  { name := fact.name,
    ty := fact.shape.primTy,
    compilerTypeId := fact.compilerTypeId }

def expectedBuiltinTypeFacts : List BuiltinTypeFact :=
  expectedIntegerRegistrationFacts.map IntegerRegistrationFact.toBuiltinTypeFact ++
  [
    { name := "bool",    ty := .bool,    compilerTypeId := expectedCompilerTypeIdBool },
    { name := "address", ty := .address, compilerTypeId := 14 },
    { name := "string",  ty := .string,  compilerTypeId := 15 },
    { name := "bytes",   ty := .bytes,   compilerTypeId := 16 },
    { name := "void",    ty := .void,    compilerTypeId := 17 } ]

/-! ## Fixed bytes -/

structure FixedBytesBoundsFact where
  min : Nat
  max : Nat
  deriving Repr, DecidableEq

def expectedFixedBytesBounds : FixedBytesBoundsFact :=
  { min := 1, max := 32 }

/-! ## TypeKind classification -/

/--
The compiler has more `TypeKind`s than the surface/core model admits.

We classify each compiler TypeKind rather than only comparing its string name.
-/
inductive TypeKindStatus where
  | modeled
  | excluded
  deriving Repr, DecidableEq

def TypeKindStatus.isExcluded : TypeKindStatus → Bool
  | .modeled => false
  | .excluded => true

structure TypeKindFact where
  compilerName : String
  status       : TypeKindStatus
  deriving Repr, DecidableEq

def expectedTypeKindFacts : List TypeKindFact :=
  [ { compilerName := "unknown",          status := .excluded },
    { compilerName := "never",            status := .excluded },
    { compilerName := "void",             status := .modeled },
    { compilerName := "bool",             status := .modeled },
    { compilerName := "integer",          status := .modeled },
    { compilerName := "comptime_integer", status := .excluded },
    { compilerName := "string",           status := .modeled },
    { compilerName := "address",          status := .modeled },
    { compilerName := "bytes",            status := .modeled },
    { compilerName := "fixed_bytes",      status := .modeled },
    { compilerName := "storage_slot",     status := .modeled },
    { compilerName := "storage_range",    status := .modeled },
    { compilerName := "external_proxy",   status := .modeled },
    { compilerName := "resource_domain",  status := .modeled },
    { compilerName := "resource_place",   status := .modeled },
    { compilerName := "named",            status := .excluded },
    { compilerName := "function",         status := .modeled },
    { compilerName := "contract",         status := .modeled },
    { compilerName := "struct_",          status := .modeled },
    { compilerName := "bitfield",         status := .modeled },
    { compilerName := "enum_",            status := .modeled },
    { compilerName := "tuple",            status := .modeled },
    { compilerName := "anonymous_struct", status := .modeled },
    { compilerName := "array",            status := .modeled },
    { compilerName := "slice",            status := .modeled },
    { compilerName := "map",              status := .modeled },
    { compilerName := "error_union",      status := .modeled },
    { compilerName := "refinement",       status := .modeled } ]

/-! ## Refinement registry -/

structure RefinementRegistryFact where
  name              : RefinementName
  compilerName      : String
  hasRuntimeGuard   : Bool
  compileTimeOnly   : Bool
  hasNativeMlirType : Bool
  pathForm          : Bool
  boundsBacked      : Bool
  deriving Repr, DecidableEq

def refinementRegistryFact (r : RefinementName) : RefinementRegistryFact :=
  let info := r.info
  { name              := r,
    compilerName      := r.compilerName,
    hasRuntimeGuard   := info.hasRuntimeGuard,
    compileTimeOnly   := info.compileTimeOnly,
    hasNativeMlirType := info.hasNativeMlirType,
    pathForm          := info.pathForm,
    boundsBacked      := info.boundsBacked }

def expectedRefinementRegistryFacts : List RefinementRegistryFact :=
  allRefinementNames.map refinementRegistryFact

/-! ## Regions -/

/--
Canonical compiler spelling for a region.

`stack` corresponds to compiler `.none`.
-/
def Region.compilerName : Region → String
  | .stack     => "none"
  | .storage   => "storage"
  | .memory    => "memory"
  | .transient => "transient"
  | .calldata  => "calldata"

/--
Compiler enum order.

Keep this explicit because sync checks also protect ordering assumptions.
-/
def regionsInCompilerOrder : List Region :=
  [.stack, .storage, .memory, .transient, .calldata]

structure RegionFact where
  region       : Region
  compilerName : String
  deriving Repr, DecidableEq

structure RegionAssignabilityFact where
  src        : Region
  dst        : Region
  assignable : Bool
  deriving Repr, DecidableEq

def expectedRegionFacts : List RegionFact :=
  regionsInCompilerOrder.map fun r =>
    { region := r, compilerName := Region.compilerName r }

def expectedRegionAssignabilityFacts : List RegionAssignabilityFact :=
  (regionsInCompilerOrder.map fun src =>
    regionsInCompilerOrder.map fun dst =>
      { src := src,
        dst := dst,
        assignable := src.assignableTo dst }).flatten

/-! ## Data-only sync projections

The generated compiler snapshot intentionally contains only primitive data
(strings, numbers, booleans, tuples). The definitions below are the trusted
typed facts above projected into that same wire shape for `Ora.Sync`.
-/

abbrev IntegerRegistrationRow := String × Bool × Nat × Nat

def IntegerRegistrationFact.toRow
    (fact : IntegerRegistrationFact) : IntegerRegistrationRow :=
  (fact.name, fact.shape.isSigned, fact.shape.bits, fact.compilerTypeId)

def expectedIntegerTypeRegistrations : List IntegerRegistrationRow :=
  expectedIntegerRegistrationFacts.map IntegerRegistrationFact.toRow

def unsignedWidthsFromIntegerRegistrations
    (rows : List IntegerRegistrationRow) : List Nat :=
  rows.filterMap fun
    | (_, false, bits, _) => some bits
    | _ => none

def signedWidthsFromIntegerRegistrations
    (rows : List IntegerRegistrationRow) : List Nat :=
  rows.filterMap fun
    | (_, true, bits, _) => some bits
    | _ => none

def expectedUIntWidths : List Nat :=
  unsignedWidthsFromIntegerRegistrations expectedIntegerTypeRegistrations

def expectedSIntWidths : List Nat :=
  signedWidthsFromIntegerRegistrations expectedIntegerTypeRegistrations

def expectedBuiltinTypeIds : List String :=
  expectedBuiltinTypeFacts.map fun f => f.name

def expectedBuiltinTypeComptimeIds : List (String × Nat) :=
  expectedBuiltinTypeFacts.map fun f => (f.name, f.compilerTypeId)

def expectedFixedBytesMin : Nat :=
  expectedFixedBytesBounds.min

def expectedFixedBytesMax : Nat :=
  expectedFixedBytesBounds.max

def expectedTypeKinds : List String :=
  expectedTypeKindFacts.map fun f => f.compilerName

def excludedTypeKinds : List String :=
  (expectedTypeKindFacts.filter fun f => f.status.isExcluded).map fun f => f.compilerName

def expectedRefinementNames : List String :=
  expectedRefinementRegistryFacts.map fun f => f.compilerName

def refinementNamesWhere (p : RefinementRegistryFact → Bool) : List String :=
  (expectedRefinementRegistryFacts.filter p).map fun f => f.compilerName

def expectedRuntimeGuardRefinementNames : List String :=
  refinementNamesWhere fun f => f.hasRuntimeGuard

def expectedCompileTimeOnlyRefinementNames : List String :=
  refinementNamesWhere fun f => f.compileTimeOnly

def expectedNativeMlirRefinementNames : List String :=
  refinementNamesWhere fun f => f.hasNativeMlirType

def expectedPathFormRefinementNames : List String :=
  refinementNamesWhere fun f => f.pathForm

def expectedBoundsBackedRefinementNames : List String :=
  refinementNamesWhere fun f => f.boundsBacked

def expectedRegions : List String :=
  expectedRegionFacts.map fun f => f.compilerName

def expectedRegionTable : List (String × String × Bool) :=
  expectedRegionAssignabilityFacts.map fun f =>
    (Region.compilerName f.src, Region.compilerName f.dst, f.assignable)

end Ora.Spec
