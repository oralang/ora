/-
Spec-side FACT INTERFACE for `check-formal-sync`.

This file projects the trusted Lean spec (`Ora.Types.*`) into the SAME data
shapes the compiler emits in `Ora/Generated/CompilerSnapshot.lean`, so that
`Ora/Sync.lean` can prove — by `decide`, in the kernel — that the compiler's
facts conform to the spec.

Nothing here is "generated": it is hand-written, trusted Lean. The generated
snapshot is DATA ONLY; all proving happens against these expected values.
-/

import Ora.Types.Prim
import Ora.Types.Region

namespace Ora.Spec

open Ora.Types

/-! ## Integer widths (bit counts), projected from the width enums -/

def uintBits : UIntWidth → Nat
  | .w8 => 8 | .w16 => 16 | .w32 => 32 | .w64 => 64 | .w128 => 128 | .w160 => 160 | .w256 => 256

def sintBits : SIntWidth → Nat
  | .w8 => 8 | .w16 => 16 | .w32 => 32 | .w64 => 64 | .w128 => 128 | .w256 => 256

def allUIntWidths : List UIntWidth := [.w8, .w16, .w32, .w64, .w128, .w160, .w256]
def allSIntWidths : List SIntWidth := [.w8, .w16, .w32, .w64, .w128, .w256]

/-- Unsigned widths the spec admits, as bit counts: `[8,16,32,64,128,160,256]`. -/
def expectedUIntWidths : List Nat := allUIntWidths.map uintBits

/-- Signed widths the spec admits, as bit counts: `[8,16,32,64,128,256]` (no 160). -/
def expectedSIntWidths : List Nat := allSIntWidths.map sintBits

/-! ## Spellable scalar builtins (`BuiltinTypeId`), in the compiler's enum order -/

def expectedBuiltinTypeIds : List String :=
  ["u8", "u16", "u32", "u64", "u128", "u160", "u256",
   "i8", "i16", "i32", "i64", "i128", "i256",
   "bool", "address", "string", "bytes", "void"]

/-! ## Fixed-bytes bounds -/

def expectedFixedBytesMin : Nat := 1
def expectedFixedBytesMax : Nat := 32

/-! ## The 28 `TypeKind`s, in the compiler's enum order (modeled + excluded together) -/

def expectedTypeKinds : List String :=
  ["unknown", "never", "void", "bool", "integer", "comptime_integer", "string", "address",
   "bytes", "fixed_bytes", "storage_slot", "storage_range", "external_proxy",
   "resource_domain", "resource_place", "named", "function", "contract", "struct_",
   "bitfield", "enum_", "tuple", "anonymous_struct", "array", "slice", "map",
   "error_union", "refinement"]

/-- The 4 compiler-internal `TypeKind`s the SURFACE model excludes (see `Ty.lean`). -/
def excludedTypeKinds : List String := ["unknown", "never", "named", "comptime_integer"]

/-! ## Regions + the assignability table, derived from the canonical relation

    `stack` is the compiler's `.none`; everything is emitted in the compiler's
    spelling and enum order so the snapshot can be compared directly. The table
    is DERIVED from `Region.assignableTo` (the canonical relation), so a drift in
    that relation changes `expectedRegionTable` automatically. -/

def regionCompilerName : Region → String
  | .stack => "none" | .storage => "storage" | .memory => "memory"
  | .transient => "transient" | .calldata => "calldata"

def regionsInCompilerOrder : List Region := [.stack, .storage, .memory, .transient, .calldata]

def expectedRegions : List String := regionsInCompilerOrder.map regionCompilerName

def expectedRegionTable : List (String × String × Bool) :=
  (regionsInCompilerOrder.map fun a =>
    regionsInCompilerOrder.map fun b =>
      (regionCompilerName a, regionCompilerName b, a.assignableTo b)).flatten

end Ora.Spec
