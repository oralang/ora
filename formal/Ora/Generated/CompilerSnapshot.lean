/-
GENERATED — DATA ONLY.  Do NOT edit by hand and do NOT add any `theorem`,
`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
`import` to this file. It contains only `def … := <literal>` facts emitted from
the compiler. The TRUSTED checks live in `Ora/Sync.lean`.

Regenerate with `scripts/check-formal-sync.sh`. Source:
src/formal/emit_compiler_snapshot.zig, src/types/builtin.zig,
src/types/semantic.zig, src/types/region_assign.zig, src/refinements/root.zig.
-/

namespace Ora.Generated

def compilerBuiltinTypeIds : List String :=
  ["u8", "u16", "u32", "u64", "u128", "u160", "u256", "i8", "i16", "i32", "i64", "i128", "i256", "bool", "address", "string", "bytes", "void"]

def compilerUIntWidths : List Nat := [8, 16, 32, 64, 128, 160, 256]
def compilerSIntWidths : List Nat := [8, 16, 32, 64, 128, 256]

def compilerFixedBytesMin : Nat := 1
def compilerFixedBytesMax : Nat := 32

def compilerTypeKinds : List String :=
  ["unknown", "never", "void", "bool", "integer", "comptime_integer", "string", "address", "bytes", "fixed_bytes", "storage_slot", "storage_range", "external_proxy", "resource_domain", "resource_place", "named", "function", "contract", "struct_", "bitfield", "enum_", "tuple", "anonymous_struct", "array", "slice", "map", "error_union", "refinement"]

def compilerRegions : List String :=
  ["none", "storage", "memory", "transient", "calldata"]

def compilerRefinementNames : List String :=
  ["MinValue", "MaxValue", "InRange", "NonZeroAddress", "NonZero", "BasisPoints", "Exact", "Scaled"]

def compilerRuntimeGuardRefinementNames : List String :=
  ["MinValue", "MaxValue", "InRange", "NonZeroAddress", "NonZero", "BasisPoints"]

def compilerCompileTimeOnlyRefinementNames : List String :=
  ["Exact", "Scaled"]

def compilerNativeMlirRefinementNames : List String :=
  ["MinValue", "MaxValue", "InRange", "NonZeroAddress", "Exact", "Scaled"]

def compilerPathFormRefinementNames : List String :=
  ["NonZeroAddress"]

def compilerBoundsBackedRefinementNames : List String :=
  ["MinValue", "MaxValue", "InRange", "NonZero", "BasisPoints"]

def compilerRegionTable : List (String × String × Bool) :=
  [("none", "none", true),
   ("none", "storage", true),
   ("none", "memory", true),
   ("none", "transient", true),
   ("none", "calldata", true),
   ("storage", "none", true),
   ("storage", "storage", true),
   ("storage", "memory", true),
   ("storage", "transient", false),
   ("storage", "calldata", false),
   ("memory", "none", true),
   ("memory", "storage", true),
   ("memory", "memory", true),
   ("memory", "transient", true),
   ("memory", "calldata", false),
   ("transient", "none", true),
   ("transient", "storage", false),
   ("transient", "memory", true),
   ("transient", "transient", true),
   ("transient", "calldata", false),
   ("calldata", "none", true),
   ("calldata", "storage", false),
   ("calldata", "memory", true),
   ("calldata", "transient", false),
   ("calldata", "calldata", true)]

end Ora.Generated
