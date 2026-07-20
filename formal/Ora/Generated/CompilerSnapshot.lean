/-
GENERATED — DATA ONLY.  Do NOT edit by hand and do NOT add any `theorem`,
`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
`import` to this file. It contains only `def … := <literal>` facts emitted from
the compiler. The TRUSTED checks live in `Ora/Sync.lean`.

Regenerate with `scripts/check-formal-sync.sh`. Source:
src/formal/emit_compiler_snapshot.zig, src/types/builtin.zig,
src/types/semantic.zig, src/types/region_assign.zig, src/refinements/root.zig,
src/formal/obligation.zig.
-/

namespace Ora.Generated

def compilerBuiltinTypeIds : List String :=
  ["u8", "u16", "u32", "u64", "u128", "u160", "u256", "i8", "i16", "i32", "i64", "i128", "i256", "bool", "address", "string", "bytes", "void"]

def compilerBuiltinTypeComptimeIds : List (String × Nat) :=
  [("u8", 1), ("u16", 2), ("u32", 3), ("u64", 4), ("u128", 5), ("u160", 18), ("u256", 6), ("i8", 7), ("i16", 8), ("i32", 9), ("i64", 10), ("i128", 11), ("i256", 12), ("bool", 13), ("address", 14), ("string", 15), ("bytes", 16), ("void", 17)]

def compilerIntegerTypeRegistrations : List (String × Bool × Nat × Nat) :=
  [("u8", false, 8, 1), ("u16", false, 16, 2), ("u32", false, 32, 3), ("u64", false, 64, 4), ("u128", false, 128, 5), ("u160", false, 160, 18), ("u256", false, 256, 6), ("i8", true, 8, 7), ("i16", true, 16, 8), ("i32", true, 32, 9), ("i64", true, 64, 10), ("i128", true, 128, 11), ("i256", true, 256, 12)]

def compilerFixedBytesMin : Nat := 1
def compilerFixedBytesMax : Nat := 32

def compilerTypeKinds : List String :=
  ["unknown", "never", "void", "bool", "integer", "comptime_integer", "string", "address", "bytes", "fixed_bytes", "storage_slot", "storage_range", "external_proxy", "resource_domain", "resource_place", "named", "function", "contract", "struct_", "bitfield", "enum_", "tuple", "anonymous_struct", "array", "slice", "map", "error_union", "refinement"]

def compilerRegions : List String :=
  ["none", "storage", "memory", "transient", "calldata"]

def compilerResourceOperations : List String :=
  ["move", "create", "destroy"]

def compilerResourceProperties : List String :=
  ["amount_non_negative", "source_sufficient", "destination_no_overflow", "same_place_identity", "conservation"]

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
