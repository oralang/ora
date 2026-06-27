/-
Ora type system — projections of `Ty` into the compiler's surface vocabulary.

  * `Ty.compilerKind` — the compiler `TypeKind` tag of a type's head
    (`@tagName(Type.kind())` on the compiler side).
  * `PrimTy.spelling` / `Ty.spelling?` — the compiler source spelling of a
    primitive type (the `BuiltinTypeId` names + `bytesN`); `none` for composites.

These feed declaration / member sync checks: a curated declaration's member type
projects to the same kind / spelling the compiler emits for it.
-/

import Ora.Types.Ty

namespace Ora.Types

/-- The compiler `TypeKind` tag for a type's head. -/
def Ty.compilerKind : Ty → String
  | .prim (.int _)        => "integer"
  | .prim .bool           => "bool"
  | .prim .address        => "address"
  | .prim .string         => "string"
  | .prim .bytes          => "bytes"
  | .prim .void           => "void"
  | .prim (.fixedBytes _) => "fixed_bytes"
  | .tuple _              => "tuple"
  | .anonStruct _         => "anonymous_struct"
  | .array _ _            => "array"
  | .slice _              => "slice"
  | .map _ _              => "map"
  | .errorUnion _ _       => "error_union"
  | .refinement _ _ _     => "refinement"
  | .struct_ _            => "struct_"
  | .enum_ _              => "enum_"
  | .bitfield _           => "bitfield"
  | .contract _           => "contract"
  | .function _ _ _       => "function"
  | .resourceDomain _ _   => "resource_domain"
  | .resourcePlace _      => "resource_place"
  | .externalProxy _      => "external_proxy"
  | .storageSlot          => "storage_slot"
  | .storageRange         => "storage_range"

/-- The compiler source spelling of a primitive (`BuiltinTypeId` / `bytesN`). -/
def PrimTy.spelling : PrimTy → String
  | .int (.uint .w8)   => "u8"   | .int (.uint .w16)  => "u16"
  | .int (.uint .w32)  => "u32"  | .int (.uint .w64)  => "u64"
  | .int (.uint .w128) => "u128" | .int (.uint .w160) => "u160"
  | .int (.uint .w256) => "u256"
  | .int (.sint .w8)   => "i8"   | .int (.sint .w16)  => "i16"
  | .int (.sint .w32)  => "i32"  | .int (.sint .w64)  => "i64"
  | .int (.sint .w128) => "i128" | .int (.sint .w256) => "i256"
  | .bool    => "bool"   | .address => "address"
  | .string  => "string" | .bytes   => "bytes"   | .void => "void"
  | .fixedBytes m => "bytes" ++ toString m.n

/-- The spelling of a primitive type; `none` for composites. -/
def Ty.spelling? : Ty → Option String
  | .prim p => some p.spelling
  | _       => none

/-! ## Sanity -/

theorem u256_kind : Ty.compilerKind (.prim u256) = "integer" := rfl
theorem map_kind : Ty.compilerKind (.map (.prim u256) (.prim u256)) = "map" := rfl
theorem resourceDomain_kind :
    Ty.compilerKind (.resourceDomain "R" (.prim u256)) = "resource_domain" := rfl

theorem u256_spelling : Ty.spelling? (.prim u256) = some "u256" := rfl
theorem i8_spelling : Ty.spelling? (.prim i8) = some "i8" := rfl
theorem map_no_spelling : Ty.spelling? (.map (.prim u256) (.prim u256)) = none := rfl

end Ora.Types
