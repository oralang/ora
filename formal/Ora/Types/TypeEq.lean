/-
Ora type system — structural type equality (`typeEql` / `locatedTypeEql`).

`DecidableEq Ty` can't be derived (`FixedBytesLen`'s proof fields, and the nested
`List Ty`). But a STRUCTURAL Bool equality needs neither — defined the same way as
`Ty.assignable`, by mutual recursion. This is the faithful mirror of the
compiler's `typeEql`, and it gives `Located` the `locatedTypeEql` it was missing.

Refinement equality compares name + base + ARGS (so `MinValue<_,1> ≠
MinValue<_,100>`).

Lawfulness: reflexivity (`Ty.beq t t = true`) IS proven, in
`Ora/Types/TypeEqLawful.lean` — kept out of the default build because the proof
unfolds the mutual `Ty.beq` and is intentionally expensive for now (~+40s); it
should be optimized before the formal gate grows much larger. STILL REMAINING:
the other direction `beq ↔ =`, i.e. a real `DecidableEq Ty`.
-/

import Ora.Types.Ty
import Ora.Types.Region

namespace Ora.Types

/-- Structural equality of primitives. `IntTy` has `DecidableEq`; `FixedBytesLen`
    is compared by its length. -/
def primBeq : PrimTy → PrimTy → Bool
  | .int e, .int a => e == a
  | .bool, .bool => true
  | .address, .address => true
  | .string, .string => true
  | .bytes, .bytes => true
  | .void, .void => true
  | .fixedBytes m, .fixedBytes n => m.n == n.n
  | _, _ => false

mutual
/-- Structural type equality. Mirrors the compiler's `typeEql`. -/
def Ty.beq : Ty → Ty → Bool
  | .prim p, .prim q => primBeq p q
  | .tuple es, .tuple as => beqList es as
  | .anonStruct fe, .anonStruct fa => beqFields fe fa
  | .array e (some ne), .array a (some na) => ne == na && Ty.beq e a
  | .array e none, .array a none => Ty.beq e a
  | .slice e, .slice a => Ty.beq e a
  | .map ke ve, .map ka va => Ty.beq ke ka && Ty.beq ve va
  | .errorUnion pe ee, .errorUnion pa ea => Ty.beq pe pa && beqList ee ea
  | .refinement n be ae, .refinement m ba aa => n == m && Ty.beq be ba && ae == aa
  | .struct_ n, .struct_ m => n == m
  | .enum_ n, .enum_ m => n == m
  | .bitfield n, .bitfield m => n == m
  | .contract n, .contract m => n == m
  | .externalProxy n, .externalProxy m => n == m
  | .storageSlot, .storageSlot => true
  | .storageRange, .storageRange => true
  | .function ne pse rse, .function na psa rsa => ne == na && beqList pse psa && beqList rse rsa
  | .resourceDomain n c, .resourceDomain m d => n == m && Ty.beq c d
  | .resourcePlace e, .resourcePlace a => Ty.beq e a
  | _, _ => false
def beqList : List Ty → List Ty → Bool
  | [], [] => true
  | e :: es, a :: as => Ty.beq e a && beqList es as
  | _, _ => false
def beqFields : List (Name × Ty) → List (Name × Ty) → Bool
  | [], [] => true
  | fe :: es, fa :: as => fe.1 == fa.1 && Ty.beq fe.2 fa.2 && beqFields es as
  | _, _ => false
end

instance : BEq Ty := ⟨Ty.beq⟩

/-- Structural equality of located types — `typeEql` ∧ same region ∧ same
    provenance. Mirrors the compiler's `locatedTypeEql` (`region.zig:18`). -/
def Located.beq (a b : Located) : Bool :=
  a.ty.beq b.ty && a.region == b.region && a.provenance == b.provenance

/-! ## Theorems -/

/-- Concrete reflexivity examples. The general `beq t t = true` is proved in
    `Ora/Types/TypeEqLawful.lean`, kept out of the default build (see that file). -/
theorem u256_beq_self : Ty.beq (.prim u256) (.prim u256) = true := rfl
theorem struct_beq_self : Ty.beq (.struct_ "P") (.struct_ "P") = true := rfl
theorem map_beq_self :
    Ty.beq (.map (.prim .address) (.prim u256)) (.map (.prim .address) (.prim u256)) = true := rfl

/-- Distinct widths are not equal (unlike `assignable`, which widens). -/
theorem u8_ne_u256 : Ty.beq (.prim u8) (.prim u256) = false := rfl

/-- Refinement ARGS are now compared: `MinValue<u256, 1>` ≠ `MinValue<u256, 100>`.
    (Before the fix these were wrongly `beq`-equal — same name + base.) -/
theorem refinement_args_distinguish :
    Ty.beq (.refinement "MinValue" (.prim u256) [.integer "1"])
           (.refinement "MinValue" (.prim u256) [.integer "100"]) = false := rfl

/-- …and equal args still compare equal. -/
theorem refinement_same_args :
    Ty.beq (.refinement "MinValue" (.prim u256) [.integer "1"])
           (.refinement "MinValue" (.prim u256) [.integer "1"]) = true := rfl

/-- Function NAME is part of equality: same params/returns, different name ⇒ not
    equal. Guards `Ty.function`'s name against being silently dropped again. -/
theorem function_name_distinguishes :
    Ty.beq (.function (some "f") [] []) (.function (some "g") [] []) = false := rfl

/-- …and same name ⇒ equal. -/
theorem function_same_name :
    Ty.beq (.function (some "f") [] []) (.function (some "f") [] []) = true := rfl

/-- Located equality distinguishes PROVENANCE — exactly what the old pair
    `Located` could not express. Same type, same region, different provenance: -/
theorem located_provenance_distinguishes :
    Located.beq
      { ty := .prim u256, region := .storage, provenance := .local }
      { ty := .prim u256, region := .storage, provenance := .storage } = false := rfl

/-- …and identical located types ARE equal. -/
theorem located_beq_self :
    Located.beq (.atRegion (.prim u256) .storage) (.atRegion (.prim u256) .storage) = true := rfl

-- LAWFULNESS (`Ty.beq` reflexivity etc.) lives in `Ora/Types/TypeEqLawful.lean`,
-- kept OUT of the default `Ora` build/sync-gate because the proofs unfold the
-- mutual `Ty.beq` and are expensive (~+40s). It is proven, just not on the hot
-- path — build it on demand with `lake build Ora.Types.TypeEqLawful`.

end Ora.Types
