/-
Ora type system — assignability (`τ` and `τ@ρ`).

Mirrors the compiler's assignability seam:
  * `typesAssignable`  (`type_descriptors.zig:299`) — type ⇒ type;
  * `integerAssignable` — same signedness, actual width ≤ expected width;
  * `isAssignable`     (`region.zig:22`) — located: type-assignable ∧ region-assignable.

Argument convention follows the compiler: `assignable expected actual` asks
"can a value of type `actual` be assigned where `expected` is wanted?".

FAITHFUL here: integer widening, leaf identity, tuple / anon-struct (same shape) /
array / slice / map structural recursion, and nominal-by-name. Reflexivity holds
for every constructor.

DEFERRED (documented, conservative) — these need more than a structural mirror:
  * refinement subtyping (`refinementSubtypeAssignable`: unwrap-to-base AND
    min/max bounds comparison) — modeled only as same name + base assignable;
  * full `error_union` subtyping (payload-as-actual, error-set membership, error
    subset) — modeled only as payload + pairwise errors;
  * anon-struct ↔ tuple positional coercion — not modeled.

`function` / `resource_*` are identity-only in the compiler (only `typeEql`
admits them); modeled faithfully here via structural equality (`Ty.beq`).
-/

import Ora.Types.Ty
import Ora.Types.Region
import Ora.Types.Internal
import Ora.Types.TypeEq

namespace Ora.Types

/-! ## Integer assignability -/

def IntTy.signed : IntTy → Bool
  | .uint _ => false
  | .sint _ => true

def IntTy.bits : IntTy → Nat
  | .uint .w8 => 8 | .uint .w16 => 16 | .uint .w32 => 32 | .uint .w64 => 64
  | .uint .w128 => 128 | .uint .w160 => 160 | .uint .w256 => 256
  | .sint .w8 => 8 | .sint .w16 => 16 | .sint .w32 => 32 | .sint .w64 => 64
  | .sint .w128 => 128 | .sint .w256 => 256

/-- `integerAssignable`: same signedness, actual no wider than expected. -/
def IntTy.assignable (expected actual : IntTy) : Bool :=
  expected.signed == actual.signed && decide (actual.bits ≤ expected.bits)

/-! ## Type assignability -/

mutual
/-- `Ty.assignable expected actual` — can `actual` be assigned where `expected`
    is wanted? -/
def Ty.assignable : Ty → Ty → Bool
  -- refinement: same name + base assignable (aligned, so the recursion stays
  -- structural). The compiler also unwraps a refinement to its base and compares
  -- bounds (`refinementSubtypeAssignable`); both are DEFERRED here.
  | .refinement n be _, .refinement m ba _ => n == m && Ty.assignable be ba
  -- integers: widening within the same signedness
  | .prim (.int e), .prim (.int a) => e.assignable a
  -- leaf primitives: identity
  | .prim .bool, .prim .bool => true
  | .prim .address, .prim .address => true
  | .prim .string, .prim .string => true
  | .prim .bytes, .prim .bytes => true
  | .prim .void, .prim .void => true
  | .prim (.fixedBytes m), .prim (.fixedBytes n) => m.n == n.n
  -- aggregates
  | .tuple es, .tuple as => assignableList es as
  | .anonStruct fe, .anonStruct fa => assignableFields fe fa
  | .array e (some ne), .array a (some na) => ne == na && Ty.assignable e a
  | .array e none, .array a none => Ty.assignable e a
  | .slice e, .slice a => Ty.assignable e a
  | .map ke ve, .map ka va => Ty.assignable ke ka && Ty.assignable ve va
  | .errorUnion pe ee, .errorUnion pa ea => Ty.assignable pe pa && assignableList ee ea
  -- nominal: same name
  | .struct_ n, .struct_ m => n == m
  | .enum_ n, .enum_ m => n == m
  | .bitfield n, .bitfield m => n == m
  | .contract n, .contract m => n == m
  | .externalProxy n, .externalProxy m => n == m
  -- capability handles
  | .storageSlot, .storageSlot => true
  | .storageRange, .storageRange => true
  -- callables / resources are IDENTITY-ONLY in the compiler (they are not in
  -- `typesAssignable`, so only `typeEql` admits them) — use structural equality.
  | .function ne pse rse, .function na psa rsa => ne == na && beqList pse psa && beqList rse rsa
  | .resourceDomain n c, .resourceDomain m d => n == m && Ty.beq c d
  | .resourcePlace e, .resourcePlace a => Ty.beq e a
  | _, _ => false
def assignableList : List Ty → List Ty → Bool
  | [], [] => true
  | e :: es, a :: as => Ty.assignable e a && assignableList es as
  | _, _ => false
def assignableFields : List (Name × Ty) → List (Name × Ty) → Bool
  | [], [] => true
  | fe :: es, fa :: as => fe.1 == fa.1 && Ty.assignable fe.2 fa.2 && assignableFields es as
  | _, _ => false
end

/-! ## Internal assignability (adds the `never` ⊥ rules) -/

/-- `InternalTy.assignable expected actual` — extends `Ty.assignable` with the
    bottom type: `never` is assignable to anything; only `never` is assignable to
    a `never` slot. -/
def InternalTy.assignable : InternalTy → InternalTy → Bool
  | _, .never => true
  | .never, .runtime _ => false
  | .runtime e, .runtime a => e.assignable a

/-! ## Located assignability (`isAssignable`) -/

/-- `Located.assignable src dst` — can a value located at `src` be assigned into
    `dst`? Type-assignable (`dst.ty` accepts `src.ty`) AND region-assignable
    (`src.region` coerces to `dst.region`). Mirrors `isAssignable(from, to)`. -/
def Located.assignable (src dst : Located) : Bool :=
  dst.ty.assignable src.ty && src.region.assignableTo dst.region

/-! ## Theorems -/

/-- Integer widening: `u8` is assignable to a `u256` slot. -/
theorem u8_to_u256 : Ty.assignable (.prim u256) (.prim u8) = true := rfl

/-- No narrowing: `u256` is not assignable to a `u8` slot. -/
theorem u256_not_to_u8 : Ty.assignable (.prim u8) (.prim u256) = false := rfl

/-- No signed/unsigned mixing: `i8` is not assignable to a `u8` slot. -/
theorem signedness_mismatch : Ty.assignable (.prim u8) (.prim i8) = false := rfl

/-- Same width, same signedness assigns. -/
theorem u256_to_u256 : Ty.assignable (.prim u256) (.prim u256) = true := rfl

/-- Nominal assignability is by name. -/
theorem struct_same_name : Ty.assignable (.struct_ "P") (.struct_ "P") = true := rfl
theorem struct_diff_name : Ty.assignable (.struct_ "P") (.struct_ "Q") = false := rfl

/-- Function assignability is identity-only and includes the name: functions
    differing only in name are not assignable. -/
theorem function_name_not_assignable :
    Ty.assignable (.function (some "f") [] []) (.function (some "g") [] []) = false := rfl

/-- A tuple of widenable elements is assignable. -/
theorem tuple_widening :
    Ty.assignable (.tuple [.prim u256, .prim u256]) (.tuple [.prim u8, .prim u16]) = true := rfl

/-- `never` is the bottom: assignable to any slot. -/
theorem never_bottom (x : InternalTy) : InternalTy.assignable x .never = true := by
  cases x <;> rfl

/-- Only `never` is assignable to a `never` slot. -/
theorem never_only_from_never (t : Ty) :
    InternalTy.assignable .never (.runtime t) = false := rfl

/-- Located: a `u8@calldata` value assigns into a `u256@memory` slot
    (widening + the `calldata → memory` region coercion). -/
theorem located_calldata_to_memory :
    Located.assignable
      (.atRegion (.prim u8) .calldata) (.atRegion (.prim u256) .memory) = true := rfl

/-- Located: nothing assigns INTO calldata from memory (read-only region). -/
theorem located_not_into_calldata :
    Located.assignable
      (.atRegion (.prim u256) .memory) (.atRegion (.prim u256) .calldata) = false := rfl

end Ora.Types
