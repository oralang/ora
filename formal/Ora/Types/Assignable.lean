/-
Ora type system — assignability (`τ` and `τ@ρ`), faithful to the compiler.

`Ty.assignable expected actual` mirrors `typesAssignable` (`src/sema/type_descriptors.zig`)
for the surface `Ty` universe: integer widening, structural recursion, nominal-by-name,
identity-only `function`/`resource_*`, REFINEMENT bound-subtyping (interval containment
via the reducible `parseInt?`/`Int.≤`, the `compareIntegerText` analogue) and ERROR-UNION
error-set SUBSET. It stays STRUCTURAL (the refinement base-check uses `Ty.beq`, not a
recursive `assignable`), so it kernel-reduces under `decide` (the sync gate) and unfolds
through the cheap `@[simp]` `rfl`-lemmas below.

DELIBERATE BOUNDARY: the compiler also unwraps across constructors (`refinement ↔ base`,
`payload ↔ error_union`); that's non-structural ⇒ well-founded ⇒ not kernel-decidable, so
it is NOT modeled (it would break the `decide` gate without `native_decide`). Hence
`Ty.assignable` is a SOUND UNDER-APPROXIMATION on cross-constructor pairs — and stays a
PREORDER (`AssignableLawful`), which the full compiler relation is not.
-/

import Ora.Types.Ty
import Ora.Types.Region
import Ora.Types.Internal
import Ora.Types.TypeEq
import Ora.Types.Refinement
import Ora.Types.RefinementTie

namespace Ora.Types

def IntTy.signed : IntTy → Bool | .uint _ => false | .sint _ => true
def IntTy.bits : IntTy → Nat
  | .uint .w8 => 8 | .uint .w16 => 16 | .uint .w32 => 32 | .uint .w64 => 64
  | .uint .w128 => 128 | .uint .w160 => 160 | .uint .w256 => 256
  | .sint .w8 => 8 | .sint .w16 => 16 | .sint .w32 => 32 | .sint .w64 => 64
  | .sint .w128 => 128 | .sint .w256 => 256
def IntTy.assignable (e a : IntTy) : Bool := e.signed == a.signed && decide (a.bits ≤ e.bits)

/-- reducible decimal-int parser (faithful to compareIntegerText on valid literals). -/
def digitVal (c : Char) : Option Nat :=
  if '0' ≤ c ∧ c ≤ '9' then some (c.toNat - '0'.toNat) else none
def parseNatAux : List Char → Nat → Bool → Option Nat
  | [], acc, seen => if seen then some acc else none
  | c :: cs, acc, _ => match digitVal c with
      | some d => parseNatAux cs (acc * 10 + d) true
      | none => none
def parseInt? (s : String) : Option Int :=
  match s.toList with
  | '-' :: rest => (parseNatAux rest 0 false).map (fun n => - (Int.ofNat n))
  | cs => (parseNatAux cs 0 false).map Int.ofNat

def intArgs : List RefinementArg → List String
  | [] => [] | .integer s :: r => s :: intArgs r | .typeMarker :: r => intArgs r
/-- parsed (min?, max?) bounds (mirrors refinement_semantics.bounds). -/
def refBounds (name : Name) (args : List RefinementArg) : Option (Option Int × Option Int) :=
  match RefinementName.ofString? name, intArgs args with
  | some .minValue, s :: _ => (parseInt? s).map fun n => (some n, none)
  | some .maxValue, s :: _ => (parseInt? s).map fun n => (none, some n)
  | some .inRange, lo :: hi :: _ => match parseInt? lo, parseInt? hi with
      | some l, some h => some (some l, some h) | _, _ => none
  | some .basisPoints, _ => some (some 0, some 10000)
  | some .nonZero, _ => some (some 1, none)
  | _, _ => none
def isNZA (name : Name) : Bool := RefinementName.ofString? name == some .nonZeroAddress
def refArgEq : RefinementArg → RefinementArg → Bool
  | .integer a, .integer b => a == b | .typeMarker, .typeMarker => true | _, _ => false
def refArgsEq : List RefinementArg → List RefinementArg → Bool
  | [], [] => true | a :: as, b :: bs => refArgEq a b && refArgsEq as bs | _, _ => false
def boundOk (em am : Option Int) (isMin : Bool) : Bool :=
  match em with
  | none => true
  | some e => match am with
      | none => false
      | some a => if isMin then decide (e ≤ a) else decide (a ≤ e)
def errorMem (e : Ty) (es : List Ty) : Bool := es.any (Ty.beq e)
def errorSubsetB (sub sup : List Ty) : Bool := sub.all (errorMem · sup)

/-- the refinement subtyping condition (bounds / NZA / semantic-eq), name+args only. -/
def refCond (ne : Name) (ae : List RefinementArg) (na : Name) (aa : List RefinementArg) : Bool :=
  if isNZA ne then isNZA na
  else if (ne == na && refArgsEq ae aa) then true
  else match refBounds ne ae, refBounds na aa with
       | some (emin, emax), some (amin, amax) => boundOk emin amin true && boundOk emax amax false
       | _, _ => false

mutual
def Ty.assignable : Ty → Ty → Bool
  | .refinement ne be ae, .refinement na ba aa => Ty.beq be ba && refCond ne ae na aa
  | .errorUnion pe ee, .errorUnion pa ea => Ty.assignable pe pa && errorSubsetB ea ee
  | .prim (.int e), .prim (.int a) => e.assignable a
  | .prim .bool, .prim .bool => true
  | .prim .address, .prim .address => true
  | .prim .string, .prim .string => true
  | .prim .bytes, .prim .bytes => true
  | .prim .void, .prim .void => true
  | .prim (.fixedBytes m), .prim (.fixedBytes n) => m.n == n.n
  | .tuple es, .tuple as => assignableList es as
  | .anonStruct fe, .anonStruct fa => assignableFields fe fa
  | .array e (some ne), .array a (some na) => ne == na && Ty.assignable e a
  | .array e none, .array a none => Ty.assignable e a
  | .slice e, .slice a => Ty.assignable e a
  | .map ke ve, .map ka va => Ty.assignable ke ka && Ty.assignable ve va
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
def assignableList : List Ty → List Ty → Bool
  | [], [] => true | e :: es, a :: as => Ty.assignable e a && assignableList es as | _, _ => false
def assignableFields : List (Name × Ty) → List (Name × Ty) → Bool
  | [], [] => true
  | (n1, t1) :: es, (n2, t2) :: as => n1 == n2 && Ty.assignable t1 t2 && assignableFields es as
  | _, _ => false
end

/-! cheap per-constructor unfolding lemmas (rfl; structural def) -/
@[simp] theorem asg_int (e a) :
    Ty.assignable (.prim (.int e)) (.prim (.int a)) = e.assignable a := rfl
@[simp] theorem asg_bool : Ty.assignable (.prim .bool) (.prim .bool) = true := rfl
@[simp] theorem asg_addr : Ty.assignable (.prim .address) (.prim .address) = true := rfl
@[simp] theorem asg_str : Ty.assignable (.prim .string) (.prim .string) = true := rfl
@[simp] theorem asg_bytes : Ty.assignable (.prim .bytes) (.prim .bytes) = true := rfl
@[simp] theorem asg_void : Ty.assignable (.prim .void) (.prim .void) = true := rfl
@[simp] theorem asg_fbytes (m n) :
    Ty.assignable (.prim (.fixedBytes m)) (.prim (.fixedBytes n)) = (m.n == n.n) := rfl
@[simp] theorem asg_tuple (a b) : Ty.assignable (.tuple a) (.tuple b) = assignableList a b := rfl
@[simp] theorem asg_anon (a b) :
    Ty.assignable (.anonStruct a) (.anonStruct b) = assignableFields a b := rfl
@[simp] theorem asg_arrayS (e a n m) :
    Ty.assignable (.array e (some n)) (.array a (some m)) = (n == m && Ty.assignable e a) := rfl
@[simp] theorem asg_arrayN (e a) :
    Ty.assignable (.array e none) (.array a none) = Ty.assignable e a := rfl
@[simp] theorem asg_slice (e a) : Ty.assignable (.slice e) (.slice a) = Ty.assignable e a := rfl
@[simp] theorem asg_map (ke ve ka va) :
    Ty.assignable (.map ke ve) (.map ka va) = (Ty.assignable ke ka && Ty.assignable ve va) := rfl
@[simp] theorem asg_eu (pe ee pa ea) :
    Ty.assignable (.errorUnion pe ee) (.errorUnion pa ea)
      = (Ty.assignable pe pa && errorSubsetB ea ee) := rfl
@[simp] theorem asg_refine (ne be ae na ba aa) :
    Ty.assignable (.refinement ne be ae) (.refinement na ba aa)
      = (Ty.beq be ba && refCond ne ae na aa) := rfl
@[simp] theorem asg_struct (n m) : Ty.assignable (.struct_ n) (.struct_ m) = (n == m) := rfl
@[simp] theorem asg_enum (n m) : Ty.assignable (.enum_ n) (.enum_ m) = (n == m) := rfl
@[simp] theorem asg_bitfield (n m) : Ty.assignable (.bitfield n) (.bitfield m) = (n == m) := rfl
@[simp] theorem asg_contract (n m) : Ty.assignable (.contract n) (.contract m) = (n == m) := rfl
@[simp] theorem asg_extproxy (n m) :
    Ty.assignable (.externalProxy n) (.externalProxy m) = (n == m) := rfl
@[simp] theorem asg_rdom (n c m d) :
    Ty.assignable (.resourceDomain n c) (.resourceDomain m d) = (n == m && Ty.beq c d) := rfl
@[simp] theorem asg_rplace (e a) :
    Ty.assignable (.resourcePlace e) (.resourcePlace a) = Ty.beq e a := rfl
@[simp] theorem asg_func (ne pse rse na psa rsa) :
    Ty.assignable (.function ne pse rse) (.function na psa rsa)
      = (ne == na && beqList pse psa && beqList rse rsa) := rfl
@[simp] theorem asgList_nil : assignableList [] [] = true := rfl
@[simp] theorem asgList_cons (x xs y ys) :
    assignableList (x::xs) (y::ys) = (Ty.assignable x y && assignableList xs ys) := rfl
@[simp] theorem asgFields_nil : assignableFields [] [] = true := rfl
@[simp] theorem asgFields_cons (x xs y ys) :
    assignableFields (x::xs) (y::ys)
      = (x.1 == y.1 && Ty.assignable x.2 y.2 && assignableFields xs ys) := rfl

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
