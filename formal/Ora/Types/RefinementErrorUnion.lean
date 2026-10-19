import Ora.Types.RefinementBridge
import Ora.Types.LocatedProjection
import Ora.Types.TypeEq

/-!
Refinement obligations on error-union returns.

`Ty.errorUnion` and `Ty.project .payload` describe the existing static shape;
`Except` supplies the Ok/Err value distinction. The success carrier is the base
carrier, not a subtype containing a proof: a refinement annotation is not evidence
that the returned value satisfies its predicate. Predicates and interpreted bounds
come from the existing refinement value/registry layer, without new bound parsing.

The model starts with an evaluated, discriminated result. Relating a concrete
compiler tag and payload extraction to these constructors is a later binding
obligation; this file establishes no ABI decoding or compiler correctness claim.
-/

namespace Ora.Types.ErrorUnionRefinement

universe u v
variable {α : Type u} {ε : Type v}

/-- No refinement obligation is generated for an error carrier. -/
def obligation? (R : α → Prop) (value : Except ε α) : Option Prop :=
  value.toOption.map R

/-- Refinement guard after discriminating the return value. -/
def Guard (R : α → Prop) : Except ε α → Prop
  | .ok payload => R payload
  | .error _ => True

/-- `none` is guard failure; an accepted return preserves its tag and value. -/
def checkedReturn? (R : α → Prop) [DecidablePred R] :
    Except ε α → Option (Except ε α)
  | .ok payload => if R payload then some (.ok payload) else none
  | .error err => some (.error err)

theorem ok_satisfies_guard {R : α → Prop} {payload : α} (h : R payload) :
    Guard R (.ok payload : Except ε α) := h

theorem ok_violates_guard {R : α → Prop} {payload : α} (h : ¬ R payload) :
    ¬ Guard R (.ok payload : Except ε α) := h

theorem ok_obligation (R : α → Prop) (payload : α) :
    obligation? R (.ok payload : Except ε α) = some (R payload) := rfl

theorem err_has_no_obligation (R : α → Prop) (err : ε) :
    obligation? R (.error err) = none := rfl

theorem err_passes_guard (R : α → Prop) (err : ε) :
    Guard R (.error err) := True.intro

theorem checked_ok_satisfies {R : α → Prop} [DecidablePred R]
    {payload : α} (h : R payload) :
    checkedReturn? R (.ok payload : Except ε α) = some (.ok payload) := by
  simp only [checkedReturn?, if_pos h]

theorem checked_ok_violates {R : α → Prop} [DecidablePred R]
    {payload : α} (h : ¬ R payload) :
    checkedReturn? R (.ok payload : Except ε α) = none := by
  simp only [checkedReturn?, if_neg h]

theorem checked_err_preserved (R : α → Prop) [DecidablePred R] (err : ε) :
    checkedReturn? R (.error err) = some (.error err) := rfl

theorem checked_return_iff (R : α → Prop) [DecidablePred R]
    (value output : Except ε α) :
    checkedReturn? R value = some output ↔ output = value ∧ Guard R value := by
  cases value with
  | ok payload =>
      by_cases h : R payload <;> simp [checkedReturn?, Guard, h, eq_comm]
  | error err => simp [checkedReturn?, Guard, eq_comm]

/-- A runtime payload read requires the actual Ok constructor, not its type. -/
theorem payload_read_iff_ok (value : Except ε α) (payload : α) :
    value.toOption = some payload ↔ value = .ok payload := by
  cases value <;> simp [Except.toOption]

/-- An Err has no success payload even if its representation shares the carrier. -/
theorem err_payload_absent (err : ε) :
    (Except.error err : Except ε α).toOption = none := rfl

/-- No unconditional total read can be a valid success-payload projection. -/
theorem discriminant_required (err : ε) :
    ¬ ∃ read : Except ε α → α, ∀ value, value.toOption = some (read value) := by
  intro ⟨read, h⟩
  have invalid := h (.error err)
  simp only [Except.toOption, reduceCtorEq] at invalid

/-- Static payload projection remains available for BOTH runtime variants. -/
theorem static_payload_shape (payload : Ty) (errors : List Ty) :
    (Ty.errorUnion payload errors).project .payload = some payload := rfl

/-- Use the existing registry predicate with already interpreted carrier bounds. -/
theorem registry_obligation [LE α] [Zero α] [OfNat α 10000]
    {name : RefinementName} {bounds : List α} {R : α → Prop}
    (h : name.runtimeDenotation bounds = some R) (value : Except ε α) :
    (name.runtimeDenotation bounds).map (fun p => obligation? p value) =
      some (obligation? R value) := by
  rw [h]
  rfl

/-- Concrete false-accept discriminator: identical wrapper types do not prove R. -/
theorem wrapper_type_does_not_discharge_obligation :
    let t := Ty.errorUnion (.refinement "MinValue" (.prim u256) [.integer "1"])
      [.enum_ "AbiDecodeError"]
    Ty.beq t t = true ∧
      ¬ Guard (Ora.Refine.MinValue 1) (.ok 0 : Except Nat Nat) := by
  constructor
  · rfl
  · exact Nat.not_succ_le_zero 0

/-- Checking an Err's raw word as T can reject a valid error return. -/
theorem unconditional_error_word_check_false_rejects :
    Guard (Ora.Refine.MinValue 1) (.error 0 : Except Nat Nat) ∧
      ¬ Ora.Refine.MinValue 1 (0 : Nat) :=
  ⟨True.intro, Nat.not_succ_le_zero 0⟩

/-- Even a word satisfying R is still NOT a success payload on the Err branch. -/
theorem satisfying_error_word_does_not_authorize_payload :
    Ora.Refine.MinValue 1 (1 : Nat) ∧
      (Except.error 1 : Except Nat Nat).toOption = none :=
  ⟨Nat.le_refl 1, rfl⟩

/-- The signed-bound case uses the existing predicate, not natural bound parsing. -/
theorem signed_floor_boundary :
    Guard (Ora.Refine.MinValue (-5)) (.ok (-5) : Except Nat Int) ∧
      ¬ Guard (Ora.Refine.MinValue (-5)) (.ok (-6) : Except Nat Int) := by
  change ((-5 : Int) ≤ -5) ∧ ¬ ((-5 : Int) ≤ -6)
  decide

end Ora.Types.ErrorUnionRefinement
