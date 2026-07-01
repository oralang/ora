/-
Ora typing judgment — `S ; Γ ; Λ ⊢ e : σ ! ϵ ⊣ Λ'` (skeleton).

This is the §5 typing relation from the type-system spec, as a Lean inductive. It is
a SKELETON: a minimal core-expression syntax and a representative set of rules, built
so the judgment is real, derivable, and reasoned about inductively — reusing the
already-proven `Ty.assignable` / `Located.assignable` for the side conditions, so
subsumption and storage typing are checked relative to this skeleton's rules.
This is not yet a compiler-wide soundness theorem.

TWO deliberate design choices, made now so they are not future drift points:

  * EFFECTS. `Effect` here mirrors the §9 lattice / `src/sema/model.zig` `Effect`
    SHAPE (6 cases: `pure | external | side_effects | writes | reads | reads_writes`,
    with `EffectFlags` and structured slots) rather than an ad-hoc type. When the
    effect layer is mechanized for real, this folds into it / gets a sync-gate pin —
    the seam is intentional, not accidental. (Slots are still bare names here; the
    structured `EffectSlot` paths of §9 are the additive next step.)
  * LOCKSET. The judgment THREADS the lockset `Λ` (input `Λ`, output `Λ'`): `lock`
    adds a slot, `unlock` removes it, and `store` REQUIRES its slot be unlocked
    (`s ∉ Λ`). This is what makes the lock discipline — and the reentrancy theorem T5
    (§10) — stateable against this skeleton, rather than deferred.
-/

import Ora.Types.Assignable
import Ora.Types.Internal
import Ora.Types.Effect

namespace Ora.Types

abbrev Var := String

/-! The effect lattice (`Effect`, `EffectFlags`, `Slot`, `Effect.join` + its monoid laws)
    lives in `Ora.Types.Effect`; it is in scope here unqualified. -/

/-! ## Core expressions (skeleton) -/

inductive Expr where
  | intLit  : Nat → Expr
  | boolLit : Bool → Expr
  | var     : Var → Expr
  | letE    : Var → Expr → Expr → Expr
  | ife     : Expr → Expr → Expr → Expr
  | load    : Slot → Expr
  | store   : Slot → Expr → Expr
  | lock    : Slot → Expr
  | unlock  : Slot → Expr
  | assign  : Var → Expr → Expr
  deriving Repr, DecidableEq

/-! ## Environments -/

abbrev Mut := Bool                              -- true = mutable (`var`), false = `let`
abbrev Context := List (Var × Located × Mut)
abbrev StorageLayout := List (Slot × Located)
abbrev Lockset := List Slot                     -- transaction-scoped write-lockset

def Context.lookup (Γ : Context) (x : Var) : Option (Located × Mut) :=
  (Γ.find? (·.1 == x)).map (fun e => (e.2.1, e.2.2))

def StorageLayout.lookup (S : StorageLayout) (s : Slot) : Option Located :=
  (S.find? (·.1 == s)).map (·.2)

/-! ## The judgment  `S ; Γ ; Λ ⊢ e : σ ! ϵ ⊣ Λ'`

    Input lockset `Λ`, output lockset `Λ'`. `sub` / `store` / `assign` discharge their
    side conditions with the proven `Located.assignable` / `Ty.assignable`; `store`
    additionally requires its slot be UNLOCKED, and `lock` / `unlock` update `Λ`. -/

inductive HasType :
    StorageLayout → Context → Lockset → Expr → Located → Effect → Lockset → Prop where
  | intLit {S Γ Λ n} :
      HasType S Γ Λ (.intLit n) ⟨.prim u256, .stack, .local⟩ .pure Λ
  | boolLit {S Γ Λ b} :
      HasType S Γ Λ (.boolLit b) ⟨.prim .bool, .stack, .local⟩ .pure Λ
  | var {S Γ Λ x σ mu} :
      Context.lookup Γ x = some (σ, mu) →
      HasType S Γ Λ (.var x) σ .pure Λ
  | sub {S Γ Λ Λ' e σ σ' ϵ} :
      HasType S Γ Λ e σ ϵ Λ' →
      Located.assignable σ σ' = true →
      HasType S Γ Λ e σ' ϵ Λ'
  | letE {S Γ Λ Λ₁ Λ₂ x e₁ e₂ σ₁ σ₂ ϵ₁ ϵ₂} :
      HasType S Γ Λ e₁ σ₁ ϵ₁ Λ₁ →
      HasType S ((x, σ₁, false) :: Γ) Λ₁ e₂ σ₂ ϵ₂ Λ₂ →
      HasType S Γ Λ (.letE x e₁ e₂) σ₂ (ϵ₁.join ϵ₂) Λ₂
  | ife {S Γ Λ Λc Λ' c t f σ ρc ϵc ϵt ϵf} :
      HasType S Γ Λ c ⟨.prim .bool, ρc, .local⟩ ϵc Λc →
      HasType S Γ Λc t σ ϵt Λ' →
      HasType S Γ Λc f σ ϵf Λ' →
      HasType S Γ Λ (.ife c t f) σ ((ϵc.join ϵt).join ϵf) Λ'
  | load {S Γ Λ s τ ρ π} :
      StorageLayout.lookup S s = some ⟨τ, ρ, π⟩ →
      HasType S Γ Λ (.load s) ⟨τ, .stack, .local⟩ (.reads [s] {}) Λ
  | store {S Γ Λ Λ₁ s e τ ρ π σ ϵ} :
      StorageLayout.lookup S s = some ⟨τ, ρ, π⟩ →
      HasType S Γ Λ e σ ϵ Λ₁ →
      Ty.assignable τ σ.ty = true →                 -- value type fits the slot
      s ∉ Λ₁ →                                       -- THE lock discipline: slot must be unlocked
      HasType S Γ Λ (.store s e) ⟨.prim .void, .stack, .local⟩
        (ϵ.join (.writes [s] {})) Λ₁
  | lock {S Γ Λ s σs} :
      StorageLayout.lookup S s = some σs →
      s ∉ Λ →                                        -- cannot re-lock an already-locked slot
      HasType S Γ Λ (.lock s) ⟨.prim .void, .stack, .local⟩
        (.side_effects { has_lock := true }) (s :: Λ)
  | unlock {S Γ Λ s σs} :
      StorageLayout.lookup S s = some σs →
      HasType S Γ Λ (.unlock s) ⟨.prim .void, .stack, .local⟩
        (.side_effects { has_unlock := true }) (Λ.erase s)
  | assign {S Γ Λ Λ₁ x e σ σ' ϵ} :
      Context.lookup Γ x = some (σ, true) →          -- must be a mutable binding
      HasType S Γ Λ e σ' ϵ Λ₁ →
      Located.assignable σ' σ = true →               -- new value assignable into the slot
      HasType S Γ Λ (.assign x e) ⟨.prim .void, .stack, .local⟩ ϵ Λ₁

namespace HasType

/-! ## Sanity: derivable examples + the lock discipline in action -/

/-- A literal types as `u256@stack`, pure, lockset unchanged. -/
example : HasType [] [] [] (.intLit 5) ⟨.prim u256, .stack, .local⟩ .pure [] := .intLit

/-- Subsumption via region coercion (`stack ↦ memory`), reusing `Located.assignable`. -/
example : HasType [] [] [] (.intLit 5) ⟨.prim u256, .memory, .local⟩ .pure [] :=
  .sub .intLit (by decide)

/-- Storing a literal into an UNLOCKED `u256@storage` slot is well-typed. -/
example :
    HasType [("bal", ⟨.prim u256, .storage, .local⟩)] [] []
      (.store "bal" (.intLit 5)) ⟨.prim .void, .stack, .local⟩ (.writes ["bal"] {}) [] :=
  .store rfl .intLit (by decide) (by decide)

/-- `lock` adds its slot to the output lockset. -/
example :
    HasType [("bal", ⟨.prim u256, .storage, .local⟩)] [] []
      (.lock "bal") ⟨.prim .void, .stack, .local⟩
      (.side_effects { has_lock := true }) ["bal"] :=
  .lock rfl (by decide)

/-- A `let` binds and threads both effects and the lockset. -/
example :
    HasType [] [] [] (.letE "x" (.intLit 1) (.var "x")) ⟨.prim u256, .stack, .local⟩ .pure [] :=
  .letE .intLit (.var rfl)

/-! ## A first meta-theorem: literals are pure (inversion over the judgment) -/

/-- Every typing of an integer literal has the pure effect — even after subsumption,
    which preserves the effect. Demonstrates inductive reasoning over `HasType`. -/
theorem intLit_pure {S Γ Λ Λ' n σ ϵ} (h : HasType S Γ Λ (.intLit n) σ ϵ Λ') : ϵ = .pure := by
  generalize he : Expr.intLit n = e at h
  induction h with
  | intLit => rfl
  | sub _ _ ih => exact ih he
  | _ => nomatch he

/-- Every well-typed expression's effect is **canonical** — so the §9 effect-monoid laws
    (`Effect.pure_join`/`join_pure`) apply to any effect the typing judgment produces. The
    leaves emit canonical effects (`pure`, `reads [s] {}`, `side_effects`), `sub`/`assign`
    inherit, and every composite is a `join` (canonical by construction). -/
theorem effect_canonical {S Γ Λ Λ' e σ ϵ} (h : HasType S Γ Λ e σ ϵ Λ') : ϵ.Canonical := by
  induction h with
  | intLit | boolLit | var _ | load _ | lock _ _ | unlock _ =>
      simp [Effect.Canonical, Effect.readSlots, Effect.writeSlots, Effect.flags, Effect.ofParts]
  | sub _ _ ih => exact ih
  | assign _ _ _ ih => exact ih
  | letE _ _ _ _ => exact Effect.join_canonical _ _
  | ife _ _ _ _ _ _ => exact Effect.join_canonical _ _
  | store _ _ _ _ _ => exact Effect.join_canonical _ _

/-- Consequence: `pure` composes away against any typing effect (monoid identity, applied
    via `effect_canonical`). -/
theorem pure_join_effect {S Γ Λ Λ' e σ ϵ} (h : HasType S Γ Λ e σ ϵ Λ') :
    Effect.join .pure ϵ = ϵ :=
  Effect.pure_join (effect_canonical h)

end HasType

end Ora.Types
