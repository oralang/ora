/-
Ora effect lattice — mirrors §9 of the type-system spec / `src/sema/model.zig` `Effect`.

A 6-case lattice (`pure | external | side_effects | writes | reads | reads_writes`) with
`EffectFlags` and slot lists, plus the join used to compose effects in the typing
judgment. `join` builds a CANONICAL effect from the union of read/write slots and the
merge of flags (`ofParts`), so `readSlots`/`writeSlots`/`flags` recover the parts.

Laws (honest about the redundancy of the 6-case representation):
  * `join_assoc` — holds UNCONDITIONALLY (both sides canonicalize to the same parts);
  * `pure_join`/`join_pure` — `pure` is the identity ON CANONICAL effects (`Canonical`),
    which is exactly what `HasType` produces. The general 6-case type has non-canonical
    inhabitants (`reads [] f`, …) for which identity would be false, so it is stated with
    the `Canonical` hypothesis rather than overclaimed.
Mathlib-free, axiom-light.
-/

namespace Ora

/-- An effect slot (a storage/transient path); a bare name in this layer. -/
abbrev Slot := String

/-- Orthogonal observable-effect flags (mirrors `model.zig` `EffectFlags`). -/
structure EffectFlags where
  has_external : Bool := false
  has_log      : Bool := false
  has_havoc    : Bool := false
  has_lock     : Bool := false
  has_unlock   : Bool := false
  deriving Repr, DecidableEq

def EffectFlags.merge (a b : EffectFlags) : EffectFlags :=
  { has_external := a.has_external || b.has_external
    has_log      := a.has_log      || b.has_log
    has_havoc    := a.has_havoc    || b.has_havoc
    has_lock     := a.has_lock     || b.has_lock
    has_unlock   := a.has_unlock   || b.has_unlock }

@[simp] theorem EffectFlags.none_merge (a : EffectFlags) : EffectFlags.merge {} a = a := by
  cases a; simp [EffectFlags.merge]
@[simp] theorem EffectFlags.merge_none (a : EffectFlags) : EffectFlags.merge a {} = a := by
  cases a; simp [EffectFlags.merge]
theorem EffectFlags.merge_assoc (a b c : EffectFlags) :
    EffectFlags.merge (EffectFlags.merge a b) c
      = EffectFlags.merge a (EffectFlags.merge b c) := by
  simp [EffectFlags.merge, Bool.or_assoc]

/-- The 6-case effect lattice (mirrors `model.zig` `Effect`). -/
inductive Effect where
  | pure
  | external
  | side_effects (flags : EffectFlags)
  | writes (slots : List Slot) (flags : EffectFlags)
  | reads  (slots : List Slot) (flags : EffectFlags)
  | reads_writes (rd wr : List Slot) (flags : EffectFlags)
  deriving Repr, DecidableEq

def Effect.readSlots : Effect → List Slot
  | .reads r _ => r | .reads_writes r _ _ => r | _ => []
def Effect.writeSlots : Effect → List Slot
  | .writes w _ => w | .reads_writes _ w _ => w | _ => []
def Effect.flags : Effect → EffectFlags
  | .pure => {} | .external => { has_external := true }
  | .side_effects f => f | .writes _ f => f | .reads _ f => f | .reads_writes _ _ f => f

/-- The canonical effect with the given read slots, write slots, and flags. -/
def Effect.ofParts (r w : List Slot) (f : EffectFlags) : Effect :=
  match r, w with
  | [], []         => if f = {} then .pure else .side_effects f
  | _ :: _, []     => .reads r f
  | [], _ :: _     => .writes w f
  | _ :: _, _ :: _ => .reads_writes r w f

@[simp] theorem Effect.ofParts_readSlots (r w : List Slot) (f : EffectFlags) :
    (Effect.ofParts r w f).readSlots = r := by
  unfold Effect.ofParts; split <;> first | rfl | (split <;> rfl)
@[simp] theorem Effect.ofParts_writeSlots (r w : List Slot) (f : EffectFlags) :
    (Effect.ofParts r w f).writeSlots = w := by
  unfold Effect.ofParts; split <;> first | rfl | (split <;> rfl)
@[simp] theorem Effect.ofParts_flags (r w : List Slot) (f : EffectFlags) :
    (Effect.ofParts r w f).flags = f := by
  unfold Effect.ofParts
  split
  · split
    · rename_i h; subst h; rfl
    · rfl
  all_goals rfl

/-- Effect join (the `⊔` of §5): slot-union, flag-merge, via the canonical `ofParts`. -/
def Effect.join (a b : Effect) : Effect :=
  Effect.ofParts (a.readSlots ++ b.readSlots) (a.writeSlots ++ b.writeSlots)
    (a.flags.merge b.flags)

@[simp] theorem Effect.join_readSlots (a b : Effect) :
    (a.join b).readSlots = a.readSlots ++ b.readSlots := by simp [Effect.join]
@[simp] theorem Effect.join_writeSlots (a b : Effect) :
    (a.join b).writeSlots = a.writeSlots ++ b.writeSlots := by simp [Effect.join]
@[simp] theorem Effect.join_flags (a b : Effect) :
    (a.join b).flags = a.flags.merge b.flags := by simp [Effect.join]

/-- `join` is associative — unconditionally, since both sides canonicalize to the union
    of slots and merge of flags. -/
theorem Effect.join_assoc (a b c : Effect) :
    (a.join b).join c = a.join (b.join c) := by
  show Effect.ofParts _ _ _ = Effect.ofParts _ _ _
  simp only [Effect.join_readSlots, Effect.join_writeSlots, Effect.join_flags,
    List.append_assoc, EffectFlags.merge_assoc]

/-- An effect is canonical when it equals the `ofParts` of its own slots/flags — i.e. it
    has no redundant empty-slot / `{}`-flag representation. `join` always produces one. -/
def Effect.Canonical (e : Effect) : Prop :=
  e = Effect.ofParts e.readSlots e.writeSlots e.flags

theorem Effect.ofParts_canonical (r w : List Slot) (f : EffectFlags) :
    (Effect.ofParts r w f).Canonical := by
  simp [Effect.Canonical]

theorem Effect.join_canonical (a b : Effect) : (a.join b).Canonical :=
  Effect.ofParts_canonical _ _ _

/-- `pure` is a left identity on canonical effects. -/
theorem Effect.pure_join {e : Effect} (h : e.Canonical) : Effect.join .pure e = e := by
  unfold Effect.join
  simp only [Effect.readSlots, Effect.writeSlots, Effect.flags, List.nil_append,
    EffectFlags.none_merge]
  exact h.symm

/-- `pure` is a right identity on canonical effects. -/
theorem Effect.join_pure {e : Effect} (h : e.Canonical) : Effect.join e .pure = e := by
  unfold Effect.join
  simp only [Effect.readSlots, Effect.writeSlots, Effect.flags, List.append_nil,
    EffectFlags.merge_none]
  exact h.symm

end Ora
