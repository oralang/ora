/-
Ora type system — component access stays located (`τ@ρ` is never lost).

The "Type System + Regions" pillar: a variable always lives WITH a region, and so
does everything reachable inside it. `Located` already guarantees this structurally
— it is a record pairing a type with a region, so a `Located` value cannot exist
without a region. This file makes the OPERATIONAL guarantee: drilling into a located
aggregate (a tuple element, struct field, array/map element, error payload, refinement
base) always yields another `Located`, in the SAME region. You can never project your
way to a bare, region-less value.

  * `Located.project`  — one access step; the component inherits the container's
    region and provenance.
  * `Located.project_region` — the invariant: region is carried through one step.
  * `Located.projectPath` / `Located.projectPath_region` — and through any depth: a
    value reached by ANY access path lives in the original variable's region.

MODELING NOTE: components inherit the container's region unchanged (a storage struct's
fields are in storage; a calldata array's elements in calldata), matching Ora's
value-based located model — there are no cross-region component references.
-/

import Ora.Types.Ty

namespace Ora.Types

/-- A single component-access step into an aggregate value. -/
inductive Access where
  | tupleIdx (i : Nat)   -- tuple element by position
  | field (n : Name)     -- anon-struct field by name
  | elem                 -- array / slice element
  | mapValue             -- map value (`m[k]`)
  | payload              -- error-union success payload
  | unwrap               -- refinement → base
  deriving Repr, DecidableEq

/-- Project the component TYPE reached by one access step (`none` if the step does
    not apply to this type's head). -/
def Ty.project : Ty → Access → Option Ty
  | .tuple ts,         .tupleIdx i => ts[i]?
  | .anonStruct fs,    .field n    => (fs.find? (fun f => f.1 == n)).map (·.2)
  | .array e _,        .elem       => some e
  | .slice e,          .elem       => some e
  | .map _ v,          .mapValue   => some v
  | .errorUnion p _,   .payload    => some p
  | .refinement _ b _, .unwrap     => some b
  | _,                 _           => none

/-- Project a component of a LOCATED type. The component inherits the container's
    region and provenance — so the result is ALWAYS located: a sub-value can never be
    reached without a region. -/
def Located.project (l : Located) (a : Access) : Option Located :=
  (l.ty.project a).map (fun t => { ty := t, region := l.region, provenance := l.provenance })

/-- THE INVARIANT: region is carried through projection — every reachable component
    lives in its container's region. -/
theorem Located.project_region {l l' : Located} {a : Access}
    (h : l.project a = some l') : l'.region = l.region := by
  simp only [Located.project, Option.map_eq_some'] at h
  obtain ⟨t, _, rfl⟩ := h; rfl

/-- Provenance is likewise preserved through projection. -/
theorem Located.project_provenance {l l' : Located} {a : Access}
    (h : l.project a = some l') : l'.provenance = l.provenance := by
  simp only [Located.project, Option.map_eq_some'] at h
  obtain ⟨t, _, rfl⟩ := h; rfl

/-! ## Access paths — drilling arbitrarily deep keeps the region -/

/-- Follow a path of access steps. -/
def Located.projectPath : Located → List Access → Option Located
  | l, []      => some l
  | l, a :: as => (l.project a).bind (fun l' => l'.projectPath as)

/-- No matter how deep you drill, the reached value lives in the ORIGINAL variable's
    region: a variable always lives with a region — and so does everything inside it. -/
theorem Located.projectPath_region :
    ∀ {l l' : Located} {p : List Access}, l.projectPath p = some l' → l'.region = l.region
  | l, l', [], h => by simp only [Located.projectPath, Option.some.injEq] at h; rw [h]
  | l, l', a :: as, h => by
      simp only [Located.projectPath, Option.bind_eq_some] at h
      obtain ⟨m, hm, hrest⟩ := h
      exact (Located.projectPath_region hrest).trans (Located.project_region hm)

/-! ## Sanity — region is inherited by components -/

/-- A field of a storage struct lives in storage. -/
example :
    Located.project { ty := .anonStruct [("x", .prim u256)], region := .storage } (.field "x")
      = some { ty := .prim u256, region := .storage } := rfl
/-- An element of a calldata array lives in calldata. -/
example :
    Located.project { ty := .array (.prim u256) none, region := .calldata } .elem
      = some { ty := .prim u256, region := .calldata } := rfl
/-- A map value of a storage map lives in storage. -/
example :
    Located.project { ty := .map (.prim .address) (.prim u256), region := .storage } .mapValue
      = some { ty := .prim u256, region := .storage } := rfl
/-- A leaf (primitive) type has no components — nothing to project. -/
example : Located.project { ty := .prim u256, region := .memory } .elem = none := rfl

end Ora.Types
