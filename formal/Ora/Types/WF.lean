/-
Ora type system — well-formedness.

`Ty` (in `Ty.lean`) is a RAW type universe: it admits syntactic shapes that are
not valid Ora types. This file is the gate (review finding 4): a `Ty` must be
WELL-FORMED before any typing / soundness proof may treat it as
compiler-admissible.

STATUS: structural scaffold. `WF` currently enforces only STRUCTURAL
well-formedness — every component type is well-formed. Width/length bounds are
already guaranteed by construction (`FixedBytesLen`, the integer-width enums).
The SEMANTIC restrictions below are documented OBLIGATIONS, not yet enforced;
several need a declaration context or a use-position that arrive in later
modules:

  TODO(verify + enforce, structural — no context needed):
    * map key types        — restrict to admissible key types (verify the exact
                             rule against the compiler before encoding).
    * error_union nesting  — payload must not itself be an `errorUnion` (confirm).
  TODO(decl-context):
    * nominal `struct_` / `enum_` / `bitfield` / `contract` names must RESOLVE in
      a declaration environment (and resolve to the right kind).
  TODO(position-context):
    * capability handles (`storageSlot` / `storageRange`) and resource types are
      illegal in ABI / external-return positions; `resourcePlace` must not escape
      its domain.

INVARIANT TO MAINTAIN: do NOT strengthen `WF` to claim compiler-admissibility
until the obligations above land. For now, `WF t` means only "structurally
well-formed" — NECESSARY, not yet SUFFICIENT.
-/

import Ora.Types.Ty

namespace Ora.Types

/--
Structural well-formedness of a type: a composite type is (structurally)
well-formed iff its component types are. Leaf types (primitives, nominal name
references, capability handles) are structurally well-formed on their own — the
*semantic* admissibility of those (name resolution, position legality) is a
separate, not-yet-encoded obligation (see the file header).
-/
inductive WF : Ty → Prop where
  | prim (p : PrimTy) : WF (.prim p)
  | tuple {ts : List Ty} (h : ∀ t ∈ ts, WF t) : WF (.tuple ts)
  | anonStruct {fs : List (Name × Ty)} (h : ∀ f ∈ fs, WF f.2) : WF (.anonStruct fs)
  | array {t : Ty} {n : Option Nat} (h : WF t) : WF (.array t n)
  | slice {t : Ty} (h : WF t) : WF (.slice t)
  | map {k v : Ty} (hk : WF k) (hv : WF v) : WF (.map k v)
  | errorUnion {p : Ty} {es : List Ty} (hp : WF p) (he : ∀ e ∈ es, WF e) :
      WF (.errorUnion p es)
  | refinement {nm : Name} {b : Ty} {as : List RefinementArg} (hb : WF b) :
      WF (.refinement nm b as)
  | struct_ (nm : Name) : WF (.struct_ nm)
  | enum_ (nm : Name) : WF (.enum_ nm)
  | bitfield (nm : Name) : WF (.bitfield nm)
  | contract (nm : Name) : WF (.contract nm)
  | function {ps rs : List Ty} (hp : ∀ p ∈ ps, WF p) (hr : ∀ r ∈ rs, WF r) :
      WF (.function ps rs)
  | resourceDomain {nm : Name} {c : Ty} (h : WF c) : WF (.resourceDomain nm c)
  | resourcePlace {d : Ty} (h : WF d) : WF (.resourcePlace d)
  | externalProxy (nm : Name) : WF (.externalProxy nm)
  | storageSlot : WF .storageSlot
  | storageRange : WF .storageRange

/-- Every primitive is structurally well-formed (sanity check that `WF` is
    inhabited and usable). -/
theorem WF.u256_wf : WF (.prim u256) := .prim u256

/-- A representative composite is structurally well-formed:
    `map<address, u256>`. -/
theorem WF.map_address_u256 : WF (.map (.prim .address) (.prim u256)) :=
  .map (.prim .address) (.prim u256)

end Ora.Types
