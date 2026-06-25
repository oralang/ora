/-
Ora type system — well-formedness and position legality.

Two distinct, compiler-verified notions (do not conflate them):

1. TYPE-INTRINSIC well-formedness `WF Γ t` — `t` is structurally admissible AND
   every nominal name it mentions resolves (to the right kind) in the
   declaration environment `Γ`. A `map` type IS well-formed.

2. POSITION legality — separate predicates on a type for a USE position. These
   mirror the compiler's `typeContains*` checks (`src/sema/type_check.zig`):
     * public / ABI surface (public params `:1916`, returns `:1938`, log fields
       `:1305`) must NOT expose an opaque capability —
       `containsStorageCapability ∨ containsResourcePlace` (`:288`);
     * runtime-value positions (params `:1974`, returns `:1987`, locals `:2001`)
       must NOT contain a `map` (maps are storage roots, not first-class
       values); and must not contain a resource place (`:2145/:2158/:2172`).

   A `map` is well-formed (#1) yet illegal as a runtime value (#2) — that split
   is the point.

NOT YET covered (next depth): storage-position rules and the richer contract /
trait declaration tables.
-/

import Ora.Types.Ty
import Ora.Types.Decl

namespace Ora.Types

/-! ## 1. Type-intrinsic well-formedness (environment-relative) -/

/--
`WF Γ t` — `t` is structurally well-formed and every nominal name in it resolves
in `Γ`. Composite types require their components WF; nominal heads require
resolution to a declaration of the matching kind.

SCOPE (don't over-read this): `WF Γ t` checks that each nominal name *resolves
to the right KIND* and recurses through the type's immediate structure.
`DeclEnvWF Γ` below checks the resolved declaration bodies. Only
`externalProxy`'s trait name is NOT checked (separate trait table).
-/
inductive WF (Γ : DeclEnv) : Ty → Prop where
  | prim (p : PrimTy) : WF Γ (.prim p)
  | tuple {ts : List Ty} (h : ∀ t ∈ ts, WF Γ t) : WF Γ (.tuple ts)
  | anonStruct {fs : List (Name × Ty)} (h : ∀ f ∈ fs, WF Γ f.2) : WF Γ (.anonStruct fs)
  | array {t : Ty} {n : Option Nat} (h : WF Γ t) : WF Γ (.array t n)
  | slice {t : Ty} (h : WF Γ t) : WF Γ (.slice t)
  | map {k v : Ty} (hk : WF Γ k) (hv : WF Γ v) : WF Γ (.map k v)
  | errorUnion {p : Ty} {es : List Ty} (hp : WF Γ p) (he : ∀ e ∈ es, WF Γ e) :
      WF Γ (.errorUnion p es)
  | refinement {nm : Name} {b : Ty} {as : List RefinementArg} (hb : WF Γ b) :
      WF Γ (.refinement nm b as)
  | struct_ {nm : Name} (h : Γ.resolvesNominal (.struct_ nm)) : WF Γ (.struct_ nm)
  | enum_ {nm : Name} (h : Γ.resolvesNominal (.enum_ nm)) : WF Γ (.enum_ nm)
  | bitfield {nm : Name} (h : Γ.resolvesNominal (.bitfield nm)) : WF Γ (.bitfield nm)
  | contract {nm : Name} (h : Γ.resolvesNominal (.contract nm)) : WF Γ (.contract nm)
  | function {nm : Option Name} {ps rs : List Ty} (hp : ∀ p ∈ ps, WF Γ p) (hr : ∀ r ∈ rs, WF Γ r) :
      WF Γ (.function nm ps rs)
  | resourceDomain {nm : Name} {c : Ty}
      (hRes : Γ.resolvesNominal (.resourceDomain nm c)) (hc : WF Γ c) :
      WF Γ (.resourceDomain nm c)
  | resourcePlace {d : Ty} (h : WF Γ d) : WF Γ (.resourcePlace d)
  -- `externalProxy`'s trait name is resolved in a separate trait table, not
  -- `DeclEnv`; no premise here yet.
  | externalProxy (nm : Name) : WF Γ (.externalProxy nm)
  | storageSlot : WF Γ .storageSlot
  | storageRange : WF Γ .storageRange

/-! ## Declaration-environment well-formedness

    `WF Γ t` only checks a nominal type's head name. `Decl.WF Γ d` checks the
    types stored inside a declaration body, and `DeclEnvWF Γ` requires every
    declaration body in the environment to be well-formed under the same
    environment. This is intentionally one layer above `WF`: recursive nominal
    declarations are allowed because a field of type `struct Node` only needs the
    `Node` head to resolve in `Γ`; we do not unfold it indefinitely.
-/

def FieldDeclWF (Γ : DeclEnv) (f : FieldDecl) : Prop :=
  WF Γ f.ty

def BitfieldFieldDeclWF (Γ : DeclEnv) (f : BitfieldFieldDecl) : Prop :=
  WF Γ f.ty

def VariantPayloadWF (Γ : DeclEnv) : VariantPayload → Prop
  | .none => True
  | .positional ts => ∀ t ∈ ts, WF Γ t
  | .named fs => ∀ f ∈ fs, FieldDeclWF Γ f

def EnumVariantDeclWF (Γ : DeclEnv) (v : EnumVariantDecl) : Prop :=
  VariantPayloadWF Γ v.payload

def DeclWF (Γ : DeclEnv) : Decl → Prop
  | .struct_ fs => ∀ f ∈ fs, FieldDeclWF Γ f
  | .enum_ vs => ∀ v ∈ vs, EnumVariantDeclWF Γ v
  | .bitfield fs => ∀ f ∈ fs, BitfieldFieldDeclWF Γ f
  -- Contract members are modeled as names only in this layer; signatures and
  -- storage variables live in the later contract-declaration layer.
  | .contract _ => True
  | .resource c => WF Γ c

def DeclEnv.NoDuplicateNames (Γ : DeclEnv) : Prop :=
  (Γ.map Prod.fst).Nodup

def DeclEnvWF (Γ : DeclEnv) : Prop :=
  DeclEnv.NoDuplicateNames Γ ∧ ∀ n d, (n, d) ∈ Γ → DeclWF Γ d

/-! ## 2. Containment predicates (mirror `typeContains*`)

    Faithful recursive mirrors of the compiler's containment checks. -/

-- Each predicate is a mutual block: the `Ty` recursion plus explicit `List Ty`
-- / `List (Name × Ty)` helpers, so the recursion is structural (Lean does not
-- accept `List.any f` with `f` the recursive function).

mutual
/-- Does `t` contain a storage-capability handle (`storage_slot`/`storage_range`)?
    Mirrors `typeContainsStorageCapability`. -/
def Ty.containsStorageCapability : Ty → Bool
  | .storageSlot => true
  | .storageRange => true
  | .tuple ts => anyStorageCap ts
  | .anonStruct fs => anyFieldStorageCap fs
  | .array t _ => t.containsStorageCapability
  | .slice t => t.containsStorageCapability
  | .map k v => k.containsStorageCapability || v.containsStorageCapability
  | .errorUnion p es => p.containsStorageCapability || anyStorageCap es
  | .refinement _ b _ => b.containsStorageCapability
  | .resourceDomain _ c => c.containsStorageCapability
  | .function _ ps rs => anyStorageCap ps || anyStorageCap rs
  | _ => false
def anyStorageCap : List Ty → Bool
  | [] => false
  | t :: ts => t.containsStorageCapability || anyStorageCap ts
def anyFieldStorageCap : List (Name × Ty) → Bool
  | [] => false
  | f :: fs => f.2.containsStorageCapability || anyFieldStorageCap fs
end

mutual
/-- Does `t` contain a resource place? Mirrors `typeContainsResourcePlace`. -/
def Ty.containsResourcePlace : Ty → Bool
  | .resourcePlace _ => true
  | .tuple ts => anyResourcePlace ts
  | .anonStruct fs => anyFieldResourcePlace fs
  | .array t _ => t.containsResourcePlace
  | .slice t => t.containsResourcePlace
  | .map k v => k.containsResourcePlace || v.containsResourcePlace
  | .errorUnion p es => p.containsResourcePlace || anyResourcePlace es
  | .refinement _ b _ => b.containsResourcePlace
  | .resourceDomain _ c => c.containsResourcePlace
  | .function _ ps rs => anyResourcePlace ps || anyResourcePlace rs
  | _ => false
def anyResourcePlace : List Ty → Bool
  | [] => false
  | t :: ts => t.containsResourcePlace || anyResourcePlace ts
def anyFieldResourcePlace : List (Name × Ty) → Bool
  | [] => false
  | f :: fs => f.2.containsResourcePlace || anyFieldResourcePlace fs
end

mutual
/-- Does `t` contain a `map`? Mirrors `typeContainsMap`. -/
def Ty.containsMap : Ty → Bool
  | .map _ _ => true
  | .tuple ts => anyMap ts
  | .anonStruct fs => anyFieldMap fs
  | .array t _ => t.containsMap
  | .slice t => t.containsMap
  | .errorUnion p es => p.containsMap || anyMap es
  | .refinement _ b _ => b.containsMap
  | .resourceDomain _ c => c.containsMap
  | .function _ ps rs => anyMap ps || anyMap rs
  | _ => false
def anyMap : List Ty → Bool
  | [] => false
  | t :: ts => t.containsMap || anyMap ts
def anyFieldMap : List (Name × Ty) → Bool
  | [] => false
  | f :: fs => f.2.containsMap || anyFieldMap fs
end

/-! ## 3. Position legality (compiler-verified) -/

/-- Exposes an opaque runtime capability — illegal at public / ABI boundaries.
    Mirrors `type_check.zig:288`. -/
def Ty.exposesOpaqueCapability (t : Ty) : Bool :=
  t.containsStorageCapability || t.containsResourcePlace

/-- May appear as a first-class runtime value (param / return / local): no map and
    no resource place. Mirrors the runtime-value rejections
    (`type_check.zig:1974/1987/2001`, `:2145/2158/2172`). -/
def Ty.validRuntimeValue (t : Ty) : Bool :=
  !t.containsMap && !t.containsResourcePlace

/-- May cross the public / ABI boundary: does not expose an opaque capability. -/
def Ty.validAbiBoundary (t : Ty) : Bool :=
  !t.exposesOpaqueCapability

/-! ## Theorems -/

/-- A primitive is well-formed in any environment. -/
theorem WF.prim_wf (Γ : DeclEnv) (p : PrimTy) : WF Γ (.prim p) := .prim p

/-- A declared struct is well-formed (resolution discharged by the env). -/
theorem WF.point_wf : WF examplePoint (.struct_ "Point") :=
  .struct_ point_resolves

/-- A declared resource domain is well-formed (name resolves + carrier WF). -/
theorem WF.resource_wf : WF exampleResource (.resourceDomain "Token" (.prim u256)) :=
  .resourceDomain resource_resolves (.prim u256)

theorem WF.resource_carrier_mismatch_not_wf :
    ¬ WF exampleResource (.resourceDomain "Token" (.prim .bool)) := by
  intro h
  cases h with
  | resourceDomain hRes _ =>
      exact resource_carrier_mismatch_does_not_resolve hRes

/-- NO DANGLING references: an undeclared nominal name is not well-formed. -/
theorem WF.dangling_struct_not_wf : ¬ WF ([] : DeclEnv) (.struct_ "Missing") := by
  intro h
  cases h with
  | struct_ hRes => obtain ⟨_, hfs⟩ := hRes; simp [DeclEnv.structFields?, DeclEnv.lookup] at hfs

theorem WF.dangling_resource_not_wf :
    ¬ WF ([] : DeclEnv) (.resourceDomain "R" (.prim u256)) := by
  intro h
  cases h with
  | resourceDomain hRes _ =>
      change DeclEnv.resourceCarrier? ([] : DeclEnv) "R" = some (.prim u256) at hRes
      simp [DeclEnv.resourceCarrier?, DeclEnv.lookup] at hRes

/-! ## Declaration-environment WF examples and regressions -/

theorem DeclEnvWF.empty : DeclEnvWF [] := by
  constructor
  · simp [DeclEnv.NoDuplicateNames]
  · intro n d h
    cases h

theorem DeclEnvWF.examplePoint : DeclEnvWF examplePoint := by
  constructor
  · change (["Point"]).Nodup
    simp
  · intro n d h
    change (n, d) ∈ [("Point", Decl.struct_ [⟨"x", .prim u256⟩, ⟨"y", .prim u256⟩])] at h
    simp at h
    rcases h with ⟨rfl, rfl⟩
    intro f hf
    simp at hf
    rcases hf with rfl | rfl
    · exact WF.prim u256
    · exact WF.prim u256

theorem DeclEnvWF.exampleResource : DeclEnvWF exampleResource := by
  constructor
  · change (["Token"]).Nodup
    simp
  · intro n d h
    change (n, d) ∈ [("Token", Decl.resource (.prim u256))] at h
    simp at h
    rcases h with ⟨rfl, rfl⟩
    exact WF.prim u256

/-- A recursive nominal definition is accepted: the field type checks by head
    resolution, not by infinite unfolding. -/
def recursiveNodeEnv : DeclEnv :=
  [("Node", .struct_ [⟨"next", .struct_ "Node"⟩])]

theorem recursiveNode_resolves :
    recursiveNodeEnv.resolvesNominal (.struct_ "Node") :=
  ⟨[⟨"next", .struct_ "Node"⟩], rfl⟩

theorem DeclEnvWF.recursiveNode : DeclEnvWF recursiveNodeEnv := by
  constructor
  · change (["Node"]).Nodup
    simp
  · intro n d h
    change (n, d) ∈ [("Node", Decl.struct_ [⟨"next", .struct_ "Node"⟩])] at h
    simp at h
    rcases h with ⟨rfl, rfl⟩
    intro f hf
    simp at hf
    rcases hf with rfl
    exact WF.struct_ recursiveNode_resolves

def danglingFieldEnv : DeclEnv :=
  [("Bad", .struct_ [⟨"x", .struct_ "Missing"⟩])]

theorem DeclEnvWF.dangling_field_not_wf :
    ¬ DeclEnvWF danglingFieldEnv := by
  intro h
  have hDecl : DeclWF danglingFieldEnv (.struct_ [⟨"x", .struct_ "Missing"⟩]) :=
    h.2 "Bad" (.struct_ [⟨"x", .struct_ "Missing"⟩]) (by simp [danglingFieldEnv])
  have hField : WF danglingFieldEnv (.struct_ "Missing") := by
    exact hDecl ⟨"x", .struct_ "Missing"⟩ (by simp [FieldDeclWF])
  cases hField with
  | struct_ hRes =>
      obtain ⟨_, hLookup⟩ := hRes
      simp [DeclEnv.structFields?, DeclEnv.lookup, danglingFieldEnv] at hLookup

def danglingResourceCarrierEnv : DeclEnv :=
  [("Token", .resource (.struct_ "Missing"))]

theorem DeclEnvWF.dangling_resource_carrier_not_wf :
    ¬ DeclEnvWF danglingResourceCarrierEnv := by
  intro h
  have hDecl : DeclWF danglingResourceCarrierEnv (.resource (.struct_ "Missing")) :=
    h.2 "Token" (.resource (.struct_ "Missing")) (by simp [danglingResourceCarrierEnv])
  cases hDecl with
  | struct_ hRes =>
      obtain ⟨_, hLookup⟩ := hRes
      simp [DeclEnv.structFields?, DeclEnv.lookup, danglingResourceCarrierEnv] at hLookup

def duplicateNameEnv : DeclEnv :=
  [("Point", .struct_ []), ("Point", .struct_ [])]

theorem DeclEnvWF.duplicate_names_not_wf :
    ¬ DeclEnvWF duplicateNameEnv := by
  intro h
  have hNoDup := h.1
  simp [DeclEnv.NoDuplicateNames, duplicateNameEnv] at hNoDup

/-- A storage-slot handle exposes an opaque capability. -/
theorem storageSlot_exposes : Ty.exposesOpaqueCapability .storageSlot = true := rfl

/-- A `map` is NOT a valid runtime value, even though it is well-formed. -/
theorem map_not_runtime_value :
    Ty.validRuntimeValue (.map (.prim .address) (.prim u256)) = false := rfl

/-- A plain `u256` is a valid runtime value and crosses the ABI boundary. -/
theorem u256_runtime_ok : Ty.validRuntimeValue (.prim u256) = true := rfl
theorem u256_abi_ok : Ty.validAbiBoundary (.prim u256) = true := rfl

/-- A capability nested inside a tuple still exposes it (containment recurses). -/
theorem nested_capability_exposed :
    Ty.exposesOpaqueCapability (.tuple [.prim u256, .storageSlot]) = true := rfl

end Ora.Types
