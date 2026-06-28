/-
Ora type system — the declaration context.

`Ty`'s nominal kinds (`struct_`, `enum_`, `bitfield`, `contract`) carry only a
`Name`; the compiler resolves that name to a definition in a declaration table.
This file is that table: a `DeclEnv` mapping names to `Decl`s, plus the lookups
that resolve a nominal name to its fields / variants / members.

This is the keystone the later layers need: well-formedness must check that every
nominal name RESOLVES (and to the right kind), and typing must look up struct
fields and enum variants. Neither can happen without this.

Grounded in `src/ast/nodes.zig`: `StructField{name, type}`,
`BitfieldField{name, type, offset?, width?}`, `EnumVariant{name, payload, value?}`,
`EnumVariantPayload{none | positional | named}`, `ContractItem{name, members}`.

SCOPE: `contract` is modeled minimally (member names only) for now — full
contract members (function signatures, storage variables) are deferred to their
own layer; only name resolution is needed here.
-/

import Ora.Types.Ty

namespace Ora.Types

/-- A struct field: a name and its type (`ast/nodes.zig:StructField`). -/
structure FieldDecl where
  name : Name
  ty   : Ty

/-- A bitfield member: name, type, and optional explicit bit offset/width
    (`ast/nodes.zig:BitfieldField` — `offset`/`width` are `?u32`). -/
structure BitfieldFieldDecl where
  name   : Name
  ty     : Ty
  offset : Option Nat
  width  : Option Nat

/-- An enum variant's payload — Ora enums are sum types
    (`ast/nodes.zig:EnumVariantPayload`). -/
inductive VariantPayload where
  | none
  | positional : List Ty → VariantPayload
  | named      : List FieldDecl → VariantPayload

/-- An enum variant: name, payload, and an optional explicit discriminant value
    (`ast/nodes.zig:EnumVariant` — `value` is `?Expr`; `none` ⇒ auto-ordinal). -/
structure EnumVariantDecl where
  name    : Name
  payload : VariantPayload
  value   : Option Nat

/--
A nominal type definition, resolved by name. The constructors match the nominal
`Ty` kinds (`struct_`/`enum_`/`bitfield`/`contract`/`resourceDomain`).
For resources, resolution includes the declared carrier type, not just the
resource name.
-/
inductive Decl where
  | struct_  : List FieldDecl → Decl
  | enum_    : List EnumVariantDecl → Decl
  | bitfield : List BitfieldFieldDecl → Decl
  | contract : List Name → Decl
  /-- A resource domain (`semantic.zig:ResourceDomainType`): its carrier type. -/
  | resource : Ty → Decl

/-- The declaration environment: nominal names ⇒ their definitions. -/
abbrev DeclEnv := List (Name × Decl)

/-- Resolve a name to its declaration (first binding wins). -/
def DeclEnv.lookup (env : DeclEnv) (n : Name) : Option Decl :=
  (env.find? (fun p => p.1 == n)).map (·.2)

/-! ## Kind-aware resolvers

    These resolve a name AND require the matching kind, so a `struct_`-named
    type cannot resolve to an enum, etc. -/

def DeclEnv.structFields? (env : DeclEnv) (n : Name) : Option (List FieldDecl) :=
  match env.lookup n with
  | some (.struct_ fs) => some fs
  | _                  => none

def DeclEnv.enumVariants? (env : DeclEnv) (n : Name) : Option (List EnumVariantDecl) :=
  match env.lookup n with
  | some (.enum_ vs) => some vs
  | _                => none

def DeclEnv.bitfieldFields? (env : DeclEnv) (n : Name) : Option (List BitfieldFieldDecl) :=
  match env.lookup n with
  | some (.bitfield fs) => some fs
  | _                   => none

def DeclEnv.contractMembers? (env : DeclEnv) (n : Name) : Option (List Name) :=
  match env.lookup n with
  | some (.contract ms) => some ms
  | _                   => none

def DeclEnv.resourceCarrier? (env : DeclEnv) (n : Name) : Option Ty :=
  match env.lookup n with
  | some (.resource c) => some c
  | _                  => none

/-! ## The WF bridge

    `resolvesNominal env t` holds when `t`'s head nominal name (if it has one)
    resolves to a declaration of the matching kind. Non-nominal types carry no
    nominal name, so they satisfy it vacuously. Well-formedness will recurse this
    over a type's sub-terms. -/

def DeclEnv.resolvesNominal (env : DeclEnv) : Ty → Prop
  | .struct_ n           => ∃ fs, env.structFields? n = some fs
  | .enum_ n             => ∃ vs, env.enumVariants? n = some vs
  | .bitfield n          => ∃ fs, env.bitfieldFields? n = some fs
  | .contract n          => ∃ ms, env.contractMembers? n = some ms
  | .resourceDomain n c  => env.resourceCarrier? n = some c
  | _                    => True

/-- A non-nominal head (here, a primitive) resolves vacuously. -/
theorem resolvesNominal_prim (env : DeclEnv) (p : PrimTy) :
    env.resolvesNominal (.prim p) := trivial

/-! ## Sanity -/

/-- `struct Point { x: u256, y: u256 }` in an environment. -/
def examplePoint : DeclEnv :=
  [("Point", .struct_ [⟨"x", .prim u256⟩, ⟨"y", .prim u256⟩])]

/-- A declared struct name resolves to a struct (and to its fields). -/
theorem point_resolves : examplePoint.resolvesNominal (.struct_ "Point") :=
  ⟨[⟨"x", .prim u256⟩, ⟨"y", .prim u256⟩], rfl⟩

/-- A name with no binding does not resolve. -/
theorem missing_does_not_resolve :
    (examplePoint.structFields? "Nope") = none := rfl

/-- Kind mismatch: `Point` is a struct, so it does not resolve as an enum. -/
theorem point_not_enum :
    (examplePoint.enumVariants? "Point") = none := rfl

/-- `resource Token` with carrier `u256` in an environment. -/
def exampleResource : DeclEnv :=
  [("Token", .resource (.prim u256))]

/-- A declared resource-domain name resolves. -/
theorem resource_resolves :
    exampleResource.resolvesNominal (.resourceDomain "Token" (.prim u256)) :=
  rfl

theorem resource_carrier_mismatch_does_not_resolve :
    ¬ exampleResource.resolvesNominal (.resourceDomain "Token" (.prim .bool)) := by
  intro h
  cases h

end Ora.Types
