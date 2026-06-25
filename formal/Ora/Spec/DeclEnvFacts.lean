/-
Spec-side FACT INTERFACE for `check-formal-declenv-sync`.

Lean owns the typed declaration model (`Ora.Types.Decl`). This file projects a
CURATED matrix of declarations into the same data shape the compiler emits in
`Ora/Generated/DeclEnvSnapshot.lean`, so `Ora/SyncDeclEnv.lean` can prove — by
`decide`, in the kernel — that the Lean `Decl` kinds map onto the compiler's
nominal `TypeKind`s (and a rename like `resource_domain` is caught).

First slice: the nominal-KIND correspondence. Richer rows (fields / variants /
member spellings) layer on later over the same curated matrix.
-/

import Ora.Types.Decl

namespace Ora.Spec

open Ora.Types

/-- The compiler `TypeKind` tag each `Decl` kind maps to (the nominal spelling
    the emitter reads via `@tagName(Type.kind())`). -/
def Decl.compilerKind : Decl → String
  | .struct_ _  => "struct_"
  | .enum_ _    => "enum_"
  | .bitfield _ => "bitfield"
  | .contract _ => "contract"
  | .resource _ => "resource_domain"

/-- The CURATED declaration matrix — a small hand-picked set, NOT the whole
    compiler universe. Mirrors the declarations the emitter builds. -/
def curatedDeclEnv : DeclEnv :=
  [ ("Point", .struct_ [⟨"x", .prim u256⟩, ⟨"y", .prim u256⟩]),
    ("Color", .enum_ [⟨"Red", .none, none⟩, ⟨"Green", .none, none⟩, ⟨"Blue", .none, none⟩]),
    ("Flags", .bitfield [⟨"a", .prim .bool, none, none⟩]),
    ("Vault", .contract ["balance", "owner"]),
    ("Token", .resource (.prim u256)) ]

/-- (declaration name, expected compiler `TypeKind` tag), in the curated order. -/
def expectedDeclKinds : List (String × String) :=
  curatedDeclEnv.map (fun p => (p.1, Decl.compilerKind p.2))

end Ora.Spec
