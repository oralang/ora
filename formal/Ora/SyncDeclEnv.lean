/-
`check-formal-declenv-sync` — the TRUSTED declaration-environment checks.

Proves, by `decide` (kernel-checked, NOT `native_decide`), that the compiler's
emitted declaration rows in `Ora/Generated/DeclEnvSnapshot.lean` match the spec's
curated model in `Ora/Spec/DeclEnvFacts.lean`. If the compiler's nominal kind
spellings drift, the `decide` fails and `lake build` goes red.
-/

import Ora.Spec.DeclEnvFacts
import Ora.Generated.DeclEnvSnapshot

namespace Ora.Sync

open Ora.Spec Ora.Generated

/-- The Lean `Decl` kinds map onto the compiler's nominal `TypeKind` tags. -/
theorem declkinds_match : compilerDeclKinds = expectedDeclKinds := by decide

/-- Each curated resource's carrier kind + spelling, as the compiler reads them
    off `carrier_type`, match the spec's `Ty.compilerKind` / `Ty.spelling?`. -/
theorem resource_carriers_match :
    compilerResourceCarriers = expectedResourceCarriers := by decide

end Ora.Sync
