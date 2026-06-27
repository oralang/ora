/-
`check-formal-type-relations-sync` — trusted checks for compiler relation rows.

The compiler emits data-only rows in `Ora/Generated/CompilerTypeRelations.lean`.
The spec owns typed cases in `Ora/Spec/TypeRelations.lean` and projects them
through Lean's `Ty.beq`, `Ty.assignable`, `Located.beq`, and
`Located.assignable`. These theorems prove the compiler rows agree with the
Lean relations over the curated matrix.
-/

import Ora.Generated.CompilerTypeRelations
import Ora.Spec.TypeRelations

namespace Ora.TypeRelationsSync

open Ora.Generated Ora.Spec.TypeRelations

theorem type_eql_rows_match :
    compilerTypeEqlRows = expectedTypeEqlRows := by decide

theorem types_assignable_rows_match :
    compilerTypesAssignableRows = expectedTypesAssignableRows := by decide

theorem located_type_eql_rows_match :
    compilerLocatedTypeEqlRows = expectedLocatedTypeEqlRows := by decide

theorem located_assignable_rows_match :
    compilerLocatedAssignableRows = expectedLocatedAssignableRows := by decide

end Ora.TypeRelationsSync
