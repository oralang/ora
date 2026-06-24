/-
Ora formal development — root module.

`lake build` builds the `Ora` library, whose root is this file. It re-exports
the library's modules so a single `import Ora` pulls in everything.

Phase 1 starts from the TYPE UNIVERSE (regions, primitives, the type lattice,
well-formedness), grounded in the compiler's `src/types/` — `semantic.zig`,
`builtin.zig`, `region.zig`. Syntax / typing / dynamics layer on top later.
-/

import Ora.Types.Region
import Ora.Types.Prim
import Ora.Types.Ty
import Ora.Types.WF
import Ora.Types.Refinement
import Ora.Spec.Facts
import Ora.Generated.CompilerSnapshot
import Ora.Sync
