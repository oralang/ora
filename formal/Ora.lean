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
import Ora.Types.RefinementValue
import Ora.Types.RefinementSubsumption
import Ora.Types.RefinementBridge
import Ora.Types.Decl
import Ora.Types.Internal
import Ora.Types.TypeEq
import Ora.Types.TypeEqLawful
import Ora.Types.Assignable
import Ora.Types.AssignableLawful
import Ora.Types.LocatedLawful
import Ora.Types.RefinementLocated
import Ora.Types.Effect
import Ora.Types.Typing
import Ora.Types.LocatedProjection
import Ora.Types.AssignableCoherence
import Ora.Types.RefinementTie
import Ora.Types.Projection
import Ora.Spec.Facts
import Ora.Generated.CompilerSnapshot
import Ora.Sync
import Ora.Spec.DeclEnvFacts
import Ora.Generated.DeclEnvSnapshot
import Ora.SyncDeclEnv
import Ora.Spec.TypeRelations
import Ora.Generated.CompilerTypeRelations
import Ora.TypeRelationsSync
import Ora.Dispatcher
import Ora.Spec.DispatcherFacts
import Ora.Generated.DispatcherStrategySnapshot
import Ora.DispatcherStrategySync
import Ora.Spec.SinoraBackendFacts
import Ora.Generated.SinoraBackendSnapshot
import Ora.SinoraBackendSync
import Ora.Obligation.Manifest
import Ora.Obligation.BitVec
import Ora.Obligation.Semantics
import Ora.Obligation.Theorems
