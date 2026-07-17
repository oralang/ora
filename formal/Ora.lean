/-
Ora formal development — root module.

`lake build` builds the `Ora` library, whose root is this file. It re-exports
the library's modules so a single `import Ora` pulls in everything.

Current scope: type-model lemmas, compiler snapshot syncs, obligation
denotation/agreement checks, storage-disjointness/totality syncs, and resource
model theorems.

Honest boundary: importing `Ora` does not import a compiler-correctness theorem
or a full language soundness theorem. The typing relation in `Types/Typing.lean`
is a core skeleton; operational semantics, preservation/progress, and
compiler-to-core correctness are future milestones.
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
import Ora.DispatcherPlannerSpec
import Ora.SinoraPlanner
import Ora.Spec.DispatcherFacts
import Ora.Generated.DispatcherStrategySnapshot
import Ora.DispatcherStrategySync
import Ora.Generated.DispatcherTableSnapshot
import Ora.DispatcherTableSync
import Ora.Spec.SinoraBackendFacts
import Ora.Generated.SinoraBackendSnapshot
import Ora.SinoraBackendSync
import Ora.Obligation.Manifest
import Ora.Obligation.BitVec
import Ora.Obligation.Semantics
import Ora.Obligation.Theorems
import Ora.Obligation.Agreement
import Ora.Resource.Model
import Ora.Resource.Theorems
import Ora.Resource.Sync
import Ora.Generated.StorageDisjointnessSnapshot
import Ora.SyncDecode
import Ora.StorageDisjointnessSync
import Ora.Generated.ObligationTotalitySnapshot
import Ora.ObligationTotalitySync
import Ora.SourceAccounting.Manifest
import Ora.SourceAccounting.Policy
import Ora.SourceAccounting.Decision
import Ora.SourceAccounting.Theorems
import Ora.SourceAccountingSync
