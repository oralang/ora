/-
Trusted sync checks for resource obligation enums.

`Ora/Generated/CompilerSnapshot.lean` emits the compiler-side enum names. This
module pins them against the Lean obligation manifest constructors, so adding or
renaming a resource operation/property on one side without the other turns the
formal sync gate red.
-/

import Ora.Obligation.Manifest
import Ora.Generated.CompilerSnapshot

namespace Ora.Resource.Sync

open Ora.Obligation Ora.Generated

def resourceOperationCompilerName : ResourceOperation → String
  | .move => "move"
  | .create => "create"
  | .destroy => "destroy"

def resourcePropertyCompilerName : ResourceProperty → String
  | .amountNonNegative => "amount_non_negative"
  | .sourceSufficient => "source_sufficient"
  | .destinationNoOverflow => "destination_no_overflow"
  | .samePlaceIdentity => "same_place_identity"
  | .conservation => "conservation"

def expectedResourceOperations : List String :=
  [.move, .create, .destroy].map resourceOperationCompilerName

def expectedResourceProperties : List String :=
  [.amountNonNegative,
   .sourceSufficient,
   .destinationNoOverflow,
   .samePlaceIdentity,
   .conservation].map resourcePropertyCompilerName

theorem resource_operations_match :
    compilerResourceOperations = expectedResourceOperations := by decide

theorem resource_properties_match :
    compilerResourceProperties = expectedResourceProperties := by decide

end Ora.Resource.Sync
