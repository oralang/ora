/-
Trusted sync check for the compiler-emitted storage path-disjointness fixture.

`Ora/Generated/StorageDisjointnessSnapshot.lean` is data-only: primitive rows
emitted from the Zig fixture table. This module decodes those rows into
`PlaceRef` values and proves the Lean relation agrees with every expected
verdict by kernel `decide`.
-/

import Ora.Obligation.Semantics
import Ora.Generated.StorageDisjointnessSnapshot
import Ora.SyncDecode

namespace Ora.StorageDisjointnessSync

open Ora.Obligation Ora.Generated Ora.SyncDecode

abbrev RawRow := String × RawPlace × RawPlace × Bool

def rowMatches : RawRow → Bool
  | (_, lhsRaw, rhsRaw, expected) =>
      match decodePlace lhsRaw, decodePlace rhsRaw with
      | some lhs, some rhs => placeDefinitelyDisjoint lhs rhs == expected
      | _, _ => false

theorem storage_disjointness_fixture_matches :
    storageDisjointnessRows.all rowMatches = true := by decide

end Ora.StorageDisjointnessSync
