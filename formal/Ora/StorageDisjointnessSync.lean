/-
Trusted sync check for the compiler-emitted storage path-disjointness fixture.

`Ora/Generated/StorageDisjointnessSnapshot.lean` is data-only: primitive rows
emitted from the Zig fixture table. This module decodes those rows into
`PlaceRef` values and proves the Lean relation agrees with every expected
verdict by kernel `decide`.
-/

import Ora.Obligation.Semantics
import Ora.Generated.StorageDisjointnessSnapshot

namespace Ora.StorageDisjointnessSync

open Ora.Obligation Ora.Generated

abbrev RawPlace := String × String × List String × List (String × String)
abbrev RawRow := String × RawPlace × RawPlace × Bool

def decodeRegion : String → Option RegionRef
  | "none" => some .none
  | "storage" => some .storage
  | "memory" => some .memory
  | "transient" => some .transient
  | "calldata" => some .calldata
  | _ => none

def decodeKey : String × String → Option PlaceKey
  | ("parameter", value) => (parseDecimalNat? value).map .parameter
  | ("comptime_parameter", value) => (parseDecimalNat? value).map .comptimeParameter
  | ("comptime_range_parameter", value) => (parseDecimalNat? value).map .comptimeRangeParameter
  | ("constant", value) => some (.constant value)
  | ("msg_sender", _) => some .msgSender
  | ("tx_origin", _) => some .txOrigin
  | ("unknown", _) => some .unknown
  | _ => none

def decodeKeys : List (String × String) → Option (List PlaceKey)
  | [] => some []
  | raw :: rest =>
      match decodeKey raw, decodeKeys rest with
      | some key, some keys => some (key :: keys)
      | _, _ => none

def decodePlace : RawPlace → Option PlaceRef
  | (root, regionName, fields, rawKeys) =>
      match decodeRegion regionName, decodeKeys rawKeys with
      | some region, some keys => some { root := root, region := region, fields := fields, keys := keys }
      | _, _ => none

def rowMatches : RawRow → Bool
  | (_, lhsRaw, rhsRaw, expected) =>
      match decodePlace lhsRaw, decodePlace rhsRaw with
      | some lhs, some rhs => placeDefinitelyDisjoint lhs rhs == expected
      | _, _ => false

theorem storage_disjointness_fixture_matches :
    storageDisjointnessRows.all rowMatches = true := by decide

end Ora.StorageDisjointnessSync
