/-
Shared decoders for compiler-emitted sync snapshots.

Every sync gate that consumes snapshot rows decodes the same wire format for
place identities (regions, place keys, free-variable ids, places). This module
is the single trusted decoder for that format: when the Zig emitters' format
moves, every gate moves together.

Trusted, hand-written. Decode failures return `none`; every consumer turns
`none` into a failed `decide` gate, so a format mismatch fails closed and
loudly. Char-list prefix matching is deliberate (transparent kernel reduction
under `decide`).
-/

import Ora.Obligation.Semantics

namespace Ora.SyncDecode

open Ora.Obligation

abbrev RawPlace := String × String × List String × List (String × String)

def decodeRegion : String → Option RegionRef
  | "none" => some .none
  | "storage" => some .storage
  | "memory" => some .memory
  | "transient" => some .transient
  | "calldata" => some .calldata
  | _ => none

def parseDecimalPrefixUntilColon : List Char → Nat → Bool → Option (Nat × List Char)
  | [], _, _ => none
  | ':' :: rest, acc, seen => if seen then some (acc, rest) else none
  | c :: rest, acc, _ =>
      match decimalDigit? c with
      | some digit => parseDecimalPrefixUntilColon rest (acc * 10 + digit) true
      | none => none

def stripPatternPrefix : List Char → Option (List Char)
  | 'p' :: 'a' :: 't' :: 't' :: 'e' :: 'r' :: 'n' :: ':' :: rest => some rest
  | _ => none

def decodeFreeVarIdString (value : String) : Option FreeVarId :=
  match value.toList with
  | 'f' :: 'i' :: 'l' :: 'e' :: ':' :: rest =>
      match parseDecimalPrefixUntilColon rest 0 false with
      | some (file_id, suffix) =>
          match stripPatternPrefix suffix with
          | some patternChars =>
              match parseDecimalNatAux patternChars 0 false with
              | some pattern_id => some { file_id := file_id, pattern_id := pattern_id }
              | none => none
          | none => none
      | none => none
  | _ => none

def decodeKey : String × String → Option PlaceKey
  | ("parameter", value) => (decodeFreeVarIdString value).map .parameter
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

end Ora.SyncDecode
