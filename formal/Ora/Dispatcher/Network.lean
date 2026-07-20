/-
Dispatcher-network topology checks, independent of planner recomputation.
-/

import Ora.Dispatcher.Decode

namespace Ora.DispatcherTableSync

abbrev RawIntent := Nat × String

structure RawNetworkSwitch where
  block : String
  defaultTarget : String
  cases : List (Nat × String)
  deriving Repr, DecidableEq

def intentSelector : RawIntent → Nat := Prod.fst
def intentLabel : RawIntent → String := Prod.snd

def dispatchNetwork : Nat → List RawNetworkSwitch → String → Nat → String
  | 0, _, block, _ => block
  | fuel + 1, switches, block, selector =>
      match switches.find? (fun sw => sw.block == block) with
      | none => block
      | some sw =>
          match sw.cases.find? (fun c => c.1 == selector) with
          | some c => c.2
          | none => dispatchNetwork fuel switches sw.defaultTarget selector

def networkNoDuplicateNat : List Nat → Bool
  | [] => true
  | value :: rest => !(rest.contains value) && networkNoDuplicateNat rest

def networkSelectorsDistinct (intents : List RawIntent) : Bool :=
  networkNoDuplicateNat (intents.map intentSelector)

def networkNoDuplicateString : List String → Bool
  | [] => true
  | value :: rest => !(rest.contains value) && networkNoDuplicateString rest

def networkBlocksDistinct (switches : List RawNetworkSwitch) : Bool :=
  networkNoDuplicateString (switches.map (·.block))

def networkCasesAuthorized
    (intents : List RawIntent) (switches : List RawNetworkSwitch) : Bool :=
  switches.all fun sw =>
    sw.cases.all fun c => intents.contains c

def networkIntentsPresentExactlyOnce
    (intents : List RawIntent) (switches : List RawNetworkSwitch) : Bool :=
  intents.all fun intent =>
    decide ((switches.foldl (fun count sw =>
      count + sw.cases.count intent) 0) = 1)

def networkDefaultsKnown (switches : List RawNetworkSwitch) : Bool :=
  switches.all fun sw =>
    sw.defaultTarget == "revert_error" ||
      switches.any (fun target => target.block == sw.defaultTarget)

def networkKnownSelectorsReachIntent
    (intents : List RawIntent)
    (switches : List RawNetworkSwitch)
    (entry : String) : Bool :=
  intents.all fun intent =>
    dispatchNetwork (switches.length + 1) switches entry (intentSelector intent) ==
      intentLabel intent

def networkDefaultReachesRevert
    (switches : List RawNetworkSwitch) (entry : String) : Bool :=
  dispatchNetwork (switches.length + 1) switches entry 4294967296 ==
    "revert_error"

def dispatcherNetworkMatches
    (intents : List RawIntent)
    (switches : List RawNetworkSwitch)
    (entry : String) : Bool :=
  networkSelectorsDistinct intents &&
    networkBlocksDistinct switches &&
    networkCasesAuthorized intents switches &&
    networkIntentsPresentExactlyOnce intents switches &&
    networkDefaultsKnown switches &&
    networkKnownSelectorsReachIntent intents switches entry &&
    networkDefaultReachesRevert switches entry

end Ora.DispatcherTableSync
