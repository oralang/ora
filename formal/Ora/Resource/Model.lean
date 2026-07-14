/-
Core resource-state model.

This module models one resource domain at a time. Domain separation is by
construction: different resource domains are different `State` values, so no
cross-domain operation is expressible here.
-/

import Ora.Obligation.BitVec

namespace Ora.Resource

abbrev Carrier := Ora.Obligation.U256

abbrev State (P : Type) := P → Carrier

def amount (s : State P) (p : P) : Carrier :=
  s p

def amountNonNegative (x : Carrier) : Prop :=
  0 ≤ x.toNat

def destinationNoOverflow (s : State P) (p : P) (x : Carrier) : Prop :=
  (s p).toNat + x.toNat < 2^256

def sourceSufficient (s : State P) (p : P) (x : Carrier) : Prop :=
  x.toNat ≤ (s p).toNat

def moveSourceSufficient (s : State P) (source _destination : P) (x : Carrier) : Prop :=
  sourceSufficient s source x

def moveDestinationNoOverflow (s : State P) (source destination : P) (x : Carrier) : Prop :=
  source = destination ∨ destinationNoOverflow s destination x

def moveGuard (s : State P) (source destination : P) (x : Carrier) : Prop :=
  moveSourceSufficient s source destination x ∧
    moveDestinationNoOverflow s source destination x

def update [DecidableEq P] (s : State P) (p : P) (value : Carrier) : State P :=
  fun candidate => if candidate = p then value else s candidate

def create [DecidableEq P] (s : State P) (p : P) (x : Carrier) : State P :=
  update s p ((s p).add x)

def destroy [DecidableEq P] (s : State P) (p : P) (x : Carrier) : State P :=
  update s p ((s p).sub x)

def move [DecidableEq P] (s : State P) (source destination : P) (x : Carrier) : State P :=
  if source = destination then
    s
  else
    update (update s source ((s source).sub x)) destination ((s destination).add x)

end Ora.Resource
