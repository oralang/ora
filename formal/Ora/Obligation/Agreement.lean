/-
Runtime agreement checks for userland Lean proof targets.

The compiler emits `Row` data beside each proof-check scratch module. This
module recomputes the manifest facts Lean can check: row ids resolve, premise
and target denotations exist, and every accepted target is covered by exactly
one row. Z3 result flags and SMT hashes remain compiler attestations.
-/

import Ora.Obligation.Theorems

namespace Ora.Obligation.Agreement

open Ora.Obligation

structure Row where
  queryId : Nat
  assumptionIds : List Nat := []
  obligationIds : List Nat := []
  z3RowMatched : Bool := false
  /-- True only when the compiler matched this proof target to a clean Z3
  UNKNOWN or UNSAT row whose result is eligible for Lean agreement. Independent
  compiler-owned proof targets are deliberately omitted from agreement rows. -/
  z3ProofCompatible : Bool := false
  constraintCount : Nat := 0
  smtlibHash : String := ""
  zigSemanticSupported : Bool := false
  usesLoopSummary : Bool := false
  loopSummarySupported : Bool := false
  deriving Repr

def valueForTy? : Option TyRef → Value
  | some ty =>
      match ty.integerShape? with
      | some shape => .integer (Ora.Integer.Value.ofNat shape 0)
      | none => .bool true
  | none => .bool false

def bindFreeVarsFromTerm (env : Env) : Term → Env
  | .variable (.free free) => env.setFree free.id (valueForTy? free.ty)
  | _ => env

/--
Witness environment for denotability checks.

Correctness invariant: `denoteFormula?`/`obligationDenotesInEnv?` return `none`
only for structural reasons (unknown ids, unsupported shapes, type mismatches,
unbound free variables) — never because of the *values* an environment assigns.
So "denotes in one total environment" is equivalent to "denotes in every
environment", PROVIDED the environment binds every free variable that can
appear. The fold below provides exactly that totality: the manifest's terms are
a flat arena, so any free variable referenced anywhere appears as a term row
and gets bound. The concrete values (true / zero) are arbitrary witnesses;
within the supported u256/bool fragment they are also well-typed. This env must
never be used to prove value-level facts — query semantics remain quantified
over all environments.
-/
def canonicalEnv (manifest : Manifest) : Env :=
  { manifest.terms.foldl bindFreeVarsFromTerm Env.empty with
    placeValue := fun _ => .integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)
    result := some (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)) }

def findAssumptionById (manifest : Manifest) (id : Nat) : Option AssumptionRow :=
  manifest.assumptions.find? (fun row => row.id == id)

def findObligationById (manifest : Manifest) (id : Nat) : Option ObligationRow :=
  manifest.obligations.find? (fun row => row.id == id)

def collectAssumptionsByIds (manifest : Manifest) : List Nat → Option (List AssumptionRow)
  | [] => some []
  | id :: rest =>
      match findAssumptionById manifest id, collectAssumptionsByIds manifest rest with
      | some row, some rows => some (row :: rows)
      | _, _ => none

def collectObligationsByIds (manifest : Manifest) : List Nat → Option (List ObligationRow)
  | [] => some []
  | id :: rest =>
      match findObligationById manifest id, collectObligationsByIds manifest rest with
      | some row, some rows => some (row :: rows)
      | _, _ => none

def assumptionsFullyDenotable? (manifest : Manifest) (rows : List AssumptionRow) : Bool :=
  (assumptionsDenoteInEnv? manifest (canonicalEnv manifest) rows).isSome

def obligationFullyDenotable? (manifest : Manifest) (row : ObligationRow) : Bool :=
  (obligationDenotesInEnv? manifest (canonicalEnv manifest) row).isSome

def obligationsFullyDenotable? (manifest : Manifest) (rows : List ObligationRow) : Bool :=
  rows.all (obligationFullyDenotable? manifest)

def rowMatches (manifest : Manifest) (row : Row) : Bool :=
  match collectAssumptionsByIds manifest row.assumptionIds with
  | some assumptions =>
      row.z3RowMatched &&
        row.z3ProofCompatible &&
        row.zigSemanticSupported &&
        manifest.wf &&
        assumptionsFullyDenotable? manifest assumptions &&
        if row.usesLoopSummary then
          row.loopSummarySupported
        else
          match collectObligationsByIds manifest row.obligationIds with
          | some obligations => obligationsFullyDenotable? manifest obligations
          | none => false
  | none => false

def countRowsFor (id : Nat) : List Row → Nat
  | [] => 0
  | row :: rest =>
      (if row.queryId == id then 1 else 0) + countRowsFor id rest

def targetCovered (id : Nat) (rows : List Row) : Bool :=
  countRowsFor id rows == 1

def targetsCovered (targetIds : List Nat) (rows : List Row) : Bool :=
  targetIds.all (fun id => targetCovered id rows)

def rowsMatch (manifest : Manifest) (rows : List Row) : Bool :=
  rows.all (rowMatches manifest)

end Ora.Obligation.Agreement
