import Ora.Loop.CallTests
import Ora.Storage.LifecycleEntry

namespace Ora.StorageEntryNegatives
open Ora.Obligation Ora.Loop Ora.StorageLifecycle Ora.Loop.CallRowTests

theorem missing_requires_cannot_witness :
    ¬ SourceWitness entryManifest { entryRow with assumptionIds := [] }
      claims semantics events :=
  malformed_entry_cannot_witness (by decide)

theorem duplicate_requires_cannot_witness :
    ¬ SourceWitness entryManifest { entryRow with assumptionIds := [1, 1] }
      claims semantics events :=
  malformed_entry_cannot_witness (by decide)

theorem foreign_owner_cannot_witness :
    ¬ SourceWitness entryManifest { entryRow with contractName := some "D" }
      claims semantics events :=
  malformed_entry_cannot_witness (by decide)

theorem missing_contract_fact_cannot_witness :
    ¬ SourceWitness entryManifest { entryRow with entryInvariantIds := [3] }
      claims semantics events :=
  malformed_entry_cannot_witness (by decide)

theorem entry_storage_fact_cannot_be_rebound :
    entryRow.entryFactsHold entryManifest (entryInput true true) ∧
      ¬ entryRow.entryFactsHold entryManifest (entryInput true false) := by
  constructor
  · unfold FunctionSummaryRow.entryFactsHold
    change ∀ formula : FormulaRef, formula ∈ [.term 0, .term 6, .term 7] → _
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil,
      forall_eq_or_imp, forall_eq, false_implies, implies_true, and_true]
    exact ⟨rfl, rfl, rfl⟩
  · intro facts
    have impossible : false = true := facts (.term 7) (by simp)
    cases impossible

def falseManifest : Manifest := {
  terms := [.boolLit false]
  assumptions := [{ id := 1, owner := "entry", kind := .requires, formula := some (.term 0) }] }
def falseRow : FunctionSummaryRow := {
  id := 1, owner := "entry", structuralKey := "test-entry", assumptionIds := [1] }

theorem contradictory_source_requires_no_witness :
    ¬ SourceWitness falseManifest falseRow claims semantics events := by
  apply contradictory_entry_cannot_witness
  intro input facts
  cases entry : falseRow.callSignature.entry? input with
  | none => simp [FunctionSummaryRow.entryFactsHold, entry] at facts
  | some state =>
      have noLocals : state.loop.locals = [] := by
        unfold FunctionSummaryRow.callSignature CallSignature.entry? at entry
        split at entry
        · exact congrArg (fun s => s.loop.locals) (Option.some.inj entry).symm
        · simp_all
      change (match some [FormulaRef.term 0], falseRow.callSignature.entry? input with
        | some formulas, some initial => ∀ formula ∈ formulas,
          ((directStorageContext falseManifest (bindStorageReads Env.empty initial.reads) []).evalFormula? formula initial).getD False
        | _, _ => False) at facts
      rw [entry] at facts
      have impossible := facts (.term 0) (by simp)
      simp [StorageFormulaContext.evalFormula?, StorageFormulaContext.stateEnv?,
        directStorageContext, noLocals, bindState?, StorageFormulaContext.formulaStorageReady,
        formulaDenotes?, StorageFormulaContext.termStorageReady, falseManifest,
        denoteFormula?, denoteValue?] at impossible

def strictSemantics : TimelineSemantics (CallFormula AdapterFormula FormulaRef AdapterChild)
    StorageLocationExpr FormulaRef AdapterChild := { (adapterSemantics {}
    (directStorageContext falseManifest Env.empty [])).withCallChecks with
  evalFormula := fun _ _ => some False }
def falseCheckpoint : AdapterEvent := .assume (.ordinary (.inl (.term 0)))

theorem retained_false_check_cannot_witness :
    ¬ SourceWitness manifest row claims strictSemantics [falseCheckpoint] := by
  apply source_witness_requires_strict_trace
  intro input state result _ _ _ execution
  cases execution with
  | cons first _ =>
      cases first with
      | assume evaluated holds =>
          change some False = some _ at evaluated
          exact (Option.some.inj evaluated).symm ▸ holds

#print axioms missing_requires_cannot_witness
#print axioms duplicate_requires_cannot_witness
#print axioms foreign_owner_cannot_witness
#print axioms missing_contract_fact_cannot_witness
#print axioms entry_storage_fact_cannot_be_rebound
#print axioms contradictory_source_requires_no_witness
#print axioms retained_false_check_cannot_witness
end Ora.StorageEntryNegatives
