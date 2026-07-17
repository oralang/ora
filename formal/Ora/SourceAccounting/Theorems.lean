import Ora.SourceAccounting.Decision

namespace Ora.SourceAccounting

theorem accountsForAllB_eq_true_iff
    (mode : CompilationMode) (m : Manifest) :
    accountsForAllB mode m = true ↔ AccountsForAll mode m := by
  simp only [accountsForAllB, AccountsForAll, SyntaxConserved, UsesComplete,
    EvidenceClosed, TraceValid, HandledOnce, PolicyCompatible,
    accountsForAllBagB, syntaxConservedB, usesCompleteB, evidenceClosedB,
    traceValidB, handledOnceB, policyCompatibleB, Bool.and_eq_true]

private theorem decideBag_accepted_iff
    (mode : CompilationMode) (m : BagManifest) :
    decideBag mode m = .accepted ↔
      wellFormedBagB m = true ∧ accountsForAllBagB mode m = true := by
  cases hw : wellFormedBagB m <;>
    cases ha : accountsForAllBagB mode m <;>
      simp [decideBag, hw, ha]

theorem accepted_iff_well_formed_and_accounted
    (mode : CompilationMode) (m : Manifest) :
    decide mode m = .accepted ↔ WellFormed m ∧ AccountsForAll mode m := by
  change decideBag mode m.bag = .accepted ↔
    wellFormedBagB m.bag = true ∧ AccountsForAll mode m
  rw [decideBag_accepted_iff]
  exact and_congr Iff.rfl (accountsForAllB_eq_true_iff mode m)

theorem accepted_implies_syntax_conserved
    {mode : CompilationMode} {m : Manifest} (h : decide mode m = .accepted) :
    SyntaxConserved m :=
  (accepted_iff_well_formed_and_accounted mode m).mp h |>.2.1

theorem accepted_implies_expected_uses_complete
    {mode : CompilationMode} {m : Manifest} (h : decide mode m = .accepted) :
    UsesComplete m :=
  (accepted_iff_well_formed_and_accounted mode m).mp h |>.2.2.1

theorem accepted_implies_no_orphan_evidence
    {mode : CompilationMode} {m : Manifest} (h : decide mode m = .accepted) :
    EvidenceClosed m :=
  (accepted_iff_well_formed_and_accounted mode m).mp h |>.2.2.2.1

theorem accepted_implies_trace_valid
    {mode : CompilationMode} {m : Manifest} (h : decide mode m = .accepted) :
    TraceValid m :=
  (accepted_iff_well_formed_and_accounted mode m).mp h |>.2.2.2.2.1

theorem accepted_implies_every_use_has_one_handling
    {mode : CompilationMode} {m : Manifest} (h : decide mode m = .accepted) :
    HandledOnce m :=
  (accepted_iff_well_formed_and_accounted mode m).mp h |>.2.2.2.2.2.1

theorem accepted_implies_policy_compatible
    {mode : CompilationMode} {m : Manifest} (h : decide mode m = .accepted) :
    PolicyCompatible mode m :=
  (accepted_iff_well_formed_and_accounted mode m).mp h |>.2.2.2.2.2.2

theorem accepted_implies_no_abandoned_fold_evidence
    {mode : CompilationMode} {m : Manifest} (h : decide mode m = .accepted) :
    m.folds.all (fun fold =>
      if fold.disposition == .abandoned then foldHasNoAuthority m fold else true) = true := by
  have ht := accepted_implies_trace_valid h
  change (noAbandonedFoldAuthorityBagB m.bag &&
    (m.bag.folds.all (foldTraceValidBag m.bag) &&
      (m.bag.predicateEvents.all (fun predicate => m.bag.folds.any fun fold =>
        fold.id == predicate.foldId && fold.events.any (fun event => traceEventMatchesPredicate event predicate)) &&
       m.bag.handlings.all (concreteHandlingTraceValidBag m.bag)))) = true at ht
  simp only [Bool.and_eq_true] at ht
  exact ht.1

theorem accepted_concrete_invariant_has_nonzero_checks
    {mode : CompilationMode} {m : Manifest} (h : decide mode m = .accepted) :
    m.handlings.all (concreteHandlingTraceValid m) = true := by
  have ht := accepted_implies_trace_valid h
  change (noAbandonedFoldAuthorityBagB m.bag &&
    (m.bag.folds.all (foldTraceValidBag m.bag) &&
      (m.bag.predicateEvents.all (fun predicate => m.bag.folds.any fun fold =>
        fold.id == predicate.foldId && fold.events.any (fun event => traceEventMatchesPredicate event predicate)) &&
       m.bag.handlings.all (concreteHandlingTraceValidBag m.bag)))) = true at ht
  simp only [Bool.and_eq_true] at ht
  exact ht.2.2.2

structure RowsPermute (left right : Manifest) : Prop where
  declaredSites : left.declaredSites.Perm right.declaredSites
  typedSites : left.typedSites.Perm right.typedSites
  generatedFactDerivations : left.generatedFactDerivations.Perm right.generatedFactDerivations
  ownerTemplates : left.ownerTemplates.Perm right.ownerTemplates
  expansions : left.expansions.Perm right.expansions
  uses : left.uses.Perm right.uses
  controlNodes : left.controlNodes.Perm right.controlNodes
  controlEdges : left.controlEdges.Perm right.controlEdges
  folds : left.folds.Perm right.folds
  predicateEvents : left.predicateEvents.Perm right.predicateEvents
  obligations : left.obligations.Perm right.obligations
  assumptions : left.assumptions.Perm right.assumptions
  queries : left.queries.Perm right.queries
  runtimeChecks : left.runtimeChecks.Perm right.runtimeChecks
  frameResults : left.frameResults.Perm right.frameResults
  stateEffects : left.stateEffects.Perm right.stateEffects
  handlings : left.handlings.Perm right.handlings

theorem RowsPermute.bag_eq {left right : Manifest} (h : RowsPermute left right) :
    left.bag = right.bag := by
  simp only [Manifest.bag, BagManifest.mk.injEq]
  exact ⟨
    RowBag.ofList_eq_of_perm h.declaredSites,
    RowBag.ofList_eq_of_perm h.typedSites,
    RowBag.ofList_eq_of_perm h.generatedFactDerivations,
    RowBag.ofList_eq_of_perm h.ownerTemplates,
    RowBag.ofList_eq_of_perm h.expansions,
    RowBag.ofList_eq_of_perm h.uses,
    RowBag.ofList_eq_of_perm h.controlNodes,
    RowBag.ofList_eq_of_perm h.controlEdges,
    RowBag.ofList_eq_of_perm h.folds,
    RowBag.ofList_eq_of_perm h.predicateEvents,
    RowBag.ofList_eq_of_perm h.obligations,
    RowBag.ofList_eq_of_perm h.assumptions,
    RowBag.ofList_eq_of_perm h.queries,
    RowBag.ofList_eq_of_perm h.runtimeChecks,
    RowBag.ofList_eq_of_perm h.frameResults,
    RowBag.ofList_eq_of_perm h.stateEffects,
    RowBag.ofList_eq_of_perm h.handlings
  ⟩

theorem decision_invariant_under_row_permutation
    {mode : CompilationMode} {left right : Manifest}
    (h : RowsPermute left right) :
    decide mode left = decide mode right := by
  simp only [decide]
  rw [h.bag_eq]

end Ora.SourceAccounting
