/-
Trusted sync check for the compiler-emitted obligation-totality fixture.

`Ora/Generated/ObligationTotalitySnapshot.lean` is data-only: primitive rows
emitted from the Zig supportability fixture matrix. This module decodes those
rows into obligation manifests and proves that every row the compiler claims is
Lean-supported has a total Lean denotation under a canonical well-typed
environment.

This is a fixture-matrix guard, not the final Z3↔Lean agreement bridge. Its job
is to keep the current supported grammar honest while the fragment grows.
-/

import Ora.Obligation.Semantics
import Ora.Generated.ObligationTotalitySnapshot
import Ora.SyncDecode

namespace Ora.ObligationTotalitySync

open Ora.Obligation Ora.Generated Ora.SyncDecode

abbrev RawTerm :=
  String × Bool × String × String × String × Nat × Nat × Option Nat × List Nat × RawPlace
abbrev RawAssumption := Nat × String × Option Nat
abbrev RawEvidence :=
  String × Nat × Nat × Nat × Nat × Nat × RawPlace × RawPlace × Nat
abbrev RawEffect := String × List RawPlace × List RawPlace × List RawEvidence
abbrev RawResource :=
  String × String × Option RawPlace × Option RawPlace × Option Nat × String
abbrev RawRow :=
  String × List RawTerm × List RawAssumption × Bool × Bool × String × Nat ×
    RawEffect × RawResource

def parseCompilerTypeId? (value : String) : Option Nat :=
  match value.toList with
  | 'c' :: 'o' :: 'm' :: 'p' :: 'i' :: 'l' :: 'e' :: 'r' :: ':' :: rest =>
      parseDecimalNatAux rest 0 false
  | _ => none

def decodeTy? (value : String) : Option TyRef :=
  match value with
  | "" => none
  | "bool" => some (.spelling "bool")
  | "i1" => some (.spelling "i1")
  | "u256" => some (.spelling "u256")
  | "uint256" => some (.spelling "uint256")
  | "i256" => some (.spelling "i256")
  | "int256" => some (.spelling "int256")
  | _ =>
      match parseCompilerTypeId? value with
      | some id => some (.compilerTypeId id)
      | none => none

def decodeUnaryOp : String → Option UnaryOp
  | "not" => some .not_
  | "neg" => some .neg
  | _ => none

def decodeBinaryOp : String → Option BinaryOp
  | "eq" => some .eq
  | "ne" => some .ne
  | "lt" => some .lt
  | "le" => some .le
  | "gt" => some .gt
  | "ge" => some .ge
  | "slt" => some .slt
  | "sle" => some .sle
  | "sgt" => some .sgt
  | "sge" => some .sge
  | "add" => some .add
  | "sub" => some .sub
  | "mul" => some .mul
  | "pow" => some .pow
  | "shl" => some .shl
  | "shr" => some .shr
  | "div" => some .div
  | "mod" => some .mod_
  | "bit_and" => some .bitAnd
  | "bit_xor" => some .bitXor
  | "and" => some .and_
  | "or" => some .or_
  | "implies" => some .implies
  | _ => none

def decodeQuantifier : String → Option Quantifier
  | "forall" => some .forall_
  | "exists" => some .exists_
  | _ => none

def decodeTerm : RawTerm → Option Term
  | (tag, boolValue, text, tyName, name, lhs, rhs, condition, args, rawPlace) =>
      match tag with
      | "bool_lit" => some (.boolLit boolValue)
      | "int_lit" =>
          match decodeTy? tyName with
          | some ty => some (.intLit { value := text, ty := some ty })
          | none => none
      | "free_var" =>
          match decodeTy? tyName with
          | some ty =>
              some (.variable (.free {
                id := { file_id := lhs, pattern_id := rhs },
                name := name,
                ty := some ty
              }))
          | none => none
      | "bound_var" =>
          match decodeTy? tyName with
          | some ty =>
              some (.variable (.bound { index := lhs, name := name, ty := some ty }))
          | none => none
      | "old" => some (.old lhs)
      | "result" => some .result
      | "place_read" =>
          match decodePlace rawPlace with
          | some place => some (.placeRead place)
          | none => none
      | "unary" =>
          match decodeUnaryOp text with
          | some op => some (.unary { op := op, operand := lhs })
          | none => none
      | "binary" =>
          match decodeBinaryOp text, decodeTy? tyName with
          | some op, some ty => some (.binary { op := op, lhs := lhs, rhs := rhs, ty := some ty })
          | some op, none =>
              if tyName == "" then
                some (.binary { op := op, lhs := lhs, rhs := rhs })
              else
                none
          | _, _ => none
      | "refinement" =>
          some (.refinementPredicate { name := text, value := lhs, args := args })
      | "quantified" =>
          match decodeQuantifier text, decodeTy? tyName with
          | some quantifier, some ty =>
              some (.quantified {
                quantifier := quantifier,
                binder := { name := name, ty := some ty },
                condition := condition,
                body := lhs
              })
          | _, _ => none
      | _ => none

def decodeTerms : List RawTerm → Option (List Term)
  | [] => some []
  | raw :: rest =>
      match decodeTerm raw, decodeTerms rest with
      | some term, some terms => some (term :: terms)
      | _, _ => none

def decodeAssumptionKind : String → Option AssumptionKind
  | "requires" => some .requires
  | "assume" => some .assume
  | "path_assume" => some .pathAssume
  | "env_assume" => some .envAssume
  | "binding" => some .binding
  | "two_state_linkage" => some .twoStateLinkage
  | "frame" => some .frame
  | "loop_invariant" => some .loopInvariant
  | "callee_obligation" => some .calleeObligation
  | "callee_ensures" => some .calleeEnsures
  | "ghost_axiom" => some .ghostAxiom
  | "goal" => some .goal
  | _ => none

def decodeAssumption : RawAssumption → Option AssumptionRow
  | (id, kindName, termId) =>
      match decodeAssumptionKind kindName with
      | some kind =>
          some {
            id := id,
            owner := "obligation_totality_fixture",
            kind := kind,
            formula := termId.map FormulaRef.term
          }
      | none => none

def decodeAssumptions : List RawAssumption → Option (List AssumptionRow)
  | [] => some []
  | raw :: rest =>
      match decodeAssumption raw, decodeAssumptions rest with
      | some row, some rows => some (row :: rows)
      | _, _ => none

def decodeEffectRelation : String → Option EffectFrameRelation
  | "write_covered_by_modifies" => some .writeCoveredByModifies
  | "read_preserved_by_frame" => some .readPreservedByFrame
  | "read_preserved_by_key_evidence" => some .readPreservedByKeyEvidence
  | "lock_covers_write" => some .lockCoversWrite
  | "external_call_frame" => some .externalCallFrame
  | _ => none

def decodeEvidenceKind : String → Option KeyDisjointEvidenceKind
  | "free_var_disequality" => some .freeVarDisequality
  | _ => none

def decodeEvidence : RawEvidence → Option KeyDisjointEvidence
  | (kindName, assumptionId, lhsFile, lhsPattern, rhsFile, rhsPattern, readRaw, writeRaw, keyIndex) =>
      match decodeEvidenceKind kindName, decodePlace readRaw, decodePlace writeRaw with
      | some kind, some read, some write =>
          some {
            kind := kind,
            assumptionId := assumptionId,
            lhs := { file_id := lhsFile, pattern_id := lhsPattern },
            rhs := { file_id := rhsFile, pattern_id := rhsPattern },
            read := read,
            write := write,
            keyIndex := keyIndex
          }
      | _, _, _ => none

def decodeEvidenceList : List RawEvidence → Option (List KeyDisjointEvidence)
  | [] => some []
  | raw :: rest =>
      match decodeEvidence raw, decodeEvidenceList rest with
      | some evidence, some tail => some (evidence :: tail)
      | _, _ => none

def decodePlaceList : List RawPlace → Option (List PlaceRef)
  | [] => some []
  | raw :: rest =>
      match decodePlace raw, decodePlaceList rest with
      | some place, some places => some (place :: places)
      | _, _ => none

def decodeEffect : RawEffect → Option EffectFrameGoal
  | (relationName, declaredRaw, actualRaw, evidenceRaw) =>
      match decodeEffectRelation relationName,
            decodePlaceList declaredRaw,
            decodePlaceList actualRaw,
            decodeEvidenceList evidenceRaw with
      | some relation, some declared, some actual, some evidence =>
          some { relation := relation, declared := declared, actual := actual, evidence := evidence }
      | _, _, _, _ => none

def decodeResourceOperation : String → Option ResourceOperation
  | "move" => some .move
  | "create" => some .create
  | "destroy" => some .destroy
  | _ => none

def decodeResourceProperty : String → Option ResourceProperty
  | "amount_non_negative" => some .amountNonNegative
  | "source_sufficient" => some .sourceSufficient
  | "destination_no_overflow" => some .destinationNoOverflow
  | "same_place_identity" => some .samePlaceIdentity
  | "conservation" => some .conservation
  | _ => none

def decodeOptionalPlace : Option RawPlace → Option (Option PlaceRef)
  | none => some none
  | some raw =>
      match decodePlace raw with
      | some place => some (some place)
      | none => none

def decodeResource : RawResource → Option ResourceGoal
  | (opName, domain, sourceRaw, destinationRaw, amountTerm, propertyName) =>
      match decodeResourceOperation opName,
            decodeOptionalPlace sourceRaw,
            decodeOptionalPlace destinationRaw,
            decodeResourceProperty propertyName with
      | some op, some source, some destination, some property =>
          some {
            op := op,
            domain := domain,
            source := source,
            destination := destination,
            amount := amountTerm.map FormulaRef.term,
            property := property
          }
      | _, _, _, _ => none

/-! ## Decoder tag pins

Each string-enum decoder above is a hand-written mirror of the Zig emitter's
name tables (`emit_obligation_totality_snapshot.zig`). A tag swap in a decoder
(e.g. decoding "sle" as `.slt`) keeps every fixture denotable, so the totality
theorem alone cannot catch it. These theorems pin the FULL decode tables by
kernel `decide`; the Zig emit side is pinned by exhaustive switches over the
enums. A rename on either side must update both, or the sync gate goes red. -/

theorem unary_op_decode_pins :
    ["not", "neg"].map decodeUnaryOp = [some .not_, some .neg] := by decide

theorem binary_op_decode_pins :
    ["eq", "ne", "lt", "le", "gt", "ge", "slt", "sle", "sgt", "sge",
     "add", "sub", "mul", "pow", "shl", "shr", "div", "mod", "bit_and", "bit_xor",
     "and", "or", "implies"].map decodeBinaryOp =
      [some .eq, some .ne, some .lt, some .le, some .gt, some .ge,
       some .slt, some .sle, some .sgt, some .sge,
       some .add, some .sub, some .mul, some .pow, some .shl, some .shr,
       some .div, some .mod_, some .bitAnd, some .bitXor,
       some .and_, some .or_, some .implies] := by decide

theorem quantifier_decode_pins :
    ["forall", "exists"].map decodeQuantifier = [some .forall_, some .exists_] := by decide

theorem region_decode_pins :
    ["none", "storage", "memory", "transient", "calldata"].map decodeRegion =
      [some .none, some .storage, some .memory, some .transient, some .calldata] := by decide

theorem assumption_kind_decode_pins :
    ["requires", "assume", "path_assume", "env_assume", "binding",
     "two_state_linkage", "frame", "loop_invariant", "callee_obligation",
     "callee_ensures", "ghost_axiom", "goal"].map decodeAssumptionKind =
      [some .requires, some .assume, some .pathAssume, some .envAssume, some .binding,
       some .twoStateLinkage, some .frame, some .loopInvariant, some .calleeObligation,
       some .calleeEnsures, some .ghostAxiom, some .goal] := by decide

theorem effect_relation_decode_pins :
    ["write_covered_by_modifies", "read_preserved_by_frame",
     "read_preserved_by_key_evidence", "lock_covers_write",
     "external_call_frame"].map decodeEffectRelation =
      [some .writeCoveredByModifies, some .readPreservedByFrame,
       some .readPreservedByKeyEvidence, some .lockCoversWrite,
       some .externalCallFrame] := by decide

theorem evidence_kind_decode_pins :
    ["free_var_disequality"].map decodeEvidenceKind = [some .freeVarDisequality] := by decide

theorem resource_operation_decode_pins :
    ["move", "create", "destroy"].map decodeResourceOperation =
      [some .move, some .create, some .destroy] := by decide

theorem resource_property_decode_pins :
    ["amount_non_negative", "source_sufficient", "destination_no_overflow",
     "same_place_identity", "conservation"].map decodeResourceProperty =
      [some .amountNonNegative, some .sourceSufficient, some .destinationNoOverflow,
       some .samePlaceIdentity, some .conservation] := by decide

theorem place_key_tag_decode_pins :
    [decodeKey ("parameter", "file:1:pattern:2"),
     decodeKey ("comptime_parameter", "3"),
     decodeKey ("comptime_range_parameter", "4"),
     decodeKey ("constant", "42"),
     decodeKey ("msg_sender", ""),
     decodeKey ("tx_origin", ""),
     decodeKey ("unknown", "")] =
      [some (.parameter { file_id := 1, pattern_id := 2 }),
       some (.comptimeParameter 3),
       some (.comptimeRangeParameter 4),
       some (.constant "42"),
       some .msgSender,
       some .txOrigin,
       some .unknown] := by decide

theorem ty_spelling_decode_pins :
    ["bool", "i1", "u256", "uint256", "i256", "int256", "compiler:12", ""].map decodeTy? =
      [some (.spelling "bool"), some (.spelling "i1"), some (.spelling "u256"),
       some (.spelling "uint256"), some (.spelling "i256"), some (.spelling "int256"),
       some (.compilerTypeId 12), none] := by decide

def valueForTy? : Option TyRef → Value
  | some ty =>
      match ty.integerShape? with
      | some shape => .integer (Ora.Integer.Value.ofNat shape 0)
      | none => .bool true
  | none => .bool false

def bindFreeVarsFromTerm (env : Env) : Term → Env
  | .variable (.free free) => env.setFree free.id (valueForTy? free.ty)
  | _ => env

def canonicalEnv (manifest : Manifest) : Env :=
  { manifest.terms.foldl bindFreeVarsFromTerm Env.empty with
    placeValue := fun _ => .integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)
    result := some (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)) }

def valueDenotes? (manifest : Manifest) (env : Env) (fuel : Nat) (id : TermId) : Bool :=
  (denoteValue? manifest env fuel id).isSome

def valueDenotesInteger? (manifest : Manifest) (env : Env) (fuel : Nat) (id : TermId) : Bool :=
  match denoteValue? manifest env fuel id with
  | some (.integer _) => true
  | _ => false

def valueDenotesU256? (manifest : Manifest) (env : Env) (fuel : Nat) (id : TermId) : Bool :=
  match denoteValue? manifest env fuel id with
  | some (.integer value) => (value.bitsFor? (.unsigned .w256)).isSome
  | _ => false

def valueDenotesBool? (manifest : Manifest) (env : Env) (fuel : Nat) (id : TermId) : Bool :=
  match denoteValue? manifest env fuel id with
  | some (.bool _) => true
  | _ => false

def valueEqDenotable?
    (manifest : Manifest)
    (env : Env)
    (fuel : Nat)
    (lhs rhs : TermId) : Bool :=
  match denoteValue? manifest env fuel lhs, denoteValue? manifest env fuel rhs with
  | some lhsValue, some rhsValue => (Value.eqProp? lhsValue rhsValue).isSome
  | _, _ => false

def refinementPredicateFullyDenotable?
    (manifest : Manifest)
    (env : Env)
    (fuel : Nat)
    (name : String)
    (value : TermId)
    (args : List TermId) : Bool :=
  let argsDenote := args.all (valueDenotesInteger? manifest env fuel)
  match name, args with
  | "NonZero", [] => valueDenotesInteger? manifest env fuel value
  | "MinValue", [_] => valueDenotesInteger? manifest env fuel value && argsDenote
  | "MaxValue", [_] => valueDenotesInteger? manifest env fuel value && argsDenote
  | "InRange", [_, _] => valueDenotesInteger? manifest env fuel value && argsDenote
  | "BasisPoints", [] => valueDenotesInteger? manifest env fuel value
  | _, _ => false

def formulaFullyDenotable? (manifest : Manifest) (env : Env) : Nat → TermId → Bool
  | 0, _ => false
  | fuel + 1, id =>
      match manifest.terms[id]? with
      | some (.boolLit _) => true
      | some (.variable _) => valueDenotesBool? manifest env fuel id
      | some (.refinementPredicate predicate) =>
          refinementPredicateFullyDenotable? manifest env fuel
            predicate.name predicate.value predicate.args
      | some (.unary unary) =>
          match unary.op with
          | .not_ =>
              formulaFullyDenotable? manifest env fuel unary.operand
          | .neg => false
      | some (.binary binary) =>
          match binary.op with
          | .eq | .ne =>
              valueEqDenotable? manifest env fuel binary.lhs binary.rhs
          | .lt | .le | .gt | .ge =>
              match binary.ty with
              | some ty =>
                  match ty.integerShape? with
                  | some (.unsigned _) =>
                      valueDenotesInteger? manifest env fuel binary.lhs &&
                        valueDenotesInteger? manifest env fuel binary.rhs
                  | _ => false
              | none => false
          | .slt | .sle | .sgt | .sge =>
              match binary.ty with
              | some ty =>
                  match ty.integerShape? with
                  | some (.signed _) =>
                      valueDenotesInteger? manifest env fuel binary.lhs &&
                        valueDenotesInteger? manifest env fuel binary.rhs
                  | _ => false
              | none => false
          | .and_ | .or_ | .implies =>
              formulaFullyDenotable? manifest env fuel binary.lhs &&
                formulaFullyDenotable? manifest env fuel binary.rhs
          | .add | .sub | .mul | .pow | .shl | .shr | .div | .mod_ |
              .bitAnd | .bitXor => false
      | some (.quantified quantified) =>
          match quantified.binder.integerShape? with
          | none => false
          | some shape =>
            let localEnv := env.pushBound (.integer (Ora.Integer.Value.ofNat shape 0))
            let conditionOk :=
              match quantified.condition with
              | none => true
              | some condition => formulaFullyDenotable? manifest localEnv fuel condition
            conditionOk && formulaFullyDenotable? manifest localEnv fuel quantified.body
      | some (.intLit _)
      | some (.old _)
      | some .result
      | some (.placeRead _) => false
      | none => false

def formulaRefFullyDenotable? (manifest : Manifest) (env : Env) : FormulaRef → Bool
  | .term id => formulaFullyDenotable? manifest env (manifest.terms.length + 1) id

def keyEvidenceFormulaFullyDenotable?
    (manifest : Manifest)
    (env : Env)
    (evidence : KeyDisjointEvidence) : Bool :=
  match manifest.assumptionById evidence.assumptionId with
  | some row =>
      if row.kind != .requires then
        false
      else
        match row.formula with
        | some (.term termId) =>
            match manifest.terms[termId]? with
            | some (.binary binary) =>
                if binary.op != .ne then
                  false
                else
                  match termFreeVarId? manifest binary.lhs,
                        termFreeVarId? manifest binary.rhs with
                  | some lhs, some rhs =>
                      freeVarPairMatches lhs rhs evidence.lhs evidence.rhs &&
                        formulaFullyDenotable? manifest env (manifest.terms.length + 1) termId
                  | _, _ => false
            | _ => false
        | _ => false
  | none => false

def keyEvidenceFullyDenotable?
    (manifest : Manifest)
    (env : Env)
    (evidence : KeyDisjointEvidence) : Bool :=
  match evidence.kind with
  | .freeVarDisequality =>
      keyEvidencePathMatches evidence.read evidence.write evidence.keyIndex
        evidence.lhs evidence.rhs &&
        keyEvidenceFormulaFullyDenotable? manifest env evidence

def evidenceListFullyDenotable?
    (manifest : Manifest)
    (env : Env)
    (evidence : List KeyDisjointEvidence) : Bool :=
  evidence.all (keyEvidenceFullyDenotable? manifest env)

def effectFrameFullyDenotable?
    (manifest : Manifest)
    (env : Env)
    (goal : EffectFrameGoal) : Bool :=
  match goal.relation with
  | .writeCoveredByModifies => true
  | .readPreservedByFrame => true
  | .readPreservedByKeyEvidence => evidenceListFullyDenotable? manifest env goal.evidence
  | .lockCoversWrite
  | .externalCallFrame => false

def resourceAmountFullyDenotable? (manifest : Manifest) (env : Env) (goal : ResourceGoal) : Bool :=
  match goal.amount with
  | some (.term id) => valueDenotesU256? manifest env (manifest.terms.length + 1) id
  | none => false

def resourceSourceKnown? (env : Env) (goal : ResourceGoal) : Bool :=
  match goal.source with
  | some source => (env.resourcePlaceKnown? source).isSome
  | none => false

def resourceDestinationKnown? (env : Env) (goal : ResourceGoal) : Bool :=
  match goal.destination with
  | some destination => (env.resourcePlaceKnown? destination).isSome
  | none => false

def resourceGoalFullyDenotable? (manifest : Manifest) (env : Env) (goal : ResourceGoal) : Bool :=
  resourceAmountFullyDenotable? manifest env goal &&
    match goal.property with
    | .amountNonNegative => true
    | .sourceSufficient =>
        match goal.op with
        | .move => resourceSourceKnown? env goal && resourceDestinationKnown? env goal
        | .destroy => resourceSourceKnown? env goal
        | .create => false
    | .destinationNoOverflow =>
        match goal.op with
        | .move => resourceSourceKnown? env goal && resourceDestinationKnown? env goal
        | .create => resourceDestinationKnown? env goal
        | .destroy => false
    | .samePlaceIdentity | .conservation =>
        match goal.op with
        | .move => resourceSourceKnown? env goal && resourceDestinationKnown? env goal
        | .create | .destroy => false

def targetTotal?
    (manifest : Manifest)
    (env : Env)
    (targetKind : String)
    (targetTerm : Nat)
    (effect : EffectFrameGoal)
    (resource : ResourceGoal) : Bool :=
  match targetKind with
  | "formula" => formulaRefFullyDenotable? manifest env (.term targetTerm)
  | "effect" => effectFrameFullyDenotable? manifest env effect
  | "resource" => resourceGoalFullyDenotable? manifest env resource
  | _ => false

def assumptionsFullyDenotable?
    (manifest : Manifest)
    (env : Env)
    (assumptions : List AssumptionRow) : Bool :=
  (assumptionsDenoteInEnv? manifest env assumptions).isSome

def rowMatches : RawRow → Bool
  | (_, rawTerms, rawAssumptions, mustSupport, zigSupported, targetKind, targetTerm,
      rawEffect, rawResource) =>
      match decodeTerms rawTerms,
            decodeAssumptions rawAssumptions,
            decodeEffect rawEffect,
            decodeResource rawResource with
      | some terms, some assumptions, some effect, some resource =>
          let manifest : Manifest := {
            terms := terms,
            assumptions := assumptions,
            obligations := [],
            proofArtifacts := []
          }
          let env := canonicalEnv manifest
          let leanSupported :=
            manifest.wf &&
              assumptionsFullyDenotable? manifest env assumptions &&
              targetTotal? manifest env targetKind targetTerm effect resource
          if mustSupport then
            zigSupported && leanSupported
          else
            (!zigSupported) && (!leanSupported)
      | _, _, _, _ => false

theorem obligation_totality_fixture_matches :
    obligationTotalityRows.all rowMatches = true := by decide


end Ora.ObligationTotalitySync
