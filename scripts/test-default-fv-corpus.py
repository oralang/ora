#!/usr/bin/env python3
"""Fault injection for the corpus harness, not a compiler/prover fault-injection mode."""
import copy
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("corpus", Path(__file__).with_name("check-default-fv-corpus.py"))
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


class CorpusGateTests(unittest.TestCase):
    def setUp(self):
        loop = {"id": 1, "owner": {"function_name": "count"}, "source": {"file": "ora-example/count.ora", "line": 2, "column": 3},
                "status": "lean_certificate_required", "excluded_by": [], "induction_query_id": 5,
                "projection_completeness": "complete"}
        loops = {"total": 1, "loops": [loop]}
        self.actual = {"exit": 0, "hex": ["count.hex"], "terminal_errors": [], "smt_errors": [],
                       "smt_diagnostics": [], "loops": corpus.loop_rows(loops)}
        self.expected = {"category": "accepted", "artifacts": 1, "explanation": "bounded counter", **self.actual}
        smt = {"summary": {"unknown": 0, "vacuous": 0, "vacuity_unknown": 0, "inconsistent_bases": 0,
                           "encoding_degraded": False, "soundness_losses": []}, "errors": [], "diagnostics": [],
               "verification": {"success": True, "verification_trust": "full"},
               "queries": [{"status": "UNSAT", "smtlib_hash": "0x1234", "vacuous": False,
                            "vacuity_unknown": False, "verified_with_caveats": False}]}
        authority = {"final_artifact_authorization": "allowed", "gates": [
            {"id": name, "status": "accepted"} for name in ("z3_userland", "lean_userland", "source_accounting_kernel")]}
        proof = {"status": "checked", "proof_count": 1, "proofs": [
            {"loop_summary_id": 1, "target_query_id": 5, "lean_status": "checked", "anti_vacuity_status": "proved",
             "authorization_claim": "loop_induction_and_nonvacuity", "loop_vc_sha256": "a" * 64,
             "axioms": ["propext", "Quot.sound"]}]}
        self.reports = [smt, loops, authority, proof, {"decision": "accepted", "failures": []}]

    def check(self):
        corpus.validate_result(self.expected, self.actual, self.reports)

    def test_accepted_proof(self):
        self.check()

    def test_retained_guard_diagnostic_is_bound_to_its_real_query(self):
        diagnostic = {"query_id": 7, "guard_id": "guard:count:input", "subtype": "GuardViolation.NonZero", "counterexample": {"x": "0"}}
        self.reports[0]["diagnostics"] = [diagnostic]
        self.reports[0]["queries"].append({"id": 7, "kind": "guard_violate", "guard_id": diagnostic["guard_id"],
            "status": "SAT", "smtlib_hash": "0x5678", "vacuous": False, "vacuity_unknown": False, "verified_with_caveats": False})
        self.reports[0]["queries"][0]["id"] = 1
        self.actual["smt_diagnostics"] = corpus.smt_diagnostics(self.reports[0])
        self.expected["smt_diagnostics"] = copy.deepcopy(self.actual["smt_diagnostics"])
        self.check()
        for key, value in (("kind", "obligation"), ("status", "UNSAT"), ("guard_id", "foreign")):
            with self.subTest(key=key):
                saved = self.reports[0]["queries"][-1][key]
                self.reports[0]["queries"][-1][key] = value
                with self.assertRaisesRegex(ValueError, "unbound retained-guard"):
                    self.check()
                self.reports[0]["queries"][-1][key] = saved

    def test_named_exclusion(self):
        self.actual.update(exit=1, hex=[])
        self.actual["loops"][0]["reasons"] = ["loop_has_call"]
        self.actual["loops"][0]["status"] = "not_verified"
        self.expected = {**copy.deepcopy(self.actual), "category": "excluded", "track": "automatic-call-loops",
                         "explanation": "default automatic calls are excluded"}
        self.reports[2]["final_artifact_authorization"] = "blocked"
        self.check()
        self.actual.update(exit=0, hex=["unexpected.hex"])
        with self.assertRaisesRegex(ValueError, "unexpected acceptance"):
            self.check()

    def test_every_authority_mutation_blocks(self):
        mutations = {
            "timeout": lambda a, r: a.update(exit="timeout"),
            "signal": lambda a, r: a.update(exit=-9),
            "missing_hex": lambda a, r: a.update(hex=[]),
            "missing_smt": lambda a, r: r.__setitem__(0, None),
            "missing_authority": lambda a, r: r.__setitem__(2, None),
            "missing_accounting": lambda a, r: r.__setitem__(4, None),
            "accounting_rejected": lambda a, r: r[4].update(decision="rejected"),
            "authority_rejected": lambda a, r: r[2].update(final_artifact_authorization="blocked"),
            "z3_rejected": lambda a, r: r[2]["gates"][0].update(status="rejected"),
            "lean_missing": lambda a, r: r[2]["gates"][1].update(status="not_requested"),
            "missing_proof": lambda a, r: r.__setitem__(3, None),
            "proof_count": lambda a, r: r[3].update(proof_count=0),
            "proof_row_missing": lambda a, r: r[3].update(proofs=[], proof_count=0),
            "wrong_query": lambda a, r: r[3]["proofs"][0].update(target_query_id=9),
            "wrong_loop": lambda a, r: r[3]["proofs"][0].update(loop_summary_id=9),
            "missing_digest": lambda a, r: r[3]["proofs"][0].update(loop_vc_sha256=None),
            "sorry": lambda a, r: r[3]["proofs"][0].update(axioms=["sorryAx"]),
            "vacuous_proof": lambda a, r: r[3]["proofs"][0].update(anti_vacuity_status="unknown"),
            "wrong_claim": lambda a, r: r[3]["proofs"][0].update(authorization_claim="function_custody"),
            "unknown_summary": lambda a, r: r[0]["summary"].update(unknown=1),
            "vacuous_summary": lambda a, r: r[0]["summary"].update(vacuous=1),
            "degraded_summary": lambda a, r: r[0]["summary"].update(encoding_degraded=True),
            "unknown_query": lambda a, r: r[0]["queries"][0].update(status="UNKNOWN"),
            "missing_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash=None),
            "degraded_query": lambda a, r: r[0]["queries"][0].update(verified_with_caveats=True),
            "changed_loop_file": lambda a, r: a["loops"][0].update(file="ora-example/foreign.ora"),
            "changed_loop_site": lambda a, r: a["loops"][0].update(line=3),
            "changed_loop_reason": lambda a, r: a["loops"][0].update(reasons=["loop_backedge_missing"]),
            "removed_loop": lambda a, r: a.update(loops=[]),
            "extra_error": lambda a, r: a.update(terminal_errors=["error: unexpected"]),
            "extra_smt_error": lambda a, r: a.update(smt_errors=[{"message": "unexpected"}]),
            "extra_smt_diagnostic": lambda a, r: a.update(smt_diagnostics=[{"message": "unexpected"}]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                actual, reports = copy.deepcopy((self.actual, self.reports))
                mutate(actual, reports)
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    corpus.validate_result(self.expected, actual, reports)

    def test_new_or_missing_file_blocks(self):
        manifest = {"files": {"ora-example/expected.ora": {}}}
        for files in ([], ["ora-example/expected.ora", "ora-example/new.ora"]):
            with self.subTest(files=files), self.assertRaisesRegex(ValueError, "unclassified/missing"):
                corpus.membership(manifest, files)

    def test_report_ambiguity_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "one.smt.report.json").write_text("{")
            with self.assertRaises(ValueError):
                corpus.read_report(directory, "smt.report.json")
            (directory / "two.smt.report.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                corpus.read_report(directory, "smt.report.json")

    def test_diagnostics_are_whole_lines(self):
        self.assertEqual(["error: division by zero"], corpus.terminal_errors("noise\nerror: division by zero\n"))
        self.assertNotEqual(corpus.terminal_errors("error: division by zero"), corpus.terminal_errors("error: division by zero; extra"))


class VerificationNegativeTests(unittest.TestCase):
    def setUp(self):
        CorpusGateTests.setUp(self)
        self.actual.update(source="ora-example/negative.ora", source_sha256="a" * 64, exit=1, hex=[])
        smt, _, authority, _, _ = self.reports
        smt.update(source_file=self.actual["source"])
        smt["verification"]["success"] = False
        smt["summary"]["inconsistent_bases"] = 1
        smt["queries"][0].update(id=1, kind="base", function_name="impossible")
        smt["errors"] = [{"type": "PreconditionViolation", "subtype": "InconsistentAssumptions",
                          "message": "verification assumptions are inconsistent in impossible"}]
        authority.update(source=self.actual["source"], final_artifact_authorization="blocked")
        authority["gates"][0]["status"] = "rejected"
        self.actual["smt_errors"] = corpus.smt_errors(smt)
        self.expected = {**copy.deepcopy(self.actual), "category": "verification_negative", "track": "C11",
                         "explanation": "exact contradictory-source refusal; never an accepted proof",
                         "rejection_evidence": corpus.rejection_evidence(smt)}

    def check(self):
        corpus.validate_result(self.expected, self.actual, self.reports)

    def test_exact_inconsistent_and_degraded_refusals(self):
        self.check()
        smt = self.reports[0]
        smt["verification"]["verification_trust"] = "degraded"
        smt["summary"].update(inconsistent_bases=0, encoding_degraded=True, soundness_losses=["unresolved_callee"],
                              degradation_reasons=["unresolved external callee has no sound state summary"])
        smt["queries"] = []
        smt["errors"] = [{"type": "EncodingDegraded", "message": smt["summary"]["degradation_reasons"][0]}]
        self.actual["smt_errors"] = corpus.smt_errors(smt)
        self.expected["smt_errors"] = copy.deepcopy(self.actual["smt_errors"])
        self.expected["rejection_evidence"] = corpus.rejection_evidence(smt)
        self.check()
        for category in ("accepted", "excluded", "invalid", "unsafe"):
            with self.subTest(category=category), self.assertRaisesRegex(ValueError, "degraded verification trust"):
                corpus.validate_result({**self.expected, "category": category}, self.actual, self.reports)

    def test_exact_pins_reject_report_and_source_mutations(self):
        mutations = {
            "foreign_source": lambda a, r: a.update(source="ora-example/other_negative.ora"),
            "source_hash": lambda a, r: a.update(source_sha256="b" * 64),
            "foreign_smt": lambda a, r: r[0].update(source_file="ora-example/foreign.ora"),
            "foreign_authority": lambda a, r: r[2].update(source="ora-example/foreign.ora"),
            "terminal": lambda a, r: a.update(terminal_errors=["error: different"]),
            "error": lambda a, r: a.update(smt_errors=[]),
            "diagnostic": lambda a, r: a.update(smt_diagnostics=[{"type": "different"}]),
            "loop_removed": lambda a, r: a.update(loops=[]),
            "loop_owner": lambda a, r: a["loops"][0].update(function="foreign"),
            "loop_file": lambda a, r: a["loops"][0].update(file="foreign.ora"),
            "loop_site": lambda a, r: a["loops"][0].update(line=999),
            "loop_status": lambda a, r: a["loops"][0].update(status="different"),
            "loop_reason": lambda a, r: a["loops"][0].update(reasons=["different"]),
            "trust": lambda a, r: r[0]["verification"].update(verification_trust="degraded"),
            "error_count": lambda a, r: r[0]["verification"].update(errors=10),
            "inconsistent_count": lambda a, r: r[0]["summary"].update(inconsistent_bases=2),
            "loss": lambda a, r: r[0]["summary"].update(soundness_losses=["unexpected"]),
            "reason": lambda a, r: r[0]["summary"].update(degradation_reason="new failure"),
            "query_removed": lambda a, r: r[0].update(queries=[]),
            "query_duplicate": lambda a, r: r[0]["queries"].append(copy.deepcopy(r[0]["queries"][0])),
            "query_id": lambda a, r: r[0]["queries"][0].update(id=8),
            "query_owner": lambda a, r: r[0]["queries"][0].update(function_name="foreign"),
            "query_kind": lambda a, r: r[0]["queries"][0].update(kind="obligation"),
            "query_status": lambda a, r: r[0]["queries"][0].update(status="SAT"),
            "query_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash="0x9876"),
            "query_constraints": lambda a, r: r[0]["queries"][0].update(smtlib="(assert false)"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                actual, reports = copy.deepcopy((self.actual, self.reports))
                mutate(actual, reports)
                with self.assertRaises(ValueError):
                    corpus.validate_result(self.expected, actual, reports)

    def test_matching_pins_cannot_excuse_fatal_states(self):
        mutations = {
            "timeout": lambda a, r: a.update(exit="timeout"),
            "signal": lambda a, r: a.update(exit=-9),
            "ice": lambda a, r: a.update(exit=2),
            "success_exit": lambda a, r: a.update(exit=0),
            "bytecode": lambda a, r: a.update(hex=["forged.hex"]),
            "missing_smt": lambda a, r: r.__setitem__(0, None),
            "missing_authority": lambda a, r: r.__setitem__(2, None),
            "malformed_authority": lambda a, r: r.__setitem__(2, {}),
            "authority_allowed": lambda a, r: r[2].update(final_artifact_authorization="allowed"),
            "authority_unknown": lambda a, r: r[2].update(final_artifact_authorization="unknown"),
            "gate_missing": lambda a, r: r[2].update(gates=[]),
            "gate_accepted": lambda a, r: r[2]["gates"][0].update(status="accepted"),
            "gate_id": lambda a, r: r[2]["gates"][0].update(id="foreign"),
            "gate_duplicate": lambda a, r: r[2]["gates"].append(copy.deepcopy(r[2]["gates"][0])),
            "success": lambda a, r: r[0]["verification"].update(success=True),
            "unknown_trust": lambda a, r: r[0]["verification"].update(verification_trust="unknown"),
            "missing_error": lambda a, r: r[0].update(errors=[]),
            "unnamed_error": lambda a, r: r[0]["errors"][0].update(type=None),
            "unknown_summary": lambda a, r: r[0]["summary"].update(unknown=1),
            "vacuous_summary": lambda a, r: r[0]["summary"].update(vacuous=1),
            "vacuity_unknown_summary": lambda a, r: r[0]["summary"].update(vacuity_unknown=1),
            "unknown_query": lambda a, r: r[0]["queries"][0].update(status="UNKNOWN"),
            "vacuous_query": lambda a, r: r[0]["queries"][0].update(vacuous=True),
            "vacuity_unknown_query": lambda a, r: r[0]["queries"][0].update(vacuity_unknown=True),
            "caveat_query": lambda a, r: r[0]["queries"][0].update(verified_with_caveats=True),
            "missing_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash=None),
            "empty_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash=""),
            "zero_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash="0x0000"),
            "malformed_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash="not-a-hash"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                expected, actual, reports = copy.deepcopy((self.expected, self.actual, self.reports))
                mutate(actual, reports)
                # Even a matching expectation cannot authorize a fatal state.
                if reports[0] is not None:
                    expected["rejection_evidence"] = corpus.rejection_evidence(reports[0])
                    actual["smt_errors"] = corpus.smt_errors(reports[0])
                    expected["smt_errors"] = copy.deepcopy(actual["smt_errors"])
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    corpus.validate_result(expected, actual, reports)

    def test_category_requires_complete_exact_expectation(self):
        for key in ("source", "source_sha256", "rejection_evidence", "track", "explanation"):
            with self.subTest(key=key):
                expected = copy.deepcopy(self.expected)
                del expected[key]
                with self.assertRaises((ValueError, KeyError)):
                    corpus.validate_result(expected, self.actual, self.reports)
        self.expected["source_sha256"] = self.actual["source_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "negative source hash"):
            self.check()

    def test_only_checkout_paths_are_normalized(self):
        root = str(corpus.ROOT)
        smt = copy.deepcopy(self.reports[0])
        for absent in (None, ""):
            smt["errors"][0]["file"] = absent
            self.assertIsNone(corpus.smt_errors(smt)[0]["file"])
        smt["errors"][0].update(file=root + "/ora-example/negative.ora",
                                message='rejected at loc("' + root + '/ora-example/negative.ora":5:9)')
        smt["queries"][0].update(file=root + "/ora-example/negative.ora")
        errors = corpus.smt_errors(smt)
        self.assertEqual(errors[0]["file"], "ora-example/negative.ora")
        self.assertEqual(errors[0]["message"], 'rejected at loc("ora-example/negative.ora":5:9)')
        evidence = corpus.rejection_evidence(smt)
        moved = copy.deepcopy(smt)
        moved["errors"][0]["file"] = "/other/checkout/ora-example/negative.ora"
        moved["errors"][0]["message"] = errors[0]["message"].replace("ora-example/", "/other/checkout/ora-example/")
        moved["queries"][0]["file"] = moved["errors"][0]["file"]
        with mock.patch.object(corpus, "ROOT", Path("/other/checkout")):
            self.assertEqual(corpus.smt_errors(moved), errors)
            self.assertEqual(corpus.rejection_evidence(moved), evidence)
            self.assertEqual(corpus.relative_paths("/foreign/checkout/file.ora"), "/foreign/checkout/file.ora")
        smt["errors"][0]["message"] += "; another reason"
        self.assertNotEqual(corpus.smt_errors(smt), errors)


class AutomaticRefusalTests(unittest.TestCase):
    """All fake compiler output is produced inside a fresh, owned temporary directory."""

    def setUp(self):
        CorpusGateTests.setUp(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(corpus, "ROOT", self.root).start()
        self.source = "ora-example/refusal.ora"
        source = self.root / self.source
        source.parent.mkdir()
        source.write_text("// test-only source identity\n")
        self.module = "namespace Test\ndef emittedQuery_5 : Prop := False\nend Test\n"
        self.reason = "automatic_nonlinear_arithmetic"
        self.actual.update(source=self.source, source_sha256=corpus.sha(source), exit=1, hex=[])
        smt, loops, authority, _, _ = self.reports
        smt.update(schema="ora.smt.report.v1", source_file=self.source, settings={"timeout_ms": 60000})
        smt["summary"].update(failed_obligations=0)
        smt["queries"][0].update(id=1, kind="loop_body_safety", function_name="count", smtlib="(assert false)")
        smt["queries"].append({**smt["queries"][0], "id": 2, "kind": "base", "status": "SAT"})
        loops.update(schema_version=1, certificate_required=1, excluded=0)
        loops["loops"][0].update(certificate_required=True, loop_source_op_id=7, loop_kind="scf_while")
        loops["loops"][0]["source"]["file"] = self.source
        loops["loops"][0]["owner"].update(tag="statement", ordinal=7)
        authority.update(schema_version=2, source=self.source, final_artifact_authorization="blocked", artifacts=[
            {"id": "lean_obligations_source", "kind": "proof_source", "owner": "lean_userland", "path": "refusal.lean.obligations.lean"},
            {"id": "source_accounting_report", "kind": "report", "owner": "source_accounting_kernel", "path": "refusal.formal.accounting.json"}])
        authority["gates"][1]["status"] = "rejected"
        authority["gates"].append({"id": "dispatcher_kernel", "status": "not_requested"})
        self.reports[3] = None
        self.reports[4] = {"schema_version": 3, "mode": "verified_full", "decision": "accepted", "primary_failure": None,
                           "failures": [], "inventory_sha256": "a" * 64, "evidence_sha256": "b" * 64,
                           "counts": {"failures": 0, "handlings_by_kind": {"reduced_scope_excluded": 0,
                                                                        "verification_disabled": 0, "rejected": 0}}}
        self.actual["loops"] = corpus.loop_rows(loops)
        self.actual["terminal_errors"] = [f"- query: emittedQuery_5 ({self.source}:2:3)",
                                          "Automatic Lean loop proof unavailable: " + self.reason]
        self.refresh_raw_hash(self.reports)
        self.expected = self.expectation(self.actual, self.reports)

    def refresh_raw_hash(self, reports):
        if reports[2] is not None and reports[4] is not None:
            for artifact in reports[2]["artifacts"]:
                if artifact["id"] == "source_accounting_report":
                    artifact["sha256"] = hashlib.sha256(json.dumps(reports[4]).encode()).hexdigest()

    def expectation(self, actual, reports):
        smt, loops, authority, _, accounting = reports
        return {**copy.deepcopy(actual), "category": "automatic_refusal", "explanation": "test all-target refusal",
                "track": "test-proof-boundary", "automatic_refusal_reason": self.reason,
                "automatic_refusal_evidence": {
                    "smt_schema": smt["schema"] if smt else None,
                    "smt": corpus.rejection_evidence(smt) if smt else None,
                    "settings": smt["settings"] if smt else None,
                    "loops": corpus.relative_paths(loops), "authority": corpus.relative_paths(authority),
                    "accounting_sha256": hashlib.sha256(json.dumps(corpus.relative_paths(accounting), sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
                    "obligations_sha256": hashlib.sha256(corpus.relative_paths(self.module).encode()).hexdigest()}}

    def collect(self, actual=None, reports=None, alter_files=None):
        actual, reports = copy.deepcopy((actual or self.actual, reports or self.reports))
        directory = Path(tempfile.mkdtemp(dir=self.root))
        self.last_directory = directory

        def communicate(**kwargs):
            suffixes = ("smt.report.json", "formal.loops.json", "formal.artifacts.json", "lean.proof.json", "formal.accounting.json")
            for suffix, report in zip(suffixes, reports):
                if report is not None:
                    (directory / ("refusal." + suffix)).write_text(json.dumps(report))
            (directory / "refusal.lean.obligations.lean").write_text(self.module)
            for name in actual["hex"]:
                (directory / name).write_text("00")
            if alter_files:
                alter_files(directory)
            return "\n".join(actual["terminal_errors"]), ""

        process = mock.Mock(returncode=actual["exit"], pid=987654321)
        process.communicate.side_effect = communicate
        with mock.patch.object(corpus.subprocess, "Popen", return_value=process):
            return corpus.run_case(Path("fake-compiler"), actual["source"], directory, 120, capture_automatic_refusal=True)

    def test_collected_refusal_and_no_caller_hash_authority(self):
        actual, reports = self.collect()
        corpus.validate_result(self.expected, actual, reports)
        with self.assertRaisesRegex(ValueError, "fresh collection"):
            corpus.validate_result(self.expected, dict(actual), reports)
        with self.assertRaisesRegex(ValueError, "collection provenance"):
            corpus.validate_result(self.expected, corpus.CollectedRefusal(actual), reports)
        forged = corpus.CollectedRefusal(actual)
        forged.directory, forged.source, forged.exit_code = self.last_directory, self.source, actual["exit"]
        with self.assertRaisesRegex(ValueError, "collection provenance"):
            corpus.validate_result(self.expected, forged, reports)
        (self.last_directory / "refusal.lean.obligations.lean").write_text("def transplanted : Prop := True\n")
        actual["automatic_refusal_evidence"] = copy.deepcopy(self.expected["automatic_refusal_evidence"])
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            corpus.validate_result(self.expected, actual, reports)

    def test_each_exact_pin_is_required(self):
        for key in ("source", "source_sha256", "automatic_refusal_reason", "automatic_refusal_evidence", "track", "explanation"):
            with self.subTest(key=key):
                actual, reports = self.collect()
                expected = copy.deepcopy(self.expected)
                del expected[key]
                with self.assertRaises((ValueError, KeyError)):
                    corpus.validate_result(expected, actual, reports)
        for key in self.expected["automatic_refusal_evidence"]:
            with self.subTest(evidence=key):
                actual, reports = self.collect()
                expected = copy.deepcopy(self.expected)
                expected["automatic_refusal_evidence"][key] = "changed"
                with self.assertRaisesRegex(ValueError, "evidence changed"):
                    corpus.validate_result(expected, actual, reports)

    def test_exact_pin_mutations(self):
        mutations = {
            "owner": lambda a, r: r[1]["loops"][0]["owner"].update(function_name="foreign"),
            "source_op": lambda a, r: r[1]["loops"][0].update(loop_source_op_id=99),
            "loop_kind": lambda a, r: r[1]["loops"][0].update(loop_kind="other"),
            "loop_id": lambda a, r: r[1]["loops"][0].update(id=99),
            "target": lambda a, r: r[1]["loops"][0].update(induction_query_id=99),
            "site": lambda a, r: r[1]["loops"][0]["source"].update(line=99),
            "query_id": lambda a, r: r[0]["queries"][0].update(id=99),
            "query_owner": lambda a, r: r[0]["queries"][0].update(function_name="foreign"),
            "query_kind": lambda a, r: r[0]["queries"][0].update(kind="obligation"),
            "query_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash="0x5678"),
            "query_constraints": lambda a, r: r[0]["queries"][0].update(smtlib="(assert true)"),
            "query_order": lambda a, r: r[0]["queries"].reverse(),
            "settings": lambda a, r: r[0]["settings"].update(timeout_ms=999),
            "accounting_body": lambda a, r: r[4].update(inventory_sha256="c" * 64),
            "artifact_registry": lambda a, r: r[2]["artifacts"].append({"id": "extra"}),
            "terminal": lambda a, r: a["terminal_errors"].append("error: another failure"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                a, r = copy.deepcopy((self.actual, self.reports))
                mutate(a, r)
                self.refresh_raw_hash(r)
                actual, reports = self.collect(a, r)
                with self.assertRaises((ValueError, KeyError)):
                    corpus.validate_result(self.expected, actual, reports)

    def test_matching_pins_cannot_excuse_fatal_states(self):
        mutations = {
            "timeout": lambda a, r: a.update(exit="timeout"),
            "crash": lambda a, r: a.update(exit=-11),
            "ice": lambda a, r: a.update(exit=2),
            "exit_zero": lambda a, r: a.update(exit=0),
            "bytecode": lambda a, r: a.update(hex=["forged.hex"]),
            "missing_smt": lambda a, r: r.__setitem__(0, None),
            "missing_loops": lambda a, r: r.__setitem__(1, None),
            "missing_authority": lambda a, r: r.__setitem__(2, None),
            "missing_accounting": lambda a, r: r.__setitem__(4, None),
            "proof_checked": lambda a, r: r.__setitem__(3, {"status": "checked"}),
            "proof_partial": lambda a, r: r.__setitem__(3, {"status": "partial"}),
            "proof_empty": lambda a, r: r.__setitem__(3, {}),
            "schema": lambda a, r: r[0].update(schema="unknown"),
            "foreign_smt": lambda a, r: r[0].update(source_file="ora-example/foreign.ora"),
            "foreign_authority": lambda a, r: r[2].update(source="ora-example/foreign.ora"),
            "foreign_loop": lambda a, r: r[1]["loops"][0]["source"].update(file="ora-example/foreign.ora"),
            "allowed": lambda a, r: r[2].update(final_artifact_authorization="allowed"),
            "unknown_authority": lambda a, r: r[2].update(final_artifact_authorization="unknown"),
            "z3_rejected": lambda a, r: r[2]["gates"][0].update(status="rejected"),
            "lean_accepted": lambda a, r: r[2]["gates"][1].update(status="accepted"),
            "lean_missing": lambda a, r: r[2]["gates"].pop(1),
            "gate_duplicate": lambda a, r: r[2]["gates"].append(copy.deepcopy(r[2]["gates"][0])),
            "gate_duplicate_same_count": lambda a, r: r[2]["gates"].__setitem__(1, copy.deepcopy(r[2]["gates"][0])),
            "accounting_gate": lambda a, r: r[2]["gates"][2].update(status="rejected"),
            "z3_false": lambda a, r: r[0]["verification"].update(success=False),
            "degraded": lambda a, r: r[0]["verification"].update(verification_trust="degraded"),
            "unknown": lambda a, r: r[0]["summary"].update(unknown=1),
            "vacuous": lambda a, r: r[0]["summary"].update(vacuous=1),
            "vacuity_unknown": lambda a, r: r[0]["summary"].update(vacuity_unknown=1),
            "inconsistent": lambda a, r: r[0]["summary"].update(inconsistent_bases=1),
            "encoding_degraded": lambda a, r: r[0]["summary"].update(encoding_degraded=True),
            "soundness_loss": lambda a, r: r[0]["summary"].update(soundness_losses=["lost fact"]),
            "failed_obligation": lambda a, r: r[0]["summary"].update(failed_obligations=1),
            "smt_error": lambda a, r: r[0].update(errors=[{"type": "failure"}]),
            "smt_diagnostic": lambda a, r: r[0].update(diagnostics=[{"message": "failure"}]),
            "sat_safety": lambda a, r: r[0]["queries"][0].update(status="SAT"),
            "unknown_query": lambda a, r: r[0]["queries"][0].update(status="UNKNOWN"),
            "query_vacuous": lambda a, r: r[0]["queries"][0].update(vacuous=True),
            "query_vacuity_unknown": lambda a, r: r[0]["queries"][0].update(vacuity_unknown=True),
            "query_caveat": lambda a, r: r[0]["queries"][0].update(verified_with_caveats=True),
            "zero_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash="0x0"),
            "malformed_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash="invalid"),
            "missing_hash": lambda a, r: r[0]["queries"][0].update(smtlib_hash=None),
            "empty_queries": lambda a, r: r[0].update(queries=[]),
            "query_duplicate": lambda a, r: r[0]["queries"].append(copy.deepcopy(r[0]["queries"][0])),
            "unknown_kind": lambda a, r: r[0]["queries"][0].update(kind="unknown"),
            "unsat_base": lambda a, r: r[0]["queries"][1].update(status="UNSAT"),
            "accounting_rejected": lambda a, r: r[4].update(decision="rejected"),
            "accounting_mode": lambda a, r: r[4].update(mode="reduced"),
            "accounting_failure": lambda a, r: r[4].update(failures=["missing_symbolic"]),
            "accounting_primary": lambda a, r: r[4].update(primary_failure="missing_symbolic"),
            "accounting_count": lambda a, r: r[4]["counts"].update(failures=1),
            "accounting_reduced": lambda a, r: r[4]["counts"]["handlings_by_kind"].update(reduced_scope_excluded=1),
            "accounting_disabled": lambda a, r: r[4]["counts"]["handlings_by_kind"].update(verification_disabled=1),
            "accounting_handling": lambda a, r: r[4]["counts"]["handlings_by_kind"].update(rejected=1),
            "accounting_digest": lambda a, r: r[4].update(inventory_sha256="invalid"),
            "loop_empty": lambda a, r: r[1].update(loops=[], total=0, certificate_required=0),
            "loop_excluded": lambda a, r: r[1]["loops"][0].update(excluded_by=["loop_backedge_missing"]),
            "loop_partial": lambda a, r: r[1]["loops"][0].update(projection_completeness="partial"),
            "loop_unrequired": lambda a, r: r[1]["loops"][0].update(certificate_required=False),
            "loop_status": lambda a, r: r[1]["loops"][0].update(status="not_verified"),
            "loop_invalid_id": lambda a, r: r[1]["loops"][0].update(id=0),
            "target_invalid": lambda a, r: r[1]["loops"][0].update(induction_query_id=False),
            "missing_reason": lambda a, r: a.update(terminal_errors=a["terminal_errors"][:1]),
            "mixed_reason": lambda a, r: a["terminal_errors"].append("Automatic Lean loop proof unavailable: automatic_term_kind_unsupported"),
            "unknown_reason": lambda a, r: a["terminal_errors"].__setitem__(1, "Automatic Lean loop proof unavailable: unknown_reason"),
            "missing_recipe": lambda a, r: a.update(terminal_errors=a["terminal_errors"][1:]),
            "duplicate_recipe": lambda a, r: a["terminal_errors"].append(a["terminal_errors"][0]),
            "foreign_recipe": lambda a, r: a["terminal_errors"].__setitem__(0, "- query: emittedQuery_5 (foreign.ora:2:3)"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                a, r = copy.deepcopy((self.actual, self.reports))
                mutate(a, r)
                self.refresh_raw_hash(r)
                actual, reports = self.collect(a, r)
                expected = self.expectation(actual, reports)  # Forge every pin to match the degraded data.
                if name == "unknown_reason":
                    expected["automatic_refusal_reason"] = "unknown_reason"
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    corpus.validate_result(expected, actual, reports)

    def test_collector_path_and_artifact_corruption(self):
        changes = {
            "wrong_owner": lambda d: None,
            "wrong_kind": lambda d: None,
            "absolute": lambda d: None,
            "traversal": lambda d: None,
            "artifact_duplicate": lambda d: None,
            "accounting_digest": lambda d: None,
            "missing": lambda d: (d / "refusal.lean.obligations.lean").unlink(),
            "duplicate": lambda d: (d / "extra.lean.obligations.lean").write_text(self.module),
            "symlink": lambda d: ((d / "refusal.lean.obligations.lean").unlink(),
                                   (d / "refusal.lean.obligations.lean").symlink_to(self.root / self.source)),
            "body": lambda d: (d / "refusal.lean.obligations.lean").write_text("def foreign : Prop := True\n"),
        }
        for name, alter in changes.items():
            with self.subTest(name=name):
                r = copy.deepcopy(self.reports)
                artifact = r[2]["artifacts"][0]
                if name in ("wrong_owner", "wrong_kind"):
                    artifact["owner" if name == "wrong_owner" else "kind"] = "foreign"
                elif name in ("absolute", "traversal"):
                    artifact["path"] = "/tmp/foreign.lean" if name == "absolute" else "../foreign.lean"
                elif name == "artifact_duplicate":
                    r[2]["artifacts"].append(copy.deepcopy(artifact))
                elif name == "accounting_digest":
                    r[2]["artifacts"][1]["sha256"] = "0" * 64
                actual, reports = self.collect(reports=r, alter_files=alter)
                expected = self.expectation(actual, reports)
                with self.assertRaisesRegex(ValueError, "artifact|refusal file|evidence changed"):
                    corpus.validate_result(expected, actual, reports)

    def test_reused_directory_and_forged_observation(self):
        actual, reports = self.collect()
        with self.assertRaisesRegex(ValueError, "empty owned directory"), mock.patch.object(corpus.subprocess, "Popen") as spawn:
            corpus.run_case(Path("fake"), self.source, self.last_directory, 120, capture_automatic_refusal=True)
        spawn.assert_not_called()
        actual["source_sha256"] = "f" * 64
        expected = self.expectation(actual, reports)
        with self.assertRaisesRegex(ValueError, "collected source changed"):
            corpus.validate_result(expected, actual, reports)
        actual["source_sha256"] = self.actual["source_sha256"]
        actual["terminal_errors"].append("error: forged")
        with self.assertRaisesRegex(ValueError, "observation changed"):
            corpus.validate_result(self.expectation(actual, reports), actual, reports)


    def test_all_targets_and_distinct_query_namespaces(self):
        a, r = copy.deepcopy((self.actual, self.reports))
        second = copy.deepcopy(r[1]["loops"][0])
        second.update(id=2, induction_query_id=2)  # Raw SMT query 2 is a base, not this Lean target.
        second["owner"]["function_name"] = "another_function"
        second["source"]["line"] = 9
        r[1]["loops"].append(second)
        r[1].update(total=2, certificate_required=2)
        a["terminal_errors"] += [f"- query: emittedQuery_2 ({self.source}:9:3)", a["terminal_errors"][1]]
        actual, reports = self.collect(a, r)
        expected = self.expectation(actual, reports)
        corpus.validate_result(expected, actual, reports)
        for key in ("id", "induction_query_id"):
            with self.subTest(key=key):
                duplicated = copy.deepcopy(r)
                duplicated[1]["loops"][1][key] = duplicated[1]["loops"][0][key]
                actual, reports = self.collect(a, duplicated)
                with self.assertRaisesRegex(ValueError, "duplicate refusal loop"):
                    corpus.validate_result(self.expectation(actual, reports), actual, reports)
        for index in (2, 3):
            with self.subTest(missing_terminal=index):
                incomplete = copy.deepcopy(a)
                incomplete["terminal_errors"].pop(index)
                actual, reports = self.collect(incomplete, r)
                with self.assertRaisesRegex(ValueError, "refusals|recipe"):
                    corpus.validate_result(self.expectation(actual, reports), actual, reports)
        for kind in ("id", "owner", "site", "order"):
            with self.subTest(loop_pin=kind):
                changed = copy.deepcopy(r)
                if kind == "order":
                    changed[1]["loops"].reverse()
                elif kind == "id":
                    changed[1]["loops"][1]["id"] = 3
                elif kind == "owner":
                    changed[1]["loops"][1]["owner"]["ordinal"] += 1
                else:
                    changed[1]["loops"][1]["source"]["column"] += 1
                actual, reports = self.collect(a, changed)
                with self.assertRaises(ValueError):
                    corpus.validate_result(expected, actual, reports)

    def test_additional_collection_corruptions(self):
        def symlink_report(directory):
            path = directory / "refusal.formal.accounting.json"
            target = self.root / "foreign-accounting.json"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
        changes = {
            "report_symlink": symlink_report,
            "duplicate_report": lambda d: (d / "extra.formal.loops.json").write_text(json.dumps(self.reports[1])),
            "malformed_report": lambda d: (d / "refusal.formal.loops.json").write_text("{"),
        }
        for name, change in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                actual, reports = self.collect(alter_files=change)
                corpus.validate_result(self.expected, actual, reports)
        actual, reports = self.collect()
        (self.last_directory / "forged.hex").write_text("00")
        with self.assertRaisesRegex(ValueError, "collected outcome changed"):
            corpus.validate_result(self.expected, actual, reports)
        actual, reports = self.collect()
        reports[0]["queries"][0]["smtlib"] = "(assert true)"
        with self.assertRaisesRegex(ValueError, "reports changed after collection"):
            corpus.validate_result(self.expectation(actual, reports), actual, reports)
        actual, reports = self.collect()
        actual["loops"][0]["function"] = "forged"
        with self.assertRaisesRegex(ValueError, "forged refusal loop observation"):
            corpus.validate_result(self.expectation(actual, reports), actual, reports)

    def test_old_categories_and_refusal_counts_stay_distinct(self):
        actual, reports = self.collect()
        for category in ("accepted", "excluded", "unsafe", "verification_negative", "unsupported_category"):
            with self.subTest(category=category), self.assertRaises((ValueError, KeyError)):
                corpus.validate_result({**self.expected, "category": category, "artifacts": 1}, actual, reports)
        # 'invalid' is still the old source-error policy; AR never routes through it.
        accepted = CorpusGateTests()
        accepted.setUp()
        accepted.actual["source"] = "ora-example/accepted.ora"
        (self.root / accepted.actual["source"]).write_text("// accepted test control\n")
        corpus.validate_result(accepted.expected, accepted.actual, accepted.reports)
        with self.assertRaisesRegex(ValueError, "unexpected acceptance"):
            expected = {**accepted.expected, "category": "automatic_refusal", "track": "future-success-tripwire"}
            corpus.validate_result(expected, accepted.actual, accepted.reports)
        manifest = {"schema": 1, "open_findings": [], "files": {
            self.source: self.expected, accepted.actual["source"]: accepted.expected}}
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        compiler = self.root / "compiler"
        compiler.write_text("test compiler identity")
        captured = io.StringIO()
        with mock.patch("sys.argv", ["checker", "--manifest", str(path), "--compiler", str(compiler),
                                      "--output", str(self.root / "main-output")]), \
             mock.patch.object(corpus.subprocess, "check_output", side_effect=["\n".join(manifest["files"]), ""]), \
             mock.patch.object(corpus, "run_case", side_effect=[(actual, reports), (accepted.actual, accepted.reports)]) as run, \
             mock.patch("sys.stdout", captured):
            corpus.main()
        self.assertTrue(run.call_args_list[0].kwargs["capture_automatic_refusal"])
        self.assertFalse(run.call_args_list[1].kwargs["capture_automatic_refusal"])
        self.assertIn('"accepted": 1, "automatic_refusal": 1', captured.getvalue())
        self.assertIn("checked loops in accepted artifacts: 1", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
