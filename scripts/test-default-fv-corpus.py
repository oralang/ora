#!/usr/bin/env python3
"""Fault injection for the corpus harness, not a compiler/prover fault-injection mode."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
