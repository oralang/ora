#!/usr/bin/env python3
"""Default FV regression gate. Expectations are reviewed data, never generated here."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/default-fv-corpus.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_report(directory, suffix):
    paths = list(directory.glob("*." + suffix))
    require(len(paths) <= 1, "ambiguous report: " + suffix)
    return json.loads(paths[0].read_text()) if paths else None


def terminal_errors(text):
    # Compare complete diagnostic lines; never filename/substring acceptance rules.
    text = re.sub(r"\x1b\[[0-9;]*m", "", text).replace(str(ROOT) + "/", "")
    return sorted(line.strip() for line in text.splitlines()
                  if re.match(r"^(?:error:|Compiler error:|Automatic Lean .* unavailable:|- [a-z_]+:)", line.strip()))


def loop_rows(report):
    if report is None:
        return []
    rows = []
    for loop in report["loops"]:
        source = loop["source"]
        rows.append({"file": source["file"].replace(str(ROOT) + "/", ""),
                     "function": loop["owner"]["function_name"],
                     "line": source["line"], "column": source["column"],
                     "status": loop["status"], "reasons": sorted(loop["excluded_by"])})
    require(report["total"] == len(rows), "loop census count mismatch")
    return rows


def relative_paths(value):
    """Only checkout-root paths vary; preserve all other report content."""
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "")
    if isinstance(value, list):
        return [relative_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: relative_paths(item) for key, item in value.items()}
    return value


def smt_errors(report):
    if report is None:
        return []
    return relative_paths([{**{key: row.get(key) for key in ("type", "message", "line", "column", "subtype")},
                            "file": row.get("file") or None}
                           for row in report["errors"]])


def smt_diagnostics(report):
    if report is None:
        return []
    return [{key: value.replace(str(ROOT) + "/", "") if isinstance(value, str) else value
             for key, value in row.items() if key != "counterexample"} for row in report["diagnostics"]]


def rejection_evidence(report):
    return relative_paths({key: report[key] for key in ("verification", "summary", "queries")})


def validate_verification_negative(expected, actual, smt, authority):
    # This records a particular refused verification, never an accepted proof.
    require(actual["source"] == expected["source"], "negative source changed")
    require(re.fullmatch(r"[0-9a-f]{64}", expected["source_sha256"] or "") and
            actual["source_sha256"] == expected["source_sha256"], "negative source hash changed")
    require(relative_paths(smt["source_file"]) == actual["source"] and
            relative_paths(authority["source"]) == actual["source"], "foreign rejection report")
    require(smt["verification"]["success"] is False, "negative verification succeeded")
    require(smt["verification"]["verification_trust"] in ("full", "degraded"), "unknown negative trust")
    require(smt["errors"] and all(row["type"] for row in smt["errors"]), "missing named negative error")
    require(authority["final_artifact_authorization"] == "blocked", "negative authority not blocked")
    gates = [gate for gate in authority["gates"] if gate["id"] == "z3_userland"]
    require(len(gates) == 1 and gates[0]["status"] == "rejected", "missing/ambiguous rejected Z3 gate")
    for query in smt["queries"]:
        digest = query["smtlib_hash"]
        require(re.fullmatch(r"0x[0-9a-f]+", digest or "") and int(digest, 16) != 0, "invalid negative solver hash")
    require(rejection_evidence(smt) == expected["rejection_evidence"], "negative evidence changed")


class CollectedRefusal(dict):
    """An in-process run with owned output paths; serialized observations are not receipts."""


_refusal_collections = {}  # Retain run identities/paths for this harness process, never from JSON.


def automatic_refusal_evidence(actual, reports):
    require(type(actual) is CollectedRefusal, "automatic refusal requires fresh collection")
    collected = _refusal_collections.get(id(actual))
    require(collected is not None and collected[0] is actual, "missing collection provenance")
    _, directory, collected_source, exit_code = collected
    require(not directory.is_symlink(), "symlinked evidence directory")
    suffixes = ("smt.report.json", "formal.loops.json", "formal.artifacts.json", "lean.proof.json", "formal.accounting.json")
    for suffix in suffixes:
        paths = list(directory.glob("*." + suffix))
        require(all(p.is_file() and not p.is_symlink() for p in paths), "foreign report file")
    require(tuple(read_report(directory, suffix) for suffix in suffixes) == tuple(reports), "reports changed after collection")
    smt, loops, authority, _, accounting = reports
    require(all(r is not None for r in (smt, loops, authority, accounting)), "missing automatic-refusal report")
    require(actual["source"] == collected_source and actual["source_sha256"] == sha(ROOT / collected_source), "collected source changed")
    text = "\n".join((directory / name).read_text() for name in ("stdout.txt", "stderr.txt"))
    require(actual["terminal_errors"] == terminal_errors(text) and actual["loops"] == loop_rows(loops) and
            actual["smt_errors"] == smt_errors(smt) and actual["smt_diagnostics"] == smt_diagnostics(smt), "observation changed")
    require(actual["exit"] == exit_code and not list(directory.rglob("*.hex")), "collected outcome changed")
    stem = Path(collected_source).stem
    blobs = {}
    for identity, suffix, kind, owner in (("lean_obligations_source", "lean.obligations.lean", "proof_source", "lean_userland"),
                                          ("source_accounting_report", "formal.accounting.json", "report", "source_accounting_kernel")):
        matches = [a for a in authority["artifacts"] if a["id"] == identity]
        require(len(matches) == 1, "missing/duplicate refusal artifact")
        artifact = matches[0]
        require((artifact["kind"], artifact["owner"], artifact["path"]) == (kind, owner, stem + "." + suffix), "foreign refusal artifact")
        path = directory / artifact["path"]
        require(list(directory.glob("*." + suffix)) == [path] and path.is_file() and not path.is_symlink() and
                path.resolve().parent == directory.resolve(), "missing/duplicate/foreign refusal file")
        blobs[identity] = path.read_bytes()
        if identity == "source_accounting_report":
            require(artifact["sha256"] == hashlib.sha256(blobs[identity]).hexdigest(), "accounting artifact hash mismatch")
    encoded = json.dumps(relative_paths(accounting), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"smt_schema": smt["schema"], "smt": rejection_evidence(smt), "settings": smt["settings"],
            "loops": relative_paths(loops), "authority": relative_paths(authority),
            "accounting_sha256": hashlib.sha256(encoded).hexdigest(),
            "obligations_sha256": hashlib.sha256(relative_paths(blobs["lean_obligations_source"].decode()).encode()).hexdigest()}


def validate_automatic_refusal(expected, actual, reports):
    smt, loops, authority, proof, accounting = reports
    require(all(r is not None for r in (smt, loops, authority, accounting)), "missing automatic-refusal report")
    require(proof is None, "automatic preflight refusal has proof report")
    require((smt["schema"], loops["schema_version"], authority["schema_version"], accounting["schema_version"]) ==
            ("ora.smt.report.v2", 1, 3, 3), "unsupported refusal report schema")
    require(actual["source"] == expected["source"] and
            re.fullmatch(r"[0-9a-f]{64}", expected["source_sha256"] or "") and
            actual["source_sha256"] == expected["source_sha256"], "automatic-refusal source changed")
    require(relative_paths(smt["source_file"]) == actual["source"] == relative_paths(authority["source"]), "foreign refusal report")
    require(authority["final_artifact_authorization"] == "blocked", "automatic refusal not blocked")
    gates = authority["gates"]
    statuses = {"z3_userland": "accepted", "source_accounting_kernel": "accepted",
                "lean_userland": "rejected", "dispatcher_kernel": "not_requested", "lifecycle_kernel": "not_requested"}
    require(len(gates) == len(statuses) and len({g["id"] for g in gates}) == len(gates) and
            {g["id"]: g["status"] for g in gates} == statuses, "invalid automatic-refusal gates")
    require(smt["verification"]["success"] is True and not smt["errors"] and not smt["diagnostics"] and
            not smt["summary"]["failed_obligations"], "automatic refusal requires successful Z3")
    require(not any(q.get("lifecycle_use") is not None for q in smt["queries"]) and
            not any(a["id"] == "storage_lifecycle_report" for a in authority["artifacts"]), "assisted automatic-refusal path changed")
    queries = smt["queries"]
    kinds = {"base": "SAT", "guard_satisfy": "SAT", "obligation": "UNSAT", "loop_body_safety": "UNSAT",
             "loop_invariant_step": "UNSAT", "guard_violate": "UNSAT"}
    require(queries and len({q["id"] for q in queries}) == len(queries), "empty/duplicate refusal queries")
    require(any(q["kind"] == "base" for q in queries) and
            any(q["kind"] in ("obligation", "loop_body_safety", "loop_invariant_step") for q in queries), "missing real refusal queries")
    for query in queries:
        require(query["kind"] in kinds and query["status"] == kinds[query["kind"]], "failed/unsupported refusal query")
        require(type(query["id"]) is int and query["id"] > 0 and
                re.fullmatch(r"0x[0-9a-f]+", query["smtlib_hash"] or "") and int(query["smtlib_hash"], 16) != 0, "invalid refusal query identity/hash")
    require(accounting["mode"] == "verified_full" and accounting["decision"] == "accepted" and
            accounting["primary_failure"] is None and not accounting["failures"] and
            accounting["counts"]["failures"] == 0, "automatic refusal requires full accounting")
    require(not any(accounting["counts"]["handlings_by_kind"][k] for k in
                    ("reduced_scope_excluded", "verification_disabled", "rejected")), "reduced refusal accounting")
    for key in ("inventory_sha256", "evidence_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", accounting[key] or ""), "invalid accounting digest")
    rows = loops["loops"]
    require(loops["total"] == loops["certificate_required"] == len(rows) > 0 and loops["excluded"] == 0, "incomplete refusal census")
    for key in ("id", "induction_query_id"):
        require(all(type(row[key]) is int and row[key] > 0 for row in rows) and
                len({row[key] for row in rows}) == len(rows), "invalid/duplicate refusal loop identity")
    recipes = []
    for row in rows:
        source = row["source"]
        require(row["projection_completeness"] == "complete" and row["status"] == "lean_certificate_required" and
                row["certificate_required"] is True and not row["excluded_by"] and row["owner"]["function_name"] and
                relative_paths(source["file"]) == actual["source"], "incomplete/foreign refusal loop")
        recipes.append(f"- query: emittedQuery_{row['induction_query_id']} ({actual['source']}:{source['line']}:{source['column']})")
    reason = expected["automatic_refusal_reason"]
    require(reason in ("automatic_nonlinear_arithmetic", "automatic_term_kind_unsupported"), "unrecognized automatic refusal")
    terminal = actual["terminal_errors"]
    require(sorted(line for line in terminal if line.startswith("Automatic Lean ")) ==
            ["Automatic Lean loop proof unavailable: " + reason] * len(rows), "mixed/incomplete automatic refusals")
    require(sorted(line for line in terminal if line.startswith("- query:")) == sorted(recipes), "automatic refusal recipe mismatch")
    require(loop_rows(loops) == actual["loops"], "forged refusal loop observation")
    require(automatic_refusal_evidence(actual, reports) == expected["automatic_refusal_evidence"], "automatic refusal evidence changed")


_lifecycle_collections = {}


def validate_lifecycle(actual, reports):
    smt, _, authority, _, _ = reports
    require(smt["schema"] == "ora.smt.report.v2" and authority["schema_version"] == 3, "unsupported lifecycle report schema")
    gates = [g for g in authority["gates"] if g["id"] == "lifecycle_kernel"]
    require(len(gates) == 1, "missing/duplicate lifecycle gate")
    uses = [q["lifecycle_use"] for q in smt["queries"] if q.get("lifecycle_use") is not None]
    artifacts = [a for a in authority["artifacts"] if a["id"] == "storage_lifecycle_report"]
    if gates[0]["status"] == "not_requested":
        require(not uses and not artifacts, "unrequested lifecycle assistance")
        return
    require(gates[0]["status"] == "accepted" and uses and len(artifacts) == 1, "unchecked lifecycle assistance")
    collected = _lifecycle_collections.get(id(actual))
    require(collected is not None and collected[0] is actual, "missing lifecycle collection provenance")
    _, directory, source, exit_code, sealed = collected
    require(actual["source"] == source and actual["exit"] == exit_code and actual["source_sha256"] == sha(ROOT / source), "lifecycle source/outcome changed")
    require(not directory.is_symlink(), "foreign lifecycle directory")
    suffixes = ("smt.report.json", "formal.loops.json", "formal.artifacts.json", "lean.proof.json", "formal.accounting.json")
    require(tuple(read_report(directory, suffix) for suffix in suffixes) == tuple(reports), "lifecycle reports changed")
    current = {p.name: sha(p) for p in directory.glob("*.json") if p.name != "run.json" and p.is_file() and not p.is_symlink()}
    require(current == sealed, "lifecycle evidence changed after collection")
    artifact = artifacts[0]
    filename = Path(source).stem + ".formal.lifecycle.json"
    require((artifact["kind"], artifact["owner"], artifact["path"]) ==
            ("report", "lifecycle_kernel", filename), "foreign lifecycle artifact")
    path = directory / filename
    require(list(directory.glob("*.formal.lifecycle.json")) == [path] and path.is_file() and not path.is_symlink() and
            path.resolve().parent == directory.resolve() and artifact["sha256"] == sha(path), "lifecycle artifact custody mismatch")
    report = json.loads(path.read_text())
    require(report["schema_version"] == 1 and relative_paths(report["source"]) == source, "foreign lifecycle report")
    evidence = report["evidence"]
    def digest(value):
        return isinstance(value, list) and len(value) == 32 and all(type(b) is int and 0 <= b <= 255 for b in value) and any(value)
    for key in ("inputs_sha256", "native_inputs_sha256", "lean_inputs_sha256", "module_sha256", "targets_sha256", "proof_sha256", "checker_sha256"):
        require(digest(evidence[key]), "invalid lifecycle digest: " + key)
    require(evidence["inputs_sha256"] == evidence["native_inputs_sha256"] == evidence["lean_inputs_sha256"], "cross-prover lifecycle mismatch")
    require(evidence["retained_checks"] is True and set(evidence["axioms"]) <= {"propext", "Quot.sound"}, "unretained checks or unchecked lifecycle axioms")
    native = evidence["native_queries"]
    keys = evidence["native_keys"]
    selectors = ("role", "claim_index", "function_id", "event_id", "function_vc_sha256")
    require(native and collections.Counter(json.dumps({k: q[k] for k in selectors}, sort_keys=True) for q in native) == collections.Counter(json.dumps(k, sort_keys=True) for k in keys) and
            len({json.dumps(k, sort_keys=True) for k in keys}) == len(keys), "incomplete lifecycle native coverage")
    for query in native:
        require(query["role"] in ("bootstrap", "prefix", "frame", "nonvacuity") and query["smtlib_hash"] > 0 and query["constraint_count"] > 0 and query["base_status"] == "sat" and
                query["status"] == ("sat" if query["role"] == "nonvacuity" else "unsat"), "failed lifecycle native proof")
    require(evidence["functions"] and evidence["slots"] and evidence["expected_uses"] == uses and
            [r["use"] for r in evidence["checked_uses"]] == uses, "incomplete lifecycle use ledger")
    for use, checked in zip(uses, evidence["checked_uses"]):
        dependency = use["dependency"]
        query = smt["queries"][use["query_index"]]
        require(query["id"] == use["query_index"] + 1 and query["lifecycle_use"] == use and
                int(query["smtlib_hash"], 16) == use["smtlib_hash"] and query["constraint_count"] == use["constraint_count"], "foreign lifecycle query use")
        require(dependency["inputs_sha256"] == evidence["inputs_sha256"] and dependency["module_sha256"] == evidence["module_sha256"] and
                dependency["claim_count"] == len(evidence["slots"]) and dependency["owner_ordinal"] == evidence["slots"][0]["owner_ordinal"] and
                any(all(f[k] == dependency[k] for k in ("function_id", "function_ordinal", "function_vc_sha256")) for f in evidence["functions"]), "foreign lifecycle entry")
        require(not any(checked[k] for k in ("vacuous", "vacuity_unknown", "degraded")) and checked["status"].upper() == query["status"], "degraded lifecycle use")
        require(checked["status"] == ("sat" if use["kind"] in ("base", "guard_satisfy") else "unsat") or
                (use["kind"] == "guard_violate" and checked["status"] == "sat" and checked["runtime_input"] is True), "failed lifecycle use")


def validate_capability_refusal(expected, actual, reports, open_findings):
    # A pinned missing capability is an open finding, never a checked proof.
    smt, _, authority, _, _ = reports
    source, digest = expected["source"], expected["source_sha256"]
    require(actual["source"] == source and actual["source_sha256"] == digest and
            re.fullmatch(r"[0-9a-f]{64}", digest or "") and int(digest, 16) != 0, "capability source changed")
    findings = [f for f in open_findings if f["id"] == expected["open_finding"]]
    require(len(findings) == 1 and findings[0]["id"].partition("/")[2] == expected["track"] and
            findings[0]["sources"].get(source) == digest and findings[0]["explanation"], "missing/mismatched capability finding")
    require(relative_paths(smt["source_file"]) == source and relative_paths(authority["source"]) == source,
            "foreign capability report")
    require(authority["final_artifact_authorization"] == "blocked", "capability authority not blocked")
    require(actual["terminal_errors"] and actual["smt_errors"] == smt_errors(smt) and
            actual["smt_diagnostics"] == smt_diagnostics(smt), "missing/inconsistent capability diagnostics")
    require(not any(e["type"] == "EncodingDegraded" for e in smt["errors"]) and
            not smt["summary"].get("degradation_reason") and not smt["summary"].get("degradation_reasons"),
            "degraded capability report")
    gates = [g for g in authority["gates"] if g["id"] == "lifecycle_kernel"]
    require(len(gates) == 1 and gates[0]["status"] == "not_requested" and
            not gates[0].get("produced_artifacts") and
            not any(a["id"] == "storage_lifecycle_report" or a["owner"] == "lifecycle_kernel" for a in authority["artifacts"]),
            "capability refusal cannot use lifecycle assistance")
    for query in smt["queries"]:
        require(query.get("lifecycle_use") is None, "capability refusal has a lifecycle use")
        require(re.fullmatch(r"0x[0-9a-f]+", query["smtlib_hash"] or "") and int(query["smtlib_hash"], 16) != 0,
                "invalid capability solver hash")


def validate_result(expected, actual, reports, open_findings=()):
    """Exact refusal pins may describe degradation, never UNKNOWN or artifact authority."""
    smt, loops, authority, proof, accounting = reports
    category = expected["category"]
    require(category in ("accepted", "excluded", "invalid", "unsafe", "verification_negative", "automatic_refusal", "capability_refusal"), "unclassified expectation")
    require(actual["exit"] in (0, 1), "crash, signal, or timeout")
    if smt is not None:
        # Inconclusive vacuity can appear only in errors, without a flagged query.
        require(not any(row["type"] == "Unknown" for row in smt["errors"]), "unknown solver error")
        summary = smt["summary"]
        for key in ("unknown", "vacuous", "vacuity_unknown"):
            require(not summary[key], "degraded solver state: " + key)
        if category != "verification_negative":
            require(smt["verification"]["verification_trust"] == "full", "degraded verification trust")
            for key in ("inconsistent_bases", "encoding_degraded"):
                require(not summary[key], "degraded solver state: " + key)
            require(not summary["soundness_losses"], "solver soundness loss")
        for query in smt["queries"]:
            require(query["status"] in ("SAT", "UNSAT"), "unknown query")
            require(query["smtlib_hash"], "missing solver hash")
            require(not any(query[k] for k in ("vacuous", "vacuity_unknown", "verified_with_caveats")),
                    "degraded query")
    require(actual["loops"] == expected["loops"], "loop identity/status/reason changed")
    require(actual["terminal_errors"] == expected["terminal_errors"], "terminal diagnostic changed")
    require(actual["smt_errors"] == expected["smt_errors"], "SMT diagnostic changed")
    require(actual["smt_diagnostics"] == expected["smt_diagnostics"], "SMT auxiliary diagnostic changed")
    require(expected["explanation"], "missing expectation justification")
    if category != "accepted":
        require(expected.get("track"), "rejection requires a named track")
        require(actual["exit"] == 1 and not actual["hex"], "unexpected acceptance / bytecode on rejection")
        if category != "invalid":
            require(smt is not None and authority is not None, "missing rejection report")
            require(authority["final_artifact_authorization"] != "allowed", "rejected source authorized")
        if category == "verification_negative":
            validate_verification_negative(expected, actual, smt, authority)
        if category == "automatic_refusal":
            validate_automatic_refusal(expected, actual, reports)
        if category == "capability_refusal":
            validate_capability_refusal(expected, actual, reports, open_findings)
        if category == "excluded":
            require(loops is not None and any(row["reasons"] for row in actual["loops"]), "missing named exclusion")
            require(not smt["errors"], "a loop exclusion cannot hide a failed source check")
        if category == "unsafe":
            require(smt is not None and smt["errors"], "missing unsafe-source counterexample")
        return
    require(actual["exit"] == 0, "accepted source rejected")
    require(len(actual["hex"]) == expected["artifacts"] and expected["artifacts"] > 0, "artifact count changed")
    require(smt is not None and authority is not None and accounting is not None, "missing authority evidence")
    require(smt["verification"]["success"] and smt["verification"]["verification_trust"] == "full", "Z3 not accepted")
    require(not smt["errors"] and not actual["terminal_errors"], "unexpected accepted-source error")
    # A retained refinement guard may fail at runtime. Bind its diagnostic to
    # the real SAT guard query; it is not a failed assertion or a proof waiver.
    for diagnostic in smt["diagnostics"]:
        matches = [q for q in smt["queries"] if q["id"] == diagnostic["query_id"]]
        require(len(matches) == 1, "missing/ambiguous diagnostic query")
        query = matches[0]
        require(query["kind"] == "guard_violate" and query["status"] == "SAT" and
                query["guard_id"] == diagnostic["guard_id"], "unbound retained-guard diagnostic")
    require(authority["final_artifact_authorization"] == "allowed", "artifact not authorized")
    gates = {g["id"]: g["status"] for g in authority["gates"]}
    require(gates.get("z3_userland") == "accepted", "Z3 gate not accepted")
    require(gates.get("source_accounting_kernel") == "accepted", "source accounting not accepted")
    require(accounting["decision"] == "accepted" and not accounting["failures"], "source accounting evidence rejected")
    validate_lifecycle(actual, reports)
    if actual["loops"]:
        require(gates.get("lean_userland") == "accepted", "Lean gate not accepted")
        require(proof is not None and proof["status"] == "checked", "missing checked Lean certificate")
        require(proof["proof_count"] == len(proof["proofs"]), "proof count mismatch")
        for loop in loops["loops"]:
            require(loop["projection_completeness"] == "complete" and not loop["excluded_by"], "incomplete loop accepted")
            matches = [p for p in proof["proofs"] if p["loop_summary_id"] == loop["id"]
                       and p["target_query_id"] == loop["induction_query_id"]]
            require(len(matches) == 1, "missing/ambiguous loop proof")
            p = matches[0]
            require(p["lean_status"] == "checked" and p["anti_vacuity_status"] == "proved", "unchecked/vacuous loop proof")
            require(p["authorization_claim"] == "loop_induction_and_nonvacuity", "wrong proof claim")
            require(re.fullmatch(r"[0-9a-f]{64}", p["loop_vc_sha256"] or ""), "missing loop digest")
            require(set(p["axioms"]) <= {"propext", "Quot.sound"}, "unapproved axiom")


def run_case(compiler, name, directory, timeout, capture_automatic_refusal=False):
    if capture_automatic_refusal:
        require(directory.is_dir() and not directory.is_symlink() and not any(directory.iterdir()), "refusal collection requires empty owned directory")
    command = [str(compiler), "emit", "--emit=smt-report,bytecode", "--out-dir", str(directory), name]
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        code = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        code = "timeout"
    (directory / "stdout.txt").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    reports = tuple(read_report(directory, suffix) for suffix in
                    ("smt.report.json", "formal.loops.json", "formal.artifacts.json", "lean.proof.json", "formal.accounting.json"))
    smt, loops, _, _, _ = reports
    actual = {"source": name, "source_sha256": sha(ROOT / name), "command": command, "exit": code,
              "seconds": round(time.monotonic() - started, 3), "hex": sorted(str(p.relative_to(directory)) for p in directory.rglob("*.hex")),
              "terminal_errors": terminal_errors(stdout + "\n" + stderr), "loops": loop_rows(loops), "smt_errors": smt_errors(smt),
              "smt_diagnostics": smt_diagnostics(smt)}
    if capture_automatic_refusal:
        actual = CollectedRefusal(actual)
        _refusal_collections[id(actual)] = (actual, directory, name, code)
    _lifecycle_collections[id(actual)] = (actual, directory, name, code,
        {p.name: sha(p) for p in directory.glob("*.json") if p.name != "run.json" and p.is_file() and not p.is_symlink()})
    (directory / "run.json").write_text(json.dumps(actual, indent=2) + "\n")
    return actual, reports


def membership(manifest, tracked):
    declared = set(manifest["files"])
    require(set(tracked) == declared, "unclassified/missing corpus inputs: " + repr(sorted(set(tracked) ^ declared)))
    for name in declared:
        require(name.startswith("ora-example/") and name.endswith(".ora") and ".." not in Path(name).parts,
                "invalid corpus path: " + name)
        require((ROOT / name).is_file(), "missing source: " + name)


def performance_hold(name, expected, open_findings):
    """A reviewed, unrun row. This provides no solver result or proof evidence."""
    require(expected["category"] == "performance_hold" and expected["source"] == name,
            "foreign performance hold")
    digest = expected["source_sha256"]
    require(re.fullmatch(r"[0-9a-f]{64}", digest or "") and int(digest, 16) != 0 and
            sha(ROOT / name) == digest, "performance hold source changed")
    require(expected.get("track") and expected["explanation"], "performance hold requires a named track and explanation")
    findings = [f for f in open_findings if f["id"] == expected["open_finding"]]
    require(len(findings) == 1 and findings[0]["id"].partition("/")[2] == expected["track"] and
            findings[0]["sources"].get(name) == digest and findings[0]["explanation"],
            "missing/mismatched performance finding")
    return {"source": name, "source_sha256": digest, "verdict": "performance_hold", "executed": False,
            "open_finding": expected["open_finding"], "track": expected["track"],
            "explanation": expected["explanation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "zig-out/bin/ora")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, help="new evidence directory; defaults to an owned temporary directory")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    require(args.timeout > 0, "timeout must be positive")
    manifest = json.loads(args.manifest.read_text())
    require(manifest["schema"] == 1, "unknown manifest schema")
    tracked = [p for p in subprocess.check_output(["git", "ls-files", "ora-example"], cwd=ROOT, text=True).splitlines() if p.endswith(".ora")]
    membership(manifest, tracked)
    output = args.output or Path(tempfile.mkdtemp(prefix="ora-default-fv-"))
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    compiler = args.compiler.resolve()
    binary_hash = sha(compiler)
    source_hashes = {name: sha(ROOT / name) for name in tracked}
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "ora-example"], cwd=ROOT, text=True).splitlines()
    result = {"binary_sha256": binary_hash, "manifest_sha256": sha(args.manifest), "untracked_not_certified": untracked,
              "open_findings": manifest["open_findings"], "runs": []}
    print("default-FV evidence: " + str(output), flush=True)
    for finding in result["open_findings"]:
        require(finding["id"] and finding["explanation"] and finding["sources"], "malformed open finding")
        for name, digest in finding["sources"].items():
            require(sha(ROOT / name) == digest, "open reproducer changed: " + name)
        print("EXPECTED OPEN (not a proof): " + finding["id"] + ": " + finding["explanation"], flush=True)
    for index, name in enumerate(tracked):
        directory = output / str(index)
        directory.mkdir()
        try:
            require(sha(compiler) == binary_hash and sha(ROOT / name) == source_hashes[name], "binary/source changed during run")
            expected = manifest["files"][name]
            if expected["category"] == "performance_hold":
                actual = performance_hold(name, expected, manifest["open_findings"])
                print(f"PERFORMANCE HOLD (NOT RUN; no proof): {name} [{actual['open_finding']}]: {actual['explanation']}", flush=True)
                (directory / "hold.json").write_text(json.dumps(actual, indent=2) + "\n")
            else:
                actual, reports = run_case(compiler, name, directory, args.timeout,
                                           capture_automatic_refusal=expected["category"] == "automatic_refusal")
                validate_result(expected, actual, reports, manifest["open_findings"])
                actual["verdict"] = expected["category"]
        except (ValueError, KeyError, TypeError, OSError) as error:
            actual = locals().get("actual", {"source": name})
            actual["verdict"] = "unexpected_failure"
            actual["failure"] = str(error)
        result["runs"].append(actual)
        print(f"{index + 1}/{len(tracked)} {name}: {actual['verdict']}", flush=True)
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        del actual
    require(sha(compiler) == binary_hash and all(sha(ROOT / n) == h for n, h in source_hashes.items()), "tree/binary changed during run")
    counts = collections.Counter(r["verdict"] for r in result["runs"])
    print("default-FV results: " + json.dumps(counts, sort_keys=True))
    print("performance holds (not run; no proof): " + str(counts["performance_hold"]))
    print("checked loops in accepted artifacts: " + str(sum(len(r["loops"]) for r in result["runs"] if r["verdict"] == "accepted")))
    require(not counts["unexpected_failure"], "unexpected corpus results; see " + str(output / "results.json"))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit("default-FV gate: " + str(error))
