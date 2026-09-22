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


def smt_errors(report):
    if report is None:
        return []
    return [{**{key: row.get(key) for key in ("type", "message", "line", "column", "subtype")},
             "file": row["file"].replace(str(ROOT) + "/", "") if row.get("file") else None}
            for row in report["errors"]]


def smt_diagnostics(report):
    if report is None:
        return []
    return [{key: value.replace(str(ROOT) + "/", "") if isinstance(value, str) else value
             for key, value in row.items() if key != "counterexample"} for row in report["diagnostics"]]


def validate_result(expected, actual, reports):
    """No expected category can excuse UNKNOWN, degradation, or missing proof evidence."""
    smt, loops, authority, proof, accounting = reports
    require(actual["exit"] in (0, 1), "crash, signal, or timeout")
    if smt is not None:
        require(smt["verification"]["verification_trust"] == "full", "degraded verification trust")
        summary = smt["summary"]
        for key in ("unknown", "vacuous", "vacuity_unknown", "inconsistent_bases", "encoding_degraded"):
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
    category = expected["category"]
    require(category in ("accepted", "excluded", "invalid", "unsafe"), "unclassified expectation")
    require(expected["explanation"], "missing expectation justification")
    if category != "accepted":
        require(expected.get("track"), "rejection requires a named track")
        require(actual["exit"] == 1 and not actual["hex"], "unexpected acceptance / bytecode on rejection")
        if category != "invalid":
            require(smt is not None and authority is not None, "missing rejection report")
            require(authority["final_artifact_authorization"] != "allowed", "rejected source authorized")
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


def run_case(compiler, name, directory, timeout):
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
    (directory / "run.json").write_text(json.dumps(actual, indent=2) + "\n")
    return actual, reports


def membership(manifest, tracked):
    declared = set(manifest["files"])
    require(set(tracked) == declared, "unclassified/missing corpus inputs: " + repr(sorted(set(tracked) ^ declared)))
    for name in declared:
        require(name.startswith("ora-example/") and name.endswith(".ora") and ".." not in Path(name).parts,
                "invalid corpus path: " + name)
        require((ROOT / name).is_file(), "missing source: " + name)


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
            actual, reports = run_case(compiler, name, directory, args.timeout)
            validate_result(manifest["files"][name], actual, reports)
            actual["verdict"] = manifest["files"][name]["category"]
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
    print("checked loops in accepted artifacts: " + str(sum(len(r["loops"]) for r in result["runs"] if r["verdict"] == "accepted")))
    require(not counts["unexpected_failure"], "unexpected corpus results; see " + str(output / "results.json"))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit("default-FV gate: " + str(error))
