#!/usr/bin/env python3
"""Build a deterministic census of Ora source loops and verifier loop queries."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import shutil
import subprocess
import sys


SCHEMA = "ora.loop_census.v3"
SCHEMA_VERSION = 3
APP_DIRECTORIES = {"apps", "dex", "vault", "locks", "protocols"}
QUERY_KINDS = ("LoopInvariantStep", "LoopBodySafety", "LoopInvariantPost")
STRATA = ("apps", "fixtures")


def find_sources(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.ora") if path.is_file())


def relative_source(path: Path, corpus_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(corpus_root.parent.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def classify_stratum(path: Path, corpus_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(corpus_root.resolve())
    except ValueError:
        return "fixtures"
    return "apps" if relative.parts and relative.parts[0] in APP_DIRECTORIES else "fixtures"


def expected_noncompile_reason(source_name: str) -> str | None:
    """Classify established negative-fixture path conventions.

    >>> expected_noncompile_reason("ora-example/locks/negative/reject.ora")
    'negative_fixture_directory'
    >>> expected_noncompile_reason("ora-example/logs/log_fail_arity.ora")
    'failure_fixture_filename'
    >>> expected_noncompile_reason("ora-example/apps/counter.ora") is None
    True
    """
    path = Path(source_name)
    for part in path.parts[:-1]:
        normalized = part.lower().replace("-", "_")
        if normalized == "negative" or normalized.startswith("negative_"):
            return "negative_fixture_directory"
    if "fail" in path.stem.lower():
        return "failure_fixture_filename"
    return None


def run_source(tool: Path, source: Path, timeout: int) -> dict:
    try:
        proc = subprocess.run(
            [str(tool), str(source.resolve())],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "measurement_failed": True,
            "failure": "census_timeout",
        }
    except OSError as exc:
        return {
            "measurement_failed": True,
            "failure": f"census_tool_error:{exc}",
        }
    if proc.returncode != 0:
        if proc.returncode < 0:
            try:
                signal_name = signal.Signals(-proc.returncode).name
            except ValueError:
                signal_name = str(-proc.returncode)
            return {
                "schema": "ora.loop_census.file.v2",
                "schema_version": 2,
                "source_file": str(source.resolve()),
                "compiling": False,
                "first_diagnostic": f"compiler_process_signal:{signal_name}",
                "measurement_errors": [],
                "loops": [],
            }
        return {
            "measurement_failed": True,
            "failure": f"census_tool_exit_{proc.returncode}",
            "stderr_tail": proc.stderr[-2000:],
        }
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "measurement_failed": True,
            "failure": f"invalid_census_json:{exc.msg}",
            "output_tail": proc.stdout[-2000:],
        }
    if not result.get("compiling", False) and str(result.get("first_diagnostic", "")).startswith("compiler_error:"):
        stderr_lines = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
        if stderr_lines:
            result["first_diagnostic"] = stderr_lines[0]
    return result


def run_production_activation(compiler: Path, source: Path, output_dir: Path, timeout: int) -> dict:
    """Collect the production formal classifier's report without running a proof gate."""
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        proc = subprocess.run(
            [
                str(compiler),
                "emit",
                "--no-verify",
                "--emit=mlir:ora",
                "--out-dir",
                str(output_dir),
                str(source.resolve()),
            ],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"measurement_failed": True, "failure": "production_activation_timeout"}
    except OSError as exc:
        return {"measurement_failed": True, "failure": f"production_activation_tool_error:{exc}"}

    if proc.returncode != 0:
        if proc.returncode < 0:
            try:
                signal_name = signal.Signals(-proc.returncode).name
            except ValueError:
                signal_name = str(-proc.returncode)
            failure = f"production_activation_signal:{signal_name}"
        else:
            failure = f"production_activation_exit_{proc.returncode}"
        return {
            "measurement_failed": True,
            "failure": failure,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    report_path = output_dir / f"{source.stem}.formal.loops.json"
    try:
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except FileNotFoundError:
        # The production emitter intentionally omits the report when no loop
        # survives into MLIR. Keep that distinct from a failed compilation.
        report = {
            "schema_version": 1,
            "record": "loop_induction_report",
            "total": 0,
            "certificate_required": 0,
            "excluded": 0,
            "loops": [],
        }
        return {"measurement_failed": False, "report": report, "report_absent": True}
    except OSError as exc:
        return {"measurement_failed": True, "failure": f"production_activation_report_missing:{exc}"}
    except json.JSONDecodeError as exc:
        return {"measurement_failed": True, "failure": f"production_activation_invalid_json:{exc.msg}"}

    if report.get("schema_version") != 1 or report.get("record") != "loop_induction_report":
        return {"measurement_failed": True, "failure": "production_activation_schema_mismatch"}
    return {"measurement_failed": False, "report": report}


def empty_stratum() -> dict:
    return {
        "summary": {
            "files_total": 0,
            "files_compiling": 0,
            "files_unexpected_noncompiling": 0,
            "files_expected_noncompiling": 0,
            "files_unmeasured": 0,
            "files_with_loops": 0,
            "files_with_invariants": 0,
            "files_with_degraded_verification_encoding": 0,
            "loops": 0,
            "invariant_bearing_loops": 0,
            "scalar_fragment_loops": 0,
            "scalar_fragment_invariant_loops": 0,
            "shape_scalar_fragment_loops": 0,
            "shape_scalar_fragment_invariant_loops": 0,
            "loop_queries": 0,
            "scalar_fragment_loop_queries": 0,
            "shape_scalar_fragment_loop_queries": 0,
            "production_activation_files_measured": 0,
            "production_activation_files_failed": 0,
            "production_loop_summaries": 0,
            "production_zero_summary_files": 0,
            "production_source_loop_delta_files": 0,
            "lean_certificate_required_loops": 0,
            "lean_certificate_excluded_loops": 0,
            "lean_certificate_required_basis_points": 0,
            "lean_exclusion_reasons": {},
            "loop_queries_in_scalar_fragment_basis_points": 0,
            "invariant_loops_in_scalar_fragment_basis_points": 0,
            "loop_queries_in_shape_scalar_fragment_basis_points": 0,
            "invariant_loops_in_shape_scalar_fragment_basis_points": 0,
            "query_counts": {kind: 0 for kind in QUERY_KINDS},
            "scalar_fragment_query_counts": {kind: 0 for kind in QUERY_KINDS},
            "shape_scalar_fragment_query_counts": {kind: 0 for kind in QUERY_KINDS},
            "loop_form_histogram": {},
            "invariant_count_histogram": {},
            "exclusion_reasons": {},
            "shape_exclusion_reasons": {},
        },
        "expected_failures": [],
        "failures": [],
        "tool_failures": [],
        "encoding_degraded_files": [],
        "measurement_errors": [],
        "production_activation_failures": [],
        "production_activation_notes": [],
        "production_loop_classifications": [],
        "loops": [],
    }


def basis_points(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return 0
    return (numerator * 10_000) // denominator


def finalize_stratum(data: dict) -> None:
    loops = sorted(
        data["loops"],
        key=lambda row: (row["file"], int(row["line"]), int(row.get("column", 0)), row["function"], row["loop_form"]),
    )
    data["loops"] = loops
    data["expected_failures"] = sorted(data["expected_failures"], key=lambda row: row["file"])
    data["failures"] = sorted(data["failures"], key=lambda row: row["file"])
    data["tool_failures"] = sorted(data["tool_failures"], key=lambda row: row["file"])
    data["encoding_degraded_files"] = sorted(data["encoding_degraded_files"])
    data["measurement_errors"] = sorted(
        data["measurement_errors"], key=lambda row: (row["file"], row.get("reason", ""), row.get("statement_id", -1))
    )
    data["production_activation_failures"] = sorted(
        data["production_activation_failures"], key=lambda row: row["file"]
    )
    data["production_activation_notes"] = sorted(
        data["production_activation_notes"], key=lambda row: row["file"]
    )
    data["production_loop_classifications"] = sorted(
        data["production_loop_classifications"],
        key=lambda row: (
            row["file"],
            int(row["source"].get("line", 0)),
            int(row["source"].get("column", 0)),
            row["owner"].get("function_name", ""),
        ),
    )

    summary = data["summary"]
    form_histogram: Counter[str] = Counter()
    invariant_histogram: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    shape_exclusions: Counter[str] = Counter()
    lean_exclusions: Counter[str] = Counter()
    files_with_loops: set[str] = set()
    files_with_invariants: set[str] = set()

    for loop in loops:
        files_with_loops.add(loop["file"])
        form_histogram[loop["loop_form"]] += 1
        invariant_histogram[str(loop["invariant_count"])] += 1
        if loop["has_invariants"]:
            files_with_invariants.add(loop["file"])
        for reason in loop["excluded_by"]:
            exclusions[reason] += 1
        for reason in loop["shape_excluded_by"]:
            shape_exclusions[reason] += 1
    for loop in data["production_loop_classifications"]:
        for reason in loop["excluded_by"]:
            lean_exclusions[reason] += 1

    summary["files_with_loops"] = len(files_with_loops)
    summary["files_with_invariants"] = len(files_with_invariants)
    summary["loop_queries_in_scalar_fragment_basis_points"] = basis_points(
        summary["scalar_fragment_loop_queries"], summary["loop_queries"]
    )
    summary["invariant_loops_in_scalar_fragment_basis_points"] = basis_points(
        summary["scalar_fragment_invariant_loops"], summary["invariant_bearing_loops"]
    )
    summary["loop_queries_in_shape_scalar_fragment_basis_points"] = basis_points(
        summary["shape_scalar_fragment_loop_queries"], summary["loop_queries"]
    )
    summary["invariant_loops_in_shape_scalar_fragment_basis_points"] = basis_points(
        summary["shape_scalar_fragment_invariant_loops"], summary["invariant_bearing_loops"]
    )
    summary["lean_certificate_required_basis_points"] = basis_points(
        summary["lean_certificate_required_loops"], summary["production_loop_summaries"]
    )
    summary["loop_form_histogram"] = dict(sorted(form_histogram.items()))
    summary["invariant_count_histogram"] = dict(sorted(invariant_histogram.items(), key=lambda item: int(item[0])))
    summary["exclusion_reasons"] = dict(sorted(exclusions.items(), key=lambda item: (-item[1], item[0])))
    summary["shape_exclusion_reasons"] = dict(sorted(shape_exclusions.items(), key=lambda item: (-item[1], item[0])))
    summary["lean_exclusion_reasons"] = dict(sorted(lean_exclusions.items(), key=lambda item: (-item[1], item[0])))


def add_file(data: dict, source_name: str, result: dict, activation: dict | None) -> None:
    summary = data["summary"]
    summary["files_total"] += 1
    if result.get("measurement_failed", False):
        summary["files_unmeasured"] += 1
        data["tool_failures"].append({"file": source_name, **result})
        return
    if not result.get("compiling", False):
        diagnostic = result.get("first_diagnostic", "missing_first_diagnostic")
        diagnostic = str(diagnostic).replace(str(Path(source_name).resolve()), source_name)
        failure = {"file": source_name, "first_diagnostic": diagnostic}
        expectation_reason = expected_noncompile_reason(source_name)
        if expectation_reason is not None:
            summary["files_expected_noncompiling"] += 1
            data["expected_failures"].append({**failure, "expectation_reason": expectation_reason})
        else:
            summary["files_unexpected_noncompiling"] += 1
            data["failures"].append(failure)
        return

    summary["files_compiling"] += 1
    for error in result.get("measurement_errors", []):
        data["measurement_errors"].append({"file": source_name, **error})
    if result.get("verification_encoding_degraded", False):
        summary["files_with_degraded_verification_encoding"] += 1
        data["encoding_degraded_files"].append(source_name)

    raw_loops = result.get("loops", [])
    if raw_loops:
        if activation is None or activation.get("measurement_failed", False):
            summary["production_activation_files_failed"] += 1
            failure = activation or {"failure": "production_activation_not_run"}
            data["production_activation_failures"].append({"file": source_name, **failure})
        else:
            summary["production_activation_files_measured"] += 1
            activation_report = activation["report"]
            production_loops = activation_report.get("loops", [])
            report_total = int(activation_report.get("total", -1))
            required = int(activation_report.get("certificate_required", -1))
            excluded = int(activation_report.get("excluded", -1))
            if report_total != len(production_loops) or required + excluded != report_total:
                summary["production_activation_files_failed"] += 1
                summary["production_activation_files_measured"] -= 1
                data["production_activation_failures"].append(
                    {
                        "file": source_name,
                        "failure": "production_activation_internal_count_mismatch",
                        "production_report_total": report_total,
                        "production_report_rows": len(production_loops),
                    }
                )
            else:
                if report_total == 0:
                    summary["production_zero_summary_files"] += 1
                if report_total != len(raw_loops):
                    summary["production_source_loop_delta_files"] += 1
                    data["production_activation_notes"].append(
                        {
                            "file": source_name,
                            "reason": "production_summary_count_differs_from_source_census",
                            "source_loops": len(raw_loops),
                            "production_loop_summaries": report_total,
                            "report_absent": bool(activation.get("report_absent", False)),
                        }
                    )
                summary["production_loop_summaries"] += report_total
                summary["lean_certificate_required_loops"] += required
                summary["lean_certificate_excluded_loops"] += excluded
                for raw_row in production_loops:
                    row = dict(raw_row)
                    source = dict(row.get("source", {}))
                    source["file"] = source_name
                    row["source"] = source
                    row["file"] = source_name
                    data["production_loop_classifications"].append(row)

    for raw_loop in raw_loops:
        loop = dict(raw_loop)
        loop["file"] = source_name
        queries = loop["queries"]
        query_total = sum(int(queries.get(kind, 0)) for kind in QUERY_KINDS)
        scalar = bool(loop["scalar_fragment"])
        shape_scalar = bool(loop["shape_scalar_fragment"])

        summary["loops"] += 1
        if loop["has_invariants"]:
            summary["invariant_bearing_loops"] += 1
        if scalar:
            summary["scalar_fragment_loops"] += 1
            if loop["has_invariants"]:
                summary["scalar_fragment_invariant_loops"] += 1
            summary["scalar_fragment_loop_queries"] += query_total
        if shape_scalar:
            summary["shape_scalar_fragment_loops"] += 1
            if loop["has_invariants"]:
                summary["shape_scalar_fragment_invariant_loops"] += 1
            summary["shape_scalar_fragment_loop_queries"] += query_total
        summary["loop_queries"] += query_total
        for kind in QUERY_KINDS:
            count = int(queries.get(kind, 0))
            summary["query_counts"][kind] += count
            if scalar:
                summary["scalar_fragment_query_counts"][kind] += count
            if shape_scalar:
                summary["shape_scalar_fragment_query_counts"][kind] += count
        data["loops"].append(loop)


def format_percent(basis_point_value: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{basis_point_value / 100:.2f}%"


def print_summary(report: dict) -> None:
    print("Ora loop census")
    for stratum in STRATA:
        summary = report["strata"][stratum]["summary"]
        print()
        print(f"{stratum} stratum")
        print(
            "  files: "
            f"total={summary['files_total']} compiling={summary['files_compiling']} "
            f"unexpected_noncompiling={summary['files_unexpected_noncompiling']} "
            f"expected_noncompiling={summary['files_expected_noncompiling']} "
            f"unmeasured={summary['files_unmeasured']} "
            f"with_loops={summary['files_with_loops']} "
            f"with_invariants={summary['files_with_invariants']} "
            f"degraded_verification_encoding={summary['files_with_degraded_verification_encoding']}"
        )
        print(
            "  loops: "
            f"total={summary['loops']} invariant_bearing={summary['invariant_bearing_loops']} "
            f"shape_scalar_fragment={summary['shape_scalar_fragment_loops']} "
            f"scalar_fragment={summary['scalar_fragment_loops']}"
        )
        print("  query kind              total  shape-scalar  scalar-fragment")
        for kind in QUERY_KINDS:
            print(
                f"  {kind:<23} {summary['query_counts'][kind]:>5} "
                f"{summary['shape_scalar_fragment_query_counts'][kind]:>13} "
                f"{summary['scalar_fragment_query_counts'][kind]:>16}"
            )
        print(
            "  loop queries inside shape-only scalar fragment: "
            f"{summary['shape_scalar_fragment_loop_queries']}/{summary['loop_queries']} "
            f"({format_percent(summary['loop_queries_in_shape_scalar_fragment_basis_points'], summary['loop_queries'])})"
        )
        print(
            "  loop queries inside scalar fragment: "
            f"{summary['scalar_fragment_loop_queries']}/{summary['loop_queries']} "
            f"({format_percent(summary['loop_queries_in_scalar_fragment_basis_points'], summary['loop_queries'])})"
        )
        print(
            "  invariant-bearing loops inside shape-only scalar fragment: "
            f"{summary['shape_scalar_fragment_invariant_loops']}/{summary['invariant_bearing_loops']} "
            f"({format_percent(summary['invariant_loops_in_shape_scalar_fragment_basis_points'], summary['invariant_bearing_loops'])})"
        )
        print(
            "  invariant-bearing loops inside scalar fragment: "
            f"{summary['scalar_fragment_invariant_loops']}/{summary['invariant_bearing_loops']} "
            f"({format_percent(summary['invariant_loops_in_scalar_fragment_basis_points'], summary['invariant_bearing_loops'])})"
        )
        print(
            "  production Lean induction activation: "
            f"required={summary['lean_certificate_required_loops']}/"
            f"{summary['production_loop_summaries']} "
            f"({format_percent(summary['lean_certificate_required_basis_points'], summary['production_loop_summaries'])}) "
            f"excluded={summary['lean_certificate_excluded_loops']} "
            f"files_measured={summary['production_activation_files_measured']} "
            f"files_failed={summary['production_activation_files_failed']} "
            f"zero_summary_files={summary['production_zero_summary_files']} "
            f"source/runtime_count_delta_files={summary['production_source_loop_delta_files']}"
        )
        print("  loop forms")
        for name, count in summary["loop_form_histogram"].items():
            print(f"    {name:<35} {count}")
        print("  ranked exclusions")
        for name, count in summary["exclusion_reasons"].items():
            print(f"    {name:<55} {count}")
        print("  ranked production Lean exclusions")
        for name, count in summary["lean_exclusion_reasons"].items():
            print(f"    {name:<55} {count}")
        if report["strata"][stratum]["expected_failures"]:
            print("  expected-noncompiling fixtures")
            for failure in report["strata"][stratum]["expected_failures"]:
                print(f"    {failure['file']}: {failure['first_diagnostic']}")
        if report["strata"][stratum]["failures"]:
            print("  unexpected noncompiling files")
            for failure in report["strata"][stratum]["failures"]:
                print(f"    {failure['file']}: {failure['first_diagnostic']}")
        if report["strata"][stratum]["measurement_errors"]:
            print("  measurement errors")
            for error in report["strata"][stratum]["measurement_errors"]:
                print(f"    {error['file']}: {error['reason']}")
        if report["strata"][stratum]["encoding_degraded_files"]:
            print("  degraded verification encoding")
            for file_name in report["strata"][stratum]["encoding_degraded_files"]:
                print(f"    {file_name}")
        if report["strata"][stratum]["tool_failures"]:
            print("  tool failures")
            for failure in report["strata"][stratum]["tool_failures"]:
                print(f"    {failure['file']}: {failure['failure']}")
        if report["strata"][stratum]["production_activation_failures"]:
            print("  production activation failures")
            for failure in report["strata"][stratum]["production_activation_failures"]:
                print(f"    {failure['file']}: {failure['failure']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=["ora-example"], help="Ora source files or directories")
    parser.add_argument("--tool", default="./zig-out/bin/ora-loop-census", help="loop census emitter binary")
    parser.add_argument("--compiler", default="./zig-out/bin/ora", help="production Ora compiler binary")
    parser.add_argument(
        "--activation-out-dir",
        default="zig-out/loop-census/formal-activation",
        help="output directory for per-file production classifier reports",
    )
    parser.add_argument("--corpus-root", default="ora-example", help="root used for paths and stratum classification")
    parser.add_argument("--json-out", help="write deterministic aggregate JSON report")
    parser.add_argument("--timeout", type=int, default=120, help="per-file timeout in seconds")
    parser.add_argument("--limit", type=int, help="limit number of sorted sources for smoke runs")
    parser.add_argument("--quiet-progress", action="store_true", help="suppress per-file progress lines")
    args = parser.parse_args()

    tool = Path(args.tool).resolve()
    if not tool.exists():
        print(f"error: loop census tool not found: {tool}", file=sys.stderr)
        return 2
    compiler = Path(args.compiler).resolve()
    if not compiler.exists():
        print(f"error: production compiler not found: {compiler}", file=sys.stderr)
        return 2
    activation_root = Path(args.activation_out_dir)

    corpus_root = Path(args.corpus_root)
    sources: list[Path] = []
    for root in args.roots:
        sources.extend(find_sources(Path(root)))
    sources = sorted(dict.fromkeys(path.resolve() for path in sources))
    if args.limit is not None:
        sources = sources[: args.limit]

    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "corpus_root": corpus_root.as_posix(),
        "strata": {stratum: empty_stratum() for stratum in STRATA},
    }

    for index, source in enumerate(sources, 1):
        source_name = relative_source(source, corpus_root)
        stratum = classify_stratum(source, corpus_root)
        if not args.quiet_progress:
            print(f"[{index}/{len(sources)}] [{stratum}] {source_name}", flush=True)
        result = run_source(tool, source, args.timeout)
        activation = None
        if (
            not result.get("measurement_failed", False)
            and result.get("compiling", False)
            and result.get("loops", [])
        ):
            activation = run_production_activation(
                compiler,
                source,
                activation_root / f"{index:04d}",
                args.timeout,
            )
        add_file(report["strata"][stratum], source_name, result, activation)

    for stratum in STRATA:
        finalize_stratum(report["strata"][stratum])

    print_summary(report)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if any(
        report["strata"][stratum]["tool_failures"]
        or report["strata"][stratum]["production_activation_failures"]
        for stratum in STRATA
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
