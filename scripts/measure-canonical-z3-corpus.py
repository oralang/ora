#!/usr/bin/env python3
"""Measure canonical-vs-live SMT hash parity across an Ora source corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections import Counter, defaultdict


COUNT_FIELDS = (
    "total",
    "live_rows",
    "matched",
    "mismatched",
    "not_annotation_pure",
    "unavailable",
    "no_live_row",
)


def find_sources(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.ora") if path.is_file())


def safe_output_dir(base: Path, source: Path) -> Path:
    rel = str(source)
    digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12]
    stem = source.with_suffix("").as_posix().replace("/", "__").replace(":", "_")
    return base / f"{stem}__{digest}"


def measurement_path(out_dir: Path, source: Path) -> Path:
    return out_dir / "verify" / f"{source.stem}.canonical-z3.measure.json"


def zero_counts() -> dict[str, int]:
    return {field: 0 for field in COUNT_FIELDS}


def add_counts(target: dict[str, int], values: dict[str, int]) -> None:
    for field in COUNT_FIELDS:
        target[field] += int(values.get(field, 0))


def run_source(ora: Path, source: Path, out_root: Path, timeout: int) -> tuple[int, Path, str]:
    out_dir = safe_output_dir(out_root, source)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ora),
        "build",
        "-o",
        str(out_dir),
        "--measure-canonical-z3",
        str(source),
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, out_dir, proc.stdout


def aggregate_measurement(aggregate: dict, source: Path, data: dict, returncode: int) -> None:
    aggregate["files_measured"] += 1
    summary = data.get("summary", {})
    aggregate["queries"] += int(summary.get("queries", 0))
    aggregate["required"] += int(summary.get("required", 0))
    if returncode != 0:
        aggregate["measured_after_failure"] += 1

    file_record = {
        "source": str(source),
        "returncode": returncode,
        "summary": data.get("summary", {}),
        "shapes": data.get("shapes", {}),
        "unsupported_reasons": data.get("unsupported_reasons", {}),
        "hash_errors": data.get("hash_errors", {}),
    }
    aggregate["files"].append(file_record)

    for shape, counts in data.get("shapes", {}).items():
        add_counts(aggregate["shapes"][shape], counts)
    aggregate["unsupported_reasons"].update(
        {name: int(count) for name, count in data.get("unsupported_reasons", {}).items() if int(count) != 0}
    )
    aggregate["hash_errors"].update(
        {name: int(count) for name, count in data.get("hash_errors", {}).items() if int(count) != 0}
    )
    for query in data.get("queries", []):
        outcome = query.get("outcome", "unknown")
        aggregate["outcomes"][outcome] += 1
        if query.get("required") is True and outcome != "match":
            aggregate["required_failures"].append({
                "source": str(source),
                "query_id": query.get("id"),
                "shape": query.get("shape"),
                "outcome": outcome,
                "reason": query.get("reason"),
            })


def print_summary(aggregate: dict) -> None:
    print("canonical Z3 corpus measurement")
    print(f"  files scanned:            {aggregate['files_scanned']}")
    print(f"  files measured:           {aggregate['files_measured']}")
    print(f"  measured after failure:   {aggregate['measured_after_failure']}")
    print(f"  failed without measure:   {len(aggregate['failures'])}")
    print(f"  queries measured:         {aggregate['queries']}")
    print(f"  required rows:            {aggregate['required']}")
    print()

    print("per-shape counts")
    print("  shape                       total  live  match  mismatch  nonpure  unavailable  no_live")
    for shape in sorted(aggregate["shapes"]):
        counts = aggregate["shapes"][shape]
        if not any(counts.values()):
            continue
        print(
            f"  {shape:<27} {counts['total']:>5} {counts['live_rows']:>5} "
            f"{counts['matched']:>6} {counts['mismatched']:>9} "
            f"{counts['not_annotation_pure']:>8} "
            f"{counts['unavailable']:>11} {counts['no_live_row']:>8}"
        )

    if aggregate["unsupported_reasons"]:
        print()
        print("unsupported reasons")
        for name, count in sorted(aggregate["unsupported_reasons"].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {name:<40} {count}")

    if aggregate["hash_errors"]:
        print()
        print("hash errors")
        for name, count in sorted(aggregate["hash_errors"].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {name:<40} {count}")

    if aggregate["required_failures"]:
        print()
        print("required row failures")
        for failure in aggregate["required_failures"][:20]:
            print(
                f"  {failure['source']} query={failure['query_id']} "
                f"shape={failure['shape']} outcome={failure['outcome']} "
                f"reason={failure['reason']}"
            )
        if len(aggregate["required_failures"]) > 20:
            print(f"  ... {len(aggregate['required_failures']) - 20} more")

    if aggregate["failures"]:
        print()
        print("failures without measurement")
        for failure in aggregate["failures"][:20]:
            print(f"  {failure['source']} ({failure['reason']})")
        if len(aggregate["failures"]) > 20:
            print(f"  ... {len(aggregate['failures']) - 20} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=["ora-example"], help="Ora source files or directories")
    parser.add_argument("--ora", default="./zig-out/bin/ora", help="Ora compiler binary")
    parser.add_argument("--out-dir", default="/tmp/ora-canonical-z3-corpus", help="temporary artifact root")
    parser.add_argument("--json-out", help="write aggregate JSON report")
    parser.add_argument("--timeout", type=int, default=120, help="per-file timeout in seconds")
    parser.add_argument("--limit", type=int, help="limit number of sources for smoke runs")
    parser.add_argument("--fail-required", action="store_true", help="fail if any required canonical row does not match")
    parser.add_argument("--min-required", type=int, default=0, help="minimum required rows expected when --fail-required is set")
    args = parser.parse_args()

    ora = Path(args.ora)
    if not ora.exists():
        print(f"error: Ora compiler not found: {ora}", file=sys.stderr)
        return 2

    sources: list[Path] = []
    for root in args.roots:
        sources.extend(find_sources(Path(root)))
    sources = sorted(dict.fromkeys(sources))
    if args.limit is not None:
        sources = sources[: args.limit]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "schema": "ora.canonical_z3.corpus_measure.v1",
        "files_scanned": len(sources),
        "files_measured": 0,
        "measured_after_failure": 0,
        "queries": 0,
        "required": 0,
        "shapes": defaultdict(zero_counts),
        "unsupported_reasons": Counter(),
        "hash_errors": Counter(),
        "outcomes": Counter(),
        "required_failures": [],
        "failures": [],
        "files": [],
    }

    for index, source in enumerate(sources, 1):
        print(f"[{index}/{len(sources)}] {source}", flush=True)
        try:
            returncode, out_dir, output = run_source(ora, source, out_root, args.timeout)
        except subprocess.TimeoutExpired:
            aggregate["failures"].append({"source": str(source), "reason": "timeout"})
            continue

        measure_file = measurement_path(out_dir, source)
        if measure_file.exists():
            with measure_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            aggregate_measurement(aggregate, source, data, returncode)
        else:
            reason = f"exit_{returncode}" if returncode != 0 else "missing_measurement"
            aggregate["failures"].append({
                "source": str(source),
                "reason": reason,
                "output_tail": output[-4000:],
            })

    serializable = {
        **aggregate,
        "shapes": dict(aggregate["shapes"]),
        "unsupported_reasons": dict(aggregate["unsupported_reasons"]),
        "hash_errors": dict(aggregate["hash_errors"]),
        "outcomes": dict(aggregate["outcomes"]),
    }
    print_summary(serializable)

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if args.fail_required:
        failed = False
        if serializable["failures"]:
            print("error: corpus measurement had files without measurement", file=sys.stderr)
            failed = True
        if serializable["required"] < args.min_required:
            print(
                f"error: required row count {serializable['required']} below minimum {args.min_required}",
                file=sys.stderr,
            )
            failed = True
        if serializable["required_failures"]:
            print(
                f"error: {len(serializable['required_failures'])} required canonical rows did not match",
                file=sys.stderr,
            )
            failed = True
        if failed:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
