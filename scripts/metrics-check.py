#!/usr/bin/env python3
"""Change-quality metrics benchmark.

Runs the metrics-snapshot harness over the conformance corpus and compares every
metric (per-call gas, per-contract bytecode size) against a committed baseline.
On each pipeline change this gives an automatic better/worse verdict — per
operation, and per CATEGORY (gas / bytecode size) with totals + percentages — so
you can tell a real broad improvement from a narrow local trick, and catch a
cross-dimension regression (e.g. smaller bytecode but more gas).

  scripts/metrics-check.py            # diff against baseline, report verdict
  scripts/metrics-check.py --update   # rewrite the baseline (review the diff!)
  scripts/metrics-check.py --check    # diff; exit 1 on ANY change (gate/CI use)
  scripts/metrics-check.py --check-size
                                      # exit 1 on bytecode-size drift only;
                                      # gas remains visible/local
  scripts/metrics-check.py --report-dir /tmp/metrics-report
                                      # also write current.tsv, compare.tsv,
                                      # changed.tsv, summary.json

Baseline: tests/conformance/metrics_snapshot.txt  (committed)
Metric category is keyed by suffix: '::__bytecode_bytes' is gated size,
'::__deploy_gas' is local deploy gas, and all remaining call metrics are local
runtime gas.
"""

from __future__ import annotations

import subprocess
import sys
import json
import os
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(os.environ.get("ORA_METRICS_HARNESS", ROOT / "zig-out" / "bin" / "metrics-snapshot"))
BASELINE = Path(os.environ.get("ORA_METRICS_BASELINE", ROOT / "tests" / "conformance" / "metrics_snapshot.txt"))
VALID_MODES = {"--diff", "--update", "--check", "--check-size"}


@dataclass
class Options:
    mode: str = "--diff"
    report_dir: Path | None = None


def category(key: str) -> str:
    if key.endswith("::__bytecode_bytes"):
        return "bytecode_bytes"
    if "::__deploy_gas" in key:
        return "deploy_gas"
    return "runtime_gas"


def gated_category(cat: str) -> bool:
    return cat == "bytecode_bytes"


def parse_args(argv: list[str]) -> Options:
    options = Options()
    mode_seen = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in VALID_MODES:
            if mode_seen:
                print("metrics-check: choose only one of --diff, --update, --check, or --check-size", file=sys.stderr)
                raise SystemExit(2)
            mode_seen = True
            options.mode = arg
        elif arg == "--report-dir":
            index += 1
            if index >= len(argv):
                print("metrics-check: --report-dir requires a path", file=sys.stderr)
                raise SystemExit(2)
            options.report_dir = Path(argv[index])
        elif arg.startswith("--report-dir="):
            options.report_dir = Path(arg.split("=", 1)[1])
        else:
            print("usage: metrics-check.py [--diff|--update|--check|--check-size] [--report-dir PATH]", file=sys.stderr)
            raise SystemExit(2)
        index += 1
    return options


def run_harness() -> dict[str, int]:
    if not HARNESS.exists():
        print("metrics-check: build the harness first: zig build metrics-snapshot", file=sys.stderr)
        raise SystemExit(2)
    p = subprocess.run([str(HARNESS)], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        print(f"metrics-check: harness failed:\n{p.stderr[:800]}", file=sys.stderr)
        raise SystemExit(1)
    out: dict[str, int] = {}
    for line in p.stdout.splitlines():
        if "\t" not in line:
            continue
        key, _, val = line.rpartition("\t")
        out[key] = int(val)
    return out


def load(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if "\t" not in line or line.startswith("#"):
            continue
        key, _, val = line.rpartition("\t")
        out[key] = int(val)
    return out


def write_baseline(current: dict[str, int]) -> None:
    head = ["# metrics baseline — <key>\\t<value>. Categories: ::__bytecode_bytes (size), else gas.",
            "# Regenerate with scripts/metrics-check.py --update and REVIEW the diff before committing."]
    body = [f"{k}\t{v}" for k, v in sorted(current.items())]
    BASELINE.write_text("\n".join(head + body) + "\n")


@dataclass
class Comparison:
    common: set[str]
    changed: list[str]
    added: list[str]
    removed: list[str]
    totals: dict[str, list[int]]


def compare(current: dict[str, int], baseline: dict[str, int]) -> Comparison:
    common = set(current) & set(baseline)
    changed = sorted(k for k in common if current[k] != baseline[k])
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))

    totals: dict[str, list[int]] = {}
    for key in common:
        cat = category(key)
        totals.setdefault(cat, [0, 0])
        totals[cat][0] += baseline[key]
        totals[cat][1] += current[key]
    return Comparison(
        common=common,
        changed=changed,
        added=added,
        removed=removed,
        totals=totals,
    )


def has_gated_drift(comparison: Comparison) -> bool:
    for key in comparison.changed + comparison.added + comparison.removed:
        if gated_category(category(key)):
            return True
    return False


def print_comparison(current: dict[str, int], baseline: dict[str, int], comparison: Comparison) -> None:
    for k in comparison.changed:
        d = current[k] - baseline[k]
        print(f"  {'▲' if d > 0 else '▼'} {k}: {baseline[k]} -> {current[k]} ({d:+d})")
    for k in comparison.added:
        print(f"  + {k}: {current[k]} (new)")
    for k in comparison.removed:
        print(f"  - {k}: {baseline[k]} (gone)")

    print("metrics-check: verdict by category (common set):")
    overall = "no change"
    for cat in sorted(comparison.totals):
        base_t, cur_t = comparison.totals[cat]
        d = cur_t - base_t
        pct = (100.0 * d / base_t) if base_t else 0.0
        verdict = "BETTER" if d < 0 else ("WORSE" if d > 0 else "same")
        gated = "gated" if gated_category(cat) else "local"
        print(f"    {cat:16} {base_t} -> {cur_t} ({d:+d}, {pct:+.2f}%)  [{verdict}, {gated}]")
        if d != 0:
            overall = "mixed/changed"
    print(f"    => {len(comparison.changed)} changed, {len(comparison.added)} new, {len(comparison.removed)} removed ({overall})")


def write_report(report_dir: Path, current: dict[str, int], baseline: dict[str, int], comparison: Comparison) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "current.tsv").write_text("".join(f"{key}\t{value}\n" for key, value in sorted(current.items())))

    compare_lines = ["category\tkeys\tbaseline_total\tcurrent_total\tdelta\tpct\tgate\tverdict"]
    categories: dict[str, dict[str, object]] = {}
    for cat in sorted(comparison.totals):
        baseline_total, current_total = comparison.totals[cat]
        delta = current_total - baseline_total
        pct = (100.0 * delta / baseline_total) if baseline_total else 0.0
        verdict = "BETTER" if delta < 0 else ("WORSE" if delta > 0 else "same")
        gate = "gated" if gated_category(cat) else "local"
        key_count = sum(1 for key in comparison.common if category(key) == cat)
        categories[cat] = {
            "keys": key_count,
            "baseline_total": baseline_total,
            "current_total": current_total,
            "delta": delta,
            "pct": pct,
            "gate": gate,
            "verdict": verdict,
        }
        compare_lines.append(
            f"{cat}\t{key_count}\t{baseline_total}\t{current_total}\t{delta}\t{pct:.4f}\t{gate}\t{verdict}"
        )
    (report_dir / "compare.tsv").write_text("\n".join(compare_lines) + "\n")

    changed_lines = ["status\tcategory\tkey\tbaseline\tcurrent\tdelta\tpct\tgate"]
    for key in comparison.changed:
        old = baseline[key]
        new = current[key]
        delta = new - old
        pct = (100.0 * delta / old) if old else 0.0
        cat = category(key)
        gate = "gated" if gated_category(cat) else "local"
        changed_lines.append(f"changed\t{cat}\t{key}\t{old}\t{new}\t{delta}\t{pct:.4f}\t{gate}")
    for key in comparison.added:
        cat = category(key)
        gate = "gated" if gated_category(cat) else "local"
        changed_lines.append(f"added\t{cat}\t{key}\t\t{current[key]}\t\t\t{gate}")
    for key in comparison.removed:
        cat = category(key)
        gate = "gated" if gated_category(cat) else "local"
        changed_lines.append(f"removed\t{cat}\t{key}\t{baseline[key]}\t\t\t\t{gate}")
    (report_dir / "changed.tsv").write_text("\n".join(changed_lines) + "\n")

    summary = {
        "baseline": str(BASELINE),
        "current_metrics": len(current),
        "baseline_metrics": len(baseline),
        "common": len(comparison.common),
        "changed": len(comparison.changed),
        "added": len(comparison.added),
        "removed": len(comparison.removed),
        "gated_drift": has_gated_drift(comparison),
        "categories": categories,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> int:
    options = parse_args(sys.argv[1:])
    mode = options.mode
    current = run_harness()

    if mode == "--update":
        write_baseline(current)
        print(f"metrics-check: wrote baseline ({len(current)} metrics) to {BASELINE.relative_to(ROOT)}")
        return 0

    baseline = load(BASELINE)
    if not baseline:
        print("metrics-check: no baseline yet — run with --update to create it.")
        return 0

    comparison = compare(current, baseline)
    print_comparison(current, baseline, comparison)
    if options.report_dir is not None:
        write_report(options.report_dir, current, baseline, comparison)
        print(f"metrics-check: wrote report to {options.report_dir}")

    if mode == "--check" and (comparison.changed or comparison.added or comparison.removed):
        print("metrics-check: metrics drift — run --update and review if intentional.", file=sys.stderr)
        return 1
    if mode == "--check-size" and has_gated_drift(comparison):
        print("metrics-check: bytecode-size drift — run --update and review if intentional.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
