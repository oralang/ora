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

Baseline: tests/conformance/metrics_snapshot.txt  (committed)
Metric category is keyed by suffix: keys containing '::__bytecode_bytes' are the
size category; everything else is gas.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "zig-out" / "bin" / "metrics-snapshot"
BASELINE = ROOT / "tests" / "conformance" / "metrics_snapshot.txt"
VALID_MODES = {"--diff", "--update", "--check"}


def category(key: str) -> str:
    return "bytecode_bytes" if key.endswith("::__bytecode_bytes") else "gas"


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


def main() -> int:
    mode = "--diff"
    mode_seen = False
    for arg in sys.argv[1:]:
        if arg not in VALID_MODES:
            print("usage: metrics-check.py [--diff|--update|--check]", file=sys.stderr)
            return 2
        if mode_seen:
            print("metrics-check: choose only one of --diff, --update, or --check", file=sys.stderr)
            return 2
        mode_seen = True
        mode = arg
    current = run_harness()

    if mode == "--update":
        write_baseline(current)
        print(f"metrics-check: wrote baseline ({len(current)} metrics) to {BASELINE.relative_to(ROOT)}")
        return 0

    baseline = load(BASELINE)
    if not baseline:
        print("metrics-check: no baseline yet — run with --update to create it.")
        return 0

    common = set(current) & set(baseline)
    changed = sorted(k for k in common if current[k] != baseline[k])
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))

    for k in changed:
        d = current[k] - baseline[k]
        print(f"  {'▲' if d > 0 else '▼'} {k}: {baseline[k]} -> {current[k]} ({d:+d})")
    for k in added:
        print(f"  + {k}: {current[k]} (new)")
    for k in removed:
        print(f"  - {k}: {baseline[k]} (gone)")

    # Per-category totals over the common set — the better/worse verdict.
    cats: dict[str, list[int]] = {}
    for k in common:
        cats.setdefault(category(k), [0, 0])
        cats[category(k)][0] += baseline[k]
        cats[category(k)][1] += current[k]

    print("metrics-check: verdict by category (common set):")
    overall = "no change"
    for cat in sorted(cats):
        base_t, cur_t = cats[cat]
        d = cur_t - base_t
        pct = (100.0 * d / base_t) if base_t else 0.0
        verdict = "BETTER" if d < 0 else ("WORSE" if d > 0 else "same")
        print(f"    {cat:16} {base_t} -> {cur_t} ({d:+d}, {pct:+.2f}%)  [{verdict}]")
        if d != 0:
            overall = "mixed/changed"
    print(f"    => {len(changed)} changed, {len(added)} new, {len(removed)} removed ({overall})")

    if mode == "--check" and (changed or added or removed):
        print("metrics-check: metrics drift — run --update and review if intentional.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
