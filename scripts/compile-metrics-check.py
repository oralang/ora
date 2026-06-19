#!/usr/bin/env python3
"""Compiler frontend metrics benchmark.

Runs the compile-metrics harness over the expected-success Ora example corpus
and compares deterministic Tier-A metrics against a committed baseline.

  scripts/compile-metrics-check.py            # diff against baseline
  scripts/compile-metrics-check.py --update   # rewrite baseline
  scripts/compile-metrics-check.py --check    # exit 1 on any Tier-A drift
  scripts/compile-metrics-check.py --time-output /tmp/now.txt --time-baseline /tmp/before.txt

Baseline: tests/compile_metrics_snapshot.txt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "zig-out" / "bin" / "compile-metrics"
BASELINE = ROOT / "tests" / "compile_metrics_snapshot.txt"


@dataclass
class Options:
    mode: str = "--diff"
    time_output: Path | None = None
    time_baseline: Path | None = None


def parse_args(argv: list[str]) -> Options:
    options = Options()
    mode_seen = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in {"--diff", "--update", "--check"}:
            if mode_seen:
                raise SystemExit("compile-metrics-check: choose only one of --diff, --update, or --check")
            mode_seen = True
            options.mode = arg
        elif arg == "--time-output":
            index += 1
            if index >= len(argv):
                raise SystemExit("compile-metrics-check: --time-output requires a path")
            options.time_output = Path(argv[index])
        elif arg.startswith("--time-output="):
            options.time_output = Path(arg.split("=", 1)[1])
        elif arg == "--time-baseline":
            index += 1
            if index >= len(argv):
                raise SystemExit("compile-metrics-check: --time-baseline requires a path")
            options.time_baseline = Path(argv[index])
        elif arg.startswith("--time-baseline="):
            options.time_baseline = Path(arg.split("=", 1)[1])
        else:
            raise SystemExit("usage: compile-metrics-check.py [--diff|--update|--check] [--time-output PATH] [--time-baseline PATH]")
        index += 1
    return options


def metric_kind(key: str) -> str:
    return key.rsplit("::", 1)[-1]


def category(key: str) -> str:
    kind = metric_kind(key)
    if kind == "__bytes_peak":
        return "bytes_peak"
    if kind == "__source_bytes":
        return "source_bytes"
    if kind == "__source_lines":
        return "source_lines"
    return kind


def lower_is_better(cat: str) -> bool:
    return cat in {"alloc_calls", "bytes_allocated", "bytes_peak"}


def run_harness(time_output: Path | None) -> dict[str, int]:
    if not HARNESS.exists():
        print("compile-metrics-check: build the harness first: zig build compile-metrics", file=sys.stderr)
        raise SystemExit(2)
    argv = [str(HARNESS)]
    if time_output is not None:
        argv += ["--time-output", str(time_output)]
    process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if process.returncode != 0:
        print(f"compile-metrics-check: harness failed:\n{process.stderr[:1600]}", file=sys.stderr)
        raise SystemExit(1)
    return parse_metrics(process.stdout)


def parse_metrics(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        if "\t" not in line or line.startswith("#"):
            continue
        key, _, value = line.rpartition("\t")
        out[key] = int(value)
    return out


def load(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return parse_metrics(path.read_text())


def write_baseline(current: dict[str, int]) -> None:
    header = [
        "# compiler metrics baseline — <key>\\t<value>.",
        "# Deterministic Tier-A only: invocations, work_count, alloc_calls, bytes_allocated, __source_bytes, __source_lines, __bytes_peak.",
        "# Regenerate with scripts/compile-metrics-check.py --update and REVIEW the diff before committing.",
    ]
    body = [f"{key}\t{value}" for key, value in sorted(current.items())]
    BASELINE.write_text("\n".join(header + body) + "\n")


def noise_key_for(key: str) -> str:
    if key.endswith("::wall_ns_min"):
        return key[: -len("::wall_ns_min")] + "::wall_ns_noise"
    if key.endswith("::wall_ns_median"):
        return key[: -len("::wall_ns_median")] + "::wall_ns_noise"
    return key


def compare_time_outputs(current_path: Path, baseline_path: Path) -> None:
    if not baseline_path.exists():
        print(f"compile-metrics-check: no time baseline at {baseline_path}; wrote current timing to {current_path}")
        return
    current = load(current_path)
    baseline = load(baseline_path)
    common = sorted(
        key for key in (set(current) & set(baseline))
        if key.endswith("::wall_ns_min") or key.endswith("::wall_ns_median")
    )
    better = worse = within_noise = 0
    notable: list[tuple[int, str, int, int, int]] = []
    for key in common:
        delta = current[key] - baseline[key]
        noise_key = noise_key_for(key)
        noise_floor = max(current.get(noise_key, 0), baseline.get(noise_key, 0))
        if abs(delta) <= noise_floor:
            within_noise += 1
            continue
        if delta < 0:
            better += 1
        else:
            worse += 1
        notable.append((abs(delta), key, baseline[key], current[key], noise_floor))

    print("compile-metrics-check: local time verdict:")
    print(f"    {better} better, {worse} worse, {within_noise} within noise over {len(common)} common timing metrics")
    for _, key, old, new, noise_floor in sorted(notable, reverse=True)[:20]:
        delta = new - old
        print(f"    {'▲' if delta > 0 else '▼'} {key}: {old} -> {new} ({delta:+d}, noise <= {noise_floor})")


def main() -> int:
    options = parse_args(sys.argv[1:])
    mode = options.mode

    baseline = load(BASELINE)
    if mode != "--update" and not baseline:
        print("compile-metrics-check: no baseline yet — run with --update to create it.")
        return 0

    current = run_harness(options.time_output)
    if options.time_output is not None and options.time_baseline is not None:
        compare_time_outputs(options.time_output, options.time_baseline)
    elif options.time_output is not None:
        print(f"compile-metrics-check: wrote local timing metrics to {options.time_output}")

    if mode == "--update":
        write_baseline(current)
        print(f"compile-metrics-check: wrote baseline ({len(current)} metrics) to {BASELINE.relative_to(ROOT)}")
        return 0

    common = set(current) & set(baseline)
    changed = sorted(key for key in common if current[key] != baseline[key])
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))

    for key in changed:
        delta = current[key] - baseline[key]
        print(f"  {'▲' if delta > 0 else '▼'} {key}: {baseline[key]} -> {current[key]} ({delta:+d})")
    for key in added:
        print(f"  + {key}: {current[key]} (new)")
    for key in removed:
        print(f"  - {key}: {baseline[key]} (gone)")

    totals: dict[str, list[int]] = {}
    for key in common:
        cat = category(key)
        totals.setdefault(cat, [0, 0])
        totals[cat][0] += baseline[key]
        totals[cat][1] += current[key]

    print("compile-metrics-check: verdict by category (common set):")
    for cat in sorted(totals):
        base_total, current_total = totals[cat]
        delta = current_total - base_total
        pct = (100.0 * delta / base_total) if base_total else 0.0
        if lower_is_better(cat):
            verdict = "BETTER" if delta < 0 else ("WORSE" if delta > 0 else "same")
        else:
            verdict = "changed" if delta != 0 else "same"
        print(f"    {cat:16} {base_total} -> {current_total} ({delta:+d}, {pct:+.2f}%)  [{verdict}]")
    print(f"    => {len(changed)} changed, {len(added)} new, {len(removed)} removed")

    if mode == "--check" and (changed or added or removed):
        print("compile-metrics-check: metrics drift — run --update and review if intentional.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
