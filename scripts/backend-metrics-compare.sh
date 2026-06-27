#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${ORA_BACKEND_METRICS_HARNESS:-$ROOT/zig-out/bin/metrics-snapshot}"
DISPATCHER_METRICS="${ORA_BACKEND_DISPATCHER_METRICS:-$ROOT/scripts/sinora-dispatcher-metrics.py}"
SNAPSHOT_ROOT="${ORA_BACKEND_METRICS_DIR:-$ROOT/tests/conformance/backend_metrics}"
RUNS_DIR="$SNAPSHOT_ROOT/runs"
RUN_ID=""
BUILD_HARNESS=1
VARIANTS=()

usage() {
  cat >&2 <<'USAGE'
usage: scripts/backend-metrics-compare.sh [--run-id ID] [--out-dir DIR] [--no-build] [variant...]

Runs the conformance gas/bytecode metrics harness once per snapshot label and
writes Sinora dispatcher-shape metrics from sinora/fixtures/dispatcher_metrics.
writes a non-overwriting snapshot under:

  tests/conformance/backend_metrics/runs/<run-id>/

Defaults: sinora.

Environment:
  ORA_BACKEND_METRICS_DIR      override snapshot root
  ORA_BACKEND_METRICS_RUN_ID   override generated run id
  ORA_BACKEND_METRICS_HARNESS  override metrics harness executable
  ORA_BACKEND_DISPATCHER_METRICS override dispatcher metrics script
USAGE
}

while (($#)); do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --run-id)
      [[ $# -ge 2 ]] || { echo "backend-metrics-compare: --run-id needs a value" >&2; exit 2; }
      RUN_ID="$2"
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || { echo "backend-metrics-compare: --out-dir needs a value" >&2; exit 2; }
      SNAPSHOT_ROOT="$2"
      RUNS_DIR="$SNAPSHOT_ROOT/runs"
      shift 2
      ;;
    --no-build)
      BUILD_HARNESS=0
      shift
      ;;
    --*)
      echo "backend-metrics-compare: unknown option '$1'" >&2
      usage
      exit 2
      ;;
    *)
      VARIANTS+=("$1")
      shift
      ;;
  esac
done

if ((${#VARIANTS[@]} == 0)); then
  VARIANTS=(sinora)
fi

if ((BUILD_HARNESS)); then
  (cd "$ROOT" && zig build metrics-snapshot sinora >/dev/null)
elif [[ ! -x "$HARNESS" ]]; then
  echo "backend-metrics-compare: missing $HARNESS; run 'zig build metrics-snapshot'" >&2
  exit 2
fi
if [[ ! -f "$DISPATCHER_METRICS" ]]; then
  echo "backend-metrics-compare: missing dispatcher metrics script: $DISPATCHER_METRICS" >&2
  exit 2
fi

mkdir -p "$RUNS_DIR"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="${ORA_BACKEND_METRICS_RUN_ID:-}"
fi
if [[ -z "$RUN_ID" ]]; then
  git_sha="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo nogit)"
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$git_sha"
fi

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|"")
    echo "backend-metrics-compare: run id must use only [A-Za-z0-9._-]" >&2
    exit 2
    ;;
esac

RUN_DIR="$RUNS_DIR/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
  echo "backend-metrics-compare: refusing to overwrite existing snapshot: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

{
  printf 'run_id\t%s\n' "$RUN_ID"
  printf 'utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'git_head\t%s\n' "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf 'git_branch\t%s\n' "$(git -C "$ROOT" branch --show-current 2>/dev/null || echo unknown)"
  printf 'variants\t%s\n' "${VARIANTS[*]}"
} > "$RUN_DIR/manifest.tsv"

for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    *[!A-Za-z0-9._-]*|"")
      echo "backend-metrics-compare: variant must use only [A-Za-z0-9._-]: '$variant'" >&2
      exit 2
      ;;
  esac
  echo "backend-metrics-compare: running $variant"
  "$HARNESS" > "$RUN_DIR/$variant.tsv"
  python3 "$DISPATCHER_METRICS" >> "$RUN_DIR/$variant.tsv"
done

python3 - "$RUN_DIR" "${VARIANTS[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
variants = sys.argv[2:]


def load(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "\t" not in line:
            continue
        key, value = line.rsplit("\t", 1)
        out[key] = int(value)
    return out


def category(key: str) -> str:
    if key.startswith("dispatcher_metrics::"):
        # Heterogeneous planner rows: the category total is a compact drift
        # checksum, while changed.tsv keeps each metric key readable.
        return "dispatcher_metrics"
    if key.endswith("::__bytecode_bytes"):
        return "bytecode_bytes"
    if "::__deploy_gas" in key:
        return "deploy_gas"
    return "runtime_gas"


data = {variant: load(run_dir / f"{variant}.tsv") for variant in variants}
summary = {
    "variants": variants,
    "totals": {},
    "comparisons": [],
}

for variant, values in data.items():
    totals: dict[str, int] = {"metrics": len(values)}
    for key, value in values.items():
        totals[category(key)] = totals.get(category(key), 0) + value
    summary["totals"][variant] = totals

comparison_lines = ["base\tcandidate\tcategory\tkeys\tbase_total\tcandidate_total\tdelta\tpct"]
changed_lines = ["base\tcandidate\tkey\tbase\tcandidate\tdelta\tpct"]

for base in variants:
    for candidate in variants:
        if base == candidate:
            continue
        base_values = data[base]
        candidate_values = data[candidate]
        common = sorted(set(base_values) & set(candidate_values))
        added = sorted(set(candidate_values) - set(base_values))
        removed = sorted(set(base_values) - set(candidate_values))
        comparison = {
            "base": base,
            "candidate": candidate,
            "common": len(common),
            "added": len(added),
            "removed": len(removed),
            "categories": {},
        }
        for cat in ("bytecode_bytes", "deploy_gas", "runtime_gas", "dispatcher_metrics"):
            keys = [key for key in common if category(key) == cat]
            base_total = sum(base_values[key] for key in keys)
            candidate_total = sum(candidate_values[key] for key in keys)
            delta = candidate_total - base_total
            pct = (100.0 * delta / base_total) if base_total else 0.0
            comparison["categories"][cat] = {
                "keys": len(keys),
                "base_total": base_total,
                "candidate_total": candidate_total,
                "delta": delta,
                "pct": pct,
            }
            comparison_lines.append(
                f"{base}\t{candidate}\t{cat}\t{len(keys)}\t{base_total}\t{candidate_total}\t{delta}\t{pct:.4f}"
            )
        for key in common:
            if base_values[key] == candidate_values[key]:
                continue
            delta = candidate_values[key] - base_values[key]
            pct = (100.0 * delta / base_values[key]) if base_values[key] else 0.0
            changed_lines.append(
                f"{base}\t{candidate}\t{key}\t{base_values[key]}\t{candidate_values[key]}\t{delta}\t{pct:.4f}"
            )
        if added:
            (run_dir / f"{base}_to_{candidate}.added.txt").write_text("\n".join(added) + "\n")
        if removed:
            (run_dir / f"{base}_to_{candidate}.removed.txt").write_text("\n".join(removed) + "\n")
        summary["comparisons"].append(comparison)

(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(run_dir / "compare.tsv").write_text("\n".join(comparison_lines) + "\n")
(run_dir / "changed.tsv").write_text("\n".join(changed_lines) + "\n")

print((run_dir / "compare.tsv").read_text(), end="")
PY

printf '%s\n' "$RUN_ID" > "$SNAPSHOT_ROOT/latest"
echo "backend-metrics-compare: wrote snapshot $RUN_DIR"
