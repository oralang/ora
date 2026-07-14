#!/usr/bin/env python3
"""Emit deterministic Sinora dispatcher metrics as TSV rows.

The backend gas/bytecode snapshot is source-level and broad. This harness is
backend-level and narrow: each fixture is a hand-written SIR switch shape that
exercises one dispatcher routing strategy. The output format intentionally
matches tests/conformance/metrics_snapshot.zig so backend-metrics-compare.sh can
track these rows in the same run history.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "sinora" / "fixtures" / "dispatcher_metrics"
DEFAULT_SINORA = ROOT / "sinora" / "zig-out" / "bin" / "sinora"


def int_field(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key)
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raise TypeError(f"metric field {key!r} is not an integer: {value!r}")


def dense_kind_id(best_dense: dict[str, Any] | None) -> int:
    if best_dense is None:
        return 0
    kind = best_dense.get("kind")
    if kind == "bit_window":
        return 1
    if kind == "multiplicative":
        return 2
    raise TypeError(f"unknown dense kind: {kind!r}")


def emit_row(key: str, value: int) -> None:
    print(f"{key}\t{value}")


def emit_metrics_for_fixture(sinora: Path, fixture: Path, tmp_dir: Path) -> None:
    metrics_path = tmp_dir / f"{fixture.stem}.json"
    command = [
        str(sinora),
        "emit-release",
        "--metrics",
        str(metrics_path),
        str(fixture),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(f"sinora-dispatcher-metrics: {fixture} failed\n")
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)

    payload = json.loads(metrics_path.read_text())
    routing = payload["switch_routing"]
    best_sparse = routing.get("best_sparse")
    best_dense = routing.get("best_dense")
    prefix = f"dispatcher_metrics::{fixture.stem}"

    emit_row(f"{prefix}::bytecode_bytes", int_field(payload, "bytecode_bytes"))
    emit_row(f"{prefix}::switches", int_field(routing, "switches"))
    emit_row(f"{prefix}::cases", int_field(routing, "cases"))
    emit_row(f"{prefix}::largest_switch_cases", int_field(routing, "largest_switch_cases"))
    emit_row(f"{prefix}::linear_worst_checks", int_field(routing, "linear_worst_checks"))
    emit_row(
        f"{prefix}::linear_known_selector_avg_checks_x1000",
        int_field(routing, "linear_known_selector_avg_checks_x1000"),
    )
    emit_row(f"{prefix}::selector_width_candidates", int_field(routing, "selector_width_candidates"))
    emit_row(f"{prefix}::chosen_linear", int_field(routing, "chosen_linear"))
    emit_row(f"{prefix}::chosen_sparse", int_field(routing, "chosen_sparse"))
    emit_row(f"{prefix}::chosen_dense", int_field(routing, "chosen_dense"))
    emit_row(f"{prefix}::sparse_candidates", int_field(routing, "sparse_candidates"))
    emit_row(f"{prefix}::dense_candidates", int_field(routing, "dense_candidates"))

    emit_row(f"{prefix}::best_sparse_bucket_bits", int_field(best_sparse or {}, "bucket_bits"))
    emit_row(f"{prefix}::best_sparse_bucket_shift", int_field(best_sparse or {}, "bucket_shift"))
    emit_row(f"{prefix}::best_sparse_bucket_count", int_field(best_sparse or {}, "bucket_count"))
    emit_row(f"{prefix}::best_sparse_max_bucket_size", int_field(best_sparse or {}, "max_bucket_size"))
    emit_row(
        f"{prefix}::best_sparse_bucket_dispatch_avg_checks_x1000",
        int_field(best_sparse or {}, "sparse_bucket_dispatch_avg_checks_x1000"),
    )
    emit_row(
        f"{prefix}::best_sparse_known_selector_avg_checks_x1000",
        int_field(best_sparse or {}, "sparse_known_selector_avg_checks_x1000"),
    )
    emit_row(
        f"{prefix}::best_sparse_total_avg_checks_x1000",
        int_field(best_sparse or {}, "sparse_total_avg_checks_x1000"),
    )

    emit_row(f"{prefix}::best_dense_kind_id", dense_kind_id(best_dense))
    emit_row(f"{prefix}::best_dense_table_slots", int_field(best_dense or {}, "table_slots"))
    emit_row(f"{prefix}::best_dense_hole_slots", int_field(best_dense or {}, "hole_slots"))
    emit_row(f"{prefix}::best_dense_load_factor_x1000", int_field(best_dense or {}, "load_factor_x1000"))
    emit_row(
        f"{prefix}::best_dense_runtime_selector_eq_checks",
        int_field(best_dense or {}, "runtime_selector_eq_checks"),
    )
    emit_row(
        f"{prefix}::best_dense_dispatch_avg_checks_x1000",
        int_field(best_dense or {}, "dense_dispatch_avg_checks_x1000"),
    )
    emit_row(
        f"{prefix}::best_dense_total_avg_checks_x1000",
        int_field(best_dense or {}, "dense_total_avg_checks_x1000"),
    )


def main() -> int:
    sinora = Path(os.environ.get("SINORA", str(DEFAULT_SINORA)))
    corpus = Path(os.environ.get("SINORA_DISPATCHER_METRICS_CORPUS", str(DEFAULT_CORPUS)))

    if not sinora.exists():
        sys.stderr.write(f"sinora-dispatcher-metrics: missing Sinora binary: {sinora}\n")
        sys.stderr.write("run `zig build sinora` first\n")
        return 2
    if not corpus.exists():
        sys.stderr.write(f"sinora-dispatcher-metrics: missing corpus directory: {corpus}\n")
        return 2

    fixtures = sorted(corpus.glob("*.sir"))
    if not fixtures:
        sys.stderr.write(f"sinora-dispatcher-metrics: no .sir fixtures under {corpus}\n")
        return 2

    with tempfile.TemporaryDirectory(prefix="sinora-dispatcher-metrics-") as tmp:
        tmp_dir = Path(tmp)
        for fixture in fixtures:
            emit_metrics_for_fixture(sinora, fixture, tmp_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
