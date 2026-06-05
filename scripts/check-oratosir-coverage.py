#!/usr/bin/env python3
"""Validate the OraToSIR de-bloat coverage manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/oratosir_debloat_coverage.json"
REQUIRED_CI_COMMANDS = {
    "zig build check-oratosir-coverage",
    "zig build check-mlir-sir",
    "zig build test-conformance",
    "zig build test-evm",
    "zig build gate-oratosir-debloat",
}


def fail(message: str) -> "None":
    print(f"check-oratosir-coverage: {message}", file=sys.stderr)
    raise SystemExit(1)


def repo_path(ref: str) -> Path:
    path = ref.split("#", 1)[0]
    if not path:
        fail(f"empty path reference in {ref!r}")
    return ROOT / path


def require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{context} must be a non-empty string")
    return value


def require_string_list(value: object, context: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        fail(f"{context} must be a list")
    if not allow_empty and not value:
        fail(f"{context} must not be empty")
    out: list[str] = []
    for index, item in enumerate(value):
        out.append(require_string(item, f"{context}[{index}]"))
    return out


def main() -> None:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        fail(f"manifest not found: {manifest_path.relative_to(ROOT)}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as err:
        fail(f"invalid JSON: {err}")

    ci_commands = set(require_string_list(manifest.get("ci_commands"), "ci_commands"))
    missing_ci = sorted(REQUIRED_CI_COMMANDS - ci_commands)
    if missing_ci:
        fail("missing CI/gate command(s): " + ", ".join(missing_ci))

    patterns = manifest.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        fail("patterns must be a non-empty list")

    seen_ids: set[str] = set()
    high_churn_count = 0
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, dict):
            fail(f"patterns[{index}] must be an object")

        pattern_id = require_string(pattern.get("id"), f"patterns[{index}].id")
        if pattern_id in seen_ids:
            fail(f"duplicate pattern id: {pattern_id}")
        seen_ids.add(pattern_id)

        for key in ("tier", "area", "conversion"):
            require_string(pattern.get(key), f"{pattern_id}.{key}")

        executing_tests = require_string_list(pattern.get("executing_tests"), f"{pattern_id}.executing_tests")
        sir_goldens = require_string_list(pattern.get("sir_goldens", []), f"{pattern_id}.sir_goldens", allow_empty=True)

        high_churn = bool(pattern.get("high_churn", False))
        if high_churn:
            high_churn_count += 1
            if not sir_goldens:
                fail(f"{pattern_id} is high-churn and must list at least one SIR golden")

        for ref in executing_tests + sir_goldens:
            path = repo_path(ref)
            if not path.exists():
                fail(f"{pattern_id} references missing file: {path.relative_to(ROOT)}")

    if high_churn_count == 0:
        fail("no high-churn patterns listed")

    print(f"check-oratosir-coverage: ok ({len(patterns)} patterns, {high_churn_count} high-churn)")


if __name__ == "__main__":
    main()
