#!/usr/bin/env python3
"""Validate the OraToSIR de-bloat coverage manifest."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/oratosir_debloat_coverage.json"
REQUIRED_CI_COMMANDS = {
    "zig build check-oratosir-coverage",
    "zig build check-mlir-sir",
    "zig build check-sir-text",
    "zig build test-conformance",
    "zig build test-evm",
    "zig build gate-oratosir-debloat",
}
ZIG_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def fail(message: str) -> "None":
    print(f"check-oratosir-coverage: {message}", file=sys.stderr)
    raise SystemExit(1)


def split_ref(ref: str) -> tuple[Path, str | None]:
    path, sep, fragment = ref.partition("#")
    if not path:
        fail(f"empty path reference in {ref!r}")
    if sep and not fragment:
        fail(f"empty fragment in reference: {ref}")
    return ROOT / path, fragment if sep else None


def validate_zig_fragment(path: Path, text: str, fragment: str, context: str) -> None:
    if " " in fragment:
        if f'test "{fragment}"' in text:
            return
        fail(f"{context} references missing Zig test {fragment!r} in {path.relative_to(ROOT)}")

    if not ZIG_SYMBOL_RE.fullmatch(fragment):
        fail(f"{context} has invalid Zig fragment {fragment!r}")

    symbol = fragment.rsplit(".", 1)[-1]
    symbol_re = re.compile(rf"\b(?:pub\s+)?(?:inline\s+)?(?:fn|const|var)\s+{re.escape(symbol)}\b")
    if symbol_re.search(text) or f'test "{fragment}"' in text:
        return
    fail(f"{context} references missing Zig symbol {fragment!r} in {path.relative_to(ROOT)}")


def validate_reference(ref: str, context: str, *, is_sir_golden: bool = False) -> None:
    path, fragment = split_ref(ref)
    if not path.exists():
        fail(f"{context} references missing file: {path.relative_to(ROOT)}")

    if path.suffix == ".zig" and fragment is None:
        fail(f"{context} references Zig file without #test-or-symbol: {ref}")
    if is_sir_golden and fragment is None:
        fail(f"{context} references SIR golden without #CHECK-fragment: {ref}")
    if fragment is None:
        return

    text = path.read_text()
    if path.suffix == ".zig":
        validate_zig_fragment(path, text, fragment, context)
    elif path.suffix == ".check":
        if any(line.startswith("// CHECK") and fragment in line for line in text.splitlines()):
            return
        fail(f"{context} references missing CHECK fragment {fragment!r} in {path.relative_to(ROOT)}")
    elif fragment not in text:
        fail(f"{context} references missing fragment {fragment!r} in {path.relative_to(ROOT)}")


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

        for ref in executing_tests:
            validate_reference(ref, pattern_id)
        for ref in sir_goldens:
            validate_reference(ref, pattern_id, is_sir_golden=True)

    if high_churn_count == 0:
        fail("no high-churn patterns listed")

    print(f"check-oratosir-coverage: ok ({len(patterns)} patterns, {high_churn_count} high-churn)")


if __name__ == "__main__":
    main()
