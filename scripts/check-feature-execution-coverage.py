#!/usr/bin/env python3
"""Validate the feature coverage manifest.

Fail-closed in both directions:
  - every feature has executed spec(s), negative expected-diagnostic spec(s),
    or an explicit skip reason;
  - every tests/conformance/*.spec.toml is claimed by at least one feature.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/conformance/feature_coverage.json"
CONFORMANCE_DIR = ROOT / "tests/conformance"
REQUIRED_CI_COMMANDS = {
    "zig build check-feature-execution-coverage",
    "zig build check-negative-corpus",
    "zig build test-conformance",
    "zig build gate",
}
ALLOWED_FEATURE_KEYS = {"id", "area", "behavior", "specs", "negative_specs", "skip"}


def fail(message: str) -> None:
    print(f"check-feature-execution-coverage: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{context} must be a non-empty string")
    return value


def validate_spec_ref(ref: str, context: str, claimed: set[str]) -> None:
    path_part, sep, fragment = ref.partition("#")
    path = ROOT / path_part
    if not path.exists():
        fail(f"{context} references missing file: {path_part}")
    if path.suffix == ".zig":
        if not sep or not fragment:
            fail(f"{context} references a Zig harness file without #fragment: {ref}")
        if fragment not in path.read_text():
            fail(f"{context} references missing fragment {fragment!r} in {path_part}")
        return
    if path.suffix != ".toml" or not path_part.endswith(".spec.toml"):
        fail(f"{context} must reference a .spec.toml or harness .zig#fragment, got: {ref}")
    if sep:
        fail(f"{context} spec reference must not carry a fragment: {ref}")
    claimed.add(path.name)


def validate_negative_ref(ref: str, context: str) -> None:
    path = ROOT / ref
    if not path.exists():
        fail(f"{context} references missing negative source: {ref}")
    if path.suffix != ".ora" or path.parent != ROOT / "tests" / "negative":
        fail(f"{context}.negative_specs must reference tests/negative/*.ora, got: {ref}")
    sidecar = path.with_suffix(".expect.toml")
    if not sidecar.exists():
        fail(f"{context} negative source is missing sidecar: {sidecar.relative_to(ROOT)}")


def main() -> None:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        fail(f"manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as err:
        fail(f"invalid JSON: {err}")

    ci_commands = manifest.get("ci_commands")
    if not isinstance(ci_commands, list):
        fail("ci_commands must be a list")
    missing_ci = sorted(REQUIRED_CI_COMMANDS - set(ci_commands))
    if missing_ci:
        fail("missing CI/gate command(s): " + ", ".join(missing_ci))

    features = manifest.get("features")
    if not isinstance(features, list) or not features:
        fail("features must be a non-empty list")

    seen_ids: set[str] = set()
    claimed_specs: set[str] = set()
    covered = 0
    negative_covered = 0
    skipped: list[str] = []

    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            fail(f"features[{index}] must be an object")
        unknown = set(feature) - ALLOWED_FEATURE_KEYS
        if unknown:
            fail(f"features[{index}] has unknown key(s): {', '.join(sorted(unknown))}")

        feature_id = require_string(feature.get("id"), f"features[{index}].id")
        if feature_id in seen_ids:
            fail(f"duplicate feature id: {feature_id}")
        seen_ids.add(feature_id)
        require_string(feature.get("area"), f"{feature_id}.area")

        has_specs = "specs" in feature
        has_negative_specs = "negative_specs" in feature
        has_skip = "skip" in feature
        if sum([has_specs, has_negative_specs, has_skip]) != 1:
            fail(f"{feature_id} must have exactly one of 'specs', 'negative_specs', or 'skip'")

        if has_skip:
            require_string(feature.get("skip"), f"{feature_id}.skip")
            skipped.append(feature_id)
            continue

        require_string(feature.get("behavior"), f"{feature_id}.behavior")
        if has_negative_specs:
            specs = feature.get("negative_specs")
            if not isinstance(specs, list) or not specs:
                fail(f"{feature_id}.negative_specs must be a non-empty list")
            for ref_index, ref in enumerate(specs):
                ref_str = require_string(ref, f"{feature_id}.negative_specs[{ref_index}]")
                validate_negative_ref(ref_str, feature_id)
            negative_covered += 1
        else:
            specs = feature.get("specs")
            if not isinstance(specs, list) or not specs:
                fail(f"{feature_id}.specs must be a non-empty list")
            for ref_index, ref in enumerate(specs):
                ref_str = require_string(ref, f"{feature_id}.specs[{ref_index}]")
                validate_spec_ref(ref_str, feature_id, claimed_specs)
            covered += 1

    all_specs = {p.name for p in CONFORMANCE_DIR.glob("*.spec.toml")}
    unclaimed = sorted(all_specs - claimed_specs)
    if unclaimed:
        fail(
            "spec(s) not claimed by any feature (add a feature entry or extend one): "
            + ", ".join(unclaimed)
        )

    print(
        f"check-feature-execution-coverage: ok "
        f"({covered} executed, {negative_covered} negative, {len(skipped)} skipped of {len(features)} features; "
        f"{len(all_specs)} specs all claimed)"
    )
    if skipped:
        print("  open coverage debt: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
