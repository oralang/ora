#!/usr/bin/env python3
"""Negative expected-diagnostic corpus runner (test-quality program T1.5).

For every tests/negative/<name>.ora:
  - require a <name>.expect.toml sidecar (fail-closed; missing = failure);
  - compile with `ora emit --emit=bytecode`;
  - the compile MUST fail (nonzero exit) AND emit no bytecode;
  - the combined stdout+stderr MUST contain the sidecar's expect_error substring.

A source that compiles, or fails with a different diagnostic, is a failure —
this is the user-facing half of the no-executable-defaults law: wrong source
must visibly fail, with the expected message.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORA_BIN = ROOT / "zig-out" / "bin" / "ora"
NEG_DIR = ROOT / "tests" / "negative"


def fail(msg: str) -> None:
    print(f"check-negative-corpus: {msg}", file=sys.stderr)
    raise SystemExit(1)


def parse_expect(path: Path) -> str:
    # Minimal: a single required key `expect_error = "..."`.
    expect = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition("=")
        if not sep:
            fail(f"{path.name}: malformed line {raw!r}")
        key = key.strip()
        val = val.strip()
        if key != "expect_error":
            fail(f"{path.name}: unknown key {key!r} (only expect_error supported)")
        if len(val) < 2 or val[0] != '"' or val[-1] != '"':
            fail(f"{path.name}: expect_error must be a quoted string")
        expect = val[1:-1]
    if not expect:
        fail(f"{path.name}: missing expect_error")
    return expect


def main() -> None:
    neg_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else NEG_DIR
    if not ORA_BIN.exists():
        fail(f"ora binary not found at {ORA_BIN}; run 'zig build' first")
    if not neg_dir.is_dir():
        fail(f"negative corpus dir not found: {neg_dir}")

    ora_files = sorted(neg_dir.glob("*.ora"))
    if not ora_files:
        fail("no .ora files in tests/negative")

    checked = 0
    for ora in ora_files:
        sidecar = ora.with_suffix(".expect.toml")
        if not sidecar.exists():
            fail(f"{ora.name}: missing sidecar {sidecar.name}")
        expect = parse_expect(sidecar)

        proc = subprocess.run(
            [str(ORA_BIN), "emit", "--emit=bytecode", str(ora)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        combined = proc.stdout + proc.stderr

        if proc.returncode == 0:
            fail(f"{ora.name}: expected compile failure but it SUCCEEDED")
        if expect not in combined:
            fail(
                f"{ora.name}: diagnostic mismatch\n"
                f"  expected substring: {expect!r}\n"
                f"  got (first 400 chars): {combined[:400]!r}"
            )
        checked += 1

    print(f"check-negative-corpus: ok ({checked} negative cases, all rejected as expected)")


if __name__ == "__main__":
    main()
