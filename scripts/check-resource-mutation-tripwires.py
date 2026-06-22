#!/usr/bin/env python3
"""Ensure resource lowering mutation tripwires stay explicit.

The resource feature's highest-risk compiler mutations are implementation-level:
removing the dynamic-alias branch, writing one side before all checks complete, or
dropping the signed amount polarity guard. We rely on SIR/FileCheck snapshots to
catch those mutations. This script keeps those snapshots from being weakened into
generic "resource compiles" checks.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"check-resource-mutation-tripwires: {message}", file=sys.stderr)
    raise SystemExit(1)


def read(path: str) -> str:
    full = ROOT / path
    if not full.is_file():
        fail(f"missing required tripwire file: {path}")
    return full.read_text(encoding="utf-8")


def require_ordered(path: str, needles: tuple[str, ...]) -> None:
    text = read(path)
    cursor = 0
    for needle in needles:
        index = text.find(needle, cursor)
        if index < 0:
            fail(f"{path}: missing ordered tripwire {needle!r}")
        cursor = index + len(needle)


def require_absent(path: str, needle: str) -> None:
    text = read(path)
    if needle in text:
        fail(f"{path}: forbidden text present: {needle!r}")


def main() -> int:
    require_ordered(
        "tests/sir_text/resource_move_dynamic_alias.check",
        (
            "eq hash_balances hash_balances_0",
            "=> [[SAME]] ? @bb5 : @bb7",
            "// Same-place path checks source sufficiency and performs no durable write.",
            "bb5 {",
            "sload hash_balances",
            "lt [[SAME_VALUE]] v2",
            "=> [[SAME_OK]] ? @bb6 : @bb0",
            "bb6 {",
            "=> @bb10",
            "// Distinct-place path loads both places, checks subtract/add, then writes both.",
            "bb7 {",
            "sstore hash_balances [[SRC_AFTER]]",
            "sstore hash_balances_0 [[DST_AFTER]]",
        ),
    )

    require_ordered(
        "tests/mlir_sir/resource_move_destination_overflow_atomic.check",
        (
            "[[SRC:%[0-9]+]] = sir.sload %hash_balances",
            "[[DST:%[0-9]+]] = sir.sload %hash_balances_{{[0-9]+}}",
            "[[SRC_AFTER:%[0-9]+]] = sir.sub [[SRC]]",
            "[[SRC_OK:%[0-9]+]] = sir.iszero",
            "sir.cond_br [[SRC_OK]]",
            "[[DST_AFTER:%[0-9]+]] = sir.add [[DST]]",
            "[[DST_OVERFLOW:%[0-9]+]] = sir.lt [[DST_AFTER]]",
            "[[DST_OK:%[0-9]+]] = sir.iszero [[DST_OVERFLOW]]",
            "sir.cond_br [[DST_OK]]",
            "sir.sstore %hash_balances",
            "sir.sstore %hash_balances_{{[0-9]+}}",
        ),
    )

    require_ordered(
        "tests/mlir_sir/resource_signed_amount_guard.check",
        (
            "[[NEG_AMOUNT:%[0-9]+]] = sir.slt %arg1",
            "[[NON_NEG:%[0-9]+]] = sir.iszero [[NEG_AMOUNT]]",
            "sir.cond_br [[NON_NEG]]",
            "[[AFTER:%[0-9]+]] = sir.add [[BEFORE]]",
            "sir.sstore %hash_debts",
        ),
    )

    require_absent("tests/mlir_sir/resource_signed_amount_guard.check", "CHECK-NOT: sir.slt")
    print("check-resource-mutation-tripwires: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
