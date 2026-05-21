#!/usr/bin/env python3

"""Run a small mutation suite against Ora's verifier.

Each case starts from a contract that should verify. The script applies one
semantic mutation and expects the verifier to reject the mutant. A mutant that
still verifies is a soundness regression signal: the verifier accepted a
contract whose stated property no longer follows from the implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Iterable


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


BASE_COUNTER = """\
comptime const std = @import("std");

contract MutationCounter {
    storage var total: u256 = 0;

    pub fn inc(amount: u256)
        requires(amount <= 10)
        requires(total <= std.constants.U256_MAX - amount)
        ensures(total == old(total) + amount)
    {
        total = total + amount;
    }
}
"""


BASE_WITHDRAW = """\
contract MutationVault {
    storage var balance: u256 = 0;

    pub fn withdraw(amount: u256)
        requires(amount <= balance)
        ensures(balance == old(balance) - amount)
    {
        balance = balance - amount;
    }
}
"""


@dataclasses.dataclass(frozen=True)
class MutationCase:
    name: str
    source: str
    function: str
    find: str
    replace: str
    reason: str


CASES = (
    MutationCase(
        name="postcondition_plus_to_minus",
        source=BASE_COUNTER,
        function="inc",
        find="ensures(total == old(total) + amount)",
        replace="ensures(total == old(total) - amount)",
        reason="postcondition no longer matches the body",
    ),
    MutationCase(
        name="body_add_to_sub",
        source=BASE_COUNTER,
        function="inc",
        find="total = total + amount;",
        replace="total = total - amount;",
        reason="body may underflow and no longer proves the postcondition",
    ),
    MutationCase(
        name="requires_weakening_underflow",
        source=BASE_WITHDRAW,
        function="withdraw",
        find="requires(amount <= balance)",
        replace="requires(true)",
        reason="weakened precondition permits checked subtraction underflow",
    ),
    MutationCase(
        name="vacuous_requires",
        source=BASE_COUNTER,
        function="inc",
        find="requires(amount <= 10)",
        replace="requires(false)",
        reason="contradictory assumptions must fail vacuity checks",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler",
        default=str(PROJECT_ROOT / "zig-out" / "bin" / "ora"),
        help="Ora compiler binary (default: ./zig-out/bin/ora)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="per compiler invocation timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="keep generated mutation files for debugging",
    )
    return parser.parse_args()


def compiler_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ORA_Z3_TIMEOUT_MS", "10000")
    return env


def run_verifier(
    compiler: pathlib.Path,
    source_path: pathlib.Path,
    artifact_dir: pathlib.Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(compiler),
            "build",
            "-o",
            str(artifact_dir),
            str(source_path),
        ],
        cwd=PROJECT_ROOT,
        env=compiler_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def mutate(case: MutationCase) -> str:
    if case.source.count(case.find) != 1:
        raise ValueError(f"{case.name}: mutation needle must appear exactly once")
    mutated = case.source.replace(case.find, case.replace, 1)
    if mutated == case.source:
        raise ValueError(f"{case.name}: mutation produced identical source")
    return mutated


def print_failure_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print("stdout:", file=sys.stderr)
        print(result.stdout[-4000:], file=sys.stderr)
    if result.stderr:
        print("stderr:", file=sys.stderr)
        print(result.stderr[-4000:], file=sys.stderr)


def run_case(case: MutationCase, compiler: pathlib.Path, tmpdir: pathlib.Path, timeout: int) -> bool:
    original_path = tmpdir / f"{case.name}.original.ora"
    mutant_path = tmpdir / f"{case.name}.mutant.ora"
    original_artifacts = tmpdir / f"{case.name}.original.artifacts"
    mutant_artifacts = tmpdir / f"{case.name}.mutant.artifacts"
    original_path.write_text(case.source, encoding="utf-8")
    mutant_path.write_text(mutate(case), encoding="utf-8")

    original = run_verifier(compiler, original_path, original_artifacts, timeout)
    if original.returncode != 0:
        print(f"[fail] {case.name}: original contract did not verify", file=sys.stderr)
        print_failure_output(original)
        return False

    mutant = run_verifier(compiler, mutant_path, mutant_artifacts, timeout)
    if mutant.returncode == 0:
        print(f"[fail] {case.name}: mutant still verified ({case.reason})", file=sys.stderr)
        print(f"       file: {mutant_path}", file=sys.stderr)
        return False

    print(f"[pass] {case.name}: mutant rejected")
    return True


def run_cases(cases: Iterable[MutationCase], compiler: pathlib.Path, timeout: int, keep_tmp: bool) -> int:
    if not compiler.exists() or not os.access(compiler, os.X_OK):
        print(f"error: compiler not found or not executable: {compiler}", file=sys.stderr)
        print("hint: run 'zig build' first", file=sys.stderr)
        return 2

    temp_context = tempfile.TemporaryDirectory(prefix="ora-mutation-")
    tmpdir = pathlib.Path(temp_context.name)
    if keep_tmp:
        temp_context.cleanup()
        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="ora-mutation-keep-"))

    try:
        total = 0
        passed = 0
        for case in cases:
            total += 1
            if run_case(case, compiler, tmpdir, timeout):
                passed += 1
        failed = total - passed
        print()
        print("Mutation verification summary")
        print("-----------------------------")
        print(f"Total:  {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        if keep_tmp:
            print(f"Temp:   {tmpdir}")
        return 0 if failed == 0 else 1
    finally:
        if not keep_tmp:
            temp_context.cleanup()


def main() -> int:
    args = parse_args()
    return run_cases(CASES, pathlib.Path(args.compiler), args.timeout, args.keep_tmp)


if __name__ == "__main__":
    raise SystemExit(main())
