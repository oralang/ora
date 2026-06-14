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


BASE_SIGNED = """\
contract MutationSigned {
    pub fn addSigned(a: i256, b: i256) -> i256
        requires(a >= 0)
        requires(b >= 0)
        requires(a <= 1000)
        requires(b <= 1000)
        ensures(result == a + b)
    {
        return a + b;
    }
}
"""


BASE_REFINEMENT = """\
contract MutationRefinement {
    pub fn atLeast(x: MinValue<u256, 100>) -> u256
        ensures(result >= 100)
    {
        return x;
    }
}
"""


BASE_ERROR_UNION = """\
contract MutationErrorUnion {
    error E;

    pub fn get(v: u256) -> !u256 | E
        requires(v <= 1000)
        ensures_ok(result == v)
    {
        return v;
    }
}
"""


BASE_ERROR_UNION_ERR = """\
contract MutationErrorUnionErr {
    error E;

    pub fn fail(v: u256) -> !u256 | E
        requires(v <= 1000)
        ensures_err(v <= 1000)
    {
        return E;
    }
}
"""


BASE_BRANCH_CONDITION = """\
contract MutationBranchCondition {
    pub fn guarded_div(x: u256, y: u256) -> u256
        requires(y != 0)
    {
        if (x / y == 0) {
            return 0;
        }
        return x / y;
    }
}
"""


BASE_IF_CONDITION = """\
contract MutationIfCondition {
    pub fn branch_div(x: u256, y: u256) -> u256
        requires(y != 0)
    {
        var out: u256 = 2;
        if (x / y == 0) {
            out = 1;
        }
        return out;
    }
}
"""


BASE_SWITCH_SCRUTINEE = """\
contract MutationSwitchScrutinee {
    pub fn switch_div(x: u256, y: u256) -> u256
        requires(y != 0)
    {
        switch (x / y) {
            0 => { return 0; }
            else => { return 1; }
        }
    }
}
"""


BASE_MAP_EFFECT = """\
contract MutationMapEffect {
    storage var balances: map<address, u256>;

    pub fn set(who: address, amount: u256)
        requires(amount <= 100)
        ensures(balances[who] == amount)
    {
        balances[who] = amount;
    }
}
"""


BASE_MAP_FRAME = """\
contract MutationMapFrame {
    storage var balances: map<address, u256>;

    pub fn setOther(who: address, other: address, amount: u256)
        requires(who != other)
        requires(amount <= 100)
        ensures(balances[other] == old(balances[other]))
    {
        balances[who] = amount;
    }
}
"""


BASE_LOOP_INVARIANT = """\
contract MutationLoopInvariant {
    storage var counter: u256 = 0;

    pub fn countTo(n: u256)
        requires(n <= 1000)
        requires(counter == 0)
        ensures(counter == n)
    {
        while (counter < n)
            invariant(counter <= n)
        {
            counter = counter + 1;
        }
    }
}
"""


BASE_STATE_INVARIANT = """\
contract MutationStateInvariant {
    storage var x: u256 = 0;

    invariant bounded(x <= 100);

    pub fn set(v: u256)
        requires(v <= 100)
    {
        x = v;
    }
}
"""


BASE_ASSERTION = """\
contract MutationAssert {
    pub fn assertBound(x: u256)
        requires(x <= 100)
    {
        assert(x <= 100, "bound");
    }
}
"""


BASE_STORAGE_BRANCH = """\
contract MutationStorageBranch {
    storage var limit: u256;

    pub fn set_limit(value: u256)
        requires(value <= 20)
    {
        limit = value;
    }

    pub fn classify(x: u256) -> u256
        requires(x <= 20)
        ensures((x <= limit && result == 1) || (x > limit && result == 2))
    {
        if (x <= limit) {
            return 1;
        }
        return 2;
    }
}
"""


BASE_QUANTIFIED = """\
contract MutationQuantified {
    pub fn check(x: u256) -> bool
        ensures(forall i: u256 where i < x => i <= x)
    {
        return true;
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
    # Per-construct mutants broaden coverage beyond the counter/vault shapes.
    MutationCase(
        name="signed_postcondition_flip",
        source=BASE_SIGNED,
        function="addSigned",
        find="ensures(result == a + b)",
        replace="ensures(result == a - b)",
        reason="signed-arithmetic postcondition no longer matches the body",
    ),
    MutationCase(
        name="signed_body_corruption",
        source=BASE_SIGNED,
        function="addSigned",
        find="return a + b;",
        replace="return a - b;",
        reason="signed body changed so it no longer proves the postcondition",
    ),
    MutationCase(
        name="refinement_bound_overclaim",
        source=BASE_REFINEMENT,
        function="atLeast",
        find="ensures(result >= 100)",
        replace="ensures(result >= 200)",
        reason="postcondition overclaims a tighter bound than the refinement gives",
    ),
    MutationCase(
        name="error_union_ok_postcondition_flip",
        source=BASE_ERROR_UNION,
        function="get",
        find="ensures_ok(result == v)",
        replace="ensures_ok(result == v + 1)",
        reason="error-union Ok postcondition no longer matches the returned value",
    ),
    MutationCase(
        name="error_union_ok_body_corruption",
        source=BASE_ERROR_UNION,
        function="get",
        find="return v;",
        replace="return v + 1;",
        reason="error-union Ok body changed so it no longer proves the postcondition",
    ),
    MutationCase(
        name="error_union_err_postcondition_flip",
        source=BASE_ERROR_UNION_ERR,
        function="fail",
        find="ensures_err(v <= 1000)",
        replace="ensures_err(v > 1000)",
        reason="error-union Err postcondition no longer follows from the precondition",
    ),
    MutationCase(
        name="branch_condition_div_precondition_removed",
        source=BASE_BRANCH_CONDITION,
        function="guarded_div",
        find="requires(y != 0)",
        replace="requires(true)",
        reason="division inside a branch condition must still prove its non-zero divisor",
    ),
    MutationCase(
        name="if_condition_div_precondition_removed",
        source=BASE_IF_CONDITION,
        function="branch_div",
        find="requires(y != 0)",
        replace="requires(true)",
        reason="division inside an if condition must prove its non-zero divisor before branch facts apply",
    ),
    MutationCase(
        name="switch_scrutinee_div_precondition_removed",
        source=BASE_SWITCH_SCRUTINEE,
        function="switch_div",
        find="requires(y != 0)",
        replace="requires(true)",
        reason="division inside a switch scrutinee must prove its non-zero divisor before case facts apply",
    ),
    MutationCase(
        name="map_store_postcondition_body_corruption",
        source=BASE_MAP_EFFECT,
        function="set",
        find="balances[who] = amount;",
        replace="balances[who] = amount + 1;",
        reason="map storage postcondition no longer matches the stored value",
    ),
    MutationCase(
        name="map_frame_alias_precondition_removed",
        source=BASE_MAP_FRAME,
        function="setOther",
        find="requires(who != other)",
        replace="requires(true)",
        reason="map frame postcondition no longer follows when keys may alias",
    ),
    MutationCase(
        name="loop_invariant_postcondition_overclaim",
        source=BASE_LOOP_INVARIANT,
        function="countTo",
        find="ensures(counter == n)",
        replace="ensures(counter == n + 1)",
        reason="loop-invariant-derived postcondition overclaims the final counter value",
    ),
    MutationCase(
        name="loop_invariant_body_corruption",
        source=BASE_LOOP_INVARIANT,
        function="countTo",
        find="counter = counter + 1;",
        replace="counter = counter + 2;",
        reason="loop body no longer preserves the invariant-derived postcondition",
    ),
    MutationCase(
        name="state_invariant_precondition_removed",
        source=BASE_STATE_INVARIANT,
        function="set",
        find="requires(v <= 100)",
        replace="requires(true)",
        reason="weakened precondition permits a storage write that violates the state invariant",
    ),
    MutationCase(
        name="assert_requires_weakened",
        source=BASE_ASSERTION,
        function="assertBound",
        find="requires(x <= 100)",
        replace="requires(true)",
        reason="assert obligation no longer follows from the weakened precondition",
    ),
    MutationCase(
        name="storage_branch_body_corruption",
        source=BASE_STORAGE_BRANCH,
        function="classify",
        find="return 1;",
        replace="return 2;",
        reason="storage-backed branch result no longer proves the conditional postcondition",
    ),
    MutationCase(
        name="quantified_postcondition_flip",
        source=BASE_QUANTIFIED,
        function="check",
        find="ensures(forall i: u256 where i < x => i <= x)",
        replace="ensures(forall i: u256 where i < x => i > x)",
        reason="quantified postcondition no longer follows for values below x",
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
