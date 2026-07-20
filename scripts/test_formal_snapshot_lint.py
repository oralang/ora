#!/usr/bin/env python3
"""Regression tests for the generated Lean snapshot grammar."""

from __future__ import annotations

import unittest

from formal_snapshot_lint import SnapshotSyntaxError, validate_snapshot


VALID = """\
/- generated data -/
namespace Ora.Generated

def rows : List (String × Option Nat × Bool) :=
  [("first", some 7, true),
   ("second", none, false)]

end Ora.Generated
"""


class SnapshotLintTests(unittest.TestCase):
    def test_accepts_multiline_literal_data(self) -> None:
        validate_snapshot(VALID)

    def assert_rejected(self, command: str) -> None:
        source = VALID.replace("end Ora.Generated", f"{command}\nend Ora.Generated")
        with self.assertRaises(SnapshotSyntaxError):
            validate_snapshot(source)

    def test_rejects_environment_and_syntax_commands(self) -> None:
        for command in (
            "attribute [simp] Ora.someTheorem",
            'notation "x" => 1',
            'local notation "x" => 1',
            "elab \"x\" : command => pure ()",
            "abbrev hidden : Nat := 1",
            "import Ora",
            "open Ora",
            "section Hidden",
            "set_option maxHeartbeats 0",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_rejects_proof_or_computed_definition_bodies(self) -> None:
        for value in ("by decide", "1 + 1", "unsafeCast 1"):
            source = VALID.replace(
                "end Ora.Generated", f"def injected : Nat := {value}\nend Ora.Generated"
            )
            with self.subTest(value=value), self.assertRaises(SnapshotSyntaxError):
                validate_snapshot(source)

    def test_requires_exact_generated_namespace(self) -> None:
        with self.assertRaises(SnapshotSyntaxError):
            validate_snapshot(VALID.replace("Ora.Generated", "Ora.Other"))


if __name__ == "__main__":
    unittest.main()
