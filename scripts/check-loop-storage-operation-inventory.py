#!/usr/bin/env python3
"""Gate the loop verifier's storage-operation inventory against TableGen.

The loop projector fails closed on unknown MLIR memory effects, but that runtime
rule is not enough to make dialect growth reviewable.  This gate expands the
Ora operation records with mlir-tblgen and pins every operation with a direct,
dynamic, or recursive memory-effect contract.  It also pins the pure/unknown
operations that carry storage provenance or require a named rejection.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TBLGEN = ROOT / "vendor/mlir/bin/mlir-tblgen"
OPS_TD = ROOT / "src/mlir/ora/td/OraOps.td"
ORA_TD_INCLUDE = ROOT / "src/mlir/ora/td"
MLIR_INCLUDE = ROOT / "vendor/mlir/include"


# operation -> (coarse loop-storage class, expanded memory-effect shape)
EXPECTED_EFFECTFUL: Mapping[str, Tuple[str, str]] = {
    "ora.assert": ("formal_marker", "write"),
    "ora.assume": ("formal_marker", "write"),
    "ora.bitfield_decl": ("declaration_only", "write"),
    "ora.create": ("resource", "read_write"),
    "ora.decreases": ("formal_marker", "write"),
    "ora.destroy": ("resource", "read_write"),
    "ora.ensures": ("formal_marker", "write"),
    "ora.enum.decl": ("declaration_only", "write"),
    "ora.error.decl": ("declaration_only", "write"),
    "ora.evm.gas": ("external_unknown", "read_write"),
    "ora.external_call": ("external_unknown", "read_write"),
    "ora.global": ("declaration_only", "write"),
    "ora.havoc": ("formal_marker", "write"),
    "ora.increases": ("formal_marker", "write"),
    "ora.invariant": ("formal_marker", "write"),
    "ora.log.decl": ("declaration_only", "write"),
    "ora.log": ("external_unknown", "write"),
    "ora.loop_control": ("formal_marker", "write"),
    "ora.mload8": ("memory_only", "dynamic"),
    "ora.mload": ("memory_only", "dynamic"),
    "ora.mstore8": ("memory_only", "dynamic"),
    "ora.mstore": ("memory_only", "dynamic"),
    "ora.map_get": ("persistent", "dynamic"),
    "ora.map_store": ("persistent", "dynamic"),
    "ora.memory.global": ("declaration_only", "write"),
    "ora.move": ("resource", "read_write"),
    "ora.refinement_guard": ("formal_marker", "write"),
    "ora.requires": ("formal_marker", "write"),
    "ora.sload": ("persistent", "dynamic"),
    "ora.sstore": ("persistent", "dynamic"),
    "ora.storage.range_erase": ("persistent", "write"),
    "ora.storage.word_load": ("persistent", "read"),
    "ora.storage.word_store": ("persistent", "write"),
    "ora.struct.decl": ("declaration_only", "write"),
    "ora.switch_expr": ("external_unknown", "recursive"),
    "ora.switch": ("external_unknown", "recursive"),
    "ora.tload": ("transient", "dynamic"),
    "ora.tstore.global": ("declaration_only", "write"),
    "ora.tstore": ("transient", "dynamic"),
    "ora.try_catch": ("external_unknown", "recursive"),
    "ora.try_stmt": ("external_unknown", "recursive"),
}


# operation -> exact projector disposition.  These operations have no non-empty
# direct effect row: the first three are pure provenance, while the last two
# deliberately remain unsupported and therefore classify as UNKNOWN at runtime.
EXPECTED_SPECIAL: Mapping[str, str] = {
    "ora.storage.derive": "persistent_provenance",
    "ora.struct_field_extract": "persistent_provenance",
    "ora.struct_field_update": "persistent_provenance",
    "ora.struct_field_store": "unsupported_aggregate_storage_access",
    "ora.tstore.guard": "unsupported_transient_storage",
}


RECORD_START = re.compile(r"^def ([A-Za-z0-9_]+) \{", re.MULTILINE)


def fail(message: str) -> None:
    print(f"check-loop-storage-operation-inventory: {message}", file=sys.stderr)
    raise SystemExit(1)


def split_records(text: str) -> Dict[str, str]:
    matches = list(RECORD_START.finditer(text))
    records: Dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        records[match.group(1)] = text[match.end() : end]
    return records


def ora_operations(records: Mapping[str, str]) -> Dict[str, str]:
    operations: Dict[str, str] = {}
    for record_name, body in records.items():
        if not re.search(r"^  Dialect opDialect = Ora_Dialect;$", body, re.MULTILINE):
            continue
        name_match = re.search(r'^  string opName = "([^"]+)";$', body, re.MULTILINE)
        if name_match is None:
            raise ValueError(f"{record_name}: Ora operation is missing opName")
        full_name = f"ora.{name_match.group(1)}"
        if full_name in operations:
            raise ValueError(f"duplicate Ora operation name: {full_name}")
        operations[full_name] = record_name
    return operations


def trait_names(body: str) -> Sequence[str]:
    match = re.search(r"^  list<Trait> traits = \[([^]]*)\];$", body, re.MULTILINE)
    if match is None or not match.group(1).strip():
        return ()
    return tuple(part.strip() for part in match.group(1).split(",") if part.strip())


def static_effects(body: str) -> Optional[Tuple[str, ...]]:
    if 'baseEffectName = "::mlir::MemoryEffects::Effect"' not in body:
        return None
    match = re.search(r"^  list<SideEffect> effects = \[([^]]*)\];$", body, re.MULTILINE)
    if match is None:
        raise ValueError("MemoryEffects trait is missing its effects list")
    return tuple(part.strip() for part in match.group(1).split(",") if part.strip())


def effect_shape(records: Mapping[str, str], operation_record: str) -> Optional[str]:
    static: list[str] = []
    dynamic = False
    recursive = False
    for trait_name in trait_names(records[operation_record]):
        body = records.get(trait_name, "")
        effects = static_effects(body)
        if effects is not None:
            static.extend(effects)
        if 'cppInterfaceName = "MemoryEffectOpInterface"' in body:
            dynamic = True
        if "HasRecursiveMemoryEffects" in body:
            recursive = True

    modes = int(bool(static)) + int(dynamic) + int(recursive)
    if modes > 1:
        raise ValueError(f"{operation_record}: ambiguous memory-effect traits")
    if recursive:
        return "recursive"
    if dynamic:
        return "dynamic"
    if not static:
        return None

    names = set(static)
    if names == {"MemRead"}:
        return "read"
    if names == {"MemWrite"}:
        return "write"
    if names == {"MemRead", "MemWrite"}:
        return "read_write"
    return "static:" + ",".join(sorted(names))


def actual_effectful(records: Mapping[str, str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for operation, record_name in ora_operations(records).items():
        shape = effect_shape(records, record_name)
        if shape is not None:
            result[operation] = shape
    return result


def compare_inventory(
    records: Mapping[str, str],
    expected_effectful: Mapping[str, Tuple[str, str]] = EXPECTED_EFFECTFUL,
    expected_special: Mapping[str, str] = EXPECTED_SPECIAL,
) -> Sequence[str]:
    errors: list[str] = []
    operations = ora_operations(records)
    actual = actual_effectful(records)
    expected_shapes = {name: row[1] for name, row in expected_effectful.items()}

    for name in sorted(actual.keys() - expected_shapes.keys()):
        errors.append(f"unclassified effectful operation: {name} ({actual[name]})")
    for name in sorted(expected_shapes.keys() - actual.keys()):
        errors.append(f"classified effectful operation disappeared or lost its effect: {name}")
    for name in sorted(actual.keys() & expected_shapes.keys()):
        if actual[name] != expected_shapes[name]:
            errors.append(
                f"effect drift for {name}: expected {expected_shapes[name]}, got {actual[name]}"
            )

    for name, disposition in sorted(expected_special.items()):
        record_name = operations.get(name)
        if record_name is None:
            errors.append(f"special projector operation disappeared: {name} ({disposition})")
            continue
        shape = effect_shape(records, record_name)
        if shape is not None:
            errors.append(
                f"special projector operation changed effect shape: {name} "
                f"({disposition}) is now {shape}"
            )
    return errors


def run_tblgen(tblgen: Path) -> str:
    command = [
        str(tblgen),
        str(OPS_TD),
        "--print-records",
        "-I",
        str(ORA_TD_INCLUDE),
        "-I",
        str(MLIR_INCLUDE),
    ]
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.PIPE)
    except FileNotFoundError:
        fail(f"missing mlir-tblgen: {tblgen}")
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or f"exit code {error.returncode}"
        fail(f"mlir-tblgen failed: {detail}")
    raise AssertionError("unreachable")


def run_self_test() -> None:
    sample = """
def ReadTrait {
  string baseEffectName = "::mlir::MemoryEffects::Effect";
  list<SideEffect> effects = [MemRead];
}
def DynamicTrait {
  string cppInterfaceName = "MemoryEffectOpInterface";
}
def RecursiveTrait {
  string trait = "::mlir::OpTrait::HasRecursiveMemoryEffects";
}
def ReadOp {
  Dialect opDialect = Ora_Dialect;
  string opName = "read";
  list<Trait> traits = [ReadTrait];
}
def DynamicOp {
  Dialect opDialect = Ora_Dialect;
  string opName = "dynamic";
  list<Trait> traits = [DynamicTrait];
}
def RecursiveOp {
  Dialect opDialect = Ora_Dialect;
  string opName = "recursive";
  list<Trait> traits = [RecursiveTrait];
}
"""
    records = split_records(sample)
    assert ora_operations(records) == {
        "ora.read": "ReadOp",
        "ora.dynamic": "DynamicOp",
        "ora.recursive": "RecursiveOp",
    }
    assert actual_effectful(records) == {
        "ora.read": "read",
        "ora.dynamic": "dynamic",
        "ora.recursive": "recursive",
    }
    expected = {
        "ora.read": ("persistent", "read"),
        "ora.dynamic": ("persistent", "dynamic"),
        "ora.recursive": ("external_unknown", "recursive"),
    }
    assert compare_inventory(records, expected, {}) == []
    missing = dict(expected)
    del missing["ora.dynamic"]
    assert compare_inventory(records, missing, {}) == [
        "unclassified effectful operation: ora.dynamic (dynamic)"
    ]
    drifted = dict(expected)
    drifted["ora.read"] = ("persistent", "write")
    assert compare_inventory(records, drifted, {}) == [
        "effect drift for ora.read: expected write, got read"
    ]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tblgen", type=Path, default=DEFAULT_TBLGEN)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        print("check-loop-storage-operation-inventory: self-test ok")
        return 0

    records = split_records(run_tblgen(args.tblgen))
    errors = compare_inventory(records)
    if errors:
        for error in errors:
            print(f"check-loop-storage-operation-inventory: {error}", file=sys.stderr)
        return 1

    print(
        "check-loop-storage-operation-inventory: ok "
        f"({len(EXPECTED_EFFECTFUL)} effectful, {len(EXPECTED_SPECIAL)} special)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
