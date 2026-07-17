#!/usr/bin/env python3
"""Run a conformance spec against Anvil and compare execution results.

The spec expectations are also checked by the in-process lib/evm runner.
Agreement corroborates lib/evm as an oracle; a divergence points to a compiler
bug, a lib/evm bug, or a hardfork mismatch.

LOCAL-FIRST, not in the blocking gate (Anvil is a live process). Pins the
hardfork to lib/evm's Ora target (OSAKA). Unsupported shapes are reported
loudly by the per-spec runner; the corpus wrapper fails on any divergence or
unhandled spec.

Usage:
  scripts/conformance-anvil-diff.py tests/conformance/counter.spec.toml
  scripts/conformance-anvil-diff.py --verified tests/conformance/resource_basic_transfer.spec.toml
  scripts/conformance-anvil-diff.py --verified --keep-proved-checks tests/conformance/resource_basic_transfer.spec.toml
"""

from __future__ import annotations

import functools
import re
import os
import subprocess
import sys
import time
from pathlib import Path
import json
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ORA = ROOT / "zig-out" / "bin" / "ora"
RPC = os.environ.get("ANVIL_RPC_URL", "http://127.0.0.1:8545")
RPC_TIMEOUT_S = int(os.environ.get("ANVIL_RPC_TIMEOUT", "30"))
RECEIPT_ATTEMPTS = int(os.environ.get("ANVIL_RECEIPT_ATTEMPTS", "100"))
RECEIPT_POLL_S = float(os.environ.get("ANVIL_RECEIPT_POLL_MS", "50")) / 1000.0
PROGRESS = os.environ.get("ANVIL_DIFF_PROGRESS", "1") != "0"
# Anvil's deterministic dev account 0 — same on every launch.
DEV_ADDRESS = "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"


def sh(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def progress(message: str) -> None:
    if PROGRESS:
        print(f"diff: {message}", file=sys.stderr, flush=True)


def cast(*args: str, rpc: bool = True) -> str:
    # Insert --rpc-url right after the subcommand; trailing flags get swallowed
    # by greedy positional args like `send --create <CODE>`.
    cmd = ["cast", args[0]]
    if rpc:
        cmd += ["--rpc-url", RPC, "--no-proxy"]
    cmd += list(args[1:])
    p = sh(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"cast {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


_funded: set[str] = set()


def rpc_request(method: str, params: list[object]) -> object:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(RPC, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=RPC_TIMEOUT_S) as response:
        payload = json.loads(response.read().decode())
    if "error" in payload:
        raise RuntimeError(f"rpc {method} failed: {payload['error']}")
    return payload.get("result")


def rpc_try(method: str, params: list[object]) -> tuple[bool, object]:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(RPC, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=RPC_TIMEOUT_S) as response:
        payload = json.loads(response.read().decode())
    if "error" in payload:
        return False, payload["error"]
    return True, payload.get("result")


def hex_quantity(value: int) -> str:
    return hex(value)


def tx_object(from_addr: str, to_addr: str | None, data: str, value: int = 0) -> dict[str, str]:
    tx = {
        "from": from_addr,
        "data": data,
        "gas": "0x989680",  # 10M, enough for adversarial conformance calls.
    }
    if to_addr is not None:
        tx["to"] = to_addr
    if value:
        tx["value"] = hex_quantity(value)
    return tx


def rpc_call(from_addr: str, to_addr: str, data: str, value: int = 0) -> str:
    return rpc_request("eth_call", [tx_object(from_addr, to_addr, data, value), "latest"])


def rpc_call_try(from_addr: str, to_addr: str, data: str, value: int = 0) -> tuple[bool, object]:
    return rpc_try("eth_call", [tx_object(from_addr, to_addr, data, value), "latest"])


def wait_receipt(tx_hash: str, stage: str) -> dict:
    progress(f"{stage}: waiting receipt {tx_hash}")
    for attempt in range(RECEIPT_ATTEMPTS):
        receipt = rpc_request("eth_getTransactionReceipt", [tx_hash])
        if receipt is not None:
            progress(f"{stage}: receipt mined status={receipt.get('status')}")
            return receipt
        time.sleep(RECEIPT_POLL_S)
    raise RuntimeError(
        f"{stage}: timed out waiting for receipt {tx_hash} "
        f"after {RECEIPT_ATTEMPTS} attempts"
    )


def send_transaction(from_addr: str, to_addr: str | None, data: str, value: int = 0, *, stage: str) -> tuple[bool, dict | object]:
    progress(f"{stage}: send transaction")
    ok, result = rpc_try("eth_sendTransaction", [tx_object(from_addr, to_addr, data, value)])
    if not ok:
        progress(f"{stage}: send rejected {result}")
        return False, result
    receipt = wait_receipt(str(result), stage)
    return receipt.get("status") in ("0x1", "1", 1, True), receipt


def rpc_code(addr: str) -> str:
    return rpc_request("eth_getCode", [addr, "latest"])


def rpc_storage(addr: str, slot: int) -> int:
    return int(rpc_request("eth_getStorageAt", [addr, hex_quantity(slot), "latest"]), 16)


def fund(addr: str) -> None:
    """Give an impersonated caller enough balance for gas + value."""
    a = addr.lower()
    if a in _funded:
        return
    rpc_request("anvil_setBalance", [addr, "0xffffffffffffffffffffffff"])
    _funded.add(a)


# --- tiny spec reader (the [deploy]/[[call]]/[[call.storage]]/[[call.log]] subset) ---------

def read_spec(path: Path) -> dict:
    deploy = {"args": []}
    contracts: list[dict] = []
    calls: list[dict] = []
    cur = None
    section = None
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "[deploy]":
            section = "deploy"
            continue
        if line == "[[contract]]":
            cur = {"args": []}
            contracts.append(cur)
            section = "contract"
            continue
        if line == "[[call]]":
            cur = {"storage": [], "logs": []}
            calls.append(cur)
            section = "call"
            continue
        if line == "[[call.storage]]":
            if cur is None:
                raise RuntimeError("[[call.storage]] before [[call]]")
            section = "storage"
            cur["storage"].append({})
            continue
        if line == "[[call.log]]":
            if cur is None:
                raise RuntimeError("[[call.log]] before [[call]]")
            section = "log"
            cur["logs"].append({})
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if section == "deploy":
            target = deploy
        elif section == "storage":
            target = cur["storage"][-1]
        elif section == "log":
            target = cur["logs"][-1]
        else:
            target = cur
        target[key] = val
    return {"deploy": deploy, "contracts": contracts, "calls": calls}


def parse_int(tok: str) -> int:
    tok = tok.strip().strip('"')
    return int(tok, 16) if tok.startswith("0x") else int(tok)


U256_MOD = 1 << 256


@functools.lru_cache(maxsize=None)
def keccak_hex(hex_data: str) -> int:
    return int(cast("keccak", "0x" + hex_data, rpc=False), 16)


def keccak_bytes(data: bytes) -> int:
    return keccak_hex(data.hex())


def encode_static_abi_word(wire_type: str, value_text: str) -> bytes:
    wire_type = normalize_int_type(wire_type.strip())
    value_text = value_text.strip().strip('"')
    if wire_type == "bool":
        value = 1 if value_text.lower() in ("true", "1") else 0
        return value.to_bytes(32, "big")
    if wire_type == "address":
        value = parse_int(value_text)
        if value < 0 or value >= (1 << 160):
            raise Unsupported(f"address key out of range: {value_text}")
        return value.to_bytes(32, "big")
    if re.fullmatch(r"uint\d+", wire_type):
        value = parse_int(value_text)
        if value < 0 or value >= U256_MOD:
            raise Unsupported(f"uint key out of range: {value_text}")
        return value.to_bytes(32, "big")
    if re.fullmatch(r"int\d+", wire_type):
        value = parse_int(value_text)
        if value < 0:
            value += U256_MOD
        if value < 0 or value >= U256_MOD:
            raise Unsupported(f"int key out of range: {value_text}")
        return value.to_bytes(32, "big")
    m = re.fullmatch(r"bytes([1-9]|[12]\d|3[12])", wire_type)
    if m:
        raw = bytes_from_hex(value_text)
        width = int(m.group(1))
        if len(raw) != width:
            raise Unsupported(f"{wire_type} key has {len(raw)} bytes")
        return raw + bytes(32 - width)
    raise Unsupported(f"unsupported map key type {wire_type}")


def parse_slot_value(value: str) -> int:
    token = value.strip().strip('"')
    return parse_slot_expression(token)


def parse_slot_expression(text: str) -> int:
    text = text.strip()
    if not text:
        raise Unsupported("empty storage slot expression")
    inner = call_inner(text, "computed")
    if inner is not None:
        parts = split_top_level(inner)
        return computed_storage_slot(parts)
    inner = call_inner(text, "map")
    if inner is not None:
        parts = split_top_level(inner)
        if len(parts) != 3:
            raise Unsupported(f"invalid map slot expression: {text}")
        return mapping_slot(parts[0], parts[1], parse_slot_expression(parts[2]))
    inner = call_inner(text, "add")
    if inner is not None:
        parts = split_top_level(inner)
        if len(parts) != 2:
            raise Unsupported(f"invalid add slot expression: {text}")
        return (parse_slot_expression(parts[0]) + parse_int(parts[1])) % U256_MOD
    inner = call_inner(text, "keccak")
    if inner is not None:
        parts = split_top_level(inner)
        if len(parts) != 1:
            raise Unsupported(f"invalid keccak slot expression: {text}")
        return keccak_slot(parse_slot_expression(parts[0]))
    return parse_int(text)


def call_inner(text: str, name: str) -> str | None:
    prefix = name + "("
    if text.startswith(prefix) and text.endswith(")"):
        return text[len(prefix) : -1]
    return None


def mapping_slot(wire_type: str, key_text: str, root_slot: int) -> int:
    wire_type = wire_type.strip()
    if wire_type == "string":
        key_word = keccak_bytes(key_text.strip().strip('"').encode()).to_bytes(32, "big")
    elif wire_type == "bytes":
        key_word = keccak_bytes(bytes_from_hex(key_text)).to_bytes(32, "big")
    else:
        key_word = encode_static_abi_word(wire_type, key_text)
    return keccak_bytes(key_word + (root_slot % U256_MOD).to_bytes(32, "big"))


def computed_storage_slot(parts: list[str]) -> int:
    if not parts or (len(parts) - 1) % 2 != 0:
        raise Unsupported(f"invalid computed storage slot expression: computed({','.join(parts)})")

    namespace = parts[0].strip().strip('"')
    if not namespace:
        raise Unsupported("computed storage namespace must be non-empty")

    key_count = (len(parts) - 1) // 2
    domain_prefix = 0x4F72614353545631  # "OraCSTV1"
    preimage = bytearray()
    preimage += domain_prefix.to_bytes(32, "big")
    preimage += key_count.to_bytes(32, "big")
    preimage += keccak_bytes(namespace.encode()).to_bytes(32, "big")

    for index in range(key_count):
        wire_type = parts[1 + index * 2]
        value = parts[2 + index * 2]
        preimage += encode_static_abi_word(wire_type, value)

    return keccak_bytes(bytes(preimage))


def keccak_slot(slot: int) -> int:
    return keccak_bytes((slot % U256_MOD).to_bytes(32, "big"))


def parse_quantity(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 16) if value.startswith("0x") else int(value)
    raise Unsupported(f"unsupported quantity {value!r}")


def gas_ceiling(call: dict) -> int | None:
    if "gas_max" not in call:
        return None
    return parse_int(call["gas_max"])


def check_receipt_gas(label: str, call: dict, receipt: object) -> bool:
    max_gas = gas_ceiling(call)
    if max_gas is None:
        return True
    if not isinstance(receipt, dict) or "gasUsed" not in receipt:
        raise RuntimeError(f"{label}: no receipt gas available for gas_max")
    used = parse_quantity(receipt["gasUsed"])
    ok = used <= max_gas
    mark = "OK" if ok else "DIVERGE"
    print(f"    [{mark}] gas -> anvil={used} max={max_gas}")
    return ok


def parse_bool(tok: str) -> bool:
    return tok.strip().strip('"').lower() in ("true", "1", "yes")


def parse_hex_data(tok: str) -> bytes:
    return bytes_from_hex(tok.strip().strip('"'))


def check_receipt_logs(label: str, call: dict, receipt: object, *, ignore_logs: bool) -> tuple[int, int]:
    expected_logs = call.get("logs", [])
    if not expected_logs or ignore_logs:
        return 0, 0
    if not isinstance(receipt, dict):
        print(f"    [DIVERGE] logs -> no transaction receipt available")
        return 1, 1
    actual_logs = receipt.get("logs")
    if not isinstance(actual_logs, list):
        print(f"    [DIVERGE] logs -> receipt has no logs array")
        return 1, 1

    checked = 0
    diverged = 0
    if len(actual_logs) != len(expected_logs):
        print(f"    [DIVERGE] logs -> anvil={len(actual_logs)} expected={len(expected_logs)}")
        return 1, 1

    for i, expected in enumerate(expected_logs):
        actual = actual_logs[i]
        if not isinstance(actual, dict):
            print(f"    [DIVERGE] log {i} -> malformed receipt log")
            checked += 1
            diverged += 1
            continue
        want_topics = [parse_int(topic) for topic in parse_args(expected.get("topics", "[]"))]
        got_topics = [parse_int(str(topic)) for topic in actual.get("topics", [])]
        want_data = parse_hex_data(expected.get("data", '"0x"'))
        got_data = parse_hex_data(str(actual.get("data", "0x")))
        ok = got_topics == want_topics and got_data == want_data
        mark = "OK" if ok else "DIVERGE"
        print(
            f"    [{mark}] log {i} -> "
            f"topics={len(got_topics)} expected_topics={len(want_topics)} "
            f"data={len(got_data)} bytes expected_data={len(want_data)} bytes"
        )
        checked += 1
        if not ok:
            diverged += 1
    return checked, diverged


def parse_args(tok: str) -> list[str]:
    """Top-level comma split that respects nested [...] arrays and (...) tuples,
    so `[[5, 8, 13]]` yields one array arg, not three."""
    tok = tok.strip()
    if tok in ("[]", ""):
        return []
    inner = tok[1:-1]
    out: list[str] = []
    depth = 0
    cur = ""
    for ch in inner:
        if ch in "[(":
            depth += 1
            cur += ch
        elif ch in "])":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            out.append(cur.strip().strip('"'))
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip().strip('"'))
    # cast wants array literals comma-packed without spaces and hex scalar
    # elements unquoted: [0xabc,0xdef], not ["0xabc","0xdef"].
    return [normalize_cast_array_arg(re.sub(r"\s+", "", a)) if a.startswith("[") else a for a in out]


def normalize_cast_array_arg(arg: str) -> str:
    return re.sub(r'"(0x[0-9a-fA-F]*)"', r"\1", arg)


RET_SPEC = re.compile(r"\{\s*(.+?)\s*=\s*(.+?)\s*\}\s*$")


def strip_0x(value: str) -> str:
    return value[2:] if value.startswith(("0x", "0X")) else value


def bytes_from_hex(value: str) -> bytes:
    text = strip_0x(value.strip())
    if len(text) % 2:
        text = "0" + text
    return bytes.fromhex(text)


def split_top_level(text: str) -> list[str]:
    out: list[str] = []
    depth = 0
    cur = ""
    for ch in text:
        if ch in "[(":
            depth += 1
            cur += ch
        elif ch in "])":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            out.append(cur.strip().strip('"'))
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip().strip('"'))
    return out


def tuple_inner(wire_type: str) -> str | None:
    wire_type = wire_type.strip()
    if len(wire_type) >= 2 and wire_type[0] == "(" and wire_type[-1] == ")":
        return wire_type[1:-1]
    return None


def tuple_fields(wire_type: str) -> list[str]:
    inner = tuple_inner(wire_type)
    if inner is None:
        raise Unsupported(f"not a tuple type: {wire_type}")
    return split_top_level(inner)


def expected_list(value: str, open_ch: str, close_ch: str) -> list[str]:
    value = value.strip()
    if len(value) < 2 or value[0] != open_ch or value[-1] != close_ch:
        raise Unsupported(f"expected {open_ch}...{close_ch}: {value}")
    return split_top_level(value[1:-1])


def abi_word(data: bytes, offset: int) -> int | None:
    if offset < 0 or offset + 32 > len(data):
        return None
    return int.from_bytes(data[offset : offset + 32], "big")


def normalize_int_type(wire_type: str) -> str:
    if re.fullmatch(r"u\d+", wire_type):
        return "uint" + wire_type[1:]
    if re.fullmatch(r"i\d+", wire_type):
        return "int" + wire_type[1:]
    return wire_type


def is_dynamic_type(wire_type: str) -> bool:
    wire_type = wire_type.strip()
    if wire_type in ("string", "bytes"):
        return True
    if wire_type.endswith("[]"):
        return True
    inner = tuple_inner(wire_type)
    if inner is not None:
        return any(is_dynamic_type(field) for field in tuple_fields(wire_type))
    return False


def compare_return_spec(ret_spec: str, raw_hex: str) -> tuple[bool, str]:
    m = RET_SPEC.fullmatch(ret_spec.strip())
    if not m:
        raise Unsupported(f"unsupported return {ret_spec}")
    wire_type = m.group(1).strip()
    expected = m.group(2).strip()
    data = bytes_from_hex(raw_hex)
    ok, observed = compare_abi_value(wire_type, expected, data, 0, root=True)
    return ok, observed


def compare_abi_value(wire_type: str, expected: str, data: bytes, offset: int, *, root: bool = False, base: int | None = None) -> tuple[bool, str]:
    wire_type = wire_type.strip()
    if root and is_dynamic_type(wire_type):
        head = abi_word(data, 0)
        if head is None:
            return False, f"short({len(data)} bytes)"
        return compare_abi_value(wire_type, expected, data, int(head), base=int(head))

    inner = tuple_inner(wire_type)
    if inner is not None:
        return compare_tuple(wire_type, expected, data, offset)

    if wire_type.endswith("[]"):
        element = wire_type[:-2]
        if element != "uint256":
            raise Unsupported(f"unsupported dynamic return array {wire_type}")
        length = abi_word(data, offset)
        if length is None:
            return False, f"short({len(data)} bytes)"
        expected_items = expected_list(expected, "[", "]")
        if int(length) != len(expected_items):
            return False, f"{wire_type}[len={length}]"
        observed: list[str] = []
        for i, item in enumerate(expected_items):
            ok, got = compare_abi_value(element, item, data, offset + 32 + i * 32)
            observed.append(got)
            if not ok:
                return False, f"{wire_type}[{','.join(observed)}]"
        return True, f"{wire_type}[{','.join(observed)}]"

    if wire_type == "string":
        length = abi_word(data, offset)
        if length is None:
            return False, f"short({len(data)} bytes)"
        start = offset + 32
        end = start + int(length)
        if end > len(data):
            return False, f"string(len={length}, short={len(data)} bytes)"
        observed_bytes = data[start:end]
        want = expected.strip().strip('"').encode()
        try:
            observed = observed_bytes.decode()
        except UnicodeDecodeError:
            observed = "0x" + observed_bytes.hex()
        return observed_bytes == want, f"string({observed})"

    return compare_static_word(wire_type, expected, data, offset)


def compare_tuple(wire_type: str, expected: str, data: bytes, offset: int) -> tuple[bool, str]:
    fields = tuple_fields(wire_type)
    expected_items = expected_list(expected, "(", ")")
    if len(fields) != len(expected_items):
        raise Unsupported(f"tuple arity mismatch in return expectation {expected}")

    observed: list[str] = []
    for i, (field, item) in enumerate(zip(fields, expected_items)):
        head = offset + i * 32
        if is_dynamic_type(field):
            rel = abi_word(data, head)
            if rel is None:
                return False, f"tuple(short={len(data)} bytes)"
            ok, got = compare_abi_value(field, item, data, offset + int(rel), base=offset)
        else:
            ok, got = compare_abi_value(field, item, data, head)
        observed.append(got)
        if not ok:
            return False, "(" + ",".join(observed) + ")"
    return True, "(" + ",".join(observed) + ")"


def compare_static_word(wire_type: str, expected: str, data: bytes, offset: int) -> tuple[bool, str]:
    word = abi_word(data, offset)
    if word is None:
        return False, f"short({len(data)} bytes)"
    et = expected.strip().strip('"')
    normalized = normalize_int_type(wire_type)
    if re.fullmatch(r"uint\d+", normalized):
        want = parse_int(et)
        return word == want, str(word)
    if re.fullmatch(r"int\d+", normalized):
        want_signed = parse_int(et)
        want_word = want_signed if want_signed >= 0 else (1 << 256) + want_signed
        observed = word if word < (1 << 255) else word - (1 << 256)
        return word == want_word, str(observed)
    if wire_type == "bool":
        want = et.lower() in ("true", "1")
        observed = word != 0
        return (word in (0, 1) and observed == want), str(observed).lower()
    if wire_type == "address":
        observed = word & ((1 << 160) - 1)
        return observed == parse_int(et), f"0x{observed:040x}"
    raise Unsupported(f"unsupported return type {wire_type}")


def parse_expected_revert(text: str) -> tuple[str, bytes]:
    value = text.strip()
    if not (value.startswith("{") and value.endswith("}")):
        raise Unsupported(f"invalid revert expectation {text}")
    body = value[1:-1].strip()
    if not body:
        return "data", b""
    if "," in body:
        raise Unsupported(f"revert expectation must contain one field: {text}")
    key, separator, raw_value = body.partition("=")
    if not separator:
        raise Unsupported(f"invalid revert expectation {text}")
    key = key.strip()
    raw_value = raw_value.strip().strip('"')
    if key == "any":
        if raw_value.lower() != "true":
            raise Unsupported("reverts.any must be true")
        return "any", b""
    if key == "selector":
        selector = bytes_from_hex(raw_value)
        if len(selector) != 4:
            raise Unsupported("revert selector must be exactly 4 bytes")
        return "selector", selector
    if key == "data":
        return "data", bytes_from_hex(raw_value)
    raise Unsupported(f"unknown revert expectation field {key}")


def extract_rpc_revert_data(error: object) -> bytes | None:
    if isinstance(error, dict):
        if "data" in error:
            extracted = extract_rpc_revert_data(error["data"])
            if extracted is not None:
                return extracted
        for key in ("result", "originalError", "error"):
            if key in error:
                extracted = extract_rpc_revert_data(error[key])
                if extracted is not None:
                    return extracted
        return None
    if isinstance(error, str):
        value = error.strip()
        if re.fullmatch(r"0[xX][0-9a-fA-F]*", value):
            return bytes_from_hex(value)
        match = re.search(r"(?:data|reverted)[^\n]*(0[xX][0-9a-fA-F]*)", value, re.IGNORECASE)
        if match:
            return bytes_from_hex(match.group(1))
    return None


def compare_revert_expectation(expectation: str, rpc_error: object) -> tuple[bool, str]:
    kind, expected = parse_expected_revert(expectation)
    if kind == "any":
        return True, "any revert"
    actual = extract_rpc_revert_data(rpc_error)
    if actual is None:
        return False, "RPC error contained no revert data"
    if kind == "selector":
        ok = len(actual) >= 4 and actual[:4] == expected
        return ok, f"data=0x{actual.hex()} expected-selector=0x{expected.hex()}"
    ok = actual == expected
    return ok, f"data=0x{actual.hex()} expected=0x{expected.hex()}"


# --- supported subset ---------------------------------------------------------

EMPTY_RET = re.compile(r"\{\s*\}")


class Unsupported(Exception):
    pass


def encode_call(sig: str, args: list[str]) -> str:
    # cast calldata handles selector + ABI encoding for static types.
    return cast("calldata", sig, *args, rpc=False)


def source_stem(path: Path) -> str:
    return path.stem


def wire_type(abi_doc: dict, type_id: str) -> str:
    entry = abi_doc["types"][type_id]
    wire = entry.get("wire", {}).get("evm-default", {})
    if "type" in wire:
        return wire["type"]
    if wire.get("as") == "tuple":
        fields = entry.get("fields", [])
        return "(" + ",".join(wire_type(abi_doc, f["typeId"]) for f in fields) + ")"
    raise Unsupported(f"unsupported ABI type {type_id}")


def constructor_sig(abi_doc: dict) -> str | None:
    for callable_doc in abi_doc.get("callables", []):
        if callable_doc.get("kind") != "constructor":
            continue
        inputs = callable_doc.get("inputs", [])
        return "constructor(" + ",".join(wire_type(abi_doc, i["typeId"]) for i in inputs) + ")"
    return None


def compile_contract(ora_path: Path, outdir: Path, compiler_args: list[str]) -> tuple[str, dict]:
    progress(f"compile {ora_path.relative_to(ROOT)}")
    p = sh([str(ORA), "emit", *compiler_args, "--emit=abi,bytecode", str(ora_path), "-o", str(outdir)])
    if p.returncode != 0:
        raise RuntimeError(f"compile failed for {ora_path}: {p.stderr[:300]}")
    stem = source_stem(ora_path)
    hex_path = outdir / f"{stem}.hex"
    abi_path = outdir / f"{stem}.abi.json"
    if not hex_path.exists():
        raise RuntimeError(f"no .hex artifact produced for {ora_path}")
    if not abi_path.exists():
        raise RuntimeError(f"no .abi.json artifact produced for {ora_path}")
    bytecode = hex_path.read_text().strip()
    if not bytecode.startswith("0x"):
        bytecode = "0x" + bytecode
    return bytecode, json.loads(abi_path.read_text())


def resolve_args(args: list[str], addresses: dict[str, str], target_addr: str | None) -> list[str]:
    resolved = []
    for arg in args:
        if arg == "$contract":
            if target_addr is None:
                raise Unsupported("$contract has no target address yet")
            resolved.append(target_addr)
        elif arg.startswith("@"):
            name = arg[1:]
            if name not in addresses:
                raise Unsupported(f"unknown contract ref @{name}")
            resolved.append(addresses[name])
        else:
            resolved.append(arg)
    return resolved


def encode_constructor_args(abi_doc: dict, args: list[str]) -> str:
    if not args:
        return ""
    sig = constructor_sig(abi_doc)
    if sig is None:
        raise Unsupported("constructor args provided but ABI has no constructor")
    encoded = cast("abi-encode", sig, *args, rpc=False)
    return encoded[2:] if encoded.startswith("0x") else encoded


def deploy_contract(bytecode: str, abi_doc: dict, args: list[str], caller: str | None, value: int, addresses: dict[str, str]) -> str:
    initcode = bytecode + encode_constructor_args(abi_doc, resolve_args(args, addresses, None))
    from_addr = caller or DEV_ADDRESS
    fund(from_addr)
    ok, receipt = send_transaction(from_addr, None, initcode, value, stage="deploy")
    if not ok or not isinstance(receipt, dict):
        raise RuntimeError(f"contract deployment reverted: {receipt}")
    address = receipt.get("contractAddress")
    if not address:
        raise RuntimeError(f"contract deployment did not return an address: {receipt}")
    progress(f"deploy: checking code at {address}")
    code = rpc_code(address)
    if code in ("", "0x"):
        raise RuntimeError(f"contract deployment produced empty code at {address}")
    return address


def parse_cli(argv: list[str]) -> tuple[Path, list[str]]:
    args = list(argv)
    verified = False
    keep_proved_checks = False
    while args and args[0].startswith("--"):
        option = args.pop(0)
        if option == "--verified":
            verified = True
        elif option == "--keep-proved-checks":
            keep_proved_checks = True
        else:
            raise Unsupported(f"unknown option {option}")
    if len(args) != 1:
        raise Unsupported("expected exactly one spec.toml path")
    if keep_proved_checks and not verified:
        raise Unsupported("--keep-proved-checks requires --verified")
    compiler_args = [] if verified else ["--no-verify"]
    if keep_proved_checks:
        compiler_args.append("--keep-proved-checks")
    return Path(args[0]), compiler_args


def main() -> int:
    try:
        spec_path, compiler_args = parse_cli(sys.argv[1:])
    except Unsupported as err:
        print(f"usage: conformance-anvil-diff.py [--verified [--keep-proved-checks]] <spec.toml>\nerror: {err}", file=sys.stderr)
        return 2
    if not spec_path.is_absolute():
        spec_path = ROOT / spec_path
    spec = read_spec(spec_path)

    source = spec["deploy"].get("source", "").strip('"')
    ora_path = ROOT / source if source else spec_path.with_suffix("").with_suffix(".ora")
    if not ora_path.exists():
        print(f"diff: source .ora not found: {ora_path}", file=sys.stderr)
        return 2

    # Compile to bytecode + abi in an isolated directory. The build target may
    # run several specs concurrently, so a shared scratch dir races artifact
    # deletion against sibling runs.
    outdir = Path(tempfile.mkdtemp(prefix="anvil-diff-", dir=ROOT / ".zig-cache"))
    try:
        return run_spec(spec, spec_path, ora_path, outdir, compiler_args)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def run_spec(spec: dict, spec_path: Path, ora_path: Path, outdir: Path, compiler_args: list[str]) -> int:

    progress(f"spec {spec_path.relative_to(ROOT)}")
    bytecode, abi_doc = compile_contract(ora_path, outdir, compiler_args)

    addresses: dict[str, str] = {}

    # Deploy from the spec's deploy caller (impersonated), so msg.sender matches.
    deployer = spec["deploy"].get("caller", "").strip('"') or None
    addr = deploy_contract(
        bytecode,
        abi_doc,
        parse_args(spec["deploy"].get("args", "[]")),
        deployer,
        parse_int(spec["deploy"].get("value", "0")),
        addresses,
    )
    addresses["self"] = addr
    print(f"diff: deployed {ora_path.name} at {addr} on Anvil")
    ignore_logs = parse_bool(spec["deploy"].get("ignore_logs", "false"))

    for contract in spec.get("contracts", []):
        name = contract.get("name", "").strip('"')
        source = contract.get("source", "").strip('"')
        if not name or not source:
            raise Unsupported("[[contract]] requires name and source")
        secondary_path = ROOT / source
        bytecode, secondary_abi = compile_contract(secondary_path, outdir, compiler_args)
        caddr = deploy_contract(
            bytecode,
            secondary_abi,
            parse_args(contract.get("args", "[]")),
            contract.get("caller", "").strip('"') or deployer,
            parse_int(contract.get("value", "0")),
            addresses,
        )
        addresses[name] = caddr
        print(f"diff: deployed {secondary_path.name} as {name} at {caddr} on Anvil")

    diverged = 0
    checked = 0
    skipped = 0
    for call in spec["calls"]:
        sig = call.get("fn", "").strip('"')
        raw_calldata = call.get("calldata", "").strip('"')
        if not sig and not raw_calldata:
            skipped += 1
            continue
        target_name = call.get("to", "").strip('"')
        target_addr = addresses.get(target_name, addr) if target_name else addr
        args = resolve_args(parse_args(call.get("args", "[]")), addresses, target_addr) if sig else []
        label = sig or f"raw:{raw_calldata[:10]}"
        calldata = raw_calldata if raw_calldata else encode_call(sig, args)

        caller = call.get("caller", "").strip('"') or deployer
        value = parse_int(call.get("value", "0"))
        from_addr = caller or DEV_ADDRESS
        fund(from_addr)

        # For every call: eth_call to read the return against committed state,
        # then SEND to commit any state change (mirrors the harness, which both
        # returns and commits per call).
        receipt_for_assertions: object | None = None
        if "returns" in call:
            ret = call["returns"]
            if EMPTY_RET.fullmatch(ret.strip()):
                ok, out = send_transaction(from_addr, target_addr, calldata, value, stage=f"{label}:tx")
                receipt_for_assertions = out
                if not ok:
                    diverged += 1
                    print(f"  [DIVERGE] {label} -> anvil tx reverted while empty return expected ({str(out)[:80]})")
                if not check_receipt_gas(label, call, out):
                    diverged += 1
                checked += 1
            else:
                progress(f"{label}: eth_call")
                out = rpc_call(from_addr, target_addr, calldata, value)
                try:
                    ok, observed = compare_return_spec(ret, out)
                except Unsupported as err:
                    print(f"  [skip] {label}: {err}")
                    skipped += 1
                    continue
                tx_ok, receipt = send_transaction(from_addr, target_addr, calldata, value, stage=f"{label}:commit")  # commit state change
                receipt_for_assertions = receipt
                if not tx_ok:
                    ok = False
                if not check_receipt_gas(label, call, receipt):
                    ok = False
                mark = "OK" if ok else "DIVERGE"
                if not ok:
                    diverged += 1
                note = f" tx_status=reverted" if not tx_ok else ""
                print(f"  [{mark}] {label} -> anvil={observed} expected(lib/evm)={ret.strip()}{note}")
                checked += 1
        elif "reverts" in call:
            progress(f"{label}: eth_call expect revert")
            ok, out = rpc_call_try(from_addr, target_addr, calldata, value)
            mark = "OK" if not ok else "DIVERGE"
            if ok:
                diverged += 1
                note = "did NOT revert"
            else:
                revert_matches, note = compare_revert_expectation(call["reverts"], out)
                if not revert_matches:
                    mark = "DIVERGE"
                    diverged += 1
                if gas_ceiling(call) is not None:
                    tx_ok, receipt = send_transaction(from_addr, target_addr, calldata, value, stage=f"{label}:revert-tx")
                    receipt_for_assertions = receipt
                    if tx_ok:
                        mark = "DIVERGE"
                        diverged += 1
                    if not check_receipt_gas(label, call, receipt):
                        mark = "DIVERGE"
                        diverged += 1
            print(f"  [{mark}] {label} -> anvil reverts ({note})")
            checked += 1
        elif "succeeds" in call:
            ok, receipt = send_transaction(from_addr, target_addr, calldata, value, stage=f"{label}:tx")
            receipt_for_assertions = receipt
            if not check_receipt_gas(label, call, receipt):
                ok = False
            mark = "OK" if ok else "DIVERGE"
            if not ok:
                diverged += 1
            print(f"  [{mark}] {label} -> anvil {'succeeds' if ok else 'REVERTED'}")
            checked += 1
        else:
            print(f"  [skip] {label}: no recognized outcome")
            skipped += 1
            continue

        # Storage-slot assertions.
        for sa in call.get("storage", []):
            slot = sa.get("slot", "").strip('"')
            want = parse_int(sa.get("value", "0"))
            got = rpc_storage(target_addr, parse_slot_value(slot))
            mark = "OK" if got == want else "DIVERGE"
            if got != want:
                diverged += 1
            print(f"    [{mark}] slot {slot} -> anvil={got} expected={want}")
            checked += 1

        log_checks, log_divergences = check_receipt_logs(
            label,
            call,
            receipt_for_assertions,
            ignore_logs=ignore_logs,
        )
        checked += log_checks
        diverged += log_divergences

    print(f"diff: {checked} checks, {diverged} divergences, {skipped} skipped (unsupported in scaffold)")
    return 1 if diverged else 0


if __name__ == "__main__":
    sys.exit(main())
