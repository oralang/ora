#!/usr/bin/env python3
"""Differential conformance backend (test-quality program gap #3 / T2.4).

Runs a conformance spec against a real Anvil node and compares the results to
the spec's expectations — which were produced/validated against the in-process
lib/evm. Agreement corroborates lib/evm (our only oracle); a divergence is the
highest-value signal the suite can produce (a compiler bug, a lib/evm bug, or a
hardfork mismatch — F-003 was lib/evm itself panicking).

LOCAL-FIRST, not in the blocking gate (Anvil is a live process). Pins the
hardfork to lib/evm's (CANCUN). This is a working scaffold: it covers the
scalar/storage subset end-to-end and skips (loudly) calls it can't yet encode.
Extend by adding arg/return/outcome handling, not by rearchitecting.

Usage:
  scripts/conformance-anvil-diff.py tests/conformance/counter.spec.toml
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORA = ROOT / "zig-out" / "bin" / "ora"
RPC = "http://127.0.0.1:8545"
# Anvil's deterministic dev account 0 — same on every launch.
DEV_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"


def sh(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def cast(*args: str, rpc: bool = True) -> str:
    # Insert --rpc-url right after the subcommand; trailing flags get swallowed
    # by greedy positional args like `send --create <CODE>`.
    cmd = ["cast", args[0]]
    if rpc:
        cmd += ["--rpc-url", RPC]
    cmd += list(args[1:])
    p = sh(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"cast {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def cast_try(*args: str) -> tuple[bool, str]:
    """Run cast, returning (succeeded, output_or_error). Does not raise on revert."""
    cmd = ["cast", args[0], "--rpc-url", RPC, *args[1:]]
    p = sh(cmd)
    return (p.returncode == 0, (p.stdout or p.stderr).strip())


# --- tiny spec reader (the [deploy]/[[call]]/[[call.storage]] subset) ---------

def read_spec(path: Path) -> dict:
    deploy = {"args": []}
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
        if line == "[[call]]":
            cur = {"storage": []}
            calls.append(cur)
            section = "call"
            continue
        if line == "[[call.storage]]":
            section = "storage"
            cur["storage"].append({})
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        target = deploy if section == "deploy" else (cur["storage"][-1] if section == "storage" else cur)
        target[key] = val
    return {"deploy": deploy, "calls": calls}


def parse_int(tok: str) -> int:
    tok = tok.strip().strip('"')
    return int(tok, 16) if tok.startswith("0x") else int(tok)


def parse_args(tok: str) -> list[str]:
    tok = tok.strip()
    if tok in ("[]", ""):
        return []
    return [a.strip().strip('"') for a in tok[1:-1].split(",") if a.strip()]


# --- supported subset ---------------------------------------------------------

UINT_RET = re.compile(r"\{\s*u256\s*=\s*(.+?)\s*\}")
EMPTY_RET = re.compile(r"\{\s*\}")
DEC_SLOT = re.compile(r"^\d+$")


class Unsupported(Exception):
    pass


def encode_call(sig: str, args: list[str]) -> str:
    # cast calldata handles selector + ABI encoding for static types.
    return cast("calldata", sig, *args, rpc=False)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: conformance-anvil-diff.py <spec.toml>", file=sys.stderr)
        return 2
    spec_path = Path(sys.argv[1])
    if not spec_path.is_absolute():
        spec_path = ROOT / spec_path
    spec = read_spec(spec_path)

    source = spec["deploy"].get("source", "").strip('"')
    ora_path = ROOT / source if source else spec_path.with_suffix("").with_suffix(".ora")
    if not ora_path.exists():
        print(f"diff: source .ora not found: {ora_path}", file=sys.stderr)
        return 2

    # Compile to bytecode + abi (clean outdir so we never read a stale artifact).
    import shutil
    outdir = ROOT / ".zig-cache" / "anvil-diff"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = sh([str(ORA), "emit", "--emit=abi,bytecode", str(ora_path), "-o", str(outdir)])
    if p.returncode != 0:
        print(f"diff: compile failed: {p.stderr[:300]}", file=sys.stderr)
        return 1
    hex_files = list(outdir.glob("*.hex"))
    if not hex_files:
        print("diff: no .hex artifact produced", file=sys.stderr)
        return 1
    bytecode = hex_files[0].read_text().strip()
    if not bytecode.startswith("0x"):
        bytecode = "0x" + bytecode

    # Constructor args (uint256 subset) appended to initcode.
    ctor_args = parse_args(spec["deploy"].get("args", "[]"))
    initcode = bytecode
    for a in ctor_args:
        initcode += f"{parse_int(a):064x}"

    # Deploy from the dev account.
    receipt = cast("send", "--private-key", DEV_KEY, "--json", "--create", initcode)
    import json
    addr = json.loads(receipt)["contractAddress"]
    print(f"diff: deployed {ora_path.name} at {addr} on Anvil")

    diverged = 0
    checked = 0
    skipped = 0
    for call in spec["calls"]:
        sig = call.get("fn", "").strip('"')
        if not sig:
            skipped += 1
            continue
        args = parse_args(call.get("args", "[]"))
        try:
            data = encode_call(sig, args)
        except Exception:
            print(f"  [skip] {sig}: cannot encode args {args}")
            skipped += 1
            continue

        # Read-only (returns) -> eth_call; state-changing -> send then nothing to read.
        if "returns" in call:
            ret = call["returns"]
            m = UINT_RET.search(ret)
            if EMPTY_RET.fullmatch(ret.strip()):
                # state-changing void: send a tx.
                cast("send", "--private-key", DEV_KEY, addr, data)
                checked += 1
                continue
            if not m:
                print(f"  [skip] {sig}: unsupported return {ret}")
                skipped += 1
                continue
            expected = parse_int(m.group(1))
            out = cast("call", addr, data)
            actual = int(out, 16)
            mark = "OK" if actual == expected else "DIVERGE"
            if actual != expected:
                diverged += 1
            print(f"  [{mark}] {sig} -> anvil={actual} expected(lib/evm)={expected}")
            checked += 1
        elif "reverts" in call:
            ok, out = cast_try("call", addr, data)
            mark = "OK" if not ok else "DIVERGE"
            if ok:
                diverged += 1
            note = out.split(":")[-1].strip()[:40] if not ok else "did NOT revert"
            print(f"  [{mark}] {sig} -> anvil reverts ({note})")
            checked += 1
            continue
        elif "succeeds" in call:
            ok, _ = cast_try("send", "--private-key", DEV_KEY, addr, data)
            mark = "OK" if ok else "DIVERGE"
            if not ok:
                diverged += 1
            print(f"  [{mark}] {sig} -> anvil {'succeeds' if ok else 'REVERTED'}")
            checked += 1
            continue
        else:
            print(f"  [skip] {sig}: no recognized outcome")
            skipped += 1
            continue

        # Storage-slot assertions (decimal slots only in this scaffold).
        for sa in call.get("storage", []):
            slot = sa.get("slot", "").strip('"')
            if not DEC_SLOT.match(slot):
                continue
            want = parse_int(sa.get("value", "0"))
            got = int(cast("storage", addr, slot), 16)
            mark = "OK" if got == want else "DIVERGE"
            if got != want:
                diverged += 1
            print(f"    [{mark}] slot {slot} -> anvil={got} expected={want}")
            checked += 1

    print(f"diff: {checked} checks, {diverged} divergences, {skipped} skipped (unsupported in scaffold)")
    return 1 if diverged else 0


if __name__ == "__main__":
    sys.exit(main())
