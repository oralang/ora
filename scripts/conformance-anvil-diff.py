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


_funded: set[str] = set()


def fund(addr: str) -> None:
    """Give an impersonated caller enough balance for gas + value."""
    a = addr.lower()
    if a in _funded:
        return
    cast("rpc", "anvil_setBalance", addr, "0xffffffffffffffffffffffff")
    _funded.add(a)


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
    # cast wants array literals comma-packed without spaces: [5,8,13]
    return [re.sub(r"\s+", "", a) if a.startswith("[") else a for a in out]


RET_TYPED = re.compile(r"\{\s*(u256|bool|address)\s*=\s*(.+?)\s*\}")


def return_matches(rtype: str, expected_tok: str, raw_hex: str) -> tuple[bool, int]:
    """Compare a 32-byte return word against the expected typed value."""
    word = int(raw_hex, 16)
    et = expected_tok.strip().strip('"')
    if rtype == "u256":
        return (word == parse_int(et), word)
    if rtype == "bool":
        want = et.lower() in ("true", "1")
        return ((word != 0) == want, word)
    if rtype == "address":
        return ((word & ((1 << 160) - 1)) == parse_int(et), word)
    return (False, word)


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
    # --no-verify: match the conformance harness (it deploys bytecode regardless
    # of the SMT verifier, which has its own gate). Otherwise contracts with
    # non-trivial obligations fail to compile here.
    p = sh([str(ORA), "emit", "--no-verify", "--emit=abi,bytecode", str(ora_path), "-o", str(outdir)])
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

    # Deploy from the spec's deploy caller (impersonated), so msg.sender matches.
    deployer = spec["deploy"].get("caller", "").strip('"') or None
    import json
    if deployer:
        fund(deployer)
        receipt = cast("send", "--from", deployer, "--unlocked", "--json", "--create", initcode)
    else:
        receipt = cast("send", "--private-key", DEV_KEY, "--json", "--create", initcode)
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

        caller = call.get("caller", "").strip('"') or deployer
        value = parse_int(call.get("value", "0"))
        from_flags = (["--from", caller, "--unlocked"] if caller else ["--private-key", DEV_KEY])
        if caller:
            fund(caller)
        val_flags = (["--value", str(value)] if value else [])
        # eth_call honors --from / --value for msg.sender / msg.value semantics.
        call_ctx = (["--from", caller] if caller else []) + val_flags

        # For every call: eth_call to read the return against committed state,
        # then SEND to commit any state change (mirrors the harness, which both
        # returns and commits per call).
        if "returns" in call:
            ret = call["returns"]
            if EMPTY_RET.fullmatch(ret.strip()):
                cast("send", *from_flags, *val_flags, addr, data)
                checked += 1
                continue
            m = RET_TYPED.search(ret)
            if not m:
                print(f"  [skip] {sig}: unsupported return {ret}")
                skipped += 1
                continue
            out = cast("call", *call_ctx, addr, data)
            ok, word = return_matches(m.group(1), m.group(2), out)
            cast("send", *from_flags, *val_flags, addr, data)  # commit state change
            mark = "OK" if ok else "DIVERGE"
            if not ok:
                diverged += 1
            print(f"  [{mark}] {sig} -> anvil={word} expected(lib/evm)={ret.strip()}")
            checked += 1
        elif "reverts" in call:
            ok, out = cast_try("call", *call_ctx, addr, data)
            mark = "OK" if not ok else "DIVERGE"
            if ok:
                diverged += 1
            note = out.split(":")[-1].strip()[:40] if not ok else "did NOT revert"
            print(f"  [{mark}] {sig} -> anvil reverts ({note})")
            checked += 1
            continue
        elif "succeeds" in call:
            ok, _ = cast_try("send", *from_flags, *val_flags, addr, data)
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
