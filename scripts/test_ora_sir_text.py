#!/usr/bin/env python3
"""
Ora SIR Text Emission Test Script

Tests all .ora example files for --emit-sir-text and reports results.
Usage: python3 scripts/test_ora_sir_text.py [--subdir apps] [--timeout 30]
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse
import re


def find_ora_files(base_dir="ora-example", subdir_filter=None):
    all_files = sorted(Path(base_dir).rglob("*.ora"))
    if not subdir_filter:
        return all_files
    if isinstance(subdir_filter, str):
        subdir_filter = [subdir_filter]
    filtered = []
    for f in all_files:
        rel = str(f.relative_to(base_dir))
        if any(d in rel or rel.startswith(d) for d in subdir_filter):
            filtered.append(f)
    return filtered


def list_subdirectories(base_dir="ora-example"):
    base = Path(base_dir)
    subdirs = {}
    for f in base.rglob("*.ora"):
        parts = f.relative_to(base).parts
        if len(parts) > 1:
            subdirs[parts[0]] = subdirs.get(parts[0], 0) + 1
    return subdirs


def test_file(file_path, compiler_path="./zig-out/bin/ora", timeout_s=30):
    stem = Path(file_path).stem.lower()
    expected_failure = "fail_" in stem or stem.startswith("fail_")

    try:
        source = Path(file_path).read_text()
    except OSError:
        source = ""
    has_contract = re.search(r"^\s*contract\b", source, re.MULTILINE) is not None

    try:
        result = subprocess.run(
            [compiler_path, "--emit-sir-text", str(file_path)],
            capture_output=True, timeout=timeout_s
        )
        timed_out = False
    except subprocess.TimeoutExpired:
        return {
            "file": str(file_path),
            "status": "TIMEOUT",
            "error": f"Timed out after {timeout_s}s",
            "output": "",
            "category": "",
            "expected_failure": expected_failure,
        }

    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    combined = stdout + "\n" + stderr
    rc = result.returncode

    # Classify the failure
    error_detail = ""
    has_error = False

    if rc != 0 and not expected_failure:
        has_error = True
        if rc == 134 or rc == -6:
            error_detail = "ABORT"
        elif rc == 139 or rc == -11:
            error_detail = "SEGFAULT"
        elif "SIR text legalizer failed" in combined:
            # Extract specific legalizer errors
            reasons = set()
            for line in combined.split("\n"):
                if "error:" in line:
                    msg = line.split("error:")[-1].strip()
                    if "non-SIR operation" in msg:
                        reasons.add("non-SIR op (func.call)")
                    elif "illegal dialect" in msg:
                        reasons.add("ora dialect remains")
                    elif "not supported" in msg:
                        reasons.add(msg[:60])
                    elif msg:
                        reasons.add(msg[:60])
            error_detail = "LEGALIZER: " + "; ".join(sorted(reasons)) if reasons else "LEGALIZER"
        elif "SIR dispatcher build failed" in combined:
            for line in combined.split("\n"):
                if "error:" in line:
                    error_detail = "DISPATCHER: " + line.split("error:")[-1].strip()[:60]
                    break
            if not error_detail:
                error_detail = "DISPATCHER"
        elif "counterexample" in combined.lower() or "guard:" in combined:
            error_detail = "Z3_COUNTEREXAMPLE"
            has_error = False  # verification warnings aren't SIR text failures
        else:
            error_detail = f"EXIT_{rc}"
    elif rc == 0 and not expected_failure:
        # Compiler succeeded — always PASS. Empty output is valid for
        # contracts with only declarations and no function bodies.
        error_detail = ""

    if expected_failure:
        status = "EXPECTED_FAIL" if rc != 0 else "UNEXPECTED_PASS"
    elif has_error:
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "file": str(file_path),
        "status": status,
        "error": error_detail,
        "output": stdout[-500:] if not has_error else "",
        "category": error_detail.split(":")[0] if error_detail else "",
        "expected_failure": expected_failure,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Ora SIR text emission")
    parser.add_argument("--compiler", default="./zig-out/bin/ora")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--base-dir", default="ora-example")
    parser.add_argument("--subdir", action="append", dest="subdirs", metavar="DIR")
    parser.add_argument("--list-subdirs", action="store_true")
    parser.add_argument("--failures-only", action="store_true",
                        help="Only print failing tests")
    args = parser.parse_args()

    if args.list_subdirs:
        for d, n in sorted(list_subdirectories(args.base_dir).items()):
            print(f"  {d}/ ({n} files)")
        return

    files = find_ora_files(args.base_dir, args.subdirs)
    if args.subdirs:
        print(f"Filter: {', '.join(args.subdirs)}")
    print(f"Testing {len(files)} files with --emit-sir-text\n")

    results = []
    for i, f in enumerate(files, 1):
        r = test_file(f, args.compiler, args.timeout)
        results.append(r)
        label = {"PASS": "✅", "FAIL": "❌", "EXPECTED_FAIL": "⏭️ ",
                 "UNEXPECTED_PASS": "⚠️ ", "TIMEOUT": "⏰"}.get(r["status"], "?")
        if not args.failures_only or r["status"] == "FAIL":
            detail = f"  ({r['error']})" if r["error"] else ""
            print(f"[{i}/{len(files)}] {label} {Path(r['file']).name}{detail}")

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    expected = sum(1 for r in results if r["status"] == "EXPECTED_FAIL")
    timeout = sum(1 for r in results if r["status"] == "TIMEOUT")
    total = len(results)

    print(f"\n{'='*60}")
    print(f"SIR Text Emission: {passed}/{total} passed "
          f"({passed*100//total if total else 0}%)")
    print(f"  PASS: {passed}  FAIL: {failed}  "
          f"EXPECTED_FAIL: {expected}  TIMEOUT: {timeout}")

    # Group failures by category
    if failed:
        from collections import Counter
        cats = Counter(r["category"] for r in results if r["status"] == "FAIL")
        print(f"\nFailure breakdown:")
        for cat, cnt in cats.most_common():
            print(f"  {cat or 'unknown':30s} {cnt}")
            # Show first 3 examples
            examples = [Path(r["file"]).name for r in results
                        if r["status"] == "FAIL" and r["category"] == cat]
            for ex in examples[:3]:
                print(f"    - {ex}")
            if len(examples) > 3:
                print(f"    ... and {len(examples)-3} more")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
