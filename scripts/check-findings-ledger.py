#!/usr/bin/env python3
"""Known-defects ledger tripwire (test-quality program; gap #2).

The gate is intentionally GREEN over known compiler/EVM defects: characterization
rows assert the current WRONG behavior so the suite locks it until it is fixed.
That means a green gate silently masks N known bugs. This check makes that
visible and keeps the ledger honest:

  - every `FINDINGS.md#f-NNN` reference in tests/ resolves to a real finding;
  - a finding marked FIXED must have ZERO remaining references (fixing a bug
    forces you to promote/remove its characterization pins — they cannot rot);
  - it prints, every gate run, exactly how many OPEN findings and how many
    characterization pins the green is hiding, by severity.

Fixing a defect makes its characterization specs go RED naturally (actual !=
pinned-wrong). When that happens the message to the developer is: the bug is
fixed — promote the pin to a real assertion and mark the finding FIXED here.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "tests" / "conformance" / "FINDINGS.md"
TESTS_DIR = ROOT / "tests"

HEADER_RE = re.compile(r"^## (F-\d+)\b", re.M)
STATUS_RE = re.compile(r"^- \*\*Status:\*\*\s*(\w+)", re.M)
SEVERITY_RE = re.compile(r"^- \*\*Severity:\*\*\s*(S\d)", re.M)
OWNER_RE = re.compile(r"^- \*\*Owner:\*\*\s*(.+)$", re.M)
REF_RE = re.compile(r"FINDINGS\.md#(f-\d+)", re.I)


def fail(msg: str) -> None:
    print(f"check-findings-ledger: {msg}", file=sys.stderr)
    raise SystemExit(1)


def parse_ledger() -> dict[str, dict]:
    if not LEDGER.exists():
        fail(f"ledger not found: {LEDGER}")
    text = LEDGER.read_text()
    # Split into per-finding sections so Status/Severity bind to their header.
    findings: dict[str, dict] = {}
    headers = list(HEADER_RE.finditer(text))
    for i, h in enumerate(headers):
        fid = h.group(1).lower()
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end]
        status_m = STATUS_RE.search(body)
        sev_m = SEVERITY_RE.search(body)
        owner_m = OWNER_RE.search(body)
        if not status_m:
            fail(f"{fid.upper()} has no Status field")
        if not sev_m:
            fail(f"{fid.upper()} has no Severity field")
        if not owner_m:
            fail(f"{fid.upper()} has no Owner field")
        findings[fid] = {
            "status": status_m.group(1).upper(),
            "severity": sev_m.group(1),
            "owner": owner_m.group(1).strip(),
        }
    if not findings:
        fail("no findings parsed from ledger")
    return findings


def collect_references() -> dict[str, list[str]]:
    refs: dict[str, list[str]] = defaultdict(list)
    for path in sorted(TESTS_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in (".toml", ".ora"):
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for m in REF_RE.finditer(text):
            refs[m.group(1).lower()].append(str(path.relative_to(ROOT)))
    return refs


def main() -> None:
    findings = parse_ledger()
    refs = collect_references()

    # 1. No dangling references.
    for fid, where in refs.items():
        if fid not in findings:
            fail(f"reference to unknown finding {fid.upper()} in {where[0]}")

    # 2. FIXED findings must have no remaining characterization pins.
    for fid, info in findings.items():
        if info["status"] == "FIXED" and refs.get(fid):
            fail(
                f"{fid.upper()} is marked FIXED but still has characterization "
                f"pins in {', '.join(sorted(set(refs[fid])))} — promote or remove them"
            )

    # 3. Loud, honest summary of what green is hiding.
    open_ids = [f for f, i in findings.items() if i["status"] == "OPEN"]
    by_sev: dict[str, int] = defaultdict(int)
    total_pins = 0
    pinned_files: set[str] = set()
    for fid in open_ids:
        by_sev[findings[fid]["severity"]] += 1
        total_pins += len(refs.get(fid, []))
        pinned_files.update(refs.get(fid, []))

    sev_summary = ", ".join(f"{by_sev[s]}×{s}" for s in sorted(by_sev)) or "none"
    print(
        f"check-findings-ledger: {len(findings)} findings tracked; "
        f"{len(open_ids)} OPEN ({sev_summary}); "
        f"{total_pins} characterization pins across {len(pinned_files)} files."
    )
    if open_ids:
        print(
            "  NOTE: a GREEN gate intentionally MASKS these known defects "
            "(characterization pins lock current wrong behavior). Not 'safe' — 'unchanged'."
        )
        for fid in sorted(open_ids):
            n = len(refs.get(fid, []))
            tag = f"{n} pins" if n else "no pins (tracked elsewhere)"
            info = findings[fid]
            print(f"    {fid.upper()} ({info['severity']}) OPEN — owner: {info['owner']} — {tag}")

    # Gap #1 visibility: the dangerous backlog must be impossible to miss.
    s1_open = sorted(f for f in open_ids if findings[f]["severity"] == "S1")
    if s1_open:
        print(
            f"  ⚠ UNFIXED S1 BACKLOG ({len(s1_open)}): "
            + ", ".join(f.upper() for f in s1_open)
            + " — these are dangerous and OPEN; they need a fix owner, not just a pin."
        )


if __name__ == "__main__":
    main()
