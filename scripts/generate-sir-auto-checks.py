#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORA_BIN = REPO / "zig-out" / "bin" / "ora"
CHECKS_DIR = REPO / "tests" / "mlir_sir"

SKIP_MARKERS = ["/invalid/", "/negative_tests/"]


def should_skip(path: Path) -> bool:
    s = str(path)
    if any(m in s for m in SKIP_MARKERS):
        return True
    name = path.name
    if name.startswith("fail_") or "/fail_" in s or "fail_region_err" in s:
        return True
    if "/logs/log_fail_" in s:
        return True
    return False


def extract_sir_output(stdout: str) -> str:
    marker = "SIR MLIR (after phase5)"
    if marker in stdout:
        stdout = stdout.split(marker, 1)[1]
    lines = [line.rstrip() for line in stdout.strip("\n").splitlines() if line.rstrip() != ""]
    for index, line in enumerate(lines):
        if line.startswith("module ") or line.startswith('"builtin.module"'):
            lines = lines[index:]
            break
    else:
        return ""
    return "\n".join(lines)


def filecheck_exact_line(line: str) -> str:
    line = line.replace(REPO.as_posix(), "{{.*}}")
    # SIR switch case lists can contain literal double brackets such as
    # [[0, 1]]. FileCheck treats that shape as a variable capture/use, so escape
    # only the double-bracket delimiter and keep other arrays readable.
    return line.replace("[[", "{{\\[\\[}}").replace("]]", "{{\\]\\]}}")


def build_full_snapshot_check(rel: str, sir: str) -> str:
    lines = [
        "// AUTO-GENERATED. Do not edit by hand.",
        f"// INPUT: {rel}",
        "// CHECK-NOT: builtin.unrealized_conversion_cast",
    ]

    sir_lines = sir.splitlines()
    if not sir_lines:
        lines.append("// CHECK: {{^$}}")
    else:
        lines.append(f"// CHECK: {filecheck_exact_line(sir_lines[0])}")
        for line in sir_lines[1:]:
            lines.append(f"// CHECK-NEXT: {filecheck_exact_line(line)}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SIR MLIR FileCheck snapshots for ora-example programs.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rewrite existing auto-generated checks instead of only creating missing ones",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ORA_FILE",
        help="generate or refresh a specific repository-relative .ora input; may be repeated",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help="per-input compiler timeout (default: 120)",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    if not ORA_BIN.exists():
        print(f"error: ora binary not found at {ORA_BIN}. Run 'zig build' first.")
        return 1

    CHECKS_DIR.mkdir(parents=True, exist_ok=True)

    covered = set()
    refresh_checks = []
    for check in CHECKS_DIR.glob("*.check"):
        try:
            text = check.read_text()
        except Exception:
            continue
        is_auto_generated = text.startswith("// AUTO-GENERATED. Do not edit by hand.")
        if "\\n// INPUT:" in text:
            text = text.replace("\\n", "\n")
        m = re.search(r"^// INPUT:\s*(.+)$", text, re.M)
        if m:
            rel = m.group(1).strip()
            covered.add(rel)
            if is_auto_generated:
                refresh_checks.append((check, rel))

    ora_files = sorted((REPO / "ora-example").rglob("*.ora"))
    if args.only:
        ora_files = [REPO / rel for rel in args.only]
    elif args.refresh:
        ora_files = [REPO / rel for _, rel in sorted(refresh_checks, key=lambda item: item[1])]

    written = 0
    failed = []
    ora_verify_flag = os.environ.get("ORA_VERIFY_FLAG", "--no-verify")
    artifact_dir = REPO / "artifacts" / "sir-checks"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for path in ora_files:
        rel = path.relative_to(REPO).as_posix()
        if should_skip(path):
            continue
        if rel in covered and not args.refresh and not args.only:
            continue

        try:
            command = [str(ORA_BIN)]
            if ora_verify_flag:
                command.append(ora_verify_flag)
            command.extend(["--emit=mlir:sir", str(path)])
            proc = subprocess.run(
                command,
                cwd=artifact_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=args.timeout,
            )
        except Exception as e:
            failed.append((rel, f"exception {e}"))
            continue
        if proc.returncode != 0:
            failed.append((rel, f"exit {proc.returncode}"))
            continue

        out = proc.stdout
        if not out.strip():
            failed.append((rel, "empty output"))
            continue

        sir = extract_sir_output(out)
        if not sir.strip():
            failed.append((rel, "empty SIR output"))
            continue

        if args.refresh and not args.only:
            check_path = next(path for path, input_rel in refresh_checks if input_rel == rel)
        else:
            check_name = "auto__" + rel.replace("/", "__").replace(".ora", ".check")
            check_path = CHECKS_DIR / check_name
        check_path.write_text(build_full_snapshot_check(rel, sir))
        written += 1

    action = "refreshed" if args.refresh else "created"
    print(f"{action} {written} checks")
    if failed:
        print("failed:")
        for rel, reason in failed:
            print(f"  {rel}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
