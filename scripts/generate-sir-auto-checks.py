#!/usr/bin/env python3
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


def main() -> int:
    if not ORA_BIN.exists():
        print(f"error: ora binary not found at {ORA_BIN}. Run 'zig build' first.")
        return 1

    CHECKS_DIR.mkdir(parents=True, exist_ok=True)

    covered = set()
    for check in CHECKS_DIR.glob("*.check"):
        try:
            text = check.read_text()
        except Exception:
            continue
        m = re.search(r"^// INPUT:\s*(.+)$", text, re.M)
        if m:
            covered.add(m.group(1).strip())

    ora_files = sorted((REPO / "ora-example").rglob("*.ora"))

    created = 0
    failed = []

    for path in ora_files:
        rel = path.relative_to(REPO).as_posix()
        if should_skip(path):
            continue
        if rel in covered:
            continue

        try:
            proc = subprocess.run(
                [str(ORA_BIN), "--emit-mlir=sir", str(path)],
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
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

        marker = "SIR MLIR (after phase5)"
        if marker in out:
            sir = out.split(marker, 1)[1]
        else:
            sir = out
        func_names = re.findall(r"func\\.func\\s+@([A-Za-z0-9_]+)", sir)

        check_name = "auto__" + rel.replace("/", "__").replace(".ora", ".check")
        check_path = CHECKS_DIR / check_name

        lines = []
        lines.append("// AUTO-GENERATED. Do not edit by hand.")
        lines.append(f"// INPUT: {rel}")
        lines.append("// CHECK-NOT: builtin.unrealized_conversion_cast")
        if func_names:
            for name in func_names:
                lines.append(f"// CHECK-LABEL: func.func @{name}")
        else:
            lines.append("// CHECK: module")

        check_path.write_text("\\n".join(lines) + "\\n")
        created += 1

    print(f"created {created} checks")
    if failed:
        print("failed:")
        for rel, reason in failed:
            print(f"  {rel}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
