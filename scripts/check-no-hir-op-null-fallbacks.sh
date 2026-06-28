#!/usr/bin/env sh
set -eu

fail() {
  echo "check-no-hir-op-null-fallbacks: $*" >&2
  exit 1
}

[ -d src/hir ] || fail "missing src/hir directory"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path("src/hir")

# Op creation failures must either diagnose/error or produce an explicitly
# marked executable fallback. Returning null, an old operand, or a prior value
# silently converts a failed MLIR construction into emittable IR.
pattern = re.compile(
    r"if\s*\(\s*(?!!)mlir\.oraOperationIsNull\([^)]*\)\s*\)\s*"
    r"(?:\{\s*)?"
    r"(?P<stmt>"
    r"return\s+(?!(?:error|try|true|false)\b)(?:null|[A-Za-z_][A-Za-z0-9_]*)\s*;"
    r"|break\s+:[A-Za-z_][A-Za-z0-9_]*\s+(?:null|[A-Za-z_][A-Za-z0-9_]*)\s*;"
    r")",
    re.DOTALL,
)

violations: list[str] = []
for path in sorted(root.rglob("*.zig")):
    if path.as_posix() == "src/hir/executable_fallbacks.zig":
        continue
    text = path.read_text()
    for match in pattern.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        snippet = " ".join(match.group("stmt").split())
        violations.append(f"{path}:{line}: {snippet}")

if violations:
    print("\n".join(violations), file=sys.stderr)
    print(
        "check-no-hir-op-null-fallbacks: MLIR op-creation failure must fail closed, not return an old operand/value/null",
        file=sys.stderr,
    )
    sys.exit(1)
PY

echo "check-no-hir-op-null-fallbacks: ok"
