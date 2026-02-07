#!/usr/bin/env bash
set -euo pipefail

FILECHECK_BIN=${FILECHECK:-./vendor/llvm-project/build-mlir/bin/FileCheck}
ORA_BIN=${ORA_BIN:-./zig-out/bin/ora}

if ! command -v "$FILECHECK_BIN" >/dev/null 2>&1; then
  echo "error: FileCheck not found. Set FILECHECK=/path/to/FileCheck" >&2
  exit 1
fi

if [ ! -x "$ORA_BIN" ]; then
  echo "error: ora binary not found at $ORA_BIN" >&2
  exit 1
fi

fail=0
while IFS= read -r -d '' check; do
  input=$(rg -m1 "^// INPUT:" "$check" | sed -E 's#^// INPUT: ##')
  if [ -z "$input" ]; then
    echo "error: missing // INPUT: in $check" >&2
    fail=1
    continue
  fi
  if [[ "$input" == path/to/* ]]; then
    continue
  fi

  if [ ! -f "$input" ]; then
    echo "error: input file not found: $input (from $check)" >&2
    fail=1
    continue
  fi

  echo "[mlir-check-sir] $check -> $input"
  "$ORA_BIN" --emit-sir "$input" 2>/dev/null | \
    awk '
      /SIR MLIR \(after phase5\)/ {f=1; print; next}
      /^module / {if (!f) f=1}
      /^"builtin.module"/ {if (!f) f=1}
      f {print}
    ' | \
    "$FILECHECK_BIN" "$check" || fail=1

done < <(find tests/mlir_sir -name "*.check" -print0 | sort -z)

exit $fail
