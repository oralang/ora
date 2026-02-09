#!/usr/bin/env bash
set -euo pipefail

FILECHECK_BIN=${FILECHECK:-./vendor/llvm-project/build-mlir/bin/FileCheck}
ORA_BIN=${ORA_BIN:-./zig-out/bin/ora}
 #ORA_VERIFY_FLAG=${ORA_VERIFY_FLAG:---no-verify}

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
  effective_check="$check"
  tmp_check=""
  # Some auto-generated checks were written with literal "\n" sequences.
  # Normalize them to real newlines for INPUT parsing and FileCheck.
  if rg -q "\\\\n// INPUT:" "$check"; then
    tmp_check=$(mktemp)
    sed 's/\\n/\
/g' "$check" > "$tmp_check"
    effective_check="$tmp_check"
  fi

  input=$(rg -m1 "^// INPUT:" "$effective_check" | sed -E 's#^// INPUT: ##' || true)
  if [ -z "$input" ]; then
    echo "error: missing // INPUT: in $check" >&2
    if [ -n "$tmp_check" ]; then rm -f "$tmp_check"; fi
    fail=1
    continue
  fi
  if [[ "$input" == path/to/* ]]; then
    if [ -n "$tmp_check" ]; then rm -f "$tmp_check"; fi
    continue
  fi

  if [ ! -f "$input" ]; then
    echo "error: input file not found: $input (from $check)" >&2
    if [ -n "$tmp_check" ]; then rm -f "$tmp_check"; fi
    fail=1
    continue
  fi

  echo "[mlir-check] $check -> $input"
  tmp_stdout=$(mktemp)
  tmp_stderr=$(mktemp)
  tmp_filtered=$(mktemp)
  #if ! "$ORA_BIN" "$ORA_VERIFY_FLAG" --emit-mlir "$input" >"$tmp_stdout" 2>"$tmp_stderr"; then
  if ! "$ORA_BIN" --emit-mlir "$input" >"$tmp_stdout" 2>"$tmp_stderr"; then
    echo "error: ora failed for $input (check: $check)" >&2
    sed -n '1,80p' "$tmp_stderr" >&2 || true
    rm -f "$tmp_stdout" "$tmp_stderr" "$tmp_filtered"
    if [ -n "$tmp_check" ]; then rm -f "$tmp_check"; fi
    fail=1
    continue
  fi

  awk '
    /Ora MLIR \(before conversion\)/ {f=1; print; next}
    /^module / {if (!f) f=1}
    f {print}
  ' "$tmp_stdout" > "$tmp_filtered"

  if [ ! -s "$tmp_filtered" ]; then
    echo "error: empty MLIR output for $input (check: $check)" >&2
    sed -n '1,80p' "$tmp_stderr" >&2 || true
    rm -f "$tmp_stdout" "$tmp_stderr" "$tmp_filtered"
    if [ -n "$tmp_check" ]; then rm -f "$tmp_check"; fi
    fail=1
    continue
  fi

  "$FILECHECK_BIN" "$effective_check" < "$tmp_filtered" || fail=1
  rm -f "$tmp_stdout" "$tmp_stderr" "$tmp_filtered"

  if [ -n "$tmp_check" ]; then rm -f "$tmp_check"; fi

done < <(find tests/mlir -name "*.check" -print0 | sort -z)

exit $fail
