#!/usr/bin/env bash
set -euo pipefail

FILECHECK_BIN=${FILECHECK:-./vendor/llvm-project/build-mlir/bin/FileCheck}
ORA_BIN=${ORA_BIN:-./zig-out/bin/ora}
ORA_VERIFY_FLAG=${ORA_VERIFY_FLAG:---no-verify}

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
  input=$(rg -m1 "^// INPUT:" "$check" | sed -E 's#^// INPUT: ##' || true)
  if [ -z "$input" ]; then
    echo "error: missing // INPUT: in $check" >&2
    fail=1
    continue
  fi

  if [ ! -f "$input" ]; then
    echo "error: input file not found: $input (from $check)" >&2
    fail=1
    continue
  fi

  echo "[sir-text-check] $check -> $input"
  expect_fail=0
  if rg -q "^// EXPECT-FAIL" "$check"; then
    expect_fail=1
  fi
  tmp_stdout=$(mktemp)
  tmp_stderr=$(mktemp)
  if "$ORA_BIN" "$ORA_VERIFY_FLAG" --emit=sir-text "$input" >"$tmp_stdout" 2>"$tmp_stderr"; then
    if [ "$expect_fail" -eq 1 ]; then
      echo "error: ora succeeded for expected-fail check $check" >&2
      rm -f "$tmp_stdout" "$tmp_stderr"
      fail=1
      continue
    fi
  else
    if [ "$expect_fail" -eq 0 ]; then
      echo "error: ora failed for $input (check: $check)" >&2
      sed -n '1,80p' "$tmp_stderr" >&2 || true
      rm -f "$tmp_stdout" "$tmp_stderr"
      fail=1
      continue
    fi
  fi

  if [ "$expect_fail" -eq 1 ]; then
    tmp_combined=$(mktemp)
    cat "$tmp_stdout" "$tmp_stderr" > "$tmp_combined"
    "$FILECHECK_BIN" "$check" < "$tmp_combined" || fail=1
    rm -f "$tmp_combined"
  else
    "$FILECHECK_BIN" --implicit-check-not=builtin.unrealized_conversion_cast "$check" < "$tmp_stdout" || fail=1
  fi
  rm -f "$tmp_stdout" "$tmp_stderr"
done < <(find tests/sir_text -name "*.check" -print0 | sort -z)

exit $fail
