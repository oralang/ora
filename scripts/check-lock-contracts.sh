#!/usr/bin/env bash
set -euo pipefail

ORA_BIN="${ORA_BIN:-./zig-out/bin/ora}"
EMIT_FLAG="${EMIT_FLAG:---emit-mlir}"

if [[ ! -x "$ORA_BIN" ]]; then
  echo "error: ora binary not found at $ORA_BIN" >&2
  exit 1
fi

positive=(
  "ora-example/locks/lock_runtime_map_guard.ora"
  "ora-example/locks/lock_runtime_array_guard.ora"
  "ora-example/locks/lock_runtime_scalar_guard.ora"
  "ora-example/locks/lock_runtime_independent_roots.ora"
)

negative=(
  "ora-example/locks/negative/fail_lock_non_tstore.ora"
  "ora-example/locks/negative/fail_lock_direct_write_same_slot.ora"
  "ora-example/locks/negative/fail_lock_direct_write_same_slot_expr_return.ora"
  "ora-example/locks/negative/fail_lock_in_expression.ora"
  "ora-example/locks/negative/fail_lock_nested_index_unsupported.ora"
  "ora-example/locks/negative/fail_lock_storage_mapping.ora"
  "ora-example/locks/negative/fail_lock_storage_struct_field_unsupported.ora"
  "ora-example/locks/negative/fail_lock_storage_nested_mapping.ora"
  "ora-example/locks/negative/fail_lock_storage_array.ora"
  "ora-example/locks/negative/fail_unlock_in_expression.ora"
)

fails=0

echo "Lock Contracts: positive compile checks"
for file in "${positive[@]}"; do
  if "$ORA_BIN" "$EMIT_FLAG" "$file" >/tmp/ora_lock_check.out 2>/tmp/ora_lock_check.err; then
    echo "  [pass] $file"
  else
    echo "  [fail] $file"
    sed -n '1,40p' /tmp/ora_lock_check.err | sed 's/^/         /'
    fails=1
  fi
done

echo
echo "Lock Contracts: negative compile checks"
for file in "${negative[@]}"; do
  if "$ORA_BIN" "$EMIT_FLAG" "$file" >/tmp/ora_lock_check.out 2>/tmp/ora_lock_check.err; then
    echo "  [fail] $file (expected compile error)"
    fails=1
  else
    echo "  [pass] $file"
  fi
done

rm -f /tmp/ora_lock_check.out /tmp/ora_lock_check.err

if [[ "$fails" -ne 0 ]]; then
  exit 1
fi

echo
echo "All lock-contract checks passed."
