#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ORA_BIN="${ORA_BIN:-$PROJECT_ROOT/zig-out/bin/ora}"
OUT_ROOT="${OUT_ROOT:-${TMPDIR:-/tmp}/ora-smt-modifies-corpus}"
MAX_QUERY_MS="${ORA_SMT_MODIFIES_MAX_QUERY_MS:-5000}"

SEMA_PASS_CASES=(
  "ora-example/corpus/modifies/pass_supported_paths.ora"
  "ora-example/corpus/modifies/pass_empty_no_writes.ora"
)

SEMA_FAIL_CASES=(
  "ora-example/corpus/modifies/fail_empty_mixed.ora|cannot be combined with non-empty"
  "ora-example/corpus/modifies/fail_empty_with_write.ora|storage write to 'total' is not covered"
  "ora-example/corpus/modifies/fail_external_storage_path.ora|only supports current-contract storage paths"
  "ora-example/corpus/modifies/fail_mixed_indexed_field_path.ora|does not support mixed indexed-field storage paths"
  "ora-example/corpus/modifies/fail_unsupported_map_key.ora|map keys must be literals"
  "ora-example/corpus/modifies/fail_write_outside_declared.ora|storage write to 'balances[other]' is not covered|storage write to 'config.admin' is not covered"
)

PASS_CASES=(
  "ora-example/smt/modifies/pass_internal_root_frame.ora"
  "ora-example/smt/modifies/pass_internal_map_key_frame.ora"
  "ora-example/smt/modifies/pass_internal_nested_map_frame.ora"
  "ora-example/smt/modifies/pass_internal_struct_field_frame.ora"
  "ora-example/smt/modifies/pass_staticcall_modifies_empty_frame.ora"
  "ora-example/smt/modifies/pass_locked_call_root_frame.ora"
  "ora-example/smt/modifies/pass_locked_call_map_key_frame.ora"
)

OPAQUE_PASS_CASES=(
  "ora-example/smt/modifies/pass_internal_map_key_frame.ora"
  "ora-example/smt/modifies/pass_internal_nested_map_frame.ora"
  "ora-example/smt/modifies/pass_internal_struct_field_frame.ora"
)

REAL_CONTRACT_PASS_CASES=(
  "ora-example/apps/erc20_verified.ora"
)

FAIL_CASES=(
  "ora-example/smt/modifies/fail_internal_map_key_alias.ora|failed to prove ensures"
  "ora-example/smt/modifies/fail_modifies_empty_unresolved_call.ora|SMT encoding degraded"
  "ora-example/smt/modifies/fail_locked_call_unlocked_root.ora|failed to prove ensures"
  "ora-example/smt/modifies/fail_locked_call_unlocked_map_key.ora|failed to prove ensures"
)

if [[ ! -x "$ORA_BIN" ]]; then
  echo "error: Ora compiler not found or not executable: $ORA_BIN" >&2
  echo "hint: run 'zig build' first" >&2
  exit 1
fi

if [[ ! "$MAX_QUERY_MS" =~ ^[0-9]+$ ]]; then
  echo "error: ORA_SMT_MODIFIES_MAX_QUERY_MS must be an integer millisecond budget, got '$MAX_QUERY_MS'" >&2
  exit 1
fi

assert_query_budget() {
  local out_dir="$1"
  local source="$2"
  local report_file
  local max_elapsed

  report_file="$(find "$out_dir/verify" -name '*.smt.report.json' -print -quit 2>/dev/null || true)"
  if [[ -z "$report_file" ]]; then
    echo "error: SMT report JSON not found for $source under $out_dir/verify" >&2
    return 1
  fi

  max_elapsed="$(
    awk '
      match($0, /"elapsed_ms"[[:space:]]*:[[:space:]]*[0-9]+/) {
        value = substr($0, RSTART, RLENGTH)
        sub(/.*:[[:space:]]*/, "", value)
        if (value + 0 > max) max = value + 0
      }
      END { print max + 0 }
    ' "$report_file"
  )"
  max_elapsed="${max_elapsed:-0}"

  if (( max_elapsed > MAX_QUERY_MS )); then
    echo "error: SMT query budget exceeded for $source: max elapsed ${max_elapsed}ms > ${MAX_QUERY_MS}ms" >&2
    echo "report: $report_file" >&2
    return 1
  fi
}

run_pass() {
  local source="$1"
  local label="$2"
  local out_dir="$OUT_ROOT/$label/$(basename "$source" .ora)"

  echo "[pass] $label $source"
  "$ORA_BIN" build --verify=full -o "$out_dir" "$PROJECT_ROOT/$source" >/dev/null 2>&1
  assert_query_budget "$out_dir" "$source"
}

run_opaque_pass() {
  local source="$1"
  local out_dir="$OUT_ROOT/opaque/$(basename "$source" .ora)"
  local log_file
  log_file="$(mktemp)"

  echo "[pass] opaque-summary $source"
  ORA_VERIFY_MAX_SUMMARY_INLINE_DEPTH=0 "$ORA_BIN" build --verify=full -o "$out_dir" "$PROJECT_ROOT/$source" >"$log_file" 2>&1
  assert_query_budget "$out_dir" "$source"
  if grep -q "precision_note" "$log_file"; then
    echo "error: opaque-summary run for $source emitted precision notes" >&2
    sed -n '1,120p' "$log_file" >&2
    rm -f "$log_file"
    return 1
  fi
  rm -f "$log_file"
}

run_fail() {
  local entry="$1"
  local source="${entry%%|*}"
  local expected="${entry#*|}"
  local out_dir="$OUT_ROOT/fail/$(basename "$source" .ora)"
  local err_file
  err_file="$(mktemp)"

  echo "[fail] $source"
  if "$ORA_BIN" build --verify=full -o "$out_dir" "$PROJECT_ROOT/$source" >"$err_file" 2>&1; then
    echo "error: expected verification failure for $source" >&2
    rm -f "$err_file"
    return 1
  fi

  IFS='|' read -r -a expected_parts <<< "$expected"
  for expected_part in "${expected_parts[@]}"; do
    if ! grep -Fq "$expected_part" "$err_file"; then
      echo "error: expected failure output for $source to contain: $expected_part" >&2
      sed -n '1,80p' "$err_file" >&2
      rm -f "$err_file"
      return 1
    fi
  done

  rm -f "$err_file"
}

main() {
  if [[ -z "$OUT_ROOT" || "$OUT_ROOT" == "/" ]]; then
    echo "error: refusing to clean unsafe OUT_ROOT: '$OUT_ROOT'" >&2
    exit 1
  fi
  rm -rf "$OUT_ROOT"
  mkdir -p "$OUT_ROOT"

  for source in "${SEMA_PASS_CASES[@]}"; do
    run_pass "$source" "sema"
  done

  for entry in "${SEMA_FAIL_CASES[@]}"; do
    run_fail "$entry"
  done

  for source in "${PASS_CASES[@]}"; do
    run_pass "$source" "default"
  done

  for source in "${OPAQUE_PASS_CASES[@]}"; do
    run_opaque_pass "$source"
  done

  for source in "${REAL_CONTRACT_PASS_CASES[@]}"; do
    run_pass "$source" "apps"
  done

  for entry in "${FAIL_CASES[@]}"; do
    run_fail "$entry"
  done

  echo "SMT modifies corpus: ok"
}

main "$@"
