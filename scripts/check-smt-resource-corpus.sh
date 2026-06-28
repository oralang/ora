#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ORA_BIN="${ORA_BIN:-$PROJECT_ROOT/zig-out/bin/ora}"
OUT_ROOT="${OUT_ROOT:-${TMPDIR:-/tmp}/ora-smt-resource-corpus}"
MAX_QUERY_MS="${ORA_SMT_RESOURCE_MAX_QUERY_MS:-5000}"
PERF_REPORT="${ORA_SMT_RESOURCE_PERF_REPORT:-$OUT_ROOT/perf.tsv}"

PASS_CASES=(
  "ora-example/smt/resources/pass_move_pair_sum.ora"
  "ora-example/smt/resources/pass_move_with_sufficient_balance.ora"
  "ora-example/smt/resources/pass_move_dynamic_alias.ora"
  "ora-example/smt/resources/pass_create_destroy_supply_delta.ora"
  "ora-example/smt/resources/pass_signed_amount_guard.ora"
  "ora-example/smt/resources/pass_signed_negative_balance.ora"
  "ora-example/smt/resources/pass_initial_create_in_init.ora"
)

FAIL_CASES=(
  "ora-example/smt/resources/fail_move_overdraw_without_precondition.ora|failed to prove contract invariant"
  "ora-example/smt/resources/fail_destination_overflow_without_bound.ora|failed to prove contract invariant"
  "ora-example/smt/resources/fail_negative_signed_amount.ora|failed to prove contract invariant"
  "ora-example/smt/resources/fail_global_supply_conservation_unstated.ora|failed to prove ensures"
  "ora-example/smt/resources/fail_write_outside_modifies.ora|not covered"
  "ora-example/smt/resources/fail_external_call_unlocked_resource_write.ora|SMT encoding degraded|unresolved external callee has no sound state summary"
  "ora-example/smt/resources/fail_signed_carrier_overflow.ora|failed to prove contract invariant"
)

if [[ ! -x "$ORA_BIN" ]]; then
  echo "error: Ora compiler not found or not executable: $ORA_BIN" >&2
  echo "hint: run 'zig build' first" >&2
  exit 1
fi

if [[ ! "$MAX_QUERY_MS" =~ ^[0-9]+$ ]]; then
  echo "error: ORA_SMT_RESOURCE_MAX_QUERY_MS must be an integer millisecond budget, got '$MAX_QUERY_MS'" >&2
  exit 1
fi

assert_query_budget() {
  local out_dir="$1"
  local source="$2"
  local report_file
  local stats
  local query_count
  local total_elapsed
  local max_elapsed

  report_file="$(find "$out_dir/verify" -name '*.smt.report.json' -print -quit 2>/dev/null || true)"
  if [[ -z "$report_file" ]]; then
    echo "error: SMT report JSON not found for $source under $out_dir/verify" >&2
    return 1
  fi

  stats="$(
    awk '
      {
        line = $0
        while (match(line, /"elapsed_ms"[[:space:]]*:[[:space:]]*[0-9]+/)) {
          value = substr(line, RSTART, RLENGTH)
          sub(/.*:[[:space:]]*/, "", value)
          count += 1
          total += value + 0
          if (value + 0 > max) max = value + 0
          line = substr(line, RSTART + RLENGTH)
        }
      }
      END { print count + 0, total + 0, max + 0 }
    ' "$report_file"
  )"
  read -r query_count total_elapsed max_elapsed <<< "$stats"
  query_count="${query_count:-0}"
  total_elapsed="${total_elapsed:-0}"
  max_elapsed="${max_elapsed:-0}"

  if (( max_elapsed > MAX_QUERY_MS )); then
    echo "error: SMT query budget exceeded for $source: max elapsed ${max_elapsed}ms > ${MAX_QUERY_MS}ms" >&2
    echo "report: $report_file" >&2
    return 1
  fi

  printf '[perf] %s queries=%s total_ms=%s max_ms=%s budget_ms=%s\n' \
    "$source" "$query_count" "$total_elapsed" "$max_elapsed" "$MAX_QUERY_MS"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$source" "$query_count" "$total_elapsed" "$max_elapsed" "$MAX_QUERY_MS" >>"$PERF_REPORT"
}

resource_effect_counts() {
  local source="$1"

  case "$source" in
    ora-example/smt/resources/pass_move_pair_sum.ora) echo "1 0 0" ;;
    ora-example/smt/resources/pass_move_with_sufficient_balance.ora) echo "1 0 0" ;;
    ora-example/smt/resources/pass_move_dynamic_alias.ora) echo "1 0 0" ;;
    ora-example/smt/resources/pass_create_destroy_supply_delta.ora) echo "0 1 1" ;;
    ora-example/smt/resources/pass_signed_amount_guard.ora) echo "0 1 0" ;;
    ora-example/smt/resources/pass_signed_negative_balance.ora) echo "1 0 0" ;;
    ora-example/smt/resources/pass_initial_create_in_init.ora) echo "0 1 0" ;;
    *)
      echo "error: missing resource effect expectation for $source" >&2
      return 1
      ;;
  esac
}

assert_resource_report_events() {
  local out_dir="$1"
  local source="$2"
  local report_json
  local report_md
  local expected_moves
  local expected_creates
  local expected_destroys
  local expected_counts

  report_json="$(find "$out_dir/verify" -name '*.smt.report.json' -print -quit 2>/dev/null || true)"
  report_md="$(find "$out_dir/verify" -name '*.smt.report.md' -print -quit 2>/dev/null || true)"
  if [[ -z "$report_json" || -z "$report_md" ]]; then
    echo "error: SMT report artifacts not found for $source under $out_dir/verify" >&2
    return 1
  fi

  expected_counts="$(resource_effect_counts "$source")" || return 1
  read -r expected_moves expected_creates expected_destroys <<<"$expected_counts"

  if ! grep -Eq -- "\"encoding_degraded\"[[:space:]]*:[[:space:]]*false" "$report_json" ||
     ! grep -Eq -- "\"soundness_losses\"[[:space:]]*:[[:space:]]*\\[[[:space:]]*\\]" "$report_json" ||
     ! grep -Eq -- "\"success\"[[:space:]]*:[[:space:]]*true" "$report_json"; then
    echo "error: SMT pass report is degraded, has soundness losses, or is not verified for $source" >&2
    echo "report: $report_json" >&2
    return 1
  fi

  if ! grep -Eq -- "\"conserved_moves\"[[:space:]]*:[[:space:]]*$expected_moves" "$report_json" ||
     ! grep -Eq -- "\"explicit_creates\"[[:space:]]*:[[:space:]]*$expected_creates" "$report_json" ||
     ! grep -Eq -- "\"explicit_destroys\"[[:space:]]*:[[:space:]]*$expected_destroys" "$report_json"; then
    echo "error: SMT report JSON resource effect counts mismatch for $source" >&2
    echo "report: $report_json" >&2
    return 1
  fi

  if ! grep -Fq -- "## Resource Effects" "$report_md" ||
     ! grep -Fq -- "- Conserved moves: \`$expected_moves\`" "$report_md" ||
     ! grep -Fq -- "- Explicit creates: \`$expected_creates\`" "$report_md" ||
     ! grep -Fq -- "- Explicit destroys: \`$expected_destroys\`" "$report_md"; then
    echo "error: SMT report markdown resource effect counts mismatch for $source" >&2
    echo "report: $report_md" >&2
    return 1
  fi

  if (( expected_moves > 0 )) && ! grep -Eq -- "\"kind\"[[:space:]]*:[[:space:]]*\"conserved_move\"" "$report_json"; then
    echo "error: SMT report JSON missing conserved_move event for $source" >&2
    echo "report: $report_json" >&2
    return 1
  fi
  if (( expected_creates > 0 )) && ! grep -Eq -- "\"kind\"[[:space:]]*:[[:space:]]*\"explicit_create\"" "$report_json"; then
    echo "error: SMT report JSON missing explicit_create event for $source" >&2
    echo "report: $report_json" >&2
    return 1
  fi
  if (( expected_destroys > 0 )) && ! grep -Eq -- "\"kind\"[[:space:]]*:[[:space:]]*\"explicit_destroy\"" "$report_json"; then
    echo "error: SMT report JSON missing explicit_destroy event for $source" >&2
    echo "report: $report_json" >&2
    return 1
  fi
}

run_pass() {
  local source="$1"
  local out_dir="$OUT_ROOT/pass/$(basename "$source" .ora)"

  echo "[pass] $source"
  "$ORA_BIN" build --verify=full -o "$out_dir" "$PROJECT_ROOT/$source" >/dev/null 2>&1
  assert_query_budget "$out_dir" "$source"
  assert_resource_report_events "$out_dir" "$source"
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
      sed -n '1,120p' "$err_file" >&2
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
  mkdir -p "$(dirname "$PERF_REPORT")"
  printf 'source\tqueries\ttotal_ms\tmax_ms\tbudget_ms\n' >"$PERF_REPORT"

  for source in "${PASS_CASES[@]}"; do
    run_pass "$source"
  done

  for entry in "${FAIL_CASES[@]}"; do
    run_fail "$entry"
  done

  echo "perf report: $PERF_REPORT"
  echo "SMT resource corpus: ok"
}

main "$@"
