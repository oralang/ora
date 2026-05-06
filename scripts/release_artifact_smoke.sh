#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${1:-$ROOT_DIR/.release-smoke-artifacts}"

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

contracts=(
  "ora-example/debugger/comptime_debug_probe.ora"
  "ora-example/debugger/partial_eval_probe.ora"
  "ora-example/debugger/nested_unroll_probe.ora"
  "ora-example/debugger/match_probe.ora"
  "ora-example/debugger/match_discard_probe.ora"
  "ora-example/debugger/constructor_value.ora"
  "ora-example/corpus/control-flow/match/error_union_match.ora"
  "ora-example/corpus/control-flow/match/result_roundtrip.ora"
  "ora-example/corpus/control-flow/match/result_named_error_match.ora"
  "ora-example/corpus/control-flow/match/result_named_error_payload_match.ora"
  "ora-example/corpus/control-flow/match/result_discard_patterns.ora"
  "ora-example/corpus/control-flow/match/result_err_discard.ora"
  "ora-example/corpus/control-flow/match/result_stateful_flow.ora"
  "ora-example/corpus/control-flow/match/result_bytes_flow.ora"
  "ora-example/corpus/control-flow/match/result_struct_flow.ora"
  "ora-example/corpus/control-flow/match/result_dynamic_bytes_input.ora"
  "ora-example/corpus/control-flow/match/result_dynamic_slice_input.ora"
  "ora-example/corpus/control-flow/match/result_helpers.ora"
  "ora-example/corpus/control-flow/match/result_payload_input.ora"
  "ora-example/corpus/control-flow/match/result_struct_input.ora"
  "ora-example/corpus/types/error-union/result_constructors.ora"
  "ora-example/corpus/types/tuple/division_builtins.ora"
  "ora-example/corpus/types/dynamic/bytes_string_len_index.ora"
  "ora-example/corpus/types/dynamic/std_bytes_helpers.ora"
  "ora-example/corpus/patterns/multi_asset.ora"
  "ora-example/vault/05_locks.ora"
  "ora-example/refinements/guards_showcase.ora"
  "ora-example/errors/try_catch.ora"
)

check_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "artifact check failed: missing or empty $path" >&2
    exit 1
  fi
}

for contract in "${contracts[@]}"; do
  stem="$(basename "${contract%.ora}")"
  out_dir="$OUT_ROOT/$stem"
  mkdir -p "$out_dir"

  echo "==> $contract"
  emit_args=(--emit-bytecode --emit-sir-text --debug-info -o "$out_dir")
  case "$contract" in
    ora-example/corpus/control-flow/match/*)
      emit_args=(--no-verify "${emit_args[@]}")
      ;;
    ora-example/corpus/types/dynamic/std_bytes_helpers.ora)
      emit_args=(--no-verify "${emit_args[@]}")
      ;;
  esac

  ./zig-out/bin/ora emit "${emit_args[@]}" "$contract" >/tmp/"$stem".release-smoke.log 2>&1 || {
    cat /tmp/"$stem".release-smoke.log >&2
    exit 1
  }

  hex_path="$out_dir/$stem.hex"
  sir_path="$out_dir/$stem.sir"
  sourcemap_path="$out_dir/$stem.sourcemap.json"
  debug_path="$out_dir/$stem.debug.json"

  check_file "$hex_path"
  check_file "$sir_path"
  check_file "$sourcemap_path"
  check_file "$debug_path"

  rg -q '^0x[0-9a-fA-F]+$' "$hex_path" || {
    echo "artifact check failed: bytecode is not hex in $hex_path" >&2
    exit 1
  }

  rg -q '"entries":\[' "$sourcemap_path" || {
    echo "artifact check failed: sourcemap missing entries in $sourcemap_path" >&2
    exit 1
  }
  rg -q '"pc":' "$sourcemap_path" || {
    echo "artifact check failed: sourcemap missing pc fields in $sourcemap_path" >&2
    exit 1
  }
  rg -q '"line":' "$sourcemap_path" || {
    echo "artifact check failed: sourcemap missing line fields in $sourcemap_path" >&2
    exit 1
  }

  rg -q '"ops":\[' "$debug_path" || {
    echo "artifact check failed: debug info missing ops array in $debug_path" >&2
    exit 1
  }
  rg -q '"statement_id":' "$debug_path" || {
    echo "artifact check failed: debug info missing statement provenance in $debug_path" >&2
    exit 1
  }

  rg -q '^fn ' "$sir_path" || {
    echo "artifact check failed: SIR text missing function bodies in $sir_path" >&2
    exit 1
  }
done

echo "artifact smoke passed: $OUT_ROOT"
