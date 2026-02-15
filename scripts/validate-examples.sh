#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ORA_BIN="$PROJECT_ROOT/zig-out/bin/ora"
EMIT_FLAG="--emit-mlir"
FAIL_FAST=0
QUIET=0
MAX_ERROR_LINES=8

usage() {
  cat <<USAGE
Validate Ora examples by compiling each .ora file with a selected emit mode.

Usage:
  ./scripts/validate-examples.sh [options] [path ...]

Options:
  --compiler <path>            Ora compiler binary (default: ./zig-out/bin/ora)
  --emit <mode>                One of: mlir|sir|sir-text|bytecode|abi|abi-extras
  --fail-fast                  Stop on first failure
  --quiet                      Print only summary and failures
  --max-error-lines <n>        Max stderr lines shown per failure (default: 8)
  -h, --help                   Show this help

Paths:
  Optional file/dir list. If omitted, defaults to ./ora-example.
USAGE
}

emit_mode_to_flag() {
  case "$1" in
    mlir) echo "--emit-mlir" ;;
    sir) echo "--emit-mlir=sir" ;;
    sir-text) echo "--emit-sir-text" ;;
    bytecode) echo "--emit-bytecode" ;;
    abi) echo "--emit-abi" ;;
    abi-extras) echo "--emit-abi-extras" ;;
    *) return 1 ;;
  esac
}

parse_args() {
  INPUTS=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --compiler)
        [[ $# -ge 2 ]] || { echo "error: --compiler requires a value" >&2; exit 1; }
        ORA_BIN="$2"
        shift 2
        ;;
      --emit)
        [[ $# -ge 2 ]] || { echo "error: --emit requires a value" >&2; exit 1; }
        EMIT_FLAG="$(emit_mode_to_flag "$2")" || {
          echo "error: invalid emit mode '$2'" >&2
          exit 1
        }
        shift 2
        ;;
      --fail-fast)
        FAIL_FAST=1
        shift
        ;;
      --quiet)
        QUIET=1
        shift
        ;;
      --max-error-lines)
        [[ $# -ge 2 ]] || { echo "error: --max-error-lines requires a value" >&2; exit 1; }
        MAX_ERROR_LINES="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      -* )
        echo "error: unknown option: $1" >&2
        exit 1
        ;;
      *)
        INPUTS+=("$1")
        shift
        ;;
    esac
  done

  if [[ "${#INPUTS[@]}" -eq 0 ]]; then
    INPUTS=("$PROJECT_ROOT/ora-example")
  fi
}

require_compiler() {
  if [[ ! -x "$ORA_BIN" ]]; then
    echo "error: Ora compiler not found or not executable: $ORA_BIN" >&2
    echo "hint: run 'zig build' first" >&2
    exit 1
  fi
}

collect_files() {
  local tmp_list
  tmp_list="$(mktemp)"

  local path
  for path in "${INPUTS[@]}"; do
    if [[ -f "$path" ]]; then
      if [[ "$path" == *.ora ]]; then
        printf "%s\n" "$path" >>"$tmp_list"
      fi
      continue
    fi

    if [[ -d "$path" ]]; then
      find "$path" -type f -name "*.ora" -print >>"$tmp_list"
      continue
    fi

    echo "warning: path not found, skipping: $path" >&2
  done

  FILES=()
  while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    FILES+=("$file")
  done < <(sort -u "$tmp_list")

  rm -f "$tmp_list"

  if [[ "${#FILES[@]}" -eq 0 ]]; then
    echo "error: no .ora files found" >&2
    exit 1
  fi
}

print_header() {
  if [[ "$QUIET" -eq 0 ]]; then
    echo "Validate Examples"
    echo "-----------------"
    echo "Compiler: $ORA_BIN"
    echo "Emit:     $EMIT_FLAG"
    echo "Files:    ${#FILES[@]}"
  fi
}

validate_one() {
  local file="$1"
  local rel="${file#$PROJECT_ROOT/}"
  local err_file
  err_file="$(mktemp)"

  if "$ORA_BIN" "$EMIT_FLAG" "$file" >/dev/null 2>"$err_file"; then
    if [[ "$QUIET" -eq 0 ]]; then
      printf "[pass] %s\n" "$rel"
    fi
    rm -f "$err_file"
    return 0
  fi

  local code=$?
  printf "[fail] %s (exit %s)\n" "$rel" "$code"
  if [[ -s "$err_file" ]]; then
    sed -n "1,${MAX_ERROR_LINES}p" "$err_file" | sed 's/^/       /'
  fi
  rm -f "$err_file"
  return "$code"
}

run_validation() {
  local total=0
  local passed=0
  local failed=0

  local file
  for file in "${FILES[@]}"; do
    total=$((total + 1))
    if validate_one "$file"; then
      passed=$((passed + 1))
    else
      failed=$((failed + 1))
      if [[ "$FAIL_FAST" -eq 1 ]]; then
        break
      fi
    fi
  done

  echo
  echo "Summary"
  echo "-------"
  echo "Total:  $total"
  echo "Passed: $passed"
  echo "Failed: $failed"

  if [[ "$failed" -gt 0 ]]; then
    exit 1
  fi
}

main() {
  parse_args "$@"
  require_compiler
  collect_files
  print_header
  run_validation
}

main "$@"
