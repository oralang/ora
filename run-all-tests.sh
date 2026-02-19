#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

MODE="fast"       # fast | full
RUN_EXAMPLES=1
RUN_MLIR=0

usage() {
  cat <<USAGE
Run Ora test suites.

Usage:
  ./run-all-tests.sh [options]

Options:
  --fast            Run fast suite (default)
  --full            Run full suite
  --mlir            Also run MLIR-specific tests (zig build test-mlir)
  --no-examples     Skip example validation script
  -h, --help        Show this help
USAGE
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --fast)
        MODE="fast"
        shift
        ;;
      --full)
        MODE="full"
        shift
        ;;
      --mlir)
        RUN_MLIR=1
        shift
        ;;
      --no-examples)
        RUN_EXAMPLES=0
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "error: unknown argument: $1" >&2
        exit 1
        ;;
    esac
  done
}

has_mlir_artifacts() {
  local lib_dir="$ROOT_DIR/vendor/mlir/lib"
  [[ -d "$lib_dir" ]] || return 1
  compgen -G "$lib_dir/libMLIR-C.*" >/dev/null || return 1
  compgen -G "$lib_dir/libMLIROraDialectC.*" >/dev/null || return 1
  compgen -G "$lib_dir/libMLIRSIRDialect.*" >/dev/null || return 1
  return 0
}

declare -a STEP_NAMES=()
declare -a STEP_CODES=()

run_step() {
  local name="$1"
  shift

  printf "\n[step] %s\n" "$name"
  if "$@"; then
    STEP_NAMES+=("$name")
    STEP_CODES+=(0)
    printf "[ok]   %s\n" "$name"
  else
    local code=$?
    STEP_NAMES+=("$name")
    STEP_CODES+=("$code")
    printf "[fail] %s (exit %s)\n" "$name" "$code"
  fi
}

skip_step() {
  local name="$1"
  local reason="$2"
  STEP_NAMES+=("$name")
  STEP_CODES+=(125)
  printf "\n[skip] %s (%s)\n" "$name" "$reason"
}

print_summary_and_exit() {
  local failures=0
  local i code status

  printf "\nTest Summary\n"
  printf "%s\n" "------------"

  for ((i=0; i<${#STEP_NAMES[@]}; i++)); do
    code="${STEP_CODES[$i]}"
    if [[ "$code" -eq 0 ]]; then
      status="PASS"
    elif [[ "$code" -eq 125 ]]; then
      status="SKIP"
    else
      status="FAIL"
      failures=$((failures + 1))
    fi
    printf "%-32s %s\n" "${STEP_NAMES[$i]}" "$status"
  done

  if [[ "$failures" -gt 0 ]]; then
    printf "\nResult: FAIL (%s failing step(s))\n" "$failures"
    exit 1
  fi

  printf "\nResult: PASS\n"
}

main() {
  parse_args "$@"
  cd "$ROOT_DIR"

  if [[ "$MODE" == "fast" ]] && has_mlir_artifacts; then
    run_step "Build (fast)" zig build -Dskip-mlir=true
    run_step "Unit tests (fast)" zig build test -Dskip-mlir=true
  else
    if [[ "$MODE" == "fast" ]]; then
      echo "[info] MLIR artifacts not found; falling back to full build/test"
    fi
    run_step "Build" zig build
    run_step "Unit tests" zig build test
  fi

  if [[ "$RUN_MLIR" -eq 1 ]]; then
    if has_mlir_artifacts; then
      run_step "MLIR tests" zig build test-mlir
    else
      skip_step "MLIR tests" "missing MLIR artifacts"
    fi
  fi

  if [[ "$RUN_EXAMPLES" -eq 1 ]]; then
    run_step "Example validation" "$ROOT_DIR/scripts/validate-examples.sh"
    run_step "Storage/TStore separation checks" "$ROOT_DIR/scripts/check-storage-tstore-separation.sh"
  fi

  print_summary_and_exit
}

main "$@"
