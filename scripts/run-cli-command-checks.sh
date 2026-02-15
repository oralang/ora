#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ORA_BIN="$PROJECT_ROOT/zig-out/bin/ora"
ORA_FILE="$PROJECT_ROOT/ora-example/apps/counter.ora"
OUT_DIR="/tmp/ora-cli-cmd-tests"
QUIET=0

usage() {
  cat <<USAGE
Run Ora CLI command checks and summarize pass/fail results.

Usage:
  ./scripts/run-cli-command-checks.sh [options]

Options:
  --compiler <path>   Ora compiler binary (default: ./zig-out/bin/ora)
  --file <path>       .ora file to use for command tests (default: ora-example/apps/counter.ora)
  --out <dir>         Output directory for logs/artifacts (default: /tmp/ora-cli-cmd-tests)
  --quiet             Print only summary and failures
  -h, --help          Show this help
USAGE
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --compiler)
        [[ $# -ge 2 ]] || { echo "error: --compiler requires a value" >&2; exit 1; }
        ORA_BIN="$2"
        shift 2
        ;;
      --file)
        [[ $# -ge 2 ]] || { echo "error: --file requires a value" >&2; exit 1; }
        ORA_FILE="$2"
        shift 2
        ;;
      --out)
        [[ $# -ge 2 ]] || { echo "error: --out requires a value" >&2; exit 1; }
        OUT_DIR="$2"
        shift 2
        ;;
      --quiet)
        QUIET=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "error: unknown option: $1" >&2
        usage
        exit 1
        ;;
    esac
  done
}

require_inputs() {
  if [[ ! -x "$ORA_BIN" ]]; then
    echo "error: compiler not found or not executable: $ORA_BIN" >&2
    echo "hint: run 'zig build' first" >&2
    exit 1
  fi

  if [[ ! -f "$ORA_FILE" ]]; then
    echo "error: input file not found: $ORA_FILE" >&2
    exit 1
  fi
}

prepare_dirs() {
  rm -rf "$OUT_DIR"
  mkdir -p "$OUT_DIR/logs"
  cp "$ORA_FILE" "$OUT_DIR/fmt_input.ora"
}

declare -a TEST_NAMES=()
declare -a TEST_EXPECT=()
declare -a TEST_CMDS=()

add_test() {
  local name="$1"
  local expected="$2"  # 0 or nonzero
  local cmd="$3"
  TEST_NAMES+=("$name")
  TEST_EXPECT+=("$expected")
  TEST_CMDS+=("$cmd")
}

build_test_matrix() {
  add_test "help" 0 '"$ORA_BIN"'
  add_test "version" 0 '"$ORA_BIN" --version'
  add_test "default build" 0 '"$ORA_BIN" "$ORA_FILE"'
  add_test "build explicit output" 0 '"$ORA_BIN" build -o "$OUT_DIR/artifacts" "$ORA_FILE"'
  add_test "emit default" 0 '"$ORA_BIN" emit "$ORA_FILE"'

  add_test "emit tokens" 0 '"$ORA_BIN" emit --emit-tokens "$ORA_FILE"'
  add_test "emit ast tree" 0 '"$ORA_BIN" emit --emit-ast "$ORA_FILE"'
  add_test "emit ast json" 0 '"$ORA_BIN" emit --emit-ast=json "$ORA_FILE"'
  add_test "emit typed ast tree" 0 '"$ORA_BIN" emit --emit-typed-ast "$ORA_FILE"'
  add_test "emit typed ast json" 0 '"$ORA_BIN" emit --emit-typed-ast=json "$ORA_FILE"'

  add_test "emit mlir ora" 0 '"$ORA_BIN" emit --emit-mlir=ora "$ORA_FILE"'
  add_test "emit mlir sir" 0 '"$ORA_BIN" emit --emit-mlir=sir "$ORA_FILE"'
  add_test "emit mlir both" 0 '"$ORA_BIN" emit --emit-mlir=both "$ORA_FILE"'
  add_test "emit sir text" 0 '"$ORA_BIN" emit --emit-sir-text "$ORA_FILE"'
  add_test "emit bytecode file" 0 '"$ORA_BIN" emit --emit-bytecode -o "$OUT_DIR/counter.hex" "$ORA_FILE"'

  add_test "emit cfg default(ora)" 0 '"$ORA_BIN" emit --emit-cfg "$ORA_FILE" > "$OUT_DIR/cfg_default.dot"'
  add_test "emit cfg ora" 0 '"$ORA_BIN" emit --emit-cfg=ora "$ORA_FILE" > "$OUT_DIR/cfg_ora.dot"'
  add_test "emit cfg sir" 0 '"$ORA_BIN" emit --emit-cfg=sir "$ORA_FILE" > "$OUT_DIR/cfg_sir.dot"'

  add_test "emit abi" 0 '"$ORA_BIN" emit --emit-abi -o "$OUT_DIR/abi" "$ORA_FILE"'
  add_test "emit abi solidity" 0 '"$ORA_BIN" emit --emit-abi-solidity -o "$OUT_DIR/abi" "$ORA_FILE"'
  add_test "emit abi extras" 0 '"$ORA_BIN" emit --emit-abi-extras -o "$OUT_DIR/abi" "$ORA_FILE"'

  add_test "verify basic" 0 '"$ORA_BIN" emit --verify=basic --emit-mlir=ora "$ORA_FILE"'
  add_test "verify full" 0 '"$ORA_BIN" emit --verify=full --emit-mlir=ora "$ORA_FILE"'
  add_test "no verify" 0 '"$ORA_BIN" emit --no-verify --emit-mlir=ora "$ORA_FILE"'
  add_test "verify toggles" 0 '"$ORA_BIN" emit --verify-calls --verify-state --verify-stats --emit-mlir=ora "$ORA_FILE"'
  add_test "disable verify toggles" 0 '"$ORA_BIN" emit --no-verify-calls --no-verify-state --emit-mlir=ora "$ORA_FILE"'
  add_test "emit smt report" 0 '"$ORA_BIN" emit --emit-smt-report -o "$OUT_DIR/verify" "$ORA_FILE"'

  add_test "opt O0" 0 '"$ORA_BIN" emit -O0 --emit-mlir=ora "$ORA_FILE"'
  add_test "opt O1" 0 '"$ORA_BIN" emit -O1 --emit-mlir=ora "$ORA_FILE"'
  add_test "opt O2" 0 '"$ORA_BIN" emit -O2 --emit-mlir=ora "$ORA_FILE"'
  add_test "no canonicalize" 0 '"$ORA_BIN" emit --no-canonicalize --emit-mlir=ora "$ORA_FILE"'
  add_test "cpp lowering stub" 0 '"$ORA_BIN" emit --cpp-lowering-stub --emit-mlir=ora "$ORA_FILE"'

  add_test "mlir print before" 0 '"$ORA_BIN" emit --emit-mlir=ora --mlir-print-ir=before "$ORA_FILE"'
  add_test "mlir print after" 0 '"$ORA_BIN" emit --emit-mlir=ora --mlir-print-ir=after "$ORA_FILE"'
  add_test "mlir print before-after filtered" 0 '"$ORA_BIN" emit --emit-mlir=ora --mlir-print-ir=before-after --mlir-print-ir-pass canonicalize "$ORA_FILE"'
  add_test "mlir pass statistics" 0 '"$ORA_BIN" emit --emit-mlir=sir --mlir-pass-statistics "$ORA_FILE"'

  add_test "custom pass pipeline" 0 '"$ORA_BIN" emit --emit-mlir=ora --mlir-pass-pipeline "builtin.module(canonicalize,cse)" --mlir-verify-each-pass --mlir-pass-timing --mlir-crash-reproducer "$OUT_DIR/reproducer.mlir" --mlir-print-op-on-diagnostic "$ORA_FILE"'

  add_test "analyze state" 0 '"$ORA_BIN" --analyze-state "$ORA_FILE"'
  add_test "metrics report" 0 '"$ORA_BIN" emit --metrics --emit-mlir=ora "$ORA_FILE"'

  add_test "fmt write" 0 '"$ORA_BIN" fmt "$OUT_DIR/fmt_input.ora"'
  add_test "fmt check" 0 '"$ORA_BIN" fmt --check "$OUT_DIR/fmt_input.ora"'
  add_test "fmt diff" 0 '"$ORA_BIN" fmt --diff "$OUT_DIR/fmt_input.ora"'
  add_test "fmt stdout" 0 '"$ORA_BIN" fmt --stdout "$OUT_DIR/fmt_input.ora"'
  add_test "fmt width" 0 '"$ORA_BIN" fmt --width 120 "$OUT_DIR/fmt_input.ora"'

  add_test "guard: timing requires pipeline" nonzero '"$ORA_BIN" emit --mlir-pass-timing "$ORA_FILE"'
  add_test "guard: print-ir-pass requires print-ir" nonzero '"$ORA_BIN" emit --mlir-print-ir-pass canonicalize "$ORA_FILE"'
}

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' /:()' '_____' | tr -cd 'a-z0-9_-'
}

run_tests() {
  local total=${#TEST_NAMES[@]}
  local passed=0
  local failed=0
  export ORA_BIN ORA_FILE OUT_DIR

  if [[ "$QUIET" -eq 0 ]]; then
    echo "CLI command checks"
    echo "------------------"
    echo "Compiler: $ORA_BIN"
    echo "Input:    $ORA_FILE"
    echo "Output:   $OUT_DIR"
    echo "Tests:    $total"
    echo
  fi

  local i
  for ((i = 0; i < total; i++)); do
    local name="${TEST_NAMES[$i]}"
    local expected="${TEST_EXPECT[$i]}"
    local cmd="${TEST_CMDS[$i]}"

    local idx
    idx=$(printf "%03d" "$((i + 1))")
    local slug
    slug=$(slugify "$name")
    local out_file="$OUT_DIR/logs/${idx}_${slug}.stdout.log"
    local err_file="$OUT_DIR/logs/${idx}_${slug}.stderr.log"
    local cmd_file="$OUT_DIR/logs/${idx}_${slug}.cmd.txt"

    printf "%s\n" "$cmd" >"$cmd_file"

    bash -lc "$cmd" >"$out_file" 2>"$err_file"
    local status=$?

    local ok=0
    if [[ "$expected" == "0" ]]; then
      if [[ "$status" -eq 0 ]]; then
        ok=1
      fi
    else
      if [[ "$status" -ne 0 ]]; then
        ok=1
      fi
    fi

    if [[ "$ok" -eq 1 ]]; then
      passed=$((passed + 1))
      if [[ "$QUIET" -eq 0 ]]; then
        printf "[pass] (%s/%s) %s\n" "$((i + 1))" "$total" "$name"
      fi
    else
      failed=$((failed + 1))
      printf "[fail] (%s/%s) %s (exit=%s expected=%s)\n" "$((i + 1))" "$total" "$name" "$status" "$expected"
      if [[ -s "$err_file" ]]; then
        sed -n '1,20p' "$err_file" | sed 's/^/       /'
      fi
      echo "       cmd: $(<"$cmd_file")"
      echo "       logs: $out_file | $err_file"
    fi
  done

  echo
  echo "Summary"
  echo "-------"
  echo "Total:  $total"
  echo "Passed: $passed"
  echo "Failed: $failed"
  echo "Logs:   $OUT_DIR/logs"

  if [[ "$failed" -gt 0 ]]; then
    exit 1
  fi
}

main() {
  parse_args "$@"
  require_inputs
  prepare_dirs
  build_test_matrix
  run_tests
}

main "$@"
