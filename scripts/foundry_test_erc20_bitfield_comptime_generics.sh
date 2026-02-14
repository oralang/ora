#!/usr/bin/env bash
set -euo pipefail

# End-to-end Anvil test harness for:
# ora-example/apps/erc20_bitfield_comptime_generics.ora
#
# Usage:
#   ./scripts/foundry_test_erc20_bitfield_comptime_generics.sh [optional-bytecode-file]
#
# Optional env:
#   RPC_URL=http://127.0.0.1:8545
#   ORA_BIN=./zig-out/bin/ora
#   ORA_SOURCE=ora-example/apps/erc20_bitfield_comptime_generics.ora
#   WORK_DIR=/tmp/ora_erc20_anvil
#   INITIAL_SUPPLY=1000

RPC_URL="${RPC_URL:-http://127.0.0.1:8545}"
ORA_BIN="${ORA_BIN:-./zig-out/bin/ora}"
ORA_SOURCE="${ORA_SOURCE:-ora-example/apps/erc20_bitfield_comptime_generics.ora}"
WORK_DIR="${WORK_DIR:-/tmp/ora_erc20_anvil}"
INITIAL_SUPPLY="${INITIAL_SUPPLY:-1000}"

DEPLOYER="${DEPLOYER:-0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266}"
ALICE="${ALICE:-$DEPLOYER}"
BOB="${BOB:-0x70997970C51812dc3A010C7d01b50e0d17dc79C8}"
CAROL="${CAROL:-0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC}"
ZERO_ADDRESS="0x0000000000000000000000000000000000000000"

export FOUNDRY_DISABLE_NIGHTLY_WARNING=1
export NO_PROXY="*"
export no_proxy="*"
export HTTP_PROXY=""
export HTTPS_PROXY=""
export ALL_PROXY=""

BYTECODE_FILE="${1:-}"
CONTRACT_ADDR=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1"
    exit 1
  fi
}

fail() {
  echo "❌ $*"
  exit 1
}

ok() {
  echo "✅ $*"
}

trim() {
  echo "$1" | tr -d '[:space:]'
}

normalize_uint() {
  local raw
  raw="$(trim "$1")"
  if [[ "$raw" =~ ^0x[0-9a-fA-F]+$ ]]; then
    cast to-dec "$raw"
  else
    echo "$raw"
  fi
}

normalize_bool() {
  local raw
  raw="$(trim "$1" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    true|1|0x1) echo "true" ;;
    false|0|0x0) echo "false" ;;
    *) echo "$raw" ;;
  esac
}

assert_eq() {
  local expected="$1"
  local got="$2"
  local msg="$3"
  if [[ "$expected" != "$got" ]]; then
    fail "$msg (expected=$expected got=$got)"
  fi
  ok "$msg"
}

assert_uint_eq() {
  local expected="$1"
  local got_raw="$2"
  local got
  got="$(normalize_uint "$got_raw")"
  assert_eq "$expected" "$got" "$3"
}

assert_bool_eq() {
  local expected="$1"
  local got_raw="$2"
  local got
  got="$(normalize_bool "$got_raw")"
  assert_eq "$expected" "$got" "$3"
}

status_is_success() {
  local s
  s="$(trim "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "$s" == "1" || "$s" == "0x1" ]]
}

status_is_failure() {
  local s
  s="$(trim "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "$s" == "0" || "$s" == "0x0" ]]
}

extract_tx_hash() {
  local text="$1"
  echo "$text" | rg -o '0x[0-9a-fA-F]{64}' | head -n1 || true
}

wait_receipt_json() {
  local tx_hash="$1"
  local i receipt
  for i in $(seq 1 100); do
    receipt="$(cast rpc --no-proxy --rpc-url "$RPC_URL" eth_getTransactionReceipt "$tx_hash" 2>/dev/null || true)"
    receipt="$(trim "$receipt")"
    if [[ -n "$receipt" && "$receipt" != "null" ]]; then
      echo "$receipt"
      return 0
    fi
    sleep 0.1
  done
  return 1
}

receipt_json_field() {
  local receipt_json="$1"
  local field="$2"
  case "$field" in
    status)
      echo "$receipt_json" | rg -o '"status"\s*:\s*"0x[0-9a-fA-F]+"' | head -n1 | awk -F'"' '{print $4}' | tr -d '[:space:]'
      ;;
    contractAddress)
      # Accept either quoted hex or null.
      local m
      m="$(echo "$receipt_json" | rg -o '"contractAddress"\s*:\s*("0x[0-9a-fA-F]{40}"|null)' | head -n1 || true)"
      if [[ -z "$m" ]]; then
        echo ""
      elif echo "$m" | rg -q 'null'; then
        echo "null"
      else
        echo "$m" | rg -o '0x[0-9a-fA-F]{40}' | head -n1 | tr -d '[:space:]'
      fi
      ;;
    *)
      fail "unsupported receipt field: $field"
      ;;
  esac
}

send_tx_hash() {
  local from="$1"
  shift
  cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$from" "$@"
}

deploy_contract() {
  local bytecode_with_ctor_args="$1"
  local deploy_out tx_hash receipt_json status addr

  deploy_out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" --create "$bytecode_with_ctor_args")"
  tx_hash="$(extract_tx_hash "$deploy_out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract deploy tx hash"

  receipt_json="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for deploy receipt"
  status="$(receipt_json_field "$receipt_json" status)"
  status_is_success "$status" || fail "deployment failed (status=$status)"

  addr="$(receipt_json_field "$receipt_json" contractAddress)"
  [[ -n "$addr" && "$addr" != "null" ]] || fail "failed to extract deployed contract address"
  echo "$addr"
}

send_expect_success() {
  local from="$1"
  shift
  local out tx_hash receipt_json status

  out="$(send_tx_hash "$from" "$CONTRACT_ADDR" "$@")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract tx hash for successful send"

  receipt_json="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for tx receipt (expected success)"
  status="$(receipt_json_field "$receipt_json" status)"
  status_is_success "$status" || fail "transaction unexpectedly failed (status=$status)"
}

send_expect_revert() {
  local from="$1"
  shift
  local out rc tx_hash receipt_json status

  set +e
  out="$(send_tx_hash "$from" "$CONTRACT_ADDR" "$@" 2>&1)"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    ok "revert observed for: $*"
    return 0
  fi

  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "expected revert, but tx hash was not found and command succeeded"

  receipt_json="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for tx receipt (expected revert)"
  status="$(receipt_json_field "$receipt_json" status)"
  if status_is_failure "$status"; then
    ok "revert observed for: $*"
    return 0
  fi

  fail "expected revert, but tx succeeded (status=$status): $*"
}

call_view() {
  cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$@"
}

call_view_from() {
  local from="$1"
  shift
  cast call --no-proxy --rpc-url "$RPC_URL" --from "$from" "$CONTRACT_ADDR" "$@"
}

balance_of() {
  call_view "balanceOf(address)(uint256)" "$1"
}

allowance_of() {
  call_view "allowance(address,address)(uint256)" "$1" "$2"
}

log_balances() {
  local label="$1"
  local alice_bal bob_bal carol_bal
  alice_bal="$(normalize_uint "$(balance_of "$ALICE")")"
  bob_bal="$(normalize_uint "$(balance_of "$BOB")")"
  carol_bal="$(normalize_uint "$(balance_of "$CAROL")")"
  echo "balances [$label]: alice=$alice_bal bob=$bob_bal carol=$carol_bal"
}

require_cmd cast
require_cmd rg
require_cmd "$ORA_BIN"

if ! cast block-number --no-proxy --rpc-url "$RPC_URL" >/dev/null 2>&1; then
  fail "cannot reach RPC at $RPC_URL (is anvil running?)"
fi

if [[ -n "$BYTECODE_FILE" ]]; then
  [[ -f "$BYTECODE_FILE" ]] || fail "bytecode file not found: $BYTECODE_FILE"
else
  mkdir -p "$WORK_DIR"
  BYTECODE_FILE="$WORK_DIR/erc20_bitfield_comptime_generics.hex"
  echo "Compiling bytecode..."
  "$ORA_BIN" --emit-bytecode -o "$BYTECODE_FILE" "$ORA_SOURCE"
fi

BYTECODE="$(tr -d '[:space:]' < "$BYTECODE_FILE")"
[[ -n "$BYTECODE" ]] || fail "empty bytecode in $BYTECODE_FILE"
if [[ "$BYTECODE" != 0x* ]]; then
  BYTECODE="0x$BYTECODE"
fi

# `init(initial_supply)` is constructor-style in Ora.
# Constructor args must be appended to creation bytecode.
CTOR_ARGS="$(cast abi-encode "f(uint256)" "$INITIAL_SUPPLY")"
DEPLOY_DATA="${BYTECODE}${CTOR_ARGS#0x}"

bytecode_chars="${#BYTECODE}"
ctor_chars="${#CTOR_ARGS}"
deploy_chars="${#DEPLOY_DATA}"
expected_deploy_chars=$((bytecode_chars + 64))

echo "bytecode chars: $bytecode_chars"
echo "ctor arg chars: $ctor_chars"
echo "deploy chars:   $deploy_chars"
if [[ "$deploy_chars" -ne "$expected_deploy_chars" ]]; then
  fail "constructor args were not appended correctly (expected deploy chars=$expected_deploy_chars got=$deploy_chars)"
fi

echo "Deploying contract..."
CONTRACT_ADDR="$(deploy_contract "$DEPLOY_DATA")"
echo "Contract address: $CONTRACT_ADDR"

echo "Running ERC20 checks..."

# 1) read-only checks
assert_uint_eq "18" "$(call_view "decimals()(uint8)")" "decimals() == 18"
assert_bool_eq "false" "$(call_view "isPaused()(bool)")" "isPaused() default false"
assert_uint_eq "$INITIAL_SUPPLY" "$(balance_of "$ALICE")" "deployer balance after init"
log_balances "after init"

# 2) approve + allowance
assert_bool_eq "true" "$(call_view_from "$ALICE" "approve(address,uint256)(bool)" "$BOB" 200)" "approve() call returns true"
send_expect_success "$ALICE" "approve(address,uint256)" "$BOB" 200
assert_uint_eq "200" "$(allowance_of "$ALICE" "$BOB")" "allowance set to 200"
log_balances "after approve"

# 3) transfer success path
assert_bool_eq "true" "$(call_view_from "$ALICE" "transfer(address,uint256)(bool)" "$BOB" 150)" "transfer() call returns true"
send_expect_success "$ALICE" "transfer(address,uint256)" "$BOB" 150
assert_uint_eq "$((INITIAL_SUPPLY - 150))" "$(balance_of "$ALICE")" "alice debited by transfer"
assert_uint_eq "150" "$(balance_of "$BOB")" "bob credited by transfer"
log_balances "after transfer alice->bob (150)"

# 4) transfer revert paths (refinement guards)
send_expect_revert "$ALICE" "transfer(address,uint256)" "$BOB" 0
send_expect_revert "$ALICE" "transfer(address,uint256)" "$ZERO_ADDRESS" 1

# 5) paused flag bitfield round-trip
send_expect_success "$ALICE" "setPaused(bool)" true
assert_bool_eq "true" "$(call_view "isPaused()(bool)")" "isPaused() after setPaused(true)"
send_expect_success "$ALICE" "setPaused(bool)" false
assert_bool_eq "false" "$(call_view "isPaused()(bool)")" "isPaused() after setPaused(false)"

# 6) transferFrom success path
assert_bool_eq "true" "$(call_view_from "$BOB" "transferFrom(address,address,uint256)(bool)" "$ALICE" "$CAROL" 120)" "transferFrom() call returns true"
send_expect_success "$BOB" "transferFrom(address,address,uint256)" "$ALICE" "$CAROL" 120
assert_uint_eq "$((INITIAL_SUPPLY - 150 - 120))" "$(balance_of "$ALICE")" "alice debited by transferFrom"
assert_uint_eq "120" "$(balance_of "$CAROL")" "carol credited by transferFrom"
assert_uint_eq "80" "$(allowance_of "$ALICE" "$BOB")" "allowance reduced to 80"
log_balances "after transferFrom alice->carol (120)"

# 7) transferFrom soft-fail path: insufficient allowance (returns false, no revert)
assert_bool_eq "false" "$(call_view_from "$BOB" "transferFrom(address,address,uint256)(bool)" "$ALICE" "$CAROL" 200)" "transferFrom() call returns false when allowance is low"
before_alice="$(normalize_uint "$(balance_of "$ALICE")")"
before_carol="$(normalize_uint "$(balance_of "$CAROL")")"
before_allowance="$(normalize_uint "$(allowance_of "$ALICE" "$BOB")")"
send_expect_success "$BOB" "transferFrom(address,address,uint256)" "$ALICE" "$CAROL" 200
after_alice="$(normalize_uint "$(balance_of "$ALICE")")"
after_carol="$(normalize_uint "$(balance_of "$CAROL")")"
after_allowance="$(normalize_uint "$(allowance_of "$ALICE" "$BOB")")"
assert_eq "$before_alice" "$after_alice" "transferFrom(false) keeps sender balance unchanged"
assert_eq "$before_carol" "$after_carol" "transferFrom(false) keeps recipient balance unchanged"
assert_eq "$before_allowance" "$after_allowance" "transferFrom(false) keeps allowance unchanged"
log_balances "after transferFrom(false) (state unchanged)"

# 8) mint success + refine guard path
assert_bool_eq "true" "$(call_view "mint(address,uint256)(bool)" "$BOB" 50)" "mint() call returns true"
send_expect_success "$ALICE" "mint(address,uint256)" "$BOB" 50
assert_uint_eq "200" "$(balance_of "$BOB")" "bob balance after mint"
bob_after_mint="$(normalize_uint "$(balance_of "$BOB")")"
echo "bob balance after mint [$bob_after_mint]"
log_balances "after mint to bob (50)"

# 8b) transfer alice -> bob after mint
assert_bool_eq "true" "$(call_view_from "$ALICE" "transfer(address,uint256)(bool)" "$BOB" 10)" "post-mint transfer() call returns true"
send_expect_success "$ALICE" "transfer(address,uint256)" "$BOB" 10
assert_uint_eq "$((INITIAL_SUPPLY - 150 - 120 - 10))" "$(balance_of "$ALICE")" "alice debited by post-mint transfer"
assert_uint_eq "210" "$(balance_of "$BOB")" "bob credited by post-mint transfer"
log_balances "after post-mint transfer alice->bob (10)"

# 8c) refine guard path checks
send_expect_revert "$ALICE" "mint(address,uint256)" "$ZERO_ADDRESS" 1
send_expect_revert "$ALICE" "mint(address,uint256)" "$BOB" 0

# 9) approve refine guard path
send_expect_revert "$ALICE" "approve(address,uint256)" "$ZERO_ADDRESS" 1

ok "ERC20BitfieldComptimeGenerics Anvil checks passed"
