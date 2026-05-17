#!/usr/bin/env bash
set -euo pipefail

# Stage-1 bytecode/storage equivalence smoke.
#
# This is intentionally narrow: it compiles a verified scalar-storage Ora
# contract, deploys the emitted bytecode to a local Anvil node, exercises the
# mutating entrypoint, and checks that the public getter and raw EVM storage
# slot 0 agree after each transaction. It also checks mapping shapes:
# - map<address, u256> at root slot 0 reads and writes keccak256(key || slot)
# - map<u256, u256> at root slot 0 uses the full 256-bit integer key in
#   keccak256(key || slot), including keys wider than an address
# - map<intN, u256> at root slots preserves narrow integer ABI spelling and
#   hashes left-padded unsigned and sign-extended signed key words
# - map<bytesN, u256> at root slots preserves fixed-bytes ABI spelling and
#   right-padded key words for bytes1, bytes4, bytes20, bytes31, and bytes32
# - map<address, map<address, u256>> at root slot 0 reads and writes
#   keccak256(spender || keccak256(owner || slot))
# - map<address, map<address, map<address, u256>>> extends the same recurrence
#   one more level and keeps asymmetric key triples isolated
# - map<address, map<address, Struct>> starts each word-sized struct field at
#   keccak256(spender || keccak256(owner || slot)) + field_index
# - slice[u256] at root slot 0 stores length at slot 0 and elements at
#   keccak256(slot) + index
# - a root-slot struct of u256 fields occupies consecutive storage slots
# - map<address, Struct> stores each word-sized struct field at
#   keccak256(key || slot) + field_index
# - slice[Struct] stores length at the root slot and each word-sized struct
#   field at keccak256(slot) + index * field_count + field_index
# - struct { values: slice[u256] } stores the slice length at the field slot
#   and elements at keccak256(field_slot) + index
# - struct { head: u256, values: slice[u256], tail: u256 } uses a non-zero
#   field slot for the dynamic member and preserves scalar neighbors
# - struct { left: slice[u256], pivot: u256, right: slice[u256] } keeps two
#   dynamic fields at independent roots while preserving the scalar neighbor
# - map<address, struct { left: slice[u256], pivot: u256, right: slice[u256] }>
#   keeps both dynamic roots scoped under the mapping cell for each account
# - struct { head: u256, duo: struct { left: slice[u256], pivot: u256,
#   right: slice[u256] }, tail: u256 } keeps nested dynamic roots independent
# - slice[slice[u256]] stores the outer length at the root slot, row lengths at
#   keccak256(root_slot) + row, and row elements at keccak256(row_slot) + col
# - map<address, slice[u256]> (spelled through a type alias) stores each slice
#   length at keccak256(address || root_slot), with elements at
#   keccak256(map_cell) + index
# - map<address, struct { u256, slice[u256], u256 }> stores scalar fields at
#   the map cell and dynamic-field elements at keccak256(map_cell + 1) + index
# - map<address, map<address, struct { u256, slice[u256], u256 }>> stores the
#   dynamic-member struct at a two-level map cell and keeps owner/spender pairs
#   isolated
# - bitfield values store bool/u8/u16 fields in one packed root slot and
#   field updates preserve neighboring bit ranges
# - custom bitfield values store explicit @bits ranges, including a signed i16
#   field and a u8 field narrowed to five storage bits
# - map<address, bitfield> stores each packed value at the map cell hash and
#   field updates preserve neighboring bit ranges plus other account keys
# - map<address, custom bitfield> stores signed/custom-width packed values at
#   the map cell hash and preserves per-account packed update locality
# - slice[bitfield] stores length at the root slot and each packed value at
#   keccak256(slot) + index with field-update locality
# - slice[custom bitfield] stores signed/custom-width packed values at
#   keccak256(slot) + index with signed-field update locality
# - struct { u256, bitfield, u256 } stores the packed value at its field slot
#   and preserves scalar neighbors across packed-field updates
# - struct { u256, custom bitfield, u256 } stores signed/custom-width packed
#   values at its field slot and preserves scalar neighbors across updates
# - slice[struct { u256, custom bitfield, u256 }] stores each three-slot row at
#   keccak256(slot) + index * 3 and preserves adjacent rows across packed-field
#   and scalar-neighbor updates
# - nested structs flatten inner word-sized fields into consecutive root slots
#   and inner-field updates preserve outer and inner neighbors
# - deeper nested structs recursively flatten 3+ levels and deep-leaf updates
#   preserve every ancestor and sibling field
# - map<address, nested struct> starts that flattened layout at the map cell
#   hash and keeps account keys isolated
# - map<address, 3-level nested struct> starts the seven-slot flattened layout
#   at the map cell hash and keeps deep updates isolated to one account
# - slice[3-level nested struct] stores each seven-slot flattened element at
#   keccak256(slot) + index * 7 and preserves adjacent elements on deep updates
# - deep nested structs with a dynamic-member leaf store the dynamic length at
#   the recursive field slot and elements at keccak256(field_slot) + index
# - high-arity external ABI entrypoints preserve 6- and 8-argument values across
#   dispatcher decoding and Sensei internal-call argument transfer
# - two deployed contracts preserve storage isolation across a state-changing
#   external call: caller storage remains unchanged while callee storage mutates
#   and caller-side storage can be driven by scalar and structured external-call
#   return data, including a single external call that both mutates callee state
#   and returns structured data consumed by the caller
# - failed external calls are catchable by `try`/`catch`; uncaught failed
#   external calls bubble as transaction reverts and roll back caller writes
# - out-of-gas external calls with no typed revert payload are catchable by
#   `try`/`catch`, and uncaught OOG failures roll back caller writes
# - a three-contract A -> B -> C call chain preserves each contract's storage
#   namespace while return data propagates across both external-call boundaries
# - a two-hop A -> B -> C try/catch path lets B catch C's revert, commit B's
#   catch-state, return normally, and let A commit based on that return value
# - an uncaught two-hop A -> B -> C revert rolls back pre-call writes in both
#   A and B while preserving C's prior state
# - transient-storage locks roll back with a reverted child call, so a caller
#   can catch the failure and call the same target again in the same transaction
# - transient-storage locks are visible to reentrant calls into the same
#   contract and still remain usable by the parent frame after the callback

RPC_URL="${RPC_URL:-http://127.0.0.1:8547}"
ORA_BIN="${ORA_BIN:-./zig-out/bin/ora}"
ORA_SOURCE="${ORA_SOURCE:-ora-example/apps/counter.ora}"
WORK_DIR="${WORK_DIR:-/tmp/ora_bytecode_storage_equiv}"
DEPLOYER="${DEPLOYER:-0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266}"
ALICE="${ALICE:-$DEPLOYER}"
BOB="${BOB:-0x70997970C51812dc3A010C7d01b50e0d17dc79C8}"
CHARLIE="${CHARLIE:-0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC}"
ZERO_ADDRESS="0x0000000000000000000000000000000000000000"

DEFAULT_RPC_URL="http://127.0.0.1:8547"
ANVIL_PID=""
CONTRACT_ADDR=""

export FOUNDRY_DISABLE_NIGHTLY_WARNING=1
export NO_PROXY="*"
export no_proxy="*"
export HTTP_PROXY=""
export HTTPS_PROXY=""
export ALL_PROXY=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

ok() {
  echo "OK: $*"
}

trim() {
  echo "$1" | tr -d '[:space:]'
}

normalize_uint() {
  local raw
  raw="$(echo "$1" | awk '{print $1}')"
  raw="$(trim "$raw")"
  if [[ "$raw" =~ ^0x[0-9a-fA-F]+$ ]]; then
    cast to-dec "$raw"
  else
    echo "$raw"
  fi
}

normalize_int() {
  local raw="$1"
  local bits="$2"

  python3 -c 'import sys
raw = sys.argv[1].strip().split()[0]
bits = int(sys.argv[2])
value = int(raw, 16) if raw.lower().startswith("0x") else int(raw)
mask = (1 << bits) - 1
value &= mask
if value >= (1 << (bits - 1)):
    value -= 1 << bits
print(value)
	' "$raw" "$bits"
}

expected_slot_hex() {
  printf "0x%064x" "$1"
}

expected_address_slot_hex() {
  local clean="${1#0x}"
  clean="${clean,,}"
  printf "0x%064s" "$clean" | tr ' ' '0'
}

extract_tx_hash() {
  echo "$1" | grep -Eo '0x[0-9a-fA-F]{64}' | head -n1 || true
}

receipt_json_field() {
  local receipt_json="$1"
  local field="$2"
  case "$field" in
    status)
      echo "$receipt_json" | grep -Eo '"status"[[:space:]]*:[[:space:]]*"0x[0-9a-fA-F]+"' | head -n1 | awk -F'"' '{print $4}' | tr -d '[:space:]'
      ;;
    contractAddress)
      local match
      match="$(echo "$receipt_json" | grep -Eo '"contractAddress"[[:space:]]*:[[:space:]]*("0x[0-9a-fA-F]{40}"|null)' | head -n1 || true)"
      if [[ -z "$match" ]]; then
        echo ""
      elif echo "$match" | grep -q 'null'; then
        echo "null"
      else
        echo "$match" | grep -Eo '0x[0-9a-fA-F]{40}' | head -n1 | tr -d '[:space:]'
      fi
      ;;
    *)
      fail "unsupported receipt field: $field"
      ;;
  esac
}

wait_receipt_json() {
  local tx_hash="$1"
  local receipt
  for _ in $(seq 1 100); do
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

rpc_reachable() {
  cast block-number --no-proxy --rpc-url "$RPC_URL" >/dev/null 2>&1
}

start_anvil_if_needed() {
  if rpc_reachable; then
    return 0
  fi

  if [[ "$RPC_URL" != "$DEFAULT_RPC_URL" ]]; then
    fail "cannot reach custom RPC_URL=$RPC_URL; start an EVM node or use the default local Anvil URL"
  fi

  require_cmd anvil
  mkdir -p "$WORK_DIR"
  anvil --host 127.0.0.1 --port 8547 >"$WORK_DIR/anvil.log" 2>&1 &
  ANVIL_PID="$!"

  for _ in $(seq 1 100); do
    if rpc_reachable; then
      ok "started local Anvil at $RPC_URL"
      return 0
    fi
    sleep 0.1
  done

  fail "timed out waiting for Anvil; see $WORK_DIR/anvil.log"
}

cleanup() {
  if [[ -n "$ANVIL_PID" ]]; then
    kill "$ANVIL_PID" >/dev/null 2>&1 || true
    wait "$ANVIL_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

deploy_contract() {
  local bytecode="$1"
  local out tx_hash receipt status addr

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" --create "$bytecode")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract deployment transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for deployment receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "deployment failed with status=$status"

  addr="$(receipt_json_field "$receipt" contractAddress)"
  [[ -n "$addr" && "$addr" != "null" ]] || fail "failed to extract deployed contract address"
  echo "$addr"
}

send_contract_tx() {
  local addr="$1"
  local label="$2"
  local signature="$3"
  shift 3
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$addr" "$signature" "$@")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract $label transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for $label receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "$label failed with status=$status"
}

send_contract_tx_expect_revert() {
  local addr="$1"
  local label="$2"
  local signature="$3"
  shift 3
  local out rc tx_hash receipt status

  set +e
  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$addr" "$signature" "$@" 2>&1)"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    ok "$label reverted"
    return 0
  fi

  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "expected $label to revert, but transaction hash was not found"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for $label revert receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x0" || "$status" == "0" ]] || fail "expected $label to revert, but status=$status"

  ok "$label reverted"
}

send_increment() {
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "increment()")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract increment transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for increment receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "increment failed with status=$status"
}

assert_storage_and_getter() {
  local expected="$1"
  local getter_raw getter slot_raw slot_expected

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get()(uint256)")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "get() mismatch: expected=$expected got=$getter"

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slot 0 mismatch: expected=$slot_expected got=$slot_raw"

  ok "get() and raw slot 0 both equal $expected"
}

compile_bytecode() {
  local source="$1"
  local output="$2"

  echo "Compiling $source" >&2
  rm -f "$output"
  "$ORA_BIN" --emit-bytecode -o "$output" "$source" >&2 || fail "bytecode compilation failed for $source"

  read_compiled_bytecode "$output"
}

compile_bytecode_without_verification() {
  local source="$1"
  local output="$2"

  echo "Compiling $source with --no-verify for deployed-runtime smoke" >&2
  rm -f "$output"
  "$ORA_BIN" --no-verify --emit-bytecode -o "$output" "$source" >&2 || fail "bytecode compilation failed for $source"

  read_compiled_bytecode "$output"
}

read_compiled_bytecode() {
  local output="$1"
  local bytecode

  bytecode="$(tr -d '[:space:]' < "$output")"
  [[ -n "$bytecode" ]] || fail "empty bytecode in $output"
  if [[ "$bytecode" != 0x* ]]; then
    bytecode="0x$bytecode"
  fi
  echo "$bytecode"
}

write_mapping_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract MappingStorageSmoke {
    storage var balances: map<address, u256>;

    pub fn set_balance(account: address, amount: u256) {
        balances[account] = amount;
    }

    pub fn get_balance(account: address) -> u256 {
        return balances[account];
    }
}
ORA
}

write_u256_mapping_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract U256MappingStorageSmoke {
    storage var scores: map<u256, u256>;

    pub fn set_score(id: u256, amount: u256) {
        scores[id] = amount;
    }

    pub fn get_score(id: u256) -> u256 {
        return scores[id];
    }
}
ORA
}

write_narrow_integer_mapping_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract NarrowIntegerMappingStorageSmoke {
    storage var u8_scores: map<u8, u256>;
    storage var u16_scores: map<u16, u256>;
    storage var u32_scores: map<u32, u256>;
    storage var u128_scores: map<u128, u256>;
    storage var i8_scores: map<i8, u256>;
    storage var i16_scores: map<i16, u256>;

    pub fn set_u8(id: u8, amount: u256) {
        u8_scores[id] = amount;
    }

    pub fn get_u8(id: u8) -> u256 {
        return u8_scores[id];
    }

    pub fn set_u16(id: u16, amount: u256) {
        u16_scores[id] = amount;
    }

    pub fn get_u16(id: u16) -> u256 {
        return u16_scores[id];
    }

    pub fn set_u32(id: u32, amount: u256) {
        u32_scores[id] = amount;
    }

    pub fn get_u32(id: u32) -> u256 {
        return u32_scores[id];
    }

    pub fn set_u128(id: u128, amount: u256) {
        u128_scores[id] = amount;
    }

    pub fn get_u128(id: u128) -> u256 {
        return u128_scores[id];
    }

    pub fn set_i8(id: i8, amount: u256) {
        i8_scores[id] = amount;
    }

    pub fn get_i8(id: i8) -> u256 {
        return i8_scores[id];
    }

    pub fn set_i16(id: i16, amount: u256) {
        i16_scores[id] = amount;
    }

    pub fn get_i16(id: i16) -> u256 {
        return i16_scores[id];
    }
}
ORA
}

write_fixed_bytes_mapping_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract FixedBytesMappingStorageSmoke {
    storage var b1_scores: map<bytes1, u256>;
    storage var b4_scores: map<bytes4, u256>;
    storage var b20_scores: map<bytes20, u256>;
    storage var b31_scores: map<bytes31, u256>;
    storage var b32_scores: map<bytes32, u256>;

    pub fn set_b1(id: bytes1, amount: u256) {
        b1_scores[id] = amount;
    }

    pub fn get_b1(id: bytes1) -> u256 {
        return b1_scores[id];
    }

    pub fn set_b4(id: bytes4, amount: u256) {
        b4_scores[id] = amount;
    }

    pub fn get_b4(id: bytes4) -> u256 {
        return b4_scores[id];
    }

    pub fn set_b20(id: bytes20, amount: u256) {
        b20_scores[id] = amount;
    }

    pub fn get_b20(id: bytes20) -> u256 {
        return b20_scores[id];
    }

    pub fn set_b31(id: bytes31, amount: u256) {
        b31_scores[id] = amount;
    }

    pub fn get_b31(id: bytes31) -> u256 {
        return b31_scores[id];
    }

    pub fn set_b32(id: bytes32, amount: u256) {
        b32_scores[id] = amount;
    }

    pub fn get_b32(id: bytes32) -> u256 {
        return b32_scores[id];
    }
}
ORA
}

write_nested_mapping_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract NestedMappingStorageSmoke {
    storage var allowances: map<address, map<address, u256>>;

    pub fn set_allowance(owner: address, spender: address, amount: u256) {
        allowances[owner][spender] = amount;
    }

    pub fn get_allowance(owner: address, spender: address) -> u256 {
        return allowances[owner][spender];
    }
}
ORA
}

write_triple_nested_mapping_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract TripleNestedMappingStorageSmoke {
    storage var limits: map<address, map<address, map<address, u256>>>;

    pub fn set_limit(owner: address, spender: address, asset: address, amount: u256) {
        limits[owner][spender][asset] = amount;
    }

    pub fn get_limit(owner: address, spender: address, asset: address) -> u256 {
        return limits[owner][spender][asset];
    }
}
ORA
}

write_nested_map_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Approval {
    amount: u256;
    nonce: u256;
    flags: u256;
}

contract NestedMapStructStorageSmoke {
    storage var approvals: map<address, map<address, Approval>>;

    pub fn set(owner: address, spender: address, amount: u256, nonce: u256, flags: u256) {
        approvals[owner][spender] = Approval { amount: amount, nonce: nonce, flags: flags };
    }

    pub fn set_nonce(owner: address, spender: address, nonce: u256) {
        approvals[owner][spender].nonce = nonce;
    }

    pub fn get_amount(owner: address, spender: address) -> u256 {
        return approvals[owner][spender].amount;
    }

    pub fn get_nonce(owner: address, spender: address) -> u256 {
        return approvals[owner][spender].nonce;
    }

    pub fn get_flags(owner: address, spender: address) -> u256 {
        return approvals[owner][spender].flags;
    }
}
ORA
}

write_dynamic_array_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract DynamicArrayStorageSmoke {
    storage var history: slice[u256];

    pub fn set(values: slice[u256]) {
        history = values;
    }

    pub fn get(index: u256) -> u256 {
        return history[index];
    }
}
ORA
}

write_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Triple {
    left: u256;
    middle: u256;
    right: u256;
}

contract StructStorageSmoke {
    storage var values: Triple;

    pub fn set(left: u256, middle: u256, right: u256) {
        values = Triple { left: left, middle: middle, right: right };
    }

    pub fn set_middle(value: u256) {
        values.middle = value;
    }

    pub fn get_left() -> u256 {
        return values.left;
    }

    pub fn get_middle() -> u256 {
        return values.middle;
    }

    pub fn get_right() -> u256 {
        return values.right;
    }
}
ORA
}

write_map_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Account {
    balance: u256;
    nonce: u256;
    flags: u256;
}

contract MapStructStorageSmoke {
    storage var accounts: map<address, Account>;

    pub fn set(account: address, balance: u256, nonce: u256, flags: u256) {
        accounts[account] = Account { balance: balance, nonce: nonce, flags: flags };
    }

    pub fn set_nonce(account: address, nonce: u256) {
        accounts[account].nonce = nonce;
    }

    pub fn get_balance(account: address) -> u256 {
        return accounts[account].balance;
    }

    pub fn get_nonce(account: address) -> u256 {
        return accounts[account].nonce;
    }

    pub fn get_flags(account: address) -> u256 {
        return accounts[account].flags;
    }
}
ORA
}

write_slice_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Entry {
    left: u256;
    middle: u256;
    right: u256;
}

contract SliceStructStorageSmoke {
    storage var entries: slice[Entry];

    pub fn set(values: slice[Entry]) {
        entries = values;
    }

    pub fn set_middle(index: u256, value: u256) {
        entries[index].middle = value;
    }

    pub fn get_left(index: u256) -> u256 {
        return entries[index].left;
    }

    pub fn get_middle(index: u256) -> u256 {
        return entries[index].middle;
    }

    pub fn get_right(index: u256) -> u256 {
        return entries[index].right;
    }
}
ORA
}

write_struct_slice_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Bag {
    values: slice[u256];
}

contract StructSliceStorageSmoke {
    storage var bag: Bag;

    pub fn set(values: slice[u256]) {
        bag.values = values;
    }

    pub fn get(index: u256) -> u256 {
        return bag.values[index];
    }
}
ORA
}

write_multi_struct_slice_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Bag {
    head: u256;
    values: slice[u256];
    tail: u256;
}

contract MultiStructSliceStorageSmoke {
    storage var bag: Bag;

    pub fn set(head: u256, values: slice[u256], tail: u256) {
        bag = Bag { head: head, values: values, tail: tail };
    }

    pub fn set_head(value: u256) {
        bag.head = value;
    }

    pub fn set_values(values: slice[u256]) {
        bag.values = values;
    }

    pub fn set_tail(value: u256) {
        bag.tail = value;
    }

    pub fn get_head() -> u256 {
        return bag.head;
    }

    pub fn get(index: u256) -> u256 {
        return bag.values[index];
    }

    pub fn get_tail() -> u256 {
        return bag.tail;
    }
}
ORA
}

write_dual_struct_slice_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Duo {
    left: slice[u256];
    pivot: u256;
    right: slice[u256];
}

contract DualStructSliceStorageSmoke {
    storage var bag: Duo;

    pub fn set(left: slice[u256], pivot: u256, right: slice[u256]) {
        bag = Duo { left: left, pivot: pivot, right: right };
    }

    pub fn set_left(values: slice[u256]) {
        bag.left = values;
    }

    pub fn set_pivot(value: u256) {
        bag.pivot = value;
    }

    pub fn set_right(values: slice[u256]) {
        bag.right = values;
    }

    pub fn get_left(index: u256) -> u256 {
        return bag.left[index];
    }

    pub fn get_pivot() -> u256 {
        return bag.pivot;
    }

    pub fn get_right(index: u256) -> u256 {
        return bag.right[index];
    }
}
ORA
}

write_slice_slice_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract SliceSliceStorageSmoke {
    storage var matrix: slice[slice[u256]];

    pub fn set(values: slice[slice[u256]]) {
        matrix = values;
    }

    pub fn get(row: u256, col: u256) -> u256 {
        return matrix[row][col];
    }
}
ORA
}

write_slice_dynamic_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Row {
    head: u256;
    values: slice[u256];
    tail: u256;
}

contract SliceDynamicStructStorageSmoke {
    storage var rows: slice[Row];

    pub fn set(values: slice[Row]) {
        rows = values;
    }

    pub fn set_head(index: u256, value: u256) {
        rows[index].head = value;
    }

    pub fn set_values(index: u256, values: slice[u256]) {
        rows[index].values = values;
    }

    pub fn set_tail(index: u256, value: u256) {
        rows[index].tail = value;
    }

    pub fn get_head(index: u256) -> u256 {
        return rows[index].head;
    }

    pub fn get_value(index: u256, col: u256) -> u256 {
        return rows[index].values[col];
    }

    pub fn get_tail(index: u256) -> u256 {
        return rows[index].tail;
    }
}
ORA
}

write_slice_nested_dual_dynamic_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Duo {
    left: slice[u256];
    pivot: u256;
    right: slice[u256];
}

struct Outer {
    head: u256;
    duo: Duo;
    tail: u256;
}

contract SliceNestedDualDynamicStructStorageSmoke {
    storage var records: slice[Outer];

    pub fn set(values: slice[Outer]) {
        records = values;
    }

    pub fn set_left(index: u256, values: slice[u256]) {
        records[index].duo.left = values;
    }

    pub fn set_pivot(index: u256, value: u256) {
        records[index].duo.pivot = value;
    }

    pub fn set_right(index: u256, values: slice[u256]) {
        records[index].duo.right = values;
    }

    pub fn set_tail(index: u256, value: u256) {
        records[index].tail = value;
    }

    pub fn get_head(index: u256) -> u256 {
        return records[index].head;
    }

    pub fn get_left(index: u256, col: u256) -> u256 {
        return records[index].duo.left[col];
    }

    pub fn get_pivot(index: u256) -> u256 {
        return records[index].duo.pivot;
    }

    pub fn get_right(index: u256, col: u256) -> u256 {
        return records[index].duo.right[col];
    }

    pub fn get_tail(index: u256) -> u256 {
        return records[index].tail;
    }
}
ORA
}

write_map_slice_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract MapSliceStorageSmoke {
    type Row = slice[u256];

    storage var rows: map<address, Row>;

    pub fn set(account: address, values: Row) {
        rows[account] = values;
    }

    pub fn get(account: address, index: u256) -> u256 {
        return rows[account][index];
    }
}
ORA
}

write_map_dynamic_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Bag {
    head: u256;
    values: slice[u256];
    tail: u256;
}

contract MapDynamicStructStorageSmoke {
    storage var bags: map<address, Bag>;

    pub fn set(account: address, head: u256, values: slice[u256], tail: u256) {
        bags[account] = Bag { head: head, values: values, tail: tail };
    }

    pub fn set_head(account: address, value: u256) {
        bags[account].head = value;
    }

    pub fn set_values(account: address, values: slice[u256]) {
        bags[account].values = values;
    }

    pub fn set_tail(account: address, value: u256) {
        bags[account].tail = value;
    }

    pub fn get_head(account: address) -> u256 {
        return bags[account].head;
    }

    pub fn get_value(account: address, index: u256) -> u256 {
        return bags[account].values[index];
    }

    pub fn get_tail(account: address) -> u256 {
        return bags[account].tail;
    }
}
ORA
}

write_map_dual_dynamic_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Duo {
    left: slice[u256];
    pivot: u256;
    right: slice[u256];
}

contract MapDualDynamicStructStorageSmoke {
    storage var bags: map<address, Duo>;

    pub fn set(account: address, left: slice[u256], pivot: u256, right: slice[u256]) {
        bags[account] = Duo { left: left, pivot: pivot, right: right };
    }

    pub fn set_left(account: address, values: slice[u256]) {
        bags[account].left = values;
    }

    pub fn set_pivot(account: address, value: u256) {
        bags[account].pivot = value;
    }

    pub fn set_right(account: address, values: slice[u256]) {
        bags[account].right = values;
    }

    pub fn get_left(account: address, index: u256) -> u256 {
        return bags[account].left[index];
    }

    pub fn get_pivot(account: address) -> u256 {
        return bags[account].pivot;
    }

    pub fn get_right(account: address, index: u256) -> u256 {
        return bags[account].right[index];
    }
}
ORA
}

write_nested_map_dynamic_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Bag {
    head: u256;
    values: slice[u256];
    tail: u256;
}

contract NestedMapDynamicStructStorageSmoke {
    storage var bags: map<address, map<address, Bag>>;

    pub fn set(owner: address, spender: address, head: u256, values: slice[u256], tail: u256) {
        bags[owner][spender] = Bag { head: head, values: values, tail: tail };
    }

    pub fn set_head(owner: address, spender: address, value: u256) {
        bags[owner][spender].head = value;
    }

    pub fn set_values(owner: address, spender: address, values: slice[u256]) {
        bags[owner][spender].values = values;
    }

    pub fn set_tail(owner: address, spender: address, value: u256) {
        bags[owner][spender].tail = value;
    }

    pub fn get_head(owner: address, spender: address) -> u256 {
        return bags[owner][spender].head;
    }

    pub fn get_value(owner: address, spender: address, index: u256) -> u256 {
        return bags[owner][spender].values[index];
    }

    pub fn get_tail(owner: address, spender: address) -> u256 {
        return bags[owner][spender].tail;
    }
}
ORA
}

write_bitfield_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
bitfield PackedFlags : u256 {
    enabled: bool;
    locked: bool;
    mode: u8;
    threshold: u16;
}

contract BitfieldStorageSmoke {
    storage var flags: PackedFlags;

    pub fn configure(mode: u8, threshold: u16) {
        let f: PackedFlags = flags;
        f.enabled = true;
        f.locked = false;
        f.mode = mode;
        f.threshold = threshold;
        flags = f;
    }

    pub fn set_locked(value: bool) {
        let f: PackedFlags = flags;
        f.locked = value;
        flags = f;
    }

    pub fn set_mode(value: u8) {
        let f: PackedFlags = flags;
        f.mode = value;
        flags = f;
    }

    pub fn is_enabled() -> bool {
        let f: PackedFlags = flags;
        return f.enabled;
    }

    pub fn is_locked() -> bool {
        let f: PackedFlags = flags;
        return f.locked;
    }

    pub fn get_mode() -> u8 {
        let f: PackedFlags = flags;
        return f.mode;
    }

    pub fn get_threshold() -> u16 {
        let f: PackedFlags = flags;
        return f.threshold;
    }
}
ORA
}

write_map_bitfield_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
bitfield PackedFlags : u256 {
    enabled: bool;
    locked: bool;
    mode: u8;
    threshold: u16;
}

contract MapBitfieldStorageSmoke {
    storage var flags: map<address, PackedFlags>;

    pub fn configure(account: address, mode: u8, threshold: u16) {
        let f: PackedFlags = flags[account];
        f.enabled = true;
        f.locked = false;
        f.mode = mode;
        f.threshold = threshold;
        flags[account] = f;
    }

    pub fn set_locked(account: address, value: bool) {
        let f: PackedFlags = flags[account];
        f.locked = value;
        flags[account] = f;
    }

    pub fn set_mode(account: address, value: u8) {
        let f: PackedFlags = flags[account];
        f.mode = value;
        flags[account] = f;
    }

    pub fn is_enabled(account: address) -> bool {
        let f: PackedFlags = flags[account];
        return f.enabled;
    }

    pub fn is_locked(account: address) -> bool {
        let f: PackedFlags = flags[account];
        return f.locked;
    }

    pub fn get_mode(account: address) -> u8 {
        let f: PackedFlags = flags[account];
        return f.mode;
    }

    pub fn get_threshold(account: address) -> u16 {
        let f: PackedFlags = flags[account];
        return f.threshold;
    }
}
ORA
}

write_map_custom_bitfield_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
bitfield CustomFlags : u256 {
    enabled: bool @bits(0..1);
    code: u8 @bits(1..6);
    delta: i16 @bits(6..22);
    amount: u32 @bits(22..54);
}

contract MapCustomBitfieldStorageSmoke {
    storage var flags: map<address, CustomFlags>;

    pub fn configure(account: address, code: u8, delta: i16, amount: u32) {
        let f: CustomFlags = flags[account];
        f.enabled = true;
        f.code = code;
        f.delta = delta;
        f.amount = amount;
        flags[account] = f;
    }

    pub fn set_delta(account: address, value: i16) {
        let f: CustomFlags = flags[account];
        f.delta = value;
        flags[account] = f;
    }

    pub fn set_code(account: address, value: u8) {
        let f: CustomFlags = flags[account];
        f.code = value;
        flags[account] = f;
    }

    pub fn is_enabled(account: address) -> bool {
        let f: CustomFlags = flags[account];
        return f.enabled;
    }

    pub fn get_code(account: address) -> u8 {
        let f: CustomFlags = flags[account];
        return f.code;
    }

    pub fn get_delta(account: address) -> i16 {
        let f: CustomFlags = flags[account];
        return f.delta;
    }

    pub fn get_amount(account: address) -> u32 {
        let f: CustomFlags = flags[account];
        return f.amount;
    }
}
ORA
}

write_custom_bitfield_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
bitfield CustomFlags : u256 {
    enabled: bool @bits(0..1);
    code: u8 @bits(1..6);
    delta: i16 @bits(6..22);
    amount: u32 @bits(22..54);
}

contract CustomBitfieldStorageSmoke {
    storage var flags: CustomFlags;

    pub fn configure(code: u8, delta: i16, amount: u32) {
        let f: CustomFlags = flags;
        f.enabled = true;
        f.code = code;
        f.delta = delta;
        f.amount = amount;
        flags = f;
    }

    pub fn set_delta(value: i16) {
        let f: CustomFlags = flags;
        f.delta = value;
        flags = f;
    }

    pub fn set_code(value: u8) {
        let f: CustomFlags = flags;
        f.code = value;
        flags = f;
    }

    pub fn is_enabled() -> bool {
        let f: CustomFlags = flags;
        return f.enabled;
    }

    pub fn get_code() -> u8 {
        let f: CustomFlags = flags;
        return f.code;
    }

    pub fn get_delta() -> i16 {
        let f: CustomFlags = flags;
        return f.delta;
    }

    pub fn get_amount() -> u32 {
        let f: CustomFlags = flags;
        return f.amount;
    }
}
ORA
}

write_slice_bitfield_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
bitfield PackedFlags : u256 {
    enabled: bool;
    locked: bool;
    mode: u8;
    threshold: u16;
}

contract SliceBitfieldStorageSmoke {
    storage var flags: slice[PackedFlags];

    pub fn set(values: slice[PackedFlags]) {
        flags = values;
    }

    pub fn set_locked(index: u256, value: bool) {
        let f: PackedFlags = flags[index];
        f.locked = value;
        flags[index] = f;
    }

    pub fn set_mode(index: u256, value: u8) {
        let f: PackedFlags = flags[index];
        f.mode = value;
        flags[index] = f;
    }

    pub fn is_enabled(index: u256) -> bool {
        let f: PackedFlags = flags[index];
        return f.enabled;
    }

    pub fn is_locked(index: u256) -> bool {
        let f: PackedFlags = flags[index];
        return f.locked;
    }

    pub fn get_mode(index: u256) -> u8 {
        let f: PackedFlags = flags[index];
        return f.mode;
    }

    pub fn get_threshold(index: u256) -> u16 {
        let f: PackedFlags = flags[index];
        return f.threshold;
    }
}
ORA
}

write_slice_custom_bitfield_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
bitfield CustomFlags : u256 {
    enabled: bool @bits(0..1);
    code: u8 @bits(1..6);
    delta: i16 @bits(6..22);
    amount: u32 @bits(22..54);
}

contract SliceCustomBitfieldStorageSmoke {
    storage var flags: slice[CustomFlags];

    pub fn set(values: slice[CustomFlags]) {
        flags = values;
    }

    pub fn set_delta(index: u256, value: i16) {
        let f: CustomFlags = flags[index];
        f.delta = value;
        flags[index] = f;
    }

    pub fn set_code(index: u256, value: u8) {
        let f: CustomFlags = flags[index];
        f.code = value;
        flags[index] = f;
    }

    pub fn is_enabled(index: u256) -> bool {
        let f: CustomFlags = flags[index];
        return f.enabled;
    }

    pub fn get_code(index: u256) -> u8 {
        let f: CustomFlags = flags[index];
        return f.code;
    }

    pub fn get_delta(index: u256) -> i16 {
        let f: CustomFlags = flags[index];
        return f.delta;
    }

    pub fn get_amount(index: u256) -> u32 {
        let f: CustomFlags = flags[index];
        return f.amount;
    }
}
ORA
}

write_struct_bitfield_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
bitfield PackedFlags : u256 {
    enabled: bool;
    locked: bool;
    mode: u8;
    threshold: u16;
}

struct Config {
    head: u256;
    flags: PackedFlags;
    tail: u256;
}

contract StructBitfieldStorageSmoke {
    storage var config: Config;

    pub fn configure(head: u256, mode: u8, threshold: u16, tail: u256) {
        let f: PackedFlags = config.flags;
        f.enabled = true;
        f.locked = false;
        f.mode = mode;
        f.threshold = threshold;
        config.head = head;
        config.flags = f;
        config.tail = tail;
    }

    pub fn set_head(value: u256) {
        config.head = value;
    }

    pub fn set_locked(value: bool) {
        let f: PackedFlags = config.flags;
        f.locked = value;
        config.flags = f;
    }

    pub fn set_mode(value: u8) {
        let f: PackedFlags = config.flags;
        f.mode = value;
        config.flags = f;
    }

    pub fn set_tail(value: u256) {
        config.tail = value;
    }

    pub fn get_head() -> u256 {
        return config.head;
    }

    pub fn is_enabled() -> bool {
        let f: PackedFlags = config.flags;
        return f.enabled;
    }

    pub fn is_locked() -> bool {
        let f: PackedFlags = config.flags;
        return f.locked;
    }

    pub fn get_mode() -> u8 {
        let f: PackedFlags = config.flags;
        return f.mode;
    }

    pub fn get_threshold() -> u16 {
        let f: PackedFlags = config.flags;
        return f.threshold;
    }

    pub fn get_tail() -> u256 {
        return config.tail;
    }
}
ORA
}

write_struct_custom_bitfield_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
bitfield CustomFlags : u256 {
    enabled: bool @bits(0..1);
    code: u8 @bits(1..6);
    delta: i16 @bits(6..22);
    amount: u32 @bits(22..54);
}

struct Config {
    head: u256;
    flags: CustomFlags;
    tail: u256;
}

contract StructCustomBitfieldStorageSmoke {
    storage var config: Config;

    pub fn configure(head: u256, code: u8, delta: i16, amount: u32, tail: u256) {
        let f: CustomFlags = config.flags;
        f.enabled = true;
        f.code = code;
        f.delta = delta;
        f.amount = amount;
        config.head = head;
        config.flags = f;
        config.tail = tail;
    }

    pub fn set_head(value: u256) {
        config.head = value;
    }

    pub fn set_delta(value: i16) {
        let f: CustomFlags = config.flags;
        f.delta = value;
        config.flags = f;
    }

    pub fn set_code(value: u8) {
        let f: CustomFlags = config.flags;
        f.code = value;
        config.flags = f;
    }

    pub fn set_tail(value: u256) {
        config.tail = value;
    }

    pub fn get_head() -> u256 {
        return config.head;
    }

    pub fn is_enabled() -> bool {
        let f: CustomFlags = config.flags;
        return f.enabled;
    }

    pub fn get_code() -> u8 {
        let f: CustomFlags = config.flags;
        return f.code;
    }

    pub fn get_delta() -> i16 {
        let f: CustomFlags = config.flags;
        return f.delta;
    }

    pub fn get_amount() -> u32 {
        let f: CustomFlags = config.flags;
        return f.amount;
    }

    pub fn get_tail() -> u256 {
        return config.tail;
    }
}
ORA
}

write_slice_struct_custom_bitfield_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
bitfield CustomFlags : u256 {
    enabled: bool @bits(0..1);
    code: u8 @bits(1..6);
    delta: i16 @bits(6..22);
    amount: u32 @bits(22..54);
}

struct Row {
    head: u256;
    flags: CustomFlags;
    tail: u256;
}

contract SliceStructCustomBitfieldStorageSmoke {
    storage var rows: slice[Row];

    pub fn set(values: slice[Row]) {
        rows = values;
    }

    pub fn set_head(index: u256, value: u256) {
        rows[index].head = value;
    }

    pub fn set_delta(index: u256, value: i16) {
        let f: CustomFlags = rows[index].flags;
        f.delta = value;
        rows[index].flags = f;
    }

    pub fn set_code(index: u256, value: u8) {
        let f: CustomFlags = rows[index].flags;
        f.code = value;
        rows[index].flags = f;
    }

    pub fn set_tail(index: u256, value: u256) {
        rows[index].tail = value;
    }

    pub fn get_head(index: u256) -> u256 {
        return rows[index].head;
    }

    pub fn is_enabled(index: u256) -> bool {
        let f: CustomFlags = rows[index].flags;
        return f.enabled;
    }

    pub fn get_code(index: u256) -> u8 {
        let f: CustomFlags = rows[index].flags;
        return f.code;
    }

    pub fn get_delta(index: u256) -> i16 {
        let f: CustomFlags = rows[index].flags;
        return f.delta;
    }

    pub fn get_amount(index: u256) -> u32 {
        let f: CustomFlags = rows[index].flags;
        return f.amount;
    }

    pub fn get_tail(index: u256) -> u256 {
        return rows[index].tail;
    }
}
ORA
}

write_nested_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Inner {
    left: u256;
    middle: u256;
    right: u256;
}

struct Outer {
    head: u256;
    inner: Inner;
    tail: u256;
}

contract NestedStructStorageSmoke {
    storage var value: Outer;

    pub fn set(head: u256, left: u256, middle: u256, right: u256, tail: u256) {
        value = Outer {
            head: head,
            inner: Inner { left: left, middle: middle, right: right },
            tail: tail,
        };
    }

    pub fn set_inner_middle(middle: u256) {
        value.inner.middle = middle;
    }

    pub fn get_head() -> u256 {
        return value.head;
    }

    pub fn get_left() -> u256 {
        return value.inner.left;
    }

    pub fn get_middle() -> u256 {
        return value.inner.middle;
    }

    pub fn get_right() -> u256 {
        return value.inner.right;
    }

    pub fn get_tail() -> u256 {
        return value.tail;
    }
}
ORA
}

write_map_nested_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Inner {
    left: u256;
    middle: u256;
    right: u256;
}

struct Outer {
    head: u256;
    inner: Inner;
    tail: u256;
}

contract MapNestedStructStorageSmoke {
    storage var accounts: map<address, Outer>;

    pub fn set(account: address, head: u256, left: u256, middle: u256, right: u256, tail: u256) {
        accounts[account] = Outer {
            head: head,
            inner: Inner { left: left, middle: middle, right: right },
            tail: tail,
        };
    }

    pub fn set_inner_middle(account: address, middle: u256) {
        accounts[account].inner.middle = middle;
    }

    pub fn get_head(account: address) -> u256 {
        return accounts[account].head;
    }

    pub fn get_left(account: address) -> u256 {
        return accounts[account].inner.left;
    }

    pub fn get_middle(account: address) -> u256 {
        return accounts[account].inner.middle;
    }

    pub fn get_right(account: address) -> u256 {
        return accounts[account].inner.right;
    }

    pub fn get_tail(account: address) -> u256 {
        return accounts[account].tail;
    }
}
ORA
}

write_deep_nested_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Leaf {
    left: u256;
    middle: u256;
    right: u256;
}

struct Mid {
    before: u256;
    leaf: Leaf;
    after: u256;
}

struct Outer {
    head: u256;
    mid: Mid;
    tail: u256;
}

contract DeepNestedStructStorageSmoke {
    storage var record: Outer;

    pub fn set(head: u256, before: u256, left: u256, middle: u256, right: u256, after: u256, tail: u256) {
        record = Outer {
            head: head,
            mid: Mid {
                before: before,
                leaf: Leaf { left: left, middle: middle, right: right },
                after: after,
            },
            tail: tail,
        };
    }

    pub fn set_leaf_middle(value: u256) {
        record.mid.leaf.middle = value;
    }

    pub fn set_mid_after(value: u256) {
        record.mid.after = value;
    }

    pub fn get_head() -> u256 {
        return record.head;
    }

    pub fn get_before() -> u256 {
        return record.mid.before;
    }

    pub fn get_left() -> u256 {
        return record.mid.leaf.left;
    }

    pub fn get_middle() -> u256 {
        return record.mid.leaf.middle;
    }

    pub fn get_right() -> u256 {
        return record.mid.leaf.right;
    }

    pub fn get_after() -> u256 {
        return record.mid.after;
    }

    pub fn get_tail() -> u256 {
        return record.tail;
    }
}
ORA
}

write_slice_deep_nested_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Leaf {
    left: u256;
    middle: u256;
    right: u256;
}

struct Mid {
    before: u256;
    leaf: Leaf;
    after: u256;
}

struct Outer {
    head: u256;
    mid: Mid;
    tail: u256;
}

contract SliceDeepNestedStructStorageSmoke {
    storage var records: slice[Outer];

    pub fn set(values: slice[Outer]) {
        records = values;
    }

    pub fn set_leaf_middle(index: u256, value: u256) {
        records[index].mid.leaf.middle = value;
    }

    pub fn set_mid_after(index: u256, value: u256) {
        records[index].mid.after = value;
    }

    pub fn get_head(index: u256) -> u256 {
        return records[index].head;
    }

    pub fn get_before(index: u256) -> u256 {
        return records[index].mid.before;
    }

    pub fn get_left(index: u256) -> u256 {
        return records[index].mid.leaf.left;
    }

    pub fn get_middle(index: u256) -> u256 {
        return records[index].mid.leaf.middle;
    }

    pub fn get_right(index: u256) -> u256 {
        return records[index].mid.leaf.right;
    }

    pub fn get_after(index: u256) -> u256 {
        return records[index].mid.after;
    }

    pub fn get_tail(index: u256) -> u256 {
        return records[index].tail;
    }
}
ORA
}

write_deep_dynamic_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Leaf {
    left: u256;
    values: slice[u256];
    right: u256;
}

struct Mid {
    before: u256;
    leaf: Leaf;
    after: u256;
}

struct Outer {
    head: u256;
    mid: Mid;
    tail: u256;
}

contract DeepDynamicStructStorageSmoke {
    storage var record: Outer;

    pub fn set(head: u256, before: u256, left: u256, values: slice[u256], right: u256, after: u256, tail: u256) {
        record = Outer {
            head: head,
            mid: Mid {
                before: before,
                leaf: Leaf { left: left, values: values, right: right },
                after: after,
            },
            tail: tail,
        };
    }

    pub fn set_values(values: slice[u256]) {
        record.mid.leaf.values = values;
    }

    pub fn set_leaf_right(value: u256) {
        record.mid.leaf.right = value;
    }

    pub fn set_mid_after(value: u256) {
        record.mid.after = value;
    }

    pub fn get_head() -> u256 {
        return record.head;
    }

    pub fn get_before() -> u256 {
        return record.mid.before;
    }

    pub fn get_left() -> u256 {
        return record.mid.leaf.left;
    }

    pub fn get_value(index: u256) -> u256 {
        return record.mid.leaf.values[index];
    }

    pub fn get_right() -> u256 {
        return record.mid.leaf.right;
    }

    pub fn get_after() -> u256 {
        return record.mid.after;
    }

    pub fn get_tail() -> u256 {
        return record.tail;
    }
}
ORA
}

write_nested_dual_dynamic_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Duo {
    left: slice[u256];
    pivot: u256;
    right: slice[u256];
}

struct Outer {
    head: u256;
    duo: Duo;
    tail: u256;
}

contract NestedDualDynamicStructStorageSmoke {
    storage var record: Outer;

    pub fn set(head: u256, left: slice[u256], pivot: u256, right: slice[u256], tail: u256) {
        record = Outer {
            head: head,
            duo: Duo { left: left, pivot: pivot, right: right },
            tail: tail,
        };
    }

    pub fn set_left(values: slice[u256]) {
        record.duo.left = values;
    }

    pub fn set_pivot(value: u256) {
        record.duo.pivot = value;
    }

    pub fn set_right(values: slice[u256]) {
        record.duo.right = values;
    }

    pub fn set_tail(value: u256) {
        record.tail = value;
    }

    pub fn get_head() -> u256 {
        return record.head;
    }

    pub fn get_left(index: u256) -> u256 {
        return record.duo.left[index];
    }

    pub fn get_pivot() -> u256 {
        return record.duo.pivot;
    }

    pub fn get_right(index: u256) -> u256 {
        return record.duo.right[index];
    }

    pub fn get_tail() -> u256 {
        return record.tail;
    }
}
ORA
}

write_map_nested_dual_dynamic_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Duo {
    left: slice[u256];
    pivot: u256;
    right: slice[u256];
}

struct Outer {
    head: u256;
    duo: Duo;
    tail: u256;
}

contract MapNestedDualDynamicStructStorageSmoke {
    storage var records: map<address, Outer>;

    pub fn set(account: address, head: u256, left: slice[u256], pivot: u256, right: slice[u256], tail: u256) {
        records[account] = Outer {
            head: head,
            duo: Duo { left: left, pivot: pivot, right: right },
            tail: tail,
        };
    }

    pub fn set_left(account: address, values: slice[u256]) {
        records[account].duo.left = values;
    }

    pub fn set_pivot(account: address, value: u256) {
        records[account].duo.pivot = value;
    }

    pub fn set_right(account: address, values: slice[u256]) {
        records[account].duo.right = values;
    }

    pub fn set_tail(account: address, value: u256) {
        records[account].tail = value;
    }

    pub fn get_head(account: address) -> u256 {
        return records[account].head;
    }

    pub fn get_left(account: address, index: u256) -> u256 {
        return records[account].duo.left[index];
    }

    pub fn get_pivot(account: address) -> u256 {
        return records[account].duo.pivot;
    }

    pub fn get_right(account: address, index: u256) -> u256 {
        return records[account].duo.right[index];
    }

    pub fn get_tail(account: address) -> u256 {
        return records[account].tail;
    }
}
ORA
}

write_map_deep_nested_struct_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Leaf {
    left: u256;
    middle: u256;
    right: u256;
}

struct Mid {
    before: u256;
    leaf: Leaf;
    after: u256;
}

struct Outer {
    head: u256;
    mid: Mid;
    tail: u256;
}

contract MapDeepNestedStructStorageSmoke {
    storage var accounts: map<address, Outer>;

    pub fn set(account: address, head: u256, before: u256, left: u256, middle: u256, right: u256, after: u256, tail: u256) {
        accounts[account] = Outer {
            head: head,
            mid: Mid {
                before: before,
                leaf: Leaf { left: left, middle: middle, right: right },
                after: after,
            },
            tail: tail,
        };
    }

    pub fn set_leaf_middle(account: address, value: u256) {
        accounts[account].mid.leaf.middle = value;
    }

    pub fn set_mid_after(account: address, value: u256) {
        accounts[account].mid.after = value;
    }

    pub fn get_head(account: address) -> u256 {
        return accounts[account].head;
    }

    pub fn get_before(account: address) -> u256 {
        return accounts[account].mid.before;
    }

    pub fn get_left(account: address) -> u256 {
        return accounts[account].mid.leaf.left;
    }

    pub fn get_middle(account: address) -> u256 {
        return accounts[account].mid.leaf.middle;
    }

    pub fn get_right(account: address) -> u256 {
        return accounts[account].mid.leaf.right;
    }

    pub fn get_after(account: address) -> u256 {
        return accounts[account].mid.after;
    }

    pub fn get_tail(account: address) -> u256 {
        return accounts[account].tail;
    }
}
ORA
}

write_high_arity_abi_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
contract HighArityAbiSmoke {
    storage var who: address;
    storage var a1: u256;
    storage var a2: u256;
    storage var a3: u256;
    storage var a4: u256;
    storage var a5: u256;
    storage var a6: u256;
    storage var a7: u256;

    pub fn set6(account: address, one: u256, two: u256, three: u256, four: u256, five: u256) {
        who = account;
        a1 = one;
        a2 = two;
        a3 = three;
        a4 = four;
        a5 = five;
    }

    pub fn set8(account: address, one: u256, two: u256, three: u256, four: u256, five: u256, six: u256, seven: u256) {
        who = account;
        a1 = one;
        a2 = two;
        a3 = three;
        a4 = four;
        a5 = five;
        a6 = six;
        a7 = seven;
    }
}
ORA
}

write_multicontract_callee_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Snapshot {
    current: u256,
    tag: u256,
}

error ExternalCallFailed;

contract ExternalCallTargetSmoke {
    storage var value: u256;

    pub fn set_value(next: u256) -> bool {
        value = next;
        return true;
    }

    pub fn get_value() -> u256 {
        return value;
    }

    pub fn snapshot() -> Snapshot {
        return Snapshot { current: value, tag: 1234 };
    }

    pub fn set_and_snapshot(next: u256) -> Snapshot {
        value = next;
        return Snapshot { current: value, tag: 5678 };
    }

    pub fn always_fail() -> !u256 | ExternalCallFailed {
        return ExternalCallFailed;
    }
}
ORA
}

write_multicontract_caller_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
extern trait Target {
    call fn set_value(self, next: u256) -> bool;
    call fn get_value(self) -> u256;
}

error ExternalCallFailed;

contract ExternalCallCallerSmoke {
    storage var local: u256;
    storage var observed: u256;

    pub fn call_target(target: address, next: u256) -> !bool | ExternalCallFailed {
        return external<Target>(target, gas: 50000).set_value(next);
    }

    pub fn pull_target(target: address) -> bool {
        try {
            let value: u256 = try external<Target>(target, gas: 50000).get_value();
            observed = value;
            return true;
        } catch {
            return false;
        }
    }

    pub fn mark_local(next: u256) {
        local = next;
    }

    pub fn get_local() -> u256 {
        return local;
    }

    pub fn get_observed() -> u256 {
        return observed;
    }
}
ORA
}

write_multicontract_snapshot_caller_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
struct Snapshot {
    current: u256,
    tag: u256,
}

extern trait TargetSnapshot {
    call fn snapshot(self) -> Snapshot;
    call fn set_and_snapshot(self, next: u256) -> Snapshot;
    call fn always_fail(self) -> u256;
}

error ExternalCallFailed;

contract ExternalCallSnapshotCallerSmoke {
    storage var observed_current: u256;
    storage var observed_tag: u256;

    pub fn pull_snapshot(target: address) -> bool {
        try {
            let snapshot: Snapshot = try external<TargetSnapshot>(target, gas: 50000).snapshot();
            observed_current = snapshot.current;
            observed_tag = snapshot.tag;
            return true;
        } catch {
            return false;
        }
    }

    pub fn push_snapshot(target: address, next: u256) -> bool {
        try {
            let snapshot: Snapshot = try external<TargetSnapshot>(target, gas: 100000).set_and_snapshot(next);
            observed_current = snapshot.current;
            observed_tag = snapshot.tag;
            return true;
        } catch {
            return false;
        }
    }

    pub fn catch_failure(target: address) -> bool {
        try {
            let value: u256 = try external<TargetSnapshot>(target, gas: 50000).always_fail();
            observed_current = value;
            observed_tag = 9090;
            return true;
        } catch {
            observed_tag = 4242;
            return false;
        }
    }

    pub fn bubble_failure(target: address) -> !u256 | ExternalCallFailed {
        observed_current = 1111;
        return external<TargetSnapshot>(target, gas: 50000).always_fail();
    }

    pub fn catch_oog(target: address) -> bool {
        try {
            let snapshot: Snapshot = try external<TargetSnapshot>(target, gas: 1).snapshot();
            observed_current = snapshot.current;
            observed_tag = snapshot.tag;
            return true;
        } catch {
            observed_tag = 5151;
            return false;
        }
    }

    pub fn bubble_oog(target: address) -> !u256 | ExternalCallFailed {
        observed_current = 2222;
        return external<TargetSnapshot>(target, gas: 1).always_fail();
    }

    pub fn get_observed_current() -> u256 {
        return observed_current;
    }

    pub fn get_observed_tag() -> u256 {
        return observed_tag;
    }
}
ORA
}

write_multicontract_chain_leaf_fixture() {
  local source="$1"

  cat >"$source" <<'ORA'
error ExternalCallFailed;

contract ExternalCallChainLeafSmoke {
    storage var value: u256;

    pub fn set_and_get(next: u256) -> u256 {
        value = next;
        return value;
    }

    pub fn always_fail() -> !u256 | ExternalCallFailed {
        return ExternalCallFailed;
    }

    pub fn get_value() -> u256 {
        return value;
    }
}
ORA
}

write_multicontract_chain_middle_fixture() {
  local source="$1"

cat >"$source" <<'ORA'
extern trait ChainLeaf {
    call fn set_and_get(self, next: u256) -> u256;
    call fn always_fail(self) -> u256;
}

error ExternalCallFailed;

contract ExternalCallChainMiddleSmoke {
    storage var local: u256;
    storage var observed_leaf: u256;

    pub fn call_leaf(leaf: address, next: u256) -> u256 {
        local = next + 2;
        try {
            let leaf_value: u256 = try external<ChainLeaf>(leaf, gas: 50000).set_and_get(next);
            observed_leaf = leaf_value;
            return leaf_value + 10;
        } catch {
            observed_leaf = 5555;
            return 0;
        }
    }

    pub fn catch_leaf_failure(leaf: address, next: u256) -> u256 {
        local = next + 20;
        try {
            let leaf_value: u256 = try external<ChainLeaf>(leaf, gas: 50000).always_fail();
            observed_leaf = leaf_value;
            return 9090;
        } catch {
            observed_leaf = 7777;
            return 8888;
        }
    }

    pub fn bubble_leaf_failure(leaf: address, next: u256) -> !u256 | ExternalCallFailed {
        local = next + 40;
        return external<ChainLeaf>(leaf, gas: 50000).always_fail();
    }

    pub fn get_local() -> u256 {
        return local;
    }

    pub fn get_observed_leaf() -> u256 {
        return observed_leaf;
    }
}
ORA
}

write_multicontract_chain_root_fixture() {
  local source="$1"

cat >"$source" <<'ORA'
extern trait ChainMiddle {
    call fn call_leaf(self, leaf: address, next: u256) -> u256;
    call fn catch_leaf_failure(self, leaf: address, next: u256) -> u256;
    call fn bubble_leaf_failure(self, leaf: address, next: u256) -> u256;
}

error ExternalCallFailed;

contract ExternalCallChainRootSmoke {
    storage var local: u256;
    storage var observed_middle: u256;

    pub fn call_chain(middle: address, leaf: address, next: u256) -> bool {
        local = next + 3;
        try {
            let middle_value: u256 = try external<ChainMiddle>(middle, gas: 100000).call_leaf(leaf, next);
            observed_middle = middle_value;
            return true;
        } catch {
            observed_middle = 6666;
            return false;
        }
    }

    pub fn call_chain_catch(middle: address, leaf: address, next: u256) -> bool {
        local = next + 30;
        try {
            let middle_value: u256 = try external<ChainMiddle>(middle, gas: 100000).catch_leaf_failure(leaf, next);
            observed_middle = middle_value;
            return true;
        } catch {
            observed_middle = 6666;
            return false;
        }
    }

    pub fn bubble_chain_failure(middle: address, leaf: address, next: u256) -> !bool | ExternalCallFailed {
        local = next + 50;
        let middle_value: u256 = try external<ChainMiddle>(middle, gas: 100000).bubble_leaf_failure(leaf, next);
        observed_middle = middle_value;
        return true;
    }

    pub fn get_local() -> u256 {
        return local;
    }

    pub fn get_observed_middle() -> u256 {
        return observed_middle;
    }
}
ORA
}

write_lock_rollback_target_fixture() {
  local source="$1"

cat >"$source" <<'ORA'
contract LockRollbackTargetSmoke {
    storage var value: u256;
    storage var marker: u256;

    fn write_value(next: u256) {
        value = next;
    }

    pub fn set_value(next: u256) -> bool {
        write_value(next);
        return true;
    }

    pub fn lock_then_guarded_write(next: u256) -> bool {
        @lock(value);
        write_value(next);
        return true;
    }

    pub fn lock_unlock_then_write(next: u256) -> bool {
        @lock(value);
        marker = 3001;
        @unlock(value);
        write_value(next);
        return true;
    }

    pub fn get_value() -> u256 {
        return value;
    }

    pub fn get_marker() -> u256 {
        return marker;
    }
}
ORA
}

write_lock_rollback_observer_fixture() {
  local source="$1"

cat >"$source" <<'ORA'
extern trait LockRollbackTarget {
    call fn lock_then_guarded_write(self, next: u256) -> bool;
    call fn set_value(self, next: u256) -> bool;
}

error ExternalCallFailed;

contract LockRollbackObserverSmoke {
    storage var marker: u256;

    pub fn catch_revert_then_set(target: address, next: u256) -> bool {
        try {
            let locked_write_ok: bool = try external<LockRollbackTarget>(target, gas: 100000).lock_then_guarded_write(next);
            if (locked_write_ok) {
                return false;
            } else {
                return false;
            }
        } catch {
            // The first call reverted while its callee frame held the transient
            // lock. The second call must still succeed in this same transaction.
        }

        try {
            let set_ok: bool = try external<LockRollbackTarget>(target, gas: 100000).set_value(next + 1);
            marker = 31;
            return set_ok;
        } catch {
            marker = 41;
            return false;
        }
    }

    pub fn get_marker() -> u256 {
        return marker;
    }
}
ORA
}

write_reentrant_lock_target_fixture() {
  local source="$1"

cat >"$source" <<'ORA'
extern trait ReentrantLockObserver {
    call fn attempt_reenter(self, target: address, next: u256) -> u256;
}

error ExternalCallFailed;

contract ReentrantLockTargetSmoke {
    storage var value: u256;
    storage var marker: u256;

    fn write_value(next: u256) {
        value = next;
    }

    pub fn reentrant_write(next: u256) -> bool {
        write_value(next);
        marker = 7000;
        return true;
    }

    pub fn lock_call_observer(observer: address, target: address, next: u256) -> bool {
        @lock(value);
        try {
            let observed: u256 = try external<ReentrantLockObserver>(observer, gas: 200000).attempt_reenter(target, next + 1);
            marker = observed;
            @unlock(value);
            write_value(next + 2);
            return true;
        } catch {
            @unlock(value);
            marker = 9001;
            return false;
        }
    }

    pub fn get_value() -> u256 {
        return value;
    }

    pub fn get_marker() -> u256 {
        return marker;
    }
}
ORA
}

write_reentrant_lock_observer_fixture() {
  local source="$1"

cat >"$source" <<'ORA'
extern trait ReentrantLockTarget {
    call fn reentrant_write(self, next: u256) -> bool;
}

error ExternalCallFailed;

contract ReentrantLockObserverSmoke {
    storage var marker: u256;

    pub fn attempt_reenter(target: address, next: u256) -> u256 {
        try {
            let ok: bool = try external<ReentrantLockTarget>(target, gas: 100000).reentrant_write(next);
            if (ok) {
                marker = 99;
                return 99;
            } else {
                marker = 98;
                return 98;
            }
        } catch {
            // 31 is the sentinel for "the target's runtime lock blocked the
            // reentrant callback and the observer caught that revert."
            marker = 31;
            return 31;
        }
    }

    pub fn get_marker() -> u256 {
        return marker;
    }
}
ORA
}

assert_contract_raw_slot() {
  local addr="$1"
  local slot="$2"
  local expected="$3"
  local label="$4"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$addr" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "$label slot $slot mismatch at $addr: expected=$slot_expected got=$slot_raw"

  ok "$label slot $slot at $addr equals $expected"
}

assert_multicontract_callee() {
  local addr="$1"
  local expected="$2"
  local getter_raw getter

  assert_contract_raw_slot "$addr" 0 "$expected" "external-call callee"
  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_value()(uint256)")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "external-call callee getter mismatch: expected=$expected got=$getter"

  ok "external-call callee getter equals $expected"
}

assert_multicontract_caller() {
  local addr="$1"
  local expected="$2"
  local getter_raw getter

  assert_contract_raw_slot "$addr" 0 "$expected" "external-call caller"
  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_local()(uint256)")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "external-call caller getter mismatch: expected=$expected got=$getter"

  ok "external-call caller getter equals $expected"
}

assert_multicontract_caller_observed() {
  local addr="$1"
  local expected="$2"
  local getter_raw getter

  assert_contract_raw_slot "$addr" 1 "$expected" "external-call caller observed"
  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_observed()(uint256)")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "external-call caller observed getter mismatch: expected=$expected got=$getter"

  ok "external-call caller observed getter equals $expected"
}

assert_multicontract_caller_snapshot() {
  local addr="$1"
  local expected_current="$2"
  local expected_tag="$3"
  local current_raw current tag_raw tag

  assert_contract_raw_slot "$addr" 0 "$expected_current" "external-call snapshot caller observed_current"
  assert_contract_raw_slot "$addr" 1 "$expected_tag" "external-call snapshot caller observed_tag"
  current_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_observed_current()(uint256)")"
  current="$(normalize_uint "$current_raw")"
  [[ "$current" == "$expected_current" ]] || fail "external-call caller observed_current getter mismatch: expected=$expected_current got=$current"

  tag_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_observed_tag()(uint256)")"
  tag="$(normalize_uint "$tag_raw")"
  [[ "$tag" == "$expected_tag" ]] || fail "external-call caller observed_tag getter mismatch: expected=$expected_tag got=$tag"

  ok "external-call caller structured observed getters equal [current=$expected_current, tag=$expected_tag]"
}

assert_multicontract_snapshot_try_success() {
  local caller_addr="$1"
  local target_addr="$2"
  local result_raw result

  result_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$caller_addr" "pull_snapshot(address)(bool)" "$target_addr")"
  result="$(normalize_bool "$result_raw")"
  [[ "$result" == "true" ]] || fail "external-call snapshot try/catch returned $result"

  ok "external-call snapshot try/catch success path returns true"
}

assert_multicontract_push_snapshot_try_success() {
  local caller_addr="$1"
  local target_addr="$2"
  local value="$3"
  local result_raw result

  result_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$caller_addr" "push_snapshot(address,uint256)(bool)" "$target_addr" "$value")"
  result="$(normalize_bool "$result_raw")"
  [[ "$result" == "true" ]] || fail "external-call mutating snapshot try/catch returned $result"

  ok "external-call mutating snapshot try/catch success path returns true"
}

assert_multicontract_failure_try_catch_returns_false() {
  local caller_addr="$1"
  local target_addr="$2"
  local result_raw result

  # eth_call pins the catch branch's return value; the later transaction pins
  # the same catch path's committed storage effects.
  result_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$caller_addr" "catch_failure(address)(bool)" "$target_addr")"
  result="$(normalize_bool "$result_raw")"
  [[ "$result" == "false" ]] || fail "external-call failure try/catch returned $result"

  ok "external-call failure try/catch path returns false"
}

send_multicontract_push_snapshot() {
  local caller_addr="$1"
  local target_addr="$2"
  local value="$3"

  send_contract_tx "$caller_addr" "external-call caller push_snapshot" "push_snapshot(address,uint256)" "$target_addr" "$value"
}

send_multicontract_catch_failure() {
  local caller_addr="$1"
  local target_addr="$2"

  send_contract_tx "$caller_addr" "external-call caller catch_failure" "catch_failure(address)" "$target_addr"
}

assert_multicontract_oog_try_catch_returns_false() {
  local caller_addr="$1"
  local target_addr="$2"
  local result_raw result

  result_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$caller_addr" "catch_oog(address)(bool)" "$target_addr")"
  result="$(normalize_bool "$result_raw")"
  [[ "$result" == "false" ]] || fail "external-call OOG try/catch returned $result"

  ok "external-call OOG try/catch path returns false"
}

send_multicontract_catch_oog() {
  local caller_addr="$1"
  local target_addr="$2"

  send_contract_tx "$caller_addr" "external-call caller catch_oog" "catch_oog(address)" "$target_addr"
}

send_multicontract_bubble_failure_expect_revert() {
  local caller_addr="$1"
  local target_addr="$2"

  send_contract_tx_expect_revert "$caller_addr" "external-call caller bubble_failure" "bubble_failure(address)" "$target_addr"
}

send_multicontract_bubble_oog_expect_revert() {
  local caller_addr="$1"
  local target_addr="$2"

  send_contract_tx_expect_revert "$caller_addr" "external-call caller bubble_oog" "bubble_oog(address)" "$target_addr"
}

assert_multicontract_chain_leaf() {
  local addr="$1"
  local expected="$2"
  local getter_raw getter

  assert_contract_raw_slot "$addr" 0 "$expected" "external-call chain leaf"
  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_value()(uint256)")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "external-call chain leaf getter mismatch: expected=$expected got=$getter"

  ok "external-call chain leaf getter equals $expected"
}

assert_multicontract_chain_middle() {
  local addr="$1"
  local expected_local="$2"
  local expected_observed="$3"
  local local_raw local_value observed_raw observed

  assert_contract_raw_slot "$addr" 0 "$expected_local" "external-call chain middle local"
  assert_contract_raw_slot "$addr" 1 "$expected_observed" "external-call chain middle observed_leaf"
  local_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_local()(uint256)")"
  local_value="$(normalize_uint "$local_raw")"
  [[ "$local_value" == "$expected_local" ]] || fail "external-call chain middle local getter mismatch: expected=$expected_local got=$local_value"

  observed_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_observed_leaf()(uint256)")"
  observed="$(normalize_uint "$observed_raw")"
  [[ "$observed" == "$expected_observed" ]] || fail "external-call chain middle observed getter mismatch: expected=$expected_observed got=$observed"

  ok "external-call chain middle getters equal [local=$expected_local, observed_leaf=$expected_observed]"
}

assert_multicontract_chain_root() {
  local addr="$1"
  local expected_local="$2"
  local expected_observed="$3"
  local local_raw local_value observed_raw observed

  assert_contract_raw_slot "$addr" 0 "$expected_local" "external-call chain root local"
  assert_contract_raw_slot "$addr" 1 "$expected_observed" "external-call chain root observed_middle"
  local_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_local()(uint256)")"
  local_value="$(normalize_uint "$local_raw")"
  [[ "$local_value" == "$expected_local" ]] || fail "external-call chain root local getter mismatch: expected=$expected_local got=$local_value"

  observed_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_observed_middle()(uint256)")"
  observed="$(normalize_uint "$observed_raw")"
  [[ "$observed" == "$expected_observed" ]] || fail "external-call chain root observed getter mismatch: expected=$expected_observed got=$observed"

  ok "external-call chain root getters equal [local=$expected_local, observed_middle=$expected_observed]"
}

send_multicontract_chain_call() {
  local root_addr="$1"
  local middle_addr="$2"
  local leaf_addr="$3"
  local value="$4"

  send_contract_tx "$root_addr" "external-call chain call_chain" "call_chain(address,address,uint256)" "$middle_addr" "$leaf_addr" "$value"
}

send_multicontract_chain_catch_call() {
  local root_addr="$1"
  local middle_addr="$2"
  local leaf_addr="$3"
  local value="$4"

  send_contract_tx "$root_addr" "external-call chain call_chain_catch" "call_chain_catch(address,address,uint256)" "$middle_addr" "$leaf_addr" "$value"
}

send_multicontract_chain_bubble_failure_expect_revert() {
  local root_addr="$1"
  local middle_addr="$2"
  local leaf_addr="$3"
  local value="$4"

  send_contract_tx_expect_revert "$root_addr" "external-call chain bubble_chain_failure" "bubble_chain_failure(address,address,uint256)" "$middle_addr" "$leaf_addr" "$value"
}

assert_lock_rollback_target() {
  local addr="$1"
  local expected_value="$2"
  local expected_marker="$3"
  local value_raw value marker_raw marker

  assert_contract_raw_slot "$addr" 0 "$expected_value" "lock rollback target value"
  assert_contract_raw_slot "$addr" 1 "$expected_marker" "lock rollback target marker"
  value_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_value()(uint256)")"
  value="$(normalize_uint "$value_raw")"
  [[ "$value" == "$expected_value" ]] || fail "lock rollback target value getter mismatch: expected=$expected_value got=$value"

  marker_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_marker()(uint256)")"
  marker="$(normalize_uint "$marker_raw")"
  [[ "$marker" == "$expected_marker" ]] || fail "lock rollback target marker getter mismatch: expected=$expected_marker got=$marker"

  ok "lock rollback target getters equal [value=$expected_value, marker=$expected_marker]"
}

assert_lock_rollback_observer() {
  local addr="$1"
  local expected_marker="$2"
  local marker_raw marker

  assert_contract_raw_slot "$addr" 0 "$expected_marker" "lock rollback observer marker"
  marker_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_marker()(uint256)")"
  marker="$(normalize_uint "$marker_raw")"
  [[ "$marker" == "$expected_marker" ]] || fail "lock rollback observer marker getter mismatch: expected=$expected_marker got=$marker"

  ok "lock rollback observer marker getter equals $expected_marker"
}

send_lock_rollback_catch_revert_then_set() {
  local observer_addr="$1"
  local target_addr="$2"
  local value="$3"

  send_contract_tx "$observer_addr" "lock rollback catch_revert_then_set" "catch_revert_then_set(address,uint256)" "$target_addr" "$value"
}

send_lock_rollback_guarded_write_expect_revert() {
  local target_addr="$1"
  local value="$2"

  send_contract_tx_expect_revert "$target_addr" "lock rollback guarded write" "lock_then_guarded_write(uint256)" "$value"
}

send_lock_rollback_unlock_then_write() {
  local target_addr="$1"
  local value="$2"

  send_contract_tx "$target_addr" "lock rollback unlock then write" "lock_unlock_then_write(uint256)" "$value"
}

assert_reentrant_lock_target() {
  local addr="$1"
  local expected_value="$2"
  local expected_marker="$3"
  local value_raw value marker_raw marker

  assert_contract_raw_slot "$addr" 0 "$expected_value" "reentrant lock target value"
  assert_contract_raw_slot "$addr" 1 "$expected_marker" "reentrant lock target marker"

  value_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_value()(uint256)")"
  value="$(normalize_uint "$value_raw")"
  [[ "$value" == "$expected_value" ]] || fail "reentrant lock target value getter mismatch: expected=$expected_value got=$value"

  marker_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_marker()(uint256)")"
  marker="$(normalize_uint "$marker_raw")"
  [[ "$marker" == "$expected_marker" ]] || fail "reentrant lock target marker getter mismatch: expected=$expected_marker got=$marker"

  ok "reentrant lock target getters equal [value=$expected_value, marker=$expected_marker]"
}

assert_reentrant_lock_observer() {
  local addr="$1"
  local expected_marker="$2"
  local marker_raw marker

  assert_contract_raw_slot "$addr" 0 "$expected_marker" "reentrant lock observer marker"
  marker_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$addr" "get_marker()(uint256)")"
  marker="$(normalize_uint "$marker_raw")"
  [[ "$marker" == "$expected_marker" ]] || fail "reentrant lock observer marker getter mismatch: expected=$expected_marker got=$marker"

  ok "reentrant lock observer marker getter equals $expected_marker"
}

send_reentrant_lock_call_observer() {
  local target_addr="$1"
  local observer_addr="$2"
  local value="$3"

  send_contract_tx "$target_addr" "reentrant lock call observer" "lock_call_observer(address,address,uint256)" "$observer_addr" "$target_addr" "$value"
}

send_reentrant_lock_direct_write() {
  local target_addr="$1"
  local value="$2"

  send_contract_tx "$target_addr" "reentrant lock direct write" "reentrant_write(uint256)" "$value"
}

assert_mapping_raw_slot() {
  local account="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(trim "$(cast index address "$account" 0)")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "mapping slot mismatch for $account at $slot: expected=$slot_expected got=$slot_raw"

  ok "raw mapping slot $slot for $account equals $expected"
}

assert_mapping_getter() {
  local account="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_balance(address)(uint256)" "$account")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "get_balance($account) mismatch: expected=$expected got=$getter"
  ok "get_balance($account) equals $expected"
}

map_u256_slot() {
  local key="$1"
  trim "$(cast index uint256 "$key" 0)"
}

map_integer_slot() {
  local abi_type="$1"
  local key="$2"
  local root_slot="$3"
  trim "$(cast index "$abi_type" -- "$key" "$root_slot")"
}

map_fixed_bytes_slot() {
  local abi_type="$1"
  local key="$2"
  local root_slot="$3"
  trim "$(cast index "$abi_type" "$key" "$root_slot")"
}

assert_u256_mapping_raw_slot() {
  local key="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(map_u256_slot "$key")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "u256 mapping slot mismatch for key $key at $slot: expected=$slot_expected got=$slot_raw"

  ok "raw u256 mapping slot $slot for key $key equals $expected"
}

assert_u256_mapping_getter() {
  local key="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_score(uint256)(uint256)" "$key")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "get_score($key) mismatch: expected=$expected got=$getter"
  ok "get_score($key) equals $expected"
}

assert_integer_mapping_raw_slot() {
  local label="$1"
  local abi_type="$2"
  local key="$3"
  local root_slot="$4"
  local expected="$5"
  local slot slot_raw slot_expected

  slot="$(map_integer_slot "$abi_type" "$key" "$root_slot")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "$label mapping slot mismatch for key $key at $slot: expected=$slot_expected got=$slot_raw"

  ok "raw $label mapping slot $slot for key $key equals $expected"
}

assert_integer_mapping_getter() {
  local label="$1"
  local abi_type="$2"
  local getter_name="$3"
  local key="$4"
  local expected="$5"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$getter_name($abi_type)(uint256)" -- "$key")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "$getter_name($key) mismatch: expected=$expected got=$getter"
  ok "$getter_name($key) equals $expected"
}

assert_fixed_bytes_mapping_raw_slot() {
  local label="$1"
  local abi_type="$2"
  local key="$3"
  local root_slot="$4"
  local expected="$5"
  local slot slot_raw slot_expected

  slot="$(map_fixed_bytes_slot "$abi_type" "$key" "$root_slot")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "$label mapping slot mismatch for key $key at $slot: expected=$slot_expected got=$slot_raw"

  ok "raw $label mapping slot $slot for key $key equals $expected"
}

assert_fixed_bytes_mapping_getter() {
  local label="$1"
  local abi_type="$2"
  local getter_name="$3"
  local key="$4"
  local expected="$5"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$getter_name($abi_type)(uint256)" "$key")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "$getter_name($key) mismatch: expected=$expected got=$getter"
  ok "$getter_name($key) equals $expected"
}

assert_nested_mapping_raw_slot() {
  local owner="$1"
  local spender="$2"
  local expected="$3"
  local inner_slot slot slot_raw slot_expected

  inner_slot="$(trim "$(cast index address "$owner" 0)")"
  slot="$(trim "$(cast index address "$spender" "$inner_slot")")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "nested mapping slot mismatch for $owner/$spender at $slot: expected=$slot_expected got=$slot_raw"

  ok "raw nested mapping slot $slot for $owner/$spender equals $expected"
}

assert_nested_mapping_getter() {
  local owner="$1"
  local spender="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_allowance(address,address)(uint256)" "$owner" "$spender")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "get_allowance($owner,$spender) mismatch: expected=$expected got=$getter"
  ok "get_allowance($owner,$spender) equals $expected"
}

triple_nested_mapping_slot() {
  local owner="$1"
  local spender="$2"
  local asset="$3"
  local owner_slot spender_slot

  owner_slot="$(trim "$(cast index address "$owner" 0)")"
  spender_slot="$(trim "$(cast index address "$spender" "$owner_slot")")"
  trim "$(cast index address "$asset" "$spender_slot")"
}

assert_triple_nested_mapping_raw_slot() {
  local owner="$1"
  local spender="$2"
  local asset="$3"
  local expected="$4"
  local slot slot_raw slot_expected

  slot="$(triple_nested_mapping_slot "$owner" "$spender" "$asset")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "triple nested mapping slot mismatch for $owner/$spender/$asset at $slot: expected=$slot_expected got=$slot_raw"

  ok "raw triple nested mapping slot $slot for $owner/$spender/$asset equals $expected"
}

assert_triple_nested_mapping_getter() {
  local owner="$1"
  local spender="$2"
  local asset="$3"
  local expected="$4"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_limit(address,address,address)(uint256)" "$owner" "$spender" "$asset")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "get_limit($owner,$spender,$asset) mismatch: expected=$expected got=$getter"
  ok "get_limit($owner,$spender,$asset) equals $expected"
}

nested_map_struct_base_slot() {
  local owner="$1"
  local spender="$2"
  local inner_slot

  inner_slot="$(trim "$(cast index address "$owner" 0)")"
  trim "$(cast index address "$spender" "$inner_slot")"
}

assert_nested_map_struct_raw_slots() {
  local owner="$1"
  local spender="$2"
  local amount="$3"
  local nonce="$4"
  local flags="$5"
  local base amount_slot nonce_slot flags_slot amount_raw nonce_raw flags_raw

  base="$(nested_map_struct_base_slot "$owner" "$spender")"
  amount_slot="$base"
  nonce_slot="$(slot_add_small "$base" 1)"
  flags_slot="$(slot_add_small "$base" 2)"
  amount_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$amount_slot")")"
  nonce_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$nonce_slot")")"
  flags_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$flags_slot")")"

  [[ "${amount_raw,,}" == "$(expected_slot_hex "$amount")" ]] || fail "nested map struct amount mismatch for $owner/$spender at $amount_slot: expected=$amount got=$amount_raw"
  [[ "${nonce_raw,,}" == "$(expected_slot_hex "$nonce")" ]] || fail "nested map struct nonce mismatch for $owner/$spender at $nonce_slot: expected=$nonce got=$nonce_raw"
  [[ "${flags_raw,,}" == "$(expected_slot_hex "$flags")" ]] || fail "nested map struct flags mismatch for $owner/$spender at $flags_slot: expected=$flags got=$flags_raw"

  ok "raw nested map struct slots for $owner/$spender equal [amount=$amount, nonce=$nonce, flags=$flags]"
}

assert_nested_map_struct_getters() {
  local owner="$1"
  local spender="$2"
  local amount="$3"
  local nonce="$4"
  local flags="$5"
  local got_amount got_nonce got_flags

  got_amount="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_amount(address,address)(uint256)" "$owner" "$spender")")"
  got_nonce="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_nonce(address,address)(uint256)" "$owner" "$spender")")"
  got_flags="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_flags(address,address)(uint256)" "$owner" "$spender")")"
  [[ "$got_amount" == "$amount" ]] || fail "nested map struct get_amount($owner,$spender) mismatch: expected=$amount got=$got_amount"
  [[ "$got_nonce" == "$nonce" ]] || fail "nested map struct get_nonce($owner,$spender) mismatch: expected=$nonce got=$got_nonce"
  [[ "$got_flags" == "$flags" ]] || fail "nested map struct get_flags($owner,$spender) mismatch: expected=$flags got=$got_flags"

  ok "nested map struct getters for $owner/$spender equal [amount=$amount, nonce=$nonce, flags=$flags]"
}

slot_add_small() {
  local slot="$1"
  local offset="$2"
  python3 -c 'import sys; print("0x%064x" % ((int(sys.argv[1], 16) + int(sys.argv[2])) % (1 << 256)))' "$slot" "$offset"
}

dynamic_array_base_slot() {
  echo "0x290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563"
}

dynamic_array_base_slot_for_root() {
  local root="$1"
  case "$root" in
    0)
      dynamic_array_base_slot
      ;;
    1)
      echo "0xb10e2d527612073b26eecdfd717e6a320cf44b4afac2b0732d9fcbe2b7fa0cf6"
      ;;
    *)
      cast keccak "$(expected_slot_hex "$root")" | tr -d '[:space:]'
      ;;
  esac
}

dynamic_array_base_slot_for_slot() {
  local slot="$1"
  if [[ "$slot" == 0x* ]]; then
    cast keccak "$slot" | tr -d '[:space:]'
  else
    dynamic_array_base_slot_for_root "$slot"
  fi
}

dynamic_array_slot() {
  local index="$1"
  slot_add_small "$(dynamic_array_base_slot)" "$index"
}

dynamic_array_slot_for_root() {
  local root="$1"
  local index="$2"
  slot_add_small "$(dynamic_array_base_slot_for_root "$root")" "$index"
}

slice_slice_row_slot() {
  local row="$1"
  dynamic_array_slot_for_root 0 "$row"
}

slice_slice_value_slot() {
  local row="$1"
  local col="$2"
  local row_slot row_base

  row_slot="$(slice_slice_row_slot "$row")"
  row_base="$(dynamic_array_base_slot_for_slot "$row_slot")"
  slot_add_small "$row_base" "$col"
}

slice_dynamic_struct_base_slot() {
  local index="$1"
  local offset=$((index * 3))

  dynamic_array_slot "$offset"
}

slice_dynamic_struct_values_root_slot() {
  local index="$1"
  local base

  base="$(slice_dynamic_struct_base_slot "$index")"
  slot_add_small "$base" 1
}

slice_dynamic_struct_value_slot() {
  local index="$1"
  local col="$2"
  local root base

  root="$(slice_dynamic_struct_values_root_slot "$index")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$col"
}

slice_nested_dual_dynamic_struct_base_slot() {
  local index="$1"
  local offset=$((index * 5))

  dynamic_array_slot "$offset"
}

slice_nested_dual_dynamic_struct_left_root_slot() {
  local index="$1"
  local base

  base="$(slice_nested_dual_dynamic_struct_base_slot "$index")"
  slot_add_small "$base" 1
}

slice_nested_dual_dynamic_struct_right_root_slot() {
  local index="$1"
  local base

  base="$(slice_nested_dual_dynamic_struct_base_slot "$index")"
  slot_add_small "$base" 3
}

slice_nested_dual_dynamic_struct_left_slot() {
  local index="$1"
  local col="$2"
  local root base

  root="$(slice_nested_dual_dynamic_struct_left_root_slot "$index")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$col"
}

slice_nested_dual_dynamic_struct_right_slot() {
  local index="$1"
  local col="$2"
  local root base

  root="$(slice_nested_dual_dynamic_struct_right_root_slot "$index")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$col"
}

deep_dynamic_struct_values_root_slot() {
  echo "0x0000000000000000000000000000000000000000000000000000000000000003"
}

deep_dynamic_struct_value_slot() {
  local col="$1"
  local root base

  root="$(deep_dynamic_struct_values_root_slot)"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$col"
}

nested_dual_dynamic_struct_left_slot() {
  local index="$1"
  local base

  base="$(dynamic_array_base_slot_for_slot 1)"
  slot_add_small "$base" "$index"
}

nested_dual_dynamic_struct_right_slot() {
  local index="$1"
  local base

  base="$(dynamic_array_base_slot_for_slot 3)"
  slot_add_small "$base" "$index"
}

map_nested_dual_dynamic_struct_base_slot() {
  local account="$1"
  trim "$(cast index address "$account" 0)"
}

map_nested_dual_dynamic_struct_left_root_slot() {
  local account="$1"
  local base

  base="$(map_nested_dual_dynamic_struct_base_slot "$account")"
  slot_add_small "$base" 1
}

map_nested_dual_dynamic_struct_right_root_slot() {
  local account="$1"
  local base

  base="$(map_nested_dual_dynamic_struct_base_slot "$account")"
  slot_add_small "$base" 3
}

map_nested_dual_dynamic_struct_left_slot() {
  local account="$1"
  local index="$2"
  local root base

  root="$(map_nested_dual_dynamic_struct_left_root_slot "$account")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$index"
}

map_nested_dual_dynamic_struct_right_slot() {
  local account="$1"
  local index="$2"
  local root base

  root="$(map_nested_dual_dynamic_struct_right_root_slot "$account")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$index"
}

map_slice_root_slot() {
  local account="$1"
  trim "$(cast index address "$account" 0)"
}

map_slice_value_slot() {
  local account="$1"
  local index="$2"
  local root base

  root="$(map_slice_root_slot "$account")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$index"
}

map_dynamic_struct_base_slot() {
  local account="$1"
  trim "$(cast index address "$account" 0)"
}

map_dynamic_struct_values_root_slot() {
  local account="$1"
  local base

  base="$(map_dynamic_struct_base_slot "$account")"
  slot_add_small "$base" 1
}

map_dynamic_struct_value_slot() {
  local account="$1"
  local index="$2"
  local root base

  root="$(map_dynamic_struct_values_root_slot "$account")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$index"
}

map_dual_dynamic_struct_base_slot() {
  local account="$1"
  trim "$(cast index address "$account" 0)"
}

map_dual_dynamic_struct_left_root_slot() {
  local account="$1"
  map_dual_dynamic_struct_base_slot "$account"
}

map_dual_dynamic_struct_right_root_slot() {
  local account="$1"
  local base

  base="$(map_dual_dynamic_struct_base_slot "$account")"
  slot_add_small "$base" 2
}

map_dual_dynamic_struct_left_slot() {
  local account="$1"
  local index="$2"
  local root base

  root="$(map_dual_dynamic_struct_left_root_slot "$account")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$index"
}

map_dual_dynamic_struct_right_slot() {
  local account="$1"
  local index="$2"
  local root base

  root="$(map_dual_dynamic_struct_right_root_slot "$account")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$index"
}

nested_map_dynamic_struct_base_slot() {
  local owner="$1"
  local spender="$2"
  local owner_slot

  owner_slot="$(trim "$(cast index address "$owner" 0)")"
  trim "$(cast index address "$spender" "$owner_slot")"
}

nested_map_dynamic_struct_values_root_slot() {
  local owner="$1"
  local spender="$2"
  local base

  base="$(nested_map_dynamic_struct_base_slot "$owner" "$spender")"
  slot_add_small "$base" 1
}

nested_map_dynamic_struct_value_slot() {
  local owner="$1"
  local spender="$2"
  local index="$3"
  local root base

  root="$(nested_map_dynamic_struct_values_root_slot "$owner" "$spender")"
  base="$(dynamic_array_base_slot_for_slot "$root")"
  slot_add_small "$base" "$index"
}

bitfield_slot_hex() {
  local enabled="$1"
  local locked="$2"
  local mode="$3"
  local threshold="$4"

  python3 -c 'import sys
enabled = int(sys.argv[1])
locked = int(sys.argv[2])
mode = int(sys.argv[3])
threshold = int(sys.argv[4])
word = enabled | (locked << 1) | (mode << 2) | (threshold << 10)
print("0x%064x" % word)
	' "$enabled" "$locked" "$mode" "$threshold"
}

custom_bitfield_slot_hex() {
  local enabled="$1"
  local code="$2"
  local delta="$3"
  local amount="$4"

  python3 -c 'import sys
enabled = int(sys.argv[1]) & 1
code = int(sys.argv[2]) & ((1 << 5) - 1)
delta = int(sys.argv[3]) & ((1 << 16) - 1)
amount = int(sys.argv[4]) & ((1 << 32) - 1)
word = enabled | (code << 1) | (delta << 6) | (amount << 22)
print("0x%064x" % word)
	' "$enabled" "$code" "$delta" "$amount"
}

map_bitfield_slot() {
  local account="$1"
  trim "$(cast index address "$account" 0)"
}

slice_bitfield_slot() {
  local index="$1"
  dynamic_array_slot "$index"
}

normalize_bool() {
  local raw
  raw="$(trim "$1")"
  case "${raw,,}" in
    true|1|0x1|0x0000000000000000000000000000000000000000000000000000000000000001)
      echo "true"
      ;;
    false|0|0x0|0x0000000000000000000000000000000000000000000000000000000000000000)
      echo "false"
      ;;
    *)
      fail "cannot normalize bool output: $raw"
      ;;
  esac
}

assert_dynamic_array_length() {
  local expected="$1"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "dynamic array length slot mismatch: expected=$slot_expected got=$slot_raw"
  ok "dynamic array root slot length equals $expected"
}

assert_dynamic_array_raw_slot() {
  local index="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(dynamic_array_slot "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "dynamic array slot mismatch for index $index at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw dynamic array slot $slot for index $index equals $expected"
}

assert_dynamic_array_getter() {
  local index="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get(uint256)(uint256)" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "get($index) mismatch: expected=$expected got=$getter"
  ok "get($index) equals $expected"
}

assert_struct_raw_slots() {
  local left="$1"
  local middle="$2"
  local right="$3"
  local slot0 slot1 slot2

  slot0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  slot2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  [[ "${slot0,,}" == "$(expected_slot_hex "$left")" ]] || fail "struct slot 0 mismatch: expected=$left got=$slot0"
  [[ "${slot1,,}" == "$(expected_slot_hex "$middle")" ]] || fail "struct slot 1 mismatch: expected=$middle got=$slot1"
  [[ "${slot2,,}" == "$(expected_slot_hex "$right")" ]] || fail "struct slot 2 mismatch: expected=$right got=$slot2"

  ok "raw struct slots equal [$left, $middle, $right]"
}

assert_struct_getters() {
  local left="$1"
  local middle="$2"
  local right="$3"
  local got_left got_middle got_right

  got_left="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left()(uint256)")")"
  got_middle="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_middle()(uint256)")")"
  got_right="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right()(uint256)")")"
  [[ "$got_left" == "$left" ]] || fail "get_left() mismatch: expected=$left got=$got_left"
  [[ "$got_middle" == "$middle" ]] || fail "get_middle() mismatch: expected=$middle got=$got_middle"
  [[ "$got_right" == "$right" ]] || fail "get_right() mismatch: expected=$right got=$got_right"

  ok "struct getters equal [$left, $middle, $right]"
}

assert_map_struct_raw_slots() {
  local account="$1"
  local balance="$2"
  local nonce="$3"
  local flags="$4"
  local base slot0 slot1 slot2 raw0 raw1 raw2

  base="$(trim "$(cast index address "$account" 0)")"
  slot0="$base"
  slot1="$(slot_add_small "$base" 1)"
  slot2="$(slot_add_small "$base" 2)"
  raw0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot0")")"
  raw1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot1")")"
  raw2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot2")")"
  [[ "${raw0,,}" == "$(expected_slot_hex "$balance")" ]] || fail "map struct balance slot mismatch for $account at $slot0: expected=$balance got=$raw0"
  [[ "${raw1,,}" == "$(expected_slot_hex "$nonce")" ]] || fail "map struct nonce slot mismatch for $account at $slot1: expected=$nonce got=$raw1"
  [[ "${raw2,,}" == "$(expected_slot_hex "$flags")" ]] || fail "map struct flags slot mismatch for $account at $slot2: expected=$flags got=$raw2"

  ok "raw map struct slots for $account equal [$balance, $nonce, $flags]"
}

assert_map_struct_getters() {
  local account="$1"
  local balance="$2"
  local nonce="$3"
  local flags="$4"
  local got_balance got_nonce got_flags

  got_balance="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_balance(address)(uint256)" "$account")")"
  got_nonce="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_nonce(address)(uint256)" "$account")")"
  got_flags="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_flags(address)(uint256)" "$account")")"
  [[ "$got_balance" == "$balance" ]] || fail "get_balance($account) mismatch: expected=$balance got=$got_balance"
  [[ "$got_nonce" == "$nonce" ]] || fail "get_nonce($account) mismatch: expected=$nonce got=$got_nonce"
  [[ "$got_flags" == "$flags" ]] || fail "get_flags($account) mismatch: expected=$flags got=$got_flags"

  ok "map struct getters for $account equal [$balance, $nonce, $flags]"
}

assert_slice_struct_length() {
  local expected="$1"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice struct length slot mismatch: expected=$slot_expected got=$slot_raw"
  ok "slice struct root slot length equals $expected"
}

assert_slice_struct_raw_slots() {
  local index="$1"
  local left="$2"
  local middle="$3"
  local right="$4"
  local base_offset slot0 slot1 slot2 raw0 raw1 raw2

  base_offset=$((index * 3))
  slot0="$(dynamic_array_slot "$base_offset")"
  slot1="$(dynamic_array_slot "$((base_offset + 1))")"
  slot2="$(dynamic_array_slot "$((base_offset + 2))")"
  raw0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot0")")"
  raw1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot1")")"
  raw2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot2")")"
  [[ "${raw0,,}" == "$(expected_slot_hex "$left")" ]] || fail "slice struct left slot mismatch for index $index at $slot0: expected=$left got=$raw0"
  [[ "${raw1,,}" == "$(expected_slot_hex "$middle")" ]] || fail "slice struct middle slot mismatch for index $index at $slot1: expected=$middle got=$raw1"
  [[ "${raw2,,}" == "$(expected_slot_hex "$right")" ]] || fail "slice struct right slot mismatch for index $index at $slot2: expected=$right got=$raw2"

  ok "raw slice struct slots for index $index equal [$left, $middle, $right]"
}

assert_slice_struct_getters() {
  local index="$1"
  local left="$2"
  local middle="$3"
  local right="$4"
  local got_left got_middle got_right

  got_left="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left(uint256)(uint256)" "$index")")"
  got_middle="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_middle(uint256)(uint256)" "$index")")"
  got_right="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right(uint256)(uint256)" "$index")")"
  [[ "$got_left" == "$left" ]] || fail "get_left($index) mismatch: expected=$left got=$got_left"
  [[ "$got_middle" == "$middle" ]] || fail "get_middle($index) mismatch: expected=$middle got=$got_middle"
  [[ "$got_right" == "$right" ]] || fail "get_right($index) mismatch: expected=$right got=$got_right"

  ok "slice struct getters for index $index equal [$left, $middle, $right]"
}

assert_struct_slice_length() {
  local expected="$1"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "struct slice length slot mismatch: expected=$slot_expected got=$slot_raw"
  ok "struct slice field root slot length equals $expected"
}

assert_struct_slice_raw_slot() {
  local index="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(dynamic_array_slot "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "struct slice slot mismatch for index $index at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw struct slice slot $slot for index $index equals $expected"
}

assert_struct_slice_getter() {
  local index="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get(uint256)(uint256)" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "struct slice get($index) mismatch: expected=$expected got=$getter"
  ok "struct slice get($index) equals $expected"
}

assert_multi_struct_slice_scalars() {
  local head="$1"
  local length="$2"
  local tail="$3"
  local raw_head raw_length raw_tail

  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  raw_length="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "multi struct slice head slot mismatch: expected=$head got=$raw_head"
  [[ "${raw_length,,}" == "$(expected_slot_hex "$length")" ]] || fail "multi struct slice length slot mismatch: expected=$length got=$raw_length"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "multi struct slice tail slot mismatch: expected=$tail got=$raw_tail"

  ok "multi struct slice scalar slots equal [head=$head, length=$length, tail=$tail]"
}

assert_multi_struct_slice_raw_slot() {
  local index="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(dynamic_array_slot_for_root 1 "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "multi struct slice slot mismatch for index $index at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw multi struct slice slot $slot for index $index equals $expected"
}

assert_multi_struct_slice_getters() {
  local head="$1"
  local tail="$2"
  local got_head got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head()(uint256)")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail()(uint256)")")"
  [[ "$got_head" == "$head" ]] || fail "multi struct get_head() mismatch: expected=$head got=$got_head"
  [[ "$got_tail" == "$tail" ]] || fail "multi struct get_tail() mismatch: expected=$tail got=$got_tail"

  ok "multi struct slice scalar getters equal [head=$head, tail=$tail]"
}

assert_multi_struct_slice_getter() {
  local index="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get(uint256)(uint256)" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "multi struct slice get($index) mismatch: expected=$expected got=$getter"
  ok "multi struct slice get($index) equals $expected"
}

assert_dual_struct_slice_scalars() {
  local left_length="$1"
  local pivot="$2"
  local right_length="$3"
  local raw_left_length raw_pivot raw_right_length

  raw_left_length="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  raw_pivot="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  raw_right_length="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  [[ "${raw_left_length,,}" == "$(expected_slot_hex "$left_length")" ]] || fail "dual struct slice left length mismatch: expected=$left_length got=$raw_left_length"
  [[ "${raw_pivot,,}" == "$(expected_slot_hex "$pivot")" ]] || fail "dual struct slice pivot mismatch: expected=$pivot got=$raw_pivot"
  [[ "${raw_right_length,,}" == "$(expected_slot_hex "$right_length")" ]] || fail "dual struct slice right length mismatch: expected=$right_length got=$raw_right_length"

  ok "dual struct slice scalar slots equal [left.length=$left_length, pivot=$pivot, right.length=$right_length]"
}

assert_dual_struct_slice_left_slot() {
  local index="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(dynamic_array_slot_for_root 0 "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "dual struct slice left[$index] mismatch at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw dual struct slice left[$index] at $slot equals $expected"
}

assert_dual_struct_slice_right_slot() {
  local index="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(dynamic_array_slot_for_root 2 "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "dual struct slice right[$index] mismatch at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw dual struct slice right[$index] at $slot equals $expected"
}

assert_dual_struct_slice_getters() {
  local pivot="$1"
  local got_pivot

  got_pivot="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_pivot()(uint256)")")"
  [[ "$got_pivot" == "$pivot" ]] || fail "dual struct slice get_pivot() mismatch: expected=$pivot got=$got_pivot"

  ok "dual struct slice scalar getter equals [pivot=$pivot]"
}

assert_dual_struct_slice_left_getter() {
  local index="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left(uint256)(uint256)" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "dual struct slice get_left($index) mismatch: expected=$expected got=$getter"
  ok "dual struct slice get_left($index) equals $expected"
}

assert_dual_struct_slice_right_getter() {
  local index="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right(uint256)(uint256)" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "dual struct slice get_right($index) mismatch: expected=$expected got=$getter"
  ok "dual struct slice get_right($index) equals $expected"
}

assert_slice_slice_outer_length() {
  local expected="$1"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice slice outer length mismatch: expected=$slot_expected got=$slot_raw"
  ok "slice slice outer length equals $expected"
}

assert_slice_slice_row_length() {
  local row="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(slice_slice_row_slot "$row")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice slice row $row length mismatch at $slot: expected=$slot_expected got=$slot_raw"
  ok "slice slice row $row length equals $expected"
}

assert_slice_slice_raw_slot() {
  local row="$1"
  local col="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(slice_slice_value_slot "$row" "$col")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice slice value mismatch for [$row][$col] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw slice slice slot $slot for [$row][$col] equals $expected"
}

assert_slice_slice_getter() {
  local row="$1"
  local col="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get(uint256,uint256)(uint256)" "$row" "$col")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "slice slice get($row,$col) mismatch: expected=$expected got=$getter"
  ok "slice slice get($row,$col) equals $expected"
}

assert_slice_dynamic_struct_length() {
  local expected="$1"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice dynamic struct outer length mismatch: expected=$slot_expected got=$slot_raw"
  ok "slice dynamic struct outer length equals $expected"
}

assert_slice_dynamic_struct_scalars() {
  local index="$1"
  local head="$2"
  local length="$3"
  local tail="$4"
  local base head_slot values_slot tail_slot raw_head raw_length raw_tail

  base="$(slice_dynamic_struct_base_slot "$index")"
  head_slot="$base"
  values_slot="$(slot_add_small "$base" 1)"
  tail_slot="$(slot_add_small "$base" 2)"
  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$head_slot")")"
  raw_length="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$values_slot")")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$tail_slot")")"
  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "slice dynamic struct row $index head mismatch at $head_slot: expected=$head got=$raw_head"
  [[ "${raw_length,,}" == "$(expected_slot_hex "$length")" ]] || fail "slice dynamic struct row $index values length mismatch at $values_slot: expected=$length got=$raw_length"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "slice dynamic struct row $index tail mismatch at $tail_slot: expected=$tail got=$raw_tail"

  ok "slice dynamic struct row $index scalar slots equal [head=$head, values.length=$length, tail=$tail]"
}

assert_slice_dynamic_struct_value_slot() {
  local index="$1"
  local col="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(slice_dynamic_struct_value_slot "$index" "$col")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice dynamic struct value mismatch for [$index].values[$col] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw slice dynamic struct slot $slot for [$index].values[$col] equals $expected"
}

assert_slice_dynamic_struct_getters() {
  local index="$1"
  local head="$2"
  local tail="$3"
  local got_head got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head(uint256)(uint256)" "$index")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail(uint256)(uint256)" "$index")")"
  [[ "$got_head" == "$head" ]] || fail "slice dynamic struct get_head($index) mismatch: expected=$head got=$got_head"
  [[ "$got_tail" == "$tail" ]] || fail "slice dynamic struct get_tail($index) mismatch: expected=$tail got=$got_tail"

  ok "slice dynamic struct scalar getters for row $index equal [head=$head, tail=$tail]"
}

assert_slice_dynamic_struct_value_getter() {
  local index="$1"
  local col="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_value(uint256,uint256)(uint256)" "$index" "$col")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "slice dynamic struct get_value($index,$col) mismatch: expected=$expected got=$getter"
  ok "slice dynamic struct get_value($index,$col) equals $expected"
}

assert_slice_nested_dual_dynamic_struct_length() {
  local expected="$1"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice nested dual dynamic outer length mismatch: expected=$slot_expected got=$slot_raw"
  ok "slice nested dual dynamic struct outer length equals $expected"
}

assert_slice_nested_dual_dynamic_struct_scalars() {
  local index="$1"
  local head="$2"
  local left_length="$3"
  local pivot="$4"
  local right_length="$5"
  local tail="$6"
  local base head_slot left_slot pivot_slot right_slot tail_slot raw_head raw_left raw_pivot raw_right raw_tail

  base="$(slice_nested_dual_dynamic_struct_base_slot "$index")"
  head_slot="$base"
  left_slot="$(slot_add_small "$base" 1)"
  pivot_slot="$(slot_add_small "$base" 2)"
  right_slot="$(slot_add_small "$base" 3)"
  tail_slot="$(slot_add_small "$base" 4)"
  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$head_slot")")"
  raw_left="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$left_slot")")"
  raw_pivot="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$pivot_slot")")"
  raw_right="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$right_slot")")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$tail_slot")")"
  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "slice nested dual dynamic row $index head mismatch at $head_slot: expected=$head got=$raw_head"
  [[ "${raw_left,,}" == "$(expected_slot_hex "$left_length")" ]] || fail "slice nested dual dynamic row $index left length mismatch at $left_slot: expected=$left_length got=$raw_left"
  [[ "${raw_pivot,,}" == "$(expected_slot_hex "$pivot")" ]] || fail "slice nested dual dynamic row $index pivot mismatch at $pivot_slot: expected=$pivot got=$raw_pivot"
  [[ "${raw_right,,}" == "$(expected_slot_hex "$right_length")" ]] || fail "slice nested dual dynamic row $index right length mismatch at $right_slot: expected=$right_length got=$raw_right"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "slice nested dual dynamic row $index tail mismatch at $tail_slot: expected=$tail got=$raw_tail"

  ok "slice nested dual dynamic struct row $index scalar slots equal [head=$head, duo=[left.length=$left_length, pivot=$pivot, right.length=$right_length], tail=$tail]"
}

assert_slice_nested_dual_dynamic_struct_left_slot() {
  local index="$1"
  local col="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(slice_nested_dual_dynamic_struct_left_slot "$index" "$col")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice nested dual dynamic left mismatch for [$index].left[$col] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw slice nested dual dynamic struct left[$index][$col] at $slot equals $expected"
}

assert_slice_nested_dual_dynamic_struct_right_slot() {
  local index="$1"
  local col="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(slice_nested_dual_dynamic_struct_right_slot "$index" "$col")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice nested dual dynamic right mismatch for [$index].right[$col] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw slice nested dual dynamic struct right[$index][$col] at $slot equals $expected"
}

assert_slice_nested_dual_dynamic_struct_getters() {
  local index="$1"
  local head="$2"
  local pivot="$3"
  local tail="$4"
  local got_head got_pivot got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head(uint256)(uint256)" "$index")")"
  got_pivot="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_pivot(uint256)(uint256)" "$index")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail(uint256)(uint256)" "$index")")"
  [[ "$got_head" == "$head" ]] || fail "slice nested dual dynamic get_head($index) mismatch: expected=$head got=$got_head"
  [[ "$got_pivot" == "$pivot" ]] || fail "slice nested dual dynamic get_pivot($index) mismatch: expected=$pivot got=$got_pivot"
  [[ "$got_tail" == "$tail" ]] || fail "slice nested dual dynamic get_tail($index) mismatch: expected=$tail got=$got_tail"

  ok "slice nested dual dynamic struct row $index scalar getters equal [head=$head, pivot=$pivot, tail=$tail]"
}

assert_slice_nested_dual_dynamic_struct_left_getter() {
  local index="$1"
  local col="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left(uint256,uint256)(uint256)" "$index" "$col")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "slice nested dual dynamic get_left($index,$col) mismatch: expected=$expected got=$getter"
  ok "slice nested dual dynamic struct get_left($index,$col) equals $expected"
}

assert_slice_nested_dual_dynamic_struct_right_getter() {
  local index="$1"
  local col="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right(uint256,uint256)(uint256)" "$index" "$col")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "slice nested dual dynamic get_right($index,$col) mismatch: expected=$expected got=$getter"
  ok "slice nested dual dynamic struct get_right($index,$col) equals $expected"
}

assert_map_slice_length() {
  local account="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(map_slice_root_slot "$account")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "map slice length mismatch for $account at $slot: expected=$slot_expected got=$slot_raw"
  ok "map slice length for $account equals $expected"
}

assert_map_slice_raw_slot() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(map_slice_value_slot "$account" "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "map slice value mismatch for $account[$index] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw map slice slot $slot for $account[$index] equals $expected"
}

assert_map_slice_getter() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get(address,uint256)(uint256)" "$account" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "map slice get($account,$index) mismatch: expected=$expected got=$getter"
  ok "map slice get($account,$index) equals $expected"
}

assert_map_dynamic_struct_scalars() {
  local account="$1"
  local head="$2"
  local length="$3"
  local tail="$4"
  local base head_slot values_slot tail_slot raw_head raw_length raw_tail

  base="$(map_dynamic_struct_base_slot "$account")"
  head_slot="$base"
  values_slot="$(slot_add_small "$base" 1)"
  tail_slot="$(slot_add_small "$base" 2)"
  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$head_slot")")"
  raw_length="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$values_slot")")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$tail_slot")")"

  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "map dynamic struct head mismatch for $account at $head_slot: expected=$head got=$raw_head"
  [[ "${raw_length,,}" == "$(expected_slot_hex "$length")" ]] || fail "map dynamic struct values length mismatch for $account at $values_slot: expected=$length got=$raw_length"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "map dynamic struct tail mismatch for $account at $tail_slot: expected=$tail got=$raw_tail"

  ok "map dynamic struct scalar slots for $account equal [head=$head, values.length=$length, tail=$tail]"
}

assert_map_dynamic_struct_value_slot() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(map_dynamic_struct_value_slot "$account" "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "map dynamic struct value mismatch for $account.values[$index] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw map dynamic struct slot $slot for $account.values[$index] equals $expected"
}

assert_map_dynamic_struct_getters() {
  local account="$1"
  local head="$2"
  local tail="$3"
  local got_head got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head(address)(uint256)" "$account")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail(address)(uint256)" "$account")")"
  [[ "$got_head" == "$head" ]] || fail "map dynamic struct get_head($account) mismatch: expected=$head got=$got_head"
  [[ "$got_tail" == "$tail" ]] || fail "map dynamic struct get_tail($account) mismatch: expected=$tail got=$got_tail"

  ok "map dynamic struct scalar getters for $account equal [head=$head, tail=$tail]"
}

assert_map_dynamic_struct_value_getter() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_value(address,uint256)(uint256)" "$account" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "map dynamic struct get_value($account,$index) mismatch: expected=$expected got=$getter"
  ok "map dynamic struct get_value($account,$index) equals $expected"
}

assert_map_dual_dynamic_struct_scalars() {
  local account="$1"
  local left_length="$2"
  local pivot="$3"
  local right_length="$4"
  local base left_slot pivot_slot right_slot raw_left raw_pivot raw_right

  base="$(map_dual_dynamic_struct_base_slot "$account")"
  left_slot="$base"
  pivot_slot="$(slot_add_small "$base" 1)"
  right_slot="$(slot_add_small "$base" 2)"
  raw_left="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$left_slot")")"
  raw_pivot="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$pivot_slot")")"
  raw_right="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$right_slot")")"

  [[ "${raw_left,,}" == "$(expected_slot_hex "$left_length")" ]] || fail "map dual dynamic struct left length mismatch for $account at $left_slot: expected=$left_length got=$raw_left"
  [[ "${raw_pivot,,}" == "$(expected_slot_hex "$pivot")" ]] || fail "map dual dynamic struct pivot mismatch for $account at $pivot_slot: expected=$pivot got=$raw_pivot"
  [[ "${raw_right,,}" == "$(expected_slot_hex "$right_length")" ]] || fail "map dual dynamic struct right length mismatch for $account at $right_slot: expected=$right_length got=$raw_right"

  ok "map dual dynamic struct scalar slots for $account equal [left.length=$left_length, pivot=$pivot, right.length=$right_length]"
}

assert_map_dual_dynamic_struct_left_slot() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(map_dual_dynamic_struct_left_slot "$account" "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "map dual dynamic struct left mismatch for $account.left[$index] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw map dual dynamic struct slot $slot for $account.left[$index] equals $expected"
}

assert_map_dual_dynamic_struct_right_slot() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(map_dual_dynamic_struct_right_slot "$account" "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "map dual dynamic struct right mismatch for $account.right[$index] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw map dual dynamic struct slot $slot for $account.right[$index] equals $expected"
}

assert_map_dual_dynamic_struct_getters() {
  local account="$1"
  local pivot="$2"
  local got_pivot

  got_pivot="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_pivot(address)(uint256)" "$account")")"
  [[ "$got_pivot" == "$pivot" ]] || fail "map dual dynamic struct get_pivot($account) mismatch: expected=$pivot got=$got_pivot"

  ok "map dual dynamic struct scalar getter for $account equals [pivot=$pivot]"
}

assert_map_dual_dynamic_struct_left_getter() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left(address,uint256)(uint256)" "$account" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "map dual dynamic struct get_left($account,$index) mismatch: expected=$expected got=$getter"
  ok "map dual dynamic struct get_left($account,$index) equals $expected"
}

assert_map_dual_dynamic_struct_right_getter() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right(address,uint256)(uint256)" "$account" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "map dual dynamic struct get_right($account,$index) mismatch: expected=$expected got=$getter"
  ok "map dual dynamic struct get_right($account,$index) equals $expected"
}

assert_nested_map_dynamic_struct_scalars() {
  local owner="$1"
  local spender="$2"
  local head="$3"
  local length="$4"
  local tail="$5"
  local base head_slot values_slot tail_slot raw_head raw_length raw_tail

  base="$(nested_map_dynamic_struct_base_slot "$owner" "$spender")"
  head_slot="$base"
  values_slot="$(slot_add_small "$base" 1)"
  tail_slot="$(slot_add_small "$base" 2)"
  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$head_slot")")"
  raw_length="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$values_slot")")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$tail_slot")")"

  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "nested map dynamic struct head mismatch for $owner/$spender at $head_slot: expected=$head got=$raw_head"
  [[ "${raw_length,,}" == "$(expected_slot_hex "$length")" ]] || fail "nested map dynamic struct values length mismatch for $owner/$spender at $values_slot: expected=$length got=$raw_length"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "nested map dynamic struct tail mismatch for $owner/$spender at $tail_slot: expected=$tail got=$raw_tail"

  ok "nested map dynamic struct scalar slots for $owner/$spender equal [head=$head, values.length=$length, tail=$tail]"
}

assert_nested_map_dynamic_struct_value_slot() {
  local owner="$1"
  local spender="$2"
  local index="$3"
  local expected="$4"
  local slot slot_raw slot_expected

  slot="$(nested_map_dynamic_struct_value_slot "$owner" "$spender" "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "nested map dynamic struct value mismatch for $owner/$spender.values[$index] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw nested map dynamic struct slot $slot for $owner/$spender.values[$index] equals $expected"
}

assert_nested_map_dynamic_struct_getters() {
  local owner="$1"
  local spender="$2"
  local head="$3"
  local tail="$4"
  local got_head got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head(address,address)(uint256)" "$owner" "$spender")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail(address,address)(uint256)" "$owner" "$spender")")"
  [[ "$got_head" == "$head" ]] || fail "nested map dynamic struct get_head($owner,$spender) mismatch: expected=$head got=$got_head"
  [[ "$got_tail" == "$tail" ]] || fail "nested map dynamic struct get_tail($owner,$spender) mismatch: expected=$tail got=$got_tail"

  ok "nested map dynamic struct scalar getters for $owner/$spender equal [head=$head, tail=$tail]"
}

assert_nested_map_dynamic_struct_value_getter() {
  local owner="$1"
  local spender="$2"
  local index="$3"
  local expected="$4"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_value(address,address,uint256)(uint256)" "$owner" "$spender" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "nested map dynamic struct get_value($owner,$spender,$index) mismatch: expected=$expected got=$getter"
  ok "nested map dynamic struct get_value($owner,$spender,$index) equals $expected"
}

assert_bitfield_raw_slot() {
  local enabled="$1"
  local locked="$2"
  local mode="$3"
  local threshold="$4"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(bitfield_slot_hex "$enabled" "$locked" "$mode" "$threshold")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "bitfield slot mismatch: expected=$slot_expected got=$slot_raw"

  ok "raw bitfield slot equals [enabled=$enabled, locked=$locked, mode=$mode, threshold=$threshold]"
}

assert_bitfield_getters() {
  local enabled="$1"
  local locked="$2"
  local mode="$3"
  local threshold="$4"
  local expected_enabled expected_locked got_enabled got_locked got_mode got_threshold

  if [[ "$enabled" == "0" ]]; then expected_enabled="false"; else expected_enabled="true"; fi
  if [[ "$locked" == "0" ]]; then expected_locked="false"; else expected_locked="true"; fi

  got_enabled="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_enabled()(bool)")")"
  got_locked="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_locked()(bool)")")"
  got_mode="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_mode()(uint8)")")"
  got_threshold="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_threshold()(uint16)")")"
  [[ "$got_enabled" == "$expected_enabled" ]] || fail "is_enabled() mismatch: expected=$expected_enabled got=$got_enabled"
  [[ "$got_locked" == "$expected_locked" ]] || fail "is_locked() mismatch: expected=$expected_locked got=$got_locked"
  [[ "$got_mode" == "$mode" ]] || fail "get_mode() mismatch: expected=$mode got=$got_mode"
  [[ "$got_threshold" == "$threshold" ]] || fail "get_threshold() mismatch: expected=$threshold got=$got_threshold"

  ok "bitfield getters equal [enabled=$enabled, locked=$locked, mode=$mode, threshold=$threshold]"
}

assert_custom_bitfield_raw_slot() {
  local enabled="$1"
  local code="$2"
  local delta="$3"
  local amount="$4"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(custom_bitfield_slot_hex "$enabled" "$code" "$delta" "$amount")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "custom bitfield slot mismatch: expected=$slot_expected got=$slot_raw"

  ok "raw custom bitfield slot equals [enabled=$enabled, code=$code, delta=$delta, amount=$amount]"
}

assert_custom_bitfield_getters() {
  local enabled="$1"
  local code="$2"
  local delta="$3"
  local amount="$4"
  local expected_enabled got_enabled got_code got_delta got_amount

  if [[ "$enabled" == "0" ]]; then expected_enabled="false"; else expected_enabled="true"; fi

  got_enabled="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_enabled()(bool)")")"
  got_code="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_code()(uint8)")")"
  got_delta="$(normalize_int "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_delta()(int16)")" 16)"
  got_amount="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_amount()(uint32)")")"
  [[ "$got_enabled" == "$expected_enabled" ]] || fail "custom bitfield is_enabled() mismatch: expected=$expected_enabled got=$got_enabled"
  [[ "$got_code" == "$code" ]] || fail "custom bitfield get_code() mismatch: expected=$code got=$got_code"
  [[ "$got_delta" == "$delta" ]] || fail "custom bitfield get_delta() mismatch: expected=$delta got=$got_delta"
  [[ "$got_amount" == "$amount" ]] || fail "custom bitfield get_amount() mismatch: expected=$amount got=$got_amount"

  ok "custom bitfield getters equal [enabled=$enabled, code=$code, delta=$delta, amount=$amount]"
}

assert_map_bitfield_raw_slot() {
  local account="$1"
  local enabled="$2"
  local locked="$3"
  local mode="$4"
  local threshold="$5"
  local slot slot_raw slot_expected

  slot="$(map_bitfield_slot "$account")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(bitfield_slot_hex "$enabled" "$locked" "$mode" "$threshold")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "map bitfield slot mismatch for $account: expected=$slot_expected got=$slot_raw"

  ok "raw map bitfield slot for $account equals [enabled=$enabled, locked=$locked, mode=$mode, threshold=$threshold]"
}

assert_map_bitfield_getters() {
  local account="$1"
  local enabled="$2"
  local locked="$3"
  local mode="$4"
  local threshold="$5"
  local expected_enabled expected_locked got_enabled got_locked got_mode got_threshold

  if [[ "$enabled" == "0" ]]; then expected_enabled="false"; else expected_enabled="true"; fi
  if [[ "$locked" == "0" ]]; then expected_locked="false"; else expected_locked="true"; fi

  got_enabled="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_enabled(address)(bool)" "$account")")"
  got_locked="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_locked(address)(bool)" "$account")")"
  got_mode="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_mode(address)(uint8)" "$account")")"
  got_threshold="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_threshold(address)(uint16)" "$account")")"
  [[ "$got_enabled" == "$expected_enabled" ]] || fail "map bitfield is_enabled($account) mismatch: expected=$expected_enabled got=$got_enabled"
  [[ "$got_locked" == "$expected_locked" ]] || fail "map bitfield is_locked($account) mismatch: expected=$expected_locked got=$got_locked"
  [[ "$got_mode" == "$mode" ]] || fail "map bitfield get_mode($account) mismatch: expected=$mode got=$got_mode"
  [[ "$got_threshold" == "$threshold" ]] || fail "map bitfield get_threshold($account) mismatch: expected=$threshold got=$got_threshold"

  ok "map bitfield getters for $account equal [enabled=$enabled, locked=$locked, mode=$mode, threshold=$threshold]"
}

assert_map_custom_bitfield_raw_slot() {
  local account="$1"
  local enabled="$2"
  local code="$3"
  local delta="$4"
  local amount="$5"
  local slot slot_raw slot_expected

  slot="$(map_bitfield_slot "$account")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(custom_bitfield_slot_hex "$enabled" "$code" "$delta" "$amount")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "map custom bitfield slot mismatch for $account: expected=$slot_expected got=$slot_raw"

  ok "raw map custom bitfield slot for $account equals [enabled=$enabled, code=$code, delta=$delta, amount=$amount]"
}

assert_map_custom_bitfield_getters() {
  local account="$1"
  local enabled="$2"
  local code="$3"
  local delta="$4"
  local amount="$5"
  local expected_enabled got_enabled got_code got_delta got_amount

  if [[ "$enabled" == "0" ]]; then expected_enabled="false"; else expected_enabled="true"; fi

  got_enabled="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_enabled(address)(bool)" "$account")")"
  got_code="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_code(address)(uint8)" "$account")")"
  got_delta="$(normalize_int "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_delta(address)(int16)" "$account")" 16)"
  got_amount="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_amount(address)(uint32)" "$account")")"
  [[ "$got_enabled" == "$expected_enabled" ]] || fail "map custom bitfield is_enabled($account) mismatch: expected=$expected_enabled got=$got_enabled"
  [[ "$got_code" == "$code" ]] || fail "map custom bitfield get_code($account) mismatch: expected=$code got=$got_code"
  [[ "$got_delta" == "$delta" ]] || fail "map custom bitfield get_delta($account) mismatch: expected=$delta got=$got_delta"
  [[ "$got_amount" == "$amount" ]] || fail "map custom bitfield get_amount($account) mismatch: expected=$amount got=$got_amount"

  ok "map custom bitfield getters for $account equal [enabled=$enabled, code=$code, delta=$delta, amount=$amount]"
}

assert_slice_bitfield_length() {
  local expected="$1"
  local length_raw length

  length_raw="$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)"
  length="$(normalize_uint "$length_raw")"
  [[ "$length" == "$expected" ]] || fail "slice bitfield length mismatch: expected=$expected got=$length"
  ok "slice bitfield length equals $expected"
}

assert_slice_bitfield_raw_slot() {
  local index="$1"
  local enabled="$2"
  local locked="$3"
  local mode="$4"
  local threshold="$5"
  local slot slot_raw slot_expected

  slot="$(slice_bitfield_slot "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(bitfield_slot_hex "$enabled" "$locked" "$mode" "$threshold")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice bitfield slot mismatch for index $index: expected=$slot_expected got=$slot_raw"

  ok "raw slice bitfield slot $index equals [enabled=$enabled, locked=$locked, mode=$mode, threshold=$threshold]"
}

assert_slice_bitfield_getters() {
  local index="$1"
  local enabled="$2"
  local locked="$3"
  local mode="$4"
  local threshold="$5"
  local expected_enabled expected_locked got_enabled got_locked got_mode got_threshold

  if [[ "$enabled" == "0" ]]; then expected_enabled="false"; else expected_enabled="true"; fi
  if [[ "$locked" == "0" ]]; then expected_locked="false"; else expected_locked="true"; fi

  got_enabled="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_enabled(uint256)(bool)" "$index")")"
  got_locked="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_locked(uint256)(bool)" "$index")")"
  got_mode="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_mode(uint256)(uint8)" "$index")")"
  got_threshold="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_threshold(uint256)(uint16)" "$index")")"
  [[ "$got_enabled" == "$expected_enabled" ]] || fail "slice bitfield is_enabled($index) mismatch: expected=$expected_enabled got=$got_enabled"
  [[ "$got_locked" == "$expected_locked" ]] || fail "slice bitfield is_locked($index) mismatch: expected=$expected_locked got=$got_locked"
  [[ "$got_mode" == "$mode" ]] || fail "slice bitfield get_mode($index) mismatch: expected=$mode got=$got_mode"
  [[ "$got_threshold" == "$threshold" ]] || fail "slice bitfield get_threshold($index) mismatch: expected=$threshold got=$got_threshold"

  ok "slice bitfield getters for index $index equal [enabled=$enabled, locked=$locked, mode=$mode, threshold=$threshold]"
}

assert_slice_custom_bitfield_raw_slot() {
  local index="$1"
  local enabled="$2"
  local code="$3"
  local delta="$4"
  local amount="$5"
  local slot slot_raw slot_expected

  slot="$(slice_bitfield_slot "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(custom_bitfield_slot_hex "$enabled" "$code" "$delta" "$amount")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice custom bitfield slot mismatch for index $index: expected=$slot_expected got=$slot_raw"

  ok "raw slice custom bitfield slot $index equals [enabled=$enabled, code=$code, delta=$delta, amount=$amount]"
}

assert_slice_custom_bitfield_getters() {
  local index="$1"
  local enabled="$2"
  local code="$3"
  local delta="$4"
  local amount="$5"
  local expected_enabled got_enabled got_code got_delta got_amount

  if [[ "$enabled" == "0" ]]; then expected_enabled="false"; else expected_enabled="true"; fi

  got_enabled="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_enabled(uint256)(bool)" "$index")")"
  got_code="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_code(uint256)(uint8)" "$index")")"
  got_delta="$(normalize_int "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_delta(uint256)(int16)" "$index")" 16)"
  got_amount="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_amount(uint256)(uint32)" "$index")")"
  [[ "$got_enabled" == "$expected_enabled" ]] || fail "slice custom bitfield is_enabled($index) mismatch: expected=$expected_enabled got=$got_enabled"
  [[ "$got_code" == "$code" ]] || fail "slice custom bitfield get_code($index) mismatch: expected=$code got=$got_code"
  [[ "$got_delta" == "$delta" ]] || fail "slice custom bitfield get_delta($index) mismatch: expected=$delta got=$got_delta"
  [[ "$got_amount" == "$amount" ]] || fail "slice custom bitfield get_amount($index) mismatch: expected=$amount got=$got_amount"

  ok "slice custom bitfield getters for index $index equal [enabled=$enabled, code=$code, delta=$delta, amount=$amount]"
}

assert_struct_bitfield_raw_slots() {
  local head="$1"
  local enabled="$2"
  local locked="$3"
  local mode="$4"
  local threshold="$5"
  local tail="$6"
  local raw_head raw_flags raw_tail flags_expected

  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  raw_flags="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  flags_expected="$(bitfield_slot_hex "$enabled" "$locked" "$mode" "$threshold")"
  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "struct bitfield head slot mismatch: expected=$head got=$raw_head"
  [[ "${raw_flags,,}" == "${flags_expected,,}" ]] || fail "struct bitfield flags slot mismatch: expected=$flags_expected got=$raw_flags"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "struct bitfield tail slot mismatch: expected=$tail got=$raw_tail"

  ok "raw struct bitfield slots equal [head=$head, flags=[$enabled,$locked,$mode,$threshold], tail=$tail]"
}

assert_struct_bitfield_getters() {
  local head="$1"
  local enabled="$2"
  local locked="$3"
  local mode="$4"
  local threshold="$5"
  local tail="$6"
  local expected_enabled expected_locked got_head got_enabled got_locked got_mode got_threshold got_tail

  if [[ "$enabled" == "0" ]]; then expected_enabled="false"; else expected_enabled="true"; fi
  if [[ "$locked" == "0" ]]; then expected_locked="false"; else expected_locked="true"; fi

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head()(uint256)")")"
  got_enabled="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_enabled()(bool)")")"
  got_locked="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_locked()(bool)")")"
  got_mode="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_mode()(uint8)")")"
  got_threshold="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_threshold()(uint16)")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail()(uint256)")")"
  [[ "$got_head" == "$head" ]] || fail "struct bitfield get_head() mismatch: expected=$head got=$got_head"
  [[ "$got_enabled" == "$expected_enabled" ]] || fail "struct bitfield is_enabled() mismatch: expected=$expected_enabled got=$got_enabled"
  [[ "$got_locked" == "$expected_locked" ]] || fail "struct bitfield is_locked() mismatch: expected=$expected_locked got=$got_locked"
  [[ "$got_mode" == "$mode" ]] || fail "struct bitfield get_mode() mismatch: expected=$mode got=$got_mode"
  [[ "$got_threshold" == "$threshold" ]] || fail "struct bitfield get_threshold() mismatch: expected=$threshold got=$got_threshold"
  [[ "$got_tail" == "$tail" ]] || fail "struct bitfield get_tail() mismatch: expected=$tail got=$got_tail"

  ok "struct bitfield getters equal [head=$head, flags=[$enabled,$locked,$mode,$threshold], tail=$tail]"
}

assert_struct_custom_bitfield_raw_slots() {
  local head="$1"
  local enabled="$2"
  local code="$3"
  local delta="$4"
  local amount="$5"
  local tail="$6"
  local raw_head raw_flags raw_tail flags_expected

  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  raw_flags="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  flags_expected="$(custom_bitfield_slot_hex "$enabled" "$code" "$delta" "$amount")"
  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "struct custom bitfield head slot mismatch: expected=$head got=$raw_head"
  [[ "${raw_flags,,}" == "${flags_expected,,}" ]] || fail "struct custom bitfield flags slot mismatch: expected=$flags_expected got=$raw_flags"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "struct custom bitfield tail slot mismatch: expected=$tail got=$raw_tail"

  ok "raw struct custom bitfield slots equal [head=$head, flags=[$enabled,$code,$delta,$amount], tail=$tail]"
}

assert_struct_custom_bitfield_getters() {
  local head="$1"
  local enabled="$2"
  local code="$3"
  local delta="$4"
  local amount="$5"
  local tail="$6"
  local expected_enabled got_head got_enabled got_code got_delta got_amount got_tail

  if [[ "$enabled" == "0" ]]; then expected_enabled="false"; else expected_enabled="true"; fi

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head()(uint256)")")"
  got_enabled="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_enabled()(bool)")")"
  got_code="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_code()(uint8)")")"
  got_delta="$(normalize_int "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_delta()(int16)")" 16)"
  got_amount="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_amount()(uint32)")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail()(uint256)")")"
  [[ "$got_head" == "$head" ]] || fail "struct custom bitfield get_head() mismatch: expected=$head got=$got_head"
  [[ "$got_enabled" == "$expected_enabled" ]] || fail "struct custom bitfield is_enabled() mismatch: expected=$expected_enabled got=$got_enabled"
  [[ "$got_code" == "$code" ]] || fail "struct custom bitfield get_code() mismatch: expected=$code got=$got_code"
  [[ "$got_delta" == "$delta" ]] || fail "struct custom bitfield get_delta() mismatch: expected=$delta got=$got_delta"
  [[ "$got_amount" == "$amount" ]] || fail "struct custom bitfield get_amount() mismatch: expected=$amount got=$got_amount"
  [[ "$got_tail" == "$tail" ]] || fail "struct custom bitfield get_tail() mismatch: expected=$tail got=$got_tail"

  ok "struct custom bitfield getters equal [head=$head, flags=[$enabled,$code,$delta,$amount], tail=$tail]"
}

assert_slice_struct_custom_bitfield_raw_slots() {
  local index="$1"
  local head="$2"
  local enabled="$3"
  local code="$4"
  local delta="$5"
  local amount="$6"
  local tail="$7"
  local base_offset slot0 slot1 slot2 raw_head raw_flags raw_tail flags_expected

  base_offset=$((index * 3))
  slot0="$(dynamic_array_slot "$base_offset")"
  slot1="$(dynamic_array_slot "$((base_offset + 1))")"
  slot2="$(dynamic_array_slot "$((base_offset + 2))")"
  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot0")")"
  raw_flags="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot1")")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot2")")"
  flags_expected="$(custom_bitfield_slot_hex "$enabled" "$code" "$delta" "$amount")"
  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "slice struct custom bitfield row $index head mismatch at $slot0: expected=$head got=$raw_head"
  [[ "${raw_flags,,}" == "${flags_expected,,}" ]] || fail "slice struct custom bitfield row $index flags mismatch at $slot1: expected=$flags_expected got=$raw_flags"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "slice struct custom bitfield row $index tail mismatch at $slot2: expected=$tail got=$raw_tail"

  ok "raw slice struct custom bitfield slots for index $index equal [head=$head, flags=[$enabled,$code,$delta,$amount], tail=$tail]"
}

assert_slice_struct_custom_bitfield_getters() {
  local index="$1"
  local head="$2"
  local enabled="$3"
  local code="$4"
  local delta="$5"
  local amount="$6"
  local tail="$7"
  local expected_enabled got_head got_enabled got_code got_delta got_amount got_tail

  if [[ "$enabled" == "0" ]]; then expected_enabled="false"; else expected_enabled="true"; fi

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head(uint256)(uint256)" "$index")")"
  got_enabled="$(normalize_bool "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "is_enabled(uint256)(bool)" "$index")")"
  got_code="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_code(uint256)(uint8)" "$index")")"
  got_delta="$(normalize_int "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_delta(uint256)(int16)" "$index")" 16)"
  got_amount="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_amount(uint256)(uint32)" "$index")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail(uint256)(uint256)" "$index")")"
  [[ "$got_head" == "$head" ]] || fail "slice struct custom bitfield get_head($index) mismatch: expected=$head got=$got_head"
  [[ "$got_enabled" == "$expected_enabled" ]] || fail "slice struct custom bitfield is_enabled($index) mismatch: expected=$expected_enabled got=$got_enabled"
  [[ "$got_code" == "$code" ]] || fail "slice struct custom bitfield get_code($index) mismatch: expected=$code got=$got_code"
  [[ "$got_delta" == "$delta" ]] || fail "slice struct custom bitfield get_delta($index) mismatch: expected=$delta got=$got_delta"
  [[ "$got_amount" == "$amount" ]] || fail "slice struct custom bitfield get_amount($index) mismatch: expected=$amount got=$got_amount"
  [[ "$got_tail" == "$tail" ]] || fail "slice struct custom bitfield get_tail($index) mismatch: expected=$tail got=$got_tail"

  ok "slice struct custom bitfield getters for index $index equal [head=$head, flags=[$enabled,$code,$delta,$amount], tail=$tail]"
}

assert_nested_struct_raw_slots() {
  local head="$1"
  local left="$2"
  local middle="$3"
  local right="$4"
  local tail="$5"
  local raw0 raw1 raw2 raw3 raw4

  raw0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  raw1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  raw2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  raw3="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 3)")"
  raw4="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 4)")"
  [[ "${raw0,,}" == "$(expected_slot_hex "$head")" ]] || fail "nested struct head slot mismatch: expected=$head got=$raw0"
  [[ "${raw1,,}" == "$(expected_slot_hex "$left")" ]] || fail "nested struct inner.left slot mismatch: expected=$left got=$raw1"
  [[ "${raw2,,}" == "$(expected_slot_hex "$middle")" ]] || fail "nested struct inner.middle slot mismatch: expected=$middle got=$raw2"
  [[ "${raw3,,}" == "$(expected_slot_hex "$right")" ]] || fail "nested struct inner.right slot mismatch: expected=$right got=$raw3"
  [[ "${raw4,,}" == "$(expected_slot_hex "$tail")" ]] || fail "nested struct tail slot mismatch: expected=$tail got=$raw4"

  ok "raw nested struct slots equal [head=$head, inner=[$left, $middle, $right], tail=$tail]"
}

assert_nested_struct_getters() {
  local head="$1"
  local left="$2"
  local middle="$3"
  local right="$4"
  local tail="$5"
  local got_head got_left got_middle got_right got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head()(uint256)")")"
  got_left="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left()(uint256)")")"
  got_middle="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_middle()(uint256)")")"
  got_right="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right()(uint256)")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail()(uint256)")")"
  [[ "$got_head" == "$head" ]] || fail "get_head() mismatch: expected=$head got=$got_head"
  [[ "$got_left" == "$left" ]] || fail "get_left() mismatch: expected=$left got=$got_left"
  [[ "$got_middle" == "$middle" ]] || fail "get_middle() mismatch: expected=$middle got=$got_middle"
  [[ "$got_right" == "$right" ]] || fail "get_right() mismatch: expected=$right got=$got_right"
  [[ "$got_tail" == "$tail" ]] || fail "get_tail() mismatch: expected=$tail got=$got_tail"

  ok "nested struct getters equal [head=$head, inner=[$left, $middle, $right], tail=$tail]"
}

assert_deep_nested_struct_raw_slots() {
  local head="$1"
  local before="$2"
  local left="$3"
  local middle="$4"
  local right="$5"
  local after="$6"
  local tail="$7"
  local raw0 raw1 raw2 raw3 raw4 raw5 raw6

  raw0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  raw1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  raw2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  raw3="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 3)")"
  raw4="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 4)")"
  raw5="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 5)")"
  raw6="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 6)")"
  [[ "${raw0,,}" == "$(expected_slot_hex "$head")" ]] || fail "deep nested struct head slot mismatch: expected=$head got=$raw0"
  [[ "${raw1,,}" == "$(expected_slot_hex "$before")" ]] || fail "deep nested struct mid.before slot mismatch: expected=$before got=$raw1"
  [[ "${raw2,,}" == "$(expected_slot_hex "$left")" ]] || fail "deep nested struct leaf.left slot mismatch: expected=$left got=$raw2"
  [[ "${raw3,,}" == "$(expected_slot_hex "$middle")" ]] || fail "deep nested struct leaf.middle slot mismatch: expected=$middle got=$raw3"
  [[ "${raw4,,}" == "$(expected_slot_hex "$right")" ]] || fail "deep nested struct leaf.right slot mismatch: expected=$right got=$raw4"
  [[ "${raw5,,}" == "$(expected_slot_hex "$after")" ]] || fail "deep nested struct mid.after slot mismatch: expected=$after got=$raw5"
  [[ "${raw6,,}" == "$(expected_slot_hex "$tail")" ]] || fail "deep nested struct tail slot mismatch: expected=$tail got=$raw6"

  ok "raw deep nested struct slots equal [head=$head, mid=[$before, leaf=[$left, $middle, $right], $after], tail=$tail]"
}

assert_deep_nested_struct_getters() {
  local head="$1"
  local before="$2"
  local left="$3"
  local middle="$4"
  local right="$5"
  local after="$6"
  local tail="$7"
  local got_head got_before got_left got_middle got_right got_after got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head()(uint256)")")"
  got_before="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_before()(uint256)")")"
  got_left="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left()(uint256)")")"
  got_middle="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_middle()(uint256)")")"
  got_right="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right()(uint256)")")"
  got_after="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_after()(uint256)")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail()(uint256)")")"
  [[ "$got_head" == "$head" ]] || fail "deep nested get_head() mismatch: expected=$head got=$got_head"
  [[ "$got_before" == "$before" ]] || fail "deep nested get_before() mismatch: expected=$before got=$got_before"
  [[ "$got_left" == "$left" ]] || fail "deep nested get_left() mismatch: expected=$left got=$got_left"
  [[ "$got_middle" == "$middle" ]] || fail "deep nested get_middle() mismatch: expected=$middle got=$got_middle"
  [[ "$got_right" == "$right" ]] || fail "deep nested get_right() mismatch: expected=$right got=$got_right"
  [[ "$got_after" == "$after" ]] || fail "deep nested get_after() mismatch: expected=$after got=$got_after"
  [[ "$got_tail" == "$tail" ]] || fail "deep nested get_tail() mismatch: expected=$tail got=$got_tail"

  ok "deep nested struct getters equal [head=$head, mid=[$before, leaf=[$left, $middle, $right], $after], tail=$tail]"
}

assert_slice_deep_nested_struct_length() {
  local expected="$1"
  local slot_raw slot_expected

  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "slice deep nested struct length slot mismatch: expected=$slot_expected got=$slot_raw"
  ok "slice deep nested struct root slot length equals $expected"
}

assert_slice_deep_nested_struct_raw_slots() {
  local index="$1"
  local head="$2"
  local before="$3"
  local left="$4"
  local middle="$5"
  local right="$6"
  local after="$7"
  local tail="$8"
  local base_offset slot0 slot1 slot2 slot3 slot4 slot5 slot6 raw0 raw1 raw2 raw3 raw4 raw5 raw6

  base_offset=$((index * 7))
  slot0="$(dynamic_array_slot "$base_offset")"
  slot1="$(dynamic_array_slot "$((base_offset + 1))")"
  slot2="$(dynamic_array_slot "$((base_offset + 2))")"
  slot3="$(dynamic_array_slot "$((base_offset + 3))")"
  slot4="$(dynamic_array_slot "$((base_offset + 4))")"
  slot5="$(dynamic_array_slot "$((base_offset + 5))")"
  slot6="$(dynamic_array_slot "$((base_offset + 6))")"
  raw0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot0")")"
  raw1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot1")")"
  raw2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot2")")"
  raw3="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot3")")"
  raw4="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot4")")"
  raw5="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot5")")"
  raw6="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot6")")"
  [[ "${raw0,,}" == "$(expected_slot_hex "$head")" ]] || fail "slice deep nested struct head slot mismatch for index $index: expected=$head got=$raw0"
  [[ "${raw1,,}" == "$(expected_slot_hex "$before")" ]] || fail "slice deep nested struct mid.before slot mismatch for index $index: expected=$before got=$raw1"
  [[ "${raw2,,}" == "$(expected_slot_hex "$left")" ]] || fail "slice deep nested struct leaf.left slot mismatch for index $index: expected=$left got=$raw2"
  [[ "${raw3,,}" == "$(expected_slot_hex "$middle")" ]] || fail "slice deep nested struct leaf.middle slot mismatch for index $index: expected=$middle got=$raw3"
  [[ "${raw4,,}" == "$(expected_slot_hex "$right")" ]] || fail "slice deep nested struct leaf.right slot mismatch for index $index: expected=$right got=$raw4"
  [[ "${raw5,,}" == "$(expected_slot_hex "$after")" ]] || fail "slice deep nested struct mid.after slot mismatch for index $index: expected=$after got=$raw5"
  [[ "${raw6,,}" == "$(expected_slot_hex "$tail")" ]] || fail "slice deep nested struct tail slot mismatch for index $index: expected=$tail got=$raw6"

  ok "raw slice deep nested struct slots for index $index equal [head=$head, mid=[$before, leaf=[$left, $middle, $right], $after], tail=$tail]"
}

assert_slice_deep_nested_struct_getters() {
  local index="$1"
  local head="$2"
  local before="$3"
  local left="$4"
  local middle="$5"
  local right="$6"
  local after="$7"
  local tail="$8"
  local got_head got_before got_left got_middle got_right got_after got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head(uint256)(uint256)" "$index")")"
  got_before="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_before(uint256)(uint256)" "$index")")"
  got_left="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left(uint256)(uint256)" "$index")")"
  got_middle="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_middle(uint256)(uint256)" "$index")")"
  got_right="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right(uint256)(uint256)" "$index")")"
  got_after="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_after(uint256)(uint256)" "$index")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail(uint256)(uint256)" "$index")")"
  [[ "$got_head" == "$head" ]] || fail "slice deep nested get_head($index) mismatch: expected=$head got=$got_head"
  [[ "$got_before" == "$before" ]] || fail "slice deep nested get_before($index) mismatch: expected=$before got=$got_before"
  [[ "$got_left" == "$left" ]] || fail "slice deep nested get_left($index) mismatch: expected=$left got=$got_left"
  [[ "$got_middle" == "$middle" ]] || fail "slice deep nested get_middle($index) mismatch: expected=$middle got=$got_middle"
  [[ "$got_right" == "$right" ]] || fail "slice deep nested get_right($index) mismatch: expected=$right got=$got_right"
  [[ "$got_after" == "$after" ]] || fail "slice deep nested get_after($index) mismatch: expected=$after got=$got_after"
  [[ "$got_tail" == "$tail" ]] || fail "slice deep nested get_tail($index) mismatch: expected=$tail got=$got_tail"

  ok "slice deep nested struct getters for index $index equal [head=$head, mid=[$before, leaf=[$left, $middle, $right], $after], tail=$tail]"
}

assert_deep_dynamic_struct_scalars() {
  local head="$1"
  local before="$2"
  local left="$3"
  local length="$4"
  local right="$5"
  local after="$6"
  local tail="$7"
  local raw0 raw1 raw2 raw3 raw4 raw5 raw6

  raw0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  raw1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  raw2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  raw3="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 3)")"
  raw4="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 4)")"
  raw5="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 5)")"
  raw6="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 6)")"
  [[ "${raw0,,}" == "$(expected_slot_hex "$head")" ]] || fail "deep dynamic struct head slot mismatch: expected=$head got=$raw0"
  [[ "${raw1,,}" == "$(expected_slot_hex "$before")" ]] || fail "deep dynamic struct mid.before slot mismatch: expected=$before got=$raw1"
  [[ "${raw2,,}" == "$(expected_slot_hex "$left")" ]] || fail "deep dynamic struct leaf.left slot mismatch: expected=$left got=$raw2"
  [[ "${raw3,,}" == "$(expected_slot_hex "$length")" ]] || fail "deep dynamic struct leaf.values length mismatch: expected=$length got=$raw3"
  [[ "${raw4,,}" == "$(expected_slot_hex "$right")" ]] || fail "deep dynamic struct leaf.right slot mismatch: expected=$right got=$raw4"
  [[ "${raw5,,}" == "$(expected_slot_hex "$after")" ]] || fail "deep dynamic struct mid.after slot mismatch: expected=$after got=$raw5"
  [[ "${raw6,,}" == "$(expected_slot_hex "$tail")" ]] || fail "deep dynamic struct tail slot mismatch: expected=$tail got=$raw6"

  ok "deep dynamic struct scalar slots equal [head=$head, mid=[$before, leaf=[$left, values.length=$length, $right], $after], tail=$tail]"
}

assert_deep_dynamic_struct_value_slot() {
  local col="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(deep_dynamic_struct_value_slot "$col")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "deep dynamic struct value mismatch for values[$col] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw deep dynamic struct value slot $slot for values[$col] equals $expected"
}

assert_deep_dynamic_struct_getters() {
  local head="$1"
  local before="$2"
  local left="$3"
  local right="$4"
  local after="$5"
  local tail="$6"
  local got_head got_before got_left got_right got_after got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head()(uint256)")")"
  got_before="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_before()(uint256)")")"
  got_left="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left()(uint256)")")"
  got_right="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right()(uint256)")")"
  got_after="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_after()(uint256)")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail()(uint256)")")"
  [[ "$got_head" == "$head" ]] || fail "deep dynamic get_head() mismatch: expected=$head got=$got_head"
  [[ "$got_before" == "$before" ]] || fail "deep dynamic get_before() mismatch: expected=$before got=$got_before"
  [[ "$got_left" == "$left" ]] || fail "deep dynamic get_left() mismatch: expected=$left got=$got_left"
  [[ "$got_right" == "$right" ]] || fail "deep dynamic get_right() mismatch: expected=$right got=$got_right"
  [[ "$got_after" == "$after" ]] || fail "deep dynamic get_after() mismatch: expected=$after got=$got_after"
  [[ "$got_tail" == "$tail" ]] || fail "deep dynamic get_tail() mismatch: expected=$tail got=$got_tail"

  ok "deep dynamic struct scalar getters equal [head=$head, mid=[$before, leaf=[$left, values, $right], $after], tail=$tail]"
}

assert_deep_dynamic_struct_value_getter() {
  local col="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_value(uint256)(uint256)" "$col")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "deep dynamic get_value($col) mismatch: expected=$expected got=$getter"
  ok "deep dynamic struct get_value($col) equals $expected"
}

assert_nested_dual_dynamic_struct_scalars() {
  local head="$1"
  local left_length="$2"
  local pivot="$3"
  local right_length="$4"
  local tail="$5"
  local raw_head raw_left raw_pivot raw_right raw_tail

  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  raw_left="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  raw_pivot="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  raw_right="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 3)")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 4)")"
  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "nested dual dynamic struct head slot mismatch: expected=$head got=$raw_head"
  [[ "${raw_left,,}" == "$(expected_slot_hex "$left_length")" ]] || fail "nested dual dynamic struct left length mismatch: expected=$left_length got=$raw_left"
  [[ "${raw_pivot,,}" == "$(expected_slot_hex "$pivot")" ]] || fail "nested dual dynamic struct pivot mismatch: expected=$pivot got=$raw_pivot"
  [[ "${raw_right,,}" == "$(expected_slot_hex "$right_length")" ]] || fail "nested dual dynamic struct right length mismatch: expected=$right_length got=$raw_right"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "nested dual dynamic struct tail slot mismatch: expected=$tail got=$raw_tail"

  ok "nested dual dynamic struct scalar slots equal [head=$head, duo=[left.length=$left_length, pivot=$pivot, right.length=$right_length], tail=$tail]"
}

assert_nested_dual_dynamic_struct_left_slot() {
  local index="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(nested_dual_dynamic_struct_left_slot "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "nested dual dynamic struct left[$index] mismatch at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw nested dual dynamic struct left[$index] at $slot equals $expected"
}

assert_nested_dual_dynamic_struct_right_slot() {
  local index="$1"
  local expected="$2"
  local slot slot_raw slot_expected

  slot="$(nested_dual_dynamic_struct_right_slot "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "nested dual dynamic struct right[$index] mismatch at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw nested dual dynamic struct right[$index] at $slot equals $expected"
}

assert_nested_dual_dynamic_struct_getters() {
  local head="$1"
  local pivot="$2"
  local tail="$3"
  local got_head got_pivot got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head()(uint256)")")"
  got_pivot="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_pivot()(uint256)")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail()(uint256)")")"
  [[ "$got_head" == "$head" ]] || fail "nested dual dynamic get_head() mismatch: expected=$head got=$got_head"
  [[ "$got_pivot" == "$pivot" ]] || fail "nested dual dynamic get_pivot() mismatch: expected=$pivot got=$got_pivot"
  [[ "$got_tail" == "$tail" ]] || fail "nested dual dynamic get_tail() mismatch: expected=$tail got=$got_tail"

  ok "nested dual dynamic struct scalar getters equal [head=$head, pivot=$pivot, tail=$tail]"
}

assert_nested_dual_dynamic_struct_left_getter() {
  local index="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left(uint256)(uint256)" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "nested dual dynamic get_left($index) mismatch: expected=$expected got=$getter"
  ok "nested dual dynamic struct get_left($index) equals $expected"
}

assert_nested_dual_dynamic_struct_right_getter() {
  local index="$1"
  local expected="$2"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right(uint256)(uint256)" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "nested dual dynamic get_right($index) mismatch: expected=$expected got=$getter"
  ok "nested dual dynamic struct get_right($index) equals $expected"
}

assert_map_nested_dual_dynamic_struct_scalars() {
  local account="$1"
  local head="$2"
  local left_length="$3"
  local pivot="$4"
  local right_length="$5"
  local tail="$6"
  local base head_slot left_slot pivot_slot right_slot tail_slot raw_head raw_left raw_pivot raw_right raw_tail

  base="$(map_nested_dual_dynamic_struct_base_slot "$account")"
  head_slot="$base"
  left_slot="$(slot_add_small "$base" 1)"
  pivot_slot="$(slot_add_small "$base" 2)"
  right_slot="$(slot_add_small "$base" 3)"
  tail_slot="$(slot_add_small "$base" 4)"
  raw_head="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$head_slot")")"
  raw_left="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$left_slot")")"
  raw_pivot="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$pivot_slot")")"
  raw_right="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$right_slot")")"
  raw_tail="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$tail_slot")")"
  [[ "${raw_head,,}" == "$(expected_slot_hex "$head")" ]] || fail "map nested dual dynamic struct head slot mismatch for $account: expected=$head got=$raw_head"
  [[ "${raw_left,,}" == "$(expected_slot_hex "$left_length")" ]] || fail "map nested dual dynamic struct left length mismatch for $account: expected=$left_length got=$raw_left"
  [[ "${raw_pivot,,}" == "$(expected_slot_hex "$pivot")" ]] || fail "map nested dual dynamic struct pivot mismatch for $account: expected=$pivot got=$raw_pivot"
  [[ "${raw_right,,}" == "$(expected_slot_hex "$right_length")" ]] || fail "map nested dual dynamic struct right length mismatch for $account: expected=$right_length got=$raw_right"
  [[ "${raw_tail,,}" == "$(expected_slot_hex "$tail")" ]] || fail "map nested dual dynamic struct tail slot mismatch for $account: expected=$tail got=$raw_tail"

  ok "map nested dual dynamic struct scalar slots for $account equal [head=$head, duo=[left.length=$left_length, pivot=$pivot, right.length=$right_length], tail=$tail]"
}

assert_map_nested_dual_dynamic_struct_left_slot() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(map_nested_dual_dynamic_struct_left_slot "$account" "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "map nested dual dynamic struct left mismatch for $account.left[$index] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw map nested dual dynamic struct left[$index] for $account at $slot equals $expected"
}

assert_map_nested_dual_dynamic_struct_right_slot() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local slot slot_raw slot_expected

  slot="$(map_nested_dual_dynamic_struct_right_slot "$account" "$index")"
  slot_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot")")"
  slot_expected="$(expected_slot_hex "$expected")"
  [[ "${slot_raw,,}" == "${slot_expected,,}" ]] || fail "map nested dual dynamic struct right mismatch for $account.right[$index] at $slot: expected=$slot_expected got=$slot_raw"
  ok "raw map nested dual dynamic struct right[$index] for $account at $slot equals $expected"
}

assert_map_nested_dual_dynamic_struct_getters() {
  local account="$1"
  local head="$2"
  local pivot="$3"
  local tail="$4"
  local got_head got_pivot got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head(address)(uint256)" "$account")")"
  got_pivot="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_pivot(address)(uint256)" "$account")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail(address)(uint256)" "$account")")"
  [[ "$got_head" == "$head" ]] || fail "map nested dual dynamic get_head($account) mismatch: expected=$head got=$got_head"
  [[ "$got_pivot" == "$pivot" ]] || fail "map nested dual dynamic get_pivot($account) mismatch: expected=$pivot got=$got_pivot"
  [[ "$got_tail" == "$tail" ]] || fail "map nested dual dynamic get_tail($account) mismatch: expected=$tail got=$got_tail"

  ok "map nested dual dynamic struct scalar getters for $account equal [head=$head, pivot=$pivot, tail=$tail]"
}

assert_map_nested_dual_dynamic_struct_left_getter() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left(address,uint256)(uint256)" "$account" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "map nested dual dynamic get_left($account,$index) mismatch: expected=$expected got=$getter"
  ok "map nested dual dynamic struct get_left($account,$index) equals $expected"
}

assert_map_nested_dual_dynamic_struct_right_getter() {
  local account="$1"
  local index="$2"
  local expected="$3"
  local getter_raw getter

  getter_raw="$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right(address,uint256)(uint256)" "$account" "$index")"
  getter="$(normalize_uint "$getter_raw")"
  [[ "$getter" == "$expected" ]] || fail "map nested dual dynamic get_right($account,$index) mismatch: expected=$expected got=$getter"
  ok "map nested dual dynamic struct get_right($account,$index) equals $expected"
}

assert_map_nested_struct_raw_slots() {
  local account="$1"
  local head="$2"
  local left="$3"
  local middle="$4"
  local right="$5"
  local tail="$6"
  local base slot0 slot1 slot2 slot3 slot4 raw0 raw1 raw2 raw3 raw4

  base="$(trim "$(cast index address "$account" 0)")"
  slot0="$base"
  slot1="$(slot_add_small "$base" 1)"
  slot2="$(slot_add_small "$base" 2)"
  slot3="$(slot_add_small "$base" 3)"
  slot4="$(slot_add_small "$base" 4)"
  raw0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot0")")"
  raw1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot1")")"
  raw2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot2")")"
  raw3="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot3")")"
  raw4="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot4")")"
  [[ "${raw0,,}" == "$(expected_slot_hex "$head")" ]] || fail "map nested struct head slot mismatch for $account: expected=$head got=$raw0"
  [[ "${raw1,,}" == "$(expected_slot_hex "$left")" ]] || fail "map nested struct inner.left slot mismatch for $account: expected=$left got=$raw1"
  [[ "${raw2,,}" == "$(expected_slot_hex "$middle")" ]] || fail "map nested struct inner.middle slot mismatch for $account: expected=$middle got=$raw2"
  [[ "${raw3,,}" == "$(expected_slot_hex "$right")" ]] || fail "map nested struct inner.right slot mismatch for $account: expected=$right got=$raw3"
  [[ "${raw4,,}" == "$(expected_slot_hex "$tail")" ]] || fail "map nested struct tail slot mismatch for $account: expected=$tail got=$raw4"

  ok "raw map nested struct slots for $account equal [head=$head, inner=[$left, $middle, $right], tail=$tail]"
}

assert_map_nested_struct_getters() {
  local account="$1"
  local head="$2"
  local left="$3"
  local middle="$4"
  local right="$5"
  local tail="$6"
  local got_head got_left got_middle got_right got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head(address)(uint256)" "$account")")"
  got_left="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left(address)(uint256)" "$account")")"
  got_middle="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_middle(address)(uint256)" "$account")")"
  got_right="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right(address)(uint256)" "$account")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail(address)(uint256)" "$account")")"
  [[ "$got_head" == "$head" ]] || fail "map nested get_head($account) mismatch: expected=$head got=$got_head"
  [[ "$got_left" == "$left" ]] || fail "map nested get_left($account) mismatch: expected=$left got=$got_left"
  [[ "$got_middle" == "$middle" ]] || fail "map nested get_middle($account) mismatch: expected=$middle got=$got_middle"
  [[ "$got_right" == "$right" ]] || fail "map nested get_right($account) mismatch: expected=$right got=$got_right"
  [[ "$got_tail" == "$tail" ]] || fail "map nested get_tail($account) mismatch: expected=$tail got=$got_tail"

  ok "map nested struct getters for $account equal [head=$head, inner=[$left, $middle, $right], tail=$tail]"
}

assert_map_deep_nested_struct_raw_slots() {
  local account="$1"
  local head="$2"
  local before="$3"
  local left="$4"
  local middle="$5"
  local right="$6"
  local after="$7"
  local tail="$8"
  local base slot0 slot1 slot2 slot3 slot4 slot5 slot6 raw0 raw1 raw2 raw3 raw4 raw5 raw6

  base="$(trim "$(cast index address "$account" 0)")"
  slot0="$base"
  slot1="$(slot_add_small "$base" 1)"
  slot2="$(slot_add_small "$base" 2)"
  slot3="$(slot_add_small "$base" 3)"
  slot4="$(slot_add_small "$base" 4)"
  slot5="$(slot_add_small "$base" 5)"
  slot6="$(slot_add_small "$base" 6)"
  raw0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot0")")"
  raw1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot1")")"
  raw2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot2")")"
  raw3="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot3")")"
  raw4="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot4")")"
  raw5="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot5")")"
  raw6="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$slot6")")"
  [[ "${raw0,,}" == "$(expected_slot_hex "$head")" ]] || fail "map deep nested struct head slot mismatch for $account: expected=$head got=$raw0"
  [[ "${raw1,,}" == "$(expected_slot_hex "$before")" ]] || fail "map deep nested struct mid.before slot mismatch for $account: expected=$before got=$raw1"
  [[ "${raw2,,}" == "$(expected_slot_hex "$left")" ]] || fail "map deep nested struct leaf.left slot mismatch for $account: expected=$left got=$raw2"
  [[ "${raw3,,}" == "$(expected_slot_hex "$middle")" ]] || fail "map deep nested struct leaf.middle slot mismatch for $account: expected=$middle got=$raw3"
  [[ "${raw4,,}" == "$(expected_slot_hex "$right")" ]] || fail "map deep nested struct leaf.right slot mismatch for $account: expected=$right got=$raw4"
  [[ "${raw5,,}" == "$(expected_slot_hex "$after")" ]] || fail "map deep nested struct mid.after slot mismatch for $account: expected=$after got=$raw5"
  [[ "${raw6,,}" == "$(expected_slot_hex "$tail")" ]] || fail "map deep nested struct tail slot mismatch for $account: expected=$tail got=$raw6"

  ok "raw map deep nested struct slots for $account equal [head=$head, mid=[$before, leaf=[$left, $middle, $right], $after], tail=$tail]"
}

assert_map_deep_nested_struct_getters() {
  local account="$1"
  local head="$2"
  local before="$3"
  local left="$4"
  local middle="$5"
  local right="$6"
  local after="$7"
  local tail="$8"
  local got_head got_before got_left got_middle got_right got_after got_tail

  got_head="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_head(address)(uint256)" "$account")")"
  got_before="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_before(address)(uint256)" "$account")")"
  got_left="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_left(address)(uint256)" "$account")")"
  got_middle="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_middle(address)(uint256)" "$account")")"
  got_right="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_right(address)(uint256)" "$account")")"
  got_after="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_after(address)(uint256)" "$account")")"
  got_tail="$(normalize_uint "$(cast call --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "get_tail(address)(uint256)" "$account")")"
  [[ "$got_head" == "$head" ]] || fail "map deep nested get_head($account) mismatch: expected=$head got=$got_head"
  [[ "$got_before" == "$before" ]] || fail "map deep nested get_before($account) mismatch: expected=$before got=$got_before"
  [[ "$got_left" == "$left" ]] || fail "map deep nested get_left($account) mismatch: expected=$left got=$got_left"
  [[ "$got_middle" == "$middle" ]] || fail "map deep nested get_middle($account) mismatch: expected=$middle got=$got_middle"
  [[ "$got_right" == "$right" ]] || fail "map deep nested get_right($account) mismatch: expected=$right got=$got_right"
  [[ "$got_after" == "$after" ]] || fail "map deep nested get_after($account) mismatch: expected=$after got=$got_after"
  [[ "$got_tail" == "$tail" ]] || fail "map deep nested get_tail($account) mismatch: expected=$tail got=$got_tail"

  ok "map deep nested struct getters for $account equal [head=$head, mid=[$before, leaf=[$left, $middle, $right], $after], tail=$tail]"
}

assert_high_arity_slots() {
  local account="$1"
  local one="$2"
  local two="$3"
  local three="$4"
  local four="$5"
  local five="$6"
  local six="${7:-0}"
  local seven="${8:-0}"
  local raw0 raw1 raw2 raw3 raw4 raw5 raw6 raw7

  raw0="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
  raw1="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 1)")"
  raw2="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 2)")"
  raw3="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 3)")"
  raw4="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 4)")"
  raw5="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 5)")"
  raw6="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 6)")"
  raw7="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 7)")"
  [[ "${raw0,,}" == "$(expected_address_slot_hex "$account")" ]] || fail "high-arity address slot mismatch: expected=$account got=$raw0"
  [[ "${raw1,,}" == "$(expected_slot_hex "$one")" ]] || fail "high-arity arg1 slot mismatch: expected=$one got=$raw1"
  [[ "${raw2,,}" == "$(expected_slot_hex "$two")" ]] || fail "high-arity arg2 slot mismatch: expected=$two got=$raw2"
  [[ "${raw3,,}" == "$(expected_slot_hex "$three")" ]] || fail "high-arity arg3 slot mismatch: expected=$three got=$raw3"
  [[ "${raw4,,}" == "$(expected_slot_hex "$four")" ]] || fail "high-arity arg4 slot mismatch: expected=$four got=$raw4"
  [[ "${raw5,,}" == "$(expected_slot_hex "$five")" ]] || fail "high-arity arg5 slot mismatch: expected=$five got=$raw5"
  [[ "${raw6,,}" == "$(expected_slot_hex "$six")" ]] || fail "high-arity arg6 slot mismatch: expected=$six got=$raw6"
  [[ "${raw7,,}" == "$(expected_slot_hex "$seven")" ]] || fail "high-arity arg7 slot mismatch: expected=$seven got=$raw7"

  ok "high-arity slots equal [account=$account, args=[$one, $two, $three, $four, $five, $six, $seven]]"
}

send_set_balance() {
  local account="$1"
  local amount="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_balance(address,uint256)" "$account" "$amount")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract set_balance transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for set_balance receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "set_balance failed with status=$status"
}

send_set_u256_mapping_score() {
  local key="$1"
  local amount="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_score(uint256,uint256)" "$key" "$amount")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract u256 mapping set_score transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for u256 mapping set_score receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "u256 mapping set_score failed with status=$status"
}

send_set_integer_mapping_score() {
  local label="$1"
  local abi_type="$2"
  local setter_name="$3"
  local key="$4"
  local amount="$5"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "$setter_name($abi_type,uint256)" -- "$key" "$amount")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract $label mapping $setter_name transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for $label mapping $setter_name receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "$label mapping $setter_name failed with status=$status"
}

send_set_fixed_bytes_mapping_score() {
  local label="$1"
  local abi_type="$2"
  local setter_name="$3"
  local key="$4"
  local amount="$5"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "$setter_name($abi_type,uint256)" "$key" "$amount")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract $label mapping $setter_name transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for $label mapping $setter_name receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "$label mapping $setter_name failed with status=$status"
}

send_set_allowance() {
  local owner="$1"
  local spender="$2"
  local amount="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_allowance(address,address,uint256)" "$owner" "$spender" "$amount")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract set_allowance transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for set_allowance receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "set_allowance failed with status=$status"
}

send_set_triple_nested_limit() {
  local owner="$1"
  local spender="$2"
  local asset="$3"
  local amount="$4"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_limit(address,address,address,uint256)" "$owner" "$spender" "$asset" "$amount")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract triple nested mapping set_limit transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for triple nested mapping set_limit receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "triple nested mapping set_limit failed with status=$status"
}

send_set_nested_map_struct() {
  local owner="$1"
  local spender="$2"
  local amount="$3"
  local nonce="$4"
  local flags="$5"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(address,address,uint256,uint256,uint256)" "$owner" "$spender" "$amount" "$nonce" "$flags")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested map struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested map struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested map struct set failed with status=$status"
}

send_set_nested_map_struct_nonce() {
  local owner="$1"
  local spender="$2"
  local nonce="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_nonce(address,address,uint256)" "$owner" "$spender" "$nonce")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested map struct set_nonce transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested map struct set_nonce receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested map struct set_nonce failed with status=$status"
}

send_set_dynamic_array() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract dynamic array set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for dynamic array set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "dynamic array set failed with status=$status"
}

send_set_struct() {
  local left="$1"
  local middle="$2"
  local right="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256,uint256,uint256)" "$left" "$middle" "$right")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct set failed with status=$status"
}

send_set_struct_middle() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_middle(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct set_middle transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct set_middle receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct set_middle failed with status=$status"
}

send_set_map_struct() {
  local account="$1"
  local balance="$2"
  local nonce="$3"
  local flags="$4"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(address,uint256,uint256,uint256)" "$account" "$balance" "$nonce" "$flags")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map struct set failed with status=$status"
}

send_set_map_struct_nonce() {
  local account="$1"
  local nonce="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_nonce(address,uint256)" "$account" "$nonce")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map struct set_nonce transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map struct set_nonce receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map struct set_nonce failed with status=$status"
}

send_set_slice_struct() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set((uint256,uint256,uint256)[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice struct set failed with status=$status"
}

send_set_slice_struct_middle() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_middle(uint256,uint256)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice struct set_middle transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice struct set_middle receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice struct set_middle failed with status=$status"
}

send_set_struct_slice() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct slice set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct slice set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct slice set failed with status=$status"
}

send_set_multi_struct_slice() {
  local head="$1"
  local values="$2"
  local tail="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256,uint256[],uint256)" "$head" "$values" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract multi struct slice set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for multi struct slice set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "multi struct slice set failed with status=$status"
}

send_set_multi_struct_slice_head() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_head(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract multi struct slice set_head transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for multi struct slice set_head receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "multi struct slice set_head failed with status=$status"
}

send_set_multi_struct_slice_values() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_values(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract multi struct slice set_values transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for multi struct slice set_values receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "multi struct slice set_values failed with status=$status"
}

send_set_multi_struct_slice_tail() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract multi struct slice set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for multi struct slice set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "multi struct slice set_tail failed with status=$status"
}

send_set_dual_struct_slice() {
  local left="$1"
  local pivot="$2"
  local right="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256[],uint256,uint256[])" "$left" "$pivot" "$right")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract dual struct slice set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for dual struct slice set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "dual struct slice set failed with status=$status"
}

send_set_dual_struct_slice_left() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_left(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract dual struct slice set_left transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for dual struct slice set_left receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "dual struct slice set_left failed with status=$status"
}

send_set_dual_struct_slice_pivot() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_pivot(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract dual struct slice set_pivot transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for dual struct slice set_pivot receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "dual struct slice set_pivot failed with status=$status"
}

send_set_dual_struct_slice_right() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_right(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract dual struct slice set_right transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for dual struct slice set_right receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "dual struct slice set_right failed with status=$status"
}

send_set_slice_slice() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256[][])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice slice set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice slice set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice slice set failed with status=$status"
}

send_set_slice_dynamic_struct() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set((uint256,uint256[],uint256)[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice dynamic struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice dynamic struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice dynamic struct set failed with status=$status"
}

send_set_slice_dynamic_struct_head() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_head(uint256,uint256)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice dynamic struct set_head transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice dynamic struct set_head receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice dynamic struct set_head failed with status=$status"
}

send_set_slice_dynamic_struct_values() {
  local index="$1"
  local values="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_values(uint256,uint256[])" "$index" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice dynamic struct set_values transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice dynamic struct set_values receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice dynamic struct set_values failed with status=$status"
}

send_set_slice_dynamic_struct_tail() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(uint256,uint256)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice dynamic struct set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice dynamic struct set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice dynamic struct set_tail failed with status=$status"
}

send_set_slice_nested_dual_dynamic_struct() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set((uint256,(uint256[],uint256,uint256[]),uint256)[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice nested dual dynamic struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice nested dual dynamic struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice nested dual dynamic struct set failed with status=$status"
}

send_set_slice_nested_dual_dynamic_struct_left() {
  local index="$1"
  local values="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_left(uint256,uint256[])" "$index" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice nested dual dynamic struct set_left transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice nested dual dynamic struct set_left receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice nested dual dynamic struct set_left failed with status=$status"
}

send_set_slice_nested_dual_dynamic_struct_pivot() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_pivot(uint256,uint256)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice nested dual dynamic struct set_pivot transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice nested dual dynamic struct set_pivot receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice nested dual dynamic struct set_pivot failed with status=$status"
}

send_set_slice_nested_dual_dynamic_struct_right() {
  local index="$1"
  local values="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_right(uint256,uint256[])" "$index" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice nested dual dynamic struct set_right transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice nested dual dynamic struct set_right receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice nested dual dynamic struct set_right failed with status=$status"
}

send_set_slice_nested_dual_dynamic_struct_tail() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(uint256,uint256)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice nested dual dynamic struct set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice nested dual dynamic struct set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice nested dual dynamic struct set_tail failed with status=$status"
}

send_set_map_slice() {
  local account="$1"
  local values="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(address,uint256[])" "$account" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map slice set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map slice set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map slice set failed with status=$status"
}

send_set_map_dynamic_struct() {
  local account="$1"
  local head="$2"
  local values="$3"
  local tail="$4"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(address,uint256,uint256[],uint256)" "$account" "$head" "$values" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map dynamic struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map dynamic struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map dynamic struct set failed with status=$status"
}

send_set_map_dynamic_struct_head() {
  local account="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_head(address,uint256)" "$account" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map dynamic struct set_head transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map dynamic struct set_head receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map dynamic struct set_head failed with status=$status"
}

send_set_map_dynamic_struct_values() {
  local account="$1"
  local values="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_values(address,uint256[])" "$account" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map dynamic struct set_values transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map dynamic struct set_values receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map dynamic struct set_values failed with status=$status"
}

send_set_map_dynamic_struct_tail() {
  local account="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(address,uint256)" "$account" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map dynamic struct set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map dynamic struct set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map dynamic struct set_tail failed with status=$status"
}

send_set_map_dual_dynamic_struct() {
  local account="$1"
  local left="$2"
  local pivot="$3"
  local right="$4"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(address,uint256[],uint256,uint256[])" "$account" "$left" "$pivot" "$right")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map dual dynamic struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map dual dynamic struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map dual dynamic struct set failed with status=$status"
}

send_set_map_dual_dynamic_struct_left() {
  local account="$1"
  local values="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_left(address,uint256[])" "$account" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map dual dynamic struct set_left transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map dual dynamic struct set_left receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map dual dynamic struct set_left failed with status=$status"
}

send_set_map_dual_dynamic_struct_pivot() {
  local account="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_pivot(address,uint256)" "$account" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map dual dynamic struct set_pivot transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map dual dynamic struct set_pivot receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map dual dynamic struct set_pivot failed with status=$status"
}

send_set_map_dual_dynamic_struct_right() {
  local account="$1"
  local values="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_right(address,uint256[])" "$account" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map dual dynamic struct set_right transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map dual dynamic struct set_right receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map dual dynamic struct set_right failed with status=$status"
}

send_set_nested_map_dynamic_struct() {
  local owner="$1"
  local spender="$2"
  local head="$3"
  local values="$4"
  local tail="$5"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(address,address,uint256,uint256[],uint256)" "$owner" "$spender" "$head" "$values" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested map dynamic struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested map dynamic struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested map dynamic struct set failed with status=$status"
}

send_set_nested_map_dynamic_struct_head() {
  local owner="$1"
  local spender="$2"
  local value="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_head(address,address,uint256)" "$owner" "$spender" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested map dynamic struct set_head transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested map dynamic struct set_head receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested map dynamic struct set_head failed with status=$status"
}

send_set_nested_map_dynamic_struct_values() {
  local owner="$1"
  local spender="$2"
  local values="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_values(address,address,uint256[])" "$owner" "$spender" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested map dynamic struct set_values transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested map dynamic struct set_values receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested map dynamic struct set_values failed with status=$status"
}

send_set_nested_map_dynamic_struct_tail() {
  local owner="$1"
  local spender="$2"
  local value="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(address,address,uint256)" "$owner" "$spender" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested map dynamic struct set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested map dynamic struct set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested map dynamic struct set_tail failed with status=$status"
}

send_configure_bitfield() {
  local mode="$1"
  local threshold="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "configure(uint8,uint16)" "$mode" "$threshold")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract bitfield configure transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for bitfield configure receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "bitfield configure failed with status=$status"
}

send_set_bitfield_locked() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_locked(bool)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract bitfield set_locked transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for bitfield set_locked receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "bitfield set_locked failed with status=$status"
}

send_set_bitfield_mode() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_mode(uint8)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract bitfield set_mode transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for bitfield set_mode receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "bitfield set_mode failed with status=$status"
}

send_configure_custom_bitfield() {
  local code="$1"
  local delta="$2"
  local amount="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "configure(uint8,int16,uint32)" "$code" "$delta" "$amount")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract custom bitfield configure transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for custom bitfield configure receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "custom bitfield configure failed with status=$status"
}

send_set_custom_bitfield_delta() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_delta(int16)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract custom bitfield set_delta transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for custom bitfield set_delta receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "custom bitfield set_delta failed with status=$status"
}

send_set_custom_bitfield_code() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_code(uint8)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract custom bitfield set_code transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for custom bitfield set_code receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "custom bitfield set_code failed with status=$status"
}

send_configure_map_bitfield() {
  local account="$1"
  local mode="$2"
  local threshold="$3"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "configure(address,uint8,uint16)" "$account" "$mode" "$threshold")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map bitfield configure transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map bitfield configure receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map bitfield configure failed with status=$status"
}

send_set_map_bitfield_locked() {
  local account="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_locked(address,bool)" "$account" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map bitfield set_locked transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map bitfield set_locked receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map bitfield set_locked failed with status=$status"
}

send_set_map_bitfield_mode() {
  local account="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_mode(address,uint8)" "$account" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map bitfield set_mode transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map bitfield set_mode receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map bitfield set_mode failed with status=$status"
}

send_configure_map_custom_bitfield() {
  local account="$1"
  local code="$2"
  local delta="$3"
  local amount="$4"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "configure(address,uint8,int16,uint32)" "$account" "$code" "$delta" "$amount")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map custom bitfield configure transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map custom bitfield configure receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map custom bitfield configure failed with status=$status"
}

send_set_map_custom_bitfield_delta() {
  local account="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_delta(address,int16)" "$account" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map custom bitfield set_delta transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map custom bitfield set_delta receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map custom bitfield set_delta failed with status=$status"
}

send_set_map_custom_bitfield_code() {
  local account="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_code(address,uint8)" "$account" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map custom bitfield set_code transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map custom bitfield set_code receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map custom bitfield set_code failed with status=$status"
}

send_set_slice_bitfield() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice bitfield set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice bitfield set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice bitfield set failed with status=$status"
}

send_set_slice_bitfield_locked() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_locked(uint256,bool)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice bitfield set_locked transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice bitfield set_locked receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice bitfield set_locked failed with status=$status"
}

send_set_slice_bitfield_mode() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_mode(uint256,uint8)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice bitfield set_mode transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice bitfield set_mode receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice bitfield set_mode failed with status=$status"
}

send_set_slice_custom_bitfield() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice custom bitfield set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice custom bitfield set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice custom bitfield set failed with status=$status"
}

send_set_slice_custom_bitfield_delta() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_delta(uint256,int16)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice custom bitfield set_delta transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice custom bitfield set_delta receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice custom bitfield set_delta failed with status=$status"
}

send_set_slice_custom_bitfield_code() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_code(uint256,uint8)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice custom bitfield set_code transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice custom bitfield set_code receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice custom bitfield set_code failed with status=$status"
}

send_configure_struct_bitfield() {
  local head="$1"
  local mode="$2"
  local threshold="$3"
  local tail="$4"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "configure(uint256,uint8,uint16,uint256)" "$head" "$mode" "$threshold" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct bitfield configure transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct bitfield configure receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct bitfield configure failed with status=$status"
}

send_set_struct_bitfield_head() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_head(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct bitfield set_head transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct bitfield set_head receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct bitfield set_head failed with status=$status"
}

send_set_struct_bitfield_locked() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_locked(bool)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct bitfield set_locked transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct bitfield set_locked receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct bitfield set_locked failed with status=$status"
}

send_set_struct_bitfield_mode() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_mode(uint8)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct bitfield set_mode transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct bitfield set_mode receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct bitfield set_mode failed with status=$status"
}

send_set_struct_bitfield_tail() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct bitfield set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct bitfield set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct bitfield set_tail failed with status=$status"
}

send_configure_struct_custom_bitfield() {
  local head="$1"
  local code="$2"
  local delta="$3"
  local amount="$4"
  local tail="$5"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "configure(uint256,uint8,int16,uint32,uint256)" "$head" "$code" "$delta" "$amount" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct custom bitfield configure transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct custom bitfield configure receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct custom bitfield configure failed with status=$status"
}

send_set_struct_custom_bitfield_head() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_head(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct custom bitfield set_head transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct custom bitfield set_head receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct custom bitfield set_head failed with status=$status"
}

send_set_struct_custom_bitfield_delta() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_delta(int16)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct custom bitfield set_delta transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct custom bitfield set_delta receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct custom bitfield set_delta failed with status=$status"
}

send_set_struct_custom_bitfield_code() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_code(uint8)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct custom bitfield set_code transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct custom bitfield set_code receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct custom bitfield set_code failed with status=$status"
}

send_set_struct_custom_bitfield_tail() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract struct custom bitfield set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for struct custom bitfield set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "struct custom bitfield set_tail failed with status=$status"
}

send_set_slice_struct_custom_bitfield() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set((uint256,uint256,uint256)[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice struct custom bitfield set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice struct custom bitfield set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice struct custom bitfield set failed with status=$status"
}

send_set_slice_struct_custom_bitfield_head() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_head(uint256,uint256)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice struct custom bitfield set_head transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice struct custom bitfield set_head receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice struct custom bitfield set_head failed with status=$status"
}

send_set_slice_struct_custom_bitfield_delta() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_delta(uint256,int16)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice struct custom bitfield set_delta transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice struct custom bitfield set_delta receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice struct custom bitfield set_delta failed with status=$status"
}

send_set_slice_struct_custom_bitfield_code() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_code(uint256,uint8)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice struct custom bitfield set_code transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice struct custom bitfield set_code receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice struct custom bitfield set_code failed with status=$status"
}

send_set_slice_struct_custom_bitfield_tail() {
  local index="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(uint256,uint256)" "$index" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice struct custom bitfield set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice struct custom bitfield set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice struct custom bitfield set_tail failed with status=$status"
}

send_set_nested_struct() {
  local head="$1"
  local left="$2"
  local middle="$3"
  local right="$4"
  local tail="$5"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256,uint256,uint256,uint256,uint256)" "$head" "$left" "$middle" "$right" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested struct set failed with status=$status"
}

send_set_nested_struct_inner_middle() {
  local middle="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_inner_middle(uint256)" "$middle")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested struct set_inner_middle transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested struct set_inner_middle receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested struct set_inner_middle failed with status=$status"
}

send_set_deep_nested_struct() {
  local head="$1"
  local before="$2"
  local left="$3"
  local middle="$4"
  local right="$5"
  local after="$6"
  local tail="$7"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256,uint256,uint256,uint256,uint256,uint256,uint256)" "$head" "$before" "$left" "$middle" "$right" "$after" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract deep nested struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for deep nested struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "deep nested struct set failed with status=$status"
}

send_set_deep_nested_struct_leaf_middle() {
  local middle="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_leaf_middle(uint256)" "$middle")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract deep nested struct set_leaf_middle transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for deep nested struct set_leaf_middle receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "deep nested struct set_leaf_middle failed with status=$status"
}

send_set_deep_nested_struct_mid_after() {
  local after="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_mid_after(uint256)" "$after")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract deep nested struct set_mid_after transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for deep nested struct set_mid_after receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "deep nested struct set_mid_after failed with status=$status"
}

send_set_slice_deep_nested_struct() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set((uint256,(uint256,(uint256,uint256,uint256),uint256),uint256)[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice deep nested struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice deep nested struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice deep nested struct set failed with status=$status"
}

send_set_slice_deep_nested_struct_leaf_middle() {
  local index="$1"
  local middle="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_leaf_middle(uint256,uint256)" "$index" "$middle")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice deep nested struct set_leaf_middle transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice deep nested struct set_leaf_middle receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice deep nested struct set_leaf_middle failed with status=$status"
}

send_set_slice_deep_nested_struct_mid_after() {
  local index="$1"
  local after="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_mid_after(uint256,uint256)" "$index" "$after")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract slice deep nested struct set_mid_after transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for slice deep nested struct set_mid_after receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "slice deep nested struct set_mid_after failed with status=$status"
}

send_set_deep_dynamic_struct() {
  local head="$1"
  local before="$2"
  local left="$3"
  local values="$4"
  local right="$5"
  local after="$6"
  local tail="$7"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256,uint256,uint256,uint256[],uint256,uint256,uint256)" "$head" "$before" "$left" "$values" "$right" "$after" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract deep dynamic struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for deep dynamic struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "deep dynamic struct set failed with status=$status"
}

send_set_deep_dynamic_struct_values() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_values(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract deep dynamic struct set_values transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for deep dynamic struct set_values receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "deep dynamic struct set_values failed with status=$status"
}

send_set_deep_dynamic_struct_leaf_right() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_leaf_right(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract deep dynamic struct set_leaf_right transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for deep dynamic struct set_leaf_right receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "deep dynamic struct set_leaf_right failed with status=$status"
}

send_set_deep_dynamic_struct_mid_after() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_mid_after(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract deep dynamic struct set_mid_after transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for deep dynamic struct set_mid_after receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "deep dynamic struct set_mid_after failed with status=$status"
}

send_set_nested_dual_dynamic_struct() {
  local head="$1"
  local left="$2"
  local pivot="$3"
  local right="$4"
  local tail="$5"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(uint256,uint256[],uint256,uint256[],uint256)" "$head" "$left" "$pivot" "$right" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested dual dynamic struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested dual dynamic struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested dual dynamic struct set failed with status=$status"
}

send_set_nested_dual_dynamic_struct_left() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_left(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested dual dynamic struct set_left transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested dual dynamic struct set_left receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested dual dynamic struct set_left failed with status=$status"
}

send_set_nested_dual_dynamic_struct_pivot() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_pivot(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested dual dynamic struct set_pivot transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested dual dynamic struct set_pivot receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested dual dynamic struct set_pivot failed with status=$status"
}

send_set_nested_dual_dynamic_struct_right() {
  local values="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_right(uint256[])" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested dual dynamic struct set_right transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested dual dynamic struct set_right receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested dual dynamic struct set_right failed with status=$status"
}

send_set_nested_dual_dynamic_struct_tail() {
  local value="$1"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(uint256)" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract nested dual dynamic struct set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for nested dual dynamic struct set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "nested dual dynamic struct set_tail failed with status=$status"
}

send_set_map_nested_dual_dynamic_struct() {
  local account="$1"
  local head="$2"
  local left="$3"
  local pivot="$4"
  local right="$5"
  local tail="$6"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(address,uint256,uint256[],uint256,uint256[],uint256)" "$account" "$head" "$left" "$pivot" "$right" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map nested dual dynamic struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map nested dual dynamic struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map nested dual dynamic struct set failed with status=$status"
}

send_set_map_nested_dual_dynamic_struct_left() {
  local account="$1"
  local values="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_left(address,uint256[])" "$account" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map nested dual dynamic struct set_left transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map nested dual dynamic struct set_left receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map nested dual dynamic struct set_left failed with status=$status"
}

send_set_map_nested_dual_dynamic_struct_pivot() {
  local account="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_pivot(address,uint256)" "$account" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map nested dual dynamic struct set_pivot transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map nested dual dynamic struct set_pivot receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map nested dual dynamic struct set_pivot failed with status=$status"
}

send_set_map_nested_dual_dynamic_struct_right() {
  local account="$1"
  local values="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_right(address,uint256[])" "$account" "$values")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map nested dual dynamic struct set_right transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map nested dual dynamic struct set_right receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map nested dual dynamic struct set_right failed with status=$status"
}

send_set_map_nested_dual_dynamic_struct_tail() {
  local account="$1"
  local value="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_tail(address,uint256)" "$account" "$value")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map nested dual dynamic struct set_tail transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map nested dual dynamic struct set_tail receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map nested dual dynamic struct set_tail failed with status=$status"
}

send_set_map_nested_struct() {
  local account="$1"
  local head="$2"
  local left="$3"
  local middle="$4"
  local right="$5"
  local tail="$6"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(address,uint256,uint256,uint256,uint256,uint256)" "$account" "$head" "$left" "$middle" "$right" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map nested struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map nested struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map nested struct set failed with status=$status"
}

send_high_arity_set6() {
  local account="$1"
  local one="$2"
  local two="$3"
  local three="$4"
  local four="$5"
  local five="$6"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set6(address,uint256,uint256,uint256,uint256,uint256)" "$account" "$one" "$two" "$three" "$four" "$five")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract high-arity set6 transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for high-arity set6 receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "high-arity set6 failed with status=$status"
}

send_high_arity_set8() {
  local account="$1"
  local one="$2"
  local two="$3"
  local three="$4"
  local four="$5"
  local five="$6"
  local six="$7"
  local seven="$8"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set8(address,uint256,uint256,uint256,uint256,uint256,uint256,uint256)" "$account" "$one" "$two" "$three" "$four" "$five" "$six" "$seven")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract high-arity set8 transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for high-arity set8 receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "high-arity set8 failed with status=$status"
}

send_multicontract_call_target() {
  local caller_addr="$1"
  local target_addr="$2"
  local value="$3"

  send_contract_tx "$caller_addr" "external-call caller call_target" "call_target(address,uint256)" "$target_addr" "$value"
}

send_multicontract_pull_target() {
  local caller_addr="$1"
  local target_addr="$2"

  send_contract_tx "$caller_addr" "external-call caller pull_target" "pull_target(address)" "$target_addr"
}

send_multicontract_pull_snapshot() {
  local caller_addr="$1"
  local target_addr="$2"

  send_contract_tx "$caller_addr" "external-call caller pull_snapshot" "pull_snapshot(address)" "$target_addr"
}

send_multicontract_mark_local() {
  local caller_addr="$1"
  local value="$2"

  send_contract_tx "$caller_addr" "external-call caller mark_local" "mark_local(uint256)" "$value"
}

send_set_map_nested_struct_inner_middle() {
  local account="$1"
  local middle="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_inner_middle(address,uint256)" "$account" "$middle")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map nested struct set_inner_middle transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map nested struct set_inner_middle receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map nested struct set_inner_middle failed with status=$status"
}

send_set_map_deep_nested_struct() {
  local account="$1"
  local head="$2"
  local before="$3"
  local left="$4"
  local middle="$5"
  local right="$6"
  local after="$7"
  local tail="$8"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set(address,uint256,uint256,uint256,uint256,uint256,uint256,uint256)" "$account" "$head" "$before" "$left" "$middle" "$right" "$after" "$tail")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map deep nested struct set transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map deep nested struct set receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map deep nested struct set failed with status=$status"
}

send_set_map_deep_nested_struct_leaf_middle() {
  local account="$1"
  local middle="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_leaf_middle(address,uint256)" "$account" "$middle")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map deep nested struct set_leaf_middle transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map deep nested struct set_leaf_middle receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map deep nested struct set_leaf_middle failed with status=$status"
}

send_set_map_deep_nested_struct_mid_after() {
  local account="$1"
  local after="$2"
  local out tx_hash receipt status

  out="$(cast send --no-proxy --rpc-url "$RPC_URL" --async --unlocked --from "$DEPLOYER" "$CONTRACT_ADDR" "set_mid_after(address,uint256)" "$account" "$after")"
  tx_hash="$(extract_tx_hash "$out")"
  [[ -n "$tx_hash" ]] || fail "failed to extract map deep nested struct set_mid_after transaction hash"

  receipt="$(wait_receipt_json "$tx_hash")" || fail "timed out waiting for map deep nested struct set_mid_after receipt"
  status="$(receipt_json_field "$receipt" status)"
  [[ "$status" == "0x1" || "$status" == "1" ]] || fail "map deep nested struct set_mid_after failed with status=$status"
}

require_cmd cast
require_cmd python3
require_cmd "$ORA_BIN"
start_anvil_if_needed

mkdir -p "$WORK_DIR"
BYTECODE_FILE="$WORK_DIR/counter.hex"

BYTECODE="$(compile_bytecode "$ORA_SOURCE" "$BYTECODE_FILE")"

echo "Deploying bytecode"
CONTRACT_ADDR="$(deploy_contract "$BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_storage_and_getter 0
for expected in 1 2 3; do
  send_increment
  assert_storage_and_getter "$expected"
done

MAPPING_SOURCE="$WORK_DIR/mapping_storage_smoke.ora"
MAPPING_BYTECODE_FILE="$WORK_DIR/mapping_storage_smoke.hex"
write_mapping_fixture "$MAPPING_SOURCE"
MAPPING_BYTECODE="$(compile_bytecode "$MAPPING_SOURCE" "$MAPPING_BYTECODE_FILE")"

echo "Deploying mapping bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAPPING_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_mapping_raw_slot "$ALICE" 0
assert_mapping_raw_slot "$BOB" 0
assert_mapping_getter "$ALICE" 0
assert_mapping_getter "$BOB" 0
send_set_balance "$ALICE" 42
assert_mapping_raw_slot "$ALICE" 42
assert_mapping_raw_slot "$BOB" 0
assert_mapping_getter "$ALICE" 42
assert_mapping_getter "$BOB" 0
send_set_balance "$BOB" 7
assert_mapping_raw_slot "$ALICE" 42
assert_mapping_raw_slot "$BOB" 7
assert_mapping_getter "$ALICE" 42
assert_mapping_getter "$BOB" 7

mapping_root_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
[[ "${mapping_root_raw,,}" == "$(expected_slot_hex 0)" ]] || fail "mapping root slot 0 should remain empty; got=$mapping_root_raw"
ok "mapping root slot 0 remains empty"

U256_MAPPING_SOURCE="$WORK_DIR/u256_mapping_storage_smoke.ora"
U256_MAPPING_BYTECODE_FILE="$WORK_DIR/u256_mapping_storage_smoke.hex"
write_u256_mapping_fixture "$U256_MAPPING_SOURCE"
U256_MAPPING_BYTECODE="$(compile_bytecode "$U256_MAPPING_SOURCE" "$U256_MAPPING_BYTECODE_FILE")"

echo "Deploying u256 mapping bytecode"
CONTRACT_ADDR="$(deploy_contract "$U256_MAPPING_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

U256_KEY_A=1
U256_KEY_B="$(python3 -c 'print(hex((1 << 200) + 7))')"

assert_u256_mapping_raw_slot "$U256_KEY_A" 0
assert_u256_mapping_raw_slot "$U256_KEY_B" 0
assert_u256_mapping_getter "$U256_KEY_A" 0
assert_u256_mapping_getter "$U256_KEY_B" 0
send_set_u256_mapping_score "$U256_KEY_A" 111
assert_u256_mapping_raw_slot "$U256_KEY_A" 111
assert_u256_mapping_raw_slot "$U256_KEY_B" 0
assert_u256_mapping_getter "$U256_KEY_A" 111
assert_u256_mapping_getter "$U256_KEY_B" 0
send_set_u256_mapping_score "$U256_KEY_B" 222
assert_u256_mapping_raw_slot "$U256_KEY_A" 111
assert_u256_mapping_raw_slot "$U256_KEY_B" 222
assert_u256_mapping_getter "$U256_KEY_A" 111
assert_u256_mapping_getter "$U256_KEY_B" 222
u256_mapping_root_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
[[ "${u256_mapping_root_raw,,}" == "$(expected_slot_hex 0)" ]] || fail "u256 mapping root slot 0 should remain empty; got=$u256_mapping_root_raw"
ok "u256 mapping root slot 0 remains empty"

NARROW_INTEGER_MAPPING_SOURCE="$WORK_DIR/narrow_integer_mapping_storage_smoke.ora"
NARROW_INTEGER_MAPPING_BYTECODE_FILE="$WORK_DIR/narrow_integer_mapping_storage_smoke.hex"
write_narrow_integer_mapping_fixture "$NARROW_INTEGER_MAPPING_SOURCE"
NARROW_INTEGER_MAPPING_BYTECODE="$(compile_bytecode "$NARROW_INTEGER_MAPPING_SOURCE" "$NARROW_INTEGER_MAPPING_BYTECODE_FILE")"

echo "Deploying narrow-integer mapping bytecode"
CONTRACT_ADDR="$(deploy_contract "$NARROW_INTEGER_MAPPING_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

UINT8_KEY_A=1
UINT8_KEY_B=255
UINT16_KEY_A=258
UINT16_KEY_B=65535
UINT32_KEY_A=16909060
UINT32_KEY_B=4294967295
UINT128_KEY_A=0x0102030405060708090a0b0c0d0e0f10
UINT128_KEY_B=0xffffffffffffffffffffffffffffffff
INT8_KEY_A=-1
INT8_KEY_B=127
INT16_KEY_A=-12345
INT16_KEY_B=12345

assert_integer_mapping_raw_slot u8 uint8 "$UINT8_KEY_A" 0 0
assert_integer_mapping_raw_slot u8 uint8 "$UINT8_KEY_B" 0 0
assert_integer_mapping_getter u8 uint8 get_u8 "$UINT8_KEY_A" 0
assert_integer_mapping_getter u8 uint8 get_u8 "$UINT8_KEY_B" 0
send_set_integer_mapping_score u8 uint8 set_u8 "$UINT8_KEY_A" 11
assert_integer_mapping_raw_slot u8 uint8 "$UINT8_KEY_A" 0 11
assert_integer_mapping_raw_slot u8 uint8 "$UINT8_KEY_B" 0 0
assert_integer_mapping_getter u8 uint8 get_u8 "$UINT8_KEY_A" 11
assert_integer_mapping_getter u8 uint8 get_u8 "$UINT8_KEY_B" 0
send_set_integer_mapping_score u8 uint8 set_u8 "$UINT8_KEY_B" 12
assert_integer_mapping_raw_slot u8 uint8 "$UINT8_KEY_A" 0 11
assert_integer_mapping_raw_slot u8 uint8 "$UINT8_KEY_B" 0 12
assert_integer_mapping_getter u8 uint8 get_u8 "$UINT8_KEY_A" 11
assert_integer_mapping_getter u8 uint8 get_u8 "$UINT8_KEY_B" 12

assert_integer_mapping_raw_slot u16 uint16 "$UINT16_KEY_A" 1 0
assert_integer_mapping_raw_slot u16 uint16 "$UINT16_KEY_B" 1 0
send_set_integer_mapping_score u16 uint16 set_u16 "$UINT16_KEY_A" 21
send_set_integer_mapping_score u16 uint16 set_u16 "$UINT16_KEY_B" 22
assert_integer_mapping_raw_slot u16 uint16 "$UINT16_KEY_A" 1 21
assert_integer_mapping_raw_slot u16 uint16 "$UINT16_KEY_B" 1 22
assert_integer_mapping_getter u16 uint16 get_u16 "$UINT16_KEY_A" 21
assert_integer_mapping_getter u16 uint16 get_u16 "$UINT16_KEY_B" 22

assert_integer_mapping_raw_slot u32 uint32 "$UINT32_KEY_A" 2 0
assert_integer_mapping_raw_slot u32 uint32 "$UINT32_KEY_B" 2 0
send_set_integer_mapping_score u32 uint32 set_u32 "$UINT32_KEY_A" 31
send_set_integer_mapping_score u32 uint32 set_u32 "$UINT32_KEY_B" 32
assert_integer_mapping_raw_slot u32 uint32 "$UINT32_KEY_A" 2 31
assert_integer_mapping_raw_slot u32 uint32 "$UINT32_KEY_B" 2 32
assert_integer_mapping_getter u32 uint32 get_u32 "$UINT32_KEY_A" 31
assert_integer_mapping_getter u32 uint32 get_u32 "$UINT32_KEY_B" 32

assert_integer_mapping_raw_slot u128 uint128 "$UINT128_KEY_A" 3 0
assert_integer_mapping_raw_slot u128 uint128 "$UINT128_KEY_B" 3 0
send_set_integer_mapping_score u128 uint128 set_u128 "$UINT128_KEY_A" 41
send_set_integer_mapping_score u128 uint128 set_u128 "$UINT128_KEY_B" 42
assert_integer_mapping_raw_slot u128 uint128 "$UINT128_KEY_A" 3 41
assert_integer_mapping_raw_slot u128 uint128 "$UINT128_KEY_B" 3 42
assert_integer_mapping_getter u128 uint128 get_u128 "$UINT128_KEY_A" 41
assert_integer_mapping_getter u128 uint128 get_u128 "$UINT128_KEY_B" 42

assert_integer_mapping_raw_slot i8 int8 "$INT8_KEY_A" 4 0
assert_integer_mapping_raw_slot i8 int8 "$INT8_KEY_B" 4 0
send_set_integer_mapping_score i8 int8 set_i8 "$INT8_KEY_A" 51
send_set_integer_mapping_score i8 int8 set_i8 "$INT8_KEY_B" 52
assert_integer_mapping_raw_slot i8 int8 "$INT8_KEY_A" 4 51
assert_integer_mapping_raw_slot i8 int8 "$INT8_KEY_B" 4 52
assert_integer_mapping_getter i8 int8 get_i8 "$INT8_KEY_A" 51
assert_integer_mapping_getter i8 int8 get_i8 "$INT8_KEY_B" 52

assert_integer_mapping_raw_slot i16 int16 "$INT16_KEY_A" 5 0
assert_integer_mapping_raw_slot i16 int16 "$INT16_KEY_B" 5 0
send_set_integer_mapping_score i16 int16 set_i16 "$INT16_KEY_A" 61
send_set_integer_mapping_score i16 int16 set_i16 "$INT16_KEY_B" 62
assert_integer_mapping_raw_slot i16 int16 "$INT16_KEY_A" 5 61
assert_integer_mapping_raw_slot i16 int16 "$INT16_KEY_B" 5 62
assert_integer_mapping_getter i16 int16 get_i16 "$INT16_KEY_A" 61
assert_integer_mapping_getter i16 int16 get_i16 "$INT16_KEY_B" 62

for narrow_integer_root_slot in 0 1 2 3 4 5; do
  narrow_integer_root_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$narrow_integer_root_slot")")"
  [[ "${narrow_integer_root_raw,,}" == "$(expected_slot_hex 0)" ]] || fail "narrow-integer mapping root slot $narrow_integer_root_slot should remain empty; got=$narrow_integer_root_raw"
done
ok "narrow-integer mapping root slots remain empty"

FIXED_BYTES_MAPPING_SOURCE="$WORK_DIR/fixed_bytes_mapping_storage_smoke.ora"
FIXED_BYTES_MAPPING_BYTECODE_FILE="$WORK_DIR/fixed_bytes_mapping_storage_smoke.hex"
write_fixed_bytes_mapping_fixture "$FIXED_BYTES_MAPPING_SOURCE"
FIXED_BYTES_MAPPING_BYTECODE="$(compile_bytecode "$FIXED_BYTES_MAPPING_SOURCE" "$FIXED_BYTES_MAPPING_BYTECODE_FILE")"

echo "Deploying fixed-bytes mapping bytecode"
CONTRACT_ADDR="$(deploy_contract "$FIXED_BYTES_MAPPING_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

FIXED_BYTES1_KEY_A="0x12"
FIXED_BYTES1_KEY_B="0xab"
FIXED_BYTES4_KEY_A="0x12345678"
FIXED_BYTES4_KEY_B="0xdeadbeef"
FIXED_BYTES20_KEY_A="0x1111111111111111111111111111111111111111"
FIXED_BYTES20_KEY_B="0x2222222222222222222222222222222222222222"
FIXED_BYTES31_KEY_A="$(python3 -c 'print("0x" + "11" * 31)')"
FIXED_BYTES31_KEY_B="$(python3 -c 'print("0x" + "aa" * 31)')"
FIXED_BYTES32_KEY_A="$(python3 -c 'print("0x" + "11" * 32)')"
FIXED_BYTES32_KEY_B="$(python3 -c 'print("0x" + "aa" * 32)')"

assert_fixed_bytes_mapping_raw_slot bytes1 bytes1 "$FIXED_BYTES1_KEY_A" 0 0
assert_fixed_bytes_mapping_raw_slot bytes1 bytes1 "$FIXED_BYTES1_KEY_B" 0 0
assert_fixed_bytes_mapping_getter bytes1 bytes1 get_b1 "$FIXED_BYTES1_KEY_A" 0
assert_fixed_bytes_mapping_getter bytes1 bytes1 get_b1 "$FIXED_BYTES1_KEY_B" 0
send_set_fixed_bytes_mapping_score bytes1 bytes1 set_b1 "$FIXED_BYTES1_KEY_A" 101
assert_fixed_bytes_mapping_raw_slot bytes1 bytes1 "$FIXED_BYTES1_KEY_A" 0 101
assert_fixed_bytes_mapping_raw_slot bytes1 bytes1 "$FIXED_BYTES1_KEY_B" 0 0
assert_fixed_bytes_mapping_getter bytes1 bytes1 get_b1 "$FIXED_BYTES1_KEY_A" 101
assert_fixed_bytes_mapping_getter bytes1 bytes1 get_b1 "$FIXED_BYTES1_KEY_B" 0
send_set_fixed_bytes_mapping_score bytes1 bytes1 set_b1 "$FIXED_BYTES1_KEY_B" 102
assert_fixed_bytes_mapping_raw_slot bytes1 bytes1 "$FIXED_BYTES1_KEY_A" 0 101
assert_fixed_bytes_mapping_raw_slot bytes1 bytes1 "$FIXED_BYTES1_KEY_B" 0 102
assert_fixed_bytes_mapping_getter bytes1 bytes1 get_b1 "$FIXED_BYTES1_KEY_A" 101
assert_fixed_bytes_mapping_getter bytes1 bytes1 get_b1 "$FIXED_BYTES1_KEY_B" 102

assert_fixed_bytes_mapping_raw_slot bytes4 bytes4 "$FIXED_BYTES4_KEY_A" 1 0
assert_fixed_bytes_mapping_raw_slot bytes4 bytes4 "$FIXED_BYTES4_KEY_B" 1 0
assert_fixed_bytes_mapping_getter bytes4 bytes4 get_b4 "$FIXED_BYTES4_KEY_A" 0
assert_fixed_bytes_mapping_getter bytes4 bytes4 get_b4 "$FIXED_BYTES4_KEY_B" 0
send_set_fixed_bytes_mapping_score bytes4 bytes4 set_b4 "$FIXED_BYTES4_KEY_A" 201
send_set_fixed_bytes_mapping_score bytes4 bytes4 set_b4 "$FIXED_BYTES4_KEY_B" 202
assert_fixed_bytes_mapping_raw_slot bytes4 bytes4 "$FIXED_BYTES4_KEY_A" 1 201
assert_fixed_bytes_mapping_raw_slot bytes4 bytes4 "$FIXED_BYTES4_KEY_B" 1 202
assert_fixed_bytes_mapping_getter bytes4 bytes4 get_b4 "$FIXED_BYTES4_KEY_A" 201
assert_fixed_bytes_mapping_getter bytes4 bytes4 get_b4 "$FIXED_BYTES4_KEY_B" 202

assert_fixed_bytes_mapping_raw_slot bytes20 bytes20 "$FIXED_BYTES20_KEY_A" 2 0
assert_fixed_bytes_mapping_raw_slot bytes20 bytes20 "$FIXED_BYTES20_KEY_B" 2 0
send_set_fixed_bytes_mapping_score bytes20 bytes20 set_b20 "$FIXED_BYTES20_KEY_A" 301
send_set_fixed_bytes_mapping_score bytes20 bytes20 set_b20 "$FIXED_BYTES20_KEY_B" 302
assert_fixed_bytes_mapping_raw_slot bytes20 bytes20 "$FIXED_BYTES20_KEY_A" 2 301
assert_fixed_bytes_mapping_raw_slot bytes20 bytes20 "$FIXED_BYTES20_KEY_B" 2 302
assert_fixed_bytes_mapping_getter bytes20 bytes20 get_b20 "$FIXED_BYTES20_KEY_A" 301
assert_fixed_bytes_mapping_getter bytes20 bytes20 get_b20 "$FIXED_BYTES20_KEY_B" 302

assert_fixed_bytes_mapping_raw_slot bytes31 bytes31 "$FIXED_BYTES31_KEY_A" 3 0
assert_fixed_bytes_mapping_raw_slot bytes31 bytes31 "$FIXED_BYTES31_KEY_B" 3 0
send_set_fixed_bytes_mapping_score bytes31 bytes31 set_b31 "$FIXED_BYTES31_KEY_A" 401
send_set_fixed_bytes_mapping_score bytes31 bytes31 set_b31 "$FIXED_BYTES31_KEY_B" 402
assert_fixed_bytes_mapping_raw_slot bytes31 bytes31 "$FIXED_BYTES31_KEY_A" 3 401
assert_fixed_bytes_mapping_raw_slot bytes31 bytes31 "$FIXED_BYTES31_KEY_B" 3 402
assert_fixed_bytes_mapping_getter bytes31 bytes31 get_b31 "$FIXED_BYTES31_KEY_A" 401
assert_fixed_bytes_mapping_getter bytes31 bytes31 get_b31 "$FIXED_BYTES31_KEY_B" 402

assert_fixed_bytes_mapping_raw_slot bytes32 bytes32 "$FIXED_BYTES32_KEY_A" 4 0
assert_fixed_bytes_mapping_raw_slot bytes32 bytes32 "$FIXED_BYTES32_KEY_B" 4 0
send_set_fixed_bytes_mapping_score bytes32 bytes32 set_b32 "$FIXED_BYTES32_KEY_A" 501
send_set_fixed_bytes_mapping_score bytes32 bytes32 set_b32 "$FIXED_BYTES32_KEY_B" 502
assert_fixed_bytes_mapping_raw_slot bytes32 bytes32 "$FIXED_BYTES32_KEY_A" 4 501
assert_fixed_bytes_mapping_raw_slot bytes32 bytes32 "$FIXED_BYTES32_KEY_B" 4 502
assert_fixed_bytes_mapping_getter bytes32 bytes32 get_b32 "$FIXED_BYTES32_KEY_A" 501
assert_fixed_bytes_mapping_getter bytes32 bytes32 get_b32 "$FIXED_BYTES32_KEY_B" 502

for fixed_bytes_root_slot in 0 1 2 3 4; do
  fixed_bytes_root_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" "$fixed_bytes_root_slot")")"
  [[ "${fixed_bytes_root_raw,,}" == "$(expected_slot_hex 0)" ]] || fail "fixed-bytes mapping root slot $fixed_bytes_root_slot should remain empty; got=$fixed_bytes_root_raw"
done
ok "fixed-bytes mapping root slots remain empty"

NESTED_MAPPING_SOURCE="$WORK_DIR/nested_mapping_storage_smoke.ora"
NESTED_MAPPING_BYTECODE_FILE="$WORK_DIR/nested_mapping_storage_smoke.hex"
write_nested_mapping_fixture "$NESTED_MAPPING_SOURCE"
NESTED_MAPPING_BYTECODE="$(compile_bytecode "$NESTED_MAPPING_SOURCE" "$NESTED_MAPPING_BYTECODE_FILE")"

echo "Deploying nested mapping bytecode"
CONTRACT_ADDR="$(deploy_contract "$NESTED_MAPPING_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_nested_mapping_raw_slot "$ALICE" "$BOB" 0
assert_nested_mapping_raw_slot "$ALICE" "$CHARLIE" 0
assert_nested_mapping_raw_slot "$BOB" "$ALICE" 0
assert_nested_mapping_getter "$ALICE" "$BOB" 0
assert_nested_mapping_getter "$ALICE" "$CHARLIE" 0
assert_nested_mapping_getter "$BOB" "$ALICE" 0
send_set_allowance "$ALICE" "$BOB" 11
assert_nested_mapping_raw_slot "$ALICE" "$BOB" 11
assert_nested_mapping_raw_slot "$ALICE" "$CHARLIE" 0
assert_nested_mapping_raw_slot "$BOB" "$ALICE" 0
assert_nested_mapping_getter "$ALICE" "$BOB" 11
assert_nested_mapping_getter "$ALICE" "$CHARLIE" 0
assert_nested_mapping_getter "$BOB" "$ALICE" 0
send_set_allowance "$BOB" "$ALICE" 13
assert_nested_mapping_raw_slot "$ALICE" "$BOB" 11
assert_nested_mapping_raw_slot "$BOB" "$ALICE" 13
assert_nested_mapping_getter "$ALICE" "$BOB" 11
assert_nested_mapping_getter "$BOB" "$ALICE" 13

nested_mapping_root_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
[[ "${nested_mapping_root_raw,,}" == "$(expected_slot_hex 0)" ]] || fail "nested mapping root slot 0 should remain empty; got=$nested_mapping_root_raw"
ok "nested mapping root slot 0 remains empty"

TRIPLE_NESTED_MAPPING_SOURCE="$WORK_DIR/triple_nested_mapping_storage_smoke.ora"
TRIPLE_NESTED_MAPPING_BYTECODE_FILE="$WORK_DIR/triple_nested_mapping_storage_smoke.hex"
write_triple_nested_mapping_fixture "$TRIPLE_NESTED_MAPPING_SOURCE"
TRIPLE_NESTED_MAPPING_BYTECODE="$(compile_bytecode "$TRIPLE_NESTED_MAPPING_SOURCE" "$TRIPLE_NESTED_MAPPING_BYTECODE_FILE")"

echo "Deploying triple nested mapping bytecode"
CONTRACT_ADDR="$(deploy_contract "$TRIPLE_NESTED_MAPPING_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_triple_nested_mapping_raw_slot "$ALICE" "$BOB" "$CHARLIE" 0
assert_triple_nested_mapping_raw_slot "$ALICE" "$CHARLIE" "$BOB" 0
assert_triple_nested_mapping_raw_slot "$BOB" "$ALICE" "$CHARLIE" 0
assert_triple_nested_mapping_getter "$ALICE" "$BOB" "$CHARLIE" 0
assert_triple_nested_mapping_getter "$ALICE" "$CHARLIE" "$BOB" 0
assert_triple_nested_mapping_getter "$BOB" "$ALICE" "$CHARLIE" 0
send_set_triple_nested_limit "$ALICE" "$BOB" "$CHARLIE" 17
assert_triple_nested_mapping_raw_slot "$ALICE" "$BOB" "$CHARLIE" 17
assert_triple_nested_mapping_raw_slot "$ALICE" "$CHARLIE" "$BOB" 0
assert_triple_nested_mapping_raw_slot "$BOB" "$ALICE" "$CHARLIE" 0
assert_triple_nested_mapping_getter "$ALICE" "$BOB" "$CHARLIE" 17
assert_triple_nested_mapping_getter "$ALICE" "$CHARLIE" "$BOB" 0
assert_triple_nested_mapping_getter "$BOB" "$ALICE" "$CHARLIE" 0
send_set_triple_nested_limit "$ALICE" "$CHARLIE" "$BOB" 19
assert_triple_nested_mapping_raw_slot "$ALICE" "$BOB" "$CHARLIE" 17
assert_triple_nested_mapping_raw_slot "$ALICE" "$CHARLIE" "$BOB" 19
assert_triple_nested_mapping_raw_slot "$BOB" "$ALICE" "$CHARLIE" 0
assert_triple_nested_mapping_getter "$ALICE" "$BOB" "$CHARLIE" 17
assert_triple_nested_mapping_getter "$ALICE" "$CHARLIE" "$BOB" 19
assert_triple_nested_mapping_getter "$BOB" "$ALICE" "$CHARLIE" 0
send_set_triple_nested_limit "$BOB" "$ALICE" "$CHARLIE" 23
assert_triple_nested_mapping_raw_slot "$ALICE" "$BOB" "$CHARLIE" 17
assert_triple_nested_mapping_raw_slot "$ALICE" "$CHARLIE" "$BOB" 19
assert_triple_nested_mapping_raw_slot "$BOB" "$ALICE" "$CHARLIE" 23
assert_triple_nested_mapping_getter "$ALICE" "$BOB" "$CHARLIE" 17
assert_triple_nested_mapping_getter "$ALICE" "$CHARLIE" "$BOB" 19
assert_triple_nested_mapping_getter "$BOB" "$ALICE" "$CHARLIE" 23

triple_nested_mapping_root_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
[[ "${triple_nested_mapping_root_raw,,}" == "$(expected_slot_hex 0)" ]] || fail "triple nested mapping root slot 0 should remain empty; got=$triple_nested_mapping_root_raw"
ok "triple nested mapping root slot 0 remains empty"

NESTED_MAP_STRUCT_SOURCE="$WORK_DIR/nested_map_struct_storage_smoke.ora"
NESTED_MAP_STRUCT_BYTECODE_FILE="$WORK_DIR/nested_map_struct_storage_smoke.hex"
write_nested_map_struct_fixture "$NESTED_MAP_STRUCT_SOURCE"
NESTED_MAP_STRUCT_BYTECODE="$(compile_bytecode "$NESTED_MAP_STRUCT_SOURCE" "$NESTED_MAP_STRUCT_BYTECODE_FILE")"

echo "Deploying nested map struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$NESTED_MAP_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_nested_map_struct_raw_slots "$ALICE" "$BOB" 0 0 0
assert_nested_map_struct_raw_slots "$ALICE" "$CHARLIE" 0 0 0
assert_nested_map_struct_raw_slots "$BOB" "$ALICE" 0 0 0
assert_nested_map_struct_getters "$ALICE" "$BOB" 0 0 0
assert_nested_map_struct_getters "$ALICE" "$CHARLIE" 0 0 0
assert_nested_map_struct_getters "$BOB" "$ALICE" 0 0 0
send_set_nested_map_struct "$ALICE" "$BOB" 101 202 303
assert_nested_map_struct_raw_slots "$ALICE" "$BOB" 101 202 303
assert_nested_map_struct_raw_slots "$ALICE" "$CHARLIE" 0 0 0
assert_nested_map_struct_raw_slots "$BOB" "$ALICE" 0 0 0
assert_nested_map_struct_getters "$ALICE" "$BOB" 101 202 303
assert_nested_map_struct_getters "$ALICE" "$CHARLIE" 0 0 0
assert_nested_map_struct_getters "$BOB" "$ALICE" 0 0 0
send_set_nested_map_struct "$BOB" "$ALICE" 404 505 606
assert_nested_map_struct_raw_slots "$ALICE" "$BOB" 101 202 303
assert_nested_map_struct_raw_slots "$BOB" "$ALICE" 404 505 606
assert_nested_map_struct_getters "$ALICE" "$BOB" 101 202 303
assert_nested_map_struct_getters "$BOB" "$ALICE" 404 505 606
send_set_nested_map_struct_nonce "$ALICE" "$BOB" 777
assert_nested_map_struct_raw_slots "$ALICE" "$BOB" 101 777 303
assert_nested_map_struct_raw_slots "$BOB" "$ALICE" 404 505 606
assert_nested_map_struct_getters "$ALICE" "$BOB" 101 777 303
assert_nested_map_struct_getters "$BOB" "$ALICE" 404 505 606

nested_map_struct_root_raw="$(trim "$(cast storage --no-proxy --rpc-url "$RPC_URL" "$CONTRACT_ADDR" 0)")"
[[ "${nested_map_struct_root_raw,,}" == "$(expected_slot_hex 0)" ]] || fail "nested map struct root slot 0 should remain empty; got=$nested_map_struct_root_raw"
ok "nested map struct root slot 0 remains empty"

DYNAMIC_ARRAY_SOURCE="$WORK_DIR/dynamic_array_storage_smoke.ora"
DYNAMIC_ARRAY_BYTECODE_FILE="$WORK_DIR/dynamic_array_storage_smoke.hex"
write_dynamic_array_fixture "$DYNAMIC_ARRAY_SOURCE"
DYNAMIC_ARRAY_BYTECODE="$(compile_bytecode "$DYNAMIC_ARRAY_SOURCE" "$DYNAMIC_ARRAY_BYTECODE_FILE")"

echo "Deploying dynamic array bytecode"
CONTRACT_ADDR="$(deploy_contract "$DYNAMIC_ARRAY_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_dynamic_array_length 0
assert_dynamic_array_raw_slot 0 0
assert_dynamic_array_raw_slot 1 0
send_set_dynamic_array "[5,8,13]"
assert_dynamic_array_length 3
assert_dynamic_array_raw_slot 0 5
assert_dynamic_array_raw_slot 1 8
assert_dynamic_array_raw_slot 2 13
assert_dynamic_array_getter 0 5
assert_dynamic_array_getter 1 8
assert_dynamic_array_getter 2 13

STRUCT_SOURCE="$WORK_DIR/struct_storage_smoke.ora"
STRUCT_BYTECODE_FILE="$WORK_DIR/struct_storage_smoke.hex"
write_struct_fixture "$STRUCT_SOURCE"
STRUCT_BYTECODE="$(compile_bytecode "$STRUCT_SOURCE" "$STRUCT_BYTECODE_FILE")"

echo "Deploying struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_struct_raw_slots 0 0 0
assert_struct_getters 0 0 0
send_set_struct 17 23 31
assert_struct_raw_slots 17 23 31
assert_struct_getters 17 23 31
send_set_struct_middle 99
assert_struct_raw_slots 17 99 31
assert_struct_getters 17 99 31

MAP_STRUCT_SOURCE="$WORK_DIR/map_struct_storage_smoke.ora"
MAP_STRUCT_BYTECODE_FILE="$WORK_DIR/map_struct_storage_smoke.hex"
write_map_struct_fixture "$MAP_STRUCT_SOURCE"
MAP_STRUCT_BYTECODE="$(compile_bytecode "$MAP_STRUCT_SOURCE" "$MAP_STRUCT_BYTECODE_FILE")"

echo "Deploying map struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAP_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_map_struct_raw_slots "$ALICE" 0 0 0
assert_map_struct_raw_slots "$BOB" 0 0 0
assert_map_struct_getters "$ALICE" 0 0 0
assert_map_struct_getters "$BOB" 0 0 0
send_set_map_struct "$ALICE" 101 202 303
assert_map_struct_raw_slots "$ALICE" 101 202 303
assert_map_struct_raw_slots "$BOB" 0 0 0
assert_map_struct_getters "$ALICE" 101 202 303
assert_map_struct_getters "$BOB" 0 0 0
send_set_map_struct "$BOB" 7 8 9
assert_map_struct_raw_slots "$ALICE" 101 202 303
assert_map_struct_raw_slots "$BOB" 7 8 9
assert_map_struct_getters "$ALICE" 101 202 303
assert_map_struct_getters "$BOB" 7 8 9
send_set_map_struct_nonce "$ALICE" 404
assert_map_struct_raw_slots "$ALICE" 101 404 303
assert_map_struct_raw_slots "$BOB" 7 8 9
assert_map_struct_getters "$ALICE" 101 404 303
assert_map_struct_getters "$BOB" 7 8 9

SLICE_STRUCT_SOURCE="$WORK_DIR/slice_struct_storage_smoke.ora"
SLICE_STRUCT_BYTECODE_FILE="$WORK_DIR/slice_struct_storage_smoke.hex"
write_slice_struct_fixture "$SLICE_STRUCT_SOURCE"
SLICE_STRUCT_BYTECODE="$(compile_bytecode "$SLICE_STRUCT_SOURCE" "$SLICE_STRUCT_BYTECODE_FILE")"

echo "Deploying slice struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$SLICE_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_slice_struct_length 0
assert_slice_struct_raw_slots 0 0 0 0
assert_slice_struct_raw_slots 1 0 0 0
send_set_slice_struct "[(11,12,13),(21,22,23)]"
assert_slice_struct_length 2
assert_slice_struct_raw_slots 0 11 12 13
assert_slice_struct_raw_slots 1 21 22 23
assert_slice_struct_getters 0 11 12 13
assert_slice_struct_getters 1 21 22 23
send_set_slice_struct_middle 1 99
assert_slice_struct_raw_slots 0 11 12 13
assert_slice_struct_raw_slots 1 21 99 23
assert_slice_struct_getters 0 11 12 13
assert_slice_struct_getters 1 21 99 23

STRUCT_SLICE_SOURCE="$WORK_DIR/struct_slice_storage_smoke.ora"
STRUCT_SLICE_BYTECODE_FILE="$WORK_DIR/struct_slice_storage_smoke.hex"
write_struct_slice_fixture "$STRUCT_SLICE_SOURCE"
STRUCT_SLICE_BYTECODE="$(compile_bytecode "$STRUCT_SLICE_SOURCE" "$STRUCT_SLICE_BYTECODE_FILE")"

echo "Deploying struct slice bytecode"
CONTRACT_ADDR="$(deploy_contract "$STRUCT_SLICE_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_struct_slice_length 0
assert_struct_slice_raw_slot 0 0
assert_struct_slice_raw_slot 1 0
send_set_struct_slice "[34,55,89]"
assert_struct_slice_length 3
assert_struct_slice_raw_slot 0 34
assert_struct_slice_raw_slot 1 55
assert_struct_slice_raw_slot 2 89
assert_struct_slice_getter 0 34
assert_struct_slice_getter 1 55
assert_struct_slice_getter 2 89

MULTI_STRUCT_SLICE_SOURCE="$WORK_DIR/multi_struct_slice_storage_smoke.ora"
MULTI_STRUCT_SLICE_BYTECODE_FILE="$WORK_DIR/multi_struct_slice_storage_smoke.hex"
write_multi_struct_slice_fixture "$MULTI_STRUCT_SLICE_SOURCE"
MULTI_STRUCT_SLICE_BYTECODE="$(compile_bytecode "$MULTI_STRUCT_SLICE_SOURCE" "$MULTI_STRUCT_SLICE_BYTECODE_FILE")"

echo "Deploying multi struct slice bytecode"
CONTRACT_ADDR="$(deploy_contract "$MULTI_STRUCT_SLICE_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_multi_struct_slice_scalars 0 0 0
assert_multi_struct_slice_raw_slot 0 0
assert_multi_struct_slice_raw_slot 1 0
assert_multi_struct_slice_getters 0 0
send_set_multi_struct_slice 17 "[34,55,89]" 31
assert_multi_struct_slice_scalars 17 3 31
assert_multi_struct_slice_raw_slot 0 34
assert_multi_struct_slice_raw_slot 1 55
assert_multi_struct_slice_raw_slot 2 89
assert_multi_struct_slice_getters 17 31
assert_multi_struct_slice_getter 0 34
assert_multi_struct_slice_getter 1 55
assert_multi_struct_slice_getter 2 89
send_set_multi_struct_slice_head 99
assert_multi_struct_slice_scalars 99 3 31
assert_multi_struct_slice_raw_slot 0 34
assert_multi_struct_slice_raw_slot 1 55
assert_multi_struct_slice_raw_slot 2 89
assert_multi_struct_slice_getters 99 31
send_set_multi_struct_slice_tail 77
assert_multi_struct_slice_scalars 99 3 77
assert_multi_struct_slice_raw_slot 0 34
assert_multi_struct_slice_raw_slot 1 55
assert_multi_struct_slice_raw_slot 2 89
assert_multi_struct_slice_getters 99 77
send_set_multi_struct_slice_values "[1,2]"
assert_multi_struct_slice_scalars 99 2 77
assert_multi_struct_slice_raw_slot 0 1
assert_multi_struct_slice_raw_slot 1 2
assert_multi_struct_slice_getters 99 77
assert_multi_struct_slice_getter 0 1
assert_multi_struct_slice_getter 1 2

DUAL_STRUCT_SLICE_SOURCE="$WORK_DIR/dual_struct_slice_storage_smoke.ora"
DUAL_STRUCT_SLICE_BYTECODE_FILE="$WORK_DIR/dual_struct_slice_storage_smoke.hex"
write_dual_struct_slice_fixture "$DUAL_STRUCT_SLICE_SOURCE"
DUAL_STRUCT_SLICE_BYTECODE="$(compile_bytecode "$DUAL_STRUCT_SLICE_SOURCE" "$DUAL_STRUCT_SLICE_BYTECODE_FILE")"

echo "Deploying dual struct slice bytecode"
CONTRACT_ADDR="$(deploy_contract "$DUAL_STRUCT_SLICE_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_dual_struct_slice_scalars 0 0 0
assert_dual_struct_slice_left_slot 0 0
assert_dual_struct_slice_right_slot 0 0
assert_dual_struct_slice_getters 0
send_set_dual_struct_slice "[11,12]" 99 "[21,22,23]"
assert_dual_struct_slice_scalars 2 99 3
assert_dual_struct_slice_left_slot 0 11
assert_dual_struct_slice_left_slot 1 12
assert_dual_struct_slice_right_slot 0 21
assert_dual_struct_slice_right_slot 1 22
assert_dual_struct_slice_right_slot 2 23
assert_dual_struct_slice_getters 99
assert_dual_struct_slice_left_getter 0 11
assert_dual_struct_slice_left_getter 1 12
assert_dual_struct_slice_right_getter 0 21
assert_dual_struct_slice_right_getter 1 22
assert_dual_struct_slice_right_getter 2 23
send_set_dual_struct_slice_left "[5]"
assert_dual_struct_slice_scalars 1 99 3
assert_dual_struct_slice_left_slot 0 5
assert_dual_struct_slice_right_slot 0 21
assert_dual_struct_slice_right_slot 1 22
assert_dual_struct_slice_right_slot 2 23
assert_dual_struct_slice_getters 99
assert_dual_struct_slice_left_getter 0 5
assert_dual_struct_slice_right_getter 0 21
assert_dual_struct_slice_right_getter 1 22
assert_dual_struct_slice_right_getter 2 23
send_set_dual_struct_slice_right "[8,9]"
assert_dual_struct_slice_scalars 1 99 2
assert_dual_struct_slice_left_slot 0 5
assert_dual_struct_slice_right_slot 0 8
assert_dual_struct_slice_right_slot 1 9
assert_dual_struct_slice_getters 99
assert_dual_struct_slice_left_getter 0 5
assert_dual_struct_slice_right_getter 0 8
assert_dual_struct_slice_right_getter 1 9
send_set_dual_struct_slice_pivot 77
assert_dual_struct_slice_scalars 1 77 2
assert_dual_struct_slice_left_slot 0 5
assert_dual_struct_slice_right_slot 0 8
assert_dual_struct_slice_right_slot 1 9
assert_dual_struct_slice_getters 77
assert_dual_struct_slice_left_getter 0 5
assert_dual_struct_slice_right_getter 0 8
assert_dual_struct_slice_right_getter 1 9

SLICE_SLICE_SOURCE="$WORK_DIR/slice_slice_storage_smoke.ora"
SLICE_SLICE_BYTECODE_FILE="$WORK_DIR/slice_slice_storage_smoke.hex"
write_slice_slice_fixture "$SLICE_SLICE_SOURCE"
SLICE_SLICE_BYTECODE="$(compile_bytecode "$SLICE_SLICE_SOURCE" "$SLICE_SLICE_BYTECODE_FILE")"

echo "Deploying slice slice bytecode"
CONTRACT_ADDR="$(deploy_contract "$SLICE_SLICE_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_slice_slice_outer_length 0
assert_slice_slice_row_length 0 0
assert_slice_slice_row_length 1 0
assert_slice_slice_raw_slot 0 0 0
assert_slice_slice_raw_slot 1 0 0
send_set_slice_slice "[[11,12],[21,22,23]]"
assert_slice_slice_outer_length 2
assert_slice_slice_row_length 0 2
assert_slice_slice_row_length 1 3
assert_slice_slice_raw_slot 0 0 11
assert_slice_slice_raw_slot 0 1 12
assert_slice_slice_raw_slot 1 0 21
assert_slice_slice_raw_slot 1 1 22
assert_slice_slice_raw_slot 1 2 23
assert_slice_slice_getter 0 0 11
assert_slice_slice_getter 0 1 12
assert_slice_slice_getter 1 0 21
assert_slice_slice_getter 1 1 22
assert_slice_slice_getter 1 2 23
send_set_slice_slice "[[7],[8,9]]"
assert_slice_slice_outer_length 2
assert_slice_slice_row_length 0 1
assert_slice_slice_row_length 1 2
assert_slice_slice_raw_slot 0 0 7
assert_slice_slice_raw_slot 1 0 8
assert_slice_slice_raw_slot 1 1 9
assert_slice_slice_getter 0 0 7
assert_slice_slice_getter 1 0 8
assert_slice_slice_getter 1 1 9

SLICE_DYNAMIC_STRUCT_SOURCE="$WORK_DIR/slice_dynamic_struct_storage_smoke.ora"
SLICE_DYNAMIC_STRUCT_BYTECODE_FILE="$WORK_DIR/slice_dynamic_struct_storage_smoke.hex"
write_slice_dynamic_struct_fixture "$SLICE_DYNAMIC_STRUCT_SOURCE"
SLICE_DYNAMIC_STRUCT_BYTECODE="$(compile_bytecode "$SLICE_DYNAMIC_STRUCT_SOURCE" "$SLICE_DYNAMIC_STRUCT_BYTECODE_FILE")"

echo "Deploying slice dynamic struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$SLICE_DYNAMIC_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_slice_dynamic_struct_length 0
assert_slice_dynamic_struct_scalars 0 0 0 0
assert_slice_dynamic_struct_scalars 1 0 0 0
assert_slice_dynamic_struct_value_slot 0 0 0
assert_slice_dynamic_struct_value_slot 1 0 0
send_set_slice_dynamic_struct "[(11,[12,13],14),(21,[22,23,24],25)]"
assert_slice_dynamic_struct_length 2
assert_slice_dynamic_struct_scalars 0 11 2 14
assert_slice_dynamic_struct_scalars 1 21 3 25
assert_slice_dynamic_struct_value_slot 0 0 12
assert_slice_dynamic_struct_value_slot 0 1 13
assert_slice_dynamic_struct_value_slot 1 0 22
assert_slice_dynamic_struct_value_slot 1 1 23
assert_slice_dynamic_struct_value_slot 1 2 24
assert_slice_dynamic_struct_getters 0 11 14
assert_slice_dynamic_struct_getters 1 21 25
assert_slice_dynamic_struct_value_getter 0 0 12
assert_slice_dynamic_struct_value_getter 0 1 13
assert_slice_dynamic_struct_value_getter 1 0 22
assert_slice_dynamic_struct_value_getter 1 1 23
assert_slice_dynamic_struct_value_getter 1 2 24
send_set_slice_dynamic_struct_tail 1 99
assert_slice_dynamic_struct_scalars 0 11 2 14
assert_slice_dynamic_struct_scalars 1 21 3 99
assert_slice_dynamic_struct_value_slot 0 0 12
assert_slice_dynamic_struct_value_slot 0 1 13
assert_slice_dynamic_struct_value_slot 1 0 22
assert_slice_dynamic_struct_value_slot 1 1 23
assert_slice_dynamic_struct_value_slot 1 2 24
assert_slice_dynamic_struct_getters 0 11 14
assert_slice_dynamic_struct_getters 1 21 99
assert_slice_dynamic_struct_value_getter 0 0 12
assert_slice_dynamic_struct_value_getter 0 1 13
assert_slice_dynamic_struct_value_getter 1 0 22
assert_slice_dynamic_struct_value_getter 1 1 23
assert_slice_dynamic_struct_value_getter 1 2 24
send_set_slice_dynamic_struct_values 0 "[7]"
assert_slice_dynamic_struct_scalars 0 11 1 14
assert_slice_dynamic_struct_scalars 1 21 3 99
assert_slice_dynamic_struct_value_slot 0 0 7
assert_slice_dynamic_struct_value_slot 1 0 22
assert_slice_dynamic_struct_value_slot 1 1 23
assert_slice_dynamic_struct_value_slot 1 2 24
assert_slice_dynamic_struct_getters 0 11 14
assert_slice_dynamic_struct_getters 1 21 99
assert_slice_dynamic_struct_value_getter 0 0 7
assert_slice_dynamic_struct_value_getter 1 0 22
assert_slice_dynamic_struct_value_getter 1 1 23
assert_slice_dynamic_struct_value_getter 1 2 24
send_set_slice_dynamic_struct_head 1 88
assert_slice_dynamic_struct_scalars 0 11 1 14
assert_slice_dynamic_struct_scalars 1 88 3 99
assert_slice_dynamic_struct_value_slot 0 0 7
assert_slice_dynamic_struct_value_slot 1 0 22
assert_slice_dynamic_struct_value_slot 1 1 23
assert_slice_dynamic_struct_value_slot 1 2 24
assert_slice_dynamic_struct_getters 0 11 14
assert_slice_dynamic_struct_getters 1 88 99
assert_slice_dynamic_struct_value_getter 0 0 7
assert_slice_dynamic_struct_value_getter 1 0 22
assert_slice_dynamic_struct_value_getter 1 1 23
assert_slice_dynamic_struct_value_getter 1 2 24

SLICE_NESTED_DUAL_DYNAMIC_STRUCT_SOURCE="$WORK_DIR/slice_nested_dual_dynamic_struct_storage_smoke.ora"
SLICE_NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE_FILE="$WORK_DIR/slice_nested_dual_dynamic_struct_storage_smoke.hex"
write_slice_nested_dual_dynamic_struct_fixture "$SLICE_NESTED_DUAL_DYNAMIC_STRUCT_SOURCE"
SLICE_NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE="$(compile_bytecode "$SLICE_NESTED_DUAL_DYNAMIC_STRUCT_SOURCE" "$SLICE_NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE_FILE")"

echo "Deploying slice nested dual dynamic struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$SLICE_NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_slice_nested_dual_dynamic_struct_length 0
assert_slice_nested_dual_dynamic_struct_scalars 0 0 0 0 0 0
assert_slice_nested_dual_dynamic_struct_scalars 1 0 0 0 0 0
assert_slice_nested_dual_dynamic_struct_left_slot 0 0 0
assert_slice_nested_dual_dynamic_struct_right_slot 0 0 0
assert_slice_nested_dual_dynamic_struct_left_slot 1 0 0
assert_slice_nested_dual_dynamic_struct_right_slot 1 0 0
send_set_slice_nested_dual_dynamic_struct "[(11,([21,22],33,[41,42,43]),55),(101,([201],303,[401,402]),505)]"
assert_slice_nested_dual_dynamic_struct_length 2
assert_slice_nested_dual_dynamic_struct_scalars 0 11 2 33 3 55
assert_slice_nested_dual_dynamic_struct_scalars 1 101 1 303 2 505
assert_slice_nested_dual_dynamic_struct_left_slot 0 0 21
assert_slice_nested_dual_dynamic_struct_left_slot 0 1 22
assert_slice_nested_dual_dynamic_struct_right_slot 0 0 41
assert_slice_nested_dual_dynamic_struct_right_slot 0 1 42
assert_slice_nested_dual_dynamic_struct_right_slot 0 2 43
assert_slice_nested_dual_dynamic_struct_left_slot 1 0 201
assert_slice_nested_dual_dynamic_struct_right_slot 1 0 401
assert_slice_nested_dual_dynamic_struct_right_slot 1 1 402
assert_slice_nested_dual_dynamic_struct_getters 0 11 33 55
assert_slice_nested_dual_dynamic_struct_getters 1 101 303 505
assert_slice_nested_dual_dynamic_struct_left_getter 0 0 21
assert_slice_nested_dual_dynamic_struct_left_getter 0 1 22
assert_slice_nested_dual_dynamic_struct_right_getter 0 0 41
assert_slice_nested_dual_dynamic_struct_right_getter 0 1 42
assert_slice_nested_dual_dynamic_struct_right_getter 0 2 43
assert_slice_nested_dual_dynamic_struct_left_getter 1 0 201
assert_slice_nested_dual_dynamic_struct_right_getter 1 0 401
assert_slice_nested_dual_dynamic_struct_right_getter 1 1 402
send_set_slice_nested_dual_dynamic_struct_left 0 "[5]"
assert_slice_nested_dual_dynamic_struct_scalars 0 11 1 33 3 55
assert_slice_nested_dual_dynamic_struct_scalars 1 101 1 303 2 505
assert_slice_nested_dual_dynamic_struct_left_slot 0 0 5
assert_slice_nested_dual_dynamic_struct_right_slot 0 0 41
assert_slice_nested_dual_dynamic_struct_right_slot 0 1 42
assert_slice_nested_dual_dynamic_struct_right_slot 0 2 43
assert_slice_nested_dual_dynamic_struct_left_slot 1 0 201
assert_slice_nested_dual_dynamic_struct_right_slot 1 0 401
assert_slice_nested_dual_dynamic_struct_right_slot 1 1 402
assert_slice_nested_dual_dynamic_struct_left_getter 0 0 5
assert_slice_nested_dual_dynamic_struct_right_getter 0 0 41
assert_slice_nested_dual_dynamic_struct_right_getter 0 1 42
assert_slice_nested_dual_dynamic_struct_right_getter 0 2 43
send_set_slice_nested_dual_dynamic_struct_right 1 "[8,9,10]"
assert_slice_nested_dual_dynamic_struct_scalars 0 11 1 33 3 55
assert_slice_nested_dual_dynamic_struct_scalars 1 101 1 303 3 505
assert_slice_nested_dual_dynamic_struct_left_slot 0 0 5
assert_slice_nested_dual_dynamic_struct_right_slot 0 0 41
assert_slice_nested_dual_dynamic_struct_right_slot 0 1 42
assert_slice_nested_dual_dynamic_struct_right_slot 0 2 43
assert_slice_nested_dual_dynamic_struct_left_slot 1 0 201
assert_slice_nested_dual_dynamic_struct_right_slot 1 0 8
assert_slice_nested_dual_dynamic_struct_right_slot 1 1 9
assert_slice_nested_dual_dynamic_struct_right_slot 1 2 10
assert_slice_nested_dual_dynamic_struct_left_getter 1 0 201
assert_slice_nested_dual_dynamic_struct_right_getter 1 0 8
assert_slice_nested_dual_dynamic_struct_right_getter 1 1 9
assert_slice_nested_dual_dynamic_struct_right_getter 1 2 10
send_set_slice_nested_dual_dynamic_struct_pivot 0 77
assert_slice_nested_dual_dynamic_struct_scalars 0 11 1 77 3 55
assert_slice_nested_dual_dynamic_struct_scalars 1 101 1 303 3 505
assert_slice_nested_dual_dynamic_struct_left_slot 0 0 5
assert_slice_nested_dual_dynamic_struct_right_slot 0 0 41
assert_slice_nested_dual_dynamic_struct_right_slot 0 1 42
assert_slice_nested_dual_dynamic_struct_right_slot 0 2 43
assert_slice_nested_dual_dynamic_struct_getters 0 11 77 55
send_set_slice_nested_dual_dynamic_struct_tail 1 909
assert_slice_nested_dual_dynamic_struct_scalars 0 11 1 77 3 55
assert_slice_nested_dual_dynamic_struct_scalars 1 101 1 303 3 909
assert_slice_nested_dual_dynamic_struct_left_slot 1 0 201
assert_slice_nested_dual_dynamic_struct_right_slot 1 0 8
assert_slice_nested_dual_dynamic_struct_right_slot 1 1 9
assert_slice_nested_dual_dynamic_struct_right_slot 1 2 10
assert_slice_nested_dual_dynamic_struct_getters 1 101 303 909

MAP_SLICE_SOURCE="$WORK_DIR/map_slice_storage_smoke.ora"
MAP_SLICE_BYTECODE_FILE="$WORK_DIR/map_slice_storage_smoke.hex"
write_map_slice_fixture "$MAP_SLICE_SOURCE"
MAP_SLICE_BYTECODE="$(compile_bytecode "$MAP_SLICE_SOURCE" "$MAP_SLICE_BYTECODE_FILE")"

echo "Deploying map slice bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAP_SLICE_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_map_slice_length "$ALICE" 0
assert_map_slice_length "$BOB" 0
assert_map_slice_raw_slot "$ALICE" 0 0
assert_map_slice_raw_slot "$BOB" 0 0
send_set_map_slice "$ALICE" "[31,41,59]"
assert_map_slice_length "$ALICE" 3
assert_map_slice_length "$BOB" 0
assert_map_slice_raw_slot "$ALICE" 0 31
assert_map_slice_raw_slot "$ALICE" 1 41
assert_map_slice_raw_slot "$ALICE" 2 59
assert_map_slice_raw_slot "$BOB" 0 0
assert_map_slice_getter "$ALICE" 0 31
assert_map_slice_getter "$ALICE" 1 41
assert_map_slice_getter "$ALICE" 2 59
send_set_map_slice "$BOB" "[7,8]"
assert_map_slice_length "$ALICE" 3
assert_map_slice_length "$BOB" 2
assert_map_slice_raw_slot "$ALICE" 0 31
assert_map_slice_raw_slot "$ALICE" 1 41
assert_map_slice_raw_slot "$ALICE" 2 59
assert_map_slice_raw_slot "$BOB" 0 7
assert_map_slice_raw_slot "$BOB" 1 8
assert_map_slice_getter "$ALICE" 0 31
assert_map_slice_getter "$ALICE" 1 41
assert_map_slice_getter "$ALICE" 2 59
assert_map_slice_getter "$BOB" 0 7
assert_map_slice_getter "$BOB" 1 8
send_set_map_slice "$ALICE" "[5]"
assert_map_slice_length "$ALICE" 1
assert_map_slice_length "$BOB" 2
assert_map_slice_raw_slot "$ALICE" 0 5
assert_map_slice_raw_slot "$BOB" 0 7
assert_map_slice_raw_slot "$BOB" 1 8
assert_map_slice_getter "$ALICE" 0 5
assert_map_slice_getter "$BOB" 0 7
assert_map_slice_getter "$BOB" 1 8

MAP_DYNAMIC_STRUCT_SOURCE="$WORK_DIR/map_dynamic_struct_storage_smoke.ora"
MAP_DYNAMIC_STRUCT_BYTECODE_FILE="$WORK_DIR/map_dynamic_struct_storage_smoke.hex"
write_map_dynamic_struct_fixture "$MAP_DYNAMIC_STRUCT_SOURCE"
MAP_DYNAMIC_STRUCT_BYTECODE="$(compile_bytecode "$MAP_DYNAMIC_STRUCT_SOURCE" "$MAP_DYNAMIC_STRUCT_BYTECODE_FILE")"

echo "Deploying map dynamic struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAP_DYNAMIC_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_map_dynamic_struct_scalars "$ALICE" 0 0 0
assert_map_dynamic_struct_scalars "$BOB" 0 0 0
assert_map_dynamic_struct_value_slot "$ALICE" 0 0
assert_map_dynamic_struct_value_slot "$BOB" 0 0
send_set_map_dynamic_struct "$ALICE" 11 "[12,13]" 14
assert_map_dynamic_struct_scalars "$ALICE" 11 2 14
assert_map_dynamic_struct_scalars "$BOB" 0 0 0
assert_map_dynamic_struct_value_slot "$ALICE" 0 12
assert_map_dynamic_struct_value_slot "$ALICE" 1 13
assert_map_dynamic_struct_value_slot "$BOB" 0 0
assert_map_dynamic_struct_getters "$ALICE" 11 14
assert_map_dynamic_struct_value_getter "$ALICE" 0 12
assert_map_dynamic_struct_value_getter "$ALICE" 1 13
assert_map_dynamic_struct_getters "$BOB" 0 0
send_set_map_dynamic_struct "$BOB" 21 "[22,23,24]" 25
assert_map_dynamic_struct_scalars "$ALICE" 11 2 14
assert_map_dynamic_struct_scalars "$BOB" 21 3 25
assert_map_dynamic_struct_value_slot "$ALICE" 0 12
assert_map_dynamic_struct_value_slot "$ALICE" 1 13
assert_map_dynamic_struct_value_slot "$BOB" 0 22
assert_map_dynamic_struct_value_slot "$BOB" 1 23
assert_map_dynamic_struct_value_slot "$BOB" 2 24
assert_map_dynamic_struct_getters "$ALICE" 11 14
assert_map_dynamic_struct_getters "$BOB" 21 25
assert_map_dynamic_struct_value_getter "$BOB" 0 22
assert_map_dynamic_struct_value_getter "$BOB" 1 23
assert_map_dynamic_struct_value_getter "$BOB" 2 24
send_set_map_dynamic_struct_values "$ALICE" "[5]"
assert_map_dynamic_struct_scalars "$ALICE" 11 1 14
assert_map_dynamic_struct_scalars "$BOB" 21 3 25
assert_map_dynamic_struct_value_slot "$ALICE" 0 5
assert_map_dynamic_struct_value_slot "$BOB" 0 22
assert_map_dynamic_struct_value_slot "$BOB" 1 23
assert_map_dynamic_struct_value_slot "$BOB" 2 24
assert_map_dynamic_struct_getters "$ALICE" 11 14
assert_map_dynamic_struct_getters "$BOB" 21 25
assert_map_dynamic_struct_value_getter "$ALICE" 0 5
assert_map_dynamic_struct_value_getter "$BOB" 0 22
assert_map_dynamic_struct_value_getter "$BOB" 1 23
assert_map_dynamic_struct_value_getter "$BOB" 2 24
send_set_map_dynamic_struct_tail "$BOB" 99
assert_map_dynamic_struct_scalars "$ALICE" 11 1 14
assert_map_dynamic_struct_scalars "$BOB" 21 3 99
assert_map_dynamic_struct_value_slot "$ALICE" 0 5
assert_map_dynamic_struct_value_slot "$BOB" 0 22
assert_map_dynamic_struct_value_slot "$BOB" 1 23
assert_map_dynamic_struct_value_slot "$BOB" 2 24
assert_map_dynamic_struct_getters "$ALICE" 11 14
assert_map_dynamic_struct_getters "$BOB" 21 99
assert_map_dynamic_struct_value_getter "$ALICE" 0 5
assert_map_dynamic_struct_value_getter "$BOB" 0 22
assert_map_dynamic_struct_value_getter "$BOB" 1 23
assert_map_dynamic_struct_value_getter "$BOB" 2 24
send_set_map_dynamic_struct_head "$ALICE" 77
assert_map_dynamic_struct_scalars "$ALICE" 77 1 14
assert_map_dynamic_struct_scalars "$BOB" 21 3 99
assert_map_dynamic_struct_value_slot "$ALICE" 0 5
assert_map_dynamic_struct_value_slot "$BOB" 0 22
assert_map_dynamic_struct_value_slot "$BOB" 1 23
assert_map_dynamic_struct_value_slot "$BOB" 2 24
assert_map_dynamic_struct_getters "$ALICE" 77 14
assert_map_dynamic_struct_getters "$BOB" 21 99
assert_map_dynamic_struct_value_getter "$ALICE" 0 5
assert_map_dynamic_struct_value_getter "$BOB" 0 22
assert_map_dynamic_struct_value_getter "$BOB" 1 23
assert_map_dynamic_struct_value_getter "$BOB" 2 24

MAP_DUAL_DYNAMIC_STRUCT_SOURCE="$WORK_DIR/map_dual_dynamic_struct_storage_smoke.ora"
MAP_DUAL_DYNAMIC_STRUCT_BYTECODE_FILE="$WORK_DIR/map_dual_dynamic_struct_storage_smoke.hex"
write_map_dual_dynamic_struct_fixture "$MAP_DUAL_DYNAMIC_STRUCT_SOURCE"
MAP_DUAL_DYNAMIC_STRUCT_BYTECODE="$(compile_bytecode "$MAP_DUAL_DYNAMIC_STRUCT_SOURCE" "$MAP_DUAL_DYNAMIC_STRUCT_BYTECODE_FILE")"

echo "Deploying map dual dynamic struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAP_DUAL_DYNAMIC_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_map_dual_dynamic_struct_scalars "$ALICE" 0 0 0
assert_map_dual_dynamic_struct_scalars "$BOB" 0 0 0
assert_map_dual_dynamic_struct_left_slot "$ALICE" 0 0
assert_map_dual_dynamic_struct_right_slot "$ALICE" 0 0
assert_map_dual_dynamic_struct_left_slot "$BOB" 0 0
assert_map_dual_dynamic_struct_right_slot "$BOB" 0 0
send_set_map_dual_dynamic_struct "$ALICE" "[11,12]" 99 "[21,22,23]"
assert_map_dual_dynamic_struct_scalars "$ALICE" 2 99 3
assert_map_dual_dynamic_struct_scalars "$BOB" 0 0 0
assert_map_dual_dynamic_struct_left_slot "$ALICE" 0 11
assert_map_dual_dynamic_struct_left_slot "$ALICE" 1 12
assert_map_dual_dynamic_struct_right_slot "$ALICE" 0 21
assert_map_dual_dynamic_struct_right_slot "$ALICE" 1 22
assert_map_dual_dynamic_struct_right_slot "$ALICE" 2 23
assert_map_dual_dynamic_struct_left_slot "$BOB" 0 0
assert_map_dual_dynamic_struct_right_slot "$BOB" 0 0
assert_map_dual_dynamic_struct_getters "$ALICE" 99
assert_map_dual_dynamic_struct_left_getter "$ALICE" 0 11
assert_map_dual_dynamic_struct_left_getter "$ALICE" 1 12
assert_map_dual_dynamic_struct_right_getter "$ALICE" 0 21
assert_map_dual_dynamic_struct_right_getter "$ALICE" 1 22
assert_map_dual_dynamic_struct_right_getter "$ALICE" 2 23
send_set_map_dual_dynamic_struct "$BOB" "[31]" 44 "[41,42]"
assert_map_dual_dynamic_struct_scalars "$ALICE" 2 99 3
assert_map_dual_dynamic_struct_scalars "$BOB" 1 44 2
assert_map_dual_dynamic_struct_left_slot "$ALICE" 0 11
assert_map_dual_dynamic_struct_left_slot "$ALICE" 1 12
assert_map_dual_dynamic_struct_right_slot "$ALICE" 0 21
assert_map_dual_dynamic_struct_right_slot "$ALICE" 1 22
assert_map_dual_dynamic_struct_right_slot "$ALICE" 2 23
assert_map_dual_dynamic_struct_left_slot "$BOB" 0 31
assert_map_dual_dynamic_struct_right_slot "$BOB" 0 41
assert_map_dual_dynamic_struct_right_slot "$BOB" 1 42
assert_map_dual_dynamic_struct_getters "$ALICE" 99
assert_map_dual_dynamic_struct_getters "$BOB" 44
assert_map_dual_dynamic_struct_left_getter "$BOB" 0 31
assert_map_dual_dynamic_struct_right_getter "$BOB" 0 41
assert_map_dual_dynamic_struct_right_getter "$BOB" 1 42
send_set_map_dual_dynamic_struct_left "$ALICE" "[5]"
assert_map_dual_dynamic_struct_scalars "$ALICE" 1 99 3
assert_map_dual_dynamic_struct_scalars "$BOB" 1 44 2
assert_map_dual_dynamic_struct_left_slot "$ALICE" 0 5
assert_map_dual_dynamic_struct_right_slot "$ALICE" 0 21
assert_map_dual_dynamic_struct_right_slot "$ALICE" 1 22
assert_map_dual_dynamic_struct_right_slot "$ALICE" 2 23
assert_map_dual_dynamic_struct_left_slot "$BOB" 0 31
assert_map_dual_dynamic_struct_right_slot "$BOB" 0 41
assert_map_dual_dynamic_struct_right_slot "$BOB" 1 42
assert_map_dual_dynamic_struct_left_getter "$ALICE" 0 5
assert_map_dual_dynamic_struct_right_getter "$ALICE" 0 21
assert_map_dual_dynamic_struct_right_getter "$ALICE" 1 22
assert_map_dual_dynamic_struct_right_getter "$ALICE" 2 23
send_set_map_dual_dynamic_struct_right "$BOB" "[8,9,10]"
assert_map_dual_dynamic_struct_scalars "$ALICE" 1 99 3
assert_map_dual_dynamic_struct_scalars "$BOB" 1 44 3
assert_map_dual_dynamic_struct_left_slot "$ALICE" 0 5
assert_map_dual_dynamic_struct_right_slot "$ALICE" 0 21
assert_map_dual_dynamic_struct_right_slot "$ALICE" 1 22
assert_map_dual_dynamic_struct_right_slot "$ALICE" 2 23
assert_map_dual_dynamic_struct_left_slot "$BOB" 0 31
assert_map_dual_dynamic_struct_right_slot "$BOB" 0 8
assert_map_dual_dynamic_struct_right_slot "$BOB" 1 9
assert_map_dual_dynamic_struct_right_slot "$BOB" 2 10
assert_map_dual_dynamic_struct_left_getter "$BOB" 0 31
assert_map_dual_dynamic_struct_right_getter "$BOB" 0 8
assert_map_dual_dynamic_struct_right_getter "$BOB" 1 9
assert_map_dual_dynamic_struct_right_getter "$BOB" 2 10
send_set_map_dual_dynamic_struct_pivot "$ALICE" 77
assert_map_dual_dynamic_struct_scalars "$ALICE" 1 77 3
assert_map_dual_dynamic_struct_scalars "$BOB" 1 44 3
assert_map_dual_dynamic_struct_left_slot "$ALICE" 0 5
assert_map_dual_dynamic_struct_right_slot "$ALICE" 0 21
assert_map_dual_dynamic_struct_right_slot "$ALICE" 1 22
assert_map_dual_dynamic_struct_right_slot "$ALICE" 2 23
assert_map_dual_dynamic_struct_getters "$ALICE" 77
assert_map_dual_dynamic_struct_left_getter "$ALICE" 0 5
assert_map_dual_dynamic_struct_right_getter "$ALICE" 0 21
assert_map_dual_dynamic_struct_right_getter "$ALICE" 1 22
assert_map_dual_dynamic_struct_right_getter "$ALICE" 2 23

MAP_NESTED_DUAL_DYNAMIC_STRUCT_SOURCE="$WORK_DIR/map_nested_dual_dynamic_struct_storage_smoke.ora"
MAP_NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE_FILE="$WORK_DIR/map_nested_dual_dynamic_struct_storage_smoke.hex"
write_map_nested_dual_dynamic_struct_fixture "$MAP_NESTED_DUAL_DYNAMIC_STRUCT_SOURCE"
MAP_NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE="$(compile_bytecode "$MAP_NESTED_DUAL_DYNAMIC_STRUCT_SOURCE" "$MAP_NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE_FILE")"

echo "Deploying map nested dual dynamic struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAP_NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_map_nested_dual_dynamic_struct_scalars "$ALICE" 0 0 0 0 0
assert_map_nested_dual_dynamic_struct_scalars "$BOB" 0 0 0 0 0
assert_map_nested_dual_dynamic_struct_left_slot "$ALICE" 0 0
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 0 0
assert_map_nested_dual_dynamic_struct_left_slot "$BOB" 0 0
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 0 0
send_set_map_nested_dual_dynamic_struct "$ALICE" 11 "[21,22]" 33 "[41,42,43]" 55
assert_map_nested_dual_dynamic_struct_scalars "$ALICE" 11 2 33 3 55
assert_map_nested_dual_dynamic_struct_scalars "$BOB" 0 0 0 0 0
assert_map_nested_dual_dynamic_struct_left_slot "$ALICE" 0 21
assert_map_nested_dual_dynamic_struct_left_slot "$ALICE" 1 22
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 0 41
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 1 42
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 2 43
assert_map_nested_dual_dynamic_struct_left_slot "$BOB" 0 0
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 0 0
assert_map_nested_dual_dynamic_struct_getters "$ALICE" 11 33 55
assert_map_nested_dual_dynamic_struct_left_getter "$ALICE" 0 21
assert_map_nested_dual_dynamic_struct_left_getter "$ALICE" 1 22
assert_map_nested_dual_dynamic_struct_right_getter "$ALICE" 0 41
assert_map_nested_dual_dynamic_struct_right_getter "$ALICE" 1 42
assert_map_nested_dual_dynamic_struct_right_getter "$ALICE" 2 43
send_set_map_nested_dual_dynamic_struct "$BOB" 101 "[201]" 303 "[401,402]" 505
assert_map_nested_dual_dynamic_struct_scalars "$ALICE" 11 2 33 3 55
assert_map_nested_dual_dynamic_struct_scalars "$BOB" 101 1 303 2 505
assert_map_nested_dual_dynamic_struct_left_slot "$ALICE" 0 21
assert_map_nested_dual_dynamic_struct_left_slot "$ALICE" 1 22
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 0 41
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 1 42
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 2 43
assert_map_nested_dual_dynamic_struct_left_slot "$BOB" 0 201
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 0 401
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 1 402
assert_map_nested_dual_dynamic_struct_getters "$ALICE" 11 33 55
assert_map_nested_dual_dynamic_struct_getters "$BOB" 101 303 505
assert_map_nested_dual_dynamic_struct_left_getter "$BOB" 0 201
assert_map_nested_dual_dynamic_struct_right_getter "$BOB" 0 401
assert_map_nested_dual_dynamic_struct_right_getter "$BOB" 1 402
send_set_map_nested_dual_dynamic_struct_left "$ALICE" "[5]"
assert_map_nested_dual_dynamic_struct_scalars "$ALICE" 11 1 33 3 55
assert_map_nested_dual_dynamic_struct_scalars "$BOB" 101 1 303 2 505
assert_map_nested_dual_dynamic_struct_left_slot "$ALICE" 0 5
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 0 41
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 1 42
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 2 43
assert_map_nested_dual_dynamic_struct_left_slot "$BOB" 0 201
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 0 401
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 1 402
assert_map_nested_dual_dynamic_struct_left_getter "$ALICE" 0 5
assert_map_nested_dual_dynamic_struct_right_getter "$ALICE" 0 41
assert_map_nested_dual_dynamic_struct_right_getter "$ALICE" 1 42
assert_map_nested_dual_dynamic_struct_right_getter "$ALICE" 2 43
send_set_map_nested_dual_dynamic_struct_right "$BOB" "[8,9,10]"
assert_map_nested_dual_dynamic_struct_scalars "$ALICE" 11 1 33 3 55
assert_map_nested_dual_dynamic_struct_scalars "$BOB" 101 1 303 3 505
assert_map_nested_dual_dynamic_struct_left_slot "$ALICE" 0 5
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 0 41
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 1 42
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 2 43
assert_map_nested_dual_dynamic_struct_left_slot "$BOB" 0 201
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 0 8
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 1 9
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 2 10
assert_map_nested_dual_dynamic_struct_left_getter "$BOB" 0 201
assert_map_nested_dual_dynamic_struct_right_getter "$BOB" 0 8
assert_map_nested_dual_dynamic_struct_right_getter "$BOB" 1 9
assert_map_nested_dual_dynamic_struct_right_getter "$BOB" 2 10
send_set_map_nested_dual_dynamic_struct_pivot "$ALICE" 77
assert_map_nested_dual_dynamic_struct_scalars "$ALICE" 11 1 77 3 55
assert_map_nested_dual_dynamic_struct_scalars "$BOB" 101 1 303 3 505
assert_map_nested_dual_dynamic_struct_left_slot "$ALICE" 0 5
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 0 41
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 1 42
assert_map_nested_dual_dynamic_struct_right_slot "$ALICE" 2 43
assert_map_nested_dual_dynamic_struct_getters "$ALICE" 11 77 55
send_set_map_nested_dual_dynamic_struct_tail "$BOB" 909
assert_map_nested_dual_dynamic_struct_scalars "$ALICE" 11 1 77 3 55
assert_map_nested_dual_dynamic_struct_scalars "$BOB" 101 1 303 3 909
assert_map_nested_dual_dynamic_struct_left_slot "$BOB" 0 201
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 0 8
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 1 9
assert_map_nested_dual_dynamic_struct_right_slot "$BOB" 2 10
assert_map_nested_dual_dynamic_struct_getters "$BOB" 101 303 909

NESTED_MAP_DYNAMIC_STRUCT_SOURCE="$WORK_DIR/nested_map_dynamic_struct_storage_smoke.ora"
NESTED_MAP_DYNAMIC_STRUCT_BYTECODE_FILE="$WORK_DIR/nested_map_dynamic_struct_storage_smoke.hex"
write_nested_map_dynamic_struct_fixture "$NESTED_MAP_DYNAMIC_STRUCT_SOURCE"
NESTED_MAP_DYNAMIC_STRUCT_BYTECODE="$(compile_bytecode "$NESTED_MAP_DYNAMIC_STRUCT_SOURCE" "$NESTED_MAP_DYNAMIC_STRUCT_BYTECODE_FILE")"

echo "Deploying nested map dynamic struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$NESTED_MAP_DYNAMIC_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_nested_map_dynamic_struct_scalars "$ALICE" "$BOB" 0 0 0
assert_nested_map_dynamic_struct_scalars "$ALICE" "$CHARLIE" 0 0 0
assert_nested_map_dynamic_struct_scalars "$BOB" "$ALICE" 0 0 0
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$BOB" 0 0
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 0 0
assert_nested_map_dynamic_struct_value_slot "$BOB" "$ALICE" 0 0
send_set_nested_map_dynamic_struct "$ALICE" "$BOB" 11 "[12,13]" 14
assert_nested_map_dynamic_struct_scalars "$ALICE" "$BOB" 11 2 14
assert_nested_map_dynamic_struct_scalars "$ALICE" "$CHARLIE" 0 0 0
assert_nested_map_dynamic_struct_scalars "$BOB" "$ALICE" 0 0 0
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$BOB" 0 12
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$BOB" 1 13
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 0 0
assert_nested_map_dynamic_struct_getters "$ALICE" "$BOB" 11 14
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$BOB" 0 12
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$BOB" 1 13
send_set_nested_map_dynamic_struct "$ALICE" "$CHARLIE" 21 "[22,23,24]" 25
assert_nested_map_dynamic_struct_scalars "$ALICE" "$BOB" 11 2 14
assert_nested_map_dynamic_struct_scalars "$ALICE" "$CHARLIE" 21 3 25
assert_nested_map_dynamic_struct_scalars "$BOB" "$ALICE" 0 0 0
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$BOB" 0 12
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$BOB" 1 13
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 0 22
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 1 23
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 2 24
assert_nested_map_dynamic_struct_getters "$ALICE" "$BOB" 11 14
assert_nested_map_dynamic_struct_getters "$ALICE" "$CHARLIE" 21 25
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$CHARLIE" 0 22
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$CHARLIE" 1 23
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$CHARLIE" 2 24
send_set_nested_map_dynamic_struct "$BOB" "$ALICE" 31 "[32]" 33
assert_nested_map_dynamic_struct_scalars "$ALICE" "$BOB" 11 2 14
assert_nested_map_dynamic_struct_scalars "$ALICE" "$CHARLIE" 21 3 25
assert_nested_map_dynamic_struct_scalars "$BOB" "$ALICE" 31 1 33
assert_nested_map_dynamic_struct_value_slot "$BOB" "$ALICE" 0 32
assert_nested_map_dynamic_struct_getters "$BOB" "$ALICE" 31 33
assert_nested_map_dynamic_struct_value_getter "$BOB" "$ALICE" 0 32
send_set_nested_map_dynamic_struct_values "$ALICE" "$BOB" "[5]"
assert_nested_map_dynamic_struct_scalars "$ALICE" "$BOB" 11 1 14
assert_nested_map_dynamic_struct_scalars "$ALICE" "$CHARLIE" 21 3 25
assert_nested_map_dynamic_struct_scalars "$BOB" "$ALICE" 31 1 33
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$BOB" 0 5
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 0 22
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 1 23
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 2 24
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$BOB" 0 5
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$CHARLIE" 0 22
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$CHARLIE" 1 23
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$CHARLIE" 2 24
send_set_nested_map_dynamic_struct_tail "$ALICE" "$CHARLIE" 99
assert_nested_map_dynamic_struct_scalars "$ALICE" "$BOB" 11 1 14
assert_nested_map_dynamic_struct_scalars "$ALICE" "$CHARLIE" 21 3 99
assert_nested_map_dynamic_struct_scalars "$BOB" "$ALICE" 31 1 33
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$BOB" 0 5
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 0 22
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 1 23
assert_nested_map_dynamic_struct_value_slot "$ALICE" "$CHARLIE" 2 24
assert_nested_map_dynamic_struct_getters "$ALICE" "$CHARLIE" 21 99
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$CHARLIE" 0 22
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$CHARLIE" 1 23
assert_nested_map_dynamic_struct_value_getter "$ALICE" "$CHARLIE" 2 24
send_set_nested_map_dynamic_struct_head "$BOB" "$ALICE" 77
assert_nested_map_dynamic_struct_scalars "$ALICE" "$BOB" 11 1 14
assert_nested_map_dynamic_struct_scalars "$ALICE" "$CHARLIE" 21 3 99
assert_nested_map_dynamic_struct_scalars "$BOB" "$ALICE" 77 1 33
assert_nested_map_dynamic_struct_value_slot "$BOB" "$ALICE" 0 32
assert_nested_map_dynamic_struct_getters "$BOB" "$ALICE" 77 33
assert_nested_map_dynamic_struct_value_getter "$BOB" "$ALICE" 0 32

BITFIELD_SOURCE="$WORK_DIR/bitfield_storage_smoke.ora"
BITFIELD_BYTECODE_FILE="$WORK_DIR/bitfield_storage_smoke.hex"
write_bitfield_fixture "$BITFIELD_SOURCE"
BITFIELD_BYTECODE="$(compile_bytecode "$BITFIELD_SOURCE" "$BITFIELD_BYTECODE_FILE")"

echo "Deploying bitfield bytecode"
CONTRACT_ADDR="$(deploy_contract "$BITFIELD_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_bitfield_raw_slot 0 0 0 0
assert_bitfield_getters 0 0 0 0
send_configure_bitfield 5 4660
assert_bitfield_raw_slot 1 0 5 4660
assert_bitfield_getters 1 0 5 4660
send_set_bitfield_locked true
assert_bitfield_raw_slot 1 1 5 4660
assert_bitfield_getters 1 1 5 4660
send_set_bitfield_mode 17
assert_bitfield_raw_slot 1 1 17 4660
assert_bitfield_getters 1 1 17 4660
send_set_bitfield_locked false
assert_bitfield_raw_slot 1 0 17 4660
assert_bitfield_getters 1 0 17 4660

CUSTOM_BITFIELD_SOURCE="$WORK_DIR/custom_bitfield_storage_smoke.ora"
CUSTOM_BITFIELD_BYTECODE_FILE="$WORK_DIR/custom_bitfield_storage_smoke.hex"
write_custom_bitfield_fixture "$CUSTOM_BITFIELD_SOURCE"
CUSTOM_BITFIELD_BYTECODE="$(compile_bytecode "$CUSTOM_BITFIELD_SOURCE" "$CUSTOM_BITFIELD_BYTECODE_FILE")"

echo "Deploying custom bitfield bytecode"
CONTRACT_ADDR="$(deploy_contract "$CUSTOM_BITFIELD_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_custom_bitfield_raw_slot 0 0 0 0
assert_custom_bitfield_getters 0 0 0 0
send_configure_custom_bitfield 31 -7 65537
assert_custom_bitfield_raw_slot 1 31 -7 65537
assert_custom_bitfield_getters 1 31 -7 65537
send_set_custom_bitfield_delta 1234
assert_custom_bitfield_raw_slot 1 31 1234 65537
assert_custom_bitfield_getters 1 31 1234 65537
send_set_custom_bitfield_code 18
assert_custom_bitfield_raw_slot 1 18 1234 65537
assert_custom_bitfield_getters 1 18 1234 65537

MAP_BITFIELD_SOURCE="$WORK_DIR/map_bitfield_storage_smoke.ora"
MAP_BITFIELD_BYTECODE_FILE="$WORK_DIR/map_bitfield_storage_smoke.hex"
write_map_bitfield_fixture "$MAP_BITFIELD_SOURCE"
MAP_BITFIELD_BYTECODE="$(compile_bytecode "$MAP_BITFIELD_SOURCE" "$MAP_BITFIELD_BYTECODE_FILE")"

echo "Deploying map bitfield bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAP_BITFIELD_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_map_bitfield_raw_slot "$ALICE" 0 0 0 0
assert_map_bitfield_raw_slot "$BOB" 0 0 0 0
assert_map_bitfield_getters "$ALICE" 0 0 0 0
assert_map_bitfield_getters "$BOB" 0 0 0 0
send_configure_map_bitfield "$ALICE" 5 4660
assert_map_bitfield_raw_slot "$ALICE" 1 0 5 4660
assert_map_bitfield_raw_slot "$BOB" 0 0 0 0
assert_map_bitfield_getters "$ALICE" 1 0 5 4660
assert_map_bitfield_getters "$BOB" 0 0 0 0
send_configure_map_bitfield "$BOB" 9 1234
assert_map_bitfield_raw_slot "$ALICE" 1 0 5 4660
assert_map_bitfield_raw_slot "$BOB" 1 0 9 1234
assert_map_bitfield_getters "$ALICE" 1 0 5 4660
assert_map_bitfield_getters "$BOB" 1 0 9 1234
send_set_map_bitfield_locked "$ALICE" true
assert_map_bitfield_raw_slot "$ALICE" 1 1 5 4660
assert_map_bitfield_raw_slot "$BOB" 1 0 9 1234
assert_map_bitfield_getters "$ALICE" 1 1 5 4660
assert_map_bitfield_getters "$BOB" 1 0 9 1234
send_set_map_bitfield_mode "$ALICE" 17
assert_map_bitfield_raw_slot "$ALICE" 1 1 17 4660
assert_map_bitfield_raw_slot "$BOB" 1 0 9 1234
assert_map_bitfield_getters "$ALICE" 1 1 17 4660
assert_map_bitfield_getters "$BOB" 1 0 9 1234

MAP_CUSTOM_BITFIELD_SOURCE="$WORK_DIR/map_custom_bitfield_storage_smoke.ora"
MAP_CUSTOM_BITFIELD_BYTECODE_FILE="$WORK_DIR/map_custom_bitfield_storage_smoke.hex"
write_map_custom_bitfield_fixture "$MAP_CUSTOM_BITFIELD_SOURCE"
MAP_CUSTOM_BITFIELD_BYTECODE="$(compile_bytecode "$MAP_CUSTOM_BITFIELD_SOURCE" "$MAP_CUSTOM_BITFIELD_BYTECODE_FILE")"

echo "Deploying map custom bitfield bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAP_CUSTOM_BITFIELD_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_map_custom_bitfield_raw_slot "$ALICE" 0 0 0 0
assert_map_custom_bitfield_raw_slot "$BOB" 0 0 0 0
assert_map_custom_bitfield_getters "$ALICE" 0 0 0 0
assert_map_custom_bitfield_getters "$BOB" 0 0 0 0
send_configure_map_custom_bitfield "$ALICE" 31 -7 65537
assert_map_custom_bitfield_raw_slot "$ALICE" 1 31 -7 65537
assert_map_custom_bitfield_raw_slot "$BOB" 0 0 0 0
assert_map_custom_bitfield_getters "$ALICE" 1 31 -7 65537
assert_map_custom_bitfield_getters "$BOB" 0 0 0 0
send_configure_map_custom_bitfield "$BOB" 18 1234 777
assert_map_custom_bitfield_raw_slot "$ALICE" 1 31 -7 65537
assert_map_custom_bitfield_raw_slot "$BOB" 1 18 1234 777
assert_map_custom_bitfield_getters "$ALICE" 1 31 -7 65537
assert_map_custom_bitfield_getters "$BOB" 1 18 1234 777
send_set_map_custom_bitfield_delta "$ALICE" -1234
assert_map_custom_bitfield_raw_slot "$ALICE" 1 31 -1234 65537
assert_map_custom_bitfield_raw_slot "$BOB" 1 18 1234 777
assert_map_custom_bitfield_getters "$ALICE" 1 31 -1234 65537
assert_map_custom_bitfield_getters "$BOB" 1 18 1234 777
send_set_map_custom_bitfield_code "$ALICE" 7
assert_map_custom_bitfield_raw_slot "$ALICE" 1 7 -1234 65537
assert_map_custom_bitfield_raw_slot "$BOB" 1 18 1234 777
assert_map_custom_bitfield_getters "$ALICE" 1 7 -1234 65537
assert_map_custom_bitfield_getters "$BOB" 1 18 1234 777

SLICE_BITFIELD_SOURCE="$WORK_DIR/slice_bitfield_storage_smoke.ora"
SLICE_BITFIELD_BYTECODE_FILE="$WORK_DIR/slice_bitfield_storage_smoke.hex"
write_slice_bitfield_fixture "$SLICE_BITFIELD_SOURCE"
SLICE_BITFIELD_BYTECODE="$(compile_bytecode "$SLICE_BITFIELD_SOURCE" "$SLICE_BITFIELD_BYTECODE_FILE")"

echo "Deploying slice bitfield bytecode"
CONTRACT_ADDR="$(deploy_contract "$SLICE_BITFIELD_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

SLICE_BITFIELD_A="$(bitfield_slot_hex 1 0 5 4660)"
SLICE_BITFIELD_B="$(bitfield_slot_hex 1 1 9 1234)"

assert_slice_bitfield_length 0
assert_slice_bitfield_raw_slot 0 0 0 0 0
assert_slice_bitfield_raw_slot 1 0 0 0 0
send_set_slice_bitfield "[$SLICE_BITFIELD_A,$SLICE_BITFIELD_B]"
assert_slice_bitfield_length 2
assert_slice_bitfield_raw_slot 0 1 0 5 4660
assert_slice_bitfield_raw_slot 1 1 1 9 1234
assert_slice_bitfield_getters 0 1 0 5 4660
assert_slice_bitfield_getters 1 1 1 9 1234
send_set_slice_bitfield_locked 0 true
assert_slice_bitfield_length 2
assert_slice_bitfield_raw_slot 0 1 1 5 4660
assert_slice_bitfield_raw_slot 1 1 1 9 1234
assert_slice_bitfield_getters 0 1 1 5 4660
assert_slice_bitfield_getters 1 1 1 9 1234
send_set_slice_bitfield_mode 1 17
assert_slice_bitfield_length 2
assert_slice_bitfield_raw_slot 0 1 1 5 4660
assert_slice_bitfield_raw_slot 1 1 1 17 1234
assert_slice_bitfield_getters 0 1 1 5 4660
assert_slice_bitfield_getters 1 1 1 17 1234

SLICE_CUSTOM_BITFIELD_SOURCE="$WORK_DIR/slice_custom_bitfield_storage_smoke.ora"
SLICE_CUSTOM_BITFIELD_BYTECODE_FILE="$WORK_DIR/slice_custom_bitfield_storage_smoke.hex"
write_slice_custom_bitfield_fixture "$SLICE_CUSTOM_BITFIELD_SOURCE"
SLICE_CUSTOM_BITFIELD_BYTECODE="$(compile_bytecode "$SLICE_CUSTOM_BITFIELD_SOURCE" "$SLICE_CUSTOM_BITFIELD_BYTECODE_FILE")"

echo "Deploying slice custom bitfield bytecode"
CONTRACT_ADDR="$(deploy_contract "$SLICE_CUSTOM_BITFIELD_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

SLICE_CUSTOM_BITFIELD_A="$(custom_bitfield_slot_hex 1 31 -7 65537)"
SLICE_CUSTOM_BITFIELD_B="$(custom_bitfield_slot_hex 1 18 1234 777)"

assert_slice_bitfield_length 0
assert_slice_custom_bitfield_raw_slot 0 0 0 0 0
assert_slice_custom_bitfield_raw_slot 1 0 0 0 0
send_set_slice_custom_bitfield "[$SLICE_CUSTOM_BITFIELD_A,$SLICE_CUSTOM_BITFIELD_B]"
assert_slice_bitfield_length 2
assert_slice_custom_bitfield_raw_slot 0 1 31 -7 65537
assert_slice_custom_bitfield_raw_slot 1 1 18 1234 777
assert_slice_custom_bitfield_getters 0 1 31 -7 65537
assert_slice_custom_bitfield_getters 1 1 18 1234 777
send_set_slice_custom_bitfield_delta 0 -1234
assert_slice_bitfield_length 2
assert_slice_custom_bitfield_raw_slot 0 1 31 -1234 65537
assert_slice_custom_bitfield_raw_slot 1 1 18 1234 777
assert_slice_custom_bitfield_getters 0 1 31 -1234 65537
assert_slice_custom_bitfield_getters 1 1 18 1234 777
send_set_slice_custom_bitfield_code 1 7
assert_slice_bitfield_length 2
assert_slice_custom_bitfield_raw_slot 0 1 31 -1234 65537
assert_slice_custom_bitfield_raw_slot 1 1 7 1234 777
assert_slice_custom_bitfield_getters 0 1 31 -1234 65537
assert_slice_custom_bitfield_getters 1 1 7 1234 777

STRUCT_BITFIELD_SOURCE="$WORK_DIR/struct_bitfield_storage_smoke.ora"
STRUCT_BITFIELD_BYTECODE_FILE="$WORK_DIR/struct_bitfield_storage_smoke.hex"
write_struct_bitfield_fixture "$STRUCT_BITFIELD_SOURCE"
STRUCT_BITFIELD_BYTECODE="$(compile_bytecode "$STRUCT_BITFIELD_SOURCE" "$STRUCT_BITFIELD_BYTECODE_FILE")"

echo "Deploying struct bitfield bytecode"
CONTRACT_ADDR="$(deploy_contract "$STRUCT_BITFIELD_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_struct_bitfield_raw_slots 0 0 0 0 0 0
assert_struct_bitfield_getters 0 0 0 0 0 0
send_configure_struct_bitfield 101 5 4660 303
assert_struct_bitfield_raw_slots 101 1 0 5 4660 303
assert_struct_bitfield_getters 101 1 0 5 4660 303
send_set_struct_bitfield_locked true
assert_struct_bitfield_raw_slots 101 1 1 5 4660 303
assert_struct_bitfield_getters 101 1 1 5 4660 303
send_set_struct_bitfield_head 707
assert_struct_bitfield_raw_slots 707 1 1 5 4660 303
assert_struct_bitfield_getters 707 1 1 5 4660 303
send_set_struct_bitfield_mode 17
assert_struct_bitfield_raw_slots 707 1 1 17 4660 303
assert_struct_bitfield_getters 707 1 1 17 4660 303
send_set_struct_bitfield_tail 909
assert_struct_bitfield_raw_slots 707 1 1 17 4660 909
assert_struct_bitfield_getters 707 1 1 17 4660 909

STRUCT_CUSTOM_BITFIELD_SOURCE="$WORK_DIR/struct_custom_bitfield_storage_smoke.ora"
STRUCT_CUSTOM_BITFIELD_BYTECODE_FILE="$WORK_DIR/struct_custom_bitfield_storage_smoke.hex"
write_struct_custom_bitfield_fixture "$STRUCT_CUSTOM_BITFIELD_SOURCE"
STRUCT_CUSTOM_BITFIELD_BYTECODE="$(compile_bytecode "$STRUCT_CUSTOM_BITFIELD_SOURCE" "$STRUCT_CUSTOM_BITFIELD_BYTECODE_FILE")"

echo "Deploying struct custom bitfield bytecode"
CONTRACT_ADDR="$(deploy_contract "$STRUCT_CUSTOM_BITFIELD_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_struct_custom_bitfield_raw_slots 0 0 0 0 0 0
assert_struct_custom_bitfield_getters 0 0 0 0 0 0
send_configure_struct_custom_bitfield 101 31 -7 65537 303
assert_struct_custom_bitfield_raw_slots 101 1 31 -7 65537 303
assert_struct_custom_bitfield_getters 101 1 31 -7 65537 303
send_set_struct_custom_bitfield_delta -1234
assert_struct_custom_bitfield_raw_slots 101 1 31 -1234 65537 303
assert_struct_custom_bitfield_getters 101 1 31 -1234 65537 303
send_set_struct_custom_bitfield_head 707
assert_struct_custom_bitfield_raw_slots 707 1 31 -1234 65537 303
assert_struct_custom_bitfield_getters 707 1 31 -1234 65537 303
send_set_struct_custom_bitfield_code 7
assert_struct_custom_bitfield_raw_slots 707 1 7 -1234 65537 303
assert_struct_custom_bitfield_getters 707 1 7 -1234 65537 303
send_set_struct_custom_bitfield_tail 909
assert_struct_custom_bitfield_raw_slots 707 1 7 -1234 65537 909
assert_struct_custom_bitfield_getters 707 1 7 -1234 65537 909

SLICE_STRUCT_CUSTOM_BITFIELD_SOURCE="$WORK_DIR/slice_struct_custom_bitfield_storage_smoke.ora"
SLICE_STRUCT_CUSTOM_BITFIELD_BYTECODE_FILE="$WORK_DIR/slice_struct_custom_bitfield_storage_smoke.hex"
write_slice_struct_custom_bitfield_fixture "$SLICE_STRUCT_CUSTOM_BITFIELD_SOURCE"
SLICE_STRUCT_CUSTOM_BITFIELD_BYTECODE="$(compile_bytecode "$SLICE_STRUCT_CUSTOM_BITFIELD_SOURCE" "$SLICE_STRUCT_CUSTOM_BITFIELD_BYTECODE_FILE")"

echo "Deploying slice struct custom bitfield bytecode"
CONTRACT_ADDR="$(deploy_contract "$SLICE_STRUCT_CUSTOM_BITFIELD_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_slice_struct_length 0
assert_slice_struct_custom_bitfield_raw_slots 0 0 0 0 0 0 0
assert_slice_struct_custom_bitfield_raw_slots 1 0 0 0 0 0 0
SLICE_STRUCT_CUSTOM_BITFIELD_A="$(custom_bitfield_slot_hex 1 31 -7 65537)"
SLICE_STRUCT_CUSTOM_BITFIELD_B="$(custom_bitfield_slot_hex 1 18 1234 777)"
send_set_slice_struct_custom_bitfield "[(11,$SLICE_STRUCT_CUSTOM_BITFIELD_A,33),(101,$SLICE_STRUCT_CUSTOM_BITFIELD_B,303)]"
assert_slice_struct_length 2
assert_slice_struct_custom_bitfield_raw_slots 0 11 1 31 -7 65537 33
assert_slice_struct_custom_bitfield_raw_slots 1 101 1 18 1234 777 303
assert_slice_struct_custom_bitfield_getters 0 11 1 31 -7 65537 33
assert_slice_struct_custom_bitfield_getters 1 101 1 18 1234 777 303
send_set_slice_struct_custom_bitfield_delta 1 -1234
assert_slice_struct_custom_bitfield_raw_slots 0 11 1 31 -7 65537 33
assert_slice_struct_custom_bitfield_raw_slots 1 101 1 18 -1234 777 303
assert_slice_struct_custom_bitfield_getters 0 11 1 31 -7 65537 33
assert_slice_struct_custom_bitfield_getters 1 101 1 18 -1234 777 303
send_set_slice_struct_custom_bitfield_head 0 707
assert_slice_struct_custom_bitfield_raw_slots 0 707 1 31 -7 65537 33
assert_slice_struct_custom_bitfield_raw_slots 1 101 1 18 -1234 777 303
assert_slice_struct_custom_bitfield_getters 0 707 1 31 -7 65537 33
assert_slice_struct_custom_bitfield_getters 1 101 1 18 -1234 777 303
send_set_slice_struct_custom_bitfield_code 0 7
assert_slice_struct_custom_bitfield_raw_slots 0 707 1 7 -7 65537 33
assert_slice_struct_custom_bitfield_raw_slots 1 101 1 18 -1234 777 303
assert_slice_struct_custom_bitfield_getters 0 707 1 7 -7 65537 33
assert_slice_struct_custom_bitfield_getters 1 101 1 18 -1234 777 303
send_set_slice_struct_custom_bitfield_tail 1 909
assert_slice_struct_custom_bitfield_raw_slots 0 707 1 7 -7 65537 33
assert_slice_struct_custom_bitfield_raw_slots 1 101 1 18 -1234 777 909
assert_slice_struct_custom_bitfield_getters 0 707 1 7 -7 65537 33
assert_slice_struct_custom_bitfield_getters 1 101 1 18 -1234 777 909

NESTED_STRUCT_SOURCE="$WORK_DIR/nested_struct_storage_smoke.ora"
NESTED_STRUCT_BYTECODE_FILE="$WORK_DIR/nested_struct_storage_smoke.hex"
write_nested_struct_fixture "$NESTED_STRUCT_SOURCE"
NESTED_STRUCT_BYTECODE="$(compile_bytecode "$NESTED_STRUCT_SOURCE" "$NESTED_STRUCT_BYTECODE_FILE")"

echo "Deploying nested struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$NESTED_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_nested_struct_raw_slots 0 0 0 0 0
assert_nested_struct_getters 0 0 0 0 0
send_set_nested_struct 11 22 33 44 55
assert_nested_struct_raw_slots 11 22 33 44 55
assert_nested_struct_getters 11 22 33 44 55
send_set_nested_struct_inner_middle 99
assert_nested_struct_raw_slots 11 22 99 44 55
assert_nested_struct_getters 11 22 99 44 55

DEEP_NESTED_STRUCT_SOURCE="$WORK_DIR/deep_nested_struct_storage_smoke.ora"
DEEP_NESTED_STRUCT_BYTECODE_FILE="$WORK_DIR/deep_nested_struct_storage_smoke.hex"
write_deep_nested_struct_fixture "$DEEP_NESTED_STRUCT_SOURCE"
DEEP_NESTED_STRUCT_BYTECODE="$(compile_bytecode "$DEEP_NESTED_STRUCT_SOURCE" "$DEEP_NESTED_STRUCT_BYTECODE_FILE")"

echo "Deploying deep nested struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$DEEP_NESTED_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_deep_nested_struct_raw_slots 0 0 0 0 0 0 0
assert_deep_nested_struct_getters 0 0 0 0 0 0 0
send_set_deep_nested_struct 11 22 33 44 55 66 77
assert_deep_nested_struct_raw_slots 11 22 33 44 55 66 77
assert_deep_nested_struct_getters 11 22 33 44 55 66 77
send_set_deep_nested_struct_leaf_middle 444
assert_deep_nested_struct_raw_slots 11 22 33 444 55 66 77
assert_deep_nested_struct_getters 11 22 33 444 55 66 77
send_set_deep_nested_struct_mid_after 666
assert_deep_nested_struct_raw_slots 11 22 33 444 55 666 77
assert_deep_nested_struct_getters 11 22 33 444 55 666 77

SLICE_DEEP_NESTED_STRUCT_SOURCE="$WORK_DIR/slice_deep_nested_struct_storage_smoke.ora"
SLICE_DEEP_NESTED_STRUCT_BYTECODE_FILE="$WORK_DIR/slice_deep_nested_struct_storage_smoke.hex"
write_slice_deep_nested_struct_fixture "$SLICE_DEEP_NESTED_STRUCT_SOURCE"
SLICE_DEEP_NESTED_STRUCT_BYTECODE="$(compile_bytecode "$SLICE_DEEP_NESTED_STRUCT_SOURCE" "$SLICE_DEEP_NESTED_STRUCT_BYTECODE_FILE")"

echo "Deploying slice deep nested struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$SLICE_DEEP_NESTED_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_slice_deep_nested_struct_length 0
assert_slice_deep_nested_struct_raw_slots 0 0 0 0 0 0 0 0
assert_slice_deep_nested_struct_raw_slots 1 0 0 0 0 0 0 0
send_set_slice_deep_nested_struct "[(11,(22,(33,44,55),66),77),(101,(202,(303,404,505),606),707)]"
assert_slice_deep_nested_struct_length 2
assert_slice_deep_nested_struct_raw_slots 0 11 22 33 44 55 66 77
assert_slice_deep_nested_struct_raw_slots 1 101 202 303 404 505 606 707
assert_slice_deep_nested_struct_getters 0 11 22 33 44 55 66 77
assert_slice_deep_nested_struct_getters 1 101 202 303 404 505 606 707
send_set_slice_deep_nested_struct_leaf_middle 1 444
assert_slice_deep_nested_struct_raw_slots 0 11 22 33 44 55 66 77
assert_slice_deep_nested_struct_raw_slots 1 101 202 303 444 505 606 707
assert_slice_deep_nested_struct_getters 0 11 22 33 44 55 66 77
assert_slice_deep_nested_struct_getters 1 101 202 303 444 505 606 707
send_set_slice_deep_nested_struct_mid_after 1 666
assert_slice_deep_nested_struct_raw_slots 0 11 22 33 44 55 66 77
assert_slice_deep_nested_struct_raw_slots 1 101 202 303 444 505 666 707
assert_slice_deep_nested_struct_getters 0 11 22 33 44 55 66 77
assert_slice_deep_nested_struct_getters 1 101 202 303 444 505 666 707

DEEP_DYNAMIC_STRUCT_SOURCE="$WORK_DIR/deep_dynamic_struct_storage_smoke.ora"
DEEP_DYNAMIC_STRUCT_BYTECODE_FILE="$WORK_DIR/deep_dynamic_struct_storage_smoke.hex"
write_deep_dynamic_struct_fixture "$DEEP_DYNAMIC_STRUCT_SOURCE"
DEEP_DYNAMIC_STRUCT_BYTECODE="$(compile_bytecode "$DEEP_DYNAMIC_STRUCT_SOURCE" "$DEEP_DYNAMIC_STRUCT_BYTECODE_FILE")"

echo "Deploying deep dynamic struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$DEEP_DYNAMIC_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_deep_dynamic_struct_scalars 0 0 0 0 0 0 0
assert_deep_dynamic_struct_value_slot 0 0
send_set_deep_dynamic_struct 11 22 33 "[44,55,66]" 77 88 99
assert_deep_dynamic_struct_scalars 11 22 33 3 77 88 99
assert_deep_dynamic_struct_value_slot 0 44
assert_deep_dynamic_struct_value_slot 1 55
assert_deep_dynamic_struct_value_slot 2 66
assert_deep_dynamic_struct_getters 11 22 33 77 88 99
assert_deep_dynamic_struct_value_getter 0 44
assert_deep_dynamic_struct_value_getter 1 55
assert_deep_dynamic_struct_value_getter 2 66
send_set_deep_dynamic_struct_leaf_right 707
assert_deep_dynamic_struct_scalars 11 22 33 3 707 88 99
assert_deep_dynamic_struct_value_slot 0 44
assert_deep_dynamic_struct_value_slot 1 55
assert_deep_dynamic_struct_value_slot 2 66
assert_deep_dynamic_struct_getters 11 22 33 707 88 99
assert_deep_dynamic_struct_value_getter 0 44
assert_deep_dynamic_struct_value_getter 1 55
assert_deep_dynamic_struct_value_getter 2 66
send_set_deep_dynamic_struct_values "[5,8]"
assert_deep_dynamic_struct_scalars 11 22 33 2 707 88 99
assert_deep_dynamic_struct_value_slot 0 5
assert_deep_dynamic_struct_value_slot 1 8
assert_deep_dynamic_struct_getters 11 22 33 707 88 99
assert_deep_dynamic_struct_value_getter 0 5
assert_deep_dynamic_struct_value_getter 1 8
send_set_deep_dynamic_struct_mid_after 909
assert_deep_dynamic_struct_scalars 11 22 33 2 707 909 99
assert_deep_dynamic_struct_value_slot 0 5
assert_deep_dynamic_struct_value_slot 1 8
assert_deep_dynamic_struct_getters 11 22 33 707 909 99
assert_deep_dynamic_struct_value_getter 0 5
assert_deep_dynamic_struct_value_getter 1 8

NESTED_DUAL_DYNAMIC_STRUCT_SOURCE="$WORK_DIR/nested_dual_dynamic_struct_storage_smoke.ora"
NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE_FILE="$WORK_DIR/nested_dual_dynamic_struct_storage_smoke.hex"
write_nested_dual_dynamic_struct_fixture "$NESTED_DUAL_DYNAMIC_STRUCT_SOURCE"
NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE="$(compile_bytecode "$NESTED_DUAL_DYNAMIC_STRUCT_SOURCE" "$NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE_FILE")"

echo "Deploying nested dual dynamic struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$NESTED_DUAL_DYNAMIC_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_nested_dual_dynamic_struct_scalars 0 0 0 0 0
assert_nested_dual_dynamic_struct_left_slot 0 0
assert_nested_dual_dynamic_struct_right_slot 0 0
send_set_nested_dual_dynamic_struct 11 "[21,22]" 33 "[41,42,43]" 55
assert_nested_dual_dynamic_struct_scalars 11 2 33 3 55
assert_nested_dual_dynamic_struct_left_slot 0 21
assert_nested_dual_dynamic_struct_left_slot 1 22
assert_nested_dual_dynamic_struct_right_slot 0 41
assert_nested_dual_dynamic_struct_right_slot 1 42
assert_nested_dual_dynamic_struct_right_slot 2 43
assert_nested_dual_dynamic_struct_getters 11 33 55
assert_nested_dual_dynamic_struct_left_getter 0 21
assert_nested_dual_dynamic_struct_left_getter 1 22
assert_nested_dual_dynamic_struct_right_getter 0 41
assert_nested_dual_dynamic_struct_right_getter 1 42
assert_nested_dual_dynamic_struct_right_getter 2 43
send_set_nested_dual_dynamic_struct_left "[5]"
assert_nested_dual_dynamic_struct_scalars 11 1 33 3 55
assert_nested_dual_dynamic_struct_left_slot 0 5
assert_nested_dual_dynamic_struct_right_slot 0 41
assert_nested_dual_dynamic_struct_right_slot 1 42
assert_nested_dual_dynamic_struct_right_slot 2 43
assert_nested_dual_dynamic_struct_left_getter 0 5
assert_nested_dual_dynamic_struct_right_getter 0 41
assert_nested_dual_dynamic_struct_right_getter 1 42
assert_nested_dual_dynamic_struct_right_getter 2 43
send_set_nested_dual_dynamic_struct_right "[8,9]"
assert_nested_dual_dynamic_struct_scalars 11 1 33 2 55
assert_nested_dual_dynamic_struct_left_slot 0 5
assert_nested_dual_dynamic_struct_right_slot 0 8
assert_nested_dual_dynamic_struct_right_slot 1 9
assert_nested_dual_dynamic_struct_left_getter 0 5
assert_nested_dual_dynamic_struct_right_getter 0 8
assert_nested_dual_dynamic_struct_right_getter 1 9
send_set_nested_dual_dynamic_struct_pivot 77
assert_nested_dual_dynamic_struct_scalars 11 1 77 2 55
assert_nested_dual_dynamic_struct_left_slot 0 5
assert_nested_dual_dynamic_struct_right_slot 0 8
assert_nested_dual_dynamic_struct_right_slot 1 9
assert_nested_dual_dynamic_struct_getters 11 77 55
send_set_nested_dual_dynamic_struct_tail 99
assert_nested_dual_dynamic_struct_scalars 11 1 77 2 99
assert_nested_dual_dynamic_struct_left_slot 0 5
assert_nested_dual_dynamic_struct_right_slot 0 8
assert_nested_dual_dynamic_struct_right_slot 1 9
assert_nested_dual_dynamic_struct_getters 11 77 99

HIGH_ARITY_SOURCE="$WORK_DIR/high_arity_abi_smoke.ora"
HIGH_ARITY_BYTECODE_FILE="$WORK_DIR/high_arity_abi_smoke.hex"
write_high_arity_abi_fixture "$HIGH_ARITY_SOURCE"
HIGH_ARITY_BYTECODE="$(compile_bytecode "$HIGH_ARITY_SOURCE" "$HIGH_ARITY_BYTECODE_FILE")"

echo "Deploying high-arity ABI bytecode"
CONTRACT_ADDR="$(deploy_contract "$HIGH_ARITY_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_high_arity_slots "$ZERO_ADDRESS" 0 0 0 0 0 0 0
send_high_arity_set6 "$ALICE" 11 22 33 44 55
assert_high_arity_slots "$ALICE" 11 22 33 44 55 0 0
send_high_arity_set8 "$BOB" 101 202 303 404 505 606 707
assert_high_arity_slots "$BOB" 101 202 303 404 505 606 707

MULTICONTRACT_CALLEE_SOURCE="$WORK_DIR/multicontract_callee_smoke.ora"
MULTICONTRACT_CALLEE_BYTECODE_FILE="$WORK_DIR/multicontract_callee_smoke.hex"
write_multicontract_callee_fixture "$MULTICONTRACT_CALLEE_SOURCE"
MULTICONTRACT_CALLEE_BYTECODE="$(compile_bytecode "$MULTICONTRACT_CALLEE_SOURCE" "$MULTICONTRACT_CALLEE_BYTECODE_FILE")"

MULTICONTRACT_CALLER_SOURCE="$WORK_DIR/multicontract_caller_smoke.ora"
MULTICONTRACT_CALLER_BYTECODE_FILE="$WORK_DIR/multicontract_caller_smoke.hex"
write_multicontract_caller_fixture "$MULTICONTRACT_CALLER_SOURCE"
# The caller intentionally contains an unresolved state-changing external call.
# Source verification correctly fails closed for that shape, so this runtime
# equivalence fixture uses --no-verify only to test deployed external-call
# bytecode behavior. This is not an SMT verification claim.
MULTICONTRACT_CALLER_BYTECODE="$(compile_bytecode_without_verification "$MULTICONTRACT_CALLER_SOURCE" "$MULTICONTRACT_CALLER_BYTECODE_FILE")"

MULTICONTRACT_SNAPSHOT_CALLER_SOURCE="$WORK_DIR/multicontract_snapshot_caller_smoke.ora"
MULTICONTRACT_SNAPSHOT_CALLER_BYTECODE_FILE="$WORK_DIR/multicontract_snapshot_caller_smoke.hex"
write_multicontract_snapshot_caller_fixture "$MULTICONTRACT_SNAPSHOT_CALLER_SOURCE"
# This caller also contains an unresolved external call, but its purpose is the
# structured return-data path: the callee returns a two-word struct and the
# caller stores both decoded fields locally.
MULTICONTRACT_SNAPSHOT_CALLER_BYTECODE="$(compile_bytecode_without_verification "$MULTICONTRACT_SNAPSHOT_CALLER_SOURCE" "$MULTICONTRACT_SNAPSHOT_CALLER_BYTECODE_FILE")"

echo "Deploying multi-contract callee bytecode"
MULTICONTRACT_CALLEE_ADDR="$(deploy_contract "$MULTICONTRACT_CALLEE_BYTECODE")"
ok "deployed multi-contract callee $MULTICONTRACT_CALLEE_ADDR"

echo "Deploying multi-contract caller bytecode"
MULTICONTRACT_CALLER_ADDR="$(deploy_contract "$MULTICONTRACT_CALLER_BYTECODE")"
ok "deployed multi-contract caller $MULTICONTRACT_CALLER_ADDR"

echo "Deploying multi-contract snapshot caller bytecode"
MULTICONTRACT_SNAPSHOT_CALLER_ADDR="$(deploy_contract "$MULTICONTRACT_SNAPSHOT_CALLER_BYTECODE")"
ok "deployed multi-contract snapshot caller $MULTICONTRACT_SNAPSHOT_CALLER_ADDR"

assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 0
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 0
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 0
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 0 0
send_multicontract_call_target "$MULTICONTRACT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR" 42
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 42
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 0
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 0
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 0 0
send_multicontract_pull_target "$MULTICONTRACT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 42
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 0
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 42
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 0 0
assert_multicontract_snapshot_try_success "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
send_multicontract_pull_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 42
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 0
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 42
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 42 1234
send_multicontract_mark_local "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 42
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 42
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 42 1234
send_multicontract_call_target "$MULTICONTRACT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR" 99
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 99
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 42
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 42 1234
send_multicontract_pull_target "$MULTICONTRACT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 99
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 42 1234
assert_multicontract_snapshot_try_success "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
send_multicontract_pull_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 99
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 99 1234
assert_multicontract_push_snapshot_try_success "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR" 123
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 99
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 99 1234
send_multicontract_push_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR" 123
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 123
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 123 5678
assert_multicontract_failure_try_catch_returns_false "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 123
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 123 5678
send_multicontract_catch_failure "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 123
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 123 4242
send_multicontract_bubble_failure_expect_revert "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 123
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 123 4242
assert_multicontract_oog_try_catch_returns_false "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 123
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 123 4242
send_multicontract_catch_oog "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 123
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 123 5151
send_multicontract_bubble_oog_expect_revert "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" "$MULTICONTRACT_CALLEE_ADDR"
assert_multicontract_callee "$MULTICONTRACT_CALLEE_ADDR" 123
assert_multicontract_caller "$MULTICONTRACT_CALLER_ADDR" 7
assert_multicontract_caller_observed "$MULTICONTRACT_CALLER_ADDR" 99
assert_multicontract_caller_snapshot "$MULTICONTRACT_SNAPSHOT_CALLER_ADDR" 123 5151

MULTICONTRACT_CHAIN_LEAF_SOURCE="$WORK_DIR/multicontract_chain_leaf_smoke.ora"
MULTICONTRACT_CHAIN_LEAF_BYTECODE_FILE="$WORK_DIR/multicontract_chain_leaf_smoke.hex"
write_multicontract_chain_leaf_fixture "$MULTICONTRACT_CHAIN_LEAF_SOURCE"
MULTICONTRACT_CHAIN_LEAF_BYTECODE="$(compile_bytecode "$MULTICONTRACT_CHAIN_LEAF_SOURCE" "$MULTICONTRACT_CHAIN_LEAF_BYTECODE_FILE")"

MULTICONTRACT_CHAIN_MIDDLE_SOURCE="$WORK_DIR/multicontract_chain_middle_smoke.ora"
MULTICONTRACT_CHAIN_MIDDLE_BYTECODE_FILE="$WORK_DIR/multicontract_chain_middle_smoke.hex"
write_multicontract_chain_middle_fixture "$MULTICONTRACT_CHAIN_MIDDLE_SOURCE"
# The middle contract contains an unresolved external call to the leaf; this is
# deployed-runtime bytecode coverage, not source-level SMT verification.
MULTICONTRACT_CHAIN_MIDDLE_BYTECODE="$(compile_bytecode_without_verification "$MULTICONTRACT_CHAIN_MIDDLE_SOURCE" "$MULTICONTRACT_CHAIN_MIDDLE_BYTECODE_FILE")"

MULTICONTRACT_CHAIN_ROOT_SOURCE="$WORK_DIR/multicontract_chain_root_smoke.ora"
MULTICONTRACT_CHAIN_ROOT_BYTECODE_FILE="$WORK_DIR/multicontract_chain_root_smoke.hex"
write_multicontract_chain_root_fixture "$MULTICONTRACT_CHAIN_ROOT_SOURCE"
# The root contract contains an unresolved external call to the middle; this is
# deployed-runtime bytecode coverage, not source-level SMT verification.
MULTICONTRACT_CHAIN_ROOT_BYTECODE="$(compile_bytecode_without_verification "$MULTICONTRACT_CHAIN_ROOT_SOURCE" "$MULTICONTRACT_CHAIN_ROOT_BYTECODE_FILE")"

echo "Deploying multi-contract chain leaf bytecode"
MULTICONTRACT_CHAIN_LEAF_ADDR="$(deploy_contract "$MULTICONTRACT_CHAIN_LEAF_BYTECODE")"
ok "deployed multi-contract chain leaf $MULTICONTRACT_CHAIN_LEAF_ADDR"

echo "Deploying multi-contract chain middle bytecode"
MULTICONTRACT_CHAIN_MIDDLE_ADDR="$(deploy_contract "$MULTICONTRACT_CHAIN_MIDDLE_BYTECODE")"
ok "deployed multi-contract chain middle $MULTICONTRACT_CHAIN_MIDDLE_ADDR"

echo "Deploying multi-contract chain root bytecode"
MULTICONTRACT_CHAIN_ROOT_ADDR="$(deploy_contract "$MULTICONTRACT_CHAIN_ROOT_BYTECODE")"
ok "deployed multi-contract chain root $MULTICONTRACT_CHAIN_ROOT_ADDR"

assert_multicontract_chain_leaf "$MULTICONTRACT_CHAIN_LEAF_ADDR" 0
assert_multicontract_chain_middle "$MULTICONTRACT_CHAIN_MIDDLE_ADDR" 0 0
assert_multicontract_chain_root "$MULTICONTRACT_CHAIN_ROOT_ADDR" 0 0
send_multicontract_chain_call "$MULTICONTRACT_CHAIN_ROOT_ADDR" "$MULTICONTRACT_CHAIN_MIDDLE_ADDR" "$MULTICONTRACT_CHAIN_LEAF_ADDR" 211
assert_multicontract_chain_leaf "$MULTICONTRACT_CHAIN_LEAF_ADDR" 211
assert_multicontract_chain_middle "$MULTICONTRACT_CHAIN_MIDDLE_ADDR" 213 211
assert_multicontract_chain_root "$MULTICONTRACT_CHAIN_ROOT_ADDR" 214 221
send_multicontract_chain_call "$MULTICONTRACT_CHAIN_ROOT_ADDR" "$MULTICONTRACT_CHAIN_MIDDLE_ADDR" "$MULTICONTRACT_CHAIN_LEAF_ADDR" 17
assert_multicontract_chain_leaf "$MULTICONTRACT_CHAIN_LEAF_ADDR" 17
assert_multicontract_chain_middle "$MULTICONTRACT_CHAIN_MIDDLE_ADDR" 19 17
assert_multicontract_chain_root "$MULTICONTRACT_CHAIN_ROOT_ADDR" 20 27
send_multicontract_chain_catch_call "$MULTICONTRACT_CHAIN_ROOT_ADDR" "$MULTICONTRACT_CHAIN_MIDDLE_ADDR" "$MULTICONTRACT_CHAIN_LEAF_ADDR" 31
assert_multicontract_chain_leaf "$MULTICONTRACT_CHAIN_LEAF_ADDR" 17
assert_multicontract_chain_middle "$MULTICONTRACT_CHAIN_MIDDLE_ADDR" 51 7777
assert_multicontract_chain_root "$MULTICONTRACT_CHAIN_ROOT_ADDR" 61 8888
send_multicontract_chain_bubble_failure_expect_revert "$MULTICONTRACT_CHAIN_ROOT_ADDR" "$MULTICONTRACT_CHAIN_MIDDLE_ADDR" "$MULTICONTRACT_CHAIN_LEAF_ADDR" 71
assert_multicontract_chain_leaf "$MULTICONTRACT_CHAIN_LEAF_ADDR" 17
assert_multicontract_chain_middle "$MULTICONTRACT_CHAIN_MIDDLE_ADDR" 51 7777
assert_multicontract_chain_root "$MULTICONTRACT_CHAIN_ROOT_ADDR" 61 8888

LOCK_ROLLBACK_TARGET_SOURCE="$WORK_DIR/lock_rollback_target_smoke.ora"
LOCK_ROLLBACK_TARGET_BYTECODE_FILE="$WORK_DIR/lock_rollback_target_smoke.hex"
write_lock_rollback_target_fixture "$LOCK_ROLLBACK_TARGET_SOURCE"
LOCK_ROLLBACK_TARGET_BYTECODE="$(compile_bytecode "$LOCK_ROLLBACK_TARGET_SOURCE" "$LOCK_ROLLBACK_TARGET_BYTECODE_FILE")"

LOCK_ROLLBACK_OBSERVER_SOURCE="$WORK_DIR/lock_rollback_observer_smoke.ora"
LOCK_ROLLBACK_OBSERVER_BYTECODE_FILE="$WORK_DIR/lock_rollback_observer_smoke.hex"
write_lock_rollback_observer_fixture "$LOCK_ROLLBACK_OBSERVER_SOURCE"
# The observer catches a reverted target call and calls the same target again in
# one transaction to prove the target frame's transient lock rolled back.
LOCK_ROLLBACK_OBSERVER_BYTECODE="$(compile_bytecode_without_verification "$LOCK_ROLLBACK_OBSERVER_SOURCE" "$LOCK_ROLLBACK_OBSERVER_BYTECODE_FILE")"

echo "Deploying lock rollback target bytecode"
LOCK_ROLLBACK_TARGET_ADDR="$(deploy_contract "$LOCK_ROLLBACK_TARGET_BYTECODE")"
ok "deployed lock rollback target $LOCK_ROLLBACK_TARGET_ADDR"

echo "Deploying lock rollback observer bytecode"
LOCK_ROLLBACK_OBSERVER_ADDR="$(deploy_contract "$LOCK_ROLLBACK_OBSERVER_BYTECODE")"
ok "deployed lock rollback observer $LOCK_ROLLBACK_OBSERVER_ADDR"

assert_lock_rollback_target "$LOCK_ROLLBACK_TARGET_ADDR" 0 0
assert_lock_rollback_observer "$LOCK_ROLLBACK_OBSERVER_ADDR" 0
send_lock_rollback_catch_revert_then_set "$LOCK_ROLLBACK_OBSERVER_ADDR" "$LOCK_ROLLBACK_TARGET_ADDR" 40
assert_lock_rollback_target "$LOCK_ROLLBACK_TARGET_ADDR" 41 0
assert_lock_rollback_observer "$LOCK_ROLLBACK_OBSERVER_ADDR" 31
send_lock_rollback_guarded_write_expect_revert "$LOCK_ROLLBACK_TARGET_ADDR" 50
assert_lock_rollback_target "$LOCK_ROLLBACK_TARGET_ADDR" 41 0
assert_lock_rollback_observer "$LOCK_ROLLBACK_OBSERVER_ADDR" 31
send_lock_rollback_unlock_then_write "$LOCK_ROLLBACK_TARGET_ADDR" 70
assert_lock_rollback_target "$LOCK_ROLLBACK_TARGET_ADDR" 70 3001
assert_lock_rollback_observer "$LOCK_ROLLBACK_OBSERVER_ADDR" 31

REENTRANT_LOCK_TARGET_SOURCE="$WORK_DIR/reentrant_lock_target_smoke.ora"
REENTRANT_LOCK_TARGET_BYTECODE_FILE="$WORK_DIR/reentrant_lock_target_smoke.hex"
write_reentrant_lock_target_fixture "$REENTRANT_LOCK_TARGET_SOURCE"
# The target intentionally holds a runtime lock across an unresolved external
# call so the observer can attempt a reentrant write into the locked contract.
REENTRANT_LOCK_TARGET_BYTECODE="$(compile_bytecode_without_verification "$REENTRANT_LOCK_TARGET_SOURCE" "$REENTRANT_LOCK_TARGET_BYTECODE_FILE")"

REENTRANT_LOCK_OBSERVER_SOURCE="$WORK_DIR/reentrant_lock_observer_smoke.ora"
REENTRANT_LOCK_OBSERVER_BYTECODE_FILE="$WORK_DIR/reentrant_lock_observer_smoke.hex"
write_reentrant_lock_observer_fixture "$REENTRANT_LOCK_OBSERVER_SOURCE"
# The observer catches the reentrant revert and records whether the callback was
# blocked by the target's transient lock.
REENTRANT_LOCK_OBSERVER_BYTECODE="$(compile_bytecode_without_verification "$REENTRANT_LOCK_OBSERVER_SOURCE" "$REENTRANT_LOCK_OBSERVER_BYTECODE_FILE")"

echo "Deploying reentrant lock target bytecode"
REENTRANT_LOCK_TARGET_ADDR="$(deploy_contract "$REENTRANT_LOCK_TARGET_BYTECODE")"
ok "deployed reentrant lock target $REENTRANT_LOCK_TARGET_ADDR"

echo "Deploying reentrant lock observer bytecode"
REENTRANT_LOCK_OBSERVER_ADDR="$(deploy_contract "$REENTRANT_LOCK_OBSERVER_BYTECODE")"
ok "deployed reentrant lock observer $REENTRANT_LOCK_OBSERVER_ADDR"

assert_reentrant_lock_target "$REENTRANT_LOCK_TARGET_ADDR" 0 0
assert_reentrant_lock_observer "$REENTRANT_LOCK_OBSERVER_ADDR" 0
send_reentrant_lock_call_observer "$REENTRANT_LOCK_TARGET_ADDR" "$REENTRANT_LOCK_OBSERVER_ADDR" 100
assert_reentrant_lock_target "$REENTRANT_LOCK_TARGET_ADDR" 102 31
assert_reentrant_lock_observer "$REENTRANT_LOCK_OBSERVER_ADDR" 31
send_reentrant_lock_direct_write "$REENTRANT_LOCK_TARGET_ADDR" 200
assert_reentrant_lock_target "$REENTRANT_LOCK_TARGET_ADDR" 200 7000
assert_reentrant_lock_observer "$REENTRANT_LOCK_OBSERVER_ADDR" 31
send_reentrant_lock_call_observer "$REENTRANT_LOCK_TARGET_ADDR" "$REENTRANT_LOCK_OBSERVER_ADDR" 300
assert_reentrant_lock_target "$REENTRANT_LOCK_TARGET_ADDR" 302 31
assert_reentrant_lock_observer "$REENTRANT_LOCK_OBSERVER_ADDR" 31

MAP_NESTED_STRUCT_SOURCE="$WORK_DIR/map_nested_struct_storage_smoke.ora"
MAP_NESTED_STRUCT_BYTECODE_FILE="$WORK_DIR/map_nested_struct_storage_smoke.hex"
write_map_nested_struct_fixture "$MAP_NESTED_STRUCT_SOURCE"
MAP_NESTED_STRUCT_BYTECODE="$(compile_bytecode "$MAP_NESTED_STRUCT_SOURCE" "$MAP_NESTED_STRUCT_BYTECODE_FILE")"

echo "Deploying map nested struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAP_NESTED_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_map_nested_struct_raw_slots "$ALICE" 0 0 0 0 0
assert_map_nested_struct_raw_slots "$BOB" 0 0 0 0 0
assert_map_nested_struct_getters "$ALICE" 0 0 0 0 0
assert_map_nested_struct_getters "$BOB" 0 0 0 0 0
send_set_map_nested_struct "$ALICE" 11 22 33 44 55
assert_map_nested_struct_raw_slots "$ALICE" 11 22 33 44 55
assert_map_nested_struct_raw_slots "$BOB" 0 0 0 0 0
assert_map_nested_struct_getters "$ALICE" 11 22 33 44 55
assert_map_nested_struct_getters "$BOB" 0 0 0 0 0
send_set_map_nested_struct "$BOB" 101 202 303 404 505
assert_map_nested_struct_raw_slots "$ALICE" 11 22 33 44 55
assert_map_nested_struct_raw_slots "$BOB" 101 202 303 404 505
assert_map_nested_struct_getters "$ALICE" 11 22 33 44 55
assert_map_nested_struct_getters "$BOB" 101 202 303 404 505
send_set_map_nested_struct_inner_middle "$ALICE" 99
assert_map_nested_struct_raw_slots "$ALICE" 11 22 99 44 55
assert_map_nested_struct_raw_slots "$BOB" 101 202 303 404 505
assert_map_nested_struct_getters "$ALICE" 11 22 99 44 55
assert_map_nested_struct_getters "$BOB" 101 202 303 404 505

MAP_DEEP_NESTED_STRUCT_SOURCE="$WORK_DIR/map_deep_nested_struct_storage_smoke.ora"
MAP_DEEP_NESTED_STRUCT_BYTECODE_FILE="$WORK_DIR/map_deep_nested_struct_storage_smoke.hex"
write_map_deep_nested_struct_fixture "$MAP_DEEP_NESTED_STRUCT_SOURCE"
MAP_DEEP_NESTED_STRUCT_BYTECODE="$(compile_bytecode "$MAP_DEEP_NESTED_STRUCT_SOURCE" "$MAP_DEEP_NESTED_STRUCT_BYTECODE_FILE")"

echo "Deploying map deep nested struct bytecode"
CONTRACT_ADDR="$(deploy_contract "$MAP_DEEP_NESTED_STRUCT_BYTECODE")"
ok "deployed $CONTRACT_ADDR"

assert_map_deep_nested_struct_raw_slots "$ALICE" 0 0 0 0 0 0 0
assert_map_deep_nested_struct_raw_slots "$BOB" 0 0 0 0 0 0 0
assert_map_deep_nested_struct_getters "$ALICE" 0 0 0 0 0 0 0
assert_map_deep_nested_struct_getters "$BOB" 0 0 0 0 0 0 0
send_set_map_deep_nested_struct "$ALICE" 11 22 33 44 55 66 77
assert_map_deep_nested_struct_raw_slots "$ALICE" 11 22 33 44 55 66 77
assert_map_deep_nested_struct_raw_slots "$BOB" 0 0 0 0 0 0 0
assert_map_deep_nested_struct_getters "$ALICE" 11 22 33 44 55 66 77
assert_map_deep_nested_struct_getters "$BOB" 0 0 0 0 0 0 0
send_set_map_deep_nested_struct "$BOB" 101 202 303 404 505 606 707
assert_map_deep_nested_struct_raw_slots "$ALICE" 11 22 33 44 55 66 77
assert_map_deep_nested_struct_raw_slots "$BOB" 101 202 303 404 505 606 707
assert_map_deep_nested_struct_getters "$ALICE" 11 22 33 44 55 66 77
assert_map_deep_nested_struct_getters "$BOB" 101 202 303 404 505 606 707
send_set_map_deep_nested_struct_leaf_middle "$ALICE" 444
assert_map_deep_nested_struct_raw_slots "$ALICE" 11 22 33 444 55 66 77
assert_map_deep_nested_struct_raw_slots "$BOB" 101 202 303 404 505 606 707
assert_map_deep_nested_struct_getters "$ALICE" 11 22 33 444 55 66 77
assert_map_deep_nested_struct_getters "$BOB" 101 202 303 404 505 606 707
send_set_map_deep_nested_struct_mid_after "$ALICE" 666
assert_map_deep_nested_struct_raw_slots "$ALICE" 11 22 33 444 55 666 77
assert_map_deep_nested_struct_raw_slots "$BOB" 101 202 303 404 505 606 707
assert_map_deep_nested_struct_getters "$ALICE" 11 22 33 444 55 666 77
assert_map_deep_nested_struct_getters "$BOB" 101 202 303 404 505 606 707

ok "bytecode storage equivalence smoke passed"
