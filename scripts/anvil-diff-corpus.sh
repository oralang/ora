#!/usr/bin/env bash
# Run the Anvil differential over the whole conformance corpus, tallying
# agreement / divergence / skipped. Broad corroboration of lib/evm (our oracle)
# against an independent EVM (Foundry/revm). LOCAL-FIRST, not in the gate.
#
# A DIVERGENCE here is high-value: lib/evm and revm disagree on real Ora
# bytecode -> a compiler bug, a lib/evm bug, or a hardfork mismatch.
#
# Usage:
#   scripts/anvil-diff-corpus.sh
#   scripts/anvil-diff-corpus.sh tests/conformance/counter.spec.toml ...
set -uo pipefail
export FOUNDRY_DISABLE_NIGHTLY_WARNING=1
export ANVIL_RPC_TIMEOUT="${ANVIL_RPC_TIMEOUT:-300}"
ANVIL_STARTUP_RPC_TIMEOUT="${ANVIL_STARTUP_RPC_TIMEOUT:-2}"
ANVIL_STARTUP_TIMEOUT_S="${ANVIL_STARTUP_TIMEOUT_S:-30}"
ANVIL_RESTART_EACH_SPEC="${ANVIL_RESTART_EACH_SPEC:-1}"
ANVIL_RESTART_GRACE_S="${ANVIL_RESTART_GRACE_S:-0.25}"
if [ -n "${ANVIL_RPC_URL:-}" ] && [ -z "${ANVIL_PORT:-}" ]; then
  ANVIL_PORT="$(python3 - "$ANVIL_RPC_URL" <<'PY'
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
if parsed.port is None:
    raise SystemExit("ANVIL_RPC_URL must include a port when ANVIL_PORT is unset")
print(parsed.port)
PY
)" || exit 2
fi
ANVIL_PORT="${ANVIL_PORT:-$(python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)}"
export ANVIL_RPC_URL="${ANVIL_RPC_URL:-http://127.0.0.1:${ANVIL_PORT}}"
ANVIL_LOG="${ANVIL_LOG:-/tmp/anvil-corpus-${ANVIL_PORT}.log}"
ANVIL_READY_LOG="${ANVIL_READY_LOG:-/tmp/anvil-ready-${ANVIL_PORT}.log}"
cd "$(git rev-parse --show-toplevel)"

STARTED=0
ANVIL_PID=""


stop_anvil() {
  if [ -n "$ANVIL_PID" ]; then
    kill "$ANVIL_PID" 2>/dev/null || true
    wait "$ANVIL_PID" 2>/dev/null || true
    ANVIL_PID=""
    sleep "$ANVIL_RESTART_GRACE_S"
  fi
}

rpc_ready() {
  python3 - "$ANVIL_STARTUP_RPC_TIMEOUT" "$ANVIL_RPC_URL" <<'PY'
import json
import sys
import urllib.request

timeout = float(sys.argv[1])
rpc_url = sys.argv[2]
body = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "eth_blockNumber",
    "params": [],
}).encode()
req = urllib.request.Request(
    rpc_url,
    data=body,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    if "error" in payload:
        print(f"rpc_ready: rpc error {payload['error']}", file=sys.stderr)
        raise SystemExit(1)
except Exception as err:
    print(f"rpc_ready: {type(err).__name__}: {err}", file=sys.stderr)
    raise SystemExit(1)
PY
}

start_anvil() {
  echo "  [anvil] starting on ${ANVIL_RPC_URL}" >&2
  anvil --hardfork osaka --auto-impersonate --port "$ANVIL_PORT" >"$ANVIL_LOG" 2>&1 &
  ANVIL_PID=$!
  STARTED=1
  started_at=$(date +%s)
  while [ $(( $(date +%s) - started_at )) -lt "$ANVIL_STARTUP_TIMEOUT_S" ]; do
    if ! kill -0 "$ANVIL_PID" 2>/dev/null; then
      echo "error: Anvil exited during startup" >&2
      if [ -s "$ANVIL_LOG" ]; then
        tail -40 "$ANVIL_LOG" >&2 || true
      fi
      return 1
    fi
    if rpc_ready >/dev/null 2>"$ANVIL_READY_LOG"; then
      echo "  [anvil] ready" >&2
      return 0
    fi
    sleep 1
  done
  echo "error: timed out waiting ${ANVIL_STARTUP_TIMEOUT_S}s for Anvil startup" >&2
  if [ -s "$ANVIL_READY_LOG" ]; then
    echo "last readiness failure:" >&2
    tail -5 "$ANVIL_READY_LOG" >&2 || true
  fi
  if [ -s "$ANVIL_LOG" ]; then
    tail -40 "$ANVIL_LOG" >&2 || true
  fi
  return 1
}

restart_anvil() {
  stop_anvil
  start_anvil
}

if ! rpc_ready >/dev/null 2>"$ANVIL_READY_LOG"; then
  start_anvil || exit 1
fi
trap '[ "$STARTED" -eq 1 ] && stop_anvil' EXIT

run_spec() {
  python3 scripts/conformance-anvil-diff.py "$1" 2>&1
}

if [ "$#" -gt 0 ]; then
  spec_paths=("$@")
else
  mapfile -t spec_paths < <(find tests/conformance -maxdepth 1 -name '*.spec.toml' | sort)
fi

agree=0 diverge=0 errored=0 specs=0
diverged_specs=""
errored_specs=""
for spec in "${spec_paths[@]}"; do
  if [ "$STARTED" -eq 1 ] && [ "$ANVIL_RESTART_EACH_SPEC" = "1" ] && [ "$specs" -gt 0 ]; then
    restart_anvil || exit 1
  fi
  specs=$((specs+1))
  echo "  [run] $(basename "$spec")" >&2
  if [ ! -f "$spec" ]; then
    errored=$((errored+1))
    errored_specs="$errored_specs $(basename "$spec")"
    echo "  [error] $(basename "$spec"): spec file not found"
    continue
  fi
  out=$(run_spec "$spec")
  rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -qi "urlopen error .*timed out"; then
    echo "  [retry] $(basename "$spec"): RPC timeout"
    if [ "$STARTED" -eq 1 ]; then
      restart_anvil || exit 1
    fi
    out=$(run_spec "$spec")
    rc=$?
  fi
  d=$(echo "$out" | sed -n 's/.*\([0-9][0-9]*\) divergences.*/\1/p' | tail -1)
  if [ "${d:-0}" -gt 0 ]; then
    diverge=$((diverge+d))
    diverged_specs="$diverged_specs $(basename "$spec")"
    echo "  [DIVERGE] $(basename "$spec"): $d"
    echo "$out" | grep DIVERGE | sed 's/^/      /'
  elif [ "$rc" -ne 0 ]; then
    errored=$((errored+1))
    errored_specs="$errored_specs $(basename "$spec")"
    echo "  [error] $(basename "$spec"): $(echo "$out" | tail -1 | cut -c1-80)"
  else
    agree=$((agree+1))
  fi
done

echo
echo "anvil-diff-corpus: $specs specs | $agree agree | $diverge divergences | $errored errored"
if [ -n "$diverged_specs" ]; then
  echo "  DIVERGED:$diverged_specs"
fi
if [ -n "$errored_specs" ]; then
  echo "  ERRORED:$errored_specs"
fi
if [ -n "$diverged_specs" ] || [ -n "$errored_specs" ]; then
  exit 1
fi
exit 0
