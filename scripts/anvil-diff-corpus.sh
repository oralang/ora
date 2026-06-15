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
export ANVIL_RPC_TIMEOUT="${ANVIL_RPC_TIMEOUT:-45}"
ANVIL_STARTUP_RPC_TIMEOUT="${ANVIL_STARTUP_RPC_TIMEOUT:-2}"
cd "$(git rev-parse --show-toplevel)"

STARTED=0
ANVIL_PID=""

stop_anvil() {
  if [ -n "$ANVIL_PID" ]; then
    kill "$ANVIL_PID" 2>/dev/null || true
    wait "$ANVIL_PID" 2>/dev/null || true
    ANVIL_PID=""
  fi
}

start_anvil() {
  anvil --hardfork osaka --auto-impersonate --silent >/tmp/anvil-corpus.log 2>&1 &
  ANVIL_PID=$!
  STARTED=1
  for _ in $(seq 1 30); do
    if cast block-number --rpc-url http://127.0.0.1:8545 --rpc-timeout "$ANVIL_STARTUP_RPC_TIMEOUT" --no-proxy >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "error: timed out waiting for Anvil startup" >&2
  return 1
}

if ! cast block-number --rpc-url http://127.0.0.1:8545 --rpc-timeout "$ANVIL_STARTUP_RPC_TIMEOUT" --no-proxy >/dev/null 2>&1; then
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
  specs=$((specs+1))
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
      stop_anvil
      start_anvil || exit 1
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
