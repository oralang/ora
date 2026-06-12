#!/usr/bin/env bash
# Run the Anvil differential over the whole conformance corpus, tallying
# agreement / divergence / skipped. Broad corroboration of lib/evm (our oracle)
# against an independent EVM (Foundry/revm). LOCAL-FIRST, not in the gate.
#
# A DIVERGENCE here is high-value: lib/evm and revm disagree on real Ora
# bytecode -> a compiler bug, a lib/evm bug, or a hardfork mismatch.
set -uo pipefail
export FOUNDRY_DISABLE_NIGHTLY_WARNING=1
cd "$(git rev-parse --show-toplevel)"

STARTED=0
if ! cast rpc eth_blockNumber --rpc-url http://127.0.0.1:8545 >/dev/null 2>&1; then
  anvil --hardfork cancun --auto-impersonate --silent >/tmp/anvil-corpus.log 2>&1 &
  STARTED=1
  sleep 3
fi
trap '[ "$STARTED" -eq 1 ] && pkill -f "anvil --hardfork" 2>/dev/null' EXIT

agree=0 diverge=0 errored=0 specs=0
diverged_specs=""
for spec in tests/conformance/*.spec.toml; do
  specs=$((specs+1))
  out=$(python3 scripts/conformance-anvil-diff.py "$spec" 2>&1)
  rc=$?
  d=$(echo "$out" | sed -n 's/.*\([0-9][0-9]*\) divergences.*/\1/p' | tail -1)
  if [ "$rc" -ne 0 ] && [ -z "$d" ]; then
    errored=$((errored+1))
    echo "  [error] $(basename "$spec"): $(echo "$out" | tail -1 | cut -c1-80)"
  elif [ "${d:-0}" -gt 0 ]; then
    diverge=$((diverge+d))
    diverged_specs="$diverged_specs $(basename "$spec")"
    echo "  [DIVERGE] $(basename "$spec"): $d"
    echo "$out" | grep DIVERGE | sed 's/^/      /'
  else
    agree=$((agree+1))
  fi
done

echo
echo "anvil-diff-corpus: $specs specs | $agree agree | $diverge divergences | $errored errored (unsupported/compile)"
if [ -n "$diverged_specs" ]; then
  echo "  DIVERGED:$diverged_specs"
  exit 1
fi
exit 0
