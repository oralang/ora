#!/usr/bin/env bash
# Proof that the Anvil differential catches F-003 (lib/evm panics on the
# huge-offset mload that the F-002 catch-path triggers).
#
# Runs the IDENTICAL hostile call through both EVMs:
#   - lib/evm (our oracle) via the single-spec runner  -> PANICS (crash)
#   - Anvil/revm via the differential                   -> clean OOG revert
# The divergence (one crashes, one reverts gracefully) IS F-003. This is the
# value of the differential: it surfaces a defect the in-process oracle cannot
# even evaluate without crashing.
#
# Exit 0 = divergence reproduced (F-003 still open / caught).
# Exit 1 = lib/evm did NOT crash -> F-003 may be fixed; promote this to a real
#          conformance spec and update FINDINGS.md.
set -uo pipefail
export FOUNDRY_DISABLE_NIGHTLY_WARNING=1

cd "$(git rev-parse --show-toplevel)"
REPRO_ORA=tests/differential/f003_repro.ora
REPRO_SPEC=tests/differential/f003_repro.spec.toml
RUNNER=zig-out/bin/conformance-one

[ -x "$RUNNER" ] || { echo "build the runner first: zig build conformance-one"; exit 2; }

echo "== lib/evm side (single-spec runner) =="
"$RUNNER" "$REPRO_ORA" "$REPRO_SPEC" >/tmp/f003_libevm.out 2>&1
LIBEVM_RC=$?
LIBEVM_PANIC=$(grep -c "panic: integer overflow" /tmp/f003_libevm.out)
echo "  exit=$LIBEVM_RC  panic_lines=$LIBEVM_PANIC"
grep -m1 "handlers_memory.zig" /tmp/f003_libevm.out | sed 's/^/  /' || true

echo "== Anvil side (differential) =="
STARTED_ANVIL=0
if ! cast rpc eth_blockNumber --rpc-url http://127.0.0.1:8545 >/dev/null 2>&1; then
  anvil --hardfork cancun --silent >/tmp/anvil.log 2>&1 &
  STARTED_ANVIL=1
  sleep 3
fi
python3 scripts/conformance-anvil-diff.py "$REPRO_SPEC" 2>&1 | grep -E "err_default|diff:" | sed 's/^/  /'
ANVIL_RC=${PIPESTATUS[0]}
[ "$STARTED_ANVIL" -eq 1 ] && pkill -f "anvil --hardfork" 2>/dev/null

echo
# lib/evm must crash (rc 134 = SIGABRT) with the F-003 panic; Anvil must not diverge.
if [ "$LIBEVM_RC" -ge 128 ] && [ "$LIBEVM_PANIC" -ge 1 ] && [ "$ANVIL_RC" -eq 0 ]; then
  echo "F-003 CAUGHT: identical call -> lib/evm PANICS (exit $LIBEVM_RC), Anvil cleanly OOG-reverts."
  echo "The differential surfaces a defect the in-process oracle cannot evaluate."
  exit 0
fi
if [ "$LIBEVM_PANIC" -eq 0 ]; then
  echo "lib/evm did NOT panic — F-003 may be fixed. Promote the repro to a conformance spec and update FINDINGS.md#f-003."
  exit 1
fi
echo "unexpected state: libevm_rc=$LIBEVM_RC anvil_rc=$ANVIL_RC"
exit 1
