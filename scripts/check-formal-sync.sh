#!/usr/bin/env bash
#
# check-formal-sync — proves the compiler's type-universe facts conform to the
# Lean spec (formal/). Three gates:
#
#   1. REGENERATE  the data-only snapshot from the compiler (src/types/*).
#   2. DATA-ONLY LINT  the snapshot — it must contain ONLY `def … := <literal>`
#      facts (no theorem/axiom/sorry/instance/import/…). This is the trust
#      boundary: the untrusted compiler emits DATA; the trusted checks live in
#      formal/Ora/Sync.lean and do all the proving.
#   3. DRIFT + PROOF: fail if the committed snapshot is stale (git diff), then
#      `lake build` — the kernel `decide` checks that compiler facts == spec.
#
# Local: run it, and if step 3 reports drift, review + commit the new snapshot.
# CI: a non-empty git diff means "compiler changed but snapshot not regenerated".
#
# Usage:  scripts/check-formal-sync.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.elan/bin:$PATH"

EMITTER="src/types/emit_formal_snapshot.zig"
SNAPSHOT="formal/Ora/Generated/CompilerSnapshot.lean"

echo "==> [1/3] regenerating $SNAPSHOT from the compiler"
zig run "$EMITTER" > "$SNAPSHOT"

echo "==> [2/3] data-only lint"
# Strip the /- … -/ header comment, then reject any non-data construct.
violations="$(sed '\#^/-#,\#^-/#d' "$SNAPSHOT" \
  | grep -nE '\b(theorem|lemma|example|axiom|sorry|instance|macro|syntax|deriving|native_decide)\b|^[[:space:]]*import|@\[' \
  || true)"
if [ -n "$violations" ]; then
  echo "  DATA-ONLY LINT FAILED — the snapshot must contain only data:" >&2
  echo "$violations" >&2
  exit 1
fi
echo "  ok (data only)"

echo "==> [3/3] drift check + lake build"
if ! git diff --quiet -- "$SNAPSHOT"; then
  echo "  SNAPSHOT DRIFT — the committed snapshot is stale vs the compiler:" >&2
  git --no-pager diff -- "$SNAPSHOT" >&2
  echo "  (review the diff; if intended, commit the regenerated snapshot)" >&2
  exit 1
fi
( cd formal && lake build )

echo "==> check-formal-sync OK"
