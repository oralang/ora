#!/usr/bin/env bash
#
# check-formal-declenv-sync — proves the compiler's nominal DECLARATION facts
# conform to the Lean declaration model (`formal/Ora/Types/Decl.lean`). Mirrors
# `check-formal-sync.sh`, but for a CURATED matrix of declarations:
#
#   1. REGENERATE  the data-only declaration snapshot from the compiler.
#   2. DATA-ONLY LINT  the snapshot — only `def … := <literal>` rows.
#   3. DRIFT + PROOF: fail if the committed snapshot is stale (git diff), then
#      `lake build` — the kernel `decide` (Ora/SyncDeclEnv.lean) checks that the
#      Lean `Decl` kinds map onto the compiler's nominal `TypeKind`s.
#
# Usage:  scripts/check-formal-declenv-sync.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.elan/bin:$PATH"

EMITTER="src/formal/emit_declenv_snapshot.zig"
SNAPSHOT="formal/Ora/Generated/DeclEnvSnapshot.lean"

echo "==> [1/3] regenerating $SNAPSHOT from the compiler"
zig run \
  --cache-dir "$ROOT/.zig-cache/formal-sync" \
  --global-cache-dir "$ROOT/.zig-cache/formal-sync-global" \
  --dep ora_formal \
  -Mroot="$EMITTER" \
  --dep ora_types --dep ora_refinements \
  -Mora_formal=src/formal.zig \
  --dep ora_refinements \
  -Mora_types=src/types/root.zig \
  -Mora_refinements=src/refinements/root.zig \
  > "$SNAPSHOT"

echo "==> [2/3] data-only lint"
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
  echo "  SNAPSHOT DRIFT — the committed declaration snapshot is stale vs the compiler:" >&2
  git --no-pager diff -- "$SNAPSHOT" >&2
  echo "  (review the diff; if intended, commit the regenerated snapshot)" >&2
  exit 1
fi
( cd formal && lake build )

echo "==> check-formal-declenv-sync OK"
