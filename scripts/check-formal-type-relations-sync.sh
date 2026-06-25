#!/usr/bin/env bash
#
# check-formal-type-relations-sync — proves selected compiler type-relation rows
# agree with the Lean type relation model.
#
#   1. Regenerate data-only relation rows from the compiler relation functions.
#   2. Lint the generated file: data only, no proof terms/imports.
#   3. Drift-check the generated snapshot, then build the Lean sync module.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.elan/bin:$PATH"

EMITTER="src/formal/emit_type_relations_snapshot.zig"
SNAPSHOT="formal/Ora/Generated/CompilerTypeRelations.lean"

echo "==> [1/3] regenerating $SNAPSHOT from compiler type relations"
zig run \
  --cache-dir "$ROOT/.zig-cache/formal-type-relations-sync" \
  --global-cache-dir "$ROOT/.zig-cache/formal-type-relations-sync-global" \
  --dep ora_formal \
  -Mroot="$EMITTER" \
  --dep ora_types \
  --dep ora_refinements \
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

echo "==> [3/3] drift check + lake build Ora.TypeRelationsSync"
if ! git diff --quiet -- "$SNAPSHOT"; then
  echo "  SNAPSHOT DRIFT — the committed type-relation snapshot is stale:" >&2
  git --no-pager diff -- "$SNAPSHOT" >&2
  echo "  (review the diff; if intended, commit the regenerated snapshot)" >&2
  exit 1
fi
( cd formal && lake build Ora.TypeRelationsSync )

echo "==> check-formal-type-relations-sync OK"
