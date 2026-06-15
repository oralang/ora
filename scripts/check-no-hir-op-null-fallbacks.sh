#!/usr/bin/env sh
set -eu

fail() {
  echo "check-no-hir-op-null-fallbacks: $*" >&2
  exit 1
}

[ -d src/hir ] || fail "missing src/hir directory"

matches="$(
  grep -R -n -E 'oraOperationIsNull\([^)]*\)\)[[:space:]]*(return|break[[:space:]]+:[[:alnum:]_]+)[[:space:]]+([[:alnum:]_]+|null);' src/hir --include="*.zig" \
    | grep -v -E 'src/hir/executable_fallbacks\.zig:' \
    | grep -v -E 'if[[:space:]]*\(!mlir\.oraOperationIsNull' \
    | grep -v -E 'return[[:space:]]+(true|false);' \
    || true
)"

if [ -n "$matches" ]; then
  echo "$matches" >&2
  fail "MLIR op-creation failure must fail closed, not return an old operand/value/null"
fi

echo "check-no-hir-op-null-fallbacks: ok"
