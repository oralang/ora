#!/usr/bin/env sh
set -eu

fail() {
  echo "check-lsp-boundary: $*" >&2
  exit 1
}

[ -d src/lsp ] || fail "missing src/lsp directory"

direct_backend_imports="$(
  grep -R -n -E '@import\("[^"]*(mlir|sir|z3|evm|backend)[^"]*"\)' src/lsp --include="*.zig" || true
)"

if [ -n "$direct_backend_imports" ]; then
  echo "$direct_backend_imports" >&2
  fail "LSP code must not directly import backend-only MLIR/SIR/Z3/EVM modules"
fi

root_backend_access="$(
  grep -R -n -E 'ora_root\.(mlir|sir|z3|evm|backend|codegen)' src/lsp --include="*.zig" || true
)"

if [ -n "$root_backend_access" ]; then
  echo "$root_backend_access" >&2
  fail "LSP code must not access backend-only modules through ora_root"
fi

echo "check-lsp-boundary: ok"
