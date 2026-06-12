#!/usr/bin/env sh
set -eu

fail() {
  echo "check-abi-layout-ownership: $*" >&2
  exit 1
}

[ -d src ] || fail "missing src directory"

matches="$(
  grep -R -n -E "StaticEncoding|LayoutNode \{|abi/layout\.zig|abi_layout\." src --include="*.zig" \
    | grep -v -E "src/abi/layout\.zig:|src/abi/layout_context\.zig:|src/abi/comptime_encoder\.zig:|src/abi/comptime_decoder\.zig:|src/abi/comptime_decoder_test_support\.zig:|src/abi/runtime_encoder\.zig:|src/abi/runtime_decoder\.zig:|src/abi\.test\.zig:" \
    | grep -v -E "src/hir/abi\.zig:|src/root\.zig:" \
    || true
)"

if [ -n "$matches" ]; then
  echo "$matches" >&2
  fail "layout-bearing code must live in src/abi/layout*.zig or sanctioned ABI materializers; tests may reference it from src/abi.test.zig"
fi

abi_decode="src/mlir/ora/lowering/OraToSIR/patterns/AbiDecode.cpp"

if ! grep -q 'failed to convert ABI return ptr-view type' "$abi_decode"; then
  fail "ABI returndata ptr-view lowering must fail closed when type conversion fails"
fi

ptr_view_fallbacks="$(
  grep -n -E 'convertedType[[:space:]]*=[^;]*ptrType|if[[:space:]]*\(!convertedType\).*ptrType|!convertedType[^?]*\?[^:]*ptrType|!convertedType[^?]*\?[^:]*:[^;]*ptrType' "$abi_decode" || true
)"

if [ -n "$ptr_view_fallbacks" ]; then
  echo "$ptr_view_fallbacks" >&2
  fail "ABI returndata ptr-view lowering must not fall back to ptrType"
fi

echo "check-abi-layout-ownership: ok"
