#!/usr/bin/env sh
set -eu

fail() {
  echo "check-abi-layout-ownership: $*" >&2
  exit 1
}

[ -d src ] || fail "missing src directory"

matches="$(
  grep -R -n -E "StaticEncoding|LayoutNode \{|abi/layout\.zig|abi_layout\." src --include="*.zig" \
    | grep -v -E "src/abi/layout\.zig:|src/abi/layout_context\.zig:|src/abi/comptime_encoder\.zig:|src/abi/runtime_encoder\.zig:|src/abi\.test\.zig:" \
    | grep -v -E "src/hir/abi\.zig:|src/root\.zig:" \
    || true
)"

if [ -n "$matches" ]; then
  echo "$matches" >&2
  fail "layout-bearing code must live in src/abi/layout*.zig or sanctioned ABI materializers; tests may reference it from src/abi.test.zig"
fi

echo "check-abi-layout-ownership: ok"
