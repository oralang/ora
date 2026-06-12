#!/usr/bin/env sh
set -eu

fail() {
  echo "check-no-width-defaults: $*" >&2
  exit 1
}

[ -d src ] || fail "missing src directory"

matches="$(
  grep -R -n -E '\.bits[[:space:]]+orelse|\.signed[[:space:]]+orelse|\.bits\.\?|\.signed\.\?|\.bits[[:space:]]*==[[:space:]]*null|\.signed[[:space:]]*==[[:space:]]*null|@as\(\?u16|@as\(\?bool|\.integer = \.\{[[:space:]]*\}|\.integer = \.\{[[:space:]]*\.spelling|integer\.spelling == null' src --include="*.zig" \
    | grep -v -E 'src/types/builtin\.zig:' \
    | grep -v -E 'src/abi/type_names\.zig:[0-9]+:.*spec\.signed orelse @compileError' \
    | grep -v -E 'src/hir/support\.zig:[0-9]+:.*spec\.signed orelse return null' \
    | grep -v -E 'src/sema/type_descriptors\.zig:[0-9]+:.*spec\.signed orelse return null' \
    || true
)"

if [ -n "$matches" ]; then
  echo "$matches" >&2
  fail "integer width/signedness must be resolved explicitly; no local defaults or empty integer descriptors"
fi

literal_matches="$(
  grep -R -n -E 'parse(Int|UnsignedInteger)Literal.*orelse[[:space:]]*0' src --include="*.zig" || true
)"

if [ -n "$literal_matches" ]; then
  echo "$literal_matches" >&2
  fail "integer literal parse failures must not default to zero"
fi

fallback_matches="$(
  grep -R -n 'unknownTypeFallbackI256' src/hir --include="*.zig" \
    | grep -v -E 'src/hir/support\.zig:[0-9]+:pub fn unknownTypeFallbackI256' \
    | grep -v -E 'src/hir/mod\.zig:[0-9]+:[[:space:]]*return support\.unknownTypeFallbackI256\(self\.context\);' \
    || true
)"

if [ -n "$fallback_matches" ]; then
  echo "$fallback_matches" >&2
  fail "unknown i256 fallback must only be returned by counted recordTypeFallback"
fi

echo "check-no-width-defaults: ok"
