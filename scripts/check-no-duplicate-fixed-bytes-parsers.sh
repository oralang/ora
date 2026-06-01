#!/usr/bin/env sh
set -eu

fail() {
  echo "check-no-duplicate-fixed-bytes-parsers: $*" >&2
  exit 1
}

[ -d src ] || fail "missing src directory"

matches="$(
  grep -R -n -E 'fn (parseFixedBytesSpelling|runtimeFixedBytesSpellingLen|isFixedBytesWireType)|std\.mem\.startsWith\(u8, [^,]+, "bytes"\)' src --include="*.zig" \
    | grep -v -E 'src/types/builtin\.zig:' \
    | grep -v -E 'src/abi/layout\.zig:[0-9]+:pub fn parseFixedBytesSpelling' \
    | grep -v -E 'src/hir/abi\.zig:[0-9]+:pub fn parseFixedBytesSpelling' \
    || true
)"

if [ -n "$matches" ]; then
  echo "$matches" >&2
  fail "fixed-bytes parsing must be owned by src/types/builtin.zig; ABI wrappers may delegate but must not parse locally"
fi

echo "check-no-duplicate-fixed-bytes-parsers: ok"
