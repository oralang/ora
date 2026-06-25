#!/usr/bin/env sh
set -eu

fail() {
  echo "check-no-scattered-process-exit: $*" >&2
  exit 1
}

[ -d src ] || fail "missing src directory"

matches="$(
  grep -R -n 'std\.process\.exit' src tests --include="*.zig" \
    | grep -v -E '^[^:]+:[0-9]+:[[:space:]]*std\.process\.exit\(code\);[[:space:]]*$' \
    || true
)"

if [ -n "$matches" ]; then
  echo "$matches" >&2
  fail "raw std.process.exit is only allowed inside a local exitCli(code) boundary; return typed errors elsewhere"
fi

echo "check-no-scattered-process-exit: ok"
