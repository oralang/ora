#!/usr/bin/env sh
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPILER="${ORA_COMPILER:-$ROOT/zig-out/bin/ora}"
OUTPUT="${ORA_SIR_FRAMEWORK_SPIKE_REPORT:-$ROOT/zig-out/metrics/sir-framework-canonicalizer-spike.md}"
TIMEOUT="${ORA_SIR_FRAMEWORK_SPIKE_TIMEOUT:-120}"
has_path=0
skip_next=0
for arg in "$@"; do
  if [ "$skip_next" -eq 1 ]; then
    skip_next=0
    continue
  fi

  case "$arg" in
    --compiler|--emit|--timeout|--output|--compiler-arg)
      skip_next=1
      ;;
    --compiler=*|--emit=*|--timeout=*|--output=*|--compiler-arg=*)
      ;;
    -*)
      ;;
    *)
      has_path=1
      ;;
  esac
done

if [ "$has_path" -eq 0 ]; then
  set -- "$@" "$ROOT/tests/conformance"
fi

python3 "$ROOT/scripts/test_ora_features.py" \
  --compiler "$COMPILER" \
  --emit bytecode \
  --timeout "$TIMEOUT" \
  --compiler-arg=--no-verify \
  --all-expected-pass \
  --output "$OUTPUT" \
  "$@"

echo "sir-framework-canonicalizer-spike: wrote $OUTPUT"
