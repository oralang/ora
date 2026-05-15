#!/usr/bin/env sh
set -eu

encoder="src/z3/encoder.zig"
grep_output="${TMPDIR:-/tmp}/ora_verifier_introspection_grep.$$"
trap 'rm -f "$grep_output"' EXIT INT HUP TERM

set -- \
  src/z3/c.zig \
  src/z3/context.zig \
  src/z3/encoder.zig \
  src/z3/errors.zig \
  src/z3/mlir_helpers.zig \
  src/z3/mod.zig \
  src/z3/solver.zig \
  src/z3/verification.zig

fail() {
  echo "check-verifier-introspection: $*" >&2
  exit 1
}

count_occurrences() {
  file="$1"
  needle="$2"
  awk -v needle="$needle" '
    index($0, needle) { count++ }
    END { print count + 0 }
  ' "$file"
}

line_of() {
  file="$1"
  needle="$2"
  awk -v needle="$needle" '
    index($0, needle) { print NR; exit }
  ' "$file"
}

forall_count="$(count_occurrences "$encoder" "Z3_mk_forall_const")"
exists_count="$(count_occurrences "$encoder" "Z3_mk_exists_const")"
[ "$forall_count" = "1" ] || fail "expected exactly one Z3_mk_forall_const in $encoder, found $forall_count"
[ "$exists_count" = "1" ] || fail "expected exactly one Z3_mk_exists_const in $encoder, found $exists_count"

helper_start="$(line_of "$encoder" "fn mkQuantifier")"
helper_end="$(line_of "$encoder" "SENTINEL: end_of_mkQuantifier_helpers")"
[ -n "$helper_start" ] || fail "missing mkQuantifier helper in $encoder"
[ -n "$helper_end" ] || fail "missing mkQuantifier sentinel in $encoder"

forall_line="$(line_of "$encoder" "Z3_mk_forall_const")"
exists_line="$(line_of "$encoder" "Z3_mk_exists_const")"
[ "$forall_line" -gt "$helper_start" ] && [ "$forall_line" -lt "$helper_end" ] ||
  fail "Z3_mk_forall_const must stay inside mkQuantifier"
[ "$exists_line" -gt "$helper_start" ] && [ "$exists_line" -lt "$helper_end" ] ||
  fail "Z3_mk_exists_const must stay inside mkQuantifier"

if grep -n "\.noteDegradationAtOp(" "$@" >"$grep_output"; then
  cat "$grep_output" >&2
  fail "production z3 code must use typed noteSoundnessLossAtOp instead of noteDegradationAtOp"
fi

for legacy in mkUndefValue degradeToUndef; do
  if grep -n "$legacy" "$@" >"$grep_output"; then
    cat "$grep_output" >&2
    fail "legacy undef/degradation API '$legacy' must not reappear in production z3 code"
  fi
done

echo "check-verifier-introspection: ok"
