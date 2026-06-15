#!/usr/bin/env sh
set -eu

encoder="src/z3/encoder.zig"
grep_output="${TMPDIR:-/tmp}/ora_verifier_introspection_grep.$$"
variants_output="${TMPDIR:-/tmp}/ora_verifier_introspection_variants.$$"
labels_output="${TMPDIR:-/tmp}/ora_verifier_introspection_labels.$$"
trap 'rm -f "$grep_output" "$variants_output" "$labels_output"' EXIT INT HUP TERM

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

check_enum_label_parity() {
  enum_name="$1"
  label_fn="$2"

  awk -v enum_name="$enum_name" '
    $0 ~ "pub const " enum_name " = enum" {
      in_enum = 1
      next
    }
    in_enum && $0 ~ /^[[:space:]]*};/ {
      exit
    }
    in_enum {
      line = $0
      sub(/\/\/.*/, "", line)
      gsub(/[[:space:],]/, "", line)
      if (line != "") print line
    }
  ' "$encoder" >"$variants_output"

  [ -s "$variants_output" ] || fail "failed to extract $enum_name variants from $encoder"

  awk -v label_fn="$label_fn" '
    $0 ~ "pub fn " label_fn "\\(" {
      in_fn = 1
      next
    }
    in_fn && $0 ~ /^[[:space:]]*};/ {
      exit
    }
    in_fn {
      line = $0
      sub(/\/\/.*/, "", line)
      if (match(line, /\.[A-Za-z0-9_]+[[:space:]]*=>[[:space:]]*"[A-Za-z0-9_]+"/)) {
        text = substr(line, RSTART, RLENGTH)
        sub(/^\./, "", text)
        split(text, parts, /[[:space:]]*=>[[:space:]]*/)
        label = parts[2]
        gsub(/"/, "", label)
        print parts[1] " " label
      }
    }
  ' "$encoder" >"$labels_output"

  [ -s "$labels_output" ] || fail "failed to extract labels from $label_fn in $encoder"

  while IFS= read -r variant; do
    grep -x "$variant $variant" "$labels_output" >/dev/null ||
      fail "$label_fn must map $enum_name.$variant to \"$variant\""
  done <"$variants_output"

  variant_count="$(wc -l <"$variants_output" | tr -d ' ')"
  label_count="$(wc -l <"$labels_output" | tr -d ' ')"
  [ "$variant_count" = "$label_count" ] ||
    fail "$label_fn label count ($label_count) does not match $enum_name variant count ($variant_count)"
}

check_enum_label_parity "SoundnessLoss" "soundnessLossLabel"
check_enum_label_parity "PrecisionNoteKind" "precisionNoteLabel"

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
