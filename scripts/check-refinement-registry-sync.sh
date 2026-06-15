#!/usr/bin/env sh
set -eu

registry="src/refinements/root.zig"
tablegen="src/mlir/ora/td/OraTypes.td"
docs="website/docs/compiler/what-ora-proves.md plan_smt.md"
grep_output="${TMPDIR:-/tmp}/ora_refinement_registry_sync_grep.$$"
trap 'rm -f "$grep_output"' EXIT INT HUP TERM

fail() {
  echo "check-refinement-registry-sync: $*" >&2
  exit 1
}

registry_entries() {
  awk '
    /pub const entries =/ { in_entries=1; next }
    in_entries && /\.name = "/ {
      line=$0
      native=line
      sub(/.*\.name = "/, "", line)
      sub(/".*/, "", line)
      if (index(native, ".has_native_mlir_type = true") > 0) {
        print line " true"
      } else if (index(native, ".has_native_mlir_type = false") > 0) {
        print line " false"
      } else {
        print line " unknown"
      }
    }
    in_entries && /^};/ { exit }
  ' "$registry"
}

entries="$(registry_entries)"
[ -n "$entries" ] || fail "failed to extract refinement registry entries from $registry"

[ -f "$tablegen" ] || fail "missing TableGen refinement surface $tablegen"

printf '%s\n' "$entries" | while read -r name native; do
  [ "$native" != "unknown" ] || fail "registry entry '$name' is missing has_native_mlir_type"
  if grep -n "Ora_Type<\"$name\"," "$tablegen" >"$grep_output"; then
    [ "$native" = "true" ] || {
      cat "$grep_output" >&2
      fail "$tablegen declares native MLIR type for non-native refinement '$name'"
    }
  else
    [ "$native" = "false" ] ||
      fail "$tablegen is missing native MLIR type for refinement '$name'"
  fi
done

for doc in $docs; do
  if [ ! -f "$doc" ]; then
    echo "check-refinement-registry-sync: note: docs surface $doc not present in repo, skipping" >&2
    continue
  fi
  printf '%s\n' "$entries" | while read -r name _native; do
    needle='`'"$name"'`'
    grep -F "$needle" "$doc" >/dev/null ||
      fail "$doc is missing backtick-wrapped refinement '$needle'"
  done
done

echo "check-refinement-registry-sync: ok"
