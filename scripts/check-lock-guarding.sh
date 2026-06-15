#!/usr/bin/env sh
set -eu

hir="src/hir/function_core.zig"

# This check assumes all current-contract storage writes that need lock guards
# are lowered through src/hir/function_core.zig. If a new HIR storage-write
# lowering file or family is added, extend this script and the user-locks
# criterion-5 evidence before treating that family as guarded.

fail() {
  echo "check-lock-guarding: $*" >&2
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

nth_line_of() {
  file="$1"
  needle="$2"
  target="$3"
  awk -v needle="$needle" -v target="$target" '
    index($0, needle) {
      count++
      if (count == target) {
        print NR
        exit
      }
    }
  ' "$file"
}

assert_count() {
  needle="$1"
  expected="$2"
  actual="$(count_occurrences "$hir" "$needle")"
  [ "$actual" = "$expected" ] ||
    fail "expected $expected occurrence(s) of '$needle' in $hir, found $actual"
}

assert_before_close() {
  label="$1"
  before="$2"
  after="$3"
  max_distance="$4"
  [ -n "$before" ] || fail "$label: missing guard line"
  [ -n "$after" ] || fail "$label: missing write line"
  [ "$before" -lt "$after" ] ||
    fail "$label: guard line $before must appear before write line $after"
  distance=$((after - before))
  [ "$distance" -le "$max_distance" ] ||
    fail "$label: guard line $before is too far from write line $after"
}

[ -f "$hir" ] || fail "missing HIR lowering source $hir"

assert_count "mlir.oraSStoreOpCreate" 1
assert_count "const op = mlir.oraMemrefStoreOpCreate(" 1
assert_count "try @This().appendMapStore(self," 1
assert_count "mlir.oraMapStoreOpCreate" 1
assert_count "maybeEmitGuardedStorageWrite(self, field.name" 1
assert_count "maybeEmitGuardedIndexedStorageWrite(self, root_name, key_value, index.range)" 2

root_guard_line="$(line_of "$hir" "maybeEmitGuardedStorageWrite(self, field.name")"
sstore_line="$(line_of "$hir" "mlir.oraSStoreOpCreate")"
assert_before_close "root storage write" "$root_guard_line" "$sstore_line" 3

indexed_memref_guard_line="$(nth_line_of "$hir" "maybeEmitGuardedIndexedStorageWrite(self, root_name, key_value, index.range)" 1)"
memref_store_line="$(line_of "$hir" "const op = mlir.oraMemrefStoreOpCreate(")"
assert_before_close "indexed memref storage write" "$indexed_memref_guard_line" "$memref_store_line" 5

indexed_map_guard_line="$(nth_line_of "$hir" "maybeEmitGuardedIndexedStorageWrite(self, root_name, key_value, index.range)" 2)"
map_store_call_line="$(line_of "$hir" "try @This().appendMapStore(self,")"
assert_before_close "indexed map storage write" "$indexed_map_guard_line" "$map_store_call_line" 3

append_map_store_line="$(line_of "$hir" "fn appendMapStore(")"
map_store_create_line="$(line_of "$hir" "mlir.oraMapStoreOpCreate")"
[ "$append_map_store_line" -lt "$map_store_create_line" ] ||
  fail "map store creation must remain encapsulated by appendMapStore"

echo "check-lock-guarding: ok"
