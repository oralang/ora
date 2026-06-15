#!/usr/bin/env sh
set -eu

fail() {
  echo "check-query-view-ownership: $*" >&2
  exit 1
}

[ -d src ] || fail "missing src directory"

local_structs="$(
  grep -R -n -E '(pub[[:space:]]+)?const[[:space:]]+(ImportQuery|TypeQuery|ModuleQuery)[[:space:]]*=[[:space:]]*struct' src --include="*.zig" || true
)"

if [ -n "$local_structs" ]; then
  echo "$local_structs" >&2
  fail "query capability structs must be owned by src/compiler_query.zig"
fi

raw_context_calls="$(
  grep -R -n -E '(query|type_query|module_query|self\.module_query\.\?)\.[[:alnum:]_]+[[:space:]]*\((query|type_query|module_query|self\.module_query\.\?)\.context' src --include="*.zig" \
    | grep -v -E 'src/compiler_query\.zig:' \
    || true
)"

if [ -n "$raw_context_calls" ]; then
  echo "$raw_context_calls" >&2
  fail "stage code must call query-view methods instead of passing query.context manually"
fi

db_wiring_outside_constructors="$(
  awk '
    /^    fn (semaQueryView|comptimeQueryView|hirQueryView)\(/ { in_constructor = 1; next }
    /^    fn [[:alnum:]_]+\(/ { in_constructor = 0 }
    /\.(ast_file|item_index|module_typecheck|lookup_item|resolve_import_alias|ensure_typecheck|resolution|module_verification_facts|const_eval)[[:space:]]*=[[:space:]]*(astFileForComptime|itemIndexForComptime|moduleTypeCheckForComptime|lookupItemForComptime|resolveImportAliasForComptime|ensureTypeCheckedForComptime|resolveNamesForComptime|moduleVerificationFactsForComptime|constEvalForComptime)/ {
      if (!in_constructor) print FILENAME ":" FNR ":" $0
    }
  ' src/db/mod.zig
)"

if [ -n "$db_wiring_outside_constructors" ]; then
  echo "$db_wiring_outside_constructors" >&2
  fail "CompilerDb query callback wiring must stay inside the view constructors"
fi

echo "check-query-view-ownership: ok"
