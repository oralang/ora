#!/usr/bin/env bash
#
# check-formal-sync — proves the compiler-emitted formal facts conform to the
# Lean spec (formal/). Four gates:
#
#   1. REGENERATE  all data-only snapshots from the compiler (`src/formal/*`):
#      type universe, curated declaration environment, type-relation rows, and
#      storage disjointness fixtures.
#   2. DATA-ONLY LINT  every snapshot — each must contain ONLY `def … := <literal>`
#      facts (no theorem/axiom/sorry/instance/import/…). This is the trust
#      boundary: the untrusted compiler emits DATA; the trusted checks live in
#      formal/Ora/*.lean and do all the proving.
#   3. DRIFT + PROOF: fail if any committed snapshot is stale (git diff), then
#      `lake build` — the kernel checks that compiler facts == spec.
#   4. AXIOM AUDIT: regenerate the ignored whole-tree theorem audit and run it
#      through Lean; only propext/Quot.sound dependencies are allowed.
#
# Local: run it, and if step 3 reports drift, review + commit the new snapshot.
# CI: a non-empty git diff means "compiler changed but snapshot not regenerated".
#
# Usage:  scripts/check-formal-sync.sh [--skip-mlir]
#
#   --skip-mlir  Pass -Dskip-mlir=true to `zig build` emitter steps. Requires a
#                prebuilt vendor/mlir (e.g. CI's mlir-build artifact); the
#                dispatcher-table emitter links the MLIR/C++ stack and this
#                skips the vendored-LLVM CMake build it would otherwise need.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.elan/bin:$PATH"

BUILD_FLAGS=()
for arg in "$@"; do
  case "$arg" in
    --skip-mlir) BUILD_FLAGS+=("-Dskip-mlir=true") ;;
    *)
      echo "error: unknown argument '$arg' (supported: --skip-mlir)" >&2
      exit 2
      ;;
  esac
done

run_emitter() {
  local label="$1"
  local emitter="$2"
  local snapshot="$3"

  echo "  - $label -> $snapshot"
  zig run \
    --cache-dir "$ROOT/.zig-cache/formal-sync" \
    --global-cache-dir "$ROOT/.zig-cache/formal-sync-global" \
    --dep ora_formal \
    -Mroot="$emitter" \
    --dep ora_types \
    --dep ora_refinements \
    -Mora_formal=src/formal.zig \
    --dep ora_refinements \
    -Mora_types=src/types/root.zig \
    -Mora_refinements=src/refinements/root.zig \
    > "$snapshot"
}

run_sinora_emitter() {
  local label="$1"
  local emitter="$2"
  local snapshot="$3"

  echo "  - $label -> $snapshot"
  zig run \
    --cache-dir "$ROOT/.zig-cache/formal-sync" \
    --global-cache-dir "$ROOT/.zig-cache/formal-sync-global" \
    --dep sinora \
    -Mroot="$emitter" \
    -Msinora=sinora/src/sinora.zig \
    > "$snapshot"
}

run_build_emitter() {
  local label="$1"
  local step="$2"
  local snapshot="$3"

  echo "  - $label -> $snapshot"
  zig build "$step" ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"} --summary none -- "$snapshot"
}

lint_snapshot() {
  local snapshot="$1"
  python3 scripts/formal_snapshot_lint.py "$snapshot"
}

check_drift() {
  local snapshot="$1"

  if ! git diff --quiet -- "$snapshot"; then
    echo "  SNAPSHOT DRIFT — $snapshot is stale vs the compiler:" >&2
    git --no-pager diff -- "$snapshot" >&2
    echo "  (review the diff; if intended, commit the regenerated snapshot)" >&2
    exit 1
  fi
}

SNAPSHOTS=(
  "formal/Ora/Generated/CompilerSnapshot.lean"
  "formal/Ora/Generated/DeclEnvSnapshot.lean"
  "formal/Ora/Generated/CompilerTypeRelations.lean"
  "formal/Ora/Generated/DispatcherStrategySnapshot.lean"
  "formal/Ora/Generated/DispatcherTableSnapshot.lean"
  "formal/Ora/Generated/SinoraBackendSnapshot.lean"
  "formal/Ora/Generated/StorageDisjointnessSnapshot.lean"
  "formal/Ora/Generated/ObligationTotalitySnapshot.lean"
  "formal/Ora/Generated/LoopTotalitySnapshot.lean"
  "formal/Ora/Generated/SourceAccountingSnapshot.lean"
)

echo "==> [1/4] regenerating formal snapshots from the compiler"
run_emitter "compiler universe" "src/formal/emit_compiler_snapshot.zig" "${SNAPSHOTS[0]}"
run_emitter "declaration environment" "src/formal/emit_declenv_snapshot.zig" "${SNAPSHOTS[1]}"
run_emitter "type relations" "src/formal/emit_type_relations_snapshot.zig" "${SNAPSHOTS[2]}"
run_sinora_emitter "dispatcher strategies" "src/formal/emit_dispatcher_strategy_snapshot.zig" "${SNAPSHOTS[3]}"
run_build_emitter "dispatcher tables" "emit-dispatcher-table-snapshot" "${SNAPSHOTS[4]}"
run_sinora_emitter "sinora backend" "src/formal/emit_sinora_backend_snapshot.zig" "${SNAPSHOTS[5]}"
run_emitter "storage disjointness" "src/formal/emit_storage_disjointness_snapshot.zig" "${SNAPSHOTS[6]}"
run_emitter "obligation totality" "src/formal/emit_obligation_totality_snapshot.zig" "${SNAPSHOTS[7]}"
run_emitter "loop totality" "src/formal/emit_loop_totality_snapshot.zig" "${SNAPSHOTS[8]}"
run_emitter "source accounting" "src/formal/emit_source_accounting_snapshot.zig" "${SNAPSHOTS[9]}"

echo "==> [2/4] data-only lint"
python3 scripts/test_formal_snapshot_lint.py
for snapshot in "${SNAPSHOTS[@]}"; do
  lint_snapshot "$snapshot"
done
echo "  ok (data only)"

echo "==> [3/4] drift check + lake build"
for snapshot in "${SNAPSHOTS[@]}"; do
  check_drift "$snapshot"
done
( cd formal && lake build )

# NOTE: the lawfulness tiers — `Ora/Types/TypeEqLawful.lean` (`Ty.beq`: reflexivity,
# `beq ↔ =`, `DecidableEq Ty`) and `Ora/Types/AssignableLawful.lean` (`Ty.assignable`
# is a preorder: reflexive + transitive) — used to be isolated opt-ins here because
# their proofs cost tens of seconds. They were restructured to induct via `Ty.recAux`
# and unfold `Ty.beq`/`Ty.assignable` through cheap per-constructor `@[simp]`
# `rfl`-lemmas (no `maxHeartbeats` bump), so each now builds in ~1.5s and both are
# imported by the `Ora` root — the `lake build` above already compiles and
# drift-checks them. No special-casing.

echo "==> [4/4] Lean axiom audit"
mkdir -p formal/.lake
python3 scripts/check-formal-trust-surface.py > formal/.lake/ora-formal-axiom-audit.lean
( cd formal && lake env lean .lake/ora-formal-axiom-audit.lean )

echo "==> check-formal-sync OK"
