#!/usr/bin/env bash
# Pre-push bar: run `zig build gate` on the COMMITTED state.
# Uncommitted work (tracked + untracked) is stashed for the duration and always
# restored. Install: ln -sf ../../scripts/pre-push-gate.sh .git/hooks/pre-push
# Bypass (emergencies only): git push --no-verify
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"

stashed=0
if [ -n "$(git status --porcelain)" ]; then
  git stash push --include-untracked --quiet -m "pre-push-gate auto-stash"
  stashed=1
  echo "pre-push-gate: uncommitted work stashed; validating committed state"
fi

restore() {
  if [ "$stashed" -eq 1 ]; then
    if ! git stash pop --quiet; then
      echo "pre-push-gate: 'git stash pop' FAILED — your uncommitted work is safe in the top stash entry ('git stash list'); resolve manually." >&2
    fi
  fi
}
trap restore EXIT

zig build gate
