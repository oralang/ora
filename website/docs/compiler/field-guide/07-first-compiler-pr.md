# Your First Compiler PR (guided "safe" changes)

This chapter gives you three starter PRs. Each PR is designed to be:
- small enough to finish
- valuable even if you’re new
- easy to review
- aligned with the pipeline stages

---

## PR 1 — Add a regression test for a typing rule (Lane A → Lane B bridge)

**Goal:** lock in a rule that prevents future regressions.

### Steps
1) create a minimal `.ora` file that triggers the diagnostic  
2) add a test expecting that diagnostic  
3) ensure spans point to the right place (if span selection is part of the fix)

### Why it’s safe
- it strengthens the project without touching core logic first
- it gives maintainers confidence to accept future refactors

---

## PR 2 — Improve one error message in lowering (Lane A friendly)

**Goal:** make one common failure actionable.

### What to do
Pick one lowering error and add:
- what happened
- why it happened
- a “next action” hint (what phase to inspect next)

### Why it’s safe
- doesn’t change semantics
- unlocks new contributors
- improves the product immediately

---

## PR 3 — Add one missing conversion pattern (Lane B starter)

**Goal:** fix a concrete “Ora → SIR conversion failed” issue.

### What to do
1) find the illegal or unsupported op in the emitted MLIR  
2) locate the appropriate conversion pattern directory  
3) add a conversion pattern for the op  
4) add a test proving the conversion now succeeds

### Why it’s safe
- conversion patterns are localized
- correctness is visible in the IR
- tests are easy to write (golden MLIR or “conversion succeeds”)

---

## Review checklist (for all PRs)

Before you open your PR:
- Can you explain the issue using the artifact ladder?
- Did you add a minimal repro file?
- Did you add or update tests?
- Does the fix touch the smallest possible phase?

If yes, you’re doing compiler work correctly.
