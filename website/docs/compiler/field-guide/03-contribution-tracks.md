# Contribution Tracks (where to start today)

This chapter exists for one reason: **so a strong engineer can contribute today** even without compiler experience.

Pick a track. Each track includes:
- what you will deliver
- what “done” looks like
- common first tasks
- what you’ll learn by doing it

---

## Track A — Tests & Minimal Repros (best first track)

**Ideal for:** any engineer.  
**Risk level:** very low.  
**Impact:** very high.

### Deliverables
- a tiny `.ora` file demonstrating a behavior (pass or fail)
- a test that checks output shape or diagnostics

### What “done” looks like
- the test fails before your fix and passes after
- the repro is small enough that anyone can understand it in 2 minutes

### Typical first tasks
- add regression tests for parse recovery
- add tests for type rules (e.g., error unions require `try`)
- add tests for MLIR emission of a small feature

### What you learn
- how to run phases in isolation
- how diagnostics flow through spans
- how changes in one phase influence downstream artifacts

---

## Track B — Diagnostics & UX

**Ideal for:** engineers who like polish.  
**Risk level:** low.  
**Impact:** high (this unlocks new contributors).

### Deliverables
- improved error message text
- improved span selection (point to the right place)
- “next action” hints (“run `ora parse`…”, “inspect typed AST…”)

### What “done” looks like
- a user reading an error understands what to do next
- the error points to the correct source region
- tests updated if they depend on exact text

### Typical first tasks
- pick one frequently hit error and add:
  - what happened
  - why it happened
  - what to do next

### What you learn
- how spans and diagnostics are wired
- where in the pipeline a particular class of errors is produced

---

## Track C — Examples & Docs

**Ideal for:** documentation-minded contributors.  
**Risk level:** very low.  
**Impact:** medium-to-high (community growth).

### Deliverables
- a short example program demonstrating one feature
- a short explanation:
  - what it demonstrates
  - expected output or expected diagnostic
  - why it matters

### What “done” looks like
- examples are small and single-purpose
- they can be used as test fixtures later

### Typical first tasks
- expand from the canonical walkthrough examples:
  - refinements
  - error unions + `try`
  - simple storage updates

### What you learn
- the language surface area
- the mapping between language features and compiler phases

---

## Track D — “Small, safe compiler changes”

**Ideal for:** engineers ready to touch internals but want low blast radius.  
**Risk level:** medium-low.

### Deliverables
One small change with tests:
- add a keyword token
- add a tiny parse rule
- add a small typing rule
- add a lowering mapping for an existing node form

### What “done” looks like
- the change is limited to 1–3 phases
- you updated or added tests
- you can explain the change using the artifact ladder

### What you learn
- how features flow end-to-end
- how to keep the compiler coherent across phases

---

## Track E — Verification (Z3/SMT), optional

**Ideal for:** people who enjoy constraints/models.  
**Risk level:** medium.  
**Impact:** high for safety features.

### Deliverables
- encode a missing MLIR op into the SMT layer
- add a test that demonstrates correctness or a useful counterexample

### What “done” looks like
- verification produces stable, understandable results
- adding new IR ops doesn’t silently “fall out of verification”

### What you learn
- how Ora’s “requires/ensures/invariant” become solver constraints
- how counterexamples improve the language/compiler

---

## How to choose (fast)

If you’re unsure:
1) start with **Track A** (tests + repros)  
2) then do **Track B** (diagnostics)  
3) then do **Track D** (small internal change)  

That progression onboards you into the compiler naturally.
