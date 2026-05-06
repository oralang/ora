# TODO: Unsat Cores

**Scope.** Implementation plan for unsat-core support in Ora's SMT verifier. This document is intentionally limited to unsat cores, vacuity detection, and proof explainability for successful proofs. It does **not** include proof-generation/certificate support.

**Status.** Core explainability support landed. This document remains focused on unsat-core explainability; adjacent proof-object and core-minimization support now exists in the implementation, but those are not the primary subject of this document.

Current implemented behavior:

- opt-in explain mode via `--explain`
- explain mode always routed through the canonical prepared-query engine
- vacuity pre-check
- unsat-core extraction for successful proofs
- optional greedy core minimization via `--minimize-cores`
  - per-query `core_minimized` reporting means the core actually shrank, not merely that minimization was enabled
- structured explain tags in SMT report JSON/markdown
- tracked/source-level assumption kinds currently supported:
  - `requires`
  - `assume`
  - `loop_invariant`
  - `path_assume`
  - `callee_ensures`
  - `goal`

Current intentional limits:

- tracked assumptions are still intentionally narrow relative to the full solver state:
  - no generic internal replay/state constraints
  - no frame glue
  - no environment glue
  - no broad imported-callee obligation tracking in explain cores
- legacy fallback no longer owns explain behavior; explain bypasses it and uses prepared queries
- no independent proof checker / certificate export
- raw proof objects, when enabled separately, are not source-level explanations
- raw proof objects are kept in JSON reports; markdown stays source-oriented

---

## Goal

Add an opt-in explainability mode that can answer:

- which assumptions were load-bearing for a successful proof
- whether a proof is vacuous because the assumptions are themselves inconsistent
- which source-level assumptions contributed to a proof

This feature must sit on top of the **same prepared-query path** used for actual verification. No separate "explain" query builder is allowed.

---

## Non-goals

- no proof-object generation
- no independent proof checker
- no minimal-core minimization
- no instrumentation of every internal solver assertion
- no legacy/live verifier path support

Proof generation is a separate feature and should remain deferred until:

1. current soundness gaps are reduced further
2. the prepared-query path is stable
3. there is a concrete requirement for certificate export

---

## Why Unsat Cores

For an obligation query, Ora is proving:

`A1 ∧ A2 ∧ ... ∧ An ∧ ¬P` is UNSAT

where:

- `A1..An` are assumptions
- `P` is the property being proved

An unsat core gives a subset of those tracked assumptions that is already enough to make the query UNSAT.

This is useful for Ora because it exposes:

- vacuous proofs
- load-bearing assumptions
- audit/debug explanations for successful proofs

It does **not** explain SAT failures and it does **not** replace model extraction.

---

## Architectural Requirements

### 1. Prepared Queries Are Canonical

Unsat-core support must be implemented only on top of the prepared-query path in [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig).

Reason:

- live/prepared divergence already caused correctness drift in the verifier
- an explanation feature is only trustworthy if it uses the same query that was actually proved

### 2. Track Only Semantic Assumptions

Do **not** wrap every `solver.assert(...)` call.

Tracked assumptions should include:

- `requires`
- assumed loop invariants
- imported callee postconditions
- explicit path assumptions
- ghost axioms
- semantic environment assumptions

Untracked assertions should include:

- proxy implication glue
- old/current linkage glue
- internal replay/state reconstruction constraints
- cache/materialization glue
- generic encoder bookkeeping constraints

If everything is tagged, cores become noisy and lose value.

### 3. Use Assumption Proxies via Implication

Use the standard Z3 pattern:

- create fresh Bool proxy `p`
- assert `p => A`
- pass `p` into `check_assumptions`

Do **not** use `p <=> A` unless a concrete need appears later.

Reason:

- implication is cheaper and more standard for unsat cores
- avoids unnecessary reverse constraints

### 4. Separate Vacuity Detection From Proof Success

Unsat cores help expose vacuity, but vacuity should not be inferred heuristically from a single proof core.

Do two checks:

1. check assumptions only
2. if satisfiable, check assumptions plus negated goal

That gives:

- **assumptions UNSAT**: vacuous / inconsistent query
- **assumptions SAT, assumptions + ¬goal UNSAT**: real proof
- **assumptions SAT, assumptions + ¬goal SAT**: failing proof
- **UNKNOWN**: solver could not decide

---

## Proposed Data Model

Add a structured assumption tag in [verification.zig](/Users/logic/Ora/Ora/src/z3/verification.zig):

```zig
const AssumptionKind = enum {
    requires,
    loop_invariant,
    callee_ensures,
    ghost_axiom,
    path_assume,
    env_assume,
    frame,
    goal,
};

const AssumptionTag = struct {
    kind: AssumptionKind,
    function_name: []const u8,
    file: []const u8,
    line: u32,
    column: u32,
    label: []const u8,
    callee_name: ?[]const u8 = null,
    guard_id: ?[]const u8 = null,
    loop_owner: ?u64 = null,
};
```

Prepared queries should be extended to carry tracked assumptions explicitly:

```zig
const TrackedAssumption = struct {
    proxy: z3.Z3_ast,
    ast: z3.Z3_ast,
    tag: AssumptionTag,
};
```

Query representation should distinguish:

- tracked assumptions
- internal constraints
- goal

---

## Solver API Changes

### `src/z3/c.zig`

Add bindings for:

```zig
pub const Z3_solver_check_assumptions = c.Z3_solver_check_assumptions;
pub const Z3_solver_get_unsat_core = c.Z3_solver_get_unsat_core;
pub const Z3_mk_fresh_const = c.Z3_mk_fresh_const;
```

### `src/z3/solver.zig`

Add wrappers:

```zig
pub fn checkAssumptions(self: *Solver, assumptions: []const c.Z3_ast) !CheckResult
pub fn getUnsatCore(self: *Solver, allocator: std.mem.Allocator) ![]c.Z3_ast
```

Keep plain `check()` unchanged.

---

## Verifier Flow

### Phase 1: Build Query

Prepared-query construction should produce:

- `internal_constraints`
- `tracked_assumptions`
- `goal_ast`

Tracked assumptions must already be source-tagged when built.

### Phase 2: Vacuity Pre-check

For each obligation:

1. push solver frame
2. assert all internal constraints
3. for each tracked assumption:
   - create proxy
   - assert `proxy => assumption`
4. run `check_assumptions(tracked_proxies)`

If result is `UNSAT`:

- classify as vacuous/inconsistent assumptions
- retrieve unsat core
- map core proxies back to `AssumptionTag`
- report vacuity and stop

### Phase 3: Real Proof Check

If the assumptions-only query is satisfiable:

1. create `goal_proxy`
2. assert `goal_proxy => not(goal_ast)`
3. run `check_assumptions(tracked_proxies + goal_proxy)`

If result is `UNSAT`:

- retrieve unsat core
- translate proxies to tags
- record success with core

If result is `SAT`:

- normal failing-proof path

If result is `UNKNOWN`:

- normal unknown/degradation path

---

## Reporting

This starts behind a CLI flag:

```text
--explain
```

Default verification remains concise and fast.

### Success output

Show:

- obligation proved
- number of relevant assumptions
- compact list of source tags

Example:

```text
verified using 4 relevant assumptions:
- requires @ transferFrom:12
- loop invariant @ transferFrom:21
- callee ensures add_safe @ transferFrom:31
- goal @ transferFrom:44
```

### Vacuous output

Show:

- obligation/query is vacuous
- inconsistent assumptions
- core assumptions that already contradict

Example:

```text
warning: obligation is vacuous; assumptions are inconsistent:
- requires @ transferFrom:12
- path assumption @ transferFrom:28
```

### Unknown output

No core is expected. Report normal UNKNOWN diagnostics.

---

## Performance Strategy

Unsat cores should start as **opt-in** only.

Reason:

- `check_assumptions` is typically slower than plain `check`
- flagship timeout cases are already sensitive

Current policy:

- normal verification: plain `check`
- `--explain`: tracked assumptions + unsat cores

If the cost is acceptable later, core support can be broadened.

---

## Implementation Order

### Landed in V1

- C bindings and solver wrappers
- prepared-query tracked assumptions
- narrow tracked kinds:
  - `requires`
  - `loop_invariant`
  - `path_assume`
  - `goal`
- vacuity pre-check
- success cores in `--explain` mode
- report integration and regression coverage

### Next steps

1. Widen tracked assumptions carefully, starting with:
   - `callee_ensures`
2. Add more explain-mode regression coverage around:
   - vacuous queries
   - loop proofs
   - stateful call summaries
3. Decide whether explain-mode report metadata should surface more summary-level counters
4. Keep proof generation deferred until solver/model/theory work is further along

---

## Tests

### Vacuity

Contract with contradictory assumptions:

- expect assumptions-only query is `UNSAT`
- expect vacuity classification
- expect unsat core over assumptions

### Normal success

Small proof with 2–3 assumptions:

- expect proof UNSAT
- expect goal proxy in the core
- expect relevant assumptions only

### Stateful call summary

Use a contract where a callee ensures is load-bearing:

- expect callee tag in the core once that category is enabled

### Loop proof

Loop invariant proof:

- expect invariant assumption to appear in the core when relevant

### Failing proof

SAT case:

- no unsat core path

### UNKNOWN

UNKNOWN case:

- no unsat core path

### Regression

With `--explain` disabled:

- current verification behavior should remain unchanged

---

## Deferred: Proof Generation

Proof generation is a different feature.

It should remain deferred because:

- it does not fix current soundness issues
- it is significantly more expensive
- it is not needed for immediate audit/debug value

When revisited later, it should be added as a separate context/solver mode and not mixed into the unsat-core rollout.

---

## Acceptance Criteria

The v1 unsat-core feature is complete when:

1. Ora can classify vacuous queries explicitly.
2. Ora can report a source-tagged unsat core for successful proofs in `--explain` mode.
3. The implementation uses the prepared-query path only.
4. Internal bookkeeping constraints do not pollute the reported core.
5. Normal verification behavior remains unchanged when `--explain` is off.
