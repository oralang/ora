# Conformance Findings Ledger

Execution-discovered compiler/EVM findings, one entry each. Corpus `.ora`/`.spec.toml` files
carry only `// see FINDINGS.md#<id>` pointers — never finding prose (test-quality program T1.6).
When a finding is fixed: flip its blocked/characterization rows as described, run
`zig build test-conformance`, and mark the entry FIXED with the commit.

---

## F-001 — `requires` not boundary-enforced; discharged checks erased (S1)

- **Status:** OPEN (escalated to SMT soundness audit, 2026-06-11)
- **What:** `requires` clauses on pub fns are verification-only assumptions — the dispatcher
  emits no guard for them. Additionally, obligations the verifier discharges *using* those
  assumptions are removed from runtime code. Net: `pub fn div(a,b) requires b != 0` has no
  zero-check at all; an external `div(7,0)` executes EVM DIV and returns **silent 0**.
  Asymmetry: the checked-add overflow assert is retained (`add(MAX,1)` reverts correctly).
- **Repro:** `tests/conformance/arithmetic_checked_revert.ora` (also minimal:
  any pub fn with `requires b != 0` + `a / b`; IR shows `ora.requires` but no assert).
- **Pinned rows (ACTIVE characterization, `arithmetic_checked_revert.spec.toml`):**
  `add(2000000,1) returns 2000001` (requires violated, executes) and
  `div(7,0) returns 0` (silent zero).
- **Flip condition:** when pub-fn requires lower to dispatcher guards, change both rows to
  `reverts` assertions.

## F-002 — catch-path unpacking of aggregate error-union payloads is miscompiled (S1-class)

- **Status:** OPEN
- **What:** `catch` after a `try` that actually returns an error, on an error union whose
  payload is an AGGREGATE (wide struct, string), dereferences a garbage payload pointer →
  huge-offset `mload`. Scalar payloads are fine (properties suite). Ok paths, plain `match`,
  and tuple-from-match all return correct values — only catch-on-actual-Err breaks; both
  payload kinds show one root cause. Likely the src-full-review "mload fresh-malloc" S1 class.
- **Repro:** `error_union_wide_carrier.ora` `err_default()`,
  `error_union_local_aggregate.ora` `err_marker()` (call-level bisected 2026-06-11).
- **Blocked rows (commented in the spec.toml files):** `err_default()` returns 777;
  `err_marker()` returns 999.
- **Flip condition:** un-comment both rows when the catch-path lowering is fixed.

## F-003 — lib/evm panics (integer overflow) on huge-offset mload gas computation

- **Status:** OPEN (lib/evm robustness, surfaced by F-002)
- **What:** `handlers_memory.zig:35` `GasConstants.GasFastestStep + mem_cost` overflows and
  PANICS the interpreter instead of failing cleanly (out-of-gas / error). The conformance
  oracle must never panic on hostile-but-valid bytecode behavior; the R9 Anvil differential
  would cross-check this class.
- **Repro:** run either F-002 blocked row.
- **Flip condition:** none in specs; fix = saturating/checked gas math returning OutOfGas.
  F-002's rows stay blocked on F-002 regardless.

## F-004 — tuple return from inside try/catch fails OraToSIR Phase3b (fails closed)

- **Status:** OPEN
- **What:** returning a `(u256, string)` tuple from within a try/catch block fails
  legalization: "unresolved materialization from !ora.tuple<i256, !ora.string> to
  !sir.ptr<1>". Same tuple from a `match` arm or plain function works. Compile-time error,
  no bytecode (Law holds) — capability gap.
- **Repro:** add `return (0, "none");` inside a `catch` block of a
  `-> (u256, string)` pub fn in `error_union_local_aggregate.ora`.
- **Blocked coverage:** try/catch paths in `error_union_local_aggregate.ora` assert via
  `.len` instead of string content.
- **Flip condition:** when the lowering lands, add content-bearing tuple returns to the
  try/catch fns and assert `(uint256,string)` rows.
