# Conformance Findings Ledger

Execution-discovered compiler/EVM findings, one entry each. Corpus `.ora`/`.spec.toml` files
carry only `// see FINDINGS.md#<id>` pointers — never finding prose (test-quality program T1.6).
When a finding is fixed: flip its blocked/characterization rows as described, run
`zig build test-conformance`, and mark the entry FIXED with the commit.

---

## F-001 — `requires` not boundary-enforced; discharged checks erased (S1)

- **Status:** OPEN (escalated to SMT soundness audit, 2026-06-11)
- **Severity:** S1
- **Owner:** compiler+smt-audit
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
- **Severity:** S1
- **Owner:** compiler
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
- **Severity:** S2
- **Owner:** lib/evm
- **What:** `handlers_memory.zig:35` `GasConstants.GasFastestStep + mem_cost` overflows and
  PANICS the interpreter instead of failing cleanly (out-of-gas / error). The conformance
  oracle must never panic on hostile-but-valid bytecode behavior; the R9 Anvil differential
  would cross-check this class.
- **Repro:** run either F-002 blocked row.
- **Flip condition:** none in specs; fix = saturating/checked gas math returning OutOfGas.
  F-002's rows stay blocked on F-002 regardless.

## F-006 — narrow error-union error path reverts empty; success encoding untrusted (S1)

- **Status:** OPEN
- **Severity:** S1
- **Owner:** compiler
- **What:** a public fn returning `!T | E` with a NARROW (<256-bit) carrier is mishandled at the
  dispatcher boundary. The WIDE carrier (`!u256 | E`) is correct, which is why the existing
  `dispatcher_error_union_revert` test (a `!u256` fn) never caught this. Confirmed 2026-06-12:
  - **Error path (CONFIRMED, reproduced twice):** a narrow `!bool | E` error reverts with EMPTY
    data (0 bytes) — the 4-byte error selector is LOST. A wide `!u256 | E` error reverts with the
    selector correctly. Lost-error-identity, S1-class.
  - **Success path (INCONSISTENT — needs isolation):** the minimal `!bool` `return true` yielded a
    zero word (reads false), but the vault's `deposit` (`!bool | E`) returned a nonzero word in the
    app context. So the success-return encoding is unreliable but not characterized to a single
    rule yet — treat narrow error-union success returns as untrusted, assert via side effects.
  Likely the narrow packing `(payload << 1) | tag` is not split into payload-return vs
  selector-revert at the dispatcher seam.
- **Repro (minimal):** `error E; pub fn f(b: bool) -> !bool | E { if (b) { return E; } return true; }`
  → `f(true)` reverts empty (selector lost); `f(false)` return-word encoding unreliable.
- **Affected coverage:** any narrow error-union fn — e.g. the vault's `deposit`/`withdraw`
  (`!bool | …`). In `app_vault_errors.spec.toml`: success calls use `succeeds = {}` (must not
  revert; bytes unchecked) with effects proven via balanceOf/getTotalDeposits; error calls pinned
  `reverts = {}` (empty).
- **Flip condition:** when narrow error-union dispatcher returns are fixed, change the vault revert
  rows to `{ selector = "0x1f2a2005" }` (ZeroAmount) / `{ selector = "0xcf479181" }`
  (InsufficientBalance), and tighten the success rows from `succeeds = {}` to `{ bool = true }`.

## F-004 — tuple return from inside try/catch fails OraToSIR Phase3b (fails closed)

- **Status:** OPEN
- **Severity:** S3
- **Owner:** compiler
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

## F-005 — trait-impl methods cannot be called from contract functions (fails closed)

- **Status:** OPEN
- **Severity:** S2
- **Owner:** compiler
- **What:** calling any trait-impl method from inside a `contract` fn fails lowering:
  `'func.call' op 'Pricing.Item.unit_price' does not reference a valid function` — the
  monomorphized impl function is not materialized for contract-context call sites. The
  `ora-example` trait corpus only calls trait methods from module-level helper fns, so this
  was never visible. Net: trait methods are currently NOT executable through the ABI at all.
- **Repro (minimal, 2026-06-12):**
  `trait P { fn unit_price(self) -> u256; }` + `struct Item { price: u256; }` +
  `impl P for Item { fn unit_price(self) -> u256 { return self.price; } }` +
  `contract C { pub fn f(p: u256) -> u256 { let i: Item = Item { price: p }; return i.unit_price(); } }`
  → exit 2, no bytecode.
- **Blocked coverage:** `traits.monomorphized_methods` manifest entry stays SKIP; the prepared
  spec (trait dispatch via `price_of`/`cost_of`) was removed from the corpus because it cannot
  compile.
- **Flip condition:** when contract-context trait calls land, restore
  `trait_method_dispatch.ora`/`.spec.toml` (content in this entry's git history, commit that
  added this section) and flip the manifest entry to covered.

## F-007 — conformance harness does not thread metered gas (blocks gas tests)

- **Status:** OPEN
- **Severity:** S3
- **Owner:** test-harness
- **What:** the conformance `evm.call` path returns `CallResult.gas_left ≈ DEFAULT_GAS` (the full
  budget), not the post-execution remaining — so `gas_used = DEFAULT_GAS - gas_left ≈ 0` for every
  call. A `gas_max` ceiling assertion was prototyped (2026-06-12) and BACKED OUT rather than ship a
  hollow check: with `gas_max = 1` a real getter still "passed". Gas testing (T4.1) is blocked until
  the harness threads true metered gas through `executeSpec` (the inner_create/call gas accounting
  needs to surface used-gas, possibly via the EVM's tracer or a gas-used field on the result).
- **Repro:** add `gas_max = 1` to any `[[call]]`; the call passes (gas_used reads as 0).
- **Flip condition:** wire real gas-used into the runner result, then re-add `gas_max` with
  teeth (a `gas_max = 1` must fail) and start ceilings on the full-app specs.
