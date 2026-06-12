# Conformance Findings Ledger

Execution-discovered compiler/EVM findings, one entry each. Corpus `.ora`/`.spec.toml` files
carry only `// see FINDINGS.md#<id>` pointers — never finding prose (test-quality program T1.6).
When a finding is fixed: flip its blocked/characterization rows as described, run
`zig build test-conformance`, and mark the entry FIXED with the commit.

---

## F-001 — `requires` not boundary-enforced; discharged checks erased (S1)

- **Status:** FIXED (runtime requires assertion emitted beside SMT precondition)
- **Severity:** S1
- **Owner:** compiler+smt-audit
- **What:** `requires` clauses on pub fns were verification-only assumptions — the dispatcher
  emitted no guard for them. Additionally, obligations the verifier discharged *using* those
  assumptions were removed from runtime code. Net: `pub fn div(a,b) requires b != 0` had no
  zero-check at all; an external `div(7,0)` executed EVM DIV and returned **silent 0**.
  Asymmetry: the checked-add overflow assert was retained (`add(MAX,1)` reverted correctly).
- **Repro:** `tests/conformance/arithmetic_checked_revert.ora` (also minimal:
  any pub fn with `requires b != 0` + `a / b`; IR shows `ora.requires` but no assert).
- **Fixed rows (`arithmetic_checked_revert.spec.toml`):**
  `add(2000000,1)`, `div(7,0)`, and `call_private_div(8,0)` now assert `reverts`.
  `ordered_requires(18,0)` pins source order: the first `requires b != 0` check runs
  before evaluating the later `a / b < 10` precondition.
- **Fix:** HIR emits `ora.requires` for SMT and a tagged executable `ora.assert` twin for
  runtime. Verification cleanup removes the pure `ora.requires` and keeps the executable
  assertion as `cf.assert`; `requires`-tagged `cf.assert` lowers to a clean revert.

## F-002 — catch-path unpacking of aggregate error-union payloads is miscompiled (S1-class)

- **Status:** FIXED (aggregate-success error-union Err payloads no longer dereference scalar error IDs)
- **Severity:** S1
- **Owner:** compiler
- **What:** `catch` after a `try` that actually returns an error, on an error union whose
  payload is an AGGREGATE (wide struct, string), dereferences a garbage payload pointer →
  huge-offset `mload`. Scalar payloads are fine (properties suite). Ok paths, plain `match`,
  and tuple-from-match all return correct values — only catch-on-actual-Err breaks; both
  payload kinds showed one root cause.
- **Repro/fixed rows:** `error_union_wide_carrier.ora` `err_default()` now returns 777;
  `error_union_local_aggregate.ora` `err_marker()` now returns 999.
- **Fix:** wide error-union return lowering now distinguishes real pointer-backed payloads
  from scalar error words temporarily relabelled as the aggregate success carrier. Scalar
  error IDs are carried as words and are not loaded as pointers before catch dispatch.

## F-003 — lib/evm panics (integer overflow) on huge-offset mload gas computation

- **Status:** OPEN (lib/evm robustness, surfaced by F-002)
- **Severity:** S2
- **Owner:** lib/evm
- **What:** `handlers_memory.zig:35` `GasConstants.GasFastestStep + mem_cost` overflows and
  PANICS the interpreter instead of failing cleanly (out-of-gas / error). The conformance
  oracle must never panic on hostile-but-valid bytecode behavior; the R9 Anvil differential
  would cross-check this class.
- **Repro:** run either F-002 blocked row. **DIFFERENTIAL-CONFIRMED 2026-06-12:**
  `bash scripts/prove-f003-differential.sh` runs the identical `err_default()` call through both
  EVMs — lib/evm PANICS (exit 134, `handlers_memory.zig:35` integer overflow in `mload`), Anvil/revm
  cleanly OOG-reverts (`EVM error MemoryOOG`). Repro at `tests/differential/f003_repro.{ora,spec.toml}`
  (kept out of `tests/conformance/` so it cannot crash the gate). This is the differential catching a
  defect the in-process oracle cannot even evaluate.
- **Flip condition:** none in specs; fix = saturating/checked gas math returning OutOfGas.
  F-002's rows stay blocked on F-002 regardless. When fixed, `prove-f003-differential.sh` exits 1
  (lib/evm no longer panics) — promote the repro to a real conformance spec then.

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

## F-008 — events emit no topic0 signature hash (not Solidity-ABI filterable)

- **Status:** OPEN
- **Severity:** S2
- **Owner:** compiler
- **What:** Ora `log` events emit with EMPTY topics and all fields packed into data — there is no
  `topic0 = keccak(eventSignature)` and no indexed-parameter topics. Execution-confirmed 2026-06-12:
  `log Recorded(who: address, amount: u256)` emits `topics=[]`, `data = abi(address,uint256)` (and
  the existing `Ping()` likewise emits `topics=[]`). Consequence: standard tooling (ethers/web3
  `getLogs` filtered by event signature or indexed args) cannot find or decode Ora contract events —
  a real interop limitation for any dapp/indexer consuming them.
- **Repro:** `tests/conformance/events_multifield.spec.toml` (asserts the current topics=[] layout).
- **Flip condition:** when Ora emits the standard `topic0 = keccak(signature)` (+ any indexed
  params as topics), update `events_multifield.spec.toml` (and `environment_log`) to the real topics.
