---
slug: ora-v0-2-proof-carrying-contracts
title: "Ora v0.2: Proof-Carrying Contracts"
authors: [axe]
tags: [release, ora, verification, adt, abi, debugger, lsp]
---

Ora v0.2 is the release where the pieces start to act like one system.

The headline features are big on their own: first-class `Result<T, E>` values,
a unified ADT model, Z3-backed verification reports, runtime ABI encoding,
dynamic public returns, traits, debugger improvements, LSP production work,
compiler metrics, CFG output, and a real MLIR optimization pipeline.

But the real v0.2 story is simpler: Ora is moving from "a compiler with
verification features" toward contracts that carry their proof obligations,
their runtime checks, their ABI shape, and their debugging evidence through one
pipeline.

<!-- truncate -->

---

## Naming: Asuka stays the release line

Asuka was v0.1 because v0.1 was the first coherent release milestone.
For v0.2, Ora keeps **Asuka** as the release track instead of minting a new
codename for every point release.

So the naming model is:

- **Asuka**: the release line
- **v0.1**: the first Asuka release
- **v0.2**: the proof-carrying-contracts release on the Asuka track

That keeps continuity without turning every compiler snapshot into a brand.

---

## Result values and error unions are real values now

Ora already had typed errors. v0.2 makes the success/error channel a first-class
value shape.

```ora
error Failure(code: u256);
error Denied(owner: address);

contract ResultDemo {
    pub fn choose(flag: bool, value: u256) -> Result<u256, Failure> {
        if (flag) {
            return Ok(value);
        }
        return Err(Failure(7));
    }

    pub fn inspect(value: Result<u256, Failure>) -> u256 {
        return match (value) {
            Ok(inner) => inner,
            Err(err) => err.code,
        };
    }
}
```

This is not just parser syntax. v0.2 carries `Result<T, E>` and error unions
through sema, HIR, Ora MLIR, SIR, the ABI dispatcher, comptime evaluation, and
the SMT encoder.

What works now:

- `Ok(...)` and `Err(...)` constructors
- `match` over success and error arms
- named and payloaded errors
- multi-error unions
- public ABI support for success and revert paths
- custom-error selector reverts
- storage/local carriers for scalar and dynamic success payloads
- comptime matching over constructed Result values
- SMT datatype encoding for enums, products, and error unions

That unlocks a style where failure is explicit, exhaustively matched, ABI
visible, and verifier-visible.

```ora
pub fn consume(value: Result<u256, Failure>, bump: u256) -> Result<u256, Failure>
    requires(bump <= 10)
{
    return match (value) {
        Ok(inner) => Ok(inner + bump),
        Err(err) => Err(Failure(err.code)),
    };
}
```

The important rule remains: Ora does not auto-wrap values into `Ok(...)` at
branch boundaries. If a function returns `!T | E`, the source should say which
path it returns. That is the no-hidden-behavior rule applied to errors.

---

## ADTs: products and sums share one model

v0.2 also unifies the ADT story.

Products are tuples, structs, and named aggregate payloads. Sums are enums,
error unions, and `Result<T, E>`. The compiler now treats these as one family
instead of separate one-off lowering paths.

```ora
struct Snapshot {
    owner: address,
    balance: u256,
}

enum Status {
    Empty,
    Ready(Snapshot),
    Closed,
}

contract AdtDemo {
    pub fn weight(status: Status) -> u256 {
        return match (status) {
            Empty => 0,
            Ready(snapshot) => snapshot.balance,
            Closed => 1,
        };
    }
}
```

The sema layer owns exhaustiveness. That matters. Codegen does not invent a
default arm to make an incomplete match lower. A missing variant is a compile
error, not bytecode with a hidden fallback.

The same model now drives:

- enum and error-union matching
- source ADT constructor lowering
- comptime constructor/match parity
- SMT datatype constructors and accessors
- ABI restrictions for shapes that do not have a Solidity-compatible encoding

That last point is important. Some types are perfectly good internal Ora values
but not valid public ABI values. v0.2 rejects those shapes instead of inventing a
private encoding at the public boundary.

---

## Verification: proof obligations, reports, and vacuity

v0.2 is the first release where the verifier report is something you can hand to
another engineer and expect them to reason from it.

The verifier is Z3-backed. It proves checked arithmetic, division safety,
assertions, postconditions, loop invariants, callee preconditions, storage-frame
facts, and the active refinement guard lexicon.

```ora
contract VerifiedVault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;

    pub fn deposit(amount: MinValue<u256, 1>)
        requires(totalDeposits <= std.constants.U256_MAX - amount)
        requires(balances[std.msg.sender()] <= std.constants.U256_MAX - amount)
        ensures(totalDeposits == old(totalDeposits) + amount)
        ensures(balances[std.msg.sender()] == old(balances[std.msg.sender()]) + amount)
    {
        let sender: NonZeroAddress = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
    }
}
```

`requires` constrains the verified body and is enforced at boundaries. `ensures`,
`assert`, loop invariants, checked arithmetic, division checks, and callee
preconditions become proof obligations. If Z3 cannot prove a required obligation,
the compiler does not emit a verified artifact.

Run:

```bash
ora build ora-example/corpus/patterns/verified_vault.ora --explain --emit=smt-report
```

The report includes:

- `verification.success`
- `verification.verification_trust`
- counterexamples for failed obligations
- degradation and soundness-loss labels
- query fragments and solver status
- explain-mode unsat cores
- optional raw Z3 proofs with `--z3-proofs`
- vacuity information

Vacuity matters. If a function's assumptions are contradictory, every obligation
can prove trivially because no valid execution exists. v0.2 reports that as
vacuous-risk instead of letting a meaningless proof look like a full success.

```ora
pub fn impossible(x: u256) -> u256
    requires(x > 10)
    requires(x < 5)
    ensures(result == 123)
{
    return 0;
}
```

The `ensures` is not trustworthy here. The assumptions are impossible. The
report tells you that.

---

## Requires, guards, asserts, and runtime checks

v0.2 also closes an important gap at the bytecode boundary. Preconditions are
not only verifier assumptions; public boundary checks are emitted as runtime
checks when needed.

The current model is:

- `requires`: assumption for verifying the body, enforced at public/call
  boundaries
- `guard`: runtime-enforced precondition shape, also verifier-visible
- `assert`: runtime check and SMT obligation
- `ensures` / `ensures_ok` / `ensures_err`: proof obligations, not automatic
  runtime postcondition checks in normal builds
- `--keep-proved-checks`: a falsification/audit option that keeps proved
  checks in runtime output instead of erasing them

This is why the release includes both SMT reports and bytecode conformance
tests. A proved property and an emitted check are different things. Ora now
makes that distinction explicit.

---

## Runtime ABI: encode, decode, dynamic returns

The ABI work in v0.2 is large because ABI bugs are not cosmetic. They are how a
contract lies to its caller.

The most important fix: source-level runtime `@abiEncode(...)` now goes through
the real runtime encoder. Before this hardening, a runtime encode could compile
to empty bytes. That is the worst possible failure mode for hash payloads.

Now this is a real runtime operation:

```ora
contract HashDemo {
    pub fn commitment(owner: address, amount: u256) -> bytes32 {
        return @keccak256(@abiEncode((owner, amount)));
    }
}
```

v0.2 also fixes dynamic public returns. Public functions returning `string`,
`bytes`, supported dynamic arrays, and supported aggregate layouts now emit the
canonical ABI head/tail shape: offset, length, padded payload.

```ora
contract DynamicReturn {
    pub fn name() -> string {
        return "ora";
    }

    pub fn values() -> slice[u256] {
        return [1, 2, 3];
    }
}
```

At the same time, unsupported public return shapes fail closed. If a type cannot
be represented honestly at the ABI boundary, the compiler rejects it instead of
shipping best-effort bytecode.

Other ABI fixes in v0.2:

- dynamic return ABI encoding
- strict external returndata decode
- runtime `@abiDecode` coverage for static and dynamic shapes
- dispatcher decode matrix coverage
- custom-error selector reverts
- narrow and wide error-union carrier fixes
- ABI layout model unification
- comptime ABI selector/signature/topic helpers

Comptime ABI builtins are now useful for interface code:

```ora
trait ERC20 {
    fn transfer(self, to: address, value: u256) -> bool;
}

contract AbiBuiltinDemo {
    pub fn transfer_selector() -> bytes4 {
        return comptime {
            @selector(ERC20.transfer);
        };
    }

    pub fn transfer_signature() -> string {
        return comptime {
            @abiSignature(ERC20.transfer);
        };
    }
}
```

---

## Comptime got broader

Ora's first design pillar is still comptime over runtime. v0.2 expands what the
compiler can evaluate before bytecode exists.

New and hardened comptime paths include:

- ABI selector/signature/event-topic helpers
- `@abiEncode` / `@abiDecode` over more static and dynamic shapes
- ADT and `Result` constructors
- `match` over comptime ADTs
- partial folding of arithmetic and ABI layouts
- bounded loop unrolling where bounds are known
- clearer fail-closed diagnostics when comptime cannot produce a value

Example:

```ora
contract ComptimeAbi {
    pub fn dynamic_array_payload_len() -> u256 {
        return comptime {
            const values = @cast(slice[u256], [1, 2, 3]);
            @abiEncode(values)[63];
        };
    }
}
```

The point is not cleverness. The point is to remove runtime work only when the
compiler can prove what it is doing.

---

## Traits and extern interfaces

v0.2 hardens the trait system and the external-call surface.

Internal traits are static-dispatch only. No runtime trait objects, no vtables,
no hidden dynamic dispatch.

```ora
trait BalanceView {
    fn balance_of(self, owner: address) -> u256;
}

impl BalanceView for Vault {
    fn balance_of(self, owner: address) -> u256 {
        return self.balances[owner];
    }
}
```

The compiler checks impl conformance: missing methods, extra methods, duplicate
visible impls, wrong signatures, and invalid trait method bodies are diagnostics.

For other contracts, `extern trait` declares ABI-compatible calls:

```ora
error InsufficientBalance(required: u256, available: u256);
error InvalidRecipient;

extern trait ERC20 {
    call fn transfer(self, to: address, amount: u256) -> bool
        errors(InsufficientBalance, InvalidRecipient);

    staticcall fn balanceOf(self, owner: address) -> u256;
}
```

v0.2 validates call/staticcall rules, gas/call shapes, selector computation,
declared error sets, trusted extern summaries, and ABI layouts for return data.
The compiler no longer treats extern calls as an untyped hole in the language.

---

## Debugger and LSP: the tools caught up

The debugger stack is now a real part of the compiler experience.

```bash
ora debug contracts/vault.ora \
  --signature 'deposit(u256)' \
  --arg 100
```

v0.2 includes source maps, debug info, TUI work, DAP server support, stack and
storage views, ABI revert decoding, watchpoints, gas and coverage overlays, and
debugger artifacts that preserve the source-to-SIR-to-EVM trail.

That matters because Ora is not trying to hide the compiler pipeline. You should
be able to inspect what the compiler did.

The LSP also moved from a syntax helper toward production tooling:

- cached document analysis
- hover
- completion
- definition
- references
- rename
- document/workspace symbols
- semantic tokens
- inlay hints
- code lenses
- call hierarchy
- cache stats
- benchmark gates

The editor now understands the v0.2 language surface: traits, impls, ADTs,
matches, errors, verification clauses, aliases, imports, and builtin docs.

---

## Metrics and CFG tooling

v0.2 adds compiler metrics so optimization work stops being guesswork.

```bash
ora build ora-example/corpus/patterns/verified_vault.ora --metrics
```

The report breaks frontend/HIR compilation into named phases:

- syntax
- ast-lower
- module-graph
- item-index
- resolve
- typecheck
- const-eval
- verify-facts
- hir-lower

It records timing, invocation counts, work counts, allocation calls, and
allocated bytes. The corpus harness mirrors the conformance metrics style:

```bash
zig build compile-metrics
scripts/compile-metrics-check.py --check
```

CFG tooling is the other half of inspectability:

```bash
ora emit --emit=cfg:sir ora-example/counter.ora
ora emit --emit=cfg:sir-diff ora-example/counter.ora
```

`cfg:sir` emits a block-level SIR control-flow graph. `cfg:sir-diff` is a debug
aid for inspecting structural changes before and after SIR optimization. It is a
graph/debugging tool, not a semantic equivalence proof.

---

## MLIR optimization: use the framework

v0.2 starts moving deterministic cleanup into MLIR itself: folders,
canonicalizers, CSE, and pass managers. That is the direction Ora should keep
taking.

The old shortcut was hand-rolled cleanup in lowering code. That got the compiler
moving, but it duplicated infrastructure MLIR already has. v0.2 begins the shift
back to the right architecture:

- SIR op folders and canonicalizers
- Ora op canonicalizers
- deterministic SIR cleanup after conversion
- Ora function canonicalization/CSE before lowering
- targeted pass-manager use
- tests pinning which folds are allowed

The rule is pragmatic: use MLIR where MLIR can own the transformation cleanly.
Keep hand-rolled code only where Ora semantics genuinely require it.

---

## Hardening: fail closed, no executable defaults

v0.2 includes a lot of work that users should never notice directly. That is the
point.

The compiler used to have too many places where an internal failure became a
plausible value: `0`, empty bytes, an unknown integer width, a placeholder
operand, a blind bitcast, or a default ABI shape.

Those are not recoveries. In a smart-contract compiler, wrong code must not
become bytecode.

The v0.2 hardening pass fixes or gates many of those paths:

- unresolved type annotations now diagnose
- unknown error names diagnose
- integer width and signedness gaps fail closed
- unsupported builtin paths fail closed
- runtime `@abiEncode` no longer returns empty bytes
- unsupported public ABI shapes reject
- dispatcher error-union failures return custom-error selectors
- narrow/wide error-union carriers preserve payload bits
- typed bitfield literals preserve carrier intent
- wide integer literals no longer truncate through host integer types
- imported-module diagnostics gate artifact emission
- destructive `-o` behavior was removed
- formatter/build mode argument confusion was hardened
- checked constant arithmetic remains a revert path after canonicalization
- verifier degradation, precision context, cap markers, vacuity, and UNKNOWN are
  visible in reports

This is not the glamorous part of a release, but it is the part that keeps a
compiler honest.

---

## Test gates got serious

v0.2 was not only feature work. The test strategy changed.

The release is guarded by:

- lib/evm conformance specs
- multi-contract conformance execution
- selected Anvil differential checks
- SIR text checks
- Ora MLIR checks
- SIR MLIR checks
- dispatcher ABI matrix checks
- OraToSIR de-bloat coverage manifest
- ABI roundtrip properties
- arithmetic reference properties
- storage load/store properties
- error-union pack/unpack properties
- bytecode equivalence smokes
- verifier mutation tests
- negative diagnostic corpus
- compile metrics snapshot checks
- CFG structural tests
- LSP smoke and benchmark gates

The big lesson from this cycle: a golden can prove text did not change, but it
cannot prove the text was right. v0.2 uses goldens, execution, property tests,
and independent EVM checks together because each catches a different class of
failure.

---

## Try v0.2

Build the compiler:

```bash
git clone https://github.com/oralang/Ora.git
cd Ora
zig build
```

Run a verified contract:

```bash
./zig-out/bin/ora build ora-example/corpus/patterns/verified_vault.ora --explain --emit=smt-report
```

Inspect lowering:

```bash
./zig-out/bin/ora emit --emit=mlir:sir,sir-text ora-example/counter.ora
```

Render a CFG:

```bash
./zig-out/bin/ora emit --emit=cfg:sir ora-example/counter.ora
```

Debug a call:

```bash
./zig-out/bin/ora debug ora-example/counter.ora --signature 'get()'
```

Measure compiler phases:

```bash
./zig-out/bin/ora build ora-example/corpus/patterns/verified_vault.ora --metrics
```

---

That is v0.2: proof-carrying contracts on the Asuka track.
