# User-Declared `modifies` Clauses

This document defines the v1 design direction for user-declared `modifies`
clauses.

Status: **v1 landed with documented deferrals**. Parser and AST accept
`modifies` as a function spec clause. Sema accepts the supported v1
current-contract storage path shapes, rejects unsupported forms fail-closed,
and checks compiler-derived current-contract storage writes against declared
paths, including `modifies()` as the empty declared set. Verifier framing is
landed for known internal callees: exact internal summaries preserve distinct
map keys and struct fields, while opaque metadata fallback preserves indexed
map paths and struct-field paths precisely and falls back to root-slot
precision for root-slot declarations.
External-call framing is landed for `staticcall`, for the existing trusted
extern caller-storage frame, and for runtime-locked root/single-key storage
paths across ordinary `call`.

The implemented v1 feature intentionally models only the current contract's own
storage. It does not attempt to describe or verify storage owned by external
contracts.

## 1. Objective

`modifies` is an optional verification clause over the current contract's
storage.

It is not a mutability permission. Users do not need `modifies` for ordinary
state-changing code to compile.

Solidity has no direct equivalent. Solidity's `view` and `pure` annotations
are coarse mutability declarations for functions that should not write state;
they do not let a state-changing function declare "I may write exactly these
slots and no others." `modifies` fills that verification gap.

This is valid Ora without `modifies`:

```ora
pub fn setBalance(user: address, value: u256) {
    balances[user] = value;
}
```

Adding `modifies` makes a proof promise:

```ora
pub fn setBalance(user: address, value: u256)
    modifies balances[user]
{
    balances[user] = value;
}
```

The verifier must then prove that the function's own writes are a subset of
the declared set. Callers may use that frame to preserve facts about storage
paths outside the set.

The core v1 rule:

> If `modifies` is absent, the function makes no storage-frame promise. If
> `modifies` is present, the compiler/verifier checks actual current-contract
> writes against the declared paths and fails closed on mismatch.

## 2. Definitions

### 2.1 Current-contract storage

Current-contract storage means storage slots owned by the contract currently
being verified.

Examples:

```ora
storage total_supply: u256;
storage balances: map<address, u256>;
```

Valid v1 `modifies` clauses name these paths:

```ora
modifies total_supply
modifies balances[user]
modifies buckets[42]
modifies balances[msg.sender]
modifies balances[tx.origin]
modifies allowances[owner][spender]
```

### 2.2 External contract storage

External contract storage is not in v1 scope.

The verifier does not try to prove which storage paths an external contract
changes internally. That storage belongs to another contract. It is outside
the current contract's storage frame.

No v1 syntax such as `callee_storage[...]` or `external_storage[...]` is
accepted.

### 2.3 Compiler-derived writes

The compiler already knows where the current contract's code writes storage.
The implementation should route those compiler-derived write paths through a
single representation, conceptually:

```text
actual_writes(function) <= declared_modifies(function)
```

If the subset check cannot be performed soundly, compilation/verification must
fail closed.

## 3. Absent `modifies`

Absent `modifies` means:

> This function declares no storage-frame promise.

It does not mean "writes no storage." It does not mean "verification may assume
all storage is preserved." It means the verifier cannot rely on a
user-declared frame for this function.

For compatibility, v1 uses the permissive default:

```ora
pub fn deposit(amount: u256) {
    balances[msg.sender] += amount;
}
```

This remains legal.

Verifier behavior:

- The compiler may still infer writes from the body.
- The verifier may use internal compiler-derived facts where already sound.
- The verifier must not invent a precise user-declared frame when none exists.
- Any proof that needs a declared frame must require `modifies` or fail closed.

Future work may add an opt-in strict mode, for example `@strict_modifies`,
where storage-writing functions must declare a frame.

## 4. Non-Empty `modifies`

A non-empty `modifies` clause declares the only current-contract storage paths
the function is allowed to write.

```ora
pub fn transfer(to: address, amount: u256)
    modifies balances[msg.sender], balances[to]
{
    balances[msg.sender] -= amount;
    balances[to] += amount;
}
```

The verifier obligation is:

```text
all actual current-contract writes are in the declared modifies set
```

If the body writes outside the set, fail closed:

```ora
pub fn bad(to: address, amount: u256)
    modifies balances[to]
{
    balances[msg.sender] -= amount; // reject: not declared
    balances[to] += amount;
}
```

The declaration is a proof contract. It is not a runtime permission switch.

### 4.1 Internal callees and migration

The declared set applies to the full effect of the function body, including
storage writes performed by internal helper calls reached from that body.

This avoids a migration cliff. A user may start by annotating one public
function without annotating every helper it calls:

```ora
pub fn transfer(to: address, amount: u256)
    modifies balances[msg.sender], balances[to], nonces[msg.sender]
{
    validateAmount(amount);
    bumpNonce(msg.sender);
    moveBalance(msg.sender, to, amount);
}
```

The verifier checks the transitive compiler-derived write set of `transfer`.
If `bumpNonce` writes `nonces[msg.sender]`, that write counts against
`transfer`'s declared frame even if `bumpNonce` has no `modifies` clause of
its own.

Helper functions may still declare their own `modifies` clauses. When they do,
those clauses are checked independently. For v1, the caller-side subset check
uses **trace-through** semantics: it walks the transitive compiler-derived
write set rather than relying on helper summaries. Summary-based checking can
be added later if performance or separate-compilation needs justify it. v1
must not require every internal helper to be annotated before a caller can use
`modifies`.

The verifier may consume a helper's sema-checked `modifies` declaration as an
internal-call frame summary. This is distinct from trusted extern summaries:
internal helper `modifies` is checked against the helper body before it is
lowered for framing, while extern summary facts are trusted at an explicit
boundary.

Source-level writes count for the modifies obligation even if a later runtime
revert would roll them back. `modifies` constrains what the source path may
attempt to write, not only what survives at the end of EVM execution.

## 5. Empty `modifies()`

`modifies()` is the explicit no-current-contract-storage-effects declaration.

Implementation status: **landed for sema-side write-set checking**. The
compiler treats `modifies()` as the empty declared set and rejects any
compiler-derived current-contract storage write in the function body.

It is different from absent `modifies`:

| Form | Meaning |
|---|---|
| no `modifies` | no user-declared frame promise |
| `modifies()` | explicit promise: this function writes no current-contract storage |

This should pass:

```ora
pub fn readBalance(user: address) -> u256
    modifies()
{
    return balances[user];
}
```

This should fail:

```ora
pub fn bad(user: address, value: u256)
    modifies()
{
    balances[user] = value; // reject
}
```

`modifies()` must not be combined with non-empty modifies clauses.

`modifies()` covers direct writes in the function's own body and transitive
internal callees. It does not by itself prove that an unprotected external
`call` cannot trigger reentrant writes. Such a function may compile if it has
no direct current-contract writes, but proofs that depend on caller-storage
preservation across the call must still satisfy the external-call framing rules
in section 6.

## 6. External Calls

### 6.1 Physical EVM rule

An ordinary external `CALL` cannot directly write the caller contract's
storage. `STATICCALL` also cannot mutate state.

So for a call like:

```ora
let ok = external<Token>(token).transfer(to, amount);
```

the external token contract cannot directly perform `SSTORE` in the current
contract's storage.

### 6.2 Return values are separate from external effects

If the current contract stores an external call's return value, that storage
write is our own write and must be covered by our function's `modifies` set.

```ora
pub fn pull(token: address, to: address, amount: u256)
    modifies last_transfer_ok
{
    let ok = external<Token>(token).transfer(to, amount);
    last_transfer_ok = ok; // current-contract write
}
```

The external call did not write `last_transfer_ok`. Our code did.

### 6.3 Reentrancy is the real risk

Although external code cannot directly write our storage, it may call back into
our contract before returning. Through that callback, our own code may write
our storage.

Therefore the verifier must distinguish:

- Direct external write to our storage: impossible for ordinary `CALL`.
- Reentrant write through our own code: possible unless prevented.

For v1:

- `staticcall`: current-contract storage is fully framed. **Implemented.**
- `call` with relevant slots locked: locked root/single-key storage paths are
  framed. **Implemented for supported runtime lock paths.**
- `call` without proven reentrancy protection: unlocked current-contract
  storage cannot be assumed preserved unless another sound summary applies.

The verifier must not use `modifies` to pretend unknown external bytecode is
verified. `modifies` only constrains the current contract's own writes.

## 7. `@lock` and `@unlock`

`@lock` and `@unlock` are runtime write gates, not only SMT hints. The sibling
soundness model is tracked in `docs/formal-specs/user-locks.md`.

The implemented runtime lowering stores lock state in transient storage and
inserts transient-storage guards before writes to guarded roots. Current
runtime lock paths are intentionally narrow: root current-contract storage
paths and single-key indexed current-contract storage roots. Struct-field
locks, nested indexed locks, and user transient-variable locks fail closed
until the lowering can represent them soundly.

The implemented verifier rule is:

```text
locked slot across external CALL => preserved slot
```

This rule is load-bearing. The encoder only applies it for lock operations
accepted by sema and backed by the runtime evidence in
`docs/formal-specs/user-locks.md`: root current-contract storage and
single-key indexed current-contract storage roots. If the compiler cannot
prove that the relevant slot is locked across the call, it must not use the
lock framing rule.

## 8. Supported v1 Syntax

Initial supported forms:

```ora
modifies total_supply
modifies balances[user]
modifies balances[msg.sender]
modifies balances[tx.origin]
modifies balances[42]
modifies allowances[owner][spender]
modifies balances[user], total_supply
```

Supported path classes:

- Root storage scalar: `total_supply`
- Root storage struct field, if the compiler can map it exactly:
  `config.owner`
- Mapping slot with a simple key: function parameter, literal,
  `msg.sender`, or `tx.origin`:
  `balances[user]`, `balances[42]`, `balances[msg.sender]`,
  `balances[tx.origin]`
- Nested mapping slot with simple keys:
  `allowances[owner][spender]`

Unsupported in v1:

```ora
modifies *
modifies balances[*]
modifies balances[users[i]]
modifies users[user].balance
modifies external_storage[...]
modifies callee_storage[...]
modifies caller_storage[...]
```

Unsupported forms fail closed with targeted diagnostics. They must not degrade
silently.

## 9. Symbolic Map Slots

Most useful storage in contracts is map-backed, so v1 must support simple map
slots.

Examples:

```ora
modifies balances[user]
modifies allowances[owner][spender]
modifies balances[msg.sender]
modifies balances[tx.origin]
modifies buckets[42]
```

The subset check compares compiler-derived slot expressions against declared
slot expressions.

For v1, keep matching structural:

- `balances[user]` matches writes to `balances[user]`.
- `balances[msg.sender]` matches writes to `balances[msg.sender]`.
- `balances[tx.origin]` matches writes to `balances[tx.origin]`.
- `buckets[42]` matches writes to `buckets[42]`.
- `allowances[owner][spender]` matches writes to
  `allowances[owner][spender]`.
- `balances[a]` and `balances[b]` are different unless they are syntactically
  the same source binding.

Do not attempt SMT equality reasoning for map keys in v1. For example, do not
try to prove `a == b` to merge `balances[a]` and `balances[b]`. That can be a
later precision improvement.

Loop-derived or aggregate-derived keys such as `balances[users[i]]` are not in
v1. They fail closed as unsupported modifies path expressions.

## 10. Trusted Extern Summaries

v1 `modifies` does not model external contract storage.

Trusted extern summaries may still describe return values and preconditions:

```ora
extern trait Token {
    call fn transfer(self, to: address, amount: u256) -> bool
        ensures result == true;
}
```

But v1 should not accept trusted extern storage effects such as:

```ora
modifies callee_storage[balances[to]]
```

Reason: external storage is outside the current contract's proof surface. If
we later want to verify multi-contract state effects, that needs a separate
design with explicit contract identities and trust boundaries.

The existing round-55 implicit caller-storage framing for trusted `call fn`
summaries with clauses is retained for v1. In the current encoder this is
represented by the `ora.trusted_extern_frame = "caller_storage"` attribute and
recognized by `isTrustedExternCallerStorageFrame`. Treat it as an implicit
trusted no-caller-storage-effects declaration for that extern call summary.

v1 does not accept explicit `modifies caller_storage[...]` syntax. The implicit
round-55 form is a compatibility bridge for trusted summaries that already
exist; explicit caller-storage effect declarations require a later design.

For current-contract storage, the relevant external-call question remains
reentrancy. That is handled by `@lock` and conservative framing, not by
pretending the external contract's storage is ours.

## 11. Fail-Closed Rules

Compilation/verification must fail closed when:

- A function with `modifies` writes outside the declared set.
- A function with `modifies()` writes any current-contract storage.
- `modifies()` is combined with non-empty `modifies`.
- A modifies path cannot be resolved to current-contract storage.
- A modifies path uses unsupported syntax.
- A proof depends on a storage frame that the verifier cannot soundly provide.
- An unsupported external-storage modifies form is used.

Fail-closed means: stop compilation or verification for that function with a
targeted diagnostic. Do not silently ignore the clause. Do not emit a weakened
proof without a soundness-loss record.

These are semantic categories, not necessarily seven separate implementation
call sites. For example, `modifies()` with a write is the empty-set instance of
"writes outside declared set", and unsupported external-storage syntax is an
unsupported-path diagnostic.

## 12. Implementation Plan

1. Define a neutral storage-effect path representation.
2. Lower compiler-derived current-contract writes into that representation.
   Recommended layer: HIR-time write-set collection, matching the existing
   effect-summary/`ora.write_slots` direction.
3. Semantically validate v1 `modifies` path expressions: resolve identifiers,
   check against the supported path classes in section 8, and lower each
   declaration into the neutral representation. Fail closed for unsupported
   forms.
4. Implement the subset check. **Landed for explicit v1 paths.**

   ```text
   actual_writes(function) <= declared_modifies(function)
   ```

   Recommended layer: post-HIR analysis over the collected write set.

5. Implement `modifies()` as the empty declared set. **Landed for sema-side
   write-set checking.**
6. Add verifier framing only after the subset check is in place. **Landed for
   known internal callees.** Exact internal summaries preserve distinct map
   keys and struct fields. Opaque metadata fallback preserves indexed map paths
   precisely, including nested maps, and preserves struct fields precisely.
   Root-slot declarations remain root-slot precise.
7. Add external-call framing:
   - `staticcall` preserves current-contract storage. **Implemented.**
   - trusted extern caller-storage frame preserves current-contract storage.
     **Implemented for the existing round-55 implicit frame.**
   - locked slots are preserved across `call`. **Implemented for supported
     runtime lock paths.**
   - unsafe unlocked storage is not assumed preserved. **Implemented
     conservatively through soundness-loss reporting.**
8. Add regression tests for every fail-closed rule in section 11. **Landed for
   the v1 semantic matrix:** writes outside the declared set, `modifies()`
   with writes, `modifies()` mixed with non-empty clauses, unsupported map-key
   syntax, unsupported external-storage names, and unresolved-call framing
   soundness loss. Regression coverage also pins internal-callee preservation
   for distinct map keys and struct fields, plus locked-call preservation for
   root storage and single-key map slots with negative controls for unlocked
   storage. Parser-level unsupported forms such as `modifies *` remain
   rejected before sema.

## 13. Non-Goals for v1

The following are explicitly deferred:

- External contract storage effects.
- `callee_storage[...]`, `caller_storage[...]`, or `external_storage[...]`
  syntax.
- Wildcards such as `modifies *` or `modifies balances[*]`.
- SMT equality reasoning for map-key aliases.
- Full multi-contract state-effect verification.
- Delegatecall. Ora currently does not model `DELEGATECALL`; if it is added
  later, this document must be revisited because delegatecall executes in the
  caller's storage context.

## 14. Precision Notes

Known internal callees are summarized by their body, so the verifier can
preserve paths that are not actually written. A helper declared as
`modifies balances[a]` and implemented by writing only `balances[a]` can support
a caller proof that `balances[b]` is preserved when the caller proves `a != b`.
Map-key paths require that kind of disjointness fact because symbolic keys can
alias. Struct fields are syntactically distinct, so preserving `config.admin`
across a helper that writes only `config.owner` needs no key-disjointness
precondition.

Opaque metadata framing consumes the same serialized `ora.modifies_slots` path
strings used by sema and HIR. For indexed map paths such as
`balances[param#0]` and nested map paths such as
`allowances[param#0][param#1]`, the fallback encoder models the call as an
opaque leaf-value update at the declared key path. That lets array theory prove
disjoint keys are preserved without quantifying over the whole map.

For struct-field paths such as `config.owner`, the fallback encoder models the
call as an opaque field-value update and reconstructs the product value from
the untouched pre-state fields. Root-slot declarations remain root-slot precise
in opaque fallback.

There are three reportable outcomes when opaque metadata is used:

1. Path-precise framing succeeds. The encoding is sound and precise for the
   declared indexed or struct-field path, and no precision note is emitted.
2. Path-precise framing cannot be applied, but metadata still covers the root.
   The encoding remains sound, falls back to a coarser root-slot opaque summary,
   and emits a precision note in the SMT report.
3. Metadata does not cover the written root. The verifier records a soundness
   loss and fails closed.

Precision notes fire for non-root path metadata when path-precise fallback
cannot be applied, including indexed paths, struct-field paths, and unsupported
mixed indexed-field shapes such as `users[param#0].balance` if such metadata is
encountered at a summary boundary. Source-level v1 `modifies` declarations for
mixed indexed-field paths fail closed in sema.

Precision notes are intentionally weaker than soundness losses: they explain
why a proof may lose precision without making the report trustedness fail by
themselves. Precision notes are typed report entries; if too many notes are
recorded, the final visible entry is `precision_note_cap_exceeded` so truncation
is explicit rather than silent. After overflow, the report contains up to nine
specific precision notes plus the cap marker.

Soundness losses use the same public truncation convention:
`soundness_loss_cap_exceeded` means the report omitted additional specific
soundness-loss entries after reaching the bounded report size. Cap markers are
truncation notices, not independent proof failures beyond the omitted entries
they summarize.

The opaque metadata path mainly serves future separate-compilation boundaries
and verifier runs that deliberately cap internal summary inlining. Current
regression coverage exercises both the opaque indexed-map encoder directly and
a known-callee `func.call` integration path with exact body summarization
disabled through `VerificationPass.setMaxSummaryInlineDepth(0)`. Production
runs can use `ORA_VERIFY_MAX_SUMMARY_INLINE_DEPTH=0` to force the same
summary-only behavior. A full separate-compilation integration test should be
added when separate-compilation makes the fallback naturally reachable without a
global summary-depth cap.

## 15. Summary

The v1 design is intentionally narrow:

> `modifies` is an optional proof clause over the current contract's own
> storage. If absent, it means no storage-frame promise. If present, actual
> compiler-derived writes must be a subset of the declared paths. External
> contracts cannot directly write our storage, but they can reenter; `@lock`
> is the runtime mechanism that lets the verifier preserve supported locked
> slots across external calls.

This scope is enough to make `modifies` useful for real verification without
pretending Ora can know arbitrary external contract behavior.
