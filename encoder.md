# ABI Encoder — Staged Implementation Plan

**Status:** design — pre-implementation. Re-review before Milestone 1 begins.

## TL;DR

One canonical ABI layout model in Zig, serialized to MLIR attributes, consumed by both a comptime materializer (`@abiEncode`) and the runtime materializer (`ora.abi_encode`). Six milestones: M1 lands the shared model (no encoding), M2 ships static comptime encoding, M3 unifies the runtime path on the same model, M4–M6 add `string`/`bytes`, dynamic arrays, and arbitrary nesting. Cross-layer parity tests anchored to `cast abi-encode` output prove correctness.

## Contents

- [Goal](#goal)
- [Why staged](#why-staged)
- [Architectural decisions](#architectural-decisions)
- [Architectural commitment](#architectural-commitment)
- [Type coverage](#type-coverage)
- [Offset semantics](#offset-semantics)
- [The milestones](#the-milestones)
- [Test strategy](#test-strategy)
- [Risk register](#risk-register)
- [Open questions](#open-questions)
- [Things deliberately out of scope](#things-deliberately-out-of-scope)
- [Sequencing constraints](#sequencing-constraints)
- [Honest framing](#honest-framing)

---

## Goal

A single canonical layout model for ABI encoding, consumed by both the
comptime evaluator (`@abiEncode(value)`) and the MLIR runtime lowering
(`ora.abi_encode`). Full coverage of the EVM ABI spec, shipped in
stages, with a stable architectural foundation laid before any
dynamic-type work.

## Why staged

Full ABI spec coverage in one landing is the wrong shape. Layout
knowledge is duplicated across three existing sites today —
`hir/abi.zig`, `module_lowering.zig`, and the C++ pass in
`SIRDispatcher.cpp` (see [Architectural commitment](#architectural-commitment)
for the inventory). A single-PR "full encoder" would either:

- **Add a fourth source of truth.** Bypass the existing sites and bolt
  the new encoder on alongside them. Cheap to land, expensive to live
  with.
- **Refactor all four at once.** Migrate the three existing sites and
  ship the new encoder in the same change, with no stable intermediate
  state to verify against.

Both are high-risk. Staging lets each milestone produce a stable,
verifiable system before the next layer of complexity lands.

---

## Architectural decisions

Three decisions that must be made up front, because they shape every
subsequent milestone:

### Decision 1: Cross-language sharing strategy

The runtime ABI encoder lives in C++ (`src/mlir/ora/lowering/OraToSIR/patterns/ControlFlow.cpp:3164`).
The comptime path is in Zig. A Zig-side encoder cannot be called directly
from the C++ pass.

Options considered:

| Option | Outcome |
|---|---|
| Duplicate the planner in C++ | Risk of divergence — rejected. |
| Serialize layout into MLIR attributes; C++ pass reads them | Self-describing IR. Single source of truth in Zig. **Chosen.** |
| C ABI bridge from C++ to Zig | Adds linking complexity, lifetime concerns. Considered but overkill. |
| Move lowering to Zig | Out of scope for the encoder work alone. |

**Decision:** Zig produces a canonical layout description; the layout
is serialized as MLIR attributes the C++ pass interprets.

The boundary is **about decisions, not about code**:

- **Zig owns type-to-layout.** All decisions about what the encoded
  shape looks like (static vs. dynamic, head/tail positions, ABI type
  selection per Ora type, padding rules) are made in Zig.
- **C++ owns runtime materialization of that layout.** The runtime
  encoder in `OraToSIR/patterns/ControlFlow.cpp` emits MLIR/SIR ops
  for padding, sign extension, dynamic offsets, lengths, loops,
  memory writes, and calldata allocation. That is real, non-trivial
  encoding work — it doesn't go away.
- **C++ must NOT make independent type/layout decisions.** Anything
  about "what to encode and how to lay it out" comes from the layout
  attribute. C++ only decides "how to emit the ops that materialize
  this layout at runtime." If a question can be answered by looking
  at types alone (without runtime values), Zig answered it before
  C++ ever sees the layout.

What's shared is **the layout model and the decisions it carries**,
not the materialization code. Comptime materializer emits bytes;
runtime materializer (C++) emits MLIR ops. Both consume the same
layout, both produce identical byte sequences for the same value.

### Decision 2: Split payload from selector-prefixed calldata

The current `ora.abi_encode` op is selector-coupled — it produces
`selector ++ payload`. This conflates two different things with
**different offset frames** (see the Offset semantics section below):
plain ABI encoding has offsets relative to byte 0 of the encoding;
selector-prefixed calldata is the same encoding placed at byte 4 of a
buffer whose first 4 bytes hold the selector.

**Decision:** explicit two-op model.

| Op / Builtin | Output | Consumer |
|---|---|---|
| `@abiEncode(value)` (user-facing) | plain ABI payload bytes | comptime user code |
| `ora.abi_encode` (internal) | plain ABI payload bytes | runtime |
| `ora.abi_encode_with_selector` (internal) | selector ++ payload | external call lowering |

User-facing `@abiEncode` returns clean payload bytes with self-relative
offsets. Selector concatenation is a separate concern: the bytes
between the selector and end-of-encoding are identical to what the
plain op produces — the selector variant just prepends 4 bytes. No
offset shifting happens at encode time. Receiver-side decoding handles
the `+4` shift via codegen-baked constants in the access pattern, not
via a runtime base register.

### Decision 3: Tree-shaped layout, not flat plan

A flat list of `(offset, value_idx, encoding_kind)` tuples works for
static-only encoding but cannot express runtime-determined offsets
needed for `string[]` and other dynamic-bearing aggregates. Element
count, tail size, and offsets within the tail are all runtime values
for dynamic arrays.

**Decision:** layout is a recursive tree.

```zig
pub const LayoutNode = union(enum) {
    static_word: struct { path: ValuePath, kind: StaticEncoding },
    dynamic_bytes: struct { path: ValuePath },           // string, bytes
    dynamic_array: struct { path: ValuePath, element: *const LayoutNode },
    fixed_array: struct { path: ValuePath, element: *const LayoutNode, length: u64 },
    tuple: []const LayoutNode,
};

pub const StaticEncoding = enum {
    uint,         // padded left, big-endian
    int,          // sign-extended, two's complement
    bool,         // 32 bytes, last byte 0/1
    address,      // 20 bytes left-padded to 32
    fixed_bytes,  // N bytes right-padded to 32
};
```

`ValuePath` identifies where in the input the value lives. It must
support both **concrete** and **loop-relative** forms because dynamic
arrays at runtime don't have a known length at layout time:

```zig
pub const ValuePathSegment = union(enum) {
    tuple_index: u32,         // concrete position in a tuple
    struct_field: []const u8, // concrete field name in a struct
    fixed_index: u32,         // concrete index in a fixed-size array
    each_element,             // loop-relative: "the current element"
                              // resolved by the materializer's enclosing loop
};

pub const ValuePath = []const ValuePathSegment;
```

The `each_element` marker is what makes `string[]` and `uint256[][]`
expressible. A `dynamic_array(path, element)` node walks the path to
reach the array, then iterates over its elements. The element's
LayoutNode uses `each_element` paths to refer to "the current element
of whatever loop is currently executing."

- **Comptime:** the materializer holds a loop variable; resolving an
  `each_element` segment returns the current iteration's value.
- **Runtime:** the materializer emits MLIR loop ops; `each_element`
  becomes a load from the loop's induction variable / element pointer.

Without this, the layout for `string[]` couldn't express "every
element is a string with the same shape" — it would need a separate
LayoutNode per array element, which is impossible when array length is
runtime.

---

## Architectural commitment

After Milestone 1, **the canonical layout model is the only thing both
layers consume**. Three existing sources of layout knowledge today get
unified:

- `src/hir/abi.zig` — canonical ABI string helpers, static word counting
- `src/hir/module_lowering.zig` — module-level layout decisions
- `src/mlir/ora/lowering/OraToSIR/SIRDispatcher.cpp:120` — C++ AbiType / AbiLayout parser

Pick one as canonical (or introduce a new top-level module) and migrate
the others to feed off it. **Do not introduce a fourth source of truth.**

M1 is not complete until all of the following are true:

- The convergence tests in `src/abi.test.zig` for context-bound module
  lowering layouts and static word counts no longer pin known
  divergences; they assert equality between `module_lowering` and
  `abi_layout`.
- The ABI manifest Result-carrier projection consumes
  `LayoutContext.planResultCarrier`; the manifest convergence test in
  `src/abi.test.zig` asserts emitted `AbiTypeNode` tuple shapes and
  component wire types match the same carrier plan.
- `zig build check-abi-layout-ownership` passes, proving no
  layout-bearing `StaticEncoding` / `LayoutNode` code exists outside
  `src/abi/layout*.zig` (tests excluded).
- `zig build test` passes with the ownership check wired into the
  normal test gate.

M2 is not complete until all of the following are true:

- The static `@abiEncode` tests in `src/compiler.test.abi.zig` pass,
  with cast-anchored byte expectations for primitives, signed integers,
  static aggregates, edge cases, refinement erasure, and every supported
  rejection diagnostic.
- The static corpus test at
  `ora-example/corpus/comptime/abi_encode_static.ora` compiles cleanly
  with no diagnostics, and every const-evaluated observation matches the
  checked expectation.
- The comptime materializer lives in
  `src/abi/comptime_encoder.zig`, consumes `LayoutContext` output, and
  does not add another layout decision path.
- `zig build check-abi-layout-ownership` passes with
  `src/abi/comptime_encoder.zig` whitelisted as the static comptime
  materializer.
- `zig build test` passes.

M3 is split into two stages so plumbing and validation cannot be
conflated. Full M3 is complete only when both M3a and M3b are complete.

M3a (runtime plumbing) is not complete until all of the following are
true:

- `src/abi/runtime_encoder.zig` is the Zig-side runtime materializer
  entry point. It consumes `LayoutContext` output and serialized
  `LayoutNode` attributes; it does not add another layout decision path.
- `ora.abi_encode` is the plain-payload runtime op and
  `ora.abi_encode_with_selector` is the selector-prefixed external-call
  op. External call lowering uses the selector variant; plain runtime
  ABI payload consumers use `ora.abi_encode`.
- The C++ lowering at
  `src/mlir/ora/lowering/OraToSIR/patterns/ControlFlow.cpp` parses the
  serialized layout attribute for static layouts. No static runtime
  path may size calldata from only `4 + 32 * operand_count` without
  checking the layout.
- The MLIR layout attribute format is exercised by a parser-side test or
  an equivalent lowering test covering every static leaf kind, fixed
  arrays, tuples, empty tuples, malformed input, and the bare single
  static layout form.
- Existing runtime ABI and extern-trait lowering tests pass unchanged.
- `zig build check-abi-layout-ownership` passes with
  `src/abi/runtime_encoder.zig` whitelisted as the runtime materializer.
- `zig build test` passes.

M3b (runtime parity validation) is not complete until all of the
following are true:

- The cross-layer runtime parity test asserts byte-for-byte equality
  between the M2 comptime bytes, runtime-produced bytes, and
  cast-anchored expected bytes for every static type in the M2 corpus.
- The MLIR layout attribute format has a serializer/parser round-trip
  test for representative static layouts. The test must fail if Zig
  emits a layout string that the C++ parser cannot consume with the same
  static materialization decisions.
- Existing runtime ABI and extern-trait lowering tests pass unchanged.
- `zig build check-abi-layout-ownership` passes.
- `zig build test` passes.

M4 is split into two stages so the first dynamic head/tail implementation
does not get conflated with runtime dynamic memory emission.

M4a (comptime dynamic `string`/`bytes`) is not complete until all of the
following are true:

- `@abiEncode` accepts dynamic `string` and `bytes` values at comptime.
- The comptime materializer emits ABI head/tail payloads for single
  dynamic values and flat/nested mixed static/dynamic tuples.
- Dynamic arrays and structs containing dynamic fields still reject with
  explicit diagnostics.
- Cast-anchored byte tests cover empty strings, non-empty strings,
  multi-byte UTF-8 strings, empty/small/aligned/unaligned bytes, and
  flat/nested mixed static/dynamic tuples.
- The dynamic corpus test at
  `ora-example/corpus/comptime/abi_encode_dynamic.ora` compiles cleanly
  with no diagnostics and checks representative offsets, lengths, and
  byte observations.
- `zig build test` passes.

M4b (runtime dynamic `string`/`bytes`) is not complete until all of the
following are true:

- The runtime materializer handles `dynamic(string)` and `dynamic(bytes)`
  layout nodes for plain payload and selector-prefixed ABI encode ops.
- Runtime tests assert byte-for-byte equality against M4a comptime bytes
  and cast-anchored expected bytes for the same dynamic cases.
- The MLIR layout parser/materializer tests cover dynamic string/bytes
  nodes and reject still-unsupported dynamic arrays.
- `zig build check-abi-layout-ownership` passes.
- `zig build test` passes.

---

## Type coverage

The `sema.Type → LayoutNode` conversion (Milestone 1) is **partial** —
not every Ora type maps to an ABI representation. This section
enumerates the contract.

### Ora types that DO have an ABI representation

| Ora type | Canonical ABI type | Encoding category |
|---|---|---|
| `u8` .. `u256` | `uint8` .. `uint256` | static word |
| `i8` .. `i256` | `int8` .. `int256` | static word (sign-extended) |
| `bool` | `bool` | static word |
| `address` | `address` | static word |
| `bytes1` .. `bytes32` | `bytes1` .. `bytes32` | static word (right-padded) |
| `bytes` (dynamic) | `bytes` | dynamic (length + bytes) |
| `string` | `string` | dynamic (length + UTF-8 bytes) |
| `[N]T` (fixed array, T encodable) | `T[N]` | static if T static, else dynamic |
| `slice<T>` / dynamic array | `T[]` | dynamic |
| `struct {...}` (encodable fields) | tuple `(T1, T2, ...)` | static if all-static, else dynamic |
| Tuple `(T1, T2, ...)` | tuple `(T1, T2, ...)` | composite |
| `enum E` | underlying integer (typically `uint8`) | static word — **NOT supported by current `canonicalAbiType` (returns `UnsupportedAbiType` for `.enum_`); Milestone 1 must unify enum handling with `module_lowering.zig` which already has separate named-enum support** |
| `bitfield Foo(uN)` | `uintN` (the underlying integer) | static word |
| `T & refinement` | `T` (refinement dropped at ABI level) | inherits T |

Two rules worth highlighting:

**Refinements are stripped.** A refined value crossing the ABI boundary
loses its refinement guarantee on the wire — the receiver sees a plain
`T`. The refinement is internal verification state, not a wire concept.
This is a soundness consideration the verifier handles; the encoder
just emits the base type.

**Bitfields encode as their underlying integer.** A `bitfield Foo(u256)`
encodes as `uint256`. The named field/bit-range layout is Ora-internal
metadata. For external tooling that needs to decode named fields, the
bitfield layout should be exposed via ABI metadata (an Ora-specific
extension alongside the standard ABI JSON), not via a different
encoding.

### Ora types that DO NOT have an ABI representation

| Ora type | Why no ABI form |
|---|---|
| `type` (type values) | Comptime-only; doesn't exist at runtime |
| Function types | EVM has selectors, not function pointers; for a function's selector, use `@selector(fn)` which returns `bytes4` |
| `trait T` references | Pure Ora concept; no wire form. For ERC-165 interface IDs use `std.interfaces.interfaceId(T)`. |
| `map<K, V>` | Solidity-style mappings aren't externally representable; they live in storage with hash-derived slots. Encode individual entries by their key. |
| Storage references / handles | Internal-only |
| `never` | No values to encode |
| Generic type parameter `T` (before monomorphization) | Not a concrete type; encoder runs on monomorphized types only |
| Error unions `!T` at user call sites (`@abiEncode(error_union_value)`) | The user-facing `@abiEncode` rejects parameter-position error unions. Ora's error model crosses contract boundaries via `errors(...)` clauses + selector matching, not as encoded parameter values. The `T` payload itself is separately ABI-representable. **Note:** compiler-internal use of error-union ABI layouts (for the result/error carriers emitted by `errors(...)` clauses) is separate — `canonicalAbiType` in `hir/abi.zig` has special-case support for those internal shapes today and that must continue to work. |
| User-declared tagged unions (if/when supported) | No direct ABI form. Users would model as `(tag, payload)` tuple manually. |

### Rejection diagnostics

The `@abiEncode` builtin rejects non-representable types with
actionable messages — tell the user what to do instead, not just that
it failed:

| Rejected type | Diagnostic |
|---|---|
| Function type | `"@abiEncode: function types cannot be encoded. For a function's 4-byte selector use @selector(fn)."` |
| `map<K, V>` | `"@abiEncode: maps cannot be encoded — they only exist in storage. Encode individual entries by their key."` |
| `type` value | `"@abiEncode: type values are comptime-only and cannot cross the ABI boundary."` |
| Trait reference | `"@abiEncode: trait references cannot be encoded. For the canonical interface ID use std.interfaces.interfaceId(T)."` |
| Generic type param | `"@abiEncode: generic type parameter '{T}' must be monomorphized before encoding."` |
| Error union | `"@abiEncode: error unions are not parameter-position types. Encode the success payload directly."` |
| Other unrepresentable | `"@abiEncode: type '{T}' has no ABI representation."` |

### Edge cases worth pinning explicitly

- **Empty tuple `()`** → empty bytes (zero-length encoding). Used for `void`-returning function results.
- **Single-element tuple `(T)` vs bare `T`** → identical encoding. Worth a test that pins this.
- **`void` function return** → encoded as empty tuple (empty bytes).
- **Empty arrays of dynamic types** (e.g., `string[]` with zero elements) → `uint256(0)` (just the length, no tail content).

### Where this contract lives

These tables describe the contract of the `sema.Type → LayoutNode`
converter. The converter is a partial function; non-representable types
produce a typed error that drives the rejection diagnostic.

When stdlib documentation exists, the user-facing version of this table
should live there as the authoritative reference. The version here in
`encoder.md` is encoder-internal — it describes what the encoder
accepts and rejects, not how Ora users should learn type-ABI mapping.

---

## Offset semantics

The ABI head/tail layout uses 32-byte offset words in the head to point
into the tail. The rules for these offsets are precise and subtle —
this section documents them explicitly because misimplementation
produces bytes that decode incorrectly without obvious failure modes.

### The per-encoding offset frame rule

**Each encoding has its own offset frame.** Offsets are byte
distances measured from the start of *that specific encoding's head*,
not from any outer encoding, not from any calldata position, not from
any memory base.

Concretely:

- **Top-level `abi.encode(args...)`**: offsets are relative to byte 0
  of the encoding's output. Byte 0 is the first byte of head[0].
- **A nested dynamic value** (e.g., a `string[]` inside a tuple) has
  its own encoding. The offsets *inside* that nested encoding are
  relative to *that* encoding's byte 0, not the outer encoding's
  byte 0.
- **Each recursion level introduces a new frame.** A `uint256[][]`
  encoding has three offset frames: the outer-tuple frame (if wrapped
  in one), the outer array's frame, and each inner array's frame.

### Worked example A — flat: `(uint256, "hello")`

```
head_size = 2 * 32 = 64 bytes (two argument slots)

byte 0   head[0]  uint256(1)              static — inline value
byte 32  head[1]  uint256(64)             offset to tail of "hello" (= byte 64, relative to byte 0 of this encoding)
byte 64  tail     uint256(5)              length of "hello"
byte 96  tail     "hello" + 27 zero bytes content padded to 32-byte boundary

Total: 128 bytes
```

The offset `64` written at byte 32 is interpreted by the receiver as
"go to byte 64 of *this encoding*". Self-consistent regardless of
where the bytes physically live.

### The dynamic-array encoding rule

For a dynamic array `T[]`:

```
encode(T[]) = uint256(length) ++ enc((T[0], ..., T[length-1]))
```

That is, a `T[]` encoding is the length word followed by a **tuple
encoding** of its elements. **Offsets inside the element head are
relative to the start of that element-head tuple — NOT relative to the
length word.** The length word is a prefix; the offset frame begins
after it.

This is the rule the next example traces. Easy to get wrong — a flat
"this whole array's bytes" frame produces offsets that decode to the
wrong positions.

### Worked example B — nested: `(uint256[][])` = `[[1, 2], [3, 4, 5]]`

The outer wrapper is the tuple of the one argument. Bytes verified
against `cast abi-encode "f(uint256[][])" "[[1,2],[3,4,5]]"`.

```
Outer tuple (one dynamic argument):
byte 0    0x20                           offset to outer array (= byte 32 in outer-tuple frame)

Outer array — starts at byte 32:
byte 32   0x02                           outer array length = 2 elements
                                         ─── element-head frame begins here (byte 64) ───
byte 64   0x40                           offset to element 0 (= 64 bytes into element-head frame)
byte 96   0xa0                           offset to element 1 (= 160 bytes into element-head frame)

Element 0 — starts at byte 128 (= byte 64 of element-head frame + offset 0x40):
byte 128  0x02                           element 0 length = 2
byte 160  0x01                           element 0 [0]
byte 192  0x02                           element 0 [1]

Element 1 — starts at byte 224 (= byte 64 of element-head frame + offset 0xa0):
byte 224  0x03                           element 1 length = 3
byte 256  0x03                           element 1 [0]
byte 288  0x04                           element 1 [1]
byte 320  0x05                           element 1 [2]
```

The key positions:

- The outer-tuple offset (`0x20`) is relative to byte 0 of the outer
  tuple's encoding. Points to byte 32 — start of the array.
- The array's element-head offsets (`0x40`, `0xa0`) are **NOT**
  relative to byte 32 (start of the array, where length lives). They
  are relative to byte 64 — **start of the element-head tuple, after
  the length word**.
- Verification: `0x40 + 64 = 128` = byte where element 0's length sits.
  `0xa0 + 64 = 224` = byte where element 1's length sits. Both match.

**This recursion is load-bearing.** A materializer that anchors array
offsets to the length-word position will be off by 32 bytes per array
level. The correct anchor is "after the length word; start of the
element-head tuple."

### EVM access semantics (informative — not the encoder's concern)

The encoder produces self-consistent bytes. Where those bytes live
and how the receiver accesses them is a downstream concern, but worth
noting briefly:

- **Calldata for a function call** = `selector (4 bytes) ++ encoding`.
  The receiver reads the selector from bytes 0..3, then accesses the
  encoding via `CALLDATALOAD(4 + position)`. The `4` is a compile-time
  constant baked into the receiver's emitted instructions, not a
  runtime base register. There is no `calldata_base` in the EVM —
  `CALLDATALOAD` takes absolute positions starting from byte 0 of
  calldata.
- **Plain `abi.encode` output** (no selector) = just the encoding. If
  it lives in memory, the receiver reads from wherever it's placed
  using absolute positions in that memory region. Offsets in the
  encoding are interpreted relative to the encoding's byte 0,
  wherever that is in memory.
- **Return data** = the encoding (no selector). Accessed via
  `RETURNDATALOAD` / `RETURNDATACOPY` with absolute positions starting
  from byte 0 of return data.

In all cases, the encoding's offset semantics are unchanged — they're
self-relative. The differences are in how the consuming code
addresses into the buffer holding the encoding, which is a codegen
concern handled separately.

### What this implies for the implementation

- The materializer (comptime or runtime) walks `LayoutNode` recursively
- Each recursive call into a nested dynamic encoding establishes a
  fresh offset accumulator starting at 0
- Offsets written into a head are relative to *that* head's frame
- Tail content is appended to *that* frame's tail buffer
- The outer materializer concatenates the nested encoding as a single
  opaque blob — it doesn't need to inspect or fix up offsets inside

For the runtime path, MLIR ops compute offsets at runtime when array
length isn't comptime-known. The recursion still applies — each
dynamic array level emits its own length+offset computations relative
to that level's frame.

### Tests must exercise the recursion

The cross-layer parity test isn't sufficient on its own — both
materializers could share the same bug and still produce matching
bytes. The correctness anchor is **matching Solidity's output for
nested dynamic structures specifically**. Cross-check tests for
Milestone 5+ must include:

- `uint256[][]` (array of arrays)
- `string[]` (array of dynamic primitives)
- `(string, string[])` (tuple mixing dynamic primitive and dynamic-of-dynamic)
- `((string, uint256), bytes)` (nested tuple with dynamics at multiple levels)
- Deeply nested (3+ levels)

If our bytes match Solidity's bytes for these specific cases, the
offset frame recursion is correct. Anything less doesn't prove it.

---

## The milestones

Each milestone produces a stable, verifiable system. The deliverable
matters more than the timeline — no time estimates here.

### Milestone 1: Canonical layout model

**Goal:** one layout description, consumed everywhere. No encoding logic
yet.

**Work:**

- Investigate the three existing layout-description sites
  (`hir/abi.zig`, `module_lowering.zig`, `SIRDispatcher.cpp`) to
  understand their shapes and overlaps
- Decide which becomes canonical, or whether a new top-level module is
  cleaner. Recommendation: a new module `src/abi/layout.zig` since the
  ABI layout is conceptually distinct from HIR or module concerns
- Define `LayoutNode` as a tree per Decision 3
- Build the converter: `sema.Type → LayoutNode`
- Decide the MLIR attribute serialization format. Candidates:
  - A string DSL (e.g., `"tuple(static_word(uint),dynamic_bytes,...)"`) — human-readable, easy to debug
  - A dense int array — compact, faster to parse
  - **Recommendation:** string DSL initially; revisit if performance matters
- Migrate the existing sites to consume the canonical model

**Deliverable:** `sema.Type → canonical layout` works for every
ABI-representable Ora type. **No encoding yet** — just the model.

**Tests:**

- Round-trip: `sema.Type → LayoutNode → canonical_abi_string` matches
  what `hir/abi.zig:canonicalAbiType` produces today (regression baseline)
- Layout parsing in C++ produces identical structure to the Zig-side
  layout for each test case
- Every ABI-representable type in Ora's test corpus has a defined
  layout

**Why this matters most:** every subsequent milestone is an incremental
capability on top of a stable model. This milestone is pure refactoring
with no user-visible behavior change, so the temptation to skip it will
be real — resist it. Skipping means adding a fourth source of truth and
discovering the architectural problem mid-implementation, when the cost
of restructuring is highest.

### Milestone 2: Static `@abiEncode`

**Goal:** `@abiEncode(value)` works at comptime for all static types.

**Scope:**

- All static primitives: `uint8..uint256`, `int8..int256`, `bool`,
  `address`, `bytes1..bytes32`
- Fixed-size arrays of static types
- Tuples (including struct values) where every field is static
- All-static nested aggregates

**Work:**

- Lexer registration of `"abiEncode"`
- Sema entry: argument check, return type `bytes`, error on
  non-ABI-representable types and on dynamic shapes outside the current
  milestone scope
- Comptime materializer: walks `(LayoutNode, ConstValue)` tree, emits
  bytes by type per static encoding rules

**Static encoding rules (locked here):**

| Type | Encoding |
|---|---|
| `uintN` (N ≤ 256) | 32 bytes, big-endian, zero-padded on the left |
| `intN` (N ≤ 256) | 32 bytes, two's complement, sign-extended on the left |
| `bool` | 32 bytes, all zeros except last byte (0x00 or 0x01) |
| `address` | 32 bytes: 12 zero bytes + 20 address bytes (left-padded) |
| `bytesM` (1 ≤ M ≤ 32) | 32 bytes: M data bytes + (32-M) zero bytes (right-padded) |
| `T[K]` (fixed array of static T) | K × encode(T) concatenated |
| Tuple of static | Concatenation of element encodings |

**Diagnostics:**

| Misuse | Diagnostic |
|---|---|
| `@abiEncode()` | `"@abiEncode expects 1 argument, found 0"` |
| `@abiEncode(a, b)` | `"@abiEncode expects 1 argument, found {n}"` |
| `@abiEncode(slice[struct with dynamic field])` | `"@abiEncode does not yet support dynamic arrays containing dynamic elements"` |
| `@abiEncode(struct with dynamic field)` | `"@abiEncode does not yet support struct values containing dynamic fields"` |
| `@abiEncode(unencodable)` | `"@abiEncode: type '{T}' has no ABI representation"` |

**Tests:**

- Each primitive with known-correct byte output (cross-checked against
  `cast abi-encode` from foundry)
- Static tuples: `(uint256, uint256)`, `(uint256, address, bool)`,
  nested static tuples
- Static fixed arrays: `bool[3]`, `uint256[2]`, `(uint256, bool)[2]`
- Edge cases: empty tuple, single-element tuple vs bare value
- Rejection diagnostics for each unsupported case

**Deliverable:** `@abiEncode(static_value)` works at comptime. Later
milestones extend the same surface to dynamic values; shapes outside the
implemented stage reject with clear, actionable diagnostics.

### Milestone 3: Runtime calldata refactor

**Goal:** runtime `ora.abi_encode` consumes the same layout model.
Static encoding is unified across comptime and runtime.

**Work:**

- Refactor `ControlFlow.cpp:3164` to consume serialized layout
  attributes instead of the current `4 + 32 * operand_count`
  assumption
- Split the op into `ora.abi_encode` (plain payload) and
  `ora.abi_encode_with_selector` (selector ++ payload) per Decision 2
- Update HIR emission (`expr_lowering.zig:1630`) to emit the right op
  for each context
- External call lowering uses `ora.abi_encode_with_selector`; other
  consumers use the plain op
- Keep dynamic types outside the M3 static runtime boundary

**Tests:**

- Existing runtime ABI tests still pass with identical byte output
- Cross-layer test: same static value through both comptime and runtime
  produces identical bytes
- Selector-coupled calldata still works for external calls

**Deliverable:** the architectural unification is complete for static
types. One layout model, two materializers, byte-identical output. The
hard part of the cross-language boundary is solved.

### Milestone 4: Dynamic `string` and `bytes`

**Goal:** the first real head/tail test. Single dynamic type at the
outer level.

**Scope:**

- `string` — `uint256(length)` + UTF-8 bytes + zero-padding to 32-byte boundary
- `bytes` (dynamic) — same shape with raw bytes

**Work:**

- `dynamic_bytes` becomes a recognized `LayoutNode` variant
- Comptime materializer handles head offset + tail content
- Runtime materializer handles the same via MLIR ops
- Mixed tuples (`(uint256, string)`, `(string, bytes)`) now type-check
  and encode

**Tests:**

- `string ""`, `"hello"`, multi-byte UTF-8
- `bytes` empty / small (< 32 bytes) / aligned-to-32 / unaligned
- `(uint256, string)` — head/tail with one static and one dynamic
- `(string, uint256, bytes)` — head/tail with alternating
- `(string, bytes)` — both dynamic
- Cross-layer: comptime and runtime produce identical bytes

**Deliverable:** `string` and `bytes` work end-to-end. The head/tail
mechanism is proven on the simplest dynamic case.

### Milestone 5: Dynamic arrays

**Goal:** `T[]` for any `T`. Runtime loops become load-bearing.

**Scope:**

- `T[]` where `T` is static: length-prefix + flat element encoding
- `T[]` where `T` is dynamic: length-prefix + per-element head/tail
- Recursive cases: arrays of arrays, arrays of mixed-content tuples

M5 is split into stages so comptime support for simple dynamic arrays
does not get conflated with runtime loop emission or recursive
element-frame handling.

M5a (comptime dynamic arrays with static elements) is not complete until
all of the following are true:

- `@abiEncode` accepts comptime-known `slice<T>` / dynamic-array values
  when `T` has a static ABI layout.
- The comptime materializer emits `offset + length + elements` for bare
  dynamic arrays and correct head/tail offsets for tuples containing
  those arrays.
- At the M5a boundary, dynamic arrays whose element layout is dynamic
  still reject with an explicit diagnostic. This slice boundary is
  superseded by M5c comptime support.
- Cast-anchored byte tests cover `uint256[]`, `address[]`, and a mixed
  tuple containing a dynamic array.
- The dynamic corpus test at
  `ora-example/corpus/comptime/abi_encode_dynamic.ora` checks
  representative array length, offset, and element bytes.
- `zig build test` passes.

M5b (runtime dynamic arrays with static elements) is not complete until
all of the following are true:

- The C++ runtime materializer parses `array(dynamic,...)` layout nodes
  whose element layout is static.
- Runtime emission handles element-count-dependent size calculation and
  per-element stores without introducing a parallel layout pipeline.
- Runtime tests assert byte-for-byte equality against M5a comptime bytes
  and cast-anchored expected bytes for the same array cases.
- `zig build check-abi-layout-ownership` passes.
- `zig build test` passes.

M5c (dynamic-element arrays with recursive offset frames). **Status:
complete.**

Verified by:

- The comptime materializer encodes `string[]`, `bytes[]`,
  `uint256[][]`, and arrays of dynamic tuples with per-element offset
  frames.
- The C++ runtime materializer parses `array(dynamic,...)` with dynamic
  element layouts and emits per-element head/tail loops.
- Cross-layer parity tests assert byte-for-byte equality against cast for
  each of: `string[]`, `bytes[]`, `uint256[][]`, `string[3]`, and
  `(uint256, string)[]`.
- The MLIR layout parser/materializer round-trip test covers
  `array(dynamic,...)` with all the above element kinds.
- `zig build check-abi-layout-ownership` passes.
- `zig build test` passes.

**Critical correctness rule:** the **per-encoding offset frame**
described in the Offset semantics section becomes load-bearing here.
Every nested dynamic encoding (e.g., each element of a `string[]`)
establishes its own offset frame. The materializer must track the
current frame recursively — flat byte-counter accumulation across
nesting levels is wrong and will produce bytes Solidity disagrees with.

**Work:**

- `dynamic_array` `LayoutNode` variant
- Comptime materializer iterates comptime-known array values, with
  per-level offset accumulator state (reset at each recursion)
- Runtime materializer emits the loop in MLIR with the same per-frame
  offset semantics (this is the new runtime-only complexity)
- Length-prefix handling

**Tests:**

- `uint256[]` empty / one / many elements
- `address[]`, `bool[]`
- `string[]` with various element lengths
- `uint256[2][]` (dynamic array of fixed arrays)
- `(uint256, string)[]` (array of mixed tuples)
- Nested: `uint256[][]`
- Cross-layer parity for all of the above

**Deliverable:** dynamic arrays work. The runtime-loop pattern is
proven and reusable for milestone 6.

### Milestone 6: Nested dynamic aggregates

**Status: complete.**

**Goal:** full EVM ABI spec coverage. Edge cases land.

**Scope:**

- Tuples and arrays of arbitrary depth containing any mix of dynamic
  and static
- Recursive head/tail across multiple nesting levels
- Empty arrays of dynamic types
- Deeply nested dynamic structures
- Struct values containing dynamic fields

**Verified by:**

- Comptime materializer encodes `((string, uint256), bytes)`,
  deep-nested tuples (5 levels), `string[][]`, dynamic structs, arrays
  of dynamic structs, and empty-array/empty-string edge cases with
  per-encoding offset frames.
- C++ runtime materializer handles the same shapes through the existing
  `AbiLayoutNode` tree. No new layout decision path is introduced.
- Cross-layer parity tests in `src/compiler.test.abi.zig` assert
  byte-for-byte equality against `cast abi-encode` for every shape.
- Isolated layout serializer/parser/materializer tests in
  `src/abi.test.zig` cover dynamic structs and deep-nested dynamic
  layouts.
- Corpus coverage in `ora-example/corpus/comptime/abi_encode_dynamic.ora`
- `zig build check-abi-layout-ownership` passes.
- `zig build test` passes.

**Deliverable:** every ABI-representable Ora type encodes correctly,
byte-for-byte matching Solidity's output. Cross-layer tests verify
comptime and runtime produce identical bytes for every test case.

---

## Test strategy

### Correctness anchor

Solidity output via `cast abi-encode` (foundry) or a small reference
contract. For each canonical case:

1. Construct the equivalent value in Solidity
2. Capture the encoded bytes (via `cast`, or a `return` from a small
   contract)
3. Lock the byte sequence in as an expected value in Ora tests
4. Assert both comptime and runtime produce those exact bytes

This is the only meaningful definition of "correct" — matching what
the EVM ecosystem produces.

### Test matrix per milestone

Each milestone adds tests for its new capability. Earlier milestones'
tests stay as regression baseline. By milestone 6, the test corpus
covers:

- Every primitive
- Static aggregates of every shape
- Each dynamic type at outer level
- Mixed static + dynamic tuples
- Dynamic arrays of every element shape
- Recursive dynamic nesting
- Edge cases (empty, deeply nested)

### Cross-layer parity test

For every non-trivial test, run the value through both comptime and
runtime materializers and assert byte-for-byte equality. **This is the
load-bearing test that proves the shared layout model works.**

If comptime and runtime ever diverge for the same input, the bug is in
one of three places: the shared layout (decisions made in Zig), the
comptime materializer's interpretation of it, or the runtime
materializer's interpretation of it. The parity test localizes the bug
to "somewhere in the encoder pipeline" but doesn't identify which
component. Diagnosing further requires comparing each layer's output
against Solidity's output independently — whichever layer is wrong
becomes evident.

---

## Risk register

**R1. The investigation for Milestone 1 may reveal more entangled
state than expected.** The three layout-description sites
(`hir/abi.zig`, `module_lowering.zig`, `SIRDispatcher.cpp`) may have
non-obvious dependencies on each other. Mitigation: investigation
deliverable is a short report mapping the existing state before
committing to which becomes canonical.

**R2. MLIR attribute serialization may be more constrained than a
string DSL suggests.** If MLIR's attribute system doesn't cleanly
support the layout DSL we want, we may need to encode as a dense int
array with a parsing helper on both sides. Plan accommodates either.

**R3. Runtime loops in MLIR for dynamic arrays (Milestone 5) require
new code patterns in the C++ pass.** The current `ControlFlow.cpp`
encoder is straight-line; adding loops touches pass infrastructure.
Mitigation: prototype the loop emission early in Milestone 5 to
de-risk before committing to the full milestone scope.

**R4. Comptime and runtime divergence in edge cases.** Even with a
shared layout model, the two materializers can produce different bytes
if either has a bug — and divergence localizes the bug to "somewhere
in the pipeline" without naming the layer (see
[Cross-layer parity test](#cross-layer-parity-test) for the
localization procedure). The parity test only catches cases we
remember to write. Mitigation: derive the test matrix from the ABI
spec itself, not from "what came to mind."

**R5. Solidity's behavior on ambiguous cases.** A few corners of the
spec are ambiguous (e.g., empty bytes, single-element-tuple
treatment, encoding of bare values vs. one-element tuples). When
Solidity and the spec disagree, follow Solidity — it's the de facto
reference. Document any cases where we make an interpretation
explicit.

**R6. Test infrastructure for capturing Solidity output.** Setting
up `cast abi-encode` or a small Solidity reference contract in the
test pipeline is non-trivial. Mitigation: hand-lock canonical values
from running `cast` once, commit them as expected bytes. Re-run
periodically rather than on every CI build.

---

## Open questions

1. **Where does the layout model live?**
   - `src/abi/layout.zig` (new module, conceptually distinct)
   - Inside `src/hir/abi.zig` (extends what's there)
   - **Recommend:** new module. ABI layout is its own concern,
     consumed by HIR, comptime, and MLIR lowering. Putting it inside
     any one of those layers is wrong.

2. **MLIR attribute encoding format.**
   - String DSL (`"tuple(static_word(uint),dynamic_bytes)"`)
   - Dense int array with type codes
   - **Recommend:** string DSL initially; revisit if profiling
     shows attribute parsing is hot.

3. **Where do canonical ABI signature strings live after Milestone 1?**
   - The layout model can derive them, but the current
     `hir/abi.zig:canonicalAbiType` is widely used. Either:
     - Keep the existing function but reimplement it on top of the
       new layout model
     - Replace callers with layout-driven signature derivation
   - **Recommend:** keep `canonicalAbiType` as a public API,
     reimplement internally on the new layout. Backward-compatible.

4. **`@abiEncodePacked` equivalent?**
   - Solidity has `abi.encodePacked` for non-spec-compliant packed
     encoding. Mostly an anti-pattern (hash collisions, ambiguity).
   - **Recommend:** do not ship. Users who need it can construct
     bytes manually via `@concat` and primitive conversions. This
     keeps the encoder spec-compliant and rejects the foot-gun.

5. **`@abiDecode` (the inverse)?**
   - Different consumer set, different complexity profile.
   - **Recommend:** out of scope for this plan. Separate effort
     after the encoder lands.

6. **Performance of comptime encoding.**
   - For very large comptime values (e.g., large arrays), comptime
     encoding could be slow. Acceptable for v1; revisit if a real
     consumer hits limits.

---

## Things deliberately out of scope

- **`@abiEncodePacked` equivalent.** Anti-pattern; not in scope.
- **`@abiDecode(bytes, type)`.** Inverse operation, different consumer
  set; separate plan if needed.
- **Higher-level convenience builtins** (e.g.,
  `@abiEncodeCall(fn, args)`). Composable from `@selector` + `@abiEncode`,
  doesn't need a dedicated primitive.
- **Custom encoding hooks per type.** ABI is the EVM spec; user-defined
  encoding doesn't apply.
- **Migration of any specific consumer.** Storage slot derivation,
  EIP-712, CREATE2 args — these are downstream of `@abiEncode` and
  evaluated independently.

---

## Sequencing constraints

- Milestone 2 depends on Milestone 1 (needs the layout model)
- Milestone 3 depends on Milestone 2 (needs working comptime encoder
  as the reference output)
- Milestone 4 depends on Milestone 3 (needs the unified architecture
  before adding new type kinds)
- Milestones 5 and 6 depend on 4

Each milestone is independently shippable. A pause after any milestone
leaves the system in a stable state — users get whatever capability
that milestone provided, with clean diagnostics for unsupported cases.

---

## Honest framing

**Approve the goal, reject the all-at-once scope.** The right first
milestone is not "full EVM ABI shared encoder." It is "one
canonical ABI layout model, static encoding shared by comptime and
runtime, with dynamic types rejected cleanly."

That gives a solid base instead of a large cross-layer refactor with
multiple hidden traps. Dynamic types arrive in subsequent milestones
on top of a proven foundation.
