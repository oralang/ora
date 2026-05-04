# Ora compiler — salsa-like query layer review

Audience: an architect reviewer who's seen Rust's `salsa` crate or
similar incremental-query systems. This is an honest read of
`src/db/mod.zig` (719 LOC) and the queries that hang off it, focused
on whether the system is doing what it claims to do.

**Bottom line:** the *shape* is salsa-shaped, but the *guarantees*
aren't. It's a hand-rolled memoization layer with manual
graph-walk invalidation, a single-threaded execution model, and
two distinct cycle-detection workarounds bolted onto specific
queries. It works for the linear `compilePackage` flow today; it
will not survive an LSP that edits files in a hot loop without
producing stale results.

## 1. Architecture, as built

`CompilerDb` (`src/db/mod.zig:92`) owns:

- `sources: SourceStore` — the leaves (packages, modules, files,
  text). Mutated through `updateSourceFile` (line 175).
- Per-file slots: `syntax_slots`, `ast_slots` (one optional
  result per `FileId`).
- Per-module slots: `module_graph_slots`, `item_index_slots`,
  `resolution_slots`, `consteval_slots`, `module_verification_slots`,
  `hir_slots` (one optional result per `ModuleId`/`PackageId`).
- Per-module hashmaps: `typecheck_slots: ArrayList(TypeCheckCache)`
  and `verification_slots: ArrayList(VerificationCache)`. Each
  inner cache is keyed on `u64` derived from
  `(kind << 32) | id` (`typeCheckCacheKey`, `src/db/mod.zig:700`).

Invalidation flows from `updateSourceFile` →
`invalidateFile(file_id)` (line 474) → walk every module, find
ones whose `file_id` matches, walk dependents through the cached
module graph, clear caches per dependent.

Public query methods (top-level entry points): `syntaxTree`,
`astFile`, `moduleGraph`, `itemIndex`, `resolveNames`,
`typeCheck`, `moduleTypeCheck`, `constEval`, `verificationFacts`,
`moduleVerificationFacts`, `lowerToHir`. Caller is `compilePackage`
in `src/driver/mod.zig` (176 LOC) and the LSP in `src/lsp/`.

## 2. What works

- **Layering is clean.** Source store → syntax → AST → module
  graph → item index → name resolution → type-check + const-eval
  → verification → HIR. Each layer's result type is its own
  struct with a `deinit`. Arena allocation per result is
  consistent.
- **Cycle handling**, in the queries that have it. Type-check
  uses an `in_progress` set + a `sentinels` map that returns
  "unknown" results during recursion (`src/db/mod.zig:21-64`).
  Const-eval has an analogous `consteval_in_progress` boolean +
  `consteval_sentinel_slots` (line 102-104). When a query
  re-enters itself the recursive caller gets a deterministic
  fallback instead of a stack overflow.
- **Topological invalidation through the module graph.** When a
  file changes, `invalidateModuleDependents` (line 520) walks
  the module DAG and invalidates every module that imports the
  changed one, transitively. This is the right abstraction.
- **No global lock or mutex.** For a single-threaded compiler
  that's the right call — keeps the code obvious and lets you
  reason about ordering.

## 3. Real concerns

### 3.1 No revision counter — invalidation is "trust the graph"

Salsa's core trick is a global revision number that increments on
every input change, plus per-query *minimum revision read* tags
on cached results. A query result is valid iff every dependency
it read has a revision ≤ the result's read-revision. This makes
stale-result detection a primitive operation.

`CompilerDb` has no revision counter. Cache validity is
"the slot is non-null" (`syntaxTree` at `src/db/mod.zig:131`-ish
pattern). `invalidateFile` clears slots by walking the module
graph forward.

The risk: if `invalidateModuleDependents` misses an edge — e.g.
because the graph itself was invalidated and re-computed *after*
a downstream cache was populated — a stale result lives forever.
There's no second-line defence.

Concrete example of how this can bite an LSP:
1. User edits `a.ora`. `invalidateFile(a.ora)` clears `a`'s
   syntax/AST and walks the module graph (which still
   references the old `a`).
2. The walk visits `b.ora` which imports `a.ora` and clears
   it. Good.
3. User edits `a.ora` again to *add* a new import of `c.ora`.
4. `invalidateFile(a.ora)` walks the now-stale graph that
   doesn't yet know about the new `a → c` edge.
5. `c`'s caches don't get cleared, but `c`'s name resolution
   may now be visible from `a`. Stale.

The mitigation today: `invalidatePackage` (line 503) drops
`module_graph_slots` for the changed file's package, so step 4's
"now-stale graph" is rebuilt before the next type-check. That
covers the example above. But the design is a series of
specific fixes for specific scenarios rather than a general
guarantee. A new query layer added later could re-introduce the
hole.

### 3.2 Two cache shapes for the same problem

`syntax_slots` etc. are `ArrayList(?*Result)` — one optional
slot per ID. `typecheck_slots` and `verification_slots` are
`ArrayList(SubCache)` where the inner cache is a hashmap keyed
on `(kind << 32) | id`. Reason: type-check and verification have
*two* sub-keys per module (`item` and `body`), and the original
slot-array shape couldn't express that.

Result: every reader has to remember which shape they're
dealing with. There's no `db.cached(query, key) -> ?Result`
abstraction — each query opens the right slot or hashmap by
name.

Lifting this to a single key/value cache (e.g. a
`StringHashMap` keyed on a `(query_name, id, sub_id)` tuple, or
a generated table per query type) would make cycle detection,
revision tracking, and parallelism easier to add later. As-is,
each new query duplicates the slot-or-hashmap decision.

### 3.3 Two cycle-detection implementations

`TypeCheckCache.in_progress` (line 23) and `consteval_in_progress`
(line 103) are the same idea written twice. The const-eval
version is even simpler: a `bool` per module, not a set keyed on
the recursive sub-call. It's coarser — once any const-eval is
running, every const-eval call short-circuits — and has a
`consteval_tainted_slots` companion that re-runs const-eval if
type-check was simultaneously running.

That taint flag is the load-bearing comment in this codebase:
const-eval calls type-check (to know what type a literal is),
type-check calls const-eval (to fold compile-time array sizes),
and the cycle isn't structurally broken, it's papered over with
a "did the other side run while I was running? then re-run me"
flag.

This works today. It is fragile to extend: any new query that
participates in either side of the cycle has to know about the
taint flag. The right fix is a Salsa-style "this query
demanded this other query — and got an Unknown sentinel — so
when the other query is ready, mark me dirty" mechanism. The
infrastructure for that doesn't exist here.

### 3.4 Silent fallback on invalidation failure

`invalidateModuleDependents` returns `!void` (line 520). The
call site (line 486) does:

```
self.invalidateModuleDependents(graph, module_record.id, &invalidated_modules) catch {
    self.invalidateModule(module_record.id);
};
```

If the dependent-walk fails (likely OOM allocating the visited
set), the fallback invalidates *only* the directly-changed
module. Every transitive dependent is now stale. There's no
log, no diagnostic, no flag the caller can check.

For a CLI compile this is fine — OOM in the invalidator means
the next phase will OOM too and the user sees an error. For an
LSP that survives partial failures, this is a silent
correctness hole.

Same pattern at line 491: `invalidated_modules.put(...) catch {}`
silently drops the bookkeeping entry; the loop continues without
knowing.

### 3.5 Cache key construction

`typeCheckCacheKey` and `verificationCacheKey` (lines 700, 707)
build a `u64` from `(@as(u64, kind_tag) << 32) | id_enum_value`.
This relies on `id_enum_value` fitting in 32 bits. There's no
`std.debug.assert` or comment explaining why that's safe. If
`ItemId` ever exceeds 32 bits (unlikely but possible after a
generic-instantiation explosion) the key collides silently.

A `packed struct { kind: u32, id: u32 }` cast to `u64` would be
both safer and self-documenting.

### 3.6 No mechanism to detect "input changed outside the API"

`SourceStore.fileMut` exists (`src/source/mod.zig`). If a
caller mutates source text without going through
`db.updateSourceFile`, no cache is invalidated and every query
returns its old result. There's no checksum, no version vector,
no `// caller asserts caches dropped` discipline.

Salsa solves this by making inputs immutable references plus a
revision tag — you can't mutate without bumping the revision. A
similar mechanism here would mean either:
- making `SourceFile.text` `[]const u8` and forcing all writes
  through `updateSourceFile`, or
- adding a per-file `revision: u32` that any cache reader
  validates before trusting its slot.

Neither is in place.

### 3.7 Single-threaded by design, not by accident

There are no mutexes anywhere in `db/mod.zig`. `in_progress` is
not a concurrent set; the type-check sentinel pattern only works
because there's exactly one thread of recursion. No `RwLock`
around the slot arrays — concurrent readers of the same slot
during a write would race.

For LSP responsiveness this is the next obvious bottleneck.
Every keystroke serializes through the same DB. A salsa-style
parallel query runner would be a major architectural change —
it'd require interior mutability, dependency-edge recording,
and revision-validated reads.

If the team wants real LSP responsiveness, the path is:
1. Add a revision counter (Section 3.1).
2. Lift the cycle-detection sets out of individual queries into
   a generic per-thread "query stack" (Section 3.3).
3. Then introduce parallelism behind a small set of typed
   locks. Don't try to add parallelism without 1 and 2.

## 4. Smaller observations

- `compilerPhaseDebugEnabled()` (line 10) re-reads `ORA_COMPILER_PHASE_DEBUG`
  from the environment on every call via a `page_allocator` alloc.
  Cheap individually, hot-path-noisy in aggregate. Cache once at
  init.
- `invalidateModule` (line 507) clears 9 different slots/maps in
  fixed order. If a new query layer is added without updating
  this method, it goes stale silently. A small `comptime` table
  of `(slot_field, deinit_fn)` pairs would catch the omission at
  build time.
- `unknownTypeCheckResult` (line 536) and the analogous
  `unknownConstEvalResult` (presumably similar) allocate fresh
  results every time the sentinel is missed. The intent is to
  cache them, but if the caller never re-asks for the same key
  the sentinel is wasted.

## 5. What I'd ask before merging more queries onto this DB

- Is there a known scenario where the LSP returns stale results?
  If yes, file the repro.
- What's the contract for `SourceStore.fileMut`? Is it caller's
  responsibility to invalidate, or is mutation-without-invalidation
  a bug?
- Are there plans to add a tenth query (say, a SIR optimization
  pass)? If so, the slot-vs-hashmap decision needs an opinion
  before that lands, not after.
- What's the test coverage for `invalidateFile`? Specifically:
  the dependents-walk hitting an edge case where the module graph
  itself changed in the same edit (Section 3.1).

## 6. Verdict

The DB does what the rest of the compiler currently asks of it.
It is **not** a general-purpose incremental query engine; it is
a topology-aware memoization layer with hand-coded invalidation
and two specific cycle-breaks. As long as the calling pattern is
"compile a package once, answer LSP queries between edits", it's
fine.

If the project goes anywhere near "edit at 60Hz, answer queries
on every keystroke" — i.e. a real IDE — every section above
becomes a bug.
