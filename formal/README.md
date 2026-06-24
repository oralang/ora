# `formal/` — Ora's Lean 4 mechanization

This directory is Ora's machine-checked formal development. It is a **standalone
Lean 4 proof project**: `lake build` elaborates it and the Lean kernel checks
the proofs. The point of Lean here is the *trusted kernel* — a small, auditable
checker — complementing the automated (but unverified) Z3/SMT discharge the
compiler uses today. See `docs/fv/VolumeI.md` ch. 39–40 ("Why Ora Will Use
Lean", "Z3 + Lean cooperation").

## Status: Phase 1 — type universe (in progress)

The toolchain scaffold is done; the type layer is now under construction,
grounded in the compiler's `src/types/`:

- `Ora/Types/Region.lean` — the 5 regions + provenance, read/write capability,
  and the implicit-coercion relation (`Region.assignableTo`) mirroring
  `src/sema/region.zig:regionAssignable`.
- `Ora/Types/Prim.lean` — the surface primitive universe (the 13 sized integers,
  `bool`, `address`, `bytesN`, `string`, `bytes`, `void`), matched by hand to
  `src/types/builtin.zig`.
- `Ora/Types/Ty.lean` — the recursive composite universe over `PrimTy` + region.

**Not yet present (and required before soundness proofs):**

- `Ora/Types/WF.lean` — well-formedness. `Ty` currently admits raw syntactic
  shapes, not valid Ora types; proofs must NOT treat an arbitrary `Ty` as
  compiler-admissible until this predicate exists.
- the core/internal layer — `never` (⊥), `ElabTy` (`unknown`/`named`), and the
  comptime layer. See the `Ty.lean` header.

## Dependency posture (important)

`formal/` is **independent of the compiler build**:

- It is **not linked into the `ora` binary** and is **not a build/runtime
  dependency**. `build.zig` does not reference `formal/`; nothing in `src/`
  imports anything here.
- Lean is a **proof-checking tool**, not part of the toolchain needed to *build
  or run* Ora. A contributor who only touches the compiler does not need Lean.
- Conversely, `lake build` here does not need Zig/MLIR/Z3.
- CI wiring (a non-gating `lake build` job) is intentionally deferred — it will
  be added later as a separate, informational check.

This separation is deliberate: the formal development can move at its own pace
and use the trusted Lean kernel without coupling the compiler's build graph to a
Lean toolchain.

## Layout

```
formal/
  lean-toolchain     # pinned Lean version (elan reads this)
  lakefile.toml      # Lake package: one library, `Ora`
  Ora.lean           # library root (re-exports modules)
  Ora/
    Types/
      Region.lean    # 5 regions + provenance, read/write capability, coercion table
      Prim.lean      # surface primitive types (ints, bool, address, bytesN, …)
      Ty.lean        # recursive composite universe over PrimTy + Region; located types
```

## Building

```sh
# One-time: install elan (the Lean toolchain manager).
#   curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
# elan auto-installs the version pinned in `lean-toolchain`.

cd formal
lake build        # elaborates + kernel-checks everything; exit 0 == proofs hold
```

## Lean tooling — the `lean4` skill

The Lean 4 proving tooling lives under the project's Claude folder:

- **`.claude/skills/lean4/`** — the committed, self-contained `lean4` skill
  (LSP-first proving, mathlib search, `sorry`/axiom analysis, plus its
  `references/`). Claude auto-discovers and uses it when editing `.lean` files —
  no install step.
- **`.claude/lean4-skills/`** — the full third-party plugin (local-only,
  gitignored): the `/lean4:*` slash commands (`draft · formalize · prove ·
  autoprove · checkpoint · review · golf · learn · doctor`) and proof agents
  (`axiom-eliminator · proof-golfer · proof-repair · sorry-filler-deep`).
  Activate with `/plugin marketplace add .claude/lean4-skills` then
  `/plugin install lean4`; for live goal inspection also install the
  `lean-lsp-mcp` MCP server and run `lake build` in `formal/` first.

Tooling only — no effect on the `ora` compiler build. Third-party, MIT
(`.claude/lean4-skills/LICENSE`).

## Decisions on record

- **Toolchain pin:** `leanprover/lean4:v4.15.0` (in `lean-toolchain`). Bump
  deliberately; keep it in sync with `mathlib` if/when that is added.
- **No `mathlib` yet.** The scaffold needs only Lean core (`rfl`, `omega`,
  `simp`, `induction`). Phase 1 may add `mathlib` via a `[[require]]` block in
  `lakefile.toml` if the soundness proofs want its tactics/lemmas — kept out for
  now to keep builds fast and the dependency surface minimal.
- **Kernel-trusted, Z3-cooperating.** Lean's role is the trusted layer in the
  "Z3 + Lean cooperation" architecture (Ch. 40), not a replacement for Z3.
