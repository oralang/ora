# Release Checklist

This checklist tracks the current release-prep state for Ora as a serious preview release.

## Release Shape

- Compiler: real and integrated
- SMT: deterministic, fail-closed, and release-gated
- Debugger: preview-quality, provenance-aware, still being polished
- Packaging: raw binaries on GitHub releases, not `.dmg`/notarized installers yet

## Compiler Gates

- [x] `zig build`
- [x] SMT success matrix
- [x] SMT expected-failure matrix
- [x] SMT degradation fail-closed matrix
- [x] Ora -> SIR cleanliness matrix
- [x] Release artifact smoke script
- [x] Artifact smoke script wired into CI
- [x] Representative diagnostics matrix

## Debugger Gates

- [x] Ora/SIR synchronized panes
- [x] provenance-aware stepping
- [x] synthetic/hoisted/duplicated metadata in debug artifacts
- [x] TStore lock visibility
- [x] removed-source-line marking
- [x] inline `:help`, `:where`, `:why-here`, `:line-info`
- [ ] final startup-stop policy cleanup
- [ ] final leak/lifetime sweep on feature-heavy paths
- [ ] one final UX pass on wording/legend density

## SMT / Verification Gates

- [x] stable function ordering
- [x] stable annotation ordering
- [x] fixed solver seed
- [x] sorted merge keys in encoder state joins
- [x] sorted counterexample rendering
- [x] source-level degradation repro
- [x] ghost `old(...)` regression coverage
- [ ] one more curated pass over larger real contracts before tag

## Release Scripts / Commands

Core local checks:

```bash
zig build
./scripts/release_artifact_smoke.sh
```

Representative verifier spot checks:

```bash
./zig-out/bin/ora build --verify ora-example/smt/verification/ghost_combined.ora
./zig-out/bin/ora build --verify ora-example/corpus/patterns/multi_asset.ora
./zig-out/bin/ora build --verify ora-example/refinements/guards_showcase.ora
```

Representative debugger spot checks:

```bash
./zig-out/bin/ora debug ora-example/debugger/comptime_debug_probe.ora \
  --init-signature 'init()' \
  --signature 'probe(u256)' \
  --arg 64

./zig-out/bin/ora debug ora-example/vault/05_locks.ora \
  --signature 'deposit(u256)' \
  --arg 25
```

## Remaining Pre-Tag Work

1. Run a final debugger stability pass on:
   - checkpoints
   - session save/load
   - exceptions
   - reverse-by-replay
2. Do one final SMT sweep on the larger examples we want to mention publicly.
3. Review website/release messaging so we describe this as a preview release, not a stable 1.0.
4. Decide whether we want only raw binaries for this release or an additional archive format.

## Current Position

If we tagged today, the honest framing would be:

- first serious Ora preview release
- integrated SMT verification
- interactive provenance-aware debugger
- raw macOS/Linux binaries in GitHub releases
