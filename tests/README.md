# Ora Test Layers

The single pre-push bar is **`zig build gate`** — it runs every layer below on the
committed state. Install the hook so it always runs against what you actually push:

```bash
ln -sf ../../scripts/pre-push-gate.sh .git/hooks/pre-push
```

## Layers

| Layer | What | Where | Run |
|---|---|---|---|
| Unit (Zig) | Co-located `*.test.zig` + `compiler.test.*` modules | `src/` | `zig build test` / `test-compiler` |
| Static tripwires | grep-based invariant checks (no-width-defaults, op-null-fallbacks, …) | `scripts/check-*.sh` | in `zig build test` |
| Ora MLIR goldens | FileCheck snapshots of the Ora dialect | `tests/mlir/` | `zig build check-mlir-ora` |
| SIR MLIR goldens | Full-text `CHECK-NEXT` snapshots of the SIR dialect | `tests/mlir_sir/` | `zig build check-mlir-sir` |
| SIR text goldens | Hand-written semantic locks on dispatcher/codegen text | `tests/sir_text/` | `zig build check-sir-text` |
| Execution conformance | `.ora`/`.spec.toml` run on the in-process `lib/evm` | `tests/conformance/` | `zig build test-conformance` |
| Feature coverage | Manifest: every feature has an executed spec or a skip reason | `tests/conformance/feature_coverage.json` | `zig build check-feature-execution-coverage` |
| Negative corpus | `.ora`/`.expect.toml`: wrong source must fail with the named diagnostic | `tests/negative/` | `zig build check-negative-corpus` |
| Verifier soundness | Bounded mutation set — verifier must reject each mutant | `scripts/verify_mutations.py` | `zig build check-verifier-mutations` |
| EVM unit | lib/evm interpreter tests | `lib/evm/` | `zig build test-evm` |

## Conformance specs (`tests/conformance/`)

A test is a `<name>.ora` + `<name>.spec.toml` pair (or a spec with `source = "..."`
referencing a contract elsewhere in the repo, e.g. an `ora-example/` app). Every
`.ora` must have a sidecar or be listed in `SKIP`. Every spec must be claimed by a
`feature_coverage.json` entry. See `tests/conformance/README.md` for the spec format;
key outcomes per `[[call]]`:

- `returns = { <type> = <value> }` — typed return assertion
- `succeeds = {}` — must succeed, return bytes unchecked (state-effect scenarios)
- `reverts = {}` / `{ selector = "0x..." }` / `{ data = "0x..." }` / `{ any = true }`
- `calldata = "0x..."` — raw hostile bytes instead of `fn`+`args` (adversarial; outcome
  restricted to `succeeds`/`reverts`)

Execution-discovered compiler/EVM bugs are recorded in
[`conformance/FINDINGS.md`](conformance/FINDINGS.md), one entry each; corpus files carry
only `// see FINDINGS.md#<id>` pointers, never finding prose.

## Regenerating goldens

MLIR/SIR goldens are auto-generated. After an intentional lowering change, regenerate
and **review the diff** (never bless blind):

```bash
python3 scripts/generate-mlir-auto-checks.py            # Ora MLIR, create missing
python3 scripts/generate-mlir-auto-checks.py --only <path.ora>   # one file
python3 scripts/generate-sir-auto-checks.py --refresh   # SIR MLIR, rewrite all
```

`tests/sir_text/` is hand-written — edit those `.check` files directly.
