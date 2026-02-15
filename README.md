# Ora

<p align="center">
  <img src="website/static/img/logo-round.png" alt="Ora logo" width="220" />
</p>

Ora is a smart-contract language and compiler for EVM with explicit semantics, verification-aware IR, and a strict MLIR-based pipeline.

## Overview

Ora is designed for teams that want compiler visibility and formal reasoning without hiding execution details. The toolchain is structured around explicit region/effect semantics, a strongly typed frontend, deterministic Ora MLIR → SIR → EVM lowering, and Z3-backed verification.

### Language features at a glance

| Area | Features |
|------|----------|
| **Types** | Unsigned/signed ints (u8–u256, i8–i256), bool, address, string, bytes; structs, enums, tuples, anonymous structs (`.{ ... }`). |
| **Refinement types** | Subtypes with constraints: `NonZero`, `InRange`, `MinValue`, `MaxValue`, `Scaled`, `Exact`, `BasisPoints`, `NonZeroAddress`; compile-time and optional runtime guards. |
| **Abstraction** | **Generics** (`comptime T: type`), generic functions and structs; **comptime** evaluation and constant folding. |
| **Control flow** | **Switch** (expressions and statements; cases, ranges, `default`); loops with **termination** (e.g. `decreases`, invariants); **labels**, `break`/`continue` to labels. |
| **Bit-level** | **Bitfields** with packed fields, optional `@at`/`@bits` layout. |
| **Errors** | **Error unions** and errors-as-values (`!T`), `try`/`catch`-style handling. |
| **Verification** | **SMT** (Z3); `requires`/`ensures`/`invariant`/`assume`; path-sensitive reasoning; refinement guards. |
| **Memory** | Explicit regions: `storage`, `memory`, `transient`, `stack`. |

## Release Track: Asuka (Pre-Release)

Asuka is the current hardening milestone. The goal is production-grade correctness of the end-to-end pipeline and verification behavior, with clear semantics and predictable outputs.

## Current Capabilities

- **Frontend:** Lexer, parser, typed AST, Ora MLIR emission.
- **Backend:** Ora MLIR → SIR MLIR / SIR text → EVM bytecode.
- **Verification:** SMT (Z3) with `requires`/`ensures`/`invariant`/`assume`, refinement guards, path-sensitive assumptions.
- See the **Language features** table above for types, generics, comptime, switch, loops, bitfields, error unions, and verification.

## Current Focus

- Lowering and backend correctness hardening for full Ora -> SIR -> EVM parity.
- Advanced verification precision (interprocedural summaries, loop inductiveness flow, quantified/state reasoning).
- Expanded end-to-end and golden coverage for compiler + verifier regressions.
- ABI and developer tooling stabilization for Asuka release quality.

## Installation and build

**Prerequisites:** Zig 0.15.x, CMake, Git, Z3, MLIR

```bash
git clone https://github.com/oralang/Ora.git
cd Ora
./setup.sh
```

Build the compiler:
```bash
zig build
```

Run tests:
```bash
zig build test
```

## Documentation

- **Language & specs:** [website/docs](website/docs) — structs, refinement types, generics, comptime, ABI, formal verification.
- **Generics (style guide):** [website/docs/generics.md](website/docs/generics.md) — `comptime T: type`, generic functions/structs, monomorphization.
- **Examples:** [ora-example](ora-example) — apps, comptime, and [ora-example/comptime/generics](ora-example/comptime/generics) for generic examples.

## Using the compiler

Run the compiler directly:
```bash
./zig-out/bin/ora <file.ora>
```

Explicit build command:
```bash
./zig-out/bin/ora build [options] <file.ora>
```

Emit/debug command (IR-focused):
```bash
./zig-out/bin/ora emit [emit-options] <file.ora>
```

See available commands and flags:
```bash
./zig-out/bin/ora
```

## Common workflows

- **Format:** `./zig-out/bin/ora fmt <file.ora>` — format source; use `--check` for CI.
- **Build artifacts (default):** `./zig-out/bin/ora <file.ora>` (same as `build`) writes:
  - `artifacts/<name>/abi/<name>.abi.json`
  - `artifacts/<name>/abi/<name>.abi.sol.json`
  - `artifacts/<name>/abi/<name>.abi.extras.json`
  - `artifacts/<name>/bin/<name>.hex`
  - `artifacts/<name>/sir/<name>.sir`
  - `artifacts/<name>/verify/<name>.smt.report.md`
  - `artifacts/<name>/verify/<name>.smt.report.json`
- **Emit MLIR/SIR for debugging:** `./zig-out/bin/ora emit --emit-mlir[=ora|sir|both] <file.ora>`, `./zig-out/bin/ora emit --emit-sir-text <file.ora>`.
- **MLIR output modes:**
  - Ora MLIR: `./zig-out/bin/ora emit --emit-mlir=ora <file.ora>`
  - SIR MLIR (after Ora->SIR): `./zig-out/bin/ora emit --emit-mlir=sir <file.ora>`
  - Both Ora + SIR MLIR: `./zig-out/bin/ora emit --emit-mlir=both <file.ora>`
  - SIR text (Sensei text IR): `./zig-out/bin/ora emit --emit-sir-text <file.ora>`
- **Emit specific outputs:** `./zig-out/bin/ora emit --emit-bytecode`, `--emit-abi`, `--emit-abi-solidity`, `--emit-abi-extras`.
- **CFG generation:** `./zig-out/bin/ora emit --emit-cfg <file.ora>` (defaults to Ora MLIR), or explicitly `--emit-cfg=ora` / `--emit-cfg=sir`.
- **Verify:** `--verify` (default), `--verify=full` for untagged asserts.
- **SMT report in emit mode:** `./zig-out/bin/ora emit --emit-smt-report <file.ora>`.
- **Tests:** `zig build test` — run the test suite. `./scripts/validate-examples.sh` — validate example fixtures.
- **CLI command matrix:** `./scripts/run-cli-command-checks.sh` — run end-to-end checks for build/emit/fmt/verification/advanced MLIR flags. Useful options: `--quiet`, `--out /tmp/ora-cli-cmd-tests`, `--file <path.ora>`, `--compiler <path-to-ora>`.

## Advanced MLIR controls

- Run a custom MLIR pipeline:
  - `./zig-out/bin/ora emit --mlir-pass-pipeline "builtin.module(canonicalize,cse)" --emit-mlir=ora <file.ora>`
- Verify each pass in that pipeline:
  - `--mlir-verify-each-pass` (requires `--mlir-pass-pipeline`)
- Enable pass timing for that pipeline:
  - `--mlir-pass-timing` (requires `--mlir-pass-pipeline`)
- Print stage snapshots:
  - `--mlir-print-ir=before|after|before-after|all`
  - Optional stage filter: `--mlir-print-ir-pass <filter>`
  - Current stage names: `lowering`, `custom-pipeline`, `canonicalize`, `ora-to-sir`, `sir-legalize`
- Save MLIR when a stage fails:
  - `--mlir-crash-reproducer <path>`
- Print module snapshot on MLIR diagnostics:
  - `--mlir-print-op-on-diagnostic`
- Print operation-count stats across stages:
  - `--mlir-pass-statistics`

## Verification timeout and tuning

Z3 verification timeout is controlled via `ORA_Z3_TIMEOUT_MS` (milliseconds):

```bash
# 5 minutes
ORA_Z3_TIMEOUT_MS=300000 ./zig-out/bin/ora emit --verify --emit-mlir ora-example/apps/defi_lending_pool.ora
```

Optional verifier env toggles:
- `ORA_Z3_PARALLEL=0|1`
- `ORA_Z3_WORKERS=<n>`
- `ORA_Z3_DEBUG=1`
- `ORA_VERIFY_MODE=basic|full`
- `ORA_VERIFY_CALLS=0|1`
- `ORA_VERIFY_STATE=0|1`
- `ORA_VERIFY_STATS=0|1`

## Verification assumptions model

- User-authored `assume(...)` statements are treated as function-level verification assumptions.
- Compiler-injected branch assumptions (from path conditions) are tracked as path-scoped assumptions.
- Base assumption-consistency checks use `requires(...)` + user `assume(...)`, while path assumptions are applied only in path-scoped obligation proving.

## Development

Generate the Ora → SIR coverage report:
```bash
python3 scripts/generate_ora_to_sir_coverage.py
```

## Notes

- Asuka is pre-release; breaking changes are expected.
- Ora → SIR parity is the primary milestone for Asuka.
- Generic style (syntax and semantics) is defined in [website/docs/generics.md](website/docs/generics.md).
