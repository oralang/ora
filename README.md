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
./zig-out/bin/ora [options] <file.ora>
```

Compile to Sensei SIR text:
```bash
./zig-out/bin/ora compile [options] <file.ora>
```

See available commands and flags:
```bash
./zig-out/bin/ora --help
```

## Common workflows

- **Format:** `./zig-out/bin/ora fmt <file.ora>` — format source; use `--check` for CI.
- **Emit IR / bytecode / ABI:** `--emit-mlir`, `--emit-mlir-sir`, `--emit-sir-text`, `--emit-bytecode`, `--emit-abi`, `--emit-abi-solidity`.
- **Verify:** `--verify` (default); `--verify=full` for untagged asserts.
- **SMT report:** `--emit-smt-report` — writes an SMT encoding audit to `<basename>.smt.report.md` and `<basename>.smt.report.json` (per-file encoding, obligations, and SMT-LIB snippets). Example: `./zig-out/bin/ora --emit-smt-report ora-example/apps/counter.ora` produces `counter.ora.smt.report.md` and `.json`.
- **Tests:** `zig build test` — run the test suite. `./scripts/validate-examples.sh` — validate example fixtures.

## Verification timeout and tuning

Z3 verification timeout is controlled via `ORA_Z3_TIMEOUT_MS` (milliseconds):

```bash
# 5 minutes
ORA_Z3_TIMEOUT_MS=300000 ./zig-out/bin/ora --verify --emit-mlir ora-example/apps/defi_lending_pool.ora
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
