# Ora

<p align="center">
  <img src="website/static/img/logo-round.png" alt="Ora logo" width="220" />
</p>

Ora is a smart-contract language and compiler for EVM with explicit semantics, verification-aware IR, and a strict MLIR-based pipeline.

## Overview

Ora is designed for teams that want compiler visibility and formal reasoning without hiding execution details. The toolchain is structured around:
- Explicit region/effect semantics (`storage`, `memory`, `transient`, `stack`).
- Strongly typed frontend with refinements and specification constructs.
- Deterministic lowering through Ora MLIR and SIR before EVM bytecode emission.
- Integrated Z3-backed verification for contract annotations and safety obligations.

## Release Track: Asuka (Pre-Release)

Asuka is the current hardening milestone. The goal is production-grade correctness of the end-to-end pipeline and verification behavior, with clear semantics and predictable outputs.

## Current Capabilities

- Frontend pipeline: lexer, parser, typed AST, and Ora MLIR emission.
- Core language features: structs, enums, mappings, switch expressions/statements, error unions, and refinement types.
- Backend pipeline: Ora MLIR -> SIR MLIR / SIR text -> EVM bytecode.
- Verification inputs: `requires`, `ensures`, `invariant`, `assume`, and refinement guards.
- Verification execution controls for mode/call/state reasoning and timeout tuning.
- Path-sensitive assumption handling (compiler path assumptions are scoped; user assumptions remain function-level).

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

```bash
# Format Ora source code
./zig-out/bin/ora fmt ora-example/apps/counter.ora

# Check if code is formatted (useful for CI)
./zig-out/bin/ora fmt --check ora-example/apps/counter.ora

# Emit MLIR (Ora dialect)
./zig-out/bin/ora --emit-mlir ora-example/apps/counter.ora

# Emit SIR MLIR (where supported)
./zig-out/bin/ora --emit-mlir-sir ora-example/apps/counter.ora

# Emit Sensei SIR text
./zig-out/bin/ora --emit-sir-text ora-example/apps/counter.ora

# Emit bytecode
./zig-out/bin/ora --emit-bytecode ora-example/apps/counter.ora

# Emit ABI (Ora-native)
./zig-out/bin/ora --emit-abi ora-example/apps/counter.ora

# Emit ABI (Solidity-compatible JSON)
./zig-out/bin/ora --emit-abi-solidity ora-example/apps/counter.ora

# Run verifier explicitly (default is enabled)
./zig-out/bin/ora --verify --emit-mlir ora-example/apps/counter.ora

# Full verification mode (includes untagged asserts)
./zig-out/bin/ora --verify=full --emit-mlir ora-example/apps/counter.ora

# Run the standard test suite
zig build test
```

Validate fixtures:
```bash
./scripts/validate-examples.sh
```

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
