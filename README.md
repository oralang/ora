# Ora

Ora is an experimental smart-contract language and compiler targeting MLIR and EVM. The project is on the **Asuka** pre-release track, focused on establishing a stable end-to-end pipeline with explicit semantics around types, regions, and effects.

## Status (Asuka, pre-release)

**Language features**
- Explicit error unions: `!T | E1 | E2` with `try`-style unwrapping.
- Region model for storage/memory/transient/stack with explicit transitions.
- Structs, enums, mappings, and composite literals.
- Switch as both statement and expression.
- Refinement types and contracts for precise value constraints.
- Formal specification syntax: `requires`, `ensures`, and `invariant` (verification pipeline in progress).

**In progress**
- Full Ora → SIR coverage with strict legality (no silent fallback).
- EVM backend parity for production-grade codegen.
- Formal verification pipeline (requires/ensures/invariants).
- ABI generation and contract metadata.

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
./zig-out/bin/ora <command> <path>
```

See available commands and flags:
```bash
./zig-out/bin/ora --help
```

## Commands (common)

```bash
# Parse and validate a file
./zig-out/bin/ora parse <path>

# Emit MLIR (Ora dialect)
./zig-out/bin/ora emit-mlir <path>

# Emit SIR MLIR (where supported)
./zig-out/bin/ora emit-sir <path>

# Emit ABI (Ora-native)
./zig-out/bin/ora emit-abi <path>

# Emit ABI (Solidity-compatible JSON)
./zig-out/bin/ora emit-abi-solidity <path>

# Run the standard test suite
zig build test
```

Validate fixtures:
```bash
./scripts/validate-examples.sh
```

## Development

Generate the Ora → SIR coverage report:
```bash
python3 scripts/generate_ora_to_sir_coverage.py
```

## Notes

- Asuka is pre-release; breaking changes are expected.
- Ora → SIR parity is the primary milestone for Asuka.
