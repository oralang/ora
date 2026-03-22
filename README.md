# Ora

<p align="center">
  <img src="website/static/img/logo-round.png" alt="Ora logo" width="220" />
</p>

Ora is a smart-contract language and compiler for EVM with explicit semantics, verification-aware IR, and a strict MLIR-based pipeline.

> **Pre-release (Asuka track).** Breaking changes are expected. Not production-ready.

## What Ora does

```ora
error InsufficientBalance(required: u256, available: u256);

comptime const std = @import("std");

contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;

    log Deposit(account: address, amount: u256);

    pub fn deposit(amount: MinValue<u256, 1>)
        requires(totalDeposits <= std.constants.U256_MAX - amount)
        ensures(totalDeposits == old(totalDeposits) + amount)
    {
        let sender: NonZeroAddress = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
        log Deposit(sender, amount);
    }

    pub fn withdraw(amount: MinValue<u256, 1>) -> !bool | InsufficientBalance {
        let sender: NonZeroAddress = std.msg.sender();
        let current: u256 = balances[sender];
        if (current < amount) { return InsufficientBalance(amount, current); }
        balances[sender] = current - amount;
        totalDeposits -= amount;
        return true;
    }

    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }
}
```

This contract uses refinement types (`MinValue`, `NonZeroAddress`), error unions (`!bool | InsufficientBalance`), specification clauses (`requires`/`ensures`/`old()`), events (`log`), and explicit storage regions — all checked by the compiler and Z3 SMT solver.

## Language features

| Area | Features |
|------|----------|
| **Types** | `u8`–`u256`, `i8`–`i256`, `bool`, `address`, `string`, `bytes`; structs, enums, tuples, arrays, maps, anonymous structs, bitfields |
| **Refinement types** | `NonZero`, `NonZeroAddress`, `MinValue`, `MaxValue`, `InRange`, `BasisPoints`, `Scaled`; compile-time proof or runtime guard |
| **Error handling** | Error unions (`!T \| E1 \| E2`), `try`/`catch`, error propagation, `errors` clause |
| **Regions** | Explicit data location: `storage`, `memory`, `calldata`, `transient`; compile-time coercion checks |
| **Control flow** | `if`/`else`, `while`, `for` (ranges, iterators), `switch` (expressions/statements, range patterns, `else`), labeled blocks, `break`/`continue` |
| **Abstraction** | Generics (`comptime T: type`), traits and `impl`, extern traits (`call`/`staticcall`), comptime evaluation |
| **Verification** | Z3 SMT: `requires`/`ensures`/`invariant`/`assume`/`assert`, `old()`, `ghost` state, `forall`/`exists`, path-sensitive reasoning |
| **Events** | `log` declarations with `indexed` fields |
| **Safety** | `@lock`/`@unlock` reentrancy guards, overflow builtins (`@addWithOverflow`, etc.), wrapping arithmetic (`+%`, `*%`) |
| **Imports** | `@import("./path.ora")`, namespace-qualified access, `ora.toml` project config |

## Installation

**Prerequisites:** Zig 0.15.x, CMake, Git, Z3, MLIR

```bash
git clone https://github.com/oralang/Ora.git
cd Ora
./setup.sh
zig build
```

Run tests:
```bash
zig build test
```

## Docker

```bash
docker pull oralang/ora:latest
docker run --rm oralang/ora:latest --help
```

Run against local files:
```bash
docker run --rm -it \
  -u "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  -w /work \
  oralang/ora:latest build ora-example/apps/erc20.ora
```

Install the Docker launcher so `ora` works like a native command:
```bash
chmod +x scripts/ora-docker scripts/install-ora-docker.sh
./scripts/install-ora-docker.sh
```

Use a specific image tag:
```bash
ORA_IMAGE=oralang/ora:v0.1.0 ora build ora-example/apps/erc20.ora
```

### Build image locally

```bash
docker build -t oralang/ora:local .
docker run --rm oralang/ora:local --help
```

## Using the compiler

Scaffold a new project:
```bash
./zig-out/bin/ora init my-project
```

Build a contract:
```bash
./zig-out/bin/ora build contracts/main.ora
```

Format source:
```bash
./zig-out/bin/ora fmt file.ora          # single file
./zig-out/bin/ora fmt contracts/        # directory (recursive)
./zig-out/bin/ora fmt --check file.ora  # CI check
```

Emit intermediate representations:
```bash
./zig-out/bin/ora emit --emit-mlir file.ora              # Ora MLIR (default)
./zig-out/bin/ora emit --emit-mlir=sir file.ora           # SIR MLIR
./zig-out/bin/ora emit --emit-mlir=both file.ora          # Ora + SIR MLIR
./zig-out/bin/ora emit --emit-sir-text file.ora           # Sensei text IR
./zig-out/bin/ora emit --emit-bytecode file.ora           # EVM bytecode
./zig-out/bin/ora emit --emit-ast file.ora                # Parsed AST
./zig-out/bin/ora emit --emit-cfg file.ora                # Control flow graph
./zig-out/bin/ora emit --emit-smt-report file.ora         # SMT verification report
```

### Build artifacts

`ora build` writes to `artifacts/<name>/`:

```
artifacts/<name>/abi/<name>.abi.json          # Ora ABI
artifacts/<name>/abi/<name>.abi.sol.json       # Solidity-compatible ABI
artifacts/<name>/abi/<name>.abi.extras.json    # Extended ABI metadata
artifacts/<name>/bin/<name>.hex               # EVM bytecode
artifacts/<name>/sir/<name>.sir               # Sensei IR
artifacts/<name>/verify/<name>.smt.report.md  # SMT report (markdown)
artifacts/<name>/verify/<name>.smt.report.json # SMT report (JSON)
```

## Imports and multi-file projects

```ora
comptime const math = @import("./math.ora");

contract Calculator {
    pub fn run() -> u256 {
        return math.add(40, 2);
    }
}
```

Imported members are always accessed through the alias (`math.add`); they are never injected into local scope.

### ora.toml

```toml
schema_version = "0.1"

[compiler]
output_dir = "./artifacts"

[[targets]]
name = "Main"
kind = "contract"
root = "contracts/main.ora"
include_paths = ["contracts", "lib"]
```

See [`docs/ora-cli-imports-config-reference.md`](docs/ora-cli-imports-config-reference.md) for the full config schema.

## Verification

Z3 verification runs by default on `ora build`. Control it with:

```bash
./zig-out/bin/ora build --verify file.ora          # default
./zig-out/bin/ora build --verify=full file.ora      # include untagged asserts
./zig-out/bin/ora build --no-verify file.ora        # skip verification
```

Environment variables for tuning:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORA_Z3_TIMEOUT_MS` | `60000` | Per-query timeout (ms) |
| `ORA_Z3_PARALLEL` | `1` | Parallel query execution |
| `ORA_Z3_WORKERS` | CPU count | Worker thread count |
| `ORA_VERIFY_MODE` | `full` | `basic` or `full` |
| `ORA_VERIFY_CALLS` | `1` | Interprocedural call summaries |
| `ORA_VERIFY_STATE` | `1` | Storage state tracking |
| `ORA_VERIFY_STATS` | `0` | Print query statistics |
| `ORA_Z3_DEBUG` | unset | Verbose Z3 debug output |

## Advanced MLIR controls

```bash
# Custom pass pipeline
./zig-out/bin/ora emit --mlir-pass-pipeline "builtin.module(canonicalize,cse)" --emit-mlir=ora file.ora

# Pipeline diagnostics
--mlir-verify-each-pass          # verify after each pass
--mlir-pass-timing               # pass timing
--mlir-pass-statistics           # operation count stats
--mlir-print-ir=before|after|all # print IR at stages
--mlir-print-ir-pass <filter>    # filter by stage name
--mlir-crash-reproducer <path>   # save MLIR on failure
--mlir-print-op-on-diagnostic    # print module on diagnostic
```

Stage names: `lowering`, `custom-pipeline`, `canonicalize`, `ora-to-sir`, `sir-legalize`.

## Documentation

- **[The Ora Little Book](website/docs/book/)** — 20-chapter progressive guide from first contract to production vault
- **[Language reference](website/docs/)** — feature docs: types, regions, error unions, traits, verification, comptime
- **[Compiler Field Guide](website/docs/compiler/field-guide/)** — contributor onboarding (14 chapters)
- **[Formal specification](docs/formal-specs/ora-2.md)** — type system calculus
- **[CLI and config reference](docs/ora-cli-imports-config-reference.md)** — full CLI, import system, and `ora.toml` schema
- **[Examples](ora-example/)** — apps, vault tiers, and feature demos

## Development

Run tests:
```bash
zig build test
```

Validate examples:
```bash
./scripts/validate-examples.sh
```

End-to-end CLI checks:
```bash
./scripts/run-cli-command-checks.sh
```

Generate Ora-to-SIR coverage report:
```bash
python3 scripts/generate_ora_to_sir_coverage.py
```
