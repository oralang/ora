# Ora

<p align="center">
  <img src="website/static/img/logo-round.png" alt="Ora logo" width="220" />
</p>

Ora is a verification-first smart-contract language and compiler for EVM. The
Asuka v0.2 release focuses on proof-carrying contracts: explicit Result values,
ADTs, SMT reports, ABI lowering, metrics, CFG inspection, and fail-closed
compiler behavior.

> **Asuka v0.2.** The language surface is still evolving, but v0.2 is a release:
> supported examples should compile, unsupported shapes should diagnose, and
> wrong code should not become bytecode.

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

    pub fn withdraw(amount: MinValue<u256, 1>) -> Result<u256, InsufficientBalance> {
        let sender: NonZeroAddress = std.msg.sender();
        let current: u256 = balances[sender];
        if (current < amount) { return Err(InsufficientBalance(amount, current)); }
        balances[sender] = current - amount;
        totalDeposits -= amount;
        return Ok(balances[sender]);
    }

    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }
}
```

This contract uses refinement types (`MinValue`, `NonZeroAddress`), Result
values (`Result<u256, InsufficientBalance>`), specification clauses
(`requires`/`ensures`/`old()`), events (`log`), and explicit storage. The
compiler checks the full surface; the SMT verifier proves the supported
properties below and fails closed when it cannot model a proof soundly.

## Asuka v0.2 highlights

- First-class `Result<T, E>` and error-union values with `Ok`, `Err`, `match`,
  `try`, ABI support, and SMT encoding.
- Unified ADT handling for structs, tuples, enums, Result/error unions, and
  source constructors.
- Z3-backed verification reports with counterexamples, trust labels, vacuity
  checks, degradation reasons, and fail-closed `UNKNOWN` handling.
- Runtime ABI hardening: `@abiEncode`, dynamic public returns, custom-error
  selector reverts, dispatcher decode coverage, and ABI layout unification.
- Comptime expansion for ABI helpers, ADT/Result values, partial folding, and
  bounded loop unrolling.
- Tooling for source-level EVM debugging, LSP features, compiler metrics, SIR
  CFG output, and MLIR/SIR inspection.
- Hardened lowering and artifact gates: unsupported shapes diagnose instead of
  emitting best-effort bytecode.

For the public verifier boundary, see
[`website/docs/compiler/what-ora-proves.md`](website/docs/compiler/what-ora-proves.md).

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
ORA_IMAGE=oralang/ora:v0.2.0 ora build ora-example/apps/erc20.ora
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
./zig-out/bin/ora emit --emit=mlir:ora file.ora           # Ora MLIR
./zig-out/bin/ora emit --emit=mlir:sir file.ora           # SIR MLIR
./zig-out/bin/ora emit --emit=mlir:both file.ora          # Ora + SIR MLIR
./zig-out/bin/ora emit --emit=sir-text file.ora           # Sensei text IR
./zig-out/bin/ora emit --emit=bytecode file.ora           # EVM bytecode
./zig-out/bin/ora emit --emit=ast file.ora                # Parsed AST
./zig-out/bin/ora emit --emit=cfg:sir file.ora            # Control flow graph
./zig-out/bin/ora build file.ora --explain --emit=smt-report # SMT verification report
```

Launch the interactive debugger:
```bash
./zig-out/bin/ora debug ora-example/arithmetic_test.ora --signature 'add(u256,u256)' --arg 7 --arg 9
```

See [`DEBUGGER.md`](DEBUGGER.md) for the debugger workflow, commands, sessions, and current limits.

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

Z3 verification runs by default on `ora build`. Build mode emits artifacts, so
soundness-reducing verification escape hatches are rejected there instead of
producing bytecode.

```bash
./zig-out/bin/ora build file.ora                                # full verification default
./zig-out/bin/ora build file.ora --verify=full --explain --emit=smt-report
./zig-out/bin/ora emit --emit=mlir:ora file.ora --verify=basic  # reduced-trust inspection
./zig-out/bin/ora emit --emit=mlir:ora file.ora --no-verify     # inspection only; not a verified artifact
```

SMT reports expose structured `soundness_losses` and `precision_notes`.
`soundness_loss_cap_exceeded` and `precision_note_cap_exceeded` are truncation
markers, not additional independent findings: they mean the bounded report list
filled and later entries were omitted.

## Documentation

- **[The Ora Little Book](website/docs/book/)** — 20-chapter progressive guide from first contract to production vault
- **[Language reference](website/docs/)** — feature docs: types, regions, error unions, traits, verification, comptime
- **[Compiler Field Guide](website/docs/compiler/field-guide/)** — contributor onboarding (14 chapters)
- **[What Ora Proves](website/docs/compiler/what-ora-proves.md)** — public SMT soundness model and trust boundaries
- **[Formal specification](docs/formal-specs/ora-2.md)** — type system calculus
- **[CLI and config reference](docs/ora-cli-imports-config-reference.md)** — full CLI, import system, and `ora.toml` schema
- **[Examples](ora-example/)** — apps, vault tiers, and feature demos

## Development

```bash
zig build test
zig build gate
```

`zig build gate` is the full local bar. It runs compiler tests, conformance,
EVM tests, MLIR/SIR snapshot checks, negative corpus checks, verifier mutation
checks, and LSP smoke checks. Use `-Dskip-mlir=true` for fast non-MLIR work;
run a full `zig build` when touching `src/mlir/**`.

For contributor workflow details, see
[`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[`Compiler Field Guide`](website/docs/compiler/field-guide/).
