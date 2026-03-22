---
title: "Chapter 19: Multi-File Projects"
description: Imports, modules, and ora.toml configuration.
sidebar_position: 19
---

# Multi-File Projects

Real contracts span multiple files. Ora supports file-level modules with namespace-qualified imports.

## Importing files

```ora
comptime const math = @import("./math.ora");

contract Calculator {
    pub fn run() -> u256 {
        return math.add(40, 2);
    }
}
```

`@import("./math.ora")` loads `math.ora` from the same directory. All public items in the imported file are accessible through the `math` namespace.

## Project configuration: ora.toml

Every Ora project has an `ora.toml` at its root:

```toml
schema_version = "0.1"

[compiler]
output_dir = "./artifacts"

[[targets]]
name = "Main"
kind = "contract"
root = "contracts/main.ora"
```

- `schema_version` — config format version (currently `"0.1"`)
- `[compiler].output_dir` — where build artifacts go
- `[[targets]]` — one or more compilation targets, each with a `name`, `kind` (`"contract"` or `"library"`), and `root` source file

### Constructor arguments

If your contract has a `pub fn init(...)` constructor, pass initial values via `init_args`:

```toml
schema_version = "0.1"

[compiler]
output_dir = "./artifacts"
init_args = ["initial_counter=0"]

[[targets]]
name = "Main"
kind = "contract"
root = "contracts/main.ora"
```

Create a project with `ora init`:

```bash
ora init my-vault
```

This generates:
```
my-vault/
├── ora.toml
├── contracts/
│   └── main.ora
└── README.md
```

## Project structure

A typical Ora project:

```
my-vault/
├── ora.toml
├── contracts/
│   ├── main.ora          # Entry point
│   ├── vault.ora         # Vault logic
│   ├── errors.ora        # Error declarations
│   └── types.ora         # Shared types
└── tests/
    └── vault_test.ora
```

## Splitting the vault

**errors.ora:**
```ora
error InsufficientBalance(required: u256, available: u256);
error ZeroAmount;
```

**types.ora:**
```ora
struct DepositRecord {
    amount: u256;
    timestamp: u256;
}

enum VaultStatus : u8 {
    Active = 0,
    Paused = 1,
    Closed = 2
}
```

**vault.ora:**
```ora
comptime const std = @import("std");
comptime const errors = @import("./errors.ora");
comptime const types = @import("./types.ora");

contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;
    storage var status: types.VaultStatus;

    pub fn deposit(amount: MinValue<u256, 1>) -> !bool | errors.ZeroAmount {
        if (amount == 0) { return errors.ZeroAmount; }
        let sender: address = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
        return true;
    }
}
```

Imported names are always qualified: `errors.ZeroAmount`, `types.VaultStatus`. No implicit flattening.

## Building

```bash
ora build contracts/vault.ora
```

The compiler resolves imports relative to the source file. The `ora.toml` configures the entry point and output directory for the full project build.

## Further reading

- [Imports and Modules](../imports) — full import reference and resolution rules
