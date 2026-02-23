---
sidebar_position: 4
---

# Imports and Modules

Ora supports splitting code across files using namespace-qualified imports.
Imported declarations are always accessed through the alias — they are never
injected into local scope.

> This documents the current Asuka-branch behavior.

## Basic usage

```ora
// math.ora
pub fn add(a: u256, b: u256) -> u256 {
    return a + b;
}
```

```ora
// main.ora
const math = @import("./math.ora");

contract Calculator {
    pub fn run() -> u256 {
        return math.add(40, 2);
    }
}
```

`math.add` stays qualified throughout compilation. There is no flattening or
renaming — the alias `math` is a namespace boundary.

## Syntax

```ora
const <alias> = @import("<specifier>");
```

- The alias is **required**. Omitting it is a compile error (`ImportAliasRequired`).
- One import per `const` declaration.
- Imports must appear at the top level (outside contracts).

## Import specifiers

### Relative imports

```ora
const math = @import("./math.ora");
const utils = @import("../lib/utils.ora");
```

- Must end with `.ora`. Missing extension is an error.

### Package imports

```ora
const math = @import("acme/math");
```

- Format is `package/module`.
- `.ora` is auto-appended when omitted.
- Resolved via `include_paths` in `ora.toml` or CLI-provided include roots.

### Built-in std

```ora
const stdlib = @import("std");
```

The `std` specifier is recognized by the compiler and provides built-in
functions and constants (e.g. `std.transaction.sender`).

## Namespace semantics

Imported members are accessed with dot-qualified syntax:

```ora
const math = @import("./math.ora");

contract Example {
    pub fn compute() -> u256 {
        // Correct: qualified access
        return math.add(1, 2);
    }
}
```

This means:

- **No local shadowing.** A local variable named `add` does not conflict with `math.add`.
- **No cross-module collisions.** Two modules can export the same name (`a.helper()` and `b.helper()` coexist).
- **Explicit dependencies.** Every external reference is prefixed with its origin.

## What can be imported

For non-entry modules, the following top-level declarations are allowed in v1:

| Declaration | Allowed |
|-------------|---------|
| `pub fn` | Yes |
| `struct` | Yes |
| `bitfield` | Yes |
| `const` (import) | Yes |
| comptime-only `fn` | Yes |
| `contract` | No (v2) |
| `enum` | No (v2) |
| `log` / `error` | No (v2) |
| top-level `var` | No |
| `init` | No |

## Nested imports

Imported modules can themselves import other modules:

```ora
// math.ora
pub fn add(a: u256, b: u256) -> u256 { return a + b; }
```

```ora
// ops.ora
const math = @import("./math.ora");

pub fn twice(x: u256) -> u256 {
    return math.add(x, x);
}
```

```ora
// Main.ora
const ops = @import("./ops.ora");

contract Main {
    pub fn run() -> u256 {
        return ops.twice(21);
    }
}
```

The compiler resolves dependencies in topological order with cycle detection.

## Error guarantees

The import resolver enforces:

- **Cycle detection** — circular imports are a compile error.
- **Duplicate alias protection** — the same alias pointing to two different modules in one file is an error.
- **Same alias, same target** — importing the same module twice with the same alias is deduplicated (not an error).
- **Target not found** — missing files produce a clear diagnostic.

## Project configuration with `ora.toml`

Scaffold a new project with `ora init`:

```bash
ora init my-project
```

This generates an `ora.toml`, `contracts/main.ora`, and a `README.md`.

Multi-file projects use `ora.toml` for target and path configuration:

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

### Key fields

| Field | Required | Description |
|-------|----------|-------------|
| `schema_version` | Yes | Must be `"0.1"` |
| `compiler.output_dir` | No | Default output directory |
| `targets[].name` | Yes | Target name |
| `targets[].root` | Yes | Entry file path |
| `targets[].kind` | No | `"contract"` (default) or `"library"` |
| `targets[].include_paths` | No | Directories for package import resolution |
| `targets[].init_args` | No | Constructor arguments as `"name=value"` pairs |

### Config discovery

The compiler searches for `ora.toml` (or `Ora.toml`) starting from the entry
file's directory and walking upward.

### Build with config

```bash
# Build all targets defined in ora.toml
ora build

# Build a specific file (auto-discovers config)
ora build contracts/main.ora
```

See the [CLI, Imports, and ora.toml Reference](https://github.com/oralang/Ora/blob/main/docs/ora-cli-imports-config-reference.md) for the full schema and CLI flag reference.
