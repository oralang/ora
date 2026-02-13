---
sidebar_position: 5
---

# Code Formatter (`ora fmt`)

Ora includes a canonical, deterministic code formatter (`ora fmt`) in the spirit of `zig fmt` and `gofmt`. It formats Ora source code into a single standard style to improve readability, reduce diffs, and enable reliable tooling.

## Quick Start

```bash
# Format a file in-place
ora fmt contract.ora

# Format multiple files
ora fmt src/*.ora

# Format a directory recursively
ora fmt src/

# Check if code is formatted (useful for CI)
ora fmt --check contract.ora

# Show diff of formatting changes
ora fmt --diff contract.ora

# Output formatted code to stdout
ora fmt --stdout contract.ora
```

## Design Principles

- **Deterministic:** Same input + same formatter version => same output
- **Idempotent:** `fmt(fmt(x)) == fmt(x)`
- **Semantics-preserving:** Formatting does not change program meaning
- **AST-driven:** Formatting is derived from the parsed AST, not regex transforms
- **Comment-preserving:** Comments are preserved without rewriting comment text
- **Tool-friendly:** Designed for editor-on-save, pre-commit hooks, CI checks

## Command Options

| Flag | Description |
|------|-------------|
| `--check` | Don't write; exit 1 if any file would change |
| `--diff` | Don't write; print diff for each file that would change |
| `--stdout` | Print formatted output to stdout (requires exactly one file input) |
| `--width <n>` | Preferred line width (default: 100) |
| `--help` | Show help |

**Rules:**
- `--stdout` rejects multiple inputs and directories
- `--check`, `--diff`, and `--stdout` are mutually exclusive

## Exit Codes

- `0`: Success (and for `--check`: already formatted)
- `1`: `--check` and at least one file would change
- `2`: Parse error, IO error, invalid invocation, internal formatter error

## Configuration

`ora fmt` can read formatting defaults from the project's `ora.toml` if present:

```toml
[format]
width = 100
```

**Precedence:**
1. CLI flags
2. `ora.toml` `[format]`
3. Built-in defaults

## Formatting Rules

### Indentation
- Uses **4 spaces** (no tabs)
- Fixed indent width in v0.1

### Braces
- Opening `{` on the same line for: `contract`, `struct`, `enum`, `fn`, and control-flow
- Closing `}` on its own line, aligned to the construct that opened it
- One space between keywords and `{`: `if (cond) {`, `fn f() {`

### Spacing
- Binary operators: one space on both sides
- Unary operators: no space to operand (`-x`, `!flag`)
- Type annotations: `name: Type` (no space before `:`, one after)
- Commas: followed by one space, no space before (`a, b`)
- No spaces inside `()` or `[]`: `f(a, b)`, `arr[i]`

### Blank Lines
- Exactly one blank line between top-level declarations
- No blank line at start or end of blocks
- No blank line immediately after `{` or before `}`

### Multiline Lists
For multiline lists (function parameters, arguments, etc.):
- One item per line
- Items indented one level from opening delimiter
- **Trailing comma required**

Example:
```ora
fn complex(
    a: u256,
    b: address,
    c: map<address, u256>,
) -> bool {
}
```

### Comments
- All comments are preserved
- Comment text is not rewritten in v0.1
- Comments maintain stable positions relative to code

## Example

### Before
```ora
contract Token{
storage var totalSupply:u256;
storage var balances:map<address,u256>;
pub fn init(initialSupply:u256){
totalSupply=initialSupply;
}
pub fn transfer(recipient:address,amount:u256)->bool{
var sender:address=std.msg.sender();
var balance:u256=balances[sender];
if(balance<amount){
return false;
}
balances[sender]=balance-amount;
balances[recipient]=balances[recipient]+amount;
return true;
}
}
```

### After
```ora
contract Token {
    storage var totalSupply: u256;
    storage var balances: map<address, u256>;

    pub fn init(initialSupply: u256) {
        totalSupply = initialSupply;
    }

    pub fn transfer(recipient: address, amount: u256) -> bool {
        var sender: address = std.msg.sender();
        var balance: u256 = balances[sender];
        if (balance < amount) {
            return false;
        }
        balances[sender] = balance - amount;
        balances[recipient] = balances[recipient] + amount;
        return true;
    }
}
```

## Integration

### Editor Integration
Configure your editor to run `ora fmt` on save:

**VS Code** (`.vscode/settings.json`):
```json
{
  "[ora]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ora.fmt"
  }
}
```

### Pre-commit Hook
Add to `.git/hooks/pre-commit`:
```bash
#!/bin/sh
ora fmt --check src/ || (echo "Code not formatted. Run 'ora fmt src/'"; exit 1)
```

### CI/CD
```yaml
# GitHub Actions example
- name: Check formatting
  run: ora fmt --check src/
```

## Full Specification

For complete details, see the [Ora Formatter Specification v0.1.0](https://github.com/oralang/Ora/blob/main/docs/ora-fmt-specification.md).

