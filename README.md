# Ora

An experimental (yet) smart-contract language and compiler targeting Yul/EVM. Ora emphasizes explicit semantics (types, memory regions, error unions) and a clean compilation pipeline suitable for analysis and verification.

## Status

- Not production-ready; syntax and semantics are evolving.
- CI builds and tests cover lexer, parser, early semantics, and Yul codegen stubs.

## Highlights

- Unified type system with explicit error unions: `!T | E1 | E2`
- Memory regions: `storage`, `memory`, `tstore`, `stack` (default) with region transition checks
- Clear, modern syntax: `->` for returns, `requires`/`ensures`, `switch` (statement/expression)
- Restricted lvalues and explicit move semantics

## Quick Start

- Prerequisites:
  - Zig 0.14.1+
  - CMake (for vendor Solidity libs)
  - Git (for submodules)

```bash
git clone https://github.com/oralang/Ora.git
cd Ora
git submodule update --init --recursive

zig build
zig build test
```

## Language Snapshot

### Contracts and declarations

```ora
contract MyContract {
    storage var balance: u256;
    immutable owner: address;
    storage const MAX_SUPPLY: u256 = 1_000_000;

    pub fn set(new_value: u256) {
        balance = new_value;
    }

    pub fn get() -> u256 {
        return balance;
    }
}
```

- Variables: `[memory_region] [var|let|const|immutable] name [: Type] [= expr];`
- Regions: `storage`, `memory`, `tstore`, default is `stack` (no qualifier).

### Types

- Primitives: `u8..u256`, `i8..i256`, `bool`, `address`, `string`, `bytes`, `void`
- Arrays/slices: `[T; N]`, `slice[T]`
- Mappings: `map[K, V]`, `doublemap[K1, K2, V]`
- Anonymous struct literals: `.{ .field = value, ... }`

### Error unions

```ora
error InsufficientBalance;
error InvalidAddress;

fn transfer(to: address, amount: u256) -> !u256 | InsufficientBalance | InvalidAddress {
    if (amount == 0) return error.InvalidAddress;
    if (balance < amount) return error.InsufficientBalance;
    balance -= amount;
    return balance;
}
```

- `!T | E1 | E2` means “returns T or one of the error tags E1/E2”.
- `try expr` unwraps success type when `expr` is an error union.

### Regions and transitions

- Reads from `storage` produce values in `stack`.
- Writes to `storage` require a `storage` lvalue target.
- `storage <- storage` bulk copies are disallowed; use element/field assignments.

Examples:

```ora
storage var s: u32;

fn f() -> void {
    let x: u32 = s;       // storage -> stack read: OK
    s = x;                // stack -> storage write: OK
}
```

### Control flow and expressions

- Return: `return expr;` or `return;` (only in `-> void`).
- If/while/for/switch, expression statements, assignments, compound assignments.
- Switch (statement or expression):

```ora
fn classify(x: u32) -> u32 {
    switch (true) {
        x == 0 => 0,
        x < 10 => 1,
        else   => 2,
    }
}
```

### Requires / Ensures (syntax defined)

```ora
fn sum(a: u32, b: u32) -> u32
    requires a <= 1_000 and b <= 1_000
    ensures  result >= a and result >= b
{
    return a + b;
}
```

### Move statement

```ora
// move amount from source to dest;
move amount from balances[sender] to balances[receiver];
```

## Grammar

- Authoritative specs:
  - `GRAMMAR.bnf`
  - `GRAMMAR.ebnf`
- Notable decisions:
  - `->` for return types
  - Unified variable declarations
  - LHS restricted to lvalues
  - Clear array/slice distinction: `[T; N]` vs `slice[T]`

## Semantics

- Phase 1: Symbol collection (top-level, contracts, function scopes, locals)
- Phase 2: Type checks and validations:
  - Return type compatibility and error unions
  - Region transition rules
  - Assignment and mutability checks
  - Switch typing (in progress)
  - Unknown-identifier walker (temporarily disabled by flag; will be re-enabled after scope hardening)

## Project Structure

```
src/            // compiler sources (Zig)
  ast/          // AST, type info
  parser/       // lexer/parser
  semantics/    // semantic analyzers
  code-bin/     // extra tools (IR/optimizer)
tests/          // fixtures and test suites
docs/           // specifications and notes
vendor/solidity // vendor libs
```

## Testing

- Run all suites: `zig build test`
- Fixture-based semantics tests live under `tests/fixtures/semantics/{valid,invalid}`.

## CLI usage

Use the installed executable:

- Lex: dump tokens
```bash
./zig-out/bin/ora lex tests/fixtures/semantics/valid/storage_region_moves.ora
```
Example output (first lines):
```
Lexing tests/fixtures/semantics/valid/storage_region_moves.ora
==================================================
Generated 44 tokens

[  0] Token{ .type = Storage, .lexeme = "storage", .range = 1:1-1:8 (0:7) }
[  1] Token{ .type = Var, .lexeme = "var", .range = 1:9-1:12 (8:11) }
[  2] Token{ .type = Identifier, .lexeme = "s", .range = 1:13-1:14 (12:13) }
[  3] Token{ .type = Colon, .lexeme = ":", .range = 1:14-1:15 (13:14) }
[  4] Token{ .type = U32, .lexeme = "u32", .range = 1:16-1:19 (15:18) }
...
```

- Parse: lex + parse and print AST summary
```bash
./zig-out/bin/ora parse tests/fixtures/semantics/valid/storage_region_moves.ora
```
Example output:
```
Parsing tests/fixtures/semantics/valid/storage_region_moves.ora
==================================================
Lexed 44 tokens
Generated 2 AST nodes

[0] Variable Storagevar 's'
[1] Function 'f' (0 params)
```

- AST JSON: generate and save AST to file
```bash
./zig-out/bin/ora -o build ast tests/fixtures/semantics/valid/storage_region_moves.ora
```
Example output:
```
Generating AST for tests/fixtures/semantics/valid/storage_region_moves.ora
==================================================
Generated 2 AST nodes
AST saved to build/storage_region_moves.ast.json
```

Notes:
- Commands shown above are verified against the current CLI (`src/main.zig`).
- `parse` prints a concise AST summary; CST building is internal and not dumped by default.

## Roadmap (abridged)

- Complete call typing and operator typing
- Re-enable and harden unknown-identifier walker
- Switch typing and exhaustiveness
- Region/immutability refinements
- Improved diagnostics and IDs
- Yul codegen completion and validation

## Contributing

- File issues for bugs and proposals
- PRs are welcome for tests, docs, analyzers, and parser work

## License

- See `LICENSE` for details.