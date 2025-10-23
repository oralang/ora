# Ora

> **Pre-ASUKA Alpha** | Early Development | Contributors Welcome

An experimental smart-contract language and compiler targeting Yul/EVM. Ora emphasizes explicit semantics (types, memory regions, error unions) and a clean compilation pipeline suitable for analysis and verification.

## Status

🚧 **Pre-Release Alpha**: Ora is under active development toward the first ASUKA release. The core compiler works, but syntax and semantics are evolving.

**What Works Now:**
- ✅ Full lexer and parser (23/29 examples pass)
- ✅ Type checking and semantic analysis
- ✅ Storage, memory, and transient storage regions
- ✅ Error unions and basic error handling
- ✅ Switch statements (expression and statement forms)
- ✅ Structs, enums, and custom types
- ✅ MLIR lowering for optimization

**In Development:**
- 🚧 Advanced for-loop syntax
- 🚧 Complete Yul backend
- 🚧 Standard library
- 🚧 Formal verification (requires/ensures)

## Highlights

- Unified type system with explicit error unions: `!T | E1 | E2`
- Memory regions: `storage`, `memory`, `tstore`, `stack` (default) with region transition checks
- Clear, modern syntax: `->` for returns, `requires`/`ensures`, `switch` (statement/expression)
- Restricted lvalues and explicit move semantics

## Quick Start

**Prerequisites:** Zig 0.15.x, CMake, Git

```bash
git clone https://github.com/oralang/Ora.git
cd Ora
./setup.sh  # Automated setup (installs deps, builds, tests)

# Or manually:
git submodule update --init --depth=1 vendor/solidity
zig build && zig build test
```

**Try it:**
```bash
# Parse a single file
./zig-out/bin/ora parse ora-example/smoke.ora

# Validate all examples
./scripts/validate-examples.sh
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
    requires(a <= 1_000 && b <= 1_000)
    ensures(result >= a && result >= b)
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
src/                    # Compiler source code (Zig)
  ast/                  # AST and type information
  parser/               # Lexer and parser
  semantics/            # Semantic analysis
  mlir/                 # MLIR lowering system
tests/                  # Test fixtures and suites
  fixtures/             # Test input files
    semantics/valid/    # Valid Ora programs
    semantics/invalid/  # Programs with expected errors
ora-example/            # Example Ora programs (validated)
docs/                   # Documentation and guides
  tech-work/            # Internal AI-assisted development docs
website/                # Documentation website (Docusaurus)
  docs/                 # User-facing documentation
scripts/                # Utility scripts (validation, etc.)
vendor/                 # External dependencies
```

## Testing

### Running Tests

```bash
# Run all compiler tests
zig build test

# Validate all example files
./scripts/validate-examples.sh

# Run specific test suite
zig build test -- <test-name>
```

### Test Organization

- **Unit tests**: Embedded in source files (`src/**/*.zig`)
- **Fixture tests**: `tests/fixtures/semantics/{valid,invalid}`
- **Example validation**: `ora-example/` (validated by script)
- **Integration tests**: `tests/compiler_e2e_test.zig`

## Advanced Features

### MLIR Integration

Ora includes MLIR (Multi-Level Intermediate Representation) lowering for advanced optimization:

```bash
./zig-out/bin/ora compile --emit-mlir contract.ora
./zig-out/bin/ora compile --mlir-passes="canonicalize,cse" contract.ora
```

See [website/docs/specifications/mlir.md](website/docs/specifications/mlir.md) for details.

## CLI Usage

```bash
# Lex (tokenize) a file
./zig-out/bin/ora lex contract.ora

# Parse and show AST
./zig-out/bin/ora parse contract.ora

# Generate AST JSON
./zig-out/bin/ora -o build ast contract.ora

# Compile (when Yul backend is complete)
./zig-out/bin/ora compile contract.ora
```

See [website/docs/specifications/api.md](website/docs/specifications/api.md) for complete CLI reference.

## Roadmap to ASUKA Release

The first stable release (ASUKA) will include:
- ✅ Complete lexer/parser pipeline
- ✅ Type checking and semantic analysis
- 🚧 Full Yul code generation
- 🚧 Basic standard library
- 📋 Comprehensive documentation
- 📋 Language specification v1.0

See [docs/roadmap-to-asuka.md](website/docs/roadmap-to-asuka.md) for details.

## Contributing

**We welcome contributors!** Ora is in active development and there are many ways to help:

- 🐛 **Bug Reports**: File issues for bugs or unexpected behavior
- 📝 **Documentation**: Improve examples, guides, or specifications
- ✅ **Testing**: Add test cases or validate examples
- 🔧 **Compiler Development**: Implement features, optimize passes, improve errors
- 💡 **Language Design**: Participate in discussions about syntax and semantics

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and development guidelines.

**Good First Issues:**
- Add more test cases to `tests/fixtures/`
- Improve error messages in parser
- Add examples to `ora-example/`
- Update documentation for current features

## Community

- **GitHub**: [oralang/Ora](https://github.com/oralang/Ora)
- **Issues**: [Report bugs](https://github.com/oralang/Ora/issues)
- **Discussions**: [Join the conversation](https://github.com/oralang/Ora/discussions)
- **Website**: [ora-lang.org](https://ora-lang.org)

## License

See `LICENSE` for details.