# Contributing to Ora

Thank you for your interest in contributing to the Ora programming language and compiler! This guide will help you get started.

## Quick Start

### Automated Setup (Recommended)

Run our setup script to install all dependencies automatically:

```bash
./setup.sh
```

This script will:
- ✅ Check for Zig 0.15.x
- ✅ Install system dependencies (CMake, Boost, OpenSSL, Clang)
- ✅ Initialize Git submodules (vendor/solidity)
- ✅ Build the compiler
- ✅ Run tests

### Manual Setup

If you prefer to set up manually or the script doesn't work for your platform:

#### 1. Install Zig

Download and install Zig 0.15.x from [ziglang.org/download](https://ziglang.org/download/).

**macOS:**
```bash
brew install zig
```

**Linux (Ubuntu/Debian):**
```bash
snap install zig --classic --beta
```

**Verify installation:**
```bash
zig version  # Should be 0.14.1 or higher
```

#### 2. Install System Dependencies

**macOS:**
```bash
brew update
brew install cmake boost openssl
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    clang \
    libc++-dev \
    libc++abi-dev \
    libboost-all-dev \
    libssl-dev \
    pkg-config \
    git
```

**Windows:**
```powershell
# Using Chocolatey
choco install cmake openssl boost-msvc-14.3

# Or using vcpkg
vcpkg install boost:x64-windows openssl:x64-windows
```

#### 3. Clone and Build

```bash
# Clone the repository
git clone https://github.com/oralang/ora.git
cd ora

# Initialize submodules
git submodule update --init --depth=1 vendor/solidity

# Build the compiler
zig build

# Run tests
zig build test
```

The first build may take 10-30 minutes as it compiles the MLIR and Solidity libraries.

## Development Workflow

### Building

```bash
# Standard build
zig build

# Release build (optimized)
zig build -Doptimize=ReleaseFast

# Clean build
rm -rf .zig-cache zig-out
zig build
```

### Testing

```bash
# Run all tests
zig build test

# Run compiler on an example
./zig-out/bin/ora compile ora-example/smoke.ora

# Test end-to-end compilation
./zig-out/bin/ora compile ora-example/basic_storage.ora
./zig-out/bin/ora --emit-yul ora-example/basic_storage.ora
```

### Code Quality

```bash
# Format code
zig fmt src/

# Check formatting
zig fmt --check src/

# Check for compile errors
zig build
```

## Project Structure

```
Ora/
├── src/                    # Compiler source code
│   ├── ast/               # Abstract Syntax Tree definitions
│   ├── parser/            # Lexer and parser
│   ├── semantics/         # Semantic analysis
│   ├── mlir/              # MLIR lowering
│   ├── lexer.zig          # Lexer implementation
│   ├── main.zig           # CLI entry point
│   └── root.zig           # Library root
├── tests/                  # Unit and integration tests
│   └── compiler_e2e_test.zig  # End-to-end tests
├── ora-example/           # Example Ora programs
├── vendor/                # External dependencies
│   ├── solidity/          # Solidity Yul compiler (submodule)
│   ├── mlir/              # Pre-built MLIR libraries
│   └── llvm-project/      # LLVM/MLIR source (build artifact)
├── docs/                  # Technical documentation
├── build.zig              # Build configuration
└── README.md              # Project overview
```

## What to Work On

### Good First Issues

Perfect for newcomers to the project:

**🐛 Bug Fixes**
- Fix parser error messages (see issues tagged `good-first-issue`)
- Handle edge cases in lexer
- Improve error recovery

**📝 Documentation**
- Add more examples to `ora-example/`
- Improve code comments
- Write tutorial guides
- Fix typos and clarify explanations

**✅ Testing**
- Add test cases to `tests/fixtures/`
- Test edge cases in type checker
- Validate example programs
- Write integration tests

**🔍 Error Messages**
- Make parser errors more descriptive
- Add source location context
- Suggest fixes in error messages
- Improve warning messages

### Intermediate Tasks

For contributors familiar with compilers:

**🔧 Parser Improvements**
- Implement missing language features
- Improve error recovery
- Add syntax sugar
- Optimize parsing performance

**🧮 Type Checker Enhancements**
- Improve type inference
- Add type narrowing
- Better error union handling
- Region transition validation

**🎯 Optimization Passes**
- Dead code elimination
- Constant folding
- Loop optimizations
- MLIR pass improvements

### Advanced Tasks

For experienced compiler developers:

**💾 Code Generation**
- Complete Yul backend
- EVM bytecode generation
- Optimization passes
- Target-specific improvements

**🔬 Formal Verification**
- Implement `requires`/`ensures` checking
- SMT solver integration
- Proof generation
- Verification examples

**🛠️ Tooling**
- LSP (Language Server Protocol)
- Debugger
- Formatter
- IDE integration

## Coding Guidelines

### Zig Style

- Follow standard Zig formatting (use `zig fmt`)
- Use descriptive variable names
- Add comments for non-obvious logic
- Keep functions focused and small

### Memory Management

- Always free allocated memory
- Use `defer` for cleanup
- Prefer arena allocators for AST construction
- Use the testing allocator in tests to catch leaks

### Error Handling

- Return errors, don't panic
- Provide helpful error messages with source locations
- Use the `ErrorHandler` for compiler diagnostics

### Testing

- Add tests for new features
- Write both positive and negative test cases
- Keep tests simple and focused
- Use descriptive test names

Example test:
```zig
test "lexer scans keywords" {
    const allocator = testing.allocator;
    const source = "contract fn pub storage";
    
    const Lexer = @import("ora").Lexer;
    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();
    
    const tokens = try lexer.scanTokens();
    defer allocator.free(tokens);
    
    try testing.expect(tokens.len >= 5);
}
```

## Submitting Changes

### Before You Submit

1. ✅ Code compiles: `zig build`
2. ✅ Tests pass: `zig build test`
3. ✅ Code is formatted: `zig fmt src/`
4. ✅ No compiler warnings

### Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Make your changes** with clear, focused commits
4. **Test thoroughly**
5. **Push** to your fork: `git push origin feature/my-feature`
6. **Open a Pull Request** with:
   - Clear description of what changed
   - Why the change is needed
   - Any related issues

### Commit Messages

Use clear, descriptive commit messages:

```
✅ Good:
- "Add support for nested struct literals"
- "Fix: Prevent panic in parser for empty input"
- "Optimize: Cache type resolution results"

❌ Bad:
- "fix stuff"
- "wip"
- "update code"
```

## Finding Your First Issue

1. **Browse Issues**: Check [GitHub Issues](https://github.com/oralang/ora/issues)
   - Filter by `good-first-issue` label
   - Filter by `help-wanted` label
   - Look for `documentation` tasks

2. **Comment on Issue**: Express interest and ask questions
   - "I'd like to work on this. Any pointers?"
   - Maintainers will guide you

3. **Start Small**: Don't try to fix everything at once
   - One feature or bug fix per PR
   - Get feedback early and often

## Development Tips for New Contributors

### Understanding the Codebase

**Start Here:**
1. Read `README.md` - project overview
2. Try `./zig-out/bin/ora parse ora-example/smoke.ora`
3. Read `GRAMMAR.bnf` - language syntax
4. Explore `src/` structure
5. Run tests: `zig build test`

**Key Files:**
- `src/lexer.zig` - Tokenization
- `src/parser/` - Syntax parsing
- `src/ast/` - Abstract syntax tree
- `src/semantics/` - Type checking and validation
- `src/mlir/` - MLIR lowering

**Common Tasks:**

*Adding a test case:*
```bash
# 1. Create test file
echo 'contract Test { storage var x: u256; }' > tests/fixtures/semantics/valid/my_test.ora

# 2. Run tests
zig build test

# 3. Verify parsing
./zig-out/bin/ora parse tests/fixtures/semantics/valid/my_test.ora
```

*Adding an example:*
```bash
# 1. Create example
cat > ora-example/my_feature.ora << 'EOF'
contract Example {
    // Your example code
}
EOF

# 2. Validate
./zig-out/bin/ora parse ora-example/my_feature.ora

# 3. Run validation suite
./scripts/validate-examples.sh
```

*Improving error messages:*
```bash
# 1. Find error site (grep for error text)
grep -r "Expected token" src/

# 2. Improve message
# 3. Test with invalid input
./zig-out/bin/ora parse tests/fixtures/parser/invalid/bad_syntax.ora

# 4. Run test suite
zig build test
```

## Getting Help

- 💬 **Questions**: [GitHub Discussions](https://github.com/oralang/ora/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/oralang/ora/issues)
- 📖 **Documentation**: See `docs/` folder
- 📚 **Guides**: Check `docs/DOCUMENTATION_GUIDE.md` for maintaining docs

## Development Tips

### Fast Iteration

For rapid development, build just the part you're working on:

```bash
# Build only the lexer
zig build-lib src/lexer.zig

# Build only the parser
zig build-lib src/parser.zig
```

### Debugging

Use Zig's built-in debugging:

```zig
const std = @import("std");

// Print debug info
std.debug.print("Value: {}\n", .{value});

// Assert conditions
std.debug.assert(value > 0);
```

### MLIR Development

To work on MLIR lowering:

```bash
# Generate MLIR output
./zig-out/bin/ora --emit-mlir contract.ora

# Validate MLIR
./validate_mlir.sh contract.ora
```

### Performance Profiling

```bash
# Build with release optimizations
zig build -Doptimize=ReleaseFast

# Profile compilation time
time ./zig-out/bin/ora compile large_contract.ora
```

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the code, not the person
- Assume good intentions

## License

By contributing to Ora, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to Ora! 🎉

