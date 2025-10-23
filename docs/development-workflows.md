# Development Workflows

Common workflows for Ora development.

## Daily Development

### Making Changes to the Parser

```bash
# 1. Make your changes to src/parser/...
vim src/parser/expression_parser.zig

# 2. Build
zig build

# 3. Test with a simple example
./zig-out/bin/ora parse ora-example/smoke.ora

# 4. Run tests
zig build test

# 5. Validate all examples
./scripts/validate-examples.sh
```

### Adding a New Language Feature

```bash
# 1. Update grammar
vim GRAMMAR.bnf

# 2. Add test fixtures
echo 'contract Test { /* new feature */ }' > tests/fixtures/semantics/valid/new_feature.ora

# 3. Implement in parser
vim src/parser/...

# 4. Add semantic rules if needed
vim src/semantics/...

# 5. Test
zig build test
./zig-out/bin/ora parse tests/fixtures/semantics/valid/new_feature.ora

# 6. Create example
cp tests/fixtures/semantics/valid/new_feature.ora ora-example/

# 7. Validate
./scripts/validate-examples.sh

# 8. Update documentation
vim website/docs/language-basics.md
```

### Adding an Example

```bash
# 1. Create example file
cat > ora-example/my_feature.ora << 'EOF'
contract Example {
    // Demonstrate feature
}
EOF

# 2. Test it parses
./zig-out/bin/ora parse ora-example/my_feature.ora

# 3. Validate all examples
./scripts/validate-examples.sh

# 4. Document in ora-example/README.md
vim ora-example/README.md

# 5. Add to website examples if needed
vim website/docs/examples.md
```

## Before Committing

### Pre-commit Checklist

```bash
# 1. Format code
zig fmt src/

# 2. Build cleanly
zig build

# 3. Pass all tests
zig build test

# 4. Validate examples
./scripts/validate-examples.sh

# 5. Check for linter errors
zig build 2>&1 | grep -i warning

# 6. Review changes
git diff

# 7. Commit with clear message
git add .
git commit -m "feat: Add support for X"
```

## Testing Workflows

### Testing Parser Changes

```bash
# Quick test
./zig-out/bin/ora parse ora-example/smoke.ora

# Test specific fixture
./zig-out/bin/ora parse tests/fixtures/semantics/valid/storage_region_moves.ora

# Test with error cases
./zig-out/bin/ora parse tests/fixtures/semantics/invalid/bad_syntax.ora

# Run parser test suite
zig build test -Dtest-filter="parser"
```

### Testing Examples After Changes

```bash
# Validate all
./scripts/validate-examples.sh

# Test specific example
./zig-out/bin/ora parse ora-example/storage/basic_storage.ora

# Show detailed errors
./zig-out/bin/ora parse ora-example/functions/basic_functions.ora 2>&1
```

### Debugging Failed Examples

```bash
# 1. Find failing example
./scripts/validate-examples.sh | grep "✗"

# 2. Test it directly
./zig-out/bin/ora parse ora-example/functions/basic_functions.ora

# 3. Look at the error
./zig-out/bin/ora parse ora-example/functions/basic_functions.ora 2>&1 | less

# 4. Check the file
cat ora-example/functions/basic_functions.ora

# 5. Check grammar
grep -A 5 "function_declaration" GRAMMAR.bnf

# 6. Fix the example or parser
vim ora-example/functions/basic_functions.ora  # or vim src/parser/...

# 7. Re-validate
./scripts/validate-examples.sh
```

## Documentation Workflows

### Updating Documentation

```bash
# 1. Make changes
vim website/docs/language-basics.md

# 2. Test examples in docs are valid
# Extract code block and test:
sed -n '/```ora/,/```/p' website/docs/language-basics.md | \
  grep -v '```' > /tmp/test.ora
./zig-out/bin/ora parse /tmp/test.ora

# 3. Preview website locally
cd website
npm install
npm run start

# 4. Update timestamps
sed -i 's/October 2025/November 2025/' website/docs/*.md
```

### Keeping Docs in Sync

```bash
# After parser changes
./scripts/validate-examples.sh

# After grammar changes
vim website/docs/specifications/grammar.md

# After feature completion
vim website/docs/language-basics.md
vim website/docs/roadmap-to-asuka.md

# Monthly maintenance
vim docs/DOCUMENTATION_GUIDE.md  # Follow checklist
```

## Release Workflows

### Preparing for Release

```bash
# 1. Ensure clean state
git status
zig build test
./scripts/validate-examples.sh

# 2. Update version numbers
vim build.zig.zon

# 3. Update documentation
vim website/docs/intro.md
vim README.md
vim CHANGELOG.md  # If exists

# 4. Run full validation
zig build clean
zig build -Doptimize=ReleaseFast
zig build test
./scripts/validate-examples.sh

# 5. Tag release
git tag -a v0.1.0-asuka -m "ASUKA Release"
git push origin v0.1.0-asuka
```

## Troubleshooting

### Build Fails

```bash
# Clean and rebuild
rm -rf zig-cache zig-out
zig build

# Check Zig version
zig version  # Should be 0.14.1+

# Check for submodules
git submodule update --init --recursive
```

### Tests Fail

```bash
# Run tests with verbose output
zig build test --summary all

# Run specific test
zig build test -Dtest-filter="lexer"

# Check test allocations
zig build test -Dtest-verbose
```

### Examples Don't Validate

```bash
# See which fail
./scripts/validate-examples.sh

# Test problematic example
./zig-out/bin/ora parse ora-example/failing-example.ora

# Compare with grammar
cat GRAMMAR.bnf | less

# Check if it's a known issue
cat ora-example/README.md
```

## Performance Profiling

### Profile Compilation

```bash
# Build optimized compiler
zig build -Doptimize=ReleaseFast

# Time compilation
time ./zig-out/bin/ora parse ora-example/max_features.ora

# Profile with instruments (macOS)
instruments -t "Time Profiler" ./zig-out/bin/ora parse ora-example/max_features.ora
```

### Memory Profiling

```bash
# Build with debug info
zig build -Doptimize=Debug

# Run with valgrind (Linux)
valgrind --leak-check=full ./zig-out/bin/ora parse ora-example/smoke.ora

# Check allocations (macOS)
leaks --atExit -- ./zig-out/bin/ora parse ora-example/smoke.ora
```

## Useful Commands

```bash
# Find all TODOs in code
grep -r "TODO" src/

# Check code size
cloc src/

# Find files modified recently
find src/ -name "*.zig" -mtime -7

# Show compiler help
./zig-out/bin/ora --help

# Generate AST JSON for inspection
./zig-out/bin/ora -o /tmp ast ora-example/smoke.ora
cat /tmp/smoke.ast.json | jq .
```

---

*For more information, see [CONTRIBUTING.md](../CONTRIBUTING.md)*

