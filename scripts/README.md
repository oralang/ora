# Ora Scripts

Utility scripts for Ora development and validation.

## Available Scripts

### `validate-examples.sh`

Validates all Ora example files to ensure they compile with the current compiler.

**Purpose**: Automatically test that all example programs parse correctly, catching syntax errors and breaking changes.

**Usage**:
```bash
./scripts/validate-examples.sh
```

**What it does**:
1. Finds all `.ora` files in `ora-example/` directory
2. Runs `ora parse` on each file
3. Reports success (✓) or failure (✗) with error details
4. Shows summary statistics

**Example Output**:
```
==================================
Validating Ora Examples
==================================

✓ ora-example/smoke.ora
✓ ora-example/storage/basic_storage.ora
✗ ora-example/loops/for_loops.ora
  Parser error at line 9: Expected property name

==================================
Summary:
  Total:  29
  Passed: 23
  Failed: 6
==================================
```

**Exit Codes**:
- `0` - All examples pass
- `1` - One or more examples fail

**When to use**:
- Before committing changes to the parser
- After modifying language syntax
- Before releases to ensure examples are valid
- When adding new examples to verify they work

**Integration with CI**:
```yaml
# .github/workflows/test.yml
- name: Validate Examples
  run: ./scripts/validate-examples.sh
```

## Script Guidelines

### Adding New Scripts

When adding a new script:

1. Make it executable: `chmod +x scripts/your-script.sh`
2. Add shebang: `#!/usr/bin/env bash`
3. Include error handling: `set -e`
4. Add help text: Show usage with `--help`
5. Document in this README
6. Update main README.md with usage

### Script Template

```bash
#!/usr/bin/env bash
# Short description of what this script does

set -e  # Exit on error

# Parse arguments
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [options]"
    echo "Description of script"
    exit 0
fi

# Script logic here
```

## Planned Scripts

The following scripts are planned but not yet implemented:

### `validate-docs.sh` (TODO)

Validate all documentation links and code snippets.

**Features**:
- Check internal markdown links
- Verify external URLs
- Extract and test code snippets
- Report broken references

**Usage** (planned):
```bash
./scripts/validate-docs.sh          # Check all docs
./scripts/validate-docs.sh --links  # Check links only
./scripts/validate-docs.sh --code   # Test code snippets only
```

### `format-examples.sh` (TODO)

Format all example files with consistent style.

**Usage** (planned):
```bash
./scripts/format-examples.sh        # Format all examples
./scripts/format-examples.sh --check # Check formatting only
```

### `generate-docs.sh` (TODO)

Generate API documentation from source code.

**Usage** (planned):
```bash
./scripts/generate-docs.sh          # Generate all docs
./scripts/generate-docs.sh --api    # Generate API docs only
```

## Contributing

When contributing scripts:

1. Follow the script template above
2. Make scripts portable (avoid platform-specific features)
3. Include error handling and helpful messages
4. Add color output for better UX (green/red for pass/fail)
5. Update this README with documentation
6. Update main project README if user-facing

## Troubleshooting

### Script Not Executable

```bash
chmod +x scripts/validate-examples.sh
```

### Script Fails to Find Compiler

Ensure you've built the project first:
```bash
zig build
```

The script expects the compiler at `./zig-out/bin/ora`.

### Colors Not Showing

The scripts use ANSI color codes. If colors don't display:
- Ensure your terminal supports colors
- Try running in a different terminal
- Colors may not work when piping output

---

*For general project documentation, see the main [README.md](../README.md)*

