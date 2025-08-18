# Ora Language Examples

This directory contains minimal examples for the Ora language, organized by feature. Each example is deliberately simplified to test only one specific language feature at a time.

## Directory Structure

- **control_flow/**: Control flow constructs (if, while, break, continue)
- **enums/**: Enum declarations and basic usage
- **errors/**: Error declarations and try-catch syntax
- **expressions/**: Expression syntax without function context
- **functions/**: Function declaration syntax
- **imports/**: Import statement syntax
- **logs/**: Log (event) declarations
- **loops/**: Loop constructs (for, while)
- **memory/**: Memory variable declarations
- **statements/**: Basic statement syntax
- **storage/**: Storage variable declarations
- **structs/**: Struct declarations
- **switch/**: Switch expressions and statements
- **transient/**: Transient storage declarations
- **types/**: Type declarations and basic usage

## Testing Approach

These minimal examples are designed for parser testing with a "divide and conquer" approach:

1. **Isolation**: Each file tests only one language feature
2. **Minimalism**: No complex interactions between features
3. **Readability**: Clear, commented examples
4. **Completeness**: Each language construct has dedicated tests

## Usage

Use these examples to test specific parts of the parser:

```bash
ora parse path/to/feature.ora
ora ast path/to/feature.ora -o output/dir
```

## Example Format

Each example follows this format:

```
// ==========================================
// FEATURE NAME TEST
// ==========================================
// This file tests [specific feature] syntax

[Minimal code demonstrating just this feature]
```

## Note on Simplicity

These examples intentionally avoid complex interactions between language features to allow isolated testing of the parser. They may not represent idiomatic Ora code but are optimized for parser testing.
