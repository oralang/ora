# Ora Examples

This directory contains example Ora programs demonstrating various language features.

## ✅ Working Examples (23/29)

The following examples parse successfully with the current compiler:

### Basic Features
- `smoke.ora` - Basic contract structure
- `storage/basic_storage.ora` - Storage variables
- `storage/storage_test_1.ora` - Storage operations
- `storage/storage_test_2.ora` - Storage operations
- `statements/contract_declaration.ora` - Contract declarations
- `statements/compound_assignments.ora` - Compound assignments (+=, -=, etc.)

### Functions & Control Flow
- `functions/basic_functions.ora` - Function declarations with requires clauses
- `control_flow/basic_control_flow.ora` - If/else statements
- `loops/while_loops.ora` - While loops

### Types & Data Structures
- `types/basic_types.ora` - Primitive types and maps
- `structs/basic_structs.ora` - Struct definitions
- `enums/basic_enums.ora` - Enum declarations
- `enums/enum_usage_test.ora` - Enum usage

### Advanced Features
- `switch/switch_expression.ora` - Switch expressions
- `switch/switch_expressions.ora` - Switch expressions
- `switch/switch_labeled.ora` - Labeled switch
- `errors/error_declarations.ora` - Error declarations
- `errors/try_catch.ora` - Try-catch blocks
- `expressions/basic_expressions.ora` - Various expressions
- `strings/simple_string.ora` - String handling
- `memory/basic_memory.ora` - Memory variables and anonymous structs
- `logs/basic_logs.ora` - Event/log declarations
- `transient/basic_transient.ora` - Transient storage (EIP-1153)
- `imports/basic_imports.ora` - Import statements

## 🚧 Examples with Unsupported Features (6/29)

The following examples use syntax not yet fully implemented:

- `loops/for_loops.ora` - For loop syntax (capture syntax `|value, index|`)
- `control_flow/break_continue.ora` - Break/continue in loops (capture syntax)
- `errors/basic_errors.ora` - Advanced error parameter passing
- `storage/storage_test_3.ora` - Complex storage operations
- `max_features.ora` - Comprehensive feature showcase (various advanced features)

## 🔍 Analysis & Development Tools

### Complexity Analysis Examples

- `complexity_example.ora` - Simple example demonstrating different function complexity levels
  ```bash
  ./zig-out/bin/ora --analyze-complexity ora-example/complexity_example.ora
  ```

- `defi_lending_pool.ora` - **Some what of a realistic DeFi lending pool contract** (700+ lines)
  - Complete lending/borrowing implementation with enums and switch statements
  - Interest rate calculations
  - Health factor monitoring
  - Liquidation mechanics
  - 21 functions demonstrating full complexity spectrum:
    * ✓ 76% Simple functions (perfect for inline)
    * ○ 19% Moderate functions (well-structured)
    * ✗ 4% Complex functions (needs refactoring)
  ```bash
  ./zig-out/bin/ora --analyze-complexity ora-example/defi_lending_pool.ora
  ```
  
  **Complexity Distribution:**
  - ✓ 11 Simple functions (73%) - Optimal for performance
  - ○ 4 Moderate functions (26%) - Well-structured business logic
  - Perfect example of maintainable smart contract design!

## Testing Examples

To validate all examples:

```bash
./scripts/validate-examples.sh
```

To test a single example:

```bash
./zig-out/bin/ora parse ora-example/smoke.ora
```

## Contributing

When adding new examples:
1. Ensure they use current Ora syntax
2. Test with `./zig-out/bin/ora parse`
3. Update this README if adding new categories
4. Keep examples focused on demonstrating specific features

