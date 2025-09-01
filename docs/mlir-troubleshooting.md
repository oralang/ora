# MLIR Lowering Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when working with the MLIR lowering system in the Ora compiler. It covers error diagnosis, debugging strategies, and resolution steps for typical problems.

**Note:** This guide focuses on issues specific to the Ora MLIR lowering implementation. For general MLIR issues, please refer to the official MLIR documentation at https://mlir.llvm.org. We use the existing MLIR framework and extend it with Ora-specific functionality.

## Common Error Categories

### 1. Type Mismatch Errors

#### Symptoms
- Error messages like "Type mismatch in binary operation"
- MLIR verification failures related to type consistency
- Compilation failures during type mapping

#### Common Causes
1. **Incompatible operand types in expressions**
   ```ora
   let result = balance + "invalid"; // u256 + string
   ```

2. **Incorrect type mapping for complex types**
   ```ora
   let array: [u256; 10] = [1, 2, "three"]; // Mixed types in array
   ```

3. **Missing type conversions**
   ```ora
   let small: u8 = 256; // Value too large for u8
   ```

#### Diagnosis Steps
1. **Check the error location**
   ```
   error: Type mismatch in binary operation
     --> contract.ora:15:8
      |
   15 |     let result = balance + "invalid";
      |                          ^ expected numeric type, found string
   ```

2. **Verify type compatibility**
   - Check that both operands in binary operations have compatible types
   - Ensure array elements have consistent types
   - Verify function argument types match parameter types

3. **Enable verbose type checking**
   ```bash
   ora compile --mlir-verify --verbose contract.ora
   ```

#### Resolution Strategies
1. **Add explicit type conversions**
   ```ora
   let result = balance + @cast(u256, amount);
   ```

2. **Use consistent types in collections**
   ```ora
   let array: [u256; 3] = [1, 2, 3]; // All u256
   ```

3. **Check type definitions**
   ```ora
   struct Balance {
       amount: u256,
       currency: string,
   }
   ```

### 2. Memory Region Violations

#### Symptoms
- Errors about invalid memory region usage
- Storage/memory/tstore constraint violations
- MLIR operations with incorrect memory space attributes

#### Common Causes
1. **Incorrect memory region annotations**
   ```ora
   memory balance: u256; // Should be storage for persistent state
   ```

2. **Cross-region assignments without proper handling**
   ```ora
   storage persistent_data: u256;
   memory temp_data: u256;
   persistent_data = temp_data; // Direct assignment may be invalid
   ```

3. **Missing region attributes in MLIR operations**

#### Diagnosis Steps
1. **Check memory region declarations**
   ```ora
   contract Token {
       storage balances: map[address, u256];  // Correct: storage
       memory temp_sum: u256;                 // Correct: temporary
       tstore pending: u256;                  // Correct: transient
   }
   ```

2. **Verify region usage patterns**
   - Storage: Persistent contract state
   - Memory: Temporary computation data
   - TStore: Transient storage window

3. **Enable memory region validation**
   ```bash
   ora compile --mlir-passes="memory-region-validation" contract.ora
   ```

#### Resolution Strategies
1. **Use correct region annotations**
   ```ora
   storage balance: u256;        // Persistent state
   memory temp_balance: u256;    // Temporary calculation
   tstore pending_tx: u256;      // Transient data
   ```

2. **Add explicit region transfers**
   ```ora
   storage_var = @storage_copy(memory_var);
   ```

3. **Validate region constraints**
   - Ensure storage variables are only modified in appropriate contexts
   - Use memory for temporary calculations
   - Use tstore for cross-transaction temporary data

### 3. Symbol Resolution Failures

#### Symptoms
- "Undefined variable" or "Undefined function" errors
- Symbol table lookup failures
- Scope-related compilation errors

#### Common Causes
1. **Variable used before declaration**
   ```ora
   let result = undeclared_var + 10; // undeclared_var not defined
   ```

2. **Scope visibility issues**
   ```ora
   if (condition) {
       let local_var = 42;
   }
   let result = local_var; // local_var out of scope
   ```

3. **Function name conflicts or typos**
   ```ora
   fn calculate() -> u256 { return 42; }
   let result = calcualte(); // Typo in function name
   ```

#### Diagnosis Steps
1. **Check variable declarations**
   - Ensure variables are declared before use
   - Verify correct spelling and case sensitivity
   - Check scope boundaries

2. **Verify function signatures**
   ```ora
   fn transfer(to: address, amount: u256) -> bool;
   // Usage must match signature exactly
   let success = transfer(recipient, 100);
   ```

3. **Enable symbol table debugging**
   ```bash
   ora compile --debug-symbols contract.ora
   ```

#### Resolution Strategies
1. **Declare variables before use**
   ```ora
   let balance: u256 = 1000;
   let result = balance + 100; // balance is now defined
   ```

2. **Manage scope correctly**
   ```ora
   let result: u256;
   if (condition) {
       let temp = calculate();
       result = temp; // Assign to outer scope variable
   }
   ```

3. **Use proper function declarations**
   ```ora
   fn helper_function(param: u256) -> u256 {
       return param * 2;
   }
   
   fn main() {
       let result = helper_function(21);
   }
   ```

### 4. MLIR Operation Construction Failures

#### Symptoms
- MLIR verifier errors
- Invalid operation construction
- Malformed MLIR IR output

#### Common Causes
1. **Incorrect operation attributes**
2. **Invalid block or region structure**
3. **Type inconsistencies in MLIR operations**
4. **Missing location information**

#### Diagnosis Steps
1. **Enable MLIR verification**
   ```bash
   ora compile --mlir-verify contract.ora
   ```

2. **Check MLIR output structure**
   ```bash
   ora compile --emit-mlir contract.ora > output.mlir
   mlir-opt --verify-diagnostics output.mlir
   ```

3. **Validate operation construction**
   - Check that all operations have proper types
   - Verify block and region structure
   - Ensure location information is attached

#### Resolution Strategies
1. **Fix operation construction**
   ```zig
   // Ensure proper type consistency
   const lhs_type = c.mlirValueGetType(lhs);
   const rhs_type = c.mlirValueGetType(rhs);
   if (!c.mlirTypeEqual(lhs_type, rhs_type)) {
       // Handle type mismatch
   }
   ```

2. **Add proper location information**
   ```zig
   const loc = locations.createLocation(span);
   c.mlirOperationSetLocation(operation, loc);
   ```

3. **Validate block structure**
   ```zig
   // Ensure blocks are properly terminated
   if (!c.mlirBlockHasTerminator(block)) {
       // Add terminator operation
   }
   ```

## Debugging Strategies

### 1. Incremental Testing

Start with minimal test cases and gradually increase complexity:

```ora
// Start with simple expressions
let a = 42;

// Add binary operations
let b = a + 10;

// Add function calls
fn double(x: u256) -> u256 { return x * 2; }
let c = double(b);

// Add control flow
if (c > 100) {
    // ...
}
```

### 2. Enable Verbose Logging

Use compiler flags to get detailed information:

```bash
# Enable all debugging output
ora compile --verbose --debug-symbols --mlir-verify contract.ora

# Enable specific debugging features
ora compile --mlir-dump-after=canonicalize contract.ora

# Enable timing information
ora compile --mlir-timing contract.ora
```

### 3. Isolate Components

Test individual components in isolation:

```zig
// Test type mapping separately
const type_mapper = TypeMapper.init(ctx);
const mlir_type = try type_mapper.toMlirType(ora_type);

// Test expression lowering separately
const expr_lowerer = ExpressionLowerer.init(/* ... */);
const result = expr_lowerer.lowerExpression(expr);

// Test statement lowering separately
const stmt_lowerer = StatementLowerer.init(/* ... */);
stmt_lowerer.lowerStatement(stmt);
```

### 4. Use MLIR Tools

Leverage MLIR's built-in debugging tools from the LLVM MLIR distribution:

```bash
# Verify MLIR correctness
mlir-opt --verify-diagnostics output.mlir

# Run specific passes
mlir-opt --canonicalize --cse output.mlir

# Print pass statistics
mlir-opt --pass-statistics output.mlir

# Enable pass timing
mlir-opt --pass-timing output.mlir

# Show available dialects and passes
mlir-opt --help
```

**Note:** These tools are part of the LLVM MLIR installation and must be available in your PATH.

## Performance Issues

### 1. Slow Compilation Times

#### Symptoms
- Long compilation times for large contracts
- Memory usage spikes during compilation
- Timeout errors in CI/CD pipelines

#### Diagnosis Steps
1. **Profile compilation phases**
   ```bash
   ora compile --timing --profile contract.ora
   ```

2. **Check memory usage**
   ```bash
   /usr/bin/time -v ora compile contract.ora
   ```

3. **Identify bottlenecks**
   - Large AST trees
   - Complex type inference
   - Expensive MLIR passes

#### Resolution Strategies
1. **Optimize AST structure**
   - Simplify complex expressions
   - Reduce nesting depth
   - Split large functions

2. **Tune MLIR passes**
   ```bash
   # Use faster pass pipeline
   ora compile --mlir-passes="canonicalize" contract.ora
   
   # Disable expensive passes
   ora compile --mlir-passes="" contract.ora
   ```

3. **Increase memory limits**
   ```bash
   # Set memory limits for compilation
   ulimit -m 4194304  # 4GB limit
   ora compile contract.ora
   ```

### 2. Memory Leaks

#### Symptoms
- Increasing memory usage during compilation
- Out-of-memory errors
- Slow garbage collection

#### Diagnosis Steps
1. **Use memory profiling tools**
   ```bash
   valgrind --tool=memcheck ora compile contract.ora
   ```

2. **Check for resource leaks**
   - MLIR context cleanup
   - Symbol table deallocation
   - Temporary string allocations

#### Resolution Strategies
1. **Ensure proper cleanup**
   ```zig
   defer result.deinit(allocator);
   defer symbol_table.deinit();
   defer error_handler.deinit();
   ```

2. **Use arena allocators for temporary data**
   ```zig
   var arena = std.heap.ArenaAllocator.init(allocator);
   defer arena.deinit();
   const temp_allocator = arena.allocator();
   ```

## Integration Issues

### 1. CLI Integration Problems

#### Symptoms
- Command-line flags not working
- Incorrect output formats
- Missing MLIR output

#### Common Causes
1. **Incorrect flag usage**
   ```bash
   # Wrong
   ora --emit-mlir compile contract.ora
   
   # Correct
   ora compile --emit-mlir contract.ora
   ```

2. **Missing MLIR backend initialization**
3. **Output redirection issues**

#### Resolution Strategies
1. **Check flag syntax**
   ```bash
   ora compile --help  # Show available flags
   ora compile --emit-mlir --output=contract.mlir contract.ora
   ```

2. **Verify MLIR initialization**
   ```zig
   // Ensure MLIR is properly initialized
   c.mlirRegisterAllDialects(registry);
   c.mlirContextAppendDialectRegistry(ctx, registry);
   ```

### 2. Build System Integration

#### Symptoms
- Build failures in CI/CD
- Inconsistent compilation results
- Missing dependencies

#### Resolution Strategies
1. **Ensure consistent build environment**
   ```yaml
   # CI configuration
   - name: Setup MLIR
     run: |
       apt-get install mlir-dev
       export MLIR_DIR=/usr/lib/mlir
   ```

2. **Add proper dependencies**
   ```zig
   // build.zig
   exe.linkSystemLibrary("MLIR");
   exe.addIncludePath("/usr/include/mlir");
   ```

## Testing and Validation

### 1. Create Minimal Reproducible Examples

When reporting issues, create minimal test cases:

```ora
// Minimal example for type mismatch
contract Test {
    fn example() {
        let x: u256 = 42;
        let y: string = "hello";
        let z = x + y; // Error: type mismatch
    }
}
```

### 2. Use FileCheck for MLIR Validation

Create tests with expected MLIR patterns:

```mlir
// CHECK: func.func @example
// CHECK: %[[C42:.*]] = arith.constant 42 : i256
// CHECK: return %[[C42]] : i256
func.func @example() -> i256 {
    %c42 = arith.constant 42 : i256
    return %c42 : i256
}
```

### 3. Regression Testing

Maintain a suite of regression tests:

```bash
# Run all MLIR tests
zig build test-mlir

# Run specific test categories
zig build test-mlir-types
zig build test-mlir-expressions
zig build test-mlir-statements
```

## Getting Help

### 1. Enable Debug Output

Always include debug information when reporting issues:

```bash
ora compile --verbose --debug-symbols --mlir-verify --emit-mlir contract.ora 2>&1 | tee debug.log
```

### 2. Provide Context Information

Include the following information:
- Ora compiler version
- MLIR version
- Operating system and architecture
- Complete error messages
- Minimal reproducible example
- Expected vs. actual behavior

### 3. Check Known Issues

Before reporting new issues:
1. Check the GitHub issues page
2. Search the documentation
3. Review recent changes in the changelog
4. Test with the latest compiler version

### 4. Community Resources

- **GitHub Issues:** Report bugs and feature requests
- **Discussions:** Ask questions and share experiences
- **Documentation:** Comprehensive guides and references
- **Examples:** Sample code and use cases

## Best Practices for Avoiding Issues

### 1. Code Organization

```ora
// Good: Clear structure and consistent naming
contract TokenContract {
    storage balances: map[address, u256];
    storage total_supply: u256;
    
    fn transfer(to: address, amount: u256) -> bool {
        requires balances[msg.sender] >= amount;
        ensures balances[to] == old(balances[to]) + amount;
        
        // Implementation
        return true;
    }
}
```

### 2. Type Safety

```ora
// Good: Explicit types and proper conversions
fn calculate_fee(amount: u256, rate: u256) -> u256 {
    let fee = (amount * rate) / 10000;
    return fee;
}

// Avoid: Implicit conversions and mixed types
fn bad_calculate(amount, rate) {
    return amount * rate / "10000"; // Type errors
}
```

### 3. Error Handling

```ora
// Good: Proper error handling
fn safe_transfer(to: address, amount: u256) -> !TransferError {
    if (balances[msg.sender] < amount) {
        return error.InsufficientBalance;
    }
    
    try move amount from msg.sender to to;
    return {};
}
```

### 4. Testing Strategy

```ora
// Good: Comprehensive test coverage
#[test]
fn test_transfer_success() {
    let contract = TokenContract.init();
    contract.mint(alice, 1000);
    
    let result = contract.transfer(bob, 100);
    assert(result.is_ok());
    assert(contract.balance_of(alice) == 900);
    assert(contract.balance_of(bob) == 100);
}

#[test]
fn test_transfer_insufficient_balance() {
    let contract = TokenContract.init();
    
    let result = contract.transfer(bob, 100);
    assert(result.is_error());
    assert(result.error() == TransferError.InsufficientBalance);
}
```

This troubleshooting guide should help developers quickly identify and resolve common issues with the MLIR lowering system.