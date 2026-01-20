# Ora Compiler API Documentation

> **⚠️ Development Status**: Evolving API surface. Names
> and availability may differ from the current CLI. Use `./zig-out/bin/ora --help`
> for authoritative commands.

## Overview

The Ora compiler is a domain-specific language compiler for smart contract
development with formal verification capabilities. The current architecture is
multi-phase:

1. **Lexical Analysis** - Tokenization of source code
2. **Syntax Analysis** - Abstract Syntax Tree generation
3. **Semantic Analysis** - Type checking and validation
4. **Ora MLIR Lowering** - Conversion to Ora-specific MLIR
5. **Sensei-IR Lowering** - Conversion to SIR
6. **Bytecode Generation** - EVM bytecode via SIR

## CLI Commands

### Basic Usage
```bash
ora <command> <file>
```

### Common Commands

Command names can change; confirm with `--help`. Common entry points include:

- `parse <file>` - Parse a .ora file to AST
- `fmt <file>` - Format Ora source
- `emit-mlir <file>` - Emit Ora MLIR
- `emit-sir <file>` - Emit SIR MLIR (where supported)
- `emit-abi <file>` - Emit Ora ABI

### Examples

```bash
# Parse a simple contract
./zig-out/bin/ora parse ora-example/smoke.ora

# Emit Ora MLIR
./zig-out/bin/ora emit-mlir ora-example/smoke.ora

# Emit SIR MLIR (where supported)
./zig-out/bin/ora emit-sir ora-example/smoke.ora
```

## Library API

The library API is not yet stabilized. The following snippets are illustrative
and may not match current names or types.

### Core Modules

#### `SIRLowering`
Lowers MLIR to Sensei-IR (SIR) intermediate representation.

```zig
// Note: API is under development
// This will lower MLIR operations to Sensei-IR (SIR) format
```

#### `SIRBackend`
Generates EVM bytecode from Sensei-IR (SIR).

```zig
// Note: API is under development
// This will use the Sensei-IR debug-backend to generate EVM bytecode
```

#### `IRBuilder`
Converts AST to High-level Intermediate Representation.

```zig
var ir_builder = IRBuilder.init(allocator);
defer ir_builder.deinit();

try ir_builder.buildFromAST(ast_nodes);
const hir_program = ir_builder.getProgramPtr();
```

#### `SemanticAnalyzer`
Performs semantic analysis with integrated formal verification.

```zig
var analyzer = SemanticAnalyzer.init(allocator);
analyzer.initSelfReferences();
defer analyzer.deinit();

const result = try analyzer.analyze(ast_nodes);
if (result.verified) {
    std.debug.print("Verification passed: {d} conditions checked\n", .{result.conditions_checked});
}
```

#### `Parser`
Parses Ora source code into an Abstract Syntax Tree.

```zig
var parser = Parser.init(allocator);
defer parser.deinit();

const ast_nodes = try parser.parse(tokens);
defer {
    for (ast_nodes) |node| {
        node.deinit(allocator);
    }
    allocator.free(ast_nodes);
}
```

#### `Lexer`
Tokenizes Ora source code.

```zig
const tokens = try Lexer.scan(source_code, allocator);
defer allocator.free(tokens);

for (tokens) |token| {
    std.debug.print("Token: {s}\n", .{@tagName(token.tag)});
}
```

## Compilation Pipeline

### Full Compilation Example

```zig
const std = @import("std");
const ora = @import("ora");

pub fn compileOraFile(allocator: std.mem.Allocator, file_path: []const u8) ![]u8 {
    // 1. Read source file
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024);
    defer allocator.free(source);
    
    // 2. Tokenize
    const tokens = try ora.Lexer.scan(source, allocator);
    defer allocator.free(tokens);
    
    // 3. Parse to AST
    var parser = ora.Parser.init(allocator);
    defer parser.deinit();
    
    const ast_nodes = try parser.parse(tokens);
    defer {
        for (ast_nodes) |node| {
            node.deinit(allocator);
        }
        allocator.free(ast_nodes);
    }
    
    // 4. Semantic Analysis
    var analyzer = ora.SemanticAnalyzer.init(allocator);
    analyzer.initSelfReferences();
    defer analyzer.deinit();
    
    const analysis_result = try analyzer.analyze(ast_nodes);
    if (!analysis_result.verified) {
        return error.VerificationFailed;
    }
    
    // 5. Generate HIR
    var ir_builder = ora.IRBuilder.init(allocator);
    defer ir_builder.deinit();
    
    try ir_builder.buildFromAST(ast_nodes);
    const hir_program = ir_builder.getProgramPtr();
    
    // 6. Lower to Sensei-IR (SIR)
    // Note: API is under development
    // This will lower MLIR to Sensei-IR (SIR) format
    
    // 7. Generate EVM bytecode from Sensei-IR
    // Note: API is under development
    // This will use the Sensei-IR debug-backend to generate bytecode
    
    // Return bytecode (caller owns)
    return allocator.dupe(u8, compile_result.bytecode);
}
```

### Incremental Compilation

```zig
// For faster development, you can stop at any stage
pub fn analyzeOnly(allocator: std.mem.Allocator, source: []const u8) !ora.SemanticAnalysis.Result {
    const tokens = try ora.Lexer.scan(source, allocator);
    defer allocator.free(tokens);
    
    var parser = ora.Parser.init(allocator);
    defer parser.deinit();
    
    const ast_nodes = try parser.parse(tokens);
    defer {
        for (ast_nodes) |node| {
            node.deinit(allocator);
        }
        allocator.free(ast_nodes);
    }
    
    var analyzer = ora.SemanticAnalyzer.init(allocator);
    analyzer.initSelfReferences();
    defer analyzer.deinit();
    
    return analyzer.analyze(ast_nodes);
}
```

## Error Handling

The compiler uses Zig's error union types for robust error handling:

```zig
const CompilerError = error{
    LexerError,
    ParseError,
    SemanticError,
    VerificationError,
    CodegenError,
    CompilationError,
    OutOfMemory,
};

// Semantic analysis errors
const SemanticError = error{
    MissingInitFunction,
    InvalidStorageAccess,
    TypeMismatch,
    UndefinedSymbol,
    DuplicateSymbol,
    InvalidFunctionCall,
    InvalidIndexAccess,
    InvalidFieldAccess,
    InvalidReturnType,
    MissingReturnStatement,
    InvalidRequiresClause,
    InvalidEnsuresClause,
    InvalidInvariant,
    VerificationFailed,
};
```

## Configuration

### Compiler Configuration

```zig
pub const CompilerConfig = struct {
    // Verification settings
    enable_formal_verification: bool = true,
    verification_timeout_ms: u32 = 30000,
    max_verification_complexity: u32 = 1000,
    
    // Optimization settings
    optimization_level: enum { none, basic, aggressive } = .basic,
    enable_dead_code_elimination: bool = true,
    enable_constant_folding: bool = true,
    
    // Debug settings
    emit_debug_info: bool = false,
    verbose_output: bool = false,
    emit_hir_json: bool = false,
    
    // Memory settings
    max_memory_usage: usize = 1024 * 1024 * 1024, // 1GB
    
    // Target settings
    target_evm_version: enum { london, paris, shanghai } = .london,
};
```

### Using Configuration

```zig
var config = ora.CompilerConfig{
    .enable_formal_verification = true,
    .optimization_level = .aggressive,
    .verbose_output = true,
};

var compiler = ora.Compiler.init(allocator, config);
defer compiler.deinit();

const result = try compiler.compileFile("my_contract.ora");
```

## Standard Library API

### Overview

Ora provides a built-in standard library (`std`) for direct access to EVM
primitives. No imports needed.

**Key Features**:
- Direct lowering to EVM-facing ops
- Type-safe (full compile-time checking)
- Always available

For comprehensive documentation, see [Standard Library](../standard-library.md).

### Quick Reference

```zig
// Block data
std.block.timestamp() -> u256      // Current block timestamp
std.block.number() -> u256         // Current block number
std.block.gaslimit() -> u256       // Block gas limit
std.block.coinbase() -> address    // Miner address
std.block.basefee() -> u256        // Base fee (EIP-1559)

// Transaction data
std.transaction.sender() -> address      // Transaction origin (EOA)
std.transaction.gasprice() -> u256       // Gas price

// Message data (call context)
std.msg.sender() -> address        // Immediate caller
std.msg.value() -> u256            // Wei sent with call
std.msg.data.size() -> u256        // Calldata size

// Constants
std.constants.ZERO_ADDRESS         // 0x000...000
std.constants.U256_MAX            // 2^256 - 1
std.constants.U128_MAX            // 2^128 - 1
std.constants.U64_MAX             // 2^64 - 1
std.constants.U32_MAX             // 2^32 - 1
```

### Usage Example

```ora
contract AccessControl {
    storage owner: address;
    
    pub fn initialize() {
        // Use std.msg.sender() for access control
        owner = std.msg.sender();
    }
    
    pub fn isOwner() -> bool {
        let caller = std.msg.sender();
        return caller == owner;
    }
    
    pub fn validRecipient(addr: address) -> bool {
        // Use std.constants for validation
        return addr != std.constants.ZERO_ADDRESS;
    }
    
    pub fn hasExpired(deadline: u256) -> bool {
        // Use std.block for time-based logic
        return std.block.timestamp() > deadline;
    }
}
```

### Compilation Flow

```
Ora Source:
  std.msg.sender()
       ↓ (Semantic Validation)
AST:
  BuiltinCall { path: "std.msg.sender", args: [] }
       ↓ (MLIR Lowering)
MLIR:
  %0 = ora.evm.caller() : i160
       ↓ (Sensei-IR Lowering)
Sensei-IR (SIR):
  fn main:
    entry -> result {
      result = caller
      iret
    }
       ↓ (Bytecode Generation)
EVM:
  CALLER (opcode 0x33)
```

Total overhead: zero (direct opcode mapping).

### Integration with Other APIs

The standard library integrates seamlessly with other compiler phases:

**Semantic Analysis**:
```zig
// In src/semantics/builtins.zig
var analyzer = SemanticAnalyzer.init(allocator);
// ... analyzer automatically validates std.* calls via BuiltinRegistry
```

**MLIR Lowering**:
```zig
// In src/mlir/expressions.zig
const expr_lowerer = ExpressionLowerer.init(..., builtin_registry);
// ... std.* calls are lowered to ora.evm.* MLIR operations
```

**Type Checking**:
```zig
// All std.* calls are type-checked at compile time
let timestamp: u256 = std.block.timestamp();  // ✅ Correct
let timestamp: address = std.block.timestamp(); // ❌ Compile error
```

---

## Capability Snapshot

- Lexical analysis with all Ora keywords
- Parser supporting contracts, functions, variables, and expressions
- AST generation with memory region support
- Type system with primitive and complex types
- Standard library built-ins
- MLIR lowering with automatic validation
- SSA transformation via mem2reg
- HIR generation and validation
- Sensei-IR (SIR) lowering and bytecode compilation
- Semantic analysis with builtin validation
- Effect tracking and analysis

## Memory Management

All modules follow Zig's explicit memory management patterns:
- Use `init()` to create instances
- Use `deinit()` to cleanup resources  
- Returned slices are owned by caller unless documented otherwise
- Use `defer` statements for automatic cleanup

Example:
```zig
var parser = Parser.init(allocator);
defer parser.deinit(); // Always cleanup

const result = try parser.parse(tokens);
defer allocator.free(result); // Caller owns result
```

## Testing

### Unit Tests

```bash
# Run all tests
zig build test

# Run specific test files
zig build test -- --test-filter "lexer"
zig build test -- --test-filter "parser"
zig build test -- --test-filter "semantic"
```

### Integration Tests

```bash
# Test with example files
zig build test-examples

# Test specific examples
zig build test-examples -- examples/simple_token.ora
```

## Integration

### Building with Sensei-IR Support

The build system includes Sensei-IR as a submodule:

```bash
zig build        # Build everything
zig build test   # Run tests
zig build docs   # Generate documentation
```

### Sensei-IR Integration

Ora uses Sensei-IR (SIR) as its backend for EVM bytecode generation:

- `sensei-ir-main/` - Sensei-IR submodule (Rust-based EVM IR)
- Integration via FFI or direct library linking (under development)
- The Sensei-IR debug-backend generates EVM bytecode from SIR

This architecture provides a clean, language-agnostic IR for EVM compilation.

## Performance Benchmarks

### Compilation Speed
- **Lexing**: ~1M tokens/second
- **Parsing**: ~100K AST nodes/second
- **Semantic Analysis**: ~50K nodes/second
- **HIR Generation**: ~75K nodes/second
- **MLIR Lowering**: ~50K nodes/second
- **Sensei-IR Lowering**: benchmarks pending

### Memory Usage
- **Typical contract**: 1-5MB peak memory
- **Large contract**: 10-50MB peak memory
- **Batch compilation**: Scales linearly

## IDE Integration

### Language Server Protocol (Planned)

```bash
# Start language server
ora lsp

# Connect from VS Code, Vim, Emacs, etc.
```

### Features (Planned)
- Syntax highlighting ✅ *Manual implementation exists*
- Error reporting
- Auto-completion
- Go-to-definition
- Hover information
- Refactoring support

## Contributing

### Development Setup

```bash
git clone https://github.com/oralang/Ora
cd Ora
zig build
```

### API Extension

To add new compiler phases:

1. Implement the module in `src/`
2. Add to the compilation pipeline
3. Update CLI commands
4. Add comprehensive tests
5. Update documentation

This API provides a solid foundation for building smart contracts with formal verification capabilities while maintaining performance and safety. 
