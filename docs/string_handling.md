# String Handling in Ora Smart Contract Language

## Overview

Ora implements a simplified string model that is optimized for smart contract development. Unlike general-purpose programming languages, Ora's string handling is deliberately restricted to improve security, reduce gas costs, and simplify implementation.

## String Limitations

1. **ASCII-only**: Strings are restricted to ASCII characters (0-127) only. Unicode support is intentionally excluded.

2. **Limited Escape Sequences**: Only the following escape sequences are supported:
   - `\n` - Newline
   - `\t` - Tab
   - `\"` - Double quote
   - `\\` - Backslash

3. **Size Restrictions**: String literals are limited to 1KB in length.

## Rationale

These limitations are intentional design choices for a smart contract language:

1. **Gas Efficiency**: 
   - Simplified string processing reduces computational overhead
   - Smaller code size for string handling routines means less deployment cost
   - Predictable behavior for gas estimation

2. **Security Benefits**:
   - Eliminating complex escape sequences reduces potential for escaping vulnerabilities
   - ASCII-only text avoids Unicode-related security issues like right-to-left overrides
   - Simpler validation means fewer potential bugs in the compiler

3. **Implementation Simplicity**:
   - Reduced complexity in lexer and parser
   - Simpler semantic validation
   - Less room for implementation errors

## Usage Guidelines

### Best Practices

1. Use strings primarily for:
   - Error messages
   - Event data
   - Simple identifiers

2. Avoid using strings for:
   - Data storage (prefer structured types)
   - Complex text processing
   - Encoding binary data (use byte arrays instead)

### Alternative Approaches

For cases where you might traditionally use strings but want to be more gas-efficient:

1. **Enums instead of string constants**
   ```
   enum ErrorCode {
     InsufficientFunds,
     Unauthorized,
     InvalidInput
   }
   ```

2. **Byte arrays for fixed-size data**
   ```
   var data: [32]u8 = [_]u8{...};
   ```

## Future Considerations

The Ora language team may consider the following enhancements in the future:

1. Adding compile-time string interning for common strings
2. Optimizing storage of string literals in the bytecode
3. Adding specialized string handling for events and error messages

However, the fundamental limitations (ASCII-only, limited escaping, size restrictions) will remain as core design principles of the language to maintain its security and efficiency advantages.
