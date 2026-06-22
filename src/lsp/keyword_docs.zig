const std = @import("std");
const lexer = @import("ora_lexer");

pub const contextual_keywords = [_][]const u8{
    "type",
    "self",
};

const docs = std.StaticStringMap([]const u8).initComptime(.{
    .{ "address", "Built-in address type." },
    .{ "and", "Logical conjunction operator." },
    .{ "as", "Converts a value to another type." },
    .{ "assert", "Verification assertion checked by the prover." },
    .{ "assume", "Verification assumption taken as given by the prover." },
    .{ "bitfield", "Declares a packed bitfield type for efficient storage." },
    .{ "bool", "Built-in boolean type." },
    .{ "break", "Exits the innermost loop." },
    .{ "bytes", "Built-in dynamically-sized byte sequence type." },
    .{ "call", "Declares or invokes a state-changing external call." },
    .{ "catch", "Handles an error from an error union." },
    .{ "comptime", "Evaluates an expression at compile time." },
    .{ "const", "Declares a compile-time constant." },
    .{ "continue", "Skips to the next iteration of the innermost loop." },
    .{ "contract", "Declares a smart contract type." },
    .{ "decreases", "Declares a decreasing termination measure." },
    .{ "else", "Alternative branch of an `if` or `switch`." },
    .{ "ensures", "Postcondition guaranteed to hold when the function returns." },
    .{ "ensures_err", "Error postcondition guaranteed to hold on error-union returns." },
    .{ "ensures_ok", "Success postcondition guaranteed to hold on successful error-union returns." },
    .{ "enum", "Declares an enumeration type." },
    .{ "error", "Declares a custom error type." },
    .{ "errors", "Declares the closed error set an extern trait method may return." },
    .{ "exists", "Existential quantifier for a value satisfying a predicate." },
    .{ "extern", "Declares an external contract interface." },
    .{ "false", "Boolean literal `false`." },
    .{ "fn", "Declares a function." },
    .{ "for", "Iterates over a range or collection." },
    .{ "forall", "Universal quantifier for all values satisfying a predicate." },
    .{ "from", "Lower bound marker used by range-style syntax." },
    .{ "ghost", "Ghost declaration that exists only for verification, not compiled code." },
    .{ "guard", "Runtime-enforced precondition checked at runtime and assumed after it passes." },
    .{ "havoc", "Assigns an arbitrary value for verification." },
    .{ "i8", "Built-in signed 8-bit integer type." },
    .{ "i16", "Built-in signed 16-bit integer type." },
    .{ "i32", "Built-in signed 32-bit integer type." },
    .{ "i64", "Built-in signed 64-bit integer type." },
    .{ "i128", "Built-in signed 128-bit integer type." },
    .{ "i256", "Built-in signed 256-bit integer type." },
    .{ "if", "Conditional branch." },
    .{ "immutable", "Declares immutable storage." },
    .{ "impl", "Implements a trait for a type." },
    .{ "inline", "Declares a private function that must be expanded at each call site." },
    .{ "import", "Imports declarations from another module." },
    .{ "increases", "Declares an increasing termination measure." },
    .{ "init", "Declares a contract initializer." },
    .{ "invariant", "Contract or loop invariant preserved across state transitions." },
    .{ "let", "Declares an immutable local binding." },
    .{ "log", "Declares an event emitted as an EVM log." },
    .{ "map", "Built-in key-value map type." },
    .{ "match", "Pattern match over values, enums, and Result/error unions." },
    .{ "memory", "Memory qualifier for temporary data within a call." },
    .{ "modifies", "Declares state locations a function may modify." },
    .{ "old", "Refers to the pre-state value of an expression in postconditions." },
    .{ "or", "Logical disjunction operator." },
    .{ "pub", "Makes a declaration publicly visible." },
    .{ "requires", "Precondition that must hold when the function is called." },
    .{ "resource", "Declares a nominal resource quantity domain." },
    .{ "result", "Refers to the return value in postconditions." },
    .{ "return", "Returns a value from the current function." },
    .{ "self", "Refers to the current contract instance." },
    .{ "slice", "Built-in slice type." },
    .{ "staticcall", "Declares or invokes a read-only external call." },
    .{ "storage", "Storage qualifier for state persisted on-chain between calls." },
    .{ "string", "Built-in string type." },
    .{ "struct", "Declares a named struct type." },
    .{ "switch", "Multi-way branch on a value." },
    .{ "to", "Upper bound marker used by range-style syntax." },
    .{ "trait", "Declares an interface trait." },
    .{ "true", "Boolean literal `true`." },
    .{ "try", "Unwraps an error union, propagating the error on failure." },
    .{ "tstore", "Transient storage qualifier cleared after each transaction." },
    .{ "type", "Declares a type alias." },
    .{ "u8", "Built-in unsigned 8-bit integer type." },
    .{ "u16", "Built-in unsigned 16-bit integer type." },
    .{ "u32", "Built-in unsigned 32-bit integer type." },
    .{ "u64", "Built-in unsigned 64-bit integer type." },
    .{ "u128", "Built-in unsigned 128-bit integer type." },
    .{ "u160", "Built-in unsigned 160-bit integer type." },
    .{ "u256", "Built-in unsigned 256-bit integer type." },
    .{ "var", "Declares a mutable variable." },
    .{ "void", "Built-in empty return type." },
    .{ "where", "Type constraint or refinement clause." },
    .{ "while", "Loop that repeats while a condition holds." },
});

comptime {
    @setEvalBranchQuota(10_000);
    const keyword_keys = lexer.keywords.kvs.keys[0..lexer.keywords.kvs.len];
    for (keyword_keys) |keyword| {
        if (docs.get(keyword) == null) {
            @compileError("missing LSP keyword documentation for '" ++ keyword ++ "'");
        }
    }
    for (contextual_keywords) |keyword| {
        if (docs.get(keyword) == null) {
            @compileError("missing LSP contextual keyword documentation for '" ++ keyword ++ "'");
        }
    }
}

pub fn documentation(keyword: []const u8) ?[]const u8 {
    return docs.get(keyword);
}
