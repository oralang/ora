// ============================================================================
// Lexer - Comprehensive Lexical Analyzer
// ============================================================================
//
// Full-featured lexer with sophisticated error diagnostics, trivia tracking,
// and comprehensive token support for the Ora language.
//
// FEATURES:
//   • Rich error diagnostics with context and suggestions
//   • Trivia tracking (comments, whitespace) for lossless printing
//   • 50+ token types including keywords, operators, literals
//   • Source position tracking (line, column, byte offset)
//   • Multi-file support with file_id
//   • Error recovery for better diagnostics
//
// SECTIONS:
//   • Error diagnostics infrastructure
//   • Source management & position tracking
//   • Trivia handling (comments & whitespace)
//   • Token types & definitions
//   • Core lexer logic & state machine
//
// ============================================================================

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import error recovery system
const error_recovery = @import("lexer/error_recovery.zig");

// Import scanner modules
const string_scanners = @import("lexer/scanners/strings.zig");
const number_scanners = @import("lexer/scanners/numbers.zig");
const identifier_scanners = @import("lexer/scanners/identifiers.zig");

// Import trivia handling
const trivia = @import("lexer/trivia.zig");

// ============================================================================
// SECTION 1: Error Diagnostics Infrastructure
// ============================================================================

/// Lexer-specific errors for better diagnostics
pub const LexerError = error{
    UnexpectedCharacter,
    UnterminatedString,
    InvalidHexLiteral,
    UnterminatedComment,
    OutOfMemory,
    InvalidEscapeSequence,
    UnterminatedRawString,
    EmptyCharacterLiteral,
    InvalidCharacterLiteral,
    InvalidCharacterInString, // For non-ASCII characters in string literals
    InvalidBinaryLiteral,
    NumberTooLarge,
    InvalidAddressFormat,
    TooManyErrors,
    InvalidBuiltinFunction,
    InvalidRangePattern,
    InvalidSwitchSyntax,
};

// Re-export diagnostic types from error_recovery for backward compatibility
pub const DiagnosticSeverity = error_recovery.DiagnosticSeverity;
pub const ErrorMessageTemplate = error_recovery.ErrorMessageTemplate;
pub const ErrorContext = error_recovery.ErrorContext;
pub const LexerDiagnostic = error_recovery.LexerDiagnostic;
pub const ErrorRecovery = error_recovery.ErrorRecovery;

// Re-export helper functions
pub const extractSourceContext = error_recovery.extractSourceContext;
pub const getErrorTemplate = error_recovery.getErrorTemplate;

// ============================================================================
// SECTION 2: Source Management & Position Tracking
// ============================================================================

/// Source range information for precise token positioning
pub const SourceRange = struct {
    start_line: u32,
    start_column: u32,
    end_line: u32,
    end_column: u32,
    start_offset: u32,
    end_offset: u32,

    pub fn format(self: SourceRange, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("{d}:{d}-{d}:{d} ({d}:{d})", .{ self.start_line, self.start_column, self.end_line, self.end_column, self.start_offset, self.end_offset });
    }
};

/// Processed token values for literals
pub const TokenValue = union(enum) {
    string: []const u8, // Processed string with escapes resolved
    character: u8, // Character literal value
    integer: u256, // Parsed integer value
    binary: u256, // Binary literal value
    hex: u256, // Hex literal value
    address: [20]u8, // Address bytes
    boolean: bool, // Boolean literal value

    pub fn format(self: TokenValue, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .string => |s| try writer.print("string(\"{s}\")", .{s}),
            .character => |c| try writer.print("char('{c}')", .{c}),
            .integer => |i| try writer.print("int({d})", .{i}),
            .binary => |b| try writer.print("bin({d})", .{b}),
            .hex => |h| try writer.print("hex({d})", .{h}),
            .address => |a| {
                try writer.writeAll("addr(0x");
                for (a) |byte| {
                    try writer.print("{x:0>2}", .{byte});
                }
                try writer.writeAll(")");
            },
            .boolean => |b| try writer.print("bool({any})", .{b}),
        }
    }
};

// ============================================================================
// SECTION 3: Trivia Handling (Comments & Whitespace)
// ============================================================================

// Re-export trivia types for backward compatibility
pub const TriviaKind = trivia.TriviaKind;
pub const TriviaPiece = trivia.TriviaPiece;
pub const StringPool = trivia.StringPool;
pub const StringProcessor = trivia.StringProcessor;

// ============================================================================
// SECTION 4: Token Types & Definitions
// ============================================================================

/// Token types for Ora
pub const TokenType = enum {
    // End of file
    Eof,

    // Keywords
    Contract,
    Pub,
    Fn,
    Let,
    Var,
    Const,
    Immutable,
    Storage,
    Memory,
    Tstore,
    Init,
    Log,
    If,
    Else,
    While,
    For,
    Break,
    Continue,
    Return,
    Requires,
    Ensures,
    Invariant,
    Old,
    Result,
    Modifies,
    Decreases,
    Increases,
    Assume,
    Havoc,
    Comptime,
    As, // reserved keyword (not currently used)
    Import,
    Struct,
    Enum,
    True,
    False,

    // Error handling keywords
    Error,
    Try,
    Catch,

    // Control flow keywords
    Switch,

    // Function modifiers
    Ghost,
    Assert,

    // Type keywords
    Void,

    // Transfer/shift keywords
    From,
    To,

    // Quantifier keywords
    Forall,
    Exists,
    Where,

    // Primitive type keywords
    U8,
    U16,
    U32,
    U64,
    U128,
    U256,
    I8,
    I16,
    I32,
    I64,
    I128,
    I256,
    Bool,
    Address,
    String,

    // Collection type keywords
    Map,
    Slice,
    Bytes,

    // Identifiers and literals
    Identifier,
    StringLiteral,
    RawStringLiteral,
    CharacterLiteral,
    IntegerLiteral,
    BinaryLiteral,
    HexLiteral,
    AddressLiteral,
    BytesLiteral,

    // Symbols and operators
    Plus, // +
    Minus, // -
    Star, // *
    Slash, // /
    Percent, // %
    StarStar, // **
    Equal, // =
    EqualEqual, // ==
    BangEqual, // !=
    Less, // <
    LessEqual, // <=
    Greater, // >
    GreaterEqual, // >=
    LessLess, // <<
    GreaterGreater, // >>
    Bang, // !
    Ampersand, // &
    AmpersandAmpersand, // &&
    Pipe, // |
    PipePipe, // ||
    Caret, // ^
    PlusEqual, // +=
    MinusEqual, // -=
    StarEqual, // *=
    SlashEqual, // /=
    PercentEqual, // %=
    Arrow, // ->

    // Delimiters
    LeftParen, // (
    RightParen, // )
    LeftBrace, // {
    RightBrace, // }
    LeftBracket, // [
    RightBracket, // ]
    Comma, // ,
    Semicolon, // ;
    /// :
    Colon, // :
    Dot, // .
    DotDot, // .. (range operator)
    DotDotDot, // ... (range operator)
    At, // @
};

/// Token with enhanced location and value information
pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    range: SourceRange,
    // For literals, store processed value separately from raw lexeme
    value: ?TokenValue = null,

    // Line and column for convenience
    line: u32,
    column: u32,

    // Lossless parsing trivia attachment (leading trivia captured; trailing optional)
    leading_trivia_start: u32 = 0,
    leading_trivia_len: u32 = 0,
    trailing_trivia_start: u32 = 0,
    trailing_trivia_len: u32 = 0,

    pub fn format(self: Token, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll("Token{ .type = ");
        try writer.writeAll(@tagName(self.type));
        try writer.writeAll(", .lexeme = \"");
        try writer.writeAll(self.lexeme);
        try writer.writeAll("\", .range = ");
        try self.range.format("", .{}, writer);
        if (self.value) |val| {
            try writer.writeAll(", .value = ");
            try val.format("", .{}, writer);
        }
        try writer.writeAll(" }");
    }
};

/// Configuration validation errors
pub const LexerConfigError = error{
    InvalidMaxErrors,
    MaxErrorsTooLarge,
    InvalidStringPoolCapacity,
    StringPoolCapacityTooLarge,
    SuggestionsRequireErrorRecovery,
    DiagnosticGroupingRequiresErrorRecovery,
    DiagnosticFilteringRequiresErrorRecovery,
};

/// Performance monitoring for lexer operations
pub const LexerPerformance = struct {
    tokens_scanned: u64 = 0,
    characters_processed: u64 = 0,
    string_interning_hits: u64 = 0,
    string_interning_misses: u64 = 0,
    error_recovery_invocations: u64 = 0,

    pub fn reset(self: *LexerPerformance) void {
        self.* = LexerPerformance{};
    }

    pub fn getTokensPerCharacter(self: *const LexerPerformance) f64 {
        if (self.characters_processed == 0) return 0.0;
        return @as(f64, @floatFromInt(self.tokens_scanned)) / @as(f64, @floatFromInt(self.characters_processed));
    }

    pub fn getStringInterningHitRate(self: *const LexerPerformance) f64 {
        const total = self.string_interning_hits + self.string_interning_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.string_interning_hits)) / @as(f64, @floatFromInt(total));
    }
};

/// Lexer configuration options
pub const LexerConfig = struct {
    // Error recovery configuration
    enable_error_recovery: bool = true,
    max_errors: u32 = 100,
    enable_suggestions: bool = true,

    // Resynchronization after errors
    enable_resync: bool = true,
    resync_max_lookahead: u32 = 256,

    // String processing configuration
    enable_string_interning: bool = true,
    string_pool_initial_capacity: u32 = 256,

    // Performance monitoring configuration
    enable_performance_monitoring: bool = false,

    // Feature toggles
    enable_raw_strings: bool = true,
    enable_character_literals: bool = true,
    enable_binary_literals: bool = true,
    enable_hex_validation: bool = true,
    enable_address_validation: bool = true,
    enable_number_overflow_checking: bool = true,

    // Diagnostic configuration
    enable_diagnostic_grouping: bool = true,
    enable_diagnostic_filtering: bool = true,
    minimum_diagnostic_severity: DiagnosticSeverity = .Error,

    // Strict mode for production use
    strict_mode: bool = false,

    /// Create default configuration
    pub fn default() LexerConfig {
        return LexerConfig{};
    }

    /// Create configuration optimized for performance
    pub fn performance() LexerConfig {
        return LexerConfig{
            .enable_error_recovery = false,
            .enable_suggestions = false,
            .enable_string_interning = true,
            .enable_performance_monitoring = true,
            .enable_diagnostic_grouping = false,
            .enable_diagnostic_filtering = false,
        };
    }

    /// Create configuration optimized for development/IDE usage
    pub fn development() LexerConfig {
        return LexerConfig{
            .enable_error_recovery = true,
            .max_errors = 1000,
            .enable_suggestions = true,
            .enable_string_interning = true,
            .enable_performance_monitoring = false,
            .enable_diagnostic_grouping = true,
            .enable_diagnostic_filtering = true,
            .minimum_diagnostic_severity = .Hint,
        };
    }

    /// Create configuration for strict parsing (no error recovery)
    pub fn strict() LexerConfig {
        return LexerConfig{
            .enable_error_recovery = false,
            .enable_suggestions = false,
            .strict_mode = true,
            .minimum_diagnostic_severity = .Error,
        };
    }

    /// Validate configuration and return errors if invalid
    pub fn validate(self: LexerConfig) LexerConfigError!void {
        if (self.max_errors == 0) {
            return LexerConfigError.InvalidMaxErrors;
        }

        if (self.max_errors > 10000) {
            return LexerConfigError.MaxErrorsTooLarge;
        }

        if (self.string_pool_initial_capacity == 0) {
            return LexerConfigError.InvalidStringPoolCapacity;
        }

        if (self.string_pool_initial_capacity > 1000000) {
            return LexerConfigError.StringPoolCapacityTooLarge;
        }

        // Validate feature combinations
        if (self.enable_suggestions and !self.enable_error_recovery) {
            return LexerConfigError.SuggestionsRequireErrorRecovery;
        }

        if (self.enable_diagnostic_grouping and !self.enable_error_recovery) {
            return LexerConfigError.DiagnosticGroupingRequiresErrorRecovery;
        }

        if (self.enable_diagnostic_filtering and !self.enable_error_recovery) {
            return LexerConfigError.DiagnosticFilteringRequiresErrorRecovery;
        }
    }

    /// Get a description of the configuration
    pub fn describe(self: LexerConfig, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(allocator);

        const writer = buffer.writer(allocator);

        try writer.writeAll("Lexer Configuration:\n");
        try writer.writeAll("==================\n");

        // Error recovery settings
        try writer.print("Error Recovery: {any}\n", .{self.enable_error_recovery});
        if (self.enable_error_recovery) {
            try writer.print("  Max Errors: {d}\n", .{self.max_errors});
            try writer.print("  Suggestions: {any}\n", .{self.enable_suggestions});
            try writer.print("  Resync: {any} (max lookahead {d})\n", .{ self.enable_resync, self.resync_max_lookahead });
        }

        // String processing settings
        try writer.print("String Interning: {any}\n", .{self.enable_string_interning});
        if (self.enable_string_interning) {
            try writer.print("  Initial Capacity: {d}\n", .{self.string_pool_initial_capacity});
        }

        // Feature toggles
        try writer.print("Raw Strings: {any}\n", .{self.enable_raw_strings});
        try writer.print("Character Literals: {any}\n", .{self.enable_character_literals});
        try writer.print("Binary Literals: {any}\n", .{self.enable_binary_literals});
        try writer.print("Hex Validation: {any}\n", .{self.enable_hex_validation});
        try writer.print("Address Validation: {any}\n", .{self.enable_address_validation});
        try writer.print("Number Overflow Checking: {any}\n", .{self.enable_number_overflow_checking});

        // Diagnostic settings
        try writer.print("Diagnostic Grouping: {any}\n", .{self.enable_diagnostic_grouping});
        try writer.print("Diagnostic Filtering: {any}\n", .{self.enable_diagnostic_filtering});
        try writer.print("Minimum Severity: {any}\n", .{self.minimum_diagnostic_severity});

        // General settings
        try writer.print("Strict Mode: {any}\n", .{self.strict_mode});
        try writer.print("Performance Monitoring: {any}\n", .{self.enable_performance_monitoring});

        return buffer.toOwnedSlice(allocator);
    }
};

/// Lexer feature enumeration for feature checking
pub const LexerFeature = enum {
    ErrorRecovery,
    Suggestions,
    StringInterning,
    PerformanceMonitoring,
    RawStrings,
    CharacterLiterals,
    BinaryLiterals,
    HexValidation,
    AddressValidation,
    NumberOverflowChecking,
    DiagnosticGrouping,
    DiagnosticFiltering,
};

/// Keywords map for efficient lookup
pub const keywords = std.StaticStringMap(TokenType).initComptime(.{
    .{ "contract", .Contract },
    .{ "pub", .Pub },
    .{ "fn", .Fn },
    .{ "let", .Let },
    .{ "var", .Var },
    .{ "const", .Const },
    .{ "immutable", .Immutable },
    .{ "storage", .Storage },
    .{ "memory", .Memory },
    .{ "tstore", .Tstore },
    .{ "init", .Init },
    .{ "log", .Log },
    .{ "if", .If },
    .{ "else", .Else },
    .{ "while", .While },
    .{ "for", .For },
    .{ "break", .Break },
    .{ "continue", .Continue },
    .{ "return", .Return },
    .{ "requires", .Requires },
    .{ "ensures", .Ensures },
    .{ "invariant", .Invariant },
    .{ "old", .Old },
    .{ "result", .Result },
    .{ "modifies", .Modifies },
    .{ "decreases", .Decreases },
    .{ "increases", .Increases },
    .{ "assume", .Assume },
    .{ "havoc", .Havoc },
    .{ "ghost", .Ghost },
    .{ "switch", .Switch },
    .{ "void", .Void },
    .{ "comptime", .Comptime },
    .{ "as", .As },
    .{ "import", .Import },
    .{ "struct", .Struct },
    .{ "enum", .Enum },
    .{ "true", .True },
    .{ "false", .False },
    .{ "error", .Error },
    .{ "try", .Try },
    .{ "catch", .Catch },
    .{ "from", .From },
    .{ "to", .To },
    .{ "forall", .Forall },
    .{ "exists", .Exists },
    .{ "where", .Where },
    .{ "assert", .Assert },
    .{ "u8", .U8 },
    .{ "u16", .U16 },
    .{ "u32", .U32 },
    .{ "u64", .U64 },
    .{ "u128", .U128 },
    .{ "u256", .U256 },
    .{ "i8", .I8 },
    .{ "i16", .I16 },
    .{ "i32", .I32 },
    .{ "i64", .I64 },
    .{ "i128", .I128 },
    .{ "i256", .I256 },
    .{ "bool", .Bool },
    .{ "address", .Address },
    .{ "string", .String },
    .{ "map", .Map },
    .{ "slice", .Slice },
    .{ "bytes", .Bytes },
});

// ============================================================================
// SECTION 5: Core Lexer Logic & State Machine
// ============================================================================

/// Lexer for Ora
pub const Lexer = struct {
    source: []const u8,
    tokens: std.ArrayList(Token),
    trivia: std.ArrayList(TriviaPiece),
    start: u32,
    current: u32,
    line: u32,
    column: u32,
    start_column: u32, // Track start position for accurate token positioning
    last_bad_char: ?u8, // Track the character that caused an error
    allocator: Allocator,

    // Arena allocator for string processing - all processed strings live here
    arena: std.heap.ArenaAllocator,

    // Error recovery system
    error_recovery: ?ErrorRecovery,
    config: LexerConfig,

    // String interning system
    string_pool: ?StringPool,

    // Performance monitoring
    performance: ?LexerPerformance,

    pub fn init(allocator: Allocator, source: []const u8) Lexer {
        return Lexer{
            .source = source,
            .tokens = std.ArrayList(Token){},
            .trivia = std.ArrayList(TriviaPiece){},
            .start = 0,
            .current = 0,
            .line = 1,
            .column = 1,
            .start_column = 1,
            .last_bad_char = null,
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .error_recovery = null,
            .config = LexerConfig.default(),
            .string_pool = null,
            .performance = null,
        };
    }

    /// Initialize lexer with custom configuration
    pub fn initWithConfig(allocator: Allocator, source: []const u8, config: LexerConfig) LexerConfigError!Lexer {
        // Validate configuration first
        try config.validate();

        var lexer = Lexer.init(allocator, source);
        lexer.config = config;

        // Initialize optional components based on configuration
        if (config.enable_error_recovery) {
            lexer.error_recovery = ErrorRecovery.init(allocator, config.max_errors);
        }

        if (config.enable_string_interning) {
            lexer.string_pool = trivia.StringPool.init(allocator);
            // Pre-allocate capacity if specified
            if (config.string_pool_initial_capacity > 0) {
                lexer.string_pool.?.strings.ensureTotalCapacity(config.string_pool_initial_capacity) catch {
                    // If allocation fails, continue with default capacity
                };
            }
        }

        if (config.enable_performance_monitoring) {
            lexer.performance = LexerPerformance{};
        }

        return lexer;
    }

    pub fn deinit(self: *Lexer) void {
        // Free the arena - this automatically frees all processed strings
        self.arena.deinit();

        // Free other components
        self.tokens.deinit(self.allocator);
        self.trivia.deinit(self.allocator);
        if (self.error_recovery) |*recovery| {
            recovery.deinit();
        }
        if (self.string_pool) |*pool| {
            pool.deinit();
        }
    }

    /// Get details about the last error
    pub fn getErrorDetails(self: *Lexer, allocator: Allocator) ![]u8 {
        if (self.last_bad_char) |c| {
            if (std.ascii.isPrint(c)) {
                return std.fmt.allocPrint(allocator, "Unexpected character '{c}' at line {}, column {}", .{ c, self.line, self.column - 1 });
            } else {
                return std.fmt.allocPrint(allocator, "Unexpected character (ASCII {}) at line {}, column {}", .{ c, self.line, self.column - 1 });
            }
        }
        return std.fmt.allocPrint(allocator, "Error at line {}, column {}", .{ self.line, self.column });
    }

    /// Get all collected diagnostics from error recovery
    pub fn getDiagnostics(self: *Lexer) []const LexerDiagnostic {
        if (self.error_recovery) |*recovery| {
            return recovery.getErrors();
        }
        return &[_]LexerDiagnostic{};
    }

    /// Create a diagnostic report with all errors and warnings
    pub fn createDiagnosticReport(self: *Lexer) ![]u8 {
        if (self.error_recovery) |*recovery| {
            return recovery.createDetailedReport(self.allocator);
        }
        return self.allocator.dupe(u8, "No diagnostics available");
    }

    /// Check if error recovery is enabled
    pub fn hasErrorRecovery(self: *Lexer) bool {
        return self.error_recovery != null;
    }

    /// Get diagnostics filtered by severity
    pub fn getDiagnosticsBySeverity(self: *Lexer, severity: DiagnosticSeverity) std.ArrayList(LexerDiagnostic) {
        if (self.error_recovery) |*recovery| {
            return recovery.getErrorsBySeverity(severity);
        }
        return std.ArrayList(LexerDiagnostic){};
    }

    /// Get diagnostics grouped by type
    pub fn getDiagnosticsByType(self: *Lexer) ?std.ArrayList(ErrorRecovery.ErrorTypeCount) {
        if (self.error_recovery) |*recovery| {
            return recovery.getErrorsByType();
        }
        return null;
    }

    /// Get diagnostics grouped by line number
    pub fn getDiagnosticsByLine(self: *Lexer) ?std.ArrayList(ErrorRecovery.LineCount) {
        if (self.error_recovery) |*recovery| {
            return recovery.getErrorsByLine();
        }
        return null;
    }

    /// Create a summary report of all diagnostics
    pub fn createDiagnosticSummary(self: *Lexer, allocator: Allocator) ![]u8 {
        if (self.error_recovery) |*recovery| {
            return recovery.createSummaryReport(allocator);
        }
        return try std.fmt.allocPrint(allocator, "No diagnostics available", .{});
    }

    /// Filter diagnostics by minimum severity level
    pub fn filterDiagnosticsBySeverity(self: *Lexer, min_severity: DiagnosticSeverity) std.ArrayList(LexerDiagnostic) {
        if (self.error_recovery) |*recovery| {
            return recovery.filterByMinimumSeverity(min_severity);
        }
        return std.ArrayList(LexerDiagnostic){};
    }

    /// Get related diagnostics (same line or nearby)
    pub fn getRelatedDiagnostics(self: *Lexer, diagnostic: LexerDiagnostic, max_distance: u32) std.ArrayList(LexerDiagnostic) {
        if (self.error_recovery) |*recovery| {
            return recovery.getRelatedErrors(diagnostic, max_distance);
        }
        return std.ArrayList(LexerDiagnostic){};
    }

    /// Get current lexer configuration
    pub fn getConfig(self: *const Lexer) LexerConfig {
        return self.config;
    }

    /// Update lexer configuration (only affects future operations)
    pub fn updateConfig(self: *Lexer, new_config: LexerConfig) LexerConfigError!void {
        // Validate new configuration
        try new_config.validate();

        // Update configuration
        self.config = new_config;

        // Reinitialize components if needed
        if (new_config.enable_error_recovery and self.error_recovery == null) {
            self.error_recovery = ErrorRecovery.init(self.allocator, new_config.max_errors);
        } else if (!new_config.enable_error_recovery and self.error_recovery != null) {
            self.error_recovery.?.deinit();
            self.error_recovery = null;
        } else if (new_config.enable_error_recovery and self.error_recovery != null) {
            // Update max errors if changed
            self.error_recovery.?.max_errors = new_config.max_errors;
        }

        if (new_config.enable_string_interning and self.string_pool == null) {
            self.string_pool = trivia.StringPool.init(self.allocator);
        } else if (!new_config.enable_string_interning and self.string_pool != null) {
            self.string_pool.?.deinit();
            self.string_pool = null;
        }

        if (new_config.enable_performance_monitoring and self.performance == null) {
            self.performance = LexerPerformance{};
        } else if (!new_config.enable_performance_monitoring) {
            self.performance = null;
        }
    }

    /// Check if a specific feature is enabled
    pub fn isFeatureEnabled(self: *const Lexer, feature: LexerFeature) bool {
        return switch (feature) {
            .ErrorRecovery => self.config.enable_error_recovery,
            .Suggestions => self.config.enable_suggestions,
            .StringInterning => self.config.enable_string_interning,
            .PerformanceMonitoring => self.config.enable_performance_monitoring,
            .RawStrings => self.config.enable_raw_strings,
            .CharacterLiterals => self.config.enable_character_literals,
            .BinaryLiterals => self.config.enable_binary_literals,
            .HexValidation => self.config.enable_hex_validation,
            .AddressValidation => self.config.enable_address_validation,
            .NumberOverflowChecking => self.config.enable_number_overflow_checking,
            .DiagnosticGrouping => self.config.enable_diagnostic_grouping,
            .DiagnosticFiltering => self.config.enable_diagnostic_filtering,
        };
    }

    /// Get performance statistics (if monitoring is enabled)
    pub fn getPerformanceStats(self: *const Lexer) ?LexerPerformance {
        return self.performance;
    }

    /// Reset performance statistics
    pub fn resetPerformanceStats(self: *Lexer) void {
        if (self.performance) |*perf| {
            perf.reset();
        }
    }

    /// Get string pool statistics (if interning is enabled)
    pub fn getStringPoolStats(self: *const Lexer) ?struct { count: u32, capacity: u32 } {
        if (self.string_pool) |*pool| {
            return .{
                .count = pool.count(),
                .capacity = @intCast(pool.strings.capacity()),
            };
        }
        return null;
    }

    /// Clear string pool (if interning is enabled)
    pub fn clearStringPool(self: *Lexer) void {
        if (self.string_pool) |*pool| {
            pool.clear();
        }
    }

    // ========================================================================
    // UTILITY METHODS
    // ========================================================================

    /// Check if lexer has any errors
    pub fn hasErrors(self: *const Lexer) bool {
        if (self.error_recovery) |recovery| {
            return recovery.getErrorCount() > 0;
        }
        return self.last_bad_char != null;
    }

    /// Get error count
    pub fn getErrorCount(self: *const Lexer) usize {
        if (self.error_recovery) |recovery| {
            return recovery.getErrorCount();
        }
        return if (self.last_bad_char != null) 1 else 0;
    }

    /// Get token at specific position
    pub fn getTokenAt(self: *const Lexer, line: u32, column: u32) ?Token {
        for (self.tokens.items) |token| {
            if (token.range.start_line == line and
                token.range.start_column <= column and
                token.range.end_column > column)
            {
                return token;
            }
        }
        return null;
    }

    /// Get all tokens
    pub fn getTokens(self: *const Lexer) []const Token {
        return self.tokens.items;
    }

    pub fn getTrivia(self: *const Lexer) []const TriviaPiece {
        return self.trivia.items;
    }

    /// Reset lexer state for reuse
    pub fn reset(self: *Lexer) void {
        self.tokens.clearAndFree(self.allocator);
        self.start = 0;
        self.current = 0;
        self.line = 1;
        self.column = 1;
        self.start_column = 1;
        self.last_bad_char = null;

        if (self.error_recovery) |*recovery| {
            recovery.clear();
        }

        if (self.string_pool) |*pool| {
            pool.clear();
        }

        if (self.performance) |*perf| {
            perf.reset();
        }
    }

    /// Set new source and reset lexer state
    pub fn setSource(self: *Lexer, source: []const u8) void {
        self.source = source;
        self.reset();
    }

    /// Record an error during scanning
    pub fn recordError(self: *Lexer, error_type: LexerError, message: []const u8) void {
        if (self.error_recovery) |*recovery| {
            const range = SourceRange{
                .start_line = self.line,
                .start_column = self.start_column,
                .end_line = self.line,
                .end_column = self.column,
                .start_offset = self.start,
                .end_offset = self.current,
            };
            recovery.recordDetailedError(error_type, range, self.source, message) catch {
                // If we can't record more errors, we've hit the limit
                return;
            };
            // Attempt resynchronization if enabled and appropriate for this error
            if (self.config.enable_resync and shouldResyncOnError(error_type)) {
                self.resyncToBoundary(self.config.resync_max_lookahead);
            }
        }
    }

    /// Record an error with a suggestion during scanning
    fn recordErrorWithSuggestion(self: *Lexer, error_type: LexerError, message: []const u8, suggestion: []const u8) void {
        if (self.error_recovery) |*recovery| {
            const range = SourceRange{
                .start_line = self.line,
                .start_column = self.start_column,
                .end_line = self.line,
                .end_column = self.column,
                .start_offset = self.start,
                .end_offset = self.current,
            };
            recovery.recordDetailedErrorWithSuggestion(error_type, range, self.source, message, suggestion) catch {
                // If we can't record more errors, we've hit the limit
                return;
            };
            if (self.config.enable_resync and shouldResyncOnError(error_type)) {
                self.resyncToBoundary(self.config.resync_max_lookahead);
            }
        }
    }

    /// Advance input to a safe boundary to resume lexing after an error
    fn resyncToBoundary(self: *Lexer, max_lookahead: u32) void {
        var walked: u32 = 0;
        while (!self.isAtEnd() and walked < max_lookahead) {
            const c = self.peek();
            // Hard boundaries: newline, braces, parens, brackets, statement delimiters
            if (c == '\n' or c == '{' or c == '}' or c == '(' or c == ')' or c == '[' or c == ']' or c == ';' or c == ',' or c == ':') {
                break;
            }
            _ = self.advance();
            walked += 1;
        }
        // Do not consume the boundary; allow normal scanning to handle it
    }

    /// Decide whether a boundary resync is appropriate for the given error
    fn shouldResyncOnError(error_type: LexerError) bool {
        return error_type == LexerError.UnterminatedString or
            error_type == LexerError.UnterminatedRawString or
            error_type == LexerError.UnterminatedComment or
            error_type == LexerError.InvalidEscapeSequence or
            error_type == LexerError.InvalidCharacterLiteral or
            error_type == LexerError.EmptyCharacterLiteral;
    }

    /// Intern a string if string interning is enabled, otherwise return the original
    fn internString(self: *Lexer, string: []const u8) LexerError![]const u8 {
        if (self.string_pool) |*pool| {
            // Track performance metrics if enabled
            if (self.performance) |*perf| {
                const hash_value = trivia.StringPool.hash(string);
                if (pool.strings.get(hash_value)) |interned_string| {
                    if (std.mem.eql(u8, interned_string, string)) {
                        perf.string_interning_hits += 1;
                    } else {
                        perf.string_interning_misses += 1;
                    }
                } else {
                    perf.string_interning_misses += 1;
                }
            }

            return pool.intern(string) catch |err| switch (err) {
                error.OutOfMemory => return LexerError.OutOfMemory,
            };
        }
        return string;
    }

    /// Tokenize the entire source
    pub fn scanTokens(self: *Lexer) LexerError![]Token {
        // Pre-allocate capacity based on source length estimate (1 token per ~8 characters)
        const estimated_tokens = @max(32, self.source.len / 8);
        try self.tokens.ensureTotalCapacity(self.allocator, estimated_tokens);

        while (!self.isAtEnd()) {
            self.start = self.current;
            self.start_column = self.column;
            // Capture leading trivia for the next token
            const trivia_start = self.trivia.items.len;
            try trivia.captureLeadingTrivia(self);
            const trivia_len: u32 = @intCast(self.trivia.items.len - trivia_start);
            self.start = self.current;
            self.start_column = self.column;
            if (self.isAtEnd()) break;
            // Debug: print current position and character

            //if (self.current < self.source.len) {
            //    const current_char = self.source[self.current];
            //    std.debug.print("Scanning at pos {}: '{}' (line {}, col {})\n", .{ self.current, current_char, self.line, self.column });
            //}

            self.scanToken() catch |err| {
                // If error recovery is enabled, continue scanning
                if (self.hasErrorRecovery()) {
                    // Record the error and continue
                    self.recordError(err, "Lexer error occurred");
                    // For localized errors (e.g., unexpected char), advancing one char is sufficient.
                    // Long-span errors will trigger boundary resync inside recordError.
                    if (!self.isAtEnd()) {
                        _ = self.advance();
                    }
                } else {
                    // If no error recovery, re-raise the error
                    return err;
                }
            };

            // Attach captured trivia as leading trivia for the token we just added
            if (self.tokens.items.len > 0 and trivia_len > 0) {
                var last = &self.tokens.items[self.tokens.items.len - 1];
                last.leading_trivia_start = @as(u32, @intCast(trivia_start));
                last.leading_trivia_len = trivia_len;
            }
        }

        // Add EOF token
        const eof_range = SourceRange{
            .start_line = self.line,
            .start_column = self.column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.current,
            .end_offset = self.current,
        };

        // Track performance metrics for EOF token if enabled
        if (self.performance) |*perf| {
            perf.tokens_scanned += 1;
        }

        try self.tokens.append(self.allocator, Token{
            .type = .Eof,
            .lexeme = "",
            .range = eof_range,
            .value = null,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.column,
        });

        return self.tokens.toOwnedSlice(self.allocator);
    }

    fn scanToken(self: *Lexer) LexerError!void {
        // Track performance metrics if enabled
        if (self.performance) |*perf| {
            perf.characters_processed += 1;
        }

        const c = self.advance();

        // Whitespace/comments are handled by captureLeadingTrivia()
        if (isWhitespace(c)) return; // should be rare due to captureLeadingTrivia

        switch (c) {

            // Single character tokens
            '(' => try self.addToken(.LeftParen),
            ')' => try self.addToken(.RightParen),
            '{' => try self.addToken(.LeftBrace),
            '}' => try self.addToken(.RightBrace),
            '[' => try self.addToken(.LeftBracket),
            ']' => try self.addToken(.RightBracket),
            ',' => try self.addToken(.Comma),
            ';' => try self.addToken(.Semicolon),
            ':' => try self.addToken(.Colon),
            '.' => {
                // Look ahead to distinguish between '.', '..', and '...'
                if (self.peek() == '.') {
                    if (self.peekNext() == '.') {
                        // We have "..." - consume the remaining two dots
                        _ = self.advance(); // consume second '.'
                        _ = self.advance(); // consume third '.'
                        try self.addToken(.DotDotDot);
                    } else {
                        // We have ".." - consume the second dot
                        _ = self.advance(); // consume second '.'
                        try self.addToken(.DotDot);
                    }
                } else {
                    // Single dot - don't consume any additional characters
                    try self.addToken(.Dot);
                }
            },
            '@' => try identifier_scanners.scanAtDirective(self),
            '^' => try self.addToken(.Caret),

            // Operators that might have compound forms
            '+' => {
                if (self.match('=')) {
                    try self.addToken(.PlusEqual);
                } else {
                    try self.addToken(.Plus);
                }
            },
            '-' => {
                if (self.match('=')) {
                    try self.addToken(.MinusEqual);
                } else if (self.match('>')) {
                    try self.addToken(.Arrow);
                } else {
                    try self.addToken(.Minus);
                }
            },
            '*' => {
                if (self.match('=')) {
                    try self.addToken(.StarEqual);
                } else if (self.match('*')) {
                    try self.addToken(.StarStar);
                } else {
                    try self.addToken(.Star);
                }
            },
            '/' => {
                if (self.match('/')) {
                    // Single-line comment already captured by captureLeadingTrivia
                    while (self.peek() != '\n' and !self.isAtEnd()) {
                        _ = self.advance();
                    }
                    return; // do not emit token
                } else if (self.match('*')) {
                    // Multi-line comment already captured by captureLeadingTrivia
                    try self.scanMultiLineComment();
                    return; // do not emit token
                } else if (self.match('=')) {
                    try self.addToken(.SlashEqual);
                } else {
                    try self.addToken(.Slash);
                }
            },
            '%' => {
                if (self.match('=')) {
                    try self.addToken(.PercentEqual);
                } else {
                    try self.addToken(.Percent);
                }
            },
            '!' => {
                if (self.match('=')) {
                    try self.addToken(.BangEqual);
                } else {
                    try self.addToken(.Bang);
                }
            },
            '=' => {
                // Support '=>' fat arrow used in switch arms and quantified expressions
                if (self.match('>')) {
                    try self.addToken(.Arrow);
                } else if (self.match('=')) {
                    try self.addToken(.EqualEqual);
                } else {
                    try self.addToken(.Equal);
                }
            },
            '<' => {
                if (self.match('=')) {
                    try self.addToken(.LessEqual);
                } else if (self.match('<')) {
                    try self.addToken(.LessLess);
                } else {
                    try self.addToken(.Less);
                }
            },
            '>' => {
                if (self.match('=')) {
                    try self.addToken(.GreaterEqual);
                } else if (self.match('>')) {
                    try self.addToken(.GreaterGreater);
                } else {
                    try self.addToken(.Greater);
                }
            },
            '&' => {
                if (self.match('&')) {
                    try self.addToken(.AmpersandAmpersand);
                } else {
                    try self.addToken(.Ampersand);
                }
            },
            '|' => {
                if (self.match('|')) {
                    try self.addToken(.PipePipe);
                } else {
                    try self.addToken(.Pipe);
                }
            },

            // String literals
            '"' => try string_scanners.scanString(self),
            'r' => {
                // Check for raw string literal r"..."
                if (self.peek() == '"') {
                    _ = self.advance(); // consume the "
                    try string_scanners.scanRawString(self);
                } else {
                    // Regular identifier starting with 'r'
                    try identifier_scanners.scanIdentifier(self);
                }
            },
            'h' => {
                // Check for hex bytes literal hex"..."
                if (self.peek() == 'e' and self.peekNext() == 'x' and self.peekNextNext() == '"') {
                    // We have "hex" - consume 'e', 'x', and '"'
                    _ = self.advance(); // consume 'e'
                    _ = self.advance(); // consume 'x'
                    _ = self.advance(); // consume '"'
                    try string_scanners.scanHexBytes(self);
                } else {
                    // Regular identifier starting with 'h'
                    try identifier_scanners.scanIdentifier(self);
                }
            },

            // Character literals
            '\'' => try string_scanners.scanCharacter(self),

            // Number literals (including hex and addresses)
            '0' => {
                if (self.match('x') or self.match('X')) {
                    try number_scanners.scanHexLiteral(self);
                } else if (self.match('b') or self.match('B')) {
                    try number_scanners.scanBinaryLiteral(self);
                } else {
                    try number_scanners.scanNumber(self);
                }
            },

            else => {
                if (isDigit(c)) {
                    try number_scanners.scanNumber(self);
                } else if (isAlpha(c)) {
                    try identifier_scanners.scanIdentifier(self);
                } else {
                    // Invalid character - use error recovery if enabled
                    self.last_bad_char = c;
                    if (self.hasErrorRecovery()) {
                        var message_buf: [128]u8 = undefined;
                        const message = if (std.ascii.isPrint(c))
                            std.fmt.bufPrint(&message_buf, "Unexpected character '{c}'", .{c}) catch "Unexpected character"
                        else
                            std.fmt.bufPrint(&message_buf, "Unexpected character (ASCII {})", .{c}) catch "Unexpected character";

                        // Get suggestion for this error
                        const context = self.source[self.current - 1 .. @min(self.current + 3, self.source.len)];
                        if (ErrorRecovery.suggestFix(LexerError.UnexpectedCharacter, context)) |suggestion| {
                            self.recordErrorWithSuggestion(LexerError.UnexpectedCharacter, message, suggestion);
                        } else {
                            self.recordError(LexerError.UnexpectedCharacter, message);
                        }

                        // Use error recovery to find next safe boundary
                        const next_boundary = ErrorRecovery.findNextTokenBoundary(self.source, self.current);
                        if (next_boundary > self.current) {
                            // Skip to the next safe boundary
                            while (self.current < next_boundary and !self.isAtEnd()) {
                                if (self.peek() == '\n') {
                                    self.line += 1;
                                    self.column = 1;
                                }
                                _ = self.advance();
                            }
                        }
                        // Continue scanning after recovery
                    } else {
                        return LexerError.UnexpectedCharacter;
                    }
                }
            },
        }
    }

    fn scanMultiLineComment(self: *Lexer) LexerError!void {
        var nesting: u32 = 1;

        while (nesting > 0 and !self.isAtEnd()) {
            if (self.peek() == '/' and self.peekNext() == '*') {
                _ = self.advance(); // consume '/'
                _ = self.advance(); // consume '*'
                nesting += 1;
            } else if (self.peek() == '*' and self.peekNext() == '/') {
                _ = self.advance(); // consume '*'
                _ = self.advance(); // consume '/'
                nesting -= 1;
            } else if (self.peek() == '\n') {
                self.line += 1;
                self.column = 1;
                _ = self.advance();
            } else {
                _ = self.advance();
            }
        }

        if (nesting > 0) {
            // Unclosed comment
            return LexerError.UnterminatedComment;
        }
    }

    pub fn isAtEnd(self: *Lexer) bool {
        return self.current >= self.source.len;
    }

    pub fn advance(self: *Lexer) u8 {
        const c = self.source[self.current];
        self.current += 1;
        self.column += 1;
        return c;
    }

    fn match(self: *Lexer, expected: u8) bool {
        if (self.isAtEnd()) return false;
        if (self.source[self.current] != expected) return false;

        self.current += 1;
        self.column += 1;
        return true;
    }

    pub fn peek(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    pub fn peekNext(self: *Lexer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    pub fn peekNextNext(self: *Lexer) u8 {
        if (self.current + 2 >= self.source.len) return 0;
        return self.source[self.current + 2];
    }

    pub fn addToken(self: *Lexer, token_type: TokenType) LexerError!void {
        const text = self.source[self.start..self.current];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Create token value for boolean literals
        var token_value: ?TokenValue = null;
        if (token_type == .True) {
            token_value = TokenValue{ .boolean = true };
        } else if (token_type == .False) {
            token_value = TokenValue{ .boolean = false };
        }

        // Track performance metrics if enabled
        if (self.performance) |*perf| {
            perf.tokens_scanned += 1;
        }

        try self.tokens.append(self.allocator, Token{
            .type = token_type,
            .lexeme = text,
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    pub fn addTokenWithInterning(self: *Lexer, token_type: TokenType) LexerError!void {
        const text = self.source[self.start..self.current];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Intern the string for identifiers and keywords to reduce memory usage
        const interned_text = try self.internString(text);

        // Create token value for boolean literals
        var token_value: ?TokenValue = null;
        if (token_type == .True) {
            token_value = TokenValue{ .boolean = true };
        } else if (token_type == .False) {
            token_value = TokenValue{ .boolean = false };
        }

        // Track performance metrics if enabled
        if (self.performance) |*perf| {
            perf.tokens_scanned += 1;
        }

        try self.tokens.append(self.allocator, Token{
            .type = token_type,
            .lexeme = interned_text,
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    /// Find the next safe token boundary for error recovery (instance method)
    fn findNextTokenBoundary(self: *Lexer) u32 {
        return ErrorRecovery.findNextTokenBoundary(self.source, self.current);
    }
};

/// Convenience function for testing - tokenizes source and returns tokens
pub fn scan(source: []const u8, allocator: Allocator) LexerError![]Token {
    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();
    return lexer.scanTokens();
}

/// Convenience function with configuration - tokenizes source with custom config
pub fn scanWithConfig(source: []const u8, allocator: Allocator, config: LexerConfig) (LexerError || LexerConfigError)![]Token {
    var lexer = try Lexer.initWithConfig(allocator, source, config);
    defer lexer.deinit();
    return lexer.scanTokens();
}

// Character classification lookup tables for performance optimization
const CHAR_DIGIT = 0x01;
const CHAR_ALPHA = 0x02;
const CHAR_HEX = 0x04;
const CHAR_BINARY = 0x08;
const CHAR_WHITESPACE = 0x10;
const CHAR_IDENTIFIER_START = 0x20;
const CHAR_IDENTIFIER_CONTINUE = 0x40;

// Pre-computed character classification table
const char_table: [256]u8 = blk: {
    var table: [256]u8 = [_]u8{0} ** 256;

    // Digits
    for ('0'..'9' + 1) |c| {
        table[c] |= CHAR_DIGIT | CHAR_HEX | CHAR_IDENTIFIER_CONTINUE;
    }

    // Binary digits
    table['0'] |= CHAR_BINARY;
    table['1'] |= CHAR_BINARY;

    // Lowercase letters
    for ('a'..'z' + 1) |c| {
        table[c] |= CHAR_ALPHA | CHAR_IDENTIFIER_START | CHAR_IDENTIFIER_CONTINUE;
    }

    // Uppercase letters
    for ('A'..'Z' + 1) |c| {
        table[c] |= CHAR_ALPHA | CHAR_IDENTIFIER_START | CHAR_IDENTIFIER_CONTINUE;
    }

    // Hex digits
    for ('a'..'f' + 1) |c| {
        table[c] |= CHAR_HEX;
    }
    for ('A'..'F' + 1) |c| {
        table[c] |= CHAR_HEX;
    }

    // Underscore
    table['_'] |= CHAR_ALPHA | CHAR_IDENTIFIER_START | CHAR_IDENTIFIER_CONTINUE;

    // Whitespace
    table[' '] |= CHAR_WHITESPACE;
    table['\t'] |= CHAR_WHITESPACE;
    table['\r'] |= CHAR_WHITESPACE;
    table['\n'] |= CHAR_WHITESPACE;

    break :blk table;
};

// Optimized helper functions using lookup tables
pub inline fn isDigit(c: u8) bool {
    return (char_table[c] & CHAR_DIGIT) != 0;
}

pub inline fn isHexDigit(c: u8) bool {
    return (char_table[c] & CHAR_HEX) != 0;
}

pub inline fn isBinaryDigit(c: u8) bool {
    return (char_table[c] & CHAR_BINARY) != 0;
}

pub inline fn isAlpha(c: u8) bool {
    return (char_table[c] & CHAR_ALPHA) != 0;
}

pub inline fn isAlphaNumeric(c: u8) bool {
    return (char_table[c] & CHAR_IDENTIFIER_CONTINUE) != 0;
}

pub inline fn isIdentifierStart(c: u8) bool {
    return (char_table[c] & CHAR_IDENTIFIER_START) != 0;
}

pub inline fn isWhitespace(c: u8) bool {
    return (char_table[c] & CHAR_WHITESPACE) != 0;
}

// Token utility functions for parser use
pub fn isKeyword(token_type: TokenType) bool {
    return switch (token_type) {
        .Contract, .Pub, .Fn, .Let, .Var, .Const, .Immutable, .Storage, .Memory, .Tstore, .Init, .Log, .If, .Else, .While, .For, .Break, .Continue, .Return, .Requires, .Ensures, .Invariant, .Old, .Result, .Modifies, .Decreases, .Increases, .Assume, .Havoc, .Switch, .Ghost, .Assert, .Void, .Comptime, .As, .Import, .Struct, .Enum, .True, .False, .Error, .Try, .Catch, .From, .To, .Forall, .Exists, .Where, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .Bool, .Address, .String, .Map, .Slice, .Bytes => true,
        else => false,
    };
}

pub fn isLiteral(token_type: TokenType) bool {
    return switch (token_type) {
        .StringLiteral, .RawStringLiteral, .CharacterLiteral, .IntegerLiteral, .BinaryLiteral, .HexLiteral, .AddressLiteral, .True, .False => true,
        else => false,
    };
}

pub fn isOperator(token_type: TokenType) bool {
    return switch (token_type) {
        .Plus, .Minus, .Star, .Slash, .Percent, .StarStar, .Equal, .EqualEqual, .BangEqual, .Less, .LessEqual, .Greater, .GreaterEqual, .Bang, .Ampersand, .Pipe, .Caret, .LessLess, .GreaterGreater, .PlusEqual, .MinusEqual, .StarEqual, .SlashEqual, .PercentEqual, .AmpersandAmpersand, .PipePipe, .Arrow, .DotDot, .DotDotDot => true,
        else => false,
    };
}

pub fn isDelimiter(token_type: TokenType) bool {
    return switch (token_type) {
        .LeftParen, .RightParen, .LeftBrace, .RightBrace, .LeftBracket, .RightBracket, .Comma, .Semicolon, .Colon, .Dot, .DotDot, .DotDotDot, .At => true,
        else => false,
    };
}
