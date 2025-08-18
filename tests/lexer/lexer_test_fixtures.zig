//! Lexer Test Fixtures - Predefined test data for lexer testing
//!
//! This module provides structured test fixtures for comprehensive lexer testing,
//! including valid token sequences, error cases, and edge cases.

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import lexer types
const lexer_module = @import("ora");
const TokenType = lexer_module.TokenType;
const LexerError = lexer_module.LexerError;

/// Test fixture for valid token sequences
pub const ValidTokenFixture = struct {
    input: []const u8,
    expected_tokens: []const ExpectedToken,
    description: []const u8,
};

/// Expected token for validation
pub const ExpectedToken = struct {
    token_type: TokenType,
    lexeme: []const u8,
    line: u32 = 1,
    column: u32 = 1,
};

/// Test fixture for error cases
pub const ErrorTestFixture = struct {
    input: []const u8,
    expected_error: LexerError,
    error_position: ErrorPosition,
    description: []const u8,
    expected_message_contains: ?[]const u8 = null,
    expected_suggestion: ?[]const u8 = null,
};

/// Expected error position
pub const ErrorPosition = struct {
    line: u32,
    column: u32,
};

/// Lexer test fixtures collection
pub const LexerTestFixtures = struct {
    /// Valid token sequences for testing token generation
    pub const VALID_TOKENS = struct {
        /// Identifier test cases
        pub const IDENTIFIERS = [_]ValidTokenFixture{
            .{
                .input = "identifier",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Identifier, .lexeme = "identifier" },
                },
                .description = "Simple identifier",
            },
            .{
                .input = "_underscore",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Identifier, .lexeme = "_underscore" },
                },
                .description = "Identifier starting with underscore",
            },
            .{
                .input = "camelCase",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Identifier, .lexeme = "camelCase" },
                },
                .description = "CamelCase identifier",
            },
            .{
                .input = "UPPER_CASE",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Identifier, .lexeme = "UPPER_CASE" },
                },
                .description = "Upper case identifier with underscore",
            },
            .{
                .input = "with123numbers",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Identifier, .lexeme = "with123numbers" },
                },
                .description = "Identifier with numbers",
            },
        };

        /// Keyword test cases
        pub const KEYWORDS = [_]ValidTokenFixture{
            .{
                .input = "contract",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Contract, .lexeme = "contract" },
                },
                .description = "Contract keyword",
            },
            .{
                .input = "fn",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Fn, .lexeme = "fn" },
                },
                .description = "Function keyword",
            },
            .{
                .input = "let",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Let, .lexeme = "let" },
                },
                .description = "Let keyword",
            },
            .{
                .input = "if",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .If, .lexeme = "if" },
                },
                .description = "If keyword",
            },
            .{
                .input = "true",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .True, .lexeme = "true" },
                },
                .description = "True boolean keyword",
            },
            .{
                .input = "false",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .False, .lexeme = "false" },
                },
                .description = "False boolean keyword",
            },
        };

        /// Number literal test cases
        pub const NUMBERS = [_]ValidTokenFixture{
            .{
                .input = "42",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .IntegerLiteral, .lexeme = "42" },
                },
                .description = "Simple integer",
            },
            .{
                .input = "0",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .IntegerLiteral, .lexeme = "0" },
                },
                .description = "Zero integer",
            },
            .{
                .input = "0x1A2B",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .HexLiteral, .lexeme = "0x1A2B" },
                },
                .description = "Hexadecimal literal",
            },
            .{
                .input = "0xDEADBEEF",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .HexLiteral, .lexeme = "0xDEADBEEF" },
                },
                .description = "Hexadecimal literal with letters",
            },
            .{
                .input = "0b1010",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .BinaryLiteral, .lexeme = "0b1010" },
                },
                .description = "Binary literal",
            },
            .{
                .input = "0b11111111",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .BinaryLiteral, .lexeme = "0b11111111" },
                },
                .description = "Binary literal with all ones",
            },
        };

        /// String literal test cases
        pub const STRINGS = [_]ValidTokenFixture{
            .{
                .input = "\"hello\"",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .StringLiteral, .lexeme = "\"hello\"" },
                },
                .description = "Simple string literal",
            },
            .{
                .input = "\"\"",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .StringLiteral, .lexeme = "\"\"" },
                },
                .description = "Empty string literal",
            },
            .{
                .input = "\"with spaces\"",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .StringLiteral, .lexeme = "\"with spaces\"" },
                },
                .description = "String with spaces",
            },
            .{
                .input = "\"with\\nescapes\"",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .StringLiteral, .lexeme = "\"with\\nescapes\"" },
                },
                .description = "String with escape sequences",
            },
            .{
                .input = "\"unicode: 🚀\"",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .StringLiteral, .lexeme = "\"unicode: 🚀\"" },
                },
                .description = "String with Unicode characters",
            },
            .{
                .input = "'c'",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .CharacterLiteral, .lexeme = "'c'" },
                },
                .description = "Character literal",
            },
            .{
                .input = "'\\n'",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .CharacterLiteral, .lexeme = "'\\n'" },
                },
                .description = "Character literal with escape",
            },
        };

        /// Operator test cases
        pub const OPERATORS = [_]ValidTokenFixture{
            .{
                .input = "+",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Plus, .lexeme = "+" },
                },
                .description = "Plus operator",
            },
            .{
                .input = "-",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Minus, .lexeme = "-" },
                },
                .description = "Minus operator",
            },
            .{
                .input = "==",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .EqualEqual, .lexeme = "==" },
                },
                .description = "Equality operator",
            },
            .{
                .input = "!=",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .BangEqual, .lexeme = "!=" },
                },
                .description = "Not equal operator",
            },
            .{
                .input = "&&",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .AmpersandAmpersand, .lexeme = "&&" },
                },
                .description = "Logical AND operator",
            },
            .{
                .input = "||",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .PipePipe, .lexeme = "||" },
                },
                .description = "Logical OR operator",
            },
        };

        /// Delimiter test cases
        pub const DELIMITERS = [_]ValidTokenFixture{
            .{
                .input = "(",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .LeftParen, .lexeme = "(" },
                },
                .description = "Left parenthesis",
            },
            .{
                .input = ")",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .RightParen, .lexeme = ")" },
                },
                .description = "Right parenthesis",
            },
            .{
                .input = "{",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .LeftBrace, .lexeme = "{" },
                },
                .description = "Left brace",
            },
            .{
                .input = "}",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .RightBrace, .lexeme = "}" },
                },
                .description = "Right brace",
            },
            .{
                .input = ";",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Semicolon, .lexeme = ";" },
                },
                .description = "Semicolon",
            },
        };

        /// Complex multi-token sequences
        pub const COMPLEX_SEQUENCES = [_]ValidTokenFixture{
            .{
                .input = "fn test() {}",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Fn, .lexeme = "fn", .line = 1, .column = 1 },
                    .{ .token_type = .Identifier, .lexeme = "test", .line = 1, .column = 4 },
                    .{ .token_type = .LeftParen, .lexeme = "(", .line = 1, .column = 8 },
                    .{ .token_type = .RightParen, .lexeme = ")", .line = 1, .column = 9 },
                    .{ .token_type = .LeftBrace, .lexeme = "{", .line = 1, .column = 11 },
                    .{ .token_type = .RightBrace, .lexeme = "}", .line = 1, .column = 12 },
                },
                .description = "Simple function declaration",
            },
            .{
                .input = "let x = 42;",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .Let, .lexeme = "let", .line = 1, .column = 1 },
                    .{ .token_type = .Identifier, .lexeme = "x", .line = 1, .column = 5 },
                    .{ .token_type = .Equal, .lexeme = "=", .line = 1, .column = 7 },
                    .{ .token_type = .IntegerLiteral, .lexeme = "42", .line = 1, .column = 9 },
                    .{ .token_type = .Semicolon, .lexeme = ";", .line = 1, .column = 11 },
                },
                .description = "Variable declaration with assignment",
            },
        };
    };

    /// Error test cases for testing error handling and recovery
    pub const ERROR_CASES = struct {
        /// Invalid character test cases
        pub const INVALID_CHARACTERS = [_]ErrorTestFixture{
            .{
                .input = "identifier @ invalid",
                .expected_error = LexerError.UnexpectedCharacter,
                .error_position = .{ .line = 1, .column = 12 },
                .description = "Unexpected @ character",
                .expected_message_contains = "unexpected character",
                .expected_suggestion = "Remove the character or use it within a string literal",
            },
            .{
                .input = "test $ symbol",
                .expected_error = LexerError.UnexpectedCharacter,
                .error_position = .{ .line = 1, .column = 6 },
                .description = "Unexpected $ character",
                .expected_message_contains = "unexpected character",
            },
            .{
                .input = "code # hash",
                .expected_error = LexerError.UnexpectedCharacter,
                .error_position = .{ .line = 1, .column = 6 },
                .description = "Unexpected # character",
                .expected_message_contains = "unexpected character",
            },
        };

        /// Unterminated string test cases
        pub const UNTERMINATED_STRINGS = [_]ErrorTestFixture{
            .{
                .input = "\"unterminated string",
                .expected_error = LexerError.UnterminatedString,
                .error_position = .{ .line = 1, .column = 1 },
                .description = "Basic unterminated string",
                .expected_message_contains = "unterminated string literal",
                .expected_suggestion = "Add a closing quote (\") to complete the string",
            },
            .{
                .input = "\"string with \\n escape",
                .expected_error = LexerError.UnterminatedString,
                .error_position = .{ .line = 1, .column = 1 },
                .description = "Unterminated string with escape sequence",
                .expected_message_contains = "unterminated string literal",
            },
            .{
                .input = "'unterminated char",
                .expected_error = LexerError.UnterminatedString,
                .error_position = .{ .line = 1, .column = 1 },
                .description = "Unterminated character literal",
                .expected_message_contains = "unterminated",
            },
        };

        /// Invalid number literal test cases
        pub const INVALID_NUMBERS = [_]ErrorTestFixture{
            .{
                .input = "0xGHIJ",
                .expected_error = LexerError.InvalidHexLiteral,
                .error_position = .{ .line = 1, .column = 1 },
                .description = "Invalid hex literal with non-hex characters",
                .expected_message_contains = "invalid hexadecimal literal",
                .expected_suggestion = "Use '0x' prefix followed by hex digits (0-9, a-f, A-F)",
            },
            .{
                .input = "0b1012",
                .expected_error = LexerError.InvalidBinaryLiteral,
                .error_position = .{ .line = 1, .column = 1 },
                .description = "Invalid binary literal with non-binary digits",
                .expected_message_contains = "invalid binary literal",
                .expected_suggestion = "Use '0b' prefix followed by binary digits (0 and 1)",
            },
            .{
                .input = "0x",
                .expected_error = LexerError.InvalidHexLiteral,
                .error_position = .{ .line = 1, .column = 1 },
                .description = "Incomplete hex literal",
                .expected_message_contains = "invalid hexadecimal literal",
            },
            .{
                .input = "0b",
                .expected_error = LexerError.InvalidBinaryLiteral,
                .error_position = .{ .line = 1, .column = 1 },
                .description = "Incomplete binary literal",
                .expected_message_contains = "invalid binary literal",
            },
        };

        /// Invalid escape sequence test cases
        pub const INVALID_ESCAPES = [_]ErrorTestFixture{
            .{
                .input = "\"invalid \\q escape\"",
                .expected_error = LexerError.InvalidEscapeSequence,
                .error_position = .{ .line = 1, .column = 9 },
                .description = "Invalid escape sequence \\q",
                .expected_message_contains = "invalid escape sequence",
                .expected_suggestion = "Use valid escape sequences like \\n, \\t, \\r, \\\\, \\\", or \\xNN",
            },
            .{
                .input = "\"incomplete \\x1 escape\"",
                .expected_error = LexerError.InvalidEscapeSequence,
                .error_position = .{ .line = 1, .column = 12 },
                .description = "Incomplete hex escape sequence",
                .expected_message_contains = "invalid escape sequence",
            },
        };

        /// Multiple error test cases
        pub const MULTIPLE_ERRORS = [_]ErrorTestFixture{
            .{
                .input = "@ invalid $ chars # here",
                .expected_error = LexerError.UnexpectedCharacter,
                .error_position = .{ .line = 1, .column = 1 },
                .description = "Multiple invalid characters in sequence",
                .expected_message_contains = "unexpected character",
            },
            .{
                .input = "0xGHI 0b123 \"unterminated",
                .expected_error = LexerError.InvalidHexLiteral,
                .error_position = .{ .line = 1, .column = 1 },
                .description = "Multiple different error types",
                .expected_message_contains = "invalid hexadecimal literal",
            },
        };
    };

    /// Edge case test fixtures
    pub const EDGE_CASES = struct {
        /// Empty and whitespace-only inputs
        pub const EMPTY_INPUTS = [_]ValidTokenFixture{
            .{
                .input = "",
                .expected_tokens = &[_]ExpectedToken{},
                .description = "Empty input",
            },
            .{
                .input = "   ",
                .expected_tokens = &[_]ExpectedToken{},
                .description = "Whitespace only",
            },
            .{
                .input = "\n\n\n",
                .expected_tokens = &[_]ExpectedToken{},
                .description = "Newlines only",
            },
            .{
                .input = "\t\t\t",
                .expected_tokens = &[_]ExpectedToken{},
                .description = "Tabs only",
            },
        };

        /// Unicode test cases
        pub const UNICODE_CASES = [_]ValidTokenFixture{
            .{
                .input = "\"🚀\"",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .StringLiteral, .lexeme = "\"🚀\"" },
                },
                .description = "Unicode emoji in string",
            },
            .{
                .input = "\"café\"",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .StringLiteral, .lexeme = "\"café\"" },
                },
                .description = "Unicode accented characters",
            },
            .{
                .input = "\"中文\"",
                .expected_tokens = &[_]ExpectedToken{
                    .{ .token_type = .StringLiteral, .lexeme = "\"中文\"" },
                },
                .description = "Unicode Chinese characters",
            },
        };

        /// Large input test cases (for performance testing)
        pub const LARGE_INPUTS = struct {
            /// Generate a large identifier for testing
            pub fn generateLargeIdentifier(allocator: Allocator, size: usize) ![]u8 {
                var buffer = try allocator.alloc(u8, size);
                buffer[0] = 'a'; // Start with a letter
                for (buffer[1..]) |*char| {
                    char.* = 'a';
                }
                return buffer;
            }

            /// Generate a large string literal for testing
            pub fn generateLargeString(allocator: Allocator, size: usize) ![]u8 {
                var buffer = try allocator.alloc(u8, size + 2); // +2 for quotes
                buffer[0] = '"';
                buffer[buffer.len - 1] = '"';
                for (buffer[1 .. buffer.len - 1]) |*char| {
                    char.* = 'x';
                }
                return buffer;
            }

            /// Generate repeated tokens for testing
            pub fn generateRepeatedTokens(allocator: Allocator, token: []const u8, count: usize) ![]u8 {
                const total_size = (token.len + 1) * count; // +1 for space
                var buffer = try allocator.alloc(u8, total_size);
                var pos: usize = 0;

                for (0..count) |i| {
                    @memcpy(buffer[pos .. pos + token.len], token);
                    pos += token.len;
                    if (i < count - 1) {
                        buffer[pos] = ' ';
                        pos += 1;
                    }
                }

                return buffer[0..pos];
            }
        };
    };

    /// Performance test fixtures
    pub const PERFORMANCE_CASES = struct {
        /// Small program for baseline performance
        pub const SMALL_PROGRAM =
            \\fn add(a: u32, b: u32) -> u32 {
            \\    return a + b;
            \\}
        ;

        /// Medium program for scaling tests
        pub const MEDIUM_PROGRAM =
            \\contract Calculator {
            \\    fn add(a: u32, b: u32) -> u32 {
            \\        return a + b;
            \\    }
            \\    
            \\    fn subtract(a: u32, b: u32) -> u32 {
            \\        return a - b;
            \\    }
            \\    
            \\    fn multiply(a: u32, b: u32) -> u32 {
            \\        return a * b;
            \\    }
            \\    
            \\    fn divide(a: u32, b: u32) -> u32 {
            \\        requires b != 0;
            \\        return a / b;
            \\    }
            \\}
        ;

        /// Generate a large program for stress testing
        pub fn generateLargeProgram(allocator: Allocator, function_count: usize) ![]u8 {
            var buffer = std.ArrayList(u8).init(allocator);
            defer buffer.deinit();

            try buffer.appendSlice("contract LargeContract {\n");

            for (0..function_count) |i| {
                try buffer.writer().print(
                    \\    fn function{}(param: u32) -> u32 {{
                    \\        let result = param + {};
                    \\        return result;
                    \\    }}
                    \\
                    \\
                , .{ i, i });
            }

            try buffer.appendSlice("}\n");
            return buffer.toOwnedSlice();
        }
    };
};

/// Utility functions for working with test fixtures
pub const FixtureUtils = struct {
    /// Load a fixture from the filesystem
    pub fn loadFixture(allocator: Allocator, path: []const u8) ![]u8 {
        const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.err("Fixture file not found: {s}", .{path});
                return err;
            },
            else => return err,
        };
        defer file.close();

        const file_size = try file.getEndPos();
        const contents = try allocator.alloc(u8, file_size);
        _ = try file.readAll(contents);

        return contents;
    }

    /// Validate that a fixture matches expected structure
    pub fn validateFixture(fixture: ValidTokenFixture) bool {
        return fixture.input.len > 0 and
            fixture.expected_tokens.len > 0 and
            fixture.description.len > 0;
    }

    /// Get all fixtures of a specific category
    pub fn getAllIdentifierFixtures() []const ValidTokenFixture {
        return &LexerTestFixtures.VALID_TOKENS.IDENTIFIERS;
    }

    pub fn getAllKeywordFixtures() []const ValidTokenFixture {
        return &LexerTestFixtures.VALID_TOKENS.KEYWORDS;
    }

    pub fn getAllNumberFixtures() []const ValidTokenFixture {
        return &LexerTestFixtures.VALID_TOKENS.NUMBERS;
    }

    pub fn getAllStringFixtures() []const ValidTokenFixture {
        return &LexerTestFixtures.VALID_TOKENS.STRINGS;
    }

    pub fn getAllOperatorFixtures() []const ValidTokenFixture {
        return &LexerTestFixtures.VALID_TOKENS.OPERATORS;
    }

    pub fn getAllDelimiterFixtures() []const ValidTokenFixture {
        return &LexerTestFixtures.VALID_TOKENS.DELIMITERS;
    }

    pub fn getAllErrorFixtures() []const ErrorTestFixture {
        return &LexerTestFixtures.ERROR_CASES.INVALID_CHARACTERS;
    }
};

// Tests for the fixtures themselves
test "ValidTokenFixture validation" {
    const fixture = LexerTestFixtures.VALID_TOKENS.IDENTIFIERS[0];
    try std.testing.expect(FixtureUtils.validateFixture(fixture));
}

test "Fixture loading utilities" {
    const identifiers = FixtureUtils.getAllIdentifierFixtures();
    try std.testing.expect(identifiers.len > 0);

    const keywords = FixtureUtils.getAllKeywordFixtures();
    try std.testing.expect(keywords.len > 0);
}
