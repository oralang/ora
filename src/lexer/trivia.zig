// ============================================================================
// Trivia Handling
// ============================================================================
//
// Handles whitespace, comments, and string processing for the lexer.
// Includes trivia tracking, string interning, and escape sequence processing.
//
// ============================================================================

const std = @import("std");
const Allocator = std.mem.Allocator;
const LexerError = @import("../lexer.zig").LexerError;
const SourceRange = @import("../lexer.zig").SourceRange;
const isWhitespace = @import("../lexer.zig").isWhitespace;

// Forward declaration - Lexer is defined in lexer.zig
const Lexer = @import("../lexer.zig").Lexer;

/// Types of trivia (whitespace and comments)
pub const TriviaKind = enum { Whitespace, Newline, LineComment, BlockComment, DocLineComment, DocBlockComment };

/// A piece of trivia (whitespace or comment) with its source location
pub const TriviaPiece = struct {
    kind: TriviaKind,
    span: SourceRange,
};

/// String interning pool for deduplicating repeated strings
pub const StringPool = struct {
    strings: std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    allocator: Allocator,

    pub fn init(allocator: Allocator) StringPool {
        return StringPool{
            .strings = std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StringPool) void {
        // free all interned strings
        var iterator = self.strings.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.strings.deinit();
    }

    /// Intern a string, returning a reference to the stored copy
    pub fn intern(self: *StringPool, string: []const u8) ![]const u8 {
        const hash_value = StringPool.hash(string);

        // check if string is already interned
        if (self.strings.get(hash_value)) |interned_string| {
            // verify it's actually the same string (handle hash collisions)
            if (std.mem.eql(u8, interned_string, string)) {
                return interned_string;
            }
        }

        // string not found, create a copy and store it
        const owned_string = try self.allocator.dupe(u8, string);
        try self.strings.put(hash_value, owned_string);
        return owned_string;
    }

    /// Hash function for strings
    pub fn hash(string: []const u8) u64 {
        return std.hash_map.hashString(string);
    }

    /// Get the number of interned strings
    pub fn count(self: *StringPool) u32 {
        return @intCast(self.strings.count());
    }

    /// Clear all interned strings
    pub fn clear(self: *StringPool) void {
        // free all interned strings
        var iterator = self.strings.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.strings.clearAndFree();
    }
};

/// String processing engine for handling escape sequences and string validation
/// Simplified for smart contract language usage:
/// - Only ASCII characters are supported
/// - Limited escape sequences (\n, \t, \", \\)
/// - Max string length is restricted to 1KB
/// This simplification improves security, reduces gas costs, and simplifies the implementation
pub const StringProcessor = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) StringProcessor {
        return StringProcessor{
            .allocator = allocator,
        };
    }

    /// Process a string literal with escape sequences
    /// Follows the simplified string model for smart contracts:
    /// - Only ASCII characters are allowed
    /// - Only supports limited escape sequences: \n, \t, \", \\
    /// - String length is limited (enforced elsewhere)
    pub fn processString(self: *StringProcessor, raw_string: []const u8) LexerError![]u8 {
        var result = std.ArrayList(u8){};
        defer result.deinit(self.allocator);

        var i: usize = 0;
        while (i < raw_string.len) {
            // check for non-ASCII characters
            if (raw_string[i] > 127) {
                return LexerError.InvalidCharacterInString;
            }

            if (raw_string[i] == '\\') {
                if (i + 1 >= raw_string.len) {
                    // trailing backslash at end of string is invalid
                    return LexerError.InvalidEscapeSequence;
                }
                const escaped_char = try StringProcessor.processEscapeSequence(raw_string[i + 1 ..]);
                try result.append(self.allocator, escaped_char.char);
                i += escaped_char.consumed + 1; // +1 for the backslash
            } else {
                try result.append(self.allocator, raw_string[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Process a single escape sequence starting after the backslash with simplified validation
    /// Only supports a minimal set of escape sequences appropriate for a smart contract language
    fn processEscapeSequence(sequence: []const u8) LexerError!struct { char: u8, consumed: usize } {
        if (sequence.len == 0) {
            return LexerError.InvalidEscapeSequence;
        }

        switch (sequence[0]) {
            // simplified set of escape sequences for smart contract language
            'n' => return .{ .char = '\n', .consumed = 1 }, // Newline
            't' => return .{ .char = '\t', .consumed = 1 }, // Tab
            '\\' => return .{ .char = '\\', .consumed = 1 }, // Backslash
            '"' => return .{ .char = '"', .consumed = 1 }, // Double quote

            // all other escape sequences are invalid in our simplified string model
            else => return LexerError.InvalidEscapeSequence,
        }
    }

    /// Validate a character literal and extract its value with enhanced error handling
    pub fn processCharacterLiteral(raw_char: []const u8) LexerError!u8 {
        // check for empty character literal
        if (raw_char.len == 0) {
            return LexerError.EmptyCharacterLiteral;
        }

        // handle simple single character literal
        if (raw_char.len == 1) {
            const char = raw_char[0];
            // validate that it's a printable character or common whitespace
            if ((char >= 32 and char <= 126) or char == '\t' or char == '\n' or char == '\r') {
                return char;
            }
            // allow null character explicitly
            if (char == 0) {
                return char;
            }
            // reject control characters and invalid bytes
            return LexerError.InvalidCharacterLiteral;
        }

        // handle escape sequences
        if (raw_char.len >= 2 and raw_char[0] == '\\') {
            const escaped = StringProcessor.processEscapeSequence(raw_char[1..]) catch |err| {
                // provide more specific error context for escape sequences
                return err;
            };

            // validate that the escape sequence consumed the entire remaining content
            if (escaped.consumed + 1 != raw_char.len) {
                return LexerError.InvalidCharacterLiteral;
            }

            return escaped.char;
        }

        // multiple characters without escape sequence
        return LexerError.InvalidCharacterLiteral;
    }

    /// Validate and process a raw string literal (future feature)
    pub fn processRawString(self: *StringProcessor, raw_string: []const u8) LexerError![]u8 {
        // for now, just return a copy of the raw string
        // this will be enhanced when raw string literals are implemented
        return self.allocator.dupe(u8, raw_string);
    }
};

/// Capture leading trivia (whitespace and comments) before the next token
pub fn captureLeadingTrivia(lexer: *Lexer) !void {
    // consume whitespace and comments, recording them as trivia pieces
    while (!lexer.isAtEnd()) {
        const c = lexer.peek();
        if (isWhitespace(c)) {
            const start_off = lexer.current;
            const start_col = lexer.column;
            if (c == '\n') {
                _ = lexer.advance();
                lexer.line += 1;
                lexer.column = 1;
                const span = SourceRange{ .start_line = lexer.line - 1, .start_column = start_col, .end_line = lexer.line, .end_column = 1, .start_offset = start_off, .end_offset = lexer.current };
                try lexer.trivia.append(lexer.allocator, TriviaPiece{ .kind = .Newline, .span = span });
            } else {
                while (isWhitespace(lexer.peek()) and lexer.peek() != '\n' and !lexer.isAtEnd()) {
                    _ = lexer.advance();
                }
                const span = SourceRange{ .start_line = lexer.line, .start_column = start_col, .end_line = lexer.line, .end_column = lexer.column, .start_offset = start_off, .end_offset = lexer.current };
                try lexer.trivia.append(lexer.allocator, TriviaPiece{ .kind = .Whitespace, .span = span });
            }
            continue;
        }
        if (c == '/' and lexer.current + 1 < lexer.source.len and lexer.source[lexer.current + 1] == '/') {
            const start_off = lexer.current;
            const start_col = lexer.column;
            _ = lexer.advance(); // '/'
            _ = lexer.advance(); // '/'
            const is_doc = lexer.peek() == '/';
            if (is_doc) _ = lexer.advance(); // consume third '/'
            while (lexer.peek() != '\n' and !lexer.isAtEnd()) _ = lexer.advance();
            const span = SourceRange{ .start_line = lexer.line, .start_column = start_col, .end_line = lexer.line, .end_column = lexer.column, .start_offset = start_off, .end_offset = lexer.current };
            // treat entire '//' to line end as line comment trivia
            try lexer.trivia.append(lexer.allocator, TriviaPiece{ .kind = if (is_doc) .DocLineComment else .LineComment, .span = span });
            continue;
        }
        if (c == '/' and lexer.current + 1 < lexer.source.len and lexer.source[lexer.current + 1] == '*') {
            // capture block comment trivia only if it is properly closed; otherwise
            // leave it to the main scanner to report UnterminatedComment.
            if (tryCaptureClosedBlockCommentTrivia(lexer)) {
                continue;
            } else {
                break;
            }
        }
        break;
    }
}

/// Try to capture a closed block comment as trivia
pub fn tryCaptureClosedBlockCommentTrivia(lexer: *Lexer) bool {
    const save_current = lexer.current;
    const save_line = lexer.line;
    const save_column = lexer.column;

    // consume '/*'
    _ = lexer.advance();
    _ = lexer.advance();
    const is_doc = lexer.peek() == '*';
    if (is_doc) _ = lexer.advance(); // '/**'
    var nesting: u32 = 1;
    while (nesting > 0 and !lexer.isAtEnd()) {
        if (lexer.peek() == '/' and lexer.peekNext() == '*') {
            _ = lexer.advance();
            _ = lexer.advance();
            nesting += 1;
        } else if (lexer.peek() == '*' and lexer.peekNext() == '/') {
            _ = lexer.advance();
            _ = lexer.advance();
            nesting -= 1;
        } else if (lexer.peek() == '\n') {
            _ = lexer.advance();
            lexer.line += 1;
            lexer.column = 1;
        } else {
            _ = lexer.advance();
        }
    }
    if (nesting == 0) {
        const span = SourceRange{ .start_line = save_line, .start_column = save_column, .end_line = lexer.line, .end_column = lexer.column, .start_offset = save_current, .end_offset = lexer.current };
        lexer.trivia.append(lexer.allocator, TriviaPiece{ .kind = if (is_doc) .DocBlockComment else .BlockComment, .span = span }) catch return false;
        return true;
    }
    // not closed; restore and let main scanner handle error
    lexer.current = save_current;
    lexer.line = save_line;
    lexer.column = save_column;
    return false;
}
