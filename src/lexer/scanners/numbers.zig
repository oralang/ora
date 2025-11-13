// ============================================================================
// Number Literal Scanners
// ============================================================================
//
// Handles scanning of numeric literals: hex, binary, decimal, and addresses.
//
// ============================================================================

const std = @import("std");
const LexerError = @import("../lexer.zig").LexerError;
const SourceRange = @import("../lexer.zig").SourceRange;
const TokenValue = @import("../lexer.zig").TokenValue;
const Token = @import("../lexer.zig").Token;
const isHexDigit = @import("../lexer.zig").isHexDigit;
const isDigit = @import("../lexer.zig").isDigit;
const isAlpha = @import("../lexer.zig").isAlpha;
const isBinaryDigit = @import("../lexer.zig").isBinaryDigit;

// Forward declaration - Lexer is defined in lexer.zig
const Lexer = @import("../lexer.zig").Lexer;
const ErrorRecovery = @import("../error_recovery.zig").ErrorRecovery;

/// Scan a hex literal (0x... or 0X...)
pub fn scanHexLiteral(lexer: *Lexer) LexerError!void {
    var digit_count: u32 = 0;

    // Scan hex digits and underscores
    while (!lexer.isAtEnd()) {
        const c = lexer.peek();
        if (isHexDigit(c)) {
            digit_count += 1;
            _ = lexer.advance();
        } else if (c == '_') {
            _ = lexer.advance(); // Skip underscore separator
        } else {
            break; // Stop at non-hex character
        }
    }

    if (digit_count == 0) {
        // Invalid hex literal (just "0x")
        return LexerError.InvalidHexLiteral;
    }

    // Check if the next character would make this invalid
    // (e.g., "0xG" should be invalid, not "0x" + "G")
    if (!lexer.isAtEnd()) {
        const next_char = lexer.peek();
        if (isAlpha(next_char) and !isHexDigit(next_char)) {
            // Invalid hex literal with non-hex letters
            return LexerError.InvalidHexLiteral;
        }
    }

    // Check if it's an address (40 hex digits)
    if (digit_count == 40) {
        try addAddressToken(lexer);
    } else {
        try addHexToken(lexer);
    }
}

/// Scan a binary literal (0b... or 0B...)
pub fn scanBinaryLiteral(lexer: *Lexer) LexerError!void {
    var digit_count: u32 = 0;
    var underscore_count: u32 = 0;
    var last_was_underscore = false;
    var first_char = true;
    var invalid_char_pos: ?u32 = null;
    var invalid_char: ?u8 = null;

    // Scan binary digits and underscores with enhanced validation
    while (!lexer.isAtEnd()) {
        const c = lexer.peek();

        if (isBinaryDigit(c)) {
            digit_count += 1;
            last_was_underscore = false;
            first_char = false;
            _ = lexer.advance();
        } else if (c == '_') {
            // Enhanced underscore validation
            if (first_char) {
                // Binary literal cannot start with underscore after 0b
                invalid_char_pos = lexer.current;
                invalid_char = c;
                break;
            }
            if (last_was_underscore) {
                // Consecutive underscores not allowed
                invalid_char_pos = lexer.current;
                invalid_char = c;
                break;
            }
            underscore_count += 1;
            last_was_underscore = true;
            first_char = false;
            _ = lexer.advance();
        } else {
            // Check if this is an invalid character that should be part of the literal
            if (isDigit(c) or isAlpha(c)) {
                invalid_char_pos = lexer.current;
                invalid_char = c;
            }
            break; // Stop at non-binary character
        }
    }

    // Enhanced error handling with specific error messages
    if (digit_count == 0) {
        // Invalid binary literal (just "0b" or "0b_") - use error recovery if enabled
        if (lexer.hasErrorRecovery()) {
            const range = SourceRange{
                .start_line = lexer.line,
                .start_column = lexer.start_column,
                .end_line = lexer.line,
                .end_column = lexer.column,
                .start_offset = lexer.start,
                .end_offset = lexer.current,
            };

            const message = if (underscore_count > 0)
                "Invalid binary literal: contains only underscores after '0b'"
            else
                "Invalid binary literal: missing digits after '0b'";

            const suggestion = "Add binary digits (0 or 1) after '0b'";

            try lexer.error_recovery.?.recordDetailedErrorWithSuggestion(LexerError.InvalidBinaryLiteral, range, lexer.source, message, suggestion);

            // Skip to next token boundary for recovery
            lexer.current = ErrorRecovery.findNextTokenBoundary(lexer.source, lexer.current);
            return; // Skip adding the token
        } else {
            return LexerError.InvalidBinaryLiteral;
        }
    }

    // Check for trailing underscore
    if (last_was_underscore) {
        if (lexer.hasErrorRecovery()) {
            const range = SourceRange{
                .start_line = lexer.line,
                .start_column = lexer.start_column,
                .end_line = lexer.line,
                .end_column = lexer.column,
                .start_offset = lexer.start,
                .end_offset = lexer.current,
            };

            const message = "Invalid binary literal: cannot end with underscore";
            const suggestion = "Remove the trailing underscore or add binary digits after it";

            try lexer.error_recovery.?.recordDetailedErrorWithSuggestion(LexerError.InvalidBinaryLiteral, range, lexer.source, message, suggestion);

            // Skip to next token boundary for recovery
            lexer.current = ErrorRecovery.findNextTokenBoundary(lexer.source, lexer.current);
            return; // Skip adding the token
        } else {
            return LexerError.InvalidBinaryLiteral;
        }
    }

    // Check for invalid characters found during scanning
    if (invalid_char_pos != null and invalid_char != null) {
        if (lexer.hasErrorRecovery()) {
            const range = SourceRange{
                .start_line = lexer.line,
                .start_column = lexer.start_column,
                .end_line = lexer.line,
                .end_column = lexer.column,
                .start_offset = lexer.start,
                .end_offset = lexer.current,
            };

            var message_buf: [256]u8 = undefined;
            const message = if (invalid_char.? == '_' and last_was_underscore)
                "Invalid binary literal: consecutive underscores not allowed"
            else if (invalid_char.? == '_')
                "Invalid binary literal: cannot start with underscore"
            else if (isDigit(invalid_char.?))
                std.fmt.bufPrint(&message_buf, "Invalid binary literal: contains invalid digit '{c}' (only 0 and 1 allowed)", .{invalid_char.?}) catch "Invalid binary literal: contains invalid digit"
            else
                std.fmt.bufPrint(&message_buf, "Invalid binary literal: contains invalid character '{c}'", .{invalid_char.?}) catch "Invalid binary literal: contains invalid character";

            const suggestion = if (invalid_char.? == '2' or invalid_char.? == '3' or invalid_char.? == '4' or
                invalid_char.? == '5' or invalid_char.? == '6' or invalid_char.? == '7' or
                invalid_char.? == '8' or invalid_char.? == '9')
                "Binary literals can only contain digits 0 and 1. Use decimal or hex literals for other digits."
            else if (isAlpha(invalid_char.?))
                "Binary literals can only contain digits 0 and 1. Remove the letter or use a hex literal."
            else
                "Use only binary digits (0 and 1) and underscores as separators in binary literals.";

            try lexer.error_recovery.?.recordDetailedErrorWithSuggestion(LexerError.InvalidBinaryLiteral, range, lexer.source, message, suggestion);

            // Skip to next token boundary for recovery
            lexer.current = ErrorRecovery.findNextTokenBoundary(lexer.source, lexer.current);
            return; // Skip adding the token
        } else {
            return LexerError.InvalidBinaryLiteral;
        }
    }

    // Check if the next character would make this invalid (lookahead validation)
    // (e.g., "0b12" should be invalid, not "0b1" + "2")
    if (!lexer.isAtEnd()) {
        const next_char = lexer.peek();
        if (isDigit(next_char) or isAlpha(next_char)) {
            // Invalid binary literal with non-binary digits - use error recovery if enabled
            if (lexer.hasErrorRecovery()) {
                const range = SourceRange{
                    .start_line = lexer.line,
                    .start_column = lexer.start_column,
                    .end_line = lexer.line,
                    .end_column = lexer.column + 1, // Include the invalid character
                    .start_offset = lexer.start,
                    .end_offset = lexer.current + 1,
                };

                var message_buf: [256]u8 = undefined;
                const message = std.fmt.bufPrint(&message_buf, "Invalid binary literal: unexpected character '{c}' after binary digits", .{next_char}) catch "Invalid binary literal: unexpected character after binary digits";

                const suggestion = if (isDigit(next_char))
                    "Binary literals can only contain digits 0 and 1. Separate with whitespace or use a different literal type."
                else
                    "Add whitespace or operator between the binary literal and the following character.";

                try lexer.error_recovery.?.recordDetailedErrorWithSuggestion(LexerError.InvalidBinaryLiteral, range, lexer.source, message, suggestion);

                // Skip to next token boundary for recovery
                lexer.current = ErrorRecovery.findNextTokenBoundary(lexer.source, lexer.current);
                return; // Skip adding the token
            } else {
                return LexerError.InvalidBinaryLiteral;
            }
        }
    }

    try addBinaryToken(lexer);
}

/// Scan a decimal number literal
pub fn scanNumber(lexer: *Lexer) LexerError!void {
    // Scan integer part
    while (isDigit(lexer.peek()) or lexer.peek() == '_') {
        _ = lexer.advance();
    }

    // Check for scientific notation
    if (lexer.peek() == 'e' or lexer.peek() == 'E') {
        _ = lexer.advance();
        if (lexer.peek() == '+' or lexer.peek() == '-') {
            _ = lexer.advance();
        }
        while (isDigit(lexer.peek())) {
            _ = lexer.advance();
        }
    }

    try addIntegerToken(lexer);
}

/// Add a binary literal token
pub fn addBinaryToken(lexer: *Lexer) LexerError!void {
    // Strip 0b/0B prefix from binary literal
    const text = lexer.source[lexer.start + 2 .. lexer.current];
    const range = SourceRange{
        .start_line = lexer.line,
        .start_column = lexer.start_column,
        .end_line = lexer.line,
        .end_column = lexer.column,
        .start_offset = lexer.start,
        .end_offset = lexer.current,
    };

    // Convert binary string to integer value with overflow checking
    const binary_value = parseBinaryToInteger(text) catch |err| {
        return err;
    };

    const token_value = TokenValue{ .binary = binary_value };

    try lexer.tokens.append(lexer.allocator, Token{
        .type = .BinaryLiteral,
        .lexeme = lexer.source[lexer.start..lexer.current], // Full lexeme including 0b prefix
        .range = range,
        .value = token_value,
        // Legacy fields for backward compatibility
        .line = lexer.line,
        .column = lexer.start_column,
    });
}

/// Parse binary string to integer
pub fn parseBinaryToInteger(binary_str: []const u8) LexerError!u256 {
    if (binary_str.len == 0) {
        return LexerError.InvalidBinaryLiteral;
    }

    var result: u256 = 0;
    var bit_count: u32 = 0;
    var underscore_count: u32 = 0;
    var last_was_underscore = false;

    // Enhanced validation during parsing
    for (binary_str, 0..) |c, i| {
        if (c == '_') {
            // Enhanced underscore validation
            underscore_count += 1;

            // Check for leading underscore
            if (i == 0) {
                return LexerError.InvalidBinaryLiteral;
            }

            // Check for consecutive underscores
            if (last_was_underscore) {
                return LexerError.InvalidBinaryLiteral;
            }

            // Check for trailing underscore (will be caught at end)
            last_was_underscore = true;
            continue; // Skip underscores used as separators
        }

        // Reset underscore flag
        last_was_underscore = false;

        // Validate binary digit
        if (c != '0' and c != '1') {
            return LexerError.InvalidBinaryLiteral;
        }

        // Check for overflow before processing (u256 has 256 bits)
        if (bit_count >= 256) {
            return LexerError.NumberTooLarge;
        }

        // Shift left and add new bit with overflow checking
        const overflow = @mulWithOverflow(result, 2);
        if (overflow[1] != 0) {
            return LexerError.NumberTooLarge;
        }
        result = overflow[0];

        if (c == '1') {
            const add_overflow = @addWithOverflow(result, 1);
            if (add_overflow[1] != 0) {
                return LexerError.NumberTooLarge;
            }
            result = add_overflow[0];
        }

        bit_count += 1;
    }

    // Check for trailing underscore
    if (last_was_underscore) {
        return LexerError.InvalidBinaryLiteral;
    }

    // Final validation
    if (bit_count == 0) {
        return LexerError.InvalidBinaryLiteral;
    }

    // Additional validation: ensure we have meaningful content
    // (not just underscores)
    if (bit_count == 0 and underscore_count > 0) {
        return LexerError.InvalidBinaryLiteral;
    }

    return result;
}

/// Add an integer token
pub fn addIntegerToken(lexer: *Lexer) LexerError!void {
    const text = lexer.source[lexer.start..lexer.current];
    const range = SourceRange{
        .start_line = lexer.line,
        .start_column = lexer.start_column,
        .end_line = lexer.line,
        .end_column = lexer.column,
        .start_offset = lexer.start,
        .end_offset = lexer.current,
    };

    // Convert decimal string to integer value with overflow checking
    const integer_value = parseDecimalToInteger(text) catch |err| {
        return err;
    };

    const token_value = TokenValue{ .integer = integer_value };

    try lexer.tokens.append(lexer.allocator, Token{
        .type = .IntegerLiteral,
        .lexeme = text,
        .range = range,
        .value = token_value,
        // Legacy fields for backward compatibility
        .line = lexer.line,
        .column = lexer.start_column,
    });
}

/// Parse decimal string to integer
pub fn parseDecimalToInteger(decimal_str: []const u8) LexerError!u256 {
    if (decimal_str.len == 0) {
        return LexerError.NumberTooLarge; // Should not happen, but handle gracefully
    }

    var result: u256 = 0;
    var digit_count: u32 = 0;

    for (decimal_str) |c| {
        if (c == '_') {
            // Skip underscores used as separators
            continue;
        }

        if (!isDigit(c)) {
            // This handles scientific notation and other non-digit characters
            // For now, we'll just skip them (scientific notation parsing would be more complex)
            continue;
        }

        // Check for overflow (u256 max is about 78 decimal digits)
        if (digit_count >= 78) {
            return LexerError.NumberTooLarge;
        }

        // Shift left by multiplying by 10 and add new digit
        const overflow = @mulWithOverflow(result, 10);
        if (overflow[1] != 0) {
            return LexerError.NumberTooLarge;
        }
        result = overflow[0];

        const digit_value = c - '0';
        const add_overflow = @addWithOverflow(result, digit_value);
        if (add_overflow[1] != 0) {
            return LexerError.NumberTooLarge;
        }
        result = add_overflow[0];

        digit_count += 1;
    }

    return result;
}

/// Add a hex literal token
pub fn addHexToken(lexer: *Lexer) LexerError!void {
    // Strip 0x/0X prefix from hex literal
    const text = lexer.source[lexer.start + 2 .. lexer.current];
    const range = SourceRange{
        .start_line = lexer.line,
        .start_column = lexer.start_column,
        .end_line = lexer.line,
        .end_column = lexer.column,
        .start_offset = lexer.start,
        .end_offset = lexer.current,
    };

    // Convert hex string to integer value with overflow checking
    const hex_value = parseHexToInteger(text) catch |err| {
        return err;
    };

    const token_value = TokenValue{ .hex = hex_value };

    try lexer.tokens.append(lexer.allocator, Token{
        .type = .HexLiteral,
        .lexeme = lexer.source[lexer.start..lexer.current], // Full lexeme including 0x prefix
        .range = range,
        .value = token_value,
        // Legacy fields for backward compatibility
        .line = lexer.line,
        .column = lexer.start_column,
    });
}

/// Add an address literal token
pub fn addAddressToken(lexer: *Lexer) LexerError!void {
    // Strip 0x/0X prefix from address literal
    const text = lexer.source[lexer.start + 2 .. lexer.current];
    const range = SourceRange{
        .start_line = lexer.line,
        .start_column = lexer.start_column,
        .end_line = lexer.line,
        .end_column = lexer.column,
        .start_offset = lexer.start,
        .end_offset = lexer.current,
    };

    // Convert address string to byte array
    const address_bytes = parseAddressToBytes(text) catch |err| {
        return err;
    };

    const token_value = TokenValue{ .address = address_bytes };

    try lexer.tokens.append(lexer.allocator, Token{
        .type = .AddressLiteral,
        .lexeme = lexer.source[lexer.start..lexer.current], // Full lexeme including 0x prefix
        .range = range,
        .value = token_value,
        // Legacy fields for backward compatibility
        .line = lexer.line,
        .column = lexer.start_column,
    });
}

/// Parse hex string to integer
pub fn parseHexToInteger(hex_str: []const u8) LexerError!u256 {
    if (hex_str.len == 0) {
        return LexerError.InvalidHexLiteral;
    }

    var result: u256 = 0;
    var digit_count: u32 = 0;

    for (hex_str) |c| {
        if (c == '_') {
            // Skip underscores used as separators
            continue;
        }

        if (!isHexDigit(c)) {
            return LexerError.InvalidHexLiteral;
        }

        // Check for overflow (u256 has 256 bits, so max 64 hex digits)
        if (digit_count >= 64) {
            return LexerError.NumberTooLarge;
        }

        // Shift left by 4 bits and add new hex digit
        const overflow = @mulWithOverflow(result, 16);
        if (overflow[1] != 0) {
            return LexerError.NumberTooLarge;
        }
        result = overflow[0];

        // Convert hex digit to value
        const digit_value: u8 = switch (c) {
            '0'...'9' => c - '0',
            'a'...'f' => c - 'a' + 10,
            'A'...'F' => c - 'A' + 10,
            else => return LexerError.InvalidHexLiteral,
        };

        const add_overflow = @addWithOverflow(result, digit_value);
        if (add_overflow[1] != 0) {
            return LexerError.NumberTooLarge;
        }
        result = add_overflow[0];

        digit_count += 1;
    }

    if (digit_count == 0) {
        return LexerError.InvalidHexLiteral;
    }

    return result;
}

/// Parse address string to bytes
pub fn parseAddressToBytes(address_str: []const u8) LexerError!([20]u8) {
    if (address_str.len != 40) {
        return LexerError.InvalidAddressFormat;
    }

    var result: [20]u8 = undefined;
    var i: usize = 0;

    while (i < 40) : (i += 2) {
        if (!isHexDigit(address_str[i]) or !isHexDigit(address_str[i + 1])) {
            return LexerError.InvalidAddressFormat;
        }

        const high_nibble: u8 = switch (address_str[i]) {
            '0'...'9' => address_str[i] - '0',
            'a'...'f' => address_str[i] - 'a' + 10,
            'A'...'F' => address_str[i] - 'A' + 10,
            else => return LexerError.InvalidAddressFormat,
        };

        const low_nibble: u8 = switch (address_str[i + 1]) {
            '0'...'9' => address_str[i + 1] - '0',
            'a'...'f' => address_str[i + 1] - 'a' + 10,
            'A'...'F' => address_str[i + 1] - 'A' + 10,
            else => return LexerError.InvalidAddressFormat,
        };

        result[i / 2] = (high_nibble << 4) | low_nibble;
    }

    return result;
}
