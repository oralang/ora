// Number literal scanners: hex, binary, decimal, addresses.

const std = @import("std");
const lexer_mod = @import("../../lexer.zig");
const LexerError = lexer_mod.LexerError;
const TokenValue = lexer_mod.TokenValue;
const Lexer = lexer_mod.Lexer;
const isHexDigit = lexer_mod.isHexDigit;
const isDigit = lexer_mod.isDigit;
const isAlpha = lexer_mod.isAlpha;
const isBinaryDigit = lexer_mod.isBinaryDigit;

/// Scan a hex literal (0x... or 0X...). Emits AddressLiteral when exactly 40 hex digits, else HexLiteral.
pub fn scanHexLiteral(lexer: *Lexer) LexerError!void {
    var digit_count: u32 = 0;

    while (!lexer.isAtEnd()) {
        const c = lexer.peek();
        if (isHexDigit(c)) {
            digit_count += 1;
            _ = lexer.advance();
        } else if (c == '_') {
            _ = lexer.advance();
        } else {
            break;
        }
    }

    if (digit_count == 0) return LexerError.InvalidHexLiteral;

    // reject "0xG" — non-hex letter directly after digits invalidates the literal
    if (!lexer.isAtEnd()) {
        const next_char = lexer.peek();
        if (isAlpha(next_char) and !isHexDigit(next_char)) {
            return LexerError.InvalidHexLiteral;
        }
    }

    if (digit_count == 40) {
        try addAddressToken(lexer);
    } else {
        try addHexToken(lexer);
    }
}

/// Scan a binary literal (0b... or 0B...).
pub fn scanBinaryLiteral(lexer: *Lexer) LexerError!void {
    var digit_count: u32 = 0;
    var underscore_count: u32 = 0;
    var last_was_underscore = false;
    var first_char = true;
    var invalid_char: ?u8 = null;

    while (!lexer.isAtEnd()) {
        const c = lexer.peek();

        if (isBinaryDigit(c)) {
            digit_count += 1;
            last_was_underscore = false;
            first_char = false;
            _ = lexer.advance();
        } else if (c == '_') {
            if (first_char or last_was_underscore) {
                invalid_char = c;
                break;
            }
            underscore_count += 1;
            last_was_underscore = true;
            first_char = false;
            _ = lexer.advance();
        } else {
            if (isDigit(c) or isAlpha(c)) invalid_char = c;
            break;
        }
    }

    if (digit_count == 0) {
        const message = if (underscore_count > 0)
            "Invalid binary literal: contains only underscores after '0b'"
        else
            "Invalid binary literal: missing digits after '0b'";
        try lexer.reportLexErrorAndAdvance(
            LexerError.InvalidBinaryLiteral,
            lexer.currentRange(),
            message,
            "Add binary digits (0 or 1) after '0b'",
        );
        return;
    }

    if (last_was_underscore) {
        try lexer.reportLexErrorAndAdvance(
            LexerError.InvalidBinaryLiteral,
            lexer.currentRange(),
            "Invalid binary literal: cannot end with underscore",
            "Remove the trailing underscore or add binary digits after it",
        );
        return;
    }

    if (invalid_char) |bad| {
        var message_buf: [256]u8 = undefined;
        const message = if (bad == '_' and last_was_underscore)
            "Invalid binary literal: consecutive underscores not allowed"
        else if (bad == '_')
            "Invalid binary literal: cannot start with underscore"
        else if (isDigit(bad))
            std.fmt.bufPrint(&message_buf, "Invalid binary literal: contains invalid digit '{c}' (only 0 and 1 allowed)", .{bad}) catch "Invalid binary literal: contains invalid digit"
        else
            std.fmt.bufPrint(&message_buf, "Invalid binary literal: contains invalid character '{c}'", .{bad}) catch "Invalid binary literal: contains invalid character";

        const suggestion = if (bad >= '2' and bad <= '9')
            "Binary literals can only contain digits 0 and 1. Use decimal or hex literals for other digits."
        else if (isAlpha(bad))
            "Binary literals can only contain digits 0 and 1. Remove the letter or use a hex literal."
        else
            "Use only binary digits (0 and 1) and underscores as separators in binary literals.";

        try lexer.reportLexErrorAndAdvance(LexerError.InvalidBinaryLiteral, lexer.currentRange(), message, suggestion);
        return;
    }

    // lookahead: reject "0b12" (digit/letter immediately after binary digits)
    if (!lexer.isAtEnd()) {
        const next_char = lexer.peek();
        if (isDigit(next_char) or isAlpha(next_char)) {
            var message_buf: [256]u8 = undefined;
            const message = std.fmt.bufPrint(&message_buf, "Invalid binary literal: unexpected character '{c}' after binary digits", .{next_char}) catch "Invalid binary literal: unexpected character after binary digits";
            const suggestion = if (isDigit(next_char))
                "Binary literals can only contain digits 0 and 1. Separate with whitespace or use a different literal type."
            else
                "Add whitespace or operator between the binary literal and the following character.";

            // span this error one char past current to highlight the bad char
            var range = lexer.currentRange();
            range.end_column += 1;
            range.end_offset += 1;
            try lexer.reportLexErrorAndAdvance(LexerError.InvalidBinaryLiteral, range, message, suggestion);
            return;
        }
    }

    try addBinaryToken(lexer);
}

/// Scan a decimal number literal (supports underscore separators and scientific notation).
pub fn scanNumber(lexer: *Lexer) LexerError!void {
    while (isDigit(lexer.peek()) or lexer.peek() == '_') {
        _ = lexer.advance();
    }

    if (lexer.peek() == 'e' or lexer.peek() == 'E') {
        _ = lexer.advance();
        if (lexer.peek() == '+' or lexer.peek() == '-') _ = lexer.advance();
        while (isDigit(lexer.peek())) _ = lexer.advance();
    }

    try addIntegerToken(lexer);
}

pub fn addBinaryToken(lexer: *Lexer) LexerError!void {
    const text = lexer.source[lexer.start + 2 .. lexer.current]; // skip 0b prefix
    if (!lexer.config.store_token_values) {
        _ = try parseBinaryToInteger(text);
        try lexer.appendTokenWithValue(.BinaryLiteral, null);
        return;
    }
    try lexer.appendTokenWithValue(.BinaryLiteral, .{ .binary = try parseBinaryToInteger(text) });
}

pub fn addIntegerToken(lexer: *Lexer) LexerError!void {
    const text = lexer.source[lexer.start..lexer.current];
    if (!lexer.config.store_token_values) {
        _ = try parseDecimalToInteger(text);
        try lexer.appendTokenWithValue(.IntegerLiteral, null);
        return;
    }
    try lexer.appendTokenWithValue(.IntegerLiteral, .{ .integer = try parseDecimalToInteger(text) });
}

pub fn addHexToken(lexer: *Lexer) LexerError!void {
    const text = lexer.source[lexer.start + 2 .. lexer.current]; // skip 0x prefix
    if (!lexer.config.store_token_values) {
        _ = try parseHexToInteger(text);
        try lexer.appendTokenWithValue(.HexLiteral, null);
        return;
    }
    try lexer.appendTokenWithValue(.HexLiteral, .{ .hex = try parseHexToInteger(text) });
}

pub fn addAddressToken(lexer: *Lexer) LexerError!void {
    const text = lexer.source[lexer.start + 2 .. lexer.current]; // skip 0x prefix
    if (!lexer.config.store_token_values) {
        _ = try parseAddressToBytes(text);
        try lexer.appendTokenWithValue(.AddressLiteral, null);
        return;
    }
    try lexer.appendTokenWithValue(.AddressLiteral, .{ .address = try parseAddressToBytes(text) });
}

/// Parse binary string to u256 with overflow + underscore validation.
pub fn parseBinaryToInteger(binary_str: []const u8) LexerError!u256 {
    if (binary_str.len == 0) return LexerError.InvalidBinaryLiteral;

    var result: u256 = 0;
    var bit_count: u32 = 0;
    var last_was_underscore = false;

    for (binary_str, 0..) |c, i| {
        if (c == '_') {
            if (i == 0 or last_was_underscore) return LexerError.InvalidBinaryLiteral;
            last_was_underscore = true;
            continue;
        }
        last_was_underscore = false;

        if (c != '0' and c != '1') return LexerError.InvalidBinaryLiteral;
        if (bit_count >= 256) return LexerError.NumberTooLarge;

        const shifted = @mulWithOverflow(result, 2);
        if (shifted[1] != 0) return LexerError.NumberTooLarge;
        result = shifted[0];

        if (c == '1') {
            const sum = @addWithOverflow(result, 1);
            if (sum[1] != 0) return LexerError.NumberTooLarge;
            result = sum[0];
        }
        bit_count += 1;
    }

    if (last_was_underscore) return LexerError.InvalidBinaryLiteral;
    if (bit_count == 0) return LexerError.InvalidBinaryLiteral;
    return result;
}

/// Parse decimal string to u256. Skips underscore separators and any non-digit chars (e.g. scientific notation `e`).
pub fn parseDecimalToInteger(decimal_str: []const u8) LexerError!u256 {
    if (decimal_str.len == 0) return LexerError.NumberTooLarge;

    var result: u256 = 0;
    var digit_count: u32 = 0;

    for (decimal_str) |c| {
        if (c == '_') continue;
        if (!isDigit(c)) continue; // scientific notation 'e'/'E'/'+'/'-' fall through here

        if (digit_count >= 78) return LexerError.NumberTooLarge;

        const shifted = @mulWithOverflow(result, 10);
        if (shifted[1] != 0) return LexerError.NumberTooLarge;
        result = shifted[0];

        const sum = @addWithOverflow(result, @as(u256, c - '0'));
        if (sum[1] != 0) return LexerError.NumberTooLarge;
        result = sum[0];

        digit_count += 1;
    }

    return result;
}

/// Parse hex string to u256.
pub fn parseHexToInteger(hex_str: []const u8) LexerError!u256 {
    if (hex_str.len == 0) return LexerError.InvalidHexLiteral;

    var result: u256 = 0;
    var digit_count: u32 = 0;

    for (hex_str) |c| {
        if (c == '_') continue;
        if (!isHexDigit(c)) return LexerError.InvalidHexLiteral;
        if (digit_count >= 64) return LexerError.NumberTooLarge;

        const shifted = @mulWithOverflow(result, 16);
        if (shifted[1] != 0) return LexerError.NumberTooLarge;
        result = shifted[0];

        const digit_value: u8 = switch (c) {
            '0'...'9' => c - '0',
            'a'...'f' => c - 'a' + 10,
            'A'...'F' => c - 'A' + 10,
            else => return LexerError.InvalidHexLiteral,
        };

        const sum = @addWithOverflow(result, @as(u256, digit_value));
        if (sum[1] != 0) return LexerError.NumberTooLarge;
        result = sum[0];

        digit_count += 1;
    }

    if (digit_count == 0) return LexerError.InvalidHexLiteral;
    return result;
}

/// Parse 40-char hex string to 20-byte address.
pub fn parseAddressToBytes(address_str: []const u8) LexerError!([20]u8) {
    if (address_str.len != 40) return LexerError.InvalidAddressFormat;

    var result: [20]u8 = undefined;
    var i: usize = 0;
    while (i < 40) : (i += 2) {
        if (!isHexDigit(address_str[i]) or !isHexDigit(address_str[i + 1])) {
            return LexerError.InvalidAddressFormat;
        }
        result[i / 2] = (hexNibble(address_str[i]) << 4) | hexNibble(address_str[i + 1]);
    }
    return result;
}

fn hexNibble(c: u8) u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => unreachable, // caller has already validated via isHexDigit
    };
}
