pub fn isValidIdentifier(name: []const u8) bool {
    if (name.len == 0) return false;
    if (!isIdentifierStart(name[0])) return false;

    for (name[1..]) |ch| {
        if (!isIdentifierContinue(ch)) return false;
    }

    return true;
}

fn isIdentifierStart(ch: u8) bool {
    return (ch >= 'a' and ch <= 'z') or
        (ch >= 'A' and ch <= 'Z') or
        ch == '_';
}

fn isIdentifierContinue(ch: u8) bool {
    return isIdentifierStart(ch) or (ch >= '0' and ch <= '9');
}
