/// Source span to track position info in the source code
pub const SourceSpan = struct {
    // file identity for multi-file projects (0 = unknown/default)
    file_id: u32 = 0,
    // 1-based caret position
    line: u32,
    column: u32,
    // byte length of the span
    length: u32,
    // start byte offset within file (for precise mapping)
    byte_offset: u32 = 0,
    // optional original slice
    lexeme: ?[]const u8 = null,
};
