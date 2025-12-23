/// Source span to track position info in the source code
pub const SourceSpan = struct {
    // File identity for multi-file projects (0 = unknown/default)
    file_id: u32 = 0,
    // 1-based caret position
    line: u32,
    column: u32,
    // Byte length of the span
    length: u32,
    // Start byte offset within file (for precise mapping)
    byte_offset: u32 = 0,
    // Optional original slice
    lexeme: ?[]const u8 = null,
};
