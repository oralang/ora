const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn buildFullDocumentEdit(
    arena: Allocator,
    source: []const u8,
    formatted: []const u8,
    encoding: text_edits.PositionEncoding,
) ![]types.TextEdit {
    if (std.mem.eql(u8, source, formatted)) {
        return try arena.alloc(types.TextEdit, 0);
    }

    const edits = try arena.alloc(types.TextEdit, 1);
    edits[0] = .{
        .range = .{
            .start = .{ .line = 0, .character = 0 },
            .end = textEndPosition(source, encoding),
        },
        .newText = try arena.dupe(u8, formatted),
    };
    return edits;
}

fn textEndPosition(source: []const u8, encoding: text_edits.PositionEncoding) types.Position {
    var line: u32 = 0;
    var character: u32 = 0;
    var i: usize = 0;

    while (i < source.len) {
        const byte = source[i];
        if (byte == '\n') {
            line += 1;
            character = 0;
            i += 1;
            continue;
        }

        switch (encoding) {
            .utf8 => {
                character += 1;
                i += 1;
            },
            .utf16, .utf32 => {
                const seq_len = std.unicode.utf8ByteSequenceLength(byte) catch {
                    character += 1;
                    i += 1;
                    continue;
                };
                if (i + seq_len > source.len) {
                    character += 1;
                    i += 1;
                    continue;
                }

                const cp = std.unicode.utf8Decode(source[i .. i + seq_len]) catch {
                    character += 1;
                    i += 1;
                    continue;
                };

                character += switch (encoding) {
                    .utf16 => if (cp <= 0xFFFF) 1 else 2,
                    .utf32 => 1,
                    .utf8 => unreachable,
                };
                i += seq_len;
            },
        }
    }

    return .{ .line = line, .character = character };
}
