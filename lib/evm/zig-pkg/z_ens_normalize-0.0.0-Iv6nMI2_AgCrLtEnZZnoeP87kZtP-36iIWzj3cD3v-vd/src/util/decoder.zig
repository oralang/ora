const std = @import("std");
const Allocator = std.mem.Allocator;

/// Decoder is a stateful bit-stream reader that decodes compressed binary data.
/// It uses bit-packed format, magic number encoding, unary encoding, and zigzag encoding.
pub const Decoder = struct {
    /// The compressed byte buffer to read from
    buf: []const u8,
    /// Current byte position in buf
    pos: usize,
    /// Magic numbers for variable-length encoding
    magic: []const i32,
    /// Current byte being processed bit-by-bit
    word: u8,
    /// Current bit mask (1, 2, 4, 8, 16, 32, 64, 128)
    bit: u8,

    /// Initialize a new decoder from a byte buffer.
    /// Note: The caller must manage the lifetime of the magic array.
    pub fn init(buf: []const u8, allocator: Allocator) !Decoder {
        var d = Decoder{
            .buf = buf,
            .pos = 0,
            .magic = &[_]i32{}, // Temporary empty slice
            .word = 0,
            .bit = 0,
        };
        d.magic = try d.readMagic(allocator);
        return d;
    }

    /// Free allocated memory (magic array)
    pub fn deinit(self: *Decoder, allocator: Allocator) void {
        allocator.free(self.magic);
    }

    /// Assert that we have reached the end of the buffer.
    /// Panics if there are unread bytes remaining.
    pub fn assertEOF(self: *Decoder) void {
        if (self.pos < self.buf.len) {
            std.debug.panic("expected eof: {d}/{d}", .{ self.pos, self.buf.len });
        }
    }

    /// Read a variable-length encoded unsigned integer using the magic table.
    pub fn ReadUnsigned(self: *Decoder) i32 {
        var a: i32 = 0;
        var w: i32 = 0;
        var i: usize = 0;
        while (true) : (i += 1) {
            w = self.magic[i];
            const n: i32 = @as(i32, 1) << @intCast(w);
            if (i + 1 == self.magic.len or !self.readBit()) {
                break;
            }
            a += n;
        }
        return a + self.readBinary(w);
    }

    /// Read an array of n integers stored as ascending deltas.
    /// Each value is prev + 1 + delta.
    pub fn ReadSortedAscending(self: *Decoder, n: i32, allocator: Allocator) ![]i32 {
        const transformFn = struct {
            fn transform(prev: i32, x: i32) i32 {
                return prev + 1 + x;
            }
        }.transform;
        return self.readArray(n, transformFn, allocator);
    }

    /// Read an array of n integers stored as signed deltas (using zigzag encoding).
    /// Each value is prev + asSigned(delta).
    pub fn ReadUnsortedDeltas(self: *Decoder, n: i32, allocator: Allocator) ![]i32 {
        const transformFn = struct {
            fn transform(prev: i32, x: i32) i32 {
                return prev + asSigned(x);
            }
        }.transform;
        return self.readArray(n, transformFn, allocator);
    }

    /// Read a string as Unicode codepoints stored using ReadUnsortedDeltas.
    /// Returns a UTF-8 encoded byte slice.
    pub fn ReadString(self: *Decoder, allocator: Allocator) ![]u8 {
        const n = self.ReadUnsigned();
        const v = try self.ReadUnsortedDeltas(n, allocator);
        defer allocator.free(v);

        // Convert codepoints to UTF-8
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(allocator);

        for (v) |cp| {
            const codepoint: u21 = @intCast(cp);
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(codepoint, &buf) catch {
                return error.InvalidCodepoint;
            };
            try result.appendSlice(allocator, buf[0..len]);
        }

        return result.toOwnedSlice(allocator);
    }

    /// Read an array of unique integers with an optimized encoding for consecutive runs.
    pub fn ReadUnique(self: *Decoder, allocator: Allocator) ![]i32 {
        const v = try self.ReadSortedAscending(self.ReadUnsigned(), allocator);
        const n = self.ReadUnsigned();

        if (n > 0) {
            const vX = try self.ReadSortedAscending(n, allocator);
            defer allocator.free(vX);
            const vS = try self.ReadUnsortedDeltas(n, allocator);
            defer allocator.free(vS);

            // Count total additional elements needed
            var total_additional: usize = 0;
            for (0..@intCast(n)) |i| {
                total_additional += @intCast(vS[i]);
            }

            // Reallocate v to fit additional elements
            const new_v = try allocator.realloc(v, v.len + total_additional);
            var idx = v.len;

            // Append consecutive runs
            for (0..@intCast(n)) |i| {
                const start = vX[i];
                const count = vS[i];
                for (0..@intCast(count)) |j| {
                    new_v[idx] = start + @as(i32, @intCast(j));
                    idx += 1;
                }
            }

            // Sort the result (matches reference implementation in utils.js:135)
            std.mem.sort(i32, new_v, {}, comptime std.sort.asc(i32));
            return new_v;
        }

        return v;
    }

    /// Same as ReadUnique() but sorts the result before returning.
    pub fn ReadSortedUnique(self: *Decoder, allocator: Allocator) ![]i32 {
        const v = try self.ReadUnique(allocator);
        std.mem.sort(i32, v, {}, comptime std.sort.asc(i32));
        return v;
    }

    /// Read the magic number table from the stream header.
    /// This is called during initialization.
    fn readMagic(self: *Decoder, allocator: Allocator) ![]i32 {
        var list: std.ArrayListUnmanaged(i32) = .{};
        errdefer list.deinit(allocator);

        var w: i32 = 0;
        while (true) {
            const dw = self.readUnary();
            if (dw == 0) break;
            w += dw;
            try list.append(allocator, w);
        }

        return list.toOwnedSlice(allocator);
    }

    /// Read a single bit from the stream.
    /// This is the foundation of all other read operations.
    fn readBit(self: *Decoder) bool {
        if (self.bit == 0) {
            self.word = self.buf[self.pos];
            self.pos += 1;
            self.bit = 1;
        }
        const bit = (self.word & self.bit) != 0;
        self.bit <<= 1;
        return bit;
    }

    /// Read a unary-encoded number.
    /// Counts consecutive 1-bits until hitting a 0-bit.
    fn readUnary(self: *Decoder) i32 {
        var x: i32 = 0;
        while (self.readBit()) {
            x += 1;
        }
        return x;
    }

    /// Read a binary number of w bits.
    /// Reads from most significant bit to least significant bit.
    fn readBinary(self: *Decoder, w: i32) i32 {
        var x: i32 = 0;
        var b: i32 = @as(i32, 1) << @intCast(w - 1);
        while (b != 0) : (b >>= 1) {
            if (self.readBit()) {
                x |= b;
            }
        }
        return x;
    }

    /// Generic array reader that reads n unsigned integers and applies a function
    /// to transform each value based on the previous value.
    fn readArray(self: *Decoder, n: i32, comptime transformFn: fn (i32, i32) i32, allocator: Allocator) ![]i32 {
        const v = try allocator.alloc(i32, @intCast(n));
        var prev: i32 = -1;
        for (0..@intCast(n)) |i| {
            const x = self.ReadUnsigned();
            v[i] = transformFn(prev, x);
            prev = v[i];
        }
        return v;
    }
};

/// Convert an unsigned integer to a signed integer using zigzag encoding.
/// Zigzag encoding: 0 -> 0, 1 -> -1, 2 -> 1, 3 -> -2, 4 -> 2, ...
pub fn asSigned(i: i32) i32 {
    if ((i & 1) != 0) {
        return ~i >> 1;
    } else {
        return i >> 1;
    }
}
