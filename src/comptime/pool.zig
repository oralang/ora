//! Constant Pool
//!
//! Compiler-wide storage for persistent compile-time constants.
//! Converts evaluator-local CtValue (with HeapId) to ConstValue (with ConstId).

const std = @import("std");
const value = @import("value.zig");
const ConstId = value.ConstId;
const ConstValue = value.ConstValue;
const ConstStruct = value.ConstStruct;
const ConstField = value.ConstField;
const ConstEnum = value.ConstEnum;
const CtValue = value.CtValue;
const HeapId = value.HeapId;
const TypeId = value.TypeId;

/// Compiler-wide constant pool. Lives for the entire compilation and owns
/// all persistent constant memory.
pub const ConstPool = struct {
    allocator: std.mem.Allocator,

    /// Main storage: ConstId indexes into this
    values: std.ArrayList(ConstValue),

    /// Blob storage (all strings/bytes live here)
    blob_arena: std.ArrayList(u8),

    /// Aggregate element storage (all []ConstId slices live here)
    element_arena: std.ArrayList(ConstId),

    /// Field storage (all []ConstField slices live here)
    field_arena: std.ArrayList(ConstField),

    /// Interning maps for deduplication
    int_intern: std.AutoHashMap(u256, ConstId),
    string_intern: std.StringHashMap(ConstId),
    bytes_intern: std.AutoHashMap(u64, ConstId), // hash -> id

    /// Sentinel for void constant (always at index 0)
    pub const VOID_ID: ConstId = 0;
    /// Sentinel for true constant (always at index 1)
    pub const TRUE_ID: ConstId = 1;
    /// Sentinel for false constant (always at index 2)
    pub const FALSE_ID: ConstId = 2;

    pub fn init(allocator: std.mem.Allocator) ConstPool {
        var pool = ConstPool{
            .allocator = allocator,
            .values = .empty,
            .blob_arena = .empty,
            .element_arena = .empty,
            .field_arena = .empty,
            .int_intern = std.AutoHashMap(u256, ConstId).init(allocator),
            .string_intern = std.StringHashMap(ConstId).init(allocator),
            .bytes_intern = std.AutoHashMap(u64, ConstId).init(allocator),
        };

        // Pre-populate common constants
        pool.values.append(allocator, .void_val) catch {};
        pool.values.append(allocator, .{ .boolean = true }) catch {};
        pool.values.append(allocator, .{ .boolean = false }) catch {};

        return pool;
    }

    pub fn deinit(self: *ConstPool) void {
        self.values.deinit(self.allocator);
        self.blob_arena.deinit(self.allocator);
        self.element_arena.deinit(self.allocator);
        self.field_arena.deinit(self.allocator);
        self.int_intern.deinit();
        self.string_intern.deinit();
        self.bytes_intern.deinit();
    }

    /// Get a stored constant by id
    pub fn get(self: *const ConstPool, id: ConstId) ConstValue {
        return self.values.items[id];
    }

    /// Error set for intern operations
    pub const InternError = error{OutOfMemory};

    /// Intern a CtValue from a CtEnv (deep-copy heap data into pool)
    pub fn intern(self: *ConstPool, env: anytype, v: CtValue) InternError!ConstId {
        return switch (v) {
            // === Primitives: check interning map or store directly ===
            .integer => |n| {
                if (self.int_intern.get(n)) |existing| return existing;
                const id = try self.store(.{ .integer = n });
                try self.int_intern.put(n, id);
                return id;
            },
            .boolean => |b| if (b) TRUE_ID else FALSE_ID,
            .address => |a| try self.store(.{ .address = a }),
            .void_val => VOID_ID,
            .type_val => |tid| try self.store(.{ .type_val = tid }),

            // === Blobs: copy into blob_arena, intern ===
            .string_ref => |heap_id| {
                const data = env.heap.getString(heap_id);
                return self.internString(data);
            },
            .bytes_ref => |heap_id| {
                const data = env.heap.getBytes(heap_id);
                return self.internBytes(data);
            },

            // === Aggregates: intern elements first, then store aggregate ===
            .array_ref => |heap_id| {
                const agg = env.heap.getArray(heap_id);
                const elem_ids = try self.internElements(env, agg.elems);
                return try self.store(.{ .array = elem_ids });
            },
            .tuple_ref => |heap_id| {
                const agg = env.heap.getTuple(heap_id);
                const elem_ids = try self.internElements(env, agg.elems);
                return try self.store(.{ .tuple = elem_ids });
            },
            .struct_ref => |heap_id| {
                const agg = env.heap.getStruct(heap_id);
                const fields = try self.internFields(env, agg.fields);
                return try self.store(.{ .struct_val = .{
                    .type_id = agg.type_id,
                    .fields = fields,
                } });
            },

            // === Enum: intern payload if present ===
            .enum_val => |e| {
                const payload_id: ?ConstId = if (e.payload) |p| blk: {
                    const payload_val = env.heap.get(p);
                    break :blk try self.intern(env, payload_val.toCtValue());
                } else null;
                return try self.store(.{ .enum_val = .{
                    .type_id = e.type_id,
                    .variant_id = e.variant_id,
                    .payload = payload_id,
                } });
            },
        };
    }

    /// Intern an integer directly (without CtEnv)
    pub fn internInt(self: *ConstPool, n: u256) !ConstId {
        if (self.int_intern.get(n)) |existing| return existing;
        const id = try self.store(.{ .integer = n });
        try self.int_intern.put(n, id);
        return id;
    }

    /// Intern a string directly (without CtEnv)
    pub fn internStringDirect(self: *ConstPool, data: []const u8) !ConstId {
        return self.internString(data);
    }

    /// Store a ConstValue and return its ConstId
    fn store(self: *ConstPool, cv: ConstValue) InternError!ConstId {
        const id: ConstId = @intCast(self.values.items.len);
        try self.values.append(self.allocator, cv);
        return id;
    }

    /// Intern a slice of CtValue elements into []ConstId (stored in element_arena)
    fn internElements(self: *ConstPool, env: anytype, elems: []const CtValue) InternError![]const ConstId {
        const start = self.element_arena.items.len;
        for (elems) |elem| {
            const elem_id = try self.intern(env, elem);
            try self.element_arena.append(self.allocator, elem_id);
        }
        return self.element_arena.items[start..];
    }

    /// Intern struct fields into []ConstField (stored in field_arena)
    fn internFields(self: *ConstPool, env: anytype, fields: anytype) InternError![]const ConstField {
        const start = self.field_arena.items.len;
        for (fields) |f| {
            const value_id = try self.intern(env, f.value);
            try self.field_arena.append(self.allocator, .{
                .field_id = f.field_id,
                .value = value_id,
            });
        }
        return self.field_arena.items[start..];
    }

    /// Intern a string (with deduplication)
    fn internString(self: *ConstPool, data: []const u8) InternError!ConstId {
        // Check if already interned (need to look up in existing blob_arena)
        if (self.string_intern.get(data)) |existing| return existing;

        // Copy into blob_arena
        const start = self.blob_arena.items.len;
        try self.blob_arena.appendSlice(self.allocator, data);
        const slice = self.blob_arena.items[start..];

        const id = try self.store(.{ .string = slice });
        try self.string_intern.put(slice, id);
        return id;
    }

    /// Intern bytes (with hash-based deduplication)
    fn internBytes(self: *ConstPool, data: []const u8) InternError!ConstId {
        const hash = std.hash.Wyhash.hash(0, data);
        if (self.bytes_intern.get(hash)) |existing| {
            // Verify actual match (hash collision check)
            const existing_data = self.get(existing).bytes;
            if (std.mem.eql(u8, existing_data, data)) {
                return existing;
            }
        }

        // Copy into blob_arena
        const start = self.blob_arena.items.len;
        try self.blob_arena.appendSlice(self.allocator, data);
        const slice = self.blob_arena.items[start..];

        const id = try self.store(.{ .bytes = slice });
        try self.bytes_intern.put(hash, id);
        return id;
    }

    /// Get number of constants stored
    pub fn count(self: *const ConstPool) usize {
        return self.values.items.len;
    }

    /// Get total memory used by blob arena
    pub fn blobMemory(self: *const ConstPool) usize {
        return self.blob_arena.items.len;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ConstPool basic operations" {
    var pool = ConstPool.init(std.testing.allocator);
    defer pool.deinit();

    // Pre-populated constants
    try std.testing.expectEqual(ConstValue.void_val, pool.get(ConstPool.VOID_ID));
    try std.testing.expectEqual(ConstValue{ .boolean = true }, pool.get(ConstPool.TRUE_ID));
    try std.testing.expectEqual(ConstValue{ .boolean = false }, pool.get(ConstPool.FALSE_ID));
}

test "ConstPool integer interning" {
    var pool = ConstPool.init(std.testing.allocator);
    defer pool.deinit();

    const id1 = try pool.internInt(42);
    const id2 = try pool.internInt(42);
    const id3 = try pool.internInt(100);

    // Same value should return same id
    try std.testing.expectEqual(id1, id2);
    // Different value should return different id
    try std.testing.expect(id1 != id3);

    try std.testing.expectEqual(@as(u256, 42), pool.get(id1).integer);
    try std.testing.expectEqual(@as(u256, 100), pool.get(id3).integer);
}

test "ConstPool string interning" {
    var pool = ConstPool.init(std.testing.allocator);
    defer pool.deinit();

    const id1 = try pool.internStringDirect("hello");
    const id2 = try pool.internStringDirect("hello");
    const id3 = try pool.internStringDirect("world");

    // Same string should return same id
    try std.testing.expectEqual(id1, id2);
    // Different string should return different id
    try std.testing.expect(id1 != id3);

    try std.testing.expectEqualStrings("hello", pool.get(id1).string);
    try std.testing.expectEqualStrings("world", pool.get(id3).string);
}
