//! Comptime Heap
//!
//! Manages heap-allocated aggregates (arrays, tuples, structs, bytes, strings)
//! with copy-on-write semantics for correct value-semantic mutation.

const std = @import("std");
const value = @import("value.zig");
const CtValue = value.CtValue;
const HeapId = value.HeapId;
const TypeId = value.TypeId;
const FieldId = value.FieldId;

/// Aggregate stored on the comptime heap
pub const CtAggregate = struct {
    /// Whether this aggregate is shared (for COW)
    shared: bool = false,

    /// The aggregate data
    data: AggregateData,

    pub const AggregateData = union(enum) {
        bytes: []u8,
        string: []u8,
        array: ArrayData,
        slice: SliceData,
        map: MapData,
        tuple: TupleData,
        struct_val: StructData,
    };

    pub const ArrayData = struct {
        elems: []CtValue,
        elem_type: ?TypeId = null,
    };

    pub const TupleData = struct {
        elems: []CtValue,
    };

    pub const SliceData = struct {
        elems: []CtValue,
        elem_type: ?TypeId = null,
    };

    pub const MapData = struct {
        entries: []MapEntry,
        key_type: ?TypeId = null,
        value_type: ?TypeId = null,
    };

    pub const MapEntry = struct {
        key: CtValue,
        value: CtValue,
    };

    pub const StructData = struct {
        type_id: TypeId,
        fields: []StructField,
    };

    pub const StructField = struct {
        field_id: FieldId,
        value: CtValue,
    };
};

/// Comptime heap for managing aggregates with COW semantics
pub const CtHeap = struct {
    allocator: std.mem.Allocator,

    /// All heap-allocated aggregates
    aggregates: std.ArrayList(CtAggregate),

    /// Free list for reuse (optional optimization)
    free_list: std.ArrayList(HeapId),

    /// Statistics
    total_bytes: usize = 0,

    pub fn init(allocator: std.mem.Allocator) CtHeap {
        return .{
            .allocator = allocator,
            .aggregates = .empty,
            .free_list = .empty,
            .total_bytes = 0,
        };
    }

    pub fn deinit(self: *CtHeap) void {
        // Free all aggregate data
        for (self.aggregates.items) |*agg| {
            self.freeAggregateData(agg);
        }
        self.aggregates.deinit(self.allocator);
        self.free_list.deinit(self.allocator);
    }

    fn freeAggregateData(self: *CtHeap, agg: *CtAggregate) void {
        switch (agg.data) {
            .bytes => |b| self.allocator.free(b),
            .string => |s| self.allocator.free(s),
            .array => |a| self.allocator.free(a.elems),
            .slice => |s| self.allocator.free(s.elems),
            .map => |m| self.allocator.free(m.entries),
            .tuple => |t| self.allocator.free(t.elems),
            .struct_val => |s| self.allocator.free(s.fields),
        }
    }

    /// Allocate a new aggregate and return its HeapId
    fn alloc(self: *CtHeap, agg: CtAggregate) !HeapId {
        // Try to reuse from free list
        if (self.free_list.items.len > 0) {
            const id = self.free_list.pop() orelse unreachable;
            self.aggregates.items[id] = agg;
            return id;
        }

        const id: HeapId = @intCast(self.aggregates.items.len);
        try self.aggregates.append(self.allocator, agg);
        return id;
    }

    /// Get aggregate by id
    pub fn get(self: *CtHeap, id: HeapId) *CtAggregate {
        return &self.aggregates.items[id];
    }

    /// Get aggregate by id (const)
    pub fn getConst(self: *const CtHeap, id: HeapId) *const CtAggregate {
        return &self.aggregates.items[id];
    }

    // ========================================================================
    // Allocation helpers
    // ========================================================================

    /// Allocate a bytes array
    pub fn allocBytes(self: *CtHeap, data: []const u8) !HeapId {
        const copy = try self.allocator.dupe(u8, data);
        return self.allocBytesOwned(copy);
    }

    /// Allocate a bytes array, taking ownership of an allocator-owned buffer.
    pub fn allocBytesOwned(self: *CtHeap, data: []u8) !HeapId {
        errdefer self.allocator.free(data);
        self.total_bytes += data.len;
        errdefer self.total_bytes -= data.len;
        return self.alloc(.{
            .data = .{ .bytes = data },
        });
    }

    /// Allocate a string
    pub fn allocString(self: *CtHeap, data: []const u8) !HeapId {
        const copy = try self.allocator.dupe(u8, data);
        return self.allocStringOwned(copy);
    }

    /// Allocate a string, taking ownership of an allocator-owned buffer.
    pub fn allocStringOwned(self: *CtHeap, data: []u8) !HeapId {
        errdefer self.allocator.free(data);
        self.total_bytes += data.len;
        errdefer self.total_bytes -= data.len;
        return self.alloc(.{
            .data = .{ .string = data },
        });
    }

    /// Allocate an array
    pub fn allocArray(self: *CtHeap, elems: []const CtValue) !HeapId {
        const copy = try self.allocator.dupe(CtValue, elems);
        return self.allocArrayOwned(copy);
    }

    /// Allocate an array, taking ownership of an allocator-owned element buffer.
    pub fn allocArrayOwned(self: *CtHeap, elems: []CtValue) !HeapId {
        errdefer self.allocator.free(elems);
        const bytes = elems.len * @sizeOf(CtValue);
        self.total_bytes += bytes;
        errdefer self.total_bytes -= bytes;
        return self.alloc(.{
            .data = .{ .array = .{ .elems = elems } },
        });
    }

    /// Allocate a slice
    pub fn allocSlice(self: *CtHeap, elems: []const CtValue) !HeapId {
        const copy = try self.allocator.dupe(CtValue, elems);
        return self.allocSliceOwned(copy);
    }

    /// Allocate a slice, taking ownership of an allocator-owned element buffer.
    pub fn allocSliceOwned(self: *CtHeap, elems: []CtValue) !HeapId {
        errdefer self.allocator.free(elems);
        const bytes = elems.len * @sizeOf(CtValue);
        self.total_bytes += bytes;
        errdefer self.total_bytes -= bytes;
        return self.alloc(.{
            .data = .{ .slice = .{ .elems = elems } },
        });
    }

    /// Allocate a map
    pub fn allocMap(self: *CtHeap, entries: []const CtAggregate.MapEntry) !HeapId {
        const copy = try self.allocator.dupe(CtAggregate.MapEntry, entries);
        return self.allocMapOwned(copy);
    }

    /// Allocate a map, taking ownership of an allocator-owned entry buffer.
    pub fn allocMapOwned(self: *CtHeap, entries: []CtAggregate.MapEntry) !HeapId {
        errdefer self.allocator.free(entries);
        const bytes = entries.len * @sizeOf(CtAggregate.MapEntry);
        self.total_bytes += bytes;
        errdefer self.total_bytes -= bytes;
        return self.alloc(.{
            .data = .{ .map = .{ .entries = entries } },
        });
    }

    /// Allocate a tuple
    pub fn allocTuple(self: *CtHeap, elems: []const CtValue) !HeapId {
        const copy = try self.allocator.dupe(CtValue, elems);
        return self.allocTupleOwned(copy);
    }

    /// Allocate a tuple, taking ownership of an allocator-owned element buffer.
    pub fn allocTupleOwned(self: *CtHeap, elems: []CtValue) !HeapId {
        errdefer self.allocator.free(elems);
        const bytes = elems.len * @sizeOf(CtValue);
        self.total_bytes += bytes;
        errdefer self.total_bytes -= bytes;
        return self.alloc(.{
            .data = .{ .tuple = .{ .elems = elems } },
        });
    }

    /// Allocate a struct
    pub fn allocStruct(self: *CtHeap, type_id: TypeId, fields: []const CtAggregate.StructField) !HeapId {
        const copy = try self.allocator.dupe(CtAggregate.StructField, fields);
        return self.allocStructOwned(type_id, copy);
    }

    /// Allocate a struct, taking ownership of an allocator-owned field buffer.
    pub fn allocStructOwned(self: *CtHeap, type_id: TypeId, fields: []CtAggregate.StructField) !HeapId {
        errdefer self.allocator.free(fields);
        const bytes = fields.len * @sizeOf(CtAggregate.StructField);
        self.total_bytes += bytes;
        errdefer self.total_bytes -= bytes;
        return self.alloc(.{
            .data = .{ .struct_val = .{ .type_id = type_id, .fields = fields } },
        });
    }

    // ========================================================================
    // Accessors for specific types
    // ========================================================================

    pub fn getBytes(self: *const CtHeap, id: HeapId) []const u8 {
        return self.aggregates.items[id].data.bytes;
    }

    pub fn getString(self: *const CtHeap, id: HeapId) []const u8 {
        return self.aggregates.items[id].data.string;
    }

    pub fn getArray(self: *const CtHeap, id: HeapId) CtAggregate.ArrayData {
        return self.aggregates.items[id].data.array;
    }

    pub fn getSlice(self: *const CtHeap, id: HeapId) CtAggregate.SliceData {
        return self.aggregates.items[id].data.slice;
    }

    pub fn getMap(self: *const CtHeap, id: HeapId) CtAggregate.MapData {
        return self.aggregates.items[id].data.map;
    }

    pub fn getTuple(self: *const CtHeap, id: HeapId) CtAggregate.TupleData {
        return self.aggregates.items[id].data.tuple;
    }

    pub fn getStruct(self: *const CtHeap, id: HeapId) CtAggregate.StructData {
        return self.aggregates.items[id].data.struct_val;
    }

    // ========================================================================
    // Copy-on-Write support
    // ========================================================================

    /// Mark an aggregate as shared (for COW)
    pub fn markShared(self: *CtHeap, id: HeapId) void {
        self.aggregates.items[id].shared = true;
    }

    /// Ensure an aggregate is unique (clone if shared)
    pub fn ensureUnique(self: *CtHeap, id: HeapId) !HeapId {
        const agg = &self.aggregates.items[id];
        if (!agg.shared) return id;

        // Clone the aggregate
        const new_id = try self.clone(id);
        self.aggregates.items[new_id].shared = false;
        return new_id;
    }

    /// Clone an aggregate (deep copy)
    pub fn clone(self: *CtHeap, id: HeapId) !HeapId {
        const src = self.aggregates.items[id];
        return switch (src.data) {
            .bytes => |b| try self.allocBytes(b),
            .string => |s| try self.allocString(s),
            .array => |a| try self.allocArray(a.elems),
            .slice => |s| try self.allocSlice(s.elems),
            .map => |m| try self.allocMap(m.entries),
            .tuple => |t| try self.allocTuple(t.elems),
            .struct_val => |s| try self.allocStruct(s.type_id, s.fields),
        };
    }

    // ========================================================================
    // Mutation helpers
    // ========================================================================

    /// Set array element (handles COW)
    pub fn setArrayElem(self: *CtHeap, id: HeapId, index: usize, val: CtValue) !HeapId {
        const unique_id = try self.ensureUnique(id);
        const arr = &self.aggregates.items[unique_id].data.array;
        if (index >= arr.elems.len) return error.IndexOutOfBounds;
        arr.elems[index] = val;
        return unique_id;
    }

    /// Set slice element (handles COW)
    pub fn setSliceElem(self: *CtHeap, id: HeapId, index: usize, val: CtValue) !HeapId {
        const unique_id = try self.ensureUnique(id);
        const slice = &self.aggregates.items[unique_id].data.slice;
        if (index >= slice.elems.len) return error.IndexOutOfBounds;
        slice.elems[index] = val;
        return unique_id;
    }

    /// Set struct field (handles COW)
    pub fn setStructField(self: *CtHeap, id: HeapId, field_id: FieldId, val: CtValue) !HeapId {
        const unique_id = try self.ensureUnique(id);
        const s = &self.aggregates.items[unique_id].data.struct_val;
        for (s.fields) |*f| {
            if (f.field_id == field_id) {
                f.value = val;
                return unique_id;
            }
        }
        return error.FieldNotFound;
    }

    /// Insert or replace a map entry (handles COW)
    pub fn setMapEntry(self: *CtHeap, id: HeapId, key: CtValue, val: CtValue, eql: *const fn (CtValue, CtValue) bool) !HeapId {
        const unique_id = try self.ensureUnique(id);
        const map = &self.aggregates.items[unique_id].data.map;
        for (map.entries) |*entry| {
            if (eql(entry.key, key)) {
                entry.value = val;
                return unique_id;
            }
        }

        const old_entries = map.entries;
        const grown = try self.allocator.alloc(CtAggregate.MapEntry, old_entries.len + 1);
        @memcpy(grown[0..old_entries.len], old_entries);
        grown[old_entries.len] = .{ .key = key, .value = val };
        self.allocator.free(old_entries);
        map.entries = grown;
        self.total_bytes += @sizeOf(CtAggregate.MapEntry);
        return unique_id;
    }

    /// Get number of aggregates
    pub fn count(self: *const CtHeap) usize {
        return self.aggregates.items.len;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CtHeap basic allocation" {
    var heap = CtHeap.init(std.testing.allocator);
    defer heap.deinit();

    const bytes_id = try heap.allocBytes("hello");
    try std.testing.expectEqualStrings("hello", heap.getBytes(bytes_id));

    const arr_id = try heap.allocArray(&[_]CtValue{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    });
    const arr = heap.getArray(arr_id);
    try std.testing.expectEqual(@as(usize, 3), arr.elems.len);
    try std.testing.expectEqual(@as(u256, 2), arr.elems[1].integer);
}

test "CtHeap copy-on-write" {
    var heap = CtHeap.init(std.testing.allocator);
    defer heap.deinit();

    // Create array
    const id1 = try heap.allocArray(&[_]CtValue{
        .{ .integer = 1 },
        .{ .integer = 2 },
    });

    // Mark as shared (simulating assignment)
    heap.markShared(id1);

    // Mutate (should create a copy)
    const id2 = try heap.setArrayElem(id1, 0, .{ .integer = 99 });

    // id2 should be different from id1
    try std.testing.expect(id1 != id2);

    // Original should be unchanged
    try std.testing.expectEqual(@as(u256, 1), heap.getArray(id1).elems[0].integer);

    // New copy should have the mutation
    try std.testing.expectEqual(@as(u256, 99), heap.getArray(id2).elems[0].integer);
}

test "CtHeap struct operations" {
    var heap = CtHeap.init(std.testing.allocator);
    defer heap.deinit();

    const id = try heap.allocStruct(42, &[_]CtAggregate.StructField{
        .{ .field_id = 0, .value = .{ .integer = 100 } },
        .{ .field_id = 1, .value = .{ .boolean = true } },
    });

    const s = heap.getStruct(id);
    try std.testing.expectEqual(@as(TypeId, 42), s.type_id);
    try std.testing.expectEqual(@as(usize, 2), s.fields.len);
    try std.testing.expectEqual(@as(u256, 100), s.fields[0].value.integer);
}
