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

    pub const StructData = struct {
        type_id: TypeId,
        fields: []StructField,
    };

    pub const StructField = struct {
        field_id: FieldId,
        value: CtValue,
    };

    /// Convert to CtValue (for enum payloads)
    pub fn toCtValue(self: *const CtAggregate) CtValue {
        return switch (self.data) {
            .bytes => unreachable, // bytes don't convert to CtValue directly
            .string => unreachable,
            .array => |a| .{ .array_ref = @intCast(@intFromPtr(self) - @intFromPtr(a.elems.ptr)) }, // placeholder
            .tuple => unreachable,
            .struct_val => unreachable,
        };
    }
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
        self.total_bytes += copy.len;
        return self.alloc(.{
            .data = .{ .bytes = copy },
        });
    }

    /// Allocate a string
    pub fn allocString(self: *CtHeap, data: []const u8) !HeapId {
        const copy = try self.allocator.dupe(u8, data);
        self.total_bytes += copy.len;
        return self.alloc(.{
            .data = .{ .string = copy },
        });
    }

    /// Allocate an array
    pub fn allocArray(self: *CtHeap, elems: []const CtValue) !HeapId {
        const copy = try self.allocator.dupe(CtValue, elems);
        self.total_bytes += copy.len * @sizeOf(CtValue);
        return self.alloc(.{
            .data = .{ .array = .{ .elems = copy } },
        });
    }

    /// Allocate a tuple
    pub fn allocTuple(self: *CtHeap, elems: []const CtValue) !HeapId {
        const copy = try self.allocator.dupe(CtValue, elems);
        self.total_bytes += copy.len * @sizeOf(CtValue);
        return self.alloc(.{
            .data = .{ .tuple = .{ .elems = copy } },
        });
    }

    /// Allocate a struct
    pub fn allocStruct(self: *CtHeap, type_id: TypeId, fields: []const CtAggregate.StructField) !HeapId {
        const copy = try self.allocator.dupe(CtAggregate.StructField, fields);
        self.total_bytes += copy.len * @sizeOf(CtAggregate.StructField);
        return self.alloc(.{
            .data = .{ .struct_val = .{ .type_id = type_id, .fields = copy } },
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
