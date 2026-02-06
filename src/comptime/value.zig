//! Comptime Value Types
//!
//! Two-layer value model:
//! - CtValue: Evaluator-internal, may contain HeapId references (env-local)
//! - ConstValue/ConstId: Persistent, stored in ConstPool (compiler-wide)

const std = @import("std");

// ============================================================================
// Handle Types
// ============================================================================

/// Slot index in CtEnv.slots
pub const SlotId = u32;

/// Heap index in CtEnv.heap (valid only within CtEnv lifetime)
pub const HeapId = u32;

/// Handle into ConstPool (persistent across compilation)
pub const ConstId = u32;

/// Stable type handle (indexes into type table)
pub const TypeId = u32;

/// Stable field handle
pub const FieldId = u32;

/// Stable enum variant handle
pub const VariantId = u32;

// ============================================================================
// CtValue (Evaluator-Internal)
// ============================================================================

/// Evaluator-internal value. May contain HeapId references that are only
/// valid within the lifetime of a CtEnv.
pub const CtValue = union(enum) {
    // Primitives (self-contained)
    integer: u256,
    boolean: bool,
    address: u160,

    // Heap-backed (HeapId valid only within CtEnv lifetime)
    bytes_ref: HeapId,
    string_ref: HeapId,
    array_ref: HeapId,
    tuple_ref: HeapId,
    struct_ref: HeapId,

    // Enum (payload may reference heap)
    enum_val: CtEnum,

    // Type-level (stable handle)
    type_val: TypeId,

    void_val,

    /// Check if value is heap-backed
    pub fn isHeapBacked(self: CtValue) bool {
        return switch (self) {
            .bytes_ref, .string_ref, .array_ref, .tuple_ref, .struct_ref => true,
            .enum_val => |e| e.payload != null,
            else => false,
        };
    }

    /// Get the heap id if heap-backed
    pub fn getHeapId(self: CtValue) ?HeapId {
        return switch (self) {
            .bytes_ref, .string_ref, .array_ref, .tuple_ref, .struct_ref => |id| id,
            else => null,
        };
    }
};

/// Enum value with optional payload
pub const CtEnum = struct {
    type_id: TypeId,
    variant_id: VariantId,
    payload: ?HeapId,
};

// ============================================================================
// ConstValue (Persistent)
// ============================================================================

/// Persistent constant value stored in ConstPool. Survives across compilation
/// phases. Aggregates contain ConstIds, not nested ConstValues.
pub const ConstValue = union(enum) {
    // Primitives
    integer: u256,
    boolean: bool,
    address: u160,

    // Blobs (interned slices, stable memory in pool)
    bytes: []const u8,
    string: []const u8,

    // Aggregates (contain ConstIds, not nested ConstValue)
    array: []const ConstId,
    tuple: []const ConstId,
    struct_val: ConstStruct,

    // Enum (payload by ConstId)
    enum_val: ConstEnum,

    // Type-level constant (stable handle)
    type_val: TypeId,

    void_val,

    /// Check if this is a primitive (non-aggregate) value
    pub fn isPrimitive(self: ConstValue) bool {
        return switch (self) {
            .integer, .boolean, .address, .type_val, .void_val => true,
            else => false,
        };
    }
};

/// Persistent struct constant
pub const ConstStruct = struct {
    type_id: TypeId,
    fields: []const ConstField,
};

/// Field within a persistent struct constant
pub const ConstField = struct {
    field_id: FieldId,
    value: ConstId,
};

/// Persistent enum constant
pub const ConstEnum = struct {
    type_id: TypeId,
    variant_id: VariantId,
    payload: ?ConstId,
};

// ============================================================================
// Tests
// ============================================================================

test "CtValue heap detection" {
    const int_val = CtValue{ .integer = 42 };
    const arr_val = CtValue{ .array_ref = 0 };

    try std.testing.expect(!int_val.isHeapBacked());
    try std.testing.expect(arr_val.isHeapBacked());
    try std.testing.expectEqual(@as(?HeapId, null), int_val.getHeapId());
    try std.testing.expectEqual(@as(?HeapId, 0), arr_val.getHeapId());
}

test "ConstValue primitive detection" {
    const int_val = ConstValue{ .integer = 42 };
    const arr_val = ConstValue{ .array = &[_]ConstId{} };

    try std.testing.expect(int_val.isPrimitive());
    try std.testing.expect(!arr_val.isPrimitive());
}
