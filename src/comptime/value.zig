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
// TypeId ↔ OraType mapping (well-known IDs for primitives)
// ============================================================================

const OraType = @import("../ast/type_info.zig").OraType;

/// Well-known TypeId values for primitive Ora types.
/// These are stable constants used by the comptime system.
pub const type_ids = struct {
    pub const u8_id: TypeId = 1;
    pub const u16_id: TypeId = 2;
    pub const u32_id: TypeId = 3;
    pub const u64_id: TypeId = 4;
    pub const u128_id: TypeId = 5;
    pub const u256_id: TypeId = 6;
    pub const i8_id: TypeId = 7;
    pub const i16_id: TypeId = 8;
    pub const i32_id: TypeId = 9;
    pub const i64_id: TypeId = 10;
    pub const i128_id: TypeId = 11;
    pub const i256_id: TypeId = 12;
    pub const bool_id: TypeId = 13;
    pub const address_id: TypeId = 14;
    pub const string_id: TypeId = 15;
    pub const bytes_id: TypeId = 16;
    pub const void_id: TypeId = 17;

    /// Convert an OraType to a well-known TypeId (primitives only).
    pub fn fromOraType(ot: OraType) ?TypeId {
        return switch (ot) {
            .u8 => u8_id,
            .u16 => u16_id,
            .u32 => u32_id,
            .u64 => u64_id,
            .u128 => u128_id,
            .u256 => u256_id,
            .i8 => i8_id,
            .i16 => i16_id,
            .i32 => i32_id,
            .i64 => i64_id,
            .i128 => i128_id,
            .i256 => i256_id,
            .bool => bool_id,
            .address => address_id,
            .string => string_id,
            .bytes => bytes_id,
            .void => void_id,
            else => null,
        };
    }

    /// Convert a well-known TypeId back to an OraType.
    pub fn toOraType(tid: TypeId) ?OraType {
        return switch (tid) {
            u8_id => OraType{ .u8 = {} },
            u16_id => OraType{ .u16 = {} },
            u32_id => OraType{ .u32 = {} },
            u64_id => OraType{ .u64 = {} },
            u128_id => OraType{ .u128 = {} },
            u256_id => OraType{ .u256 = {} },
            i8_id => OraType{ .i8 = {} },
            i16_id => OraType{ .i16 = {} },
            i32_id => OraType{ .i32 = {} },
            i64_id => OraType{ .i64 = {} },
            i128_id => OraType{ .i128 = {} },
            i256_id => OraType{ .i256 = {} },
            bool_id => OraType{ .bool = {} },
            address_id => OraType{ .address = {} },
            string_id => OraType{ .string = {} },
            bytes_id => OraType{ .bytes = {} },
            void_id => OraType{ .void = {} },
            else => null,
        };
    }
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
