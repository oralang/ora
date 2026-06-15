//! Comptime Value Types
//!
//! Evaluator-local value model. CtValue may contain HeapId references that are
//! valid only within one CtEnv lifetime.

const std = @import("std");

// ============================================================================
// Handle Types
// ============================================================================

/// Slot index in CtEnv.slots
pub const SlotId = u32;

/// Heap index in CtEnv.heap (valid only within CtEnv lifetime)
pub const HeapId = u32;

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
    slice_ref: HeapId,
    map_ref: HeapId,
    tuple_ref: HeapId,
    struct_ref: HeapId,

    // ADT/sum value (payload may reference heap)
    adt_val: CtAdt,
    error_union_val: CtErrorUnion,

    // Type-level (stable handle)
    type_val: TypeId,

    void_val,

    /// Check if value is heap-backed
    pub fn isHeapBacked(self: CtValue) bool {
        return switch (self) {
            .bytes_ref, .string_ref, .array_ref, .slice_ref, .map_ref, .tuple_ref, .struct_ref => true,
            .adt_val => |e| e.payload != null,
            .error_union_val => true,
            else => false,
        };
    }

    /// Get the heap id if heap-backed
    pub fn getHeapId(self: CtValue) ?HeapId {
        return switch (self) {
            .bytes_ref, .string_ref, .array_ref, .slice_ref, .map_ref, .tuple_ref, .struct_ref => |id| id,
            .adt_val => |value| value.payload,
            .error_union_val => |value| value.payload,
            else => null,
        };
    }
};

/// ADT/sum value with optional payload.
pub const CtAdt = struct {
    type_id: TypeId,
    variant_id: VariantId,
    payload: ?HeapId,
};

pub const CtEnum = CtAdt;

/// Result/error-union value. Payload is a one-element tuple containing either
/// the success value or the error value.
pub const CtErrorUnion = struct {
    is_error: bool,
    payload: HeapId,
};

const builtin = @import("ora_types").builtin;

/// Well-known TypeId values for primitive Ora types.
/// These are stable constants used by the comptime system.
pub const type_ids = struct {
    fn builtinComptimeTypeId(comptime id: builtin.BuiltinTypeId) TypeId {
        return @intCast(builtin.lookupBuiltinById(id).comptime_type_id);
    }

    pub const u8_id: TypeId = builtinComptimeTypeId(.u8);
    pub const u16_id: TypeId = builtinComptimeTypeId(.u16);
    pub const u32_id: TypeId = builtinComptimeTypeId(.u32);
    pub const u64_id: TypeId = builtinComptimeTypeId(.u64);
    pub const u128_id: TypeId = builtinComptimeTypeId(.u128);
    pub const u256_id: TypeId = builtinComptimeTypeId(.u256);
    pub const i8_id: TypeId = builtinComptimeTypeId(.i8);
    pub const i16_id: TypeId = builtinComptimeTypeId(.i16);
    pub const i32_id: TypeId = builtinComptimeTypeId(.i32);
    pub const i64_id: TypeId = builtinComptimeTypeId(.i64);
    pub const i128_id: TypeId = builtinComptimeTypeId(.i128);
    pub const i256_id: TypeId = builtinComptimeTypeId(.i256);
    pub const bool_id: TypeId = builtinComptimeTypeId(.bool);
    pub const address_id: TypeId = builtinComptimeTypeId(.address);
    pub const string_id: TypeId = builtinComptimeTypeId(.string);
    pub const bytes_id: TypeId = builtinComptimeTypeId(.bytes);
    pub const void_id: TypeId = builtinComptimeTypeId(.void);
    pub const u160_id: TypeId = builtinComptimeTypeId(.u160);
};

// ============================================================================
// Tests
// ============================================================================

test "CtValue heap detection" {
    const int_val = CtValue{ .integer = 42 };
    const arr_val = CtValue{ .array_ref = 0 };
    const adt_payload_val = CtValue{ .adt_val = .{
        .type_id = 100,
        .variant_id = 1,
        .payload = 7,
    } };
    const adt_payloadless_val = CtValue{ .adt_val = .{
        .type_id = 100,
        .variant_id = 0,
        .payload = null,
    } };

    try std.testing.expect(!int_val.isHeapBacked());
    try std.testing.expect(arr_val.isHeapBacked());
    try std.testing.expect(adt_payload_val.isHeapBacked());
    try std.testing.expect(!adt_payloadless_val.isHeapBacked());
    try std.testing.expectEqual(@as(?HeapId, null), int_val.getHeapId());
    try std.testing.expectEqual(@as(?HeapId, 0), arr_val.getHeapId());
    try std.testing.expectEqual(@as(?HeapId, 7), adt_payload_val.getHeapId());
    try std.testing.expectEqual(@as(?HeapId, null), adt_payloadless_val.getHeapId());
}
