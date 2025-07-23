const std = @import("std");
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Allocator = std.mem.Allocator;

/// Ora IR Version
pub const IR_VERSION = "1.0";

/// Source location for debug information
pub const SourceLocation = struct {
    line: u32,
    column: u32,
    length: u32,
};

/// Memory regions as defined in the formal specification
pub const Region = enum {
    stack,
    memory,
    storage,
    tstore,
    const_,
    immutable,

    pub fn toString(self: Region) []const u8 {
        return switch (self) {
            .stack => "stack",
            .memory => "memory",
            .storage => "storage",
            .tstore => "tstore",
            .const_ => "const",
            .immutable => "immutable",
        };
    }
};

/// Mutability specification
pub const Mutability = enum {
    let,
    variable,

    pub fn toString(self: Mutability) []const u8 {
        return switch (self) {
            .let => "let",
            .variable => "var",
        };
    }
};

/// Visibility specification
pub const Visibility = enum {
    public,
    internal,
    private,

    pub fn toString(self: Visibility) []const u8 {
        return switch (self) {
            .public => "public",
            .internal => "internal",
            .private => "private",
        };
    }
};

/// Type system implementation
pub const Type = union(enum) {
    primitive: PrimitiveType,
    mapping: MappingType,
    slice: SliceType,
    custom: CustomType,
    struct_type: StructType, // Add struct type support
    enum_type: EnumType, // Add enum type support

    // Error handling types
    error_union: ErrorUnionType,
    result: ResultType,

    pub const PrimitiveType = enum {
        u8,
        u16,
        u32,
        u64,
        u128,
        u256,
        i8,
        i16,
        i32,
        i64,
        i128,
        i256,
        bool,
        address,
        string,
        bytes,

        pub fn toString(self: PrimitiveType) []const u8 {
            return switch (self) {
                .u8 => "u8",
                .u16 => "u16",
                .u32 => "u32",
                .u64 => "u64",
                .u128 => "u128",
                .u256 => "u256",
                .i8 => "i8",
                .i16 => "i16",
                .i32 => "i32",
                .i64 => "i64",
                .i128 => "i128",
                .i256 => "i256",
                .bool => "bool",
                .address => "address",
                .string => "string",
                .bytes => "bytes",
            };
        }
    };

    pub const MappingType = struct {
        key_type: *Type,
        value_type: *Type,
    };

    pub const SliceType = struct {
        element_type: *Type,
    };

    pub const CustomType = struct {
        name: []const u8,
    };

    pub const StructType = struct {
        name: []const u8,
        fields: []StructField,
        layout: ?StructLayout, // Memory layout information
        origin_type: ?*const @import("typer.zig").StructType, // Reference to original type

        pub const StructLayout = struct {
            total_size: u32,
            storage_slots: u32,
            alignment: u32,
            packed_efficiently: bool,
        };
    };

    pub const StructField = struct {
        name: []const u8,
        field_type: *Type,
        offset: u32, // Byte offset within struct
        slot: u32, // Storage slot number
        slot_offset: u32, // Offset within storage slot
    };

    pub const ErrorUnionType = struct {
        success_type: *Type,
    };

    pub const ResultType = struct {
        ok_type: *Type,
        error_type: *Type,
    };

    pub const EnumType = struct {
        name: []const u8,
        variants: []EnumVariant,

        pub const EnumVariant = struct {
            name: []const u8,
            value: *Expression,
        };
    };

    pub fn deinit(self: *Type, allocator: Allocator) void {
        switch (self.*) {
            .primitive => {},
            .custom => {},
            .mapping => |*mapping| {
                mapping.key_type.deinit(allocator);
                allocator.destroy(mapping.key_type);
                mapping.value_type.deinit(allocator);
                allocator.destroy(mapping.value_type);
            },
            .slice => |*slice| {
                slice.element_type.deinit(allocator);
                allocator.destroy(slice.element_type);
            },
            .error_union => |*error_union| {
                error_union.success_type.deinit(allocator);
                allocator.destroy(error_union.success_type);
            },
            .result => |*result| {
                result.ok_type.deinit(allocator);
                allocator.destroy(result.ok_type);
                result.error_type.deinit(allocator);
                allocator.destroy(result.error_type);
            },
            .struct_type => |*struct_type| {
                for (struct_type.fields) |*field| {
                    field.field_type.deinit(allocator);
                    allocator.destroy(field.field_type);
                }
                allocator.free(struct_type.fields);
            },
            .enum_type => |*enum_type| {
                for (enum_type.variants) |*variant| {
                    variant.value.deinit();
                    allocator.destroy(variant.value);
                }
                allocator.free(enum_type.variants);
            },
        }
    }

    pub fn isCompatibleWith(self: *const Type, other: *const Type) bool {
        // Implementation of type compatibility rules
        return switch (self.*) {
            .primitive => |p1| switch (other.*) {
                .primitive => |p2| p1 == p2 or self.isPromotableToType(other),
                else => false,
            },
            .mapping => |m1| switch (other.*) {
                .mapping => |m2| m1.key_type.isCompatibleWith(m2.key_type) and
                    m1.value_type.isCompatibleWith(m2.value_type),
                else => false,
            },
            .slice => |s1| switch (other.*) {
                .slice => |s2| s1.element_type.isCompatibleWith(s2.element_type),
                else => false,
            },
            .custom => |c1| switch (other.*) {
                .custom => |c2| std.mem.eql(u8, c1.name, c2.name),
                else => false,
            },
            .error_union => |eu1| switch (other.*) {
                .error_union => |eu2| eu1.success_type.isCompatibleWith(eu2.success_type),
                else => false,
            },
            .result => |r1| switch (other.*) {
                .result => |r2| r1.ok_type.isCompatibleWith(r2.ok_type) and
                    r1.error_type.isCompatibleWith(r2.error_type),
                else => false,
            },
            .struct_type => |struct_type1| switch (other.*) {
                .struct_type => |struct_type2| {
                    // Struct types are compatible if they have the same name
                    return std.mem.eql(u8, struct_type1.name, struct_type2.name);
                },
                else => false,
            },
            .enum_type => |enum_type1| switch (other.*) {
                .enum_type => |enum_type2| {
                    // Enum types are compatible if they have the same name
                    return std.mem.eql(u8, enum_type1.name, enum_type2.name);
                },
                else => false,
            },
        };
    }

    /// Check if this type can be promoted to another type
    pub fn isPromotableToType(self: *const Type, target: *const Type) bool {
        return switch (self.*) {
            .primitive => |p1| switch (target.*) {
                .primitive => |p2| self.isNumericPromotion(p1, p2),
                else => false,
            },
            else => false,
        };
    }

    /// Check if a numeric type can be promoted to another numeric type
    fn isNumericPromotion(self: *const Type, from: Type.PrimitiveType, to: Type.PrimitiveType) bool {
        _ = self;
        // Define promotion hierarchy:
        // Unsigned: u8 → u16 → u32 → u64 → u128 → u256
        // Signed: i8 → i16 → i32 → i64 → i128 → i256
        // Mixed: unsigned can promote to signed of same or larger size
        const unsigned_promotion = [_]Type.PrimitiveType{ .u8, .u16, .u32, .u64, .u128, .u256 };
        const signed_promotion = [_]Type.PrimitiveType{ .i8, .i16, .i32, .i64, .i128, .i256 };

        // Check unsigned promotion
        var from_idx: ?usize = null;
        var to_idx: ?usize = null;

        for (unsigned_promotion, 0..) |typ, i| {
            if (typ == from) from_idx = i;
            if (typ == to) to_idx = i;
        }

        if (from_idx != null and to_idx != null) {
            return from_idx.? <= to_idx.?;
        }

        // Check signed promotion
        from_idx = null;
        to_idx = null;

        for (signed_promotion, 0..) |typ, i| {
            if (typ == from) from_idx = i;
            if (typ == to) to_idx = i;
        }

        if (from_idx != null and to_idx != null) {
            return from_idx.? <= to_idx.?;
        }

        // Check unsigned to signed promotion (u8 -> i16, u16 -> i32, etc.)
        if (from == .u8 and to == .i16) return true;
        if (from == .u16 and to == .i32) return true;
        if (from == .u32 and to == .i64) return true;
        if (from == .u64 and to == .i128) return true;
        if (from == .u128 and to == .i256) return true;

        return false;
    }

    /// Get the common type for two types (used in binary operations)
    pub fn getCommonType(self: *const Type, other: *const Type, allocator: Allocator) ?Type {
        _ = allocator; // For future use with complex types

        return switch (self.*) {
            .primitive => |p1| switch (other.*) {
                .primitive => |p2| {
                    if (p1 == p2) return self.*;

                    // Promote to larger numeric type
                    if (self.isNumericPromotion(p1, p2)) return other.*;
                    if (self.isNumericPromotion(p2, p1)) return self.*;

                    return null;
                },
                else => null,
            },
            else => if (self.isCompatibleWith(other)) self.* else null,
        };
    }

    /// Infer the type of a literal value
    pub fn inferLiteralType(literal: *const Literal) Type {
        return switch (literal.*) {
            .integer => |int_str| {
                // Check if it's a negative number
                const is_negative = int_str.len > 0 and int_str[0] == '-';
                const abs_str = if (is_negative) int_str[1..] else int_str;

                if (is_negative) {
                    // Try parsing as different signed integer types
                    if (std.fmt.parseInt(i8, abs_str, 10)) |_| {
                        return Type{ .primitive = .i8 };
                    } else |_| {}

                    if (std.fmt.parseInt(i16, abs_str, 10)) |_| {
                        return Type{ .primitive = .i16 };
                    } else |_| {}

                    if (std.fmt.parseInt(i32, abs_str, 10)) |_| {
                        return Type{ .primitive = .i32 };
                    } else |_| {}

                    if (std.fmt.parseInt(i64, abs_str, 10)) |_| {
                        return Type{ .primitive = .i64 };
                    } else |_| {}

                    if (std.fmt.parseInt(i128, abs_str, 10)) |_| {
                        return Type{ .primitive = .i128 };
                    } else |_| {}

                    // Default to i256 for very large negative numbers
                    return Type{ .primitive = .i256 };
                } else {
                    // Parse integer to determine smallest suitable type
                    if (std.fmt.parseInt(u8, int_str, 10)) |_| {
                        return Type{ .primitive = .u8 };
                    } else |_| {}

                    if (std.fmt.parseInt(u16, int_str, 10)) |_| {
                        return Type{ .primitive = .u16 };
                    } else |_| {}

                    if (std.fmt.parseInt(u32, int_str, 10)) |_| {
                        return Type{ .primitive = .u32 };
                    } else |_| {}

                    if (std.fmt.parseInt(u64, int_str, 10)) |_| {
                        return Type{ .primitive = .u64 };
                    } else |_| {}

                    if (std.fmt.parseInt(u128, int_str, 10)) |_| {
                        return Type{ .primitive = .u128 };
                    } else |_| {}

                    // Default to u256 for very large numbers
                    return Type{ .primitive = .u256 };
                }
            },
            .string => Type{ .primitive = .string },
            .boolean => Type{ .primitive = .bool },
            .address => Type{ .primitive = .address },
        };
    }

    /// Check if two types are structurally equal
    pub fn isEqual(self: *const Type, other: *const Type) bool {
        if (std.meta.activeTag(self.*) != std.meta.activeTag(other.*)) {
            return false;
        }

        return switch (self.*) {
            .primitive => |p1| p1 == other.primitive,
            .struct_type => |s1| blk: {
                const s2 = other.struct_type;
                if (!std.mem.eql(u8, s1.name, s2.name)) break :blk false;
                if (s1.fields.len != s2.fields.len) break :blk false;

                for (s1.fields, s2.fields) |f1, f2| {
                    if (!std.mem.eql(u8, f1.name, f2.name)) break :blk false;
                    if (!f1.field_type.isEqual(f2.field_type)) break :blk false;
                }
                break :blk true;
            },
            .mapping => |m1| {
                const m2 = other.mapping;
                return m1.key_type.isEqual(m2.key_type) and m1.value_type.isEqual(m2.value_type);
            },
            .slice => |s1| s1.element_type.isEqual(other.slice.element_type),
            .custom => |c1| std.mem.eql(u8, c1.name, other.custom.name),
            else => false,
        };
    }

    /// Get the field type for a struct field
    pub fn getStructFieldType(self: *const Type, field_name: []const u8) ?*const Type {
        switch (self.*) {
            .struct_type => |struct_type| {
                for (struct_type.fields) |field| {
                    if (std.mem.eql(u8, field.name, field_name)) {
                        return field.field_type;
                    }
                }
                return null;
            },
            else => return null,
        }
    }

    /// Get struct field layout information
    pub fn getStructFieldLayout(self: *const Type, field_name: []const u8) ?StructField {
        switch (self.*) {
            .struct_type => |struct_type| {
                for (struct_type.fields) |field| {
                    if (std.mem.eql(u8, field.name, field_name)) {
                        return field;
                    }
                }
                return null;
            },
            else => return null,
        }
    }

    /// Check if a type has a specific field
    pub fn hasField(self: *const Type, field_name: []const u8) bool {
        return self.getStructFieldType(field_name) != null;
    }

    /// Get the size of a type in bytes
    pub fn getSize(self: *const Type) u32 {
        return switch (self.*) {
            .primitive => |p| switch (p) {
                .u8, .i8, .bool => 1,
                .u16, .i16 => 2,
                .u32, .i32 => 4,
                .u64, .i64 => 8,
                .u128, .i128 => 16,
                .u256, .i256 => 32,
                .address => 20,
                .string, .bytes => 32, // Dynamic size, return slot size
            },
            .struct_type => |struct_type| {
                if (struct_type.layout) |layout| {
                    return layout.total_size;
                } else {
                    // Calculate size from fields
                    var size: u32 = 0;
                    for (struct_type.fields) |field| {
                        size += field.field_type.getSize();
                    }
                    return size;
                }
            },
            .mapping => 32, // Storage slot size
            .slice => 32, // Dynamic size, return slot size
            .custom => 32, // Unknown size, assume slot size
            else => 32,
        };
    }
};

/// Effect system implementation
pub const Effect = struct {
    type: EffectType,
    path: AccessPath,
    condition: ?*Expression,

    pub const EffectType = enum {
        read,
        write,
        lock,
        unlock,
        emit,
    };
};

/// Access path for effects
pub const AccessPath = struct {
    base: []const u8,
    selectors: []PathSelector,
    region: Region,

    pub const PathSelector = union(enum) {
        field: struct { name: []const u8 },
        index: struct { index: *Expression },
    };
};

/// Effect set for tracking all effects
pub const EffectSet = struct {
    effects: ArrayList(Effect),
    allocator: Allocator,

    pub fn init(allocator: Allocator) EffectSet {
        return EffectSet{
            .effects = ArrayList(Effect).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *EffectSet) void {
        self.effects.deinit();
    }

    pub fn add(self: *EffectSet, effect: Effect) !void {
        try self.effects.append(effect);
    }

    pub fn merge(self: *EffectSet, other: *const EffectSet) !void {
        for (other.effects.items) |effect| {
            try self.add(effect);
        }
    }

    pub fn hasConflict(self: *const EffectSet) bool {
        // Check for lock violations and other conflicts
        for (self.effects.items) |effect1| {
            for (self.effects.items) |effect2| {
                if (self.conflictsWith(effect1, effect2)) {
                    return true;
                }
            }
        }
        return false;
    }

    /// Returns true if this effect is a state change (modifies storage)
    pub fn isStateEffect(effect: Effect) bool {
        return switch (effect.type) {
            .write => true,
            .lock => true,
            .unlock => true,
            .read => false,
            .emit => false,
        };
    }

    /// Returns true if this effect is observable (emits logs/events)
    pub fn isObservableEffect(effect: Effect) bool {
        return switch (effect.type) {
            .write => false,
            .lock => false,
            .unlock => false,
            .read => false,
            .emit => true,
        };
    }

    /// Filter effects to only include state effects
    pub fn filterStateEffects(self: *const EffectSet, allocator: Allocator) !EffectSet {
        var state_effects = EffectSet.init(allocator);
        for (self.effects.items) |effect| {
            if (isStateEffect(effect)) {
                try state_effects.add(effect);
            }
        }
        return state_effects;
    }

    /// Filter effects to only include observable effects
    pub fn filterObservableEffects(self: *const EffectSet, allocator: Allocator) !EffectSet {
        var observable_effects = EffectSet.init(allocator);
        for (self.effects.items) |effect| {
            if (isObservableEffect(effect)) {
                try observable_effects.add(effect);
            }
        }
        return observable_effects;
    }

    fn conflictsWith(self: *const EffectSet, e1: Effect, e2: Effect) bool {
        // Two effects conflict if they access the same path and at least one is a write
        _ = self;
        if (!pathsEqual(e1.path, e2.path)) return false;

        return (e1.type == .write or e2.type == .write) and
            (e1.type != e2.type);
    }

    fn pathsEqual(p1: AccessPath, p2: AccessPath) bool {
        if (!std.mem.eql(u8, p1.base, p2.base)) return false;
        if (p1.region != p2.region) return false;
        if (p1.selectors.len != p2.selectors.len) return false;

        for (p1.selectors, 0..) |s1, i| {
            if (!selectorsEqual(s1, p2.selectors[i])) return false;
        }

        return true;
    }

    fn selectorsEqual(s1: AccessPath.PathSelector, s2: AccessPath.PathSelector) bool {
        return switch (s1) {
            .field => |f1| switch (s2) {
                .field => |f2| std.mem.eql(u8, f1.name, f2.name),
                .index => false,
            },
            .index => |idx1| switch (s2) {
                .index => |idx2| expressionsEqual(idx1.index, idx2.index),
                .field => false,
            },
        };
    }

    fn expressionsEqual(e1: *Expression, e2: *Expression) bool {
        // Simplified expression equality for demonstration
        _ = e1;
        _ = e2;
        return false; // In practice, implement proper expression comparison
    }
};

/// HIR Node types
pub const HIRProgram = struct {
    version: []const u8,
    contracts: []Contract,
    allocator: Allocator,

    pub fn init(allocator: Allocator) HIRProgram {
        return HIRProgram{
            .version = IR_VERSION,
            .contracts = &[_]Contract{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HIRProgram) void {
        for (self.contracts) |*contract| {
            contract.deinit();
        }
        self.allocator.free(self.contracts);
    }

    pub fn addContract(self: *HIRProgram, contract: Contract) !void {
        const new_contracts = try self.allocator.realloc(self.contracts, self.contracts.len + 1);
        new_contracts[new_contracts.len - 1] = contract;
        self.contracts = new_contracts;
    }
};

pub const Contract = struct {
    name: []const u8,
    storage: []StorageVariable,
    functions: []Function,
    events: []Event,
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8) Contract {
        return Contract{
            .name = name,
            .storage = &[_]StorageVariable{},
            .functions = &[_]Function{},
            .events = &[_]Event{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Contract) void {
        for (self.storage) |*storage| {
            storage.deinit();
        }
        for (self.functions) |*function| {
            function.deinit();
        }
        for (self.events) |*event| {
            event.deinit();
        }
        self.allocator.free(self.storage);
        self.allocator.free(self.functions);
        self.allocator.free(self.events);
    }
};

pub const StorageVariable = struct {
    name: []const u8,
    type: Type,
    region: Region,
    mutable: bool,
    slot: ?u32,
    value: ?*Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *StorageVariable) void {
        self.type.deinit(self.allocator);
        if (self.value) |value| {
            value.deinit();
            self.allocator.destroy(value);
        }
    }
};

/// Function effect metadata for tooling
pub const FunctionEffects = struct {
    writes_storage: bool,
    reads_storage: bool,
    writes_transient: bool,
    reads_transient: bool,
    emits_logs: bool,
    calls_other: bool,
    modifies_state: bool,
    is_pure: bool,

    pub fn init() FunctionEffects {
        return FunctionEffects{
            .writes_storage = false,
            .reads_storage = false,
            .writes_transient = false,
            .reads_transient = false,
            .emits_logs = false,
            .calls_other = false,
            .modifies_state = false,
            .is_pure = true,
        };
    }

    pub fn markStorageWrite(self: *FunctionEffects) void {
        self.writes_storage = true;
        self.modifies_state = true;
        self.is_pure = false;
    }

    pub fn markStorageRead(self: *FunctionEffects) void {
        self.reads_storage = true;
        self.is_pure = false;
    }

    pub fn markTransientWrite(self: *FunctionEffects) void {
        self.writes_transient = true;
        self.modifies_state = true;
        self.is_pure = false;
    }

    pub fn markTransientRead(self: *FunctionEffects) void {
        self.reads_transient = true;
        self.is_pure = false;
    }

    pub fn markLogEmission(self: *FunctionEffects) void {
        self.emits_logs = true;
        self.modifies_state = true;
        self.is_pure = false;
    }

    pub fn markExternalCall(self: *FunctionEffects) void {
        self.calls_other = true;
        self.is_pure = false;
    }
};

pub const Function = struct {
    name: []const u8,
    visibility: Visibility,
    parameters: []Parameter,
    return_type: ?Type,
    requires: []Expression,
    ensures: []Expression,
    body: Block,
    state_effects: EffectSet,
    observable_effects: EffectSet,
    effects: FunctionEffects,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *Function) void {
        for (self.parameters) |*param| {
            param.deinit(self.allocator);
        }
        if (self.return_type) |*ret_type| {
            ret_type.deinit(self.allocator);
        }
        for (self.requires) |*req| {
            req.deinit();
        }
        for (self.ensures) |*ens| {
            ens.deinit();
        }
        self.body.deinit();
        self.state_effects.deinit();
        self.observable_effects.deinit();

        // Free the arrays themselves
        self.allocator.free(self.parameters);
        self.allocator.free(self.requires);
        self.allocator.free(self.ensures);
    }
};

pub const Parameter = struct {
    name: []const u8,
    type: Type,
    location: SourceLocation,

    pub fn deinit(self: *Parameter, allocator: Allocator) void {
        self.type.deinit(allocator);
    }
};

pub const Event = struct {
    name: []const u8,
    fields: []EventField,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *Event) void {
        for (self.fields) |*field| {
            field.deinit(self.allocator);
        }
        self.allocator.free(self.fields);
    }
};

pub const EventField = struct {
    name: []const u8,
    type: Type,
    indexed: bool,
    location: SourceLocation,

    pub fn deinit(self: *EventField, allocator: Allocator) void {
        self.type.deinit(allocator);
    }
};

pub const Block = struct {
    statements: []Statement,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *Block) void {
        for (self.statements) |*stmt| {
            stmt.deinit();
        }
        self.allocator.free(self.statements);
    }
};

pub const Statement = union(enum) {
    variable_decl: VariableDecl,
    assignment: Assignment,
    compound_assignment: CompoundAssignment,
    if_statement: IfStatement,
    while_statement: WhileStatement,
    return_statement: ReturnStatement,
    expression_statement: ExpressionStatement,
    lock_statement: LockStatement,
    unlock_statement: UnlockStatement,

    // Error handling statements
    error_decl: ErrorDecl,
    try_statement: TryStatement,
    error_return: ErrorReturn,

    pub fn deinit(self: *Statement) void {
        switch (self.*) {
            .variable_decl => |*vd| vd.deinit(),
            .assignment => |*a| a.deinit(),
            .compound_assignment => |*ca| ca.deinit(),
            .if_statement => |*is| is.deinit(),
            .while_statement => |*ws| ws.deinit(),
            .return_statement => |*rs| rs.deinit(),
            .expression_statement => |*es| es.deinit(),
            .lock_statement => |*ls| ls.deinit(),
            .unlock_statement => |*us| us.deinit(),
            .error_decl => |*ed| ed.deinit(),
            .try_statement => |*ts| ts.deinit(),
            .error_return => |*er| er.deinit(),
        }
    }
};

pub const VariableDecl = struct {
    name: []const u8,
    type: Type,
    region: Region,
    mutable: bool,
    value: ?*Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *VariableDecl) void {
        self.type.deinit(self.allocator);
        if (self.value) |value| {
            value.deinit();
            self.allocator.destroy(value);
        }
    }
};

pub const Assignment = struct {
    target: *Expression,
    value: *Expression,
    location: SourceLocation,

    pub fn deinit(self: *Assignment) void {
        self.target.deinit();
        self.value.deinit();
    }
};

pub const CompoundAssignment = struct {
    target: *Expression,
    operator: CompoundAssignmentOp,
    value: *Expression,
    location: SourceLocation,

    pub const CompoundAssignmentOp = enum {
        plus_equal,
        minus_equal,
        star_equal,
        slash_equal,
        percent_equal,
    };

    pub fn deinit(self: *CompoundAssignment) void {
        self.target.deinit();
        self.value.deinit();
    }
};

pub const IfStatement = struct {
    condition: *Expression,
    then_branch: Block,
    else_branch: ?Block,
    location: SourceLocation,

    pub fn deinit(self: *IfStatement) void {
        self.condition.deinit();
        self.then_branch.deinit();
        if (self.else_branch) |*else_b| {
            else_b.deinit();
        }
    }
};

pub const WhileStatement = struct {
    condition: *Expression,
    body: Block,
    invariants: []Expression,
    location: SourceLocation,

    pub fn deinit(self: *WhileStatement) void {
        self.condition.deinit();
        self.body.deinit();
        for (self.invariants) |*inv| {
            inv.deinit();
        }
    }
};

pub const ReturnStatement = struct {
    value: ?*Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *ReturnStatement) void {
        if (self.value) |value| {
            value.deinit();
            self.allocator.destroy(value);
        }
    }
};

pub const ExpressionStatement = struct {
    expression: *Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *ExpressionStatement) void {
        self.expression.deinit();
        self.allocator.destroy(self.expression);
    }
};

pub const LockStatement = struct {
    path: AccessPath,
    location: SourceLocation,

    pub fn deinit(self: *LockStatement) void {
        _ = self;
    }
};

pub const UnlockStatement = struct {
    path: AccessPath,
    location: SourceLocation,

    pub fn deinit(self: *UnlockStatement) void {
        _ = self;
    }
};

/// Error declaration statement
pub const ErrorDecl = struct {
    name: []const u8,
    location: SourceLocation,

    pub fn deinit(self: *ErrorDecl) void {
        _ = self;
    }
};

/// Try-catch statement
pub const TryStatement = struct {
    try_block: Block,
    catch_block: ?CatchBlock,
    location: SourceLocation,

    pub const CatchBlock = struct {
        error_variable: ?[]const u8,
        block: Block,
        location: SourceLocation,
    };

    pub fn deinit(self: *TryStatement) void {
        self.try_block.deinit();
        if (self.catch_block) |*cb| {
            cb.block.deinit();
        }
    }
};

/// Error return statement
pub const ErrorReturn = struct {
    error_name: []const u8,
    location: SourceLocation,

    pub fn deinit(self: *ErrorReturn) void {
        _ = self;
    }
};

pub const Expression = union(enum) {
    binary: BinaryExpression,
    unary: UnaryExpression,
    call: CallExpression,
    index: IndexExpression,
    field: FieldExpression,
    transfer: TransferExpression,
    shift: ShiftExpression,
    old: OldExpression,
    literal: Literal,
    identifier: Identifier,

    // Error handling expressions
    try_expr: TryExpression,
    error_value: ErrorValue,
    error_cast: ErrorCast,

    // Struct instantiation
    struct_instantiation: StructInstantiationExpression,

    pub fn deinit(self: *Expression) void {
        switch (self.*) {
            .binary => |*be| be.deinit(),
            .unary => |*ue| ue.deinit(),
            .call => |*ce| ce.deinit(),
            .index => |*ie| ie.deinit(),
            .field => |*fe| fe.deinit(),
            .transfer => |*te| te.deinit(),
            .shift => |*se| se.deinit(),
            .old => |*oe| oe.deinit(),
            .literal => |*l| l.deinit(),
            .identifier => |*i| i.deinit(),
            .try_expr => |*te| te.deinit(),
            .error_value => |*ev| ev.deinit(),
            .error_cast => |*ec| ec.deinit(),
            .struct_instantiation => |*si| si.deinit(),
        }
    }

    pub fn getLocation(self: *const Expression) SourceLocation {
        return switch (self.*) {
            .binary => |*be| be.location,
            .unary => |*ue| ue.location,
            .call => |*ce| ce.location,
            .index => |*ie| ie.location,
            .field => |*fe| fe.location,
            .transfer => |*te| te.location,
            .shift => |*se| se.location,
            .old => |*oe| oe.location,
            .literal => SourceLocation{ .line = 0, .column = 0, .length = 0 }, // Literals don't have location
            .identifier => |*i| i.location,
            .try_expr => |*te| te.location,
            .error_value => |*ev| ev.location,
            .error_cast => |*ec| ec.location,
            .struct_instantiation => |*si| si.location,
        };
    }
};

pub const BinaryExpression = struct {
    left: *Expression,
    operator: BinaryOp,
    right: *Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub const BinaryOp = enum {
        plus,
        minus,
        star,
        slash,
        percent,
        equal_equal,
        bang_equal,
        less,
        less_equal,
        greater,
        greater_equal,
        and_,
        or_,
        bit_and,
        bit_or,
        bit_xor,
        shift_left,
        shift_right,
    };

    pub fn deinit(self: *BinaryExpression) void {
        self.left.deinit();
        self.allocator.destroy(self.left);
        self.right.deinit();
        self.allocator.destroy(self.right);
    }
};

pub const UnaryExpression = struct {
    operator: UnaryOp,
    operand: *Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub const UnaryOp = enum {
        minus,
        bang,
        bit_not,
    };

    pub fn deinit(self: *UnaryExpression) void {
        self.operand.deinit();
        self.allocator.destroy(self.operand);
    }
};

pub const CallExpression = struct {
    callee: *Expression,
    arguments: []Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *CallExpression) void {
        self.callee.deinit();
        self.allocator.destroy(self.callee);
        for (self.arguments) |*arg| {
            arg.deinit();
        }
        self.allocator.free(self.arguments);
    }
};

pub const IndexExpression = struct {
    target: *Expression,
    index: *Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *IndexExpression) void {
        self.target.deinit();
        self.allocator.destroy(self.target);
        self.index.deinit();
        self.allocator.destroy(self.index);
    }
};

pub const FieldExpression = struct {
    target: *Expression,
    field: []const u8,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *FieldExpression) void {
        self.target.deinit();
        self.allocator.destroy(self.target);
    }
};

pub const TransferExpression = struct {
    from: *Expression,
    to: *Expression,
    amount: *Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *TransferExpression) void {
        self.from.deinit();
        self.allocator.destroy(self.from);
        self.to.deinit();
        self.allocator.destroy(self.to);
        self.amount.deinit();
        self.allocator.destroy(self.amount);
    }
};

pub const ShiftExpression = struct {
    mapping: *Expression,
    source: *Expression,
    dest: *Expression,
    amount: *Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *ShiftExpression) void {
        self.mapping.deinit();
        self.allocator.destroy(self.mapping);
        self.source.deinit();
        self.allocator.destroy(self.source);
        self.dest.deinit();
        self.allocator.destroy(self.dest);
        self.amount.deinit();
        self.allocator.destroy(self.amount);
    }
};

pub const OldExpression = struct {
    expression: *Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *OldExpression) void {
        self.expression.deinit();
        self.allocator.destroy(self.expression);
    }
};

pub const Literal = union(enum) {
    integer: []const u8,
    string: []const u8,
    boolean: bool,
    address: []const u8,

    pub fn deinit(self: *Literal) void {
        _ = self;
    }
};

pub const Identifier = struct {
    name: []const u8,
    location: SourceLocation,

    pub fn deinit(self: *Identifier) void {
        _ = self;
    }
};

/// Try expression
pub const TryExpression = struct {
    expression: *Expression,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *TryExpression) void {
        self.expression.deinit();
        self.allocator.destroy(self.expression);
    }
};

/// Error value expression
pub const ErrorValue = struct {
    error_name: []const u8,
    location: SourceLocation,

    pub fn deinit(self: *ErrorValue) void {
        _ = self;
    }
};

/// Error cast expression
pub const ErrorCast = struct {
    operand: *Expression,
    target_type: Type,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *ErrorCast) void {
        self.operand.deinit();
        self.allocator.destroy(self.operand);
        self.target_type.deinit(self.allocator);
    }
};

/// Struct instantiation expression
pub const StructInstantiationExpression = struct {
    struct_type: Type, // Should be a struct_type variant
    field_values: []StructFieldValue,
    location: SourceLocation,
    allocator: Allocator,

    pub fn deinit(self: *StructInstantiationExpression) void {
        for (self.field_values) |*field_value| {
            field_value.deinit();
        }
        self.allocator.free(self.field_values);
        self.struct_type.deinit(self.allocator);
    }
};

/// Struct field value for instantiation
pub const StructFieldValue = struct {
    field_name: []const u8,
    value: *Expression,
    allocator: Allocator,

    pub fn deinit(self: *StructFieldValue) void {
        self.value.deinit();
        self.allocator.destroy(self.value);
    }
};

/// Validation context for IR well-formedness checking
pub const ValidationContext = struct {
    allocator: Allocator,
    errors: ArrayList(ValidationError),
    warnings: ArrayList(ValidationError),

    pub fn init(allocator: Allocator) ValidationContext {
        return ValidationContext{
            .allocator = allocator,
            .errors = ArrayList(ValidationError).init(allocator),
            .warnings = ArrayList(ValidationError).init(allocator),
        };
    }

    pub fn deinit(self: *ValidationContext) void {
        // Free allocated error messages only
        for (self.errors.items) |error_| {
            if (error_.allocated) {
                self.allocator.free(error_.message);
            }
        }
        for (self.warnings.items) |warning| {
            if (warning.allocated) {
                self.allocator.free(warning.message);
            }
        }
        self.errors.deinit();
        self.warnings.deinit();
    }

    pub fn addError(self: *ValidationContext, error_: ValidationError) !void {
        try self.errors.append(error_);
    }

    pub fn addWarning(self: *ValidationContext, warning: ValidationError) !void {
        try self.warnings.append(warning);
    }

    pub fn hasErrors(self: *const ValidationContext) bool {
        return self.errors.items.len > 0;
    }
};

pub const ValidationError = struct {
    message: []const u8,
    location: SourceLocation,
    kind: ValidationErrorKind,
    allocated: bool, // Track if message was allocated and needs freeing

    pub const ValidationErrorKind = enum {
        type_error,
        region_error,
        effect_error,
        lock_error,
        memory_safety_error,
        symbol_error,
        mutability_error,
        contract_error,
        control_flow_error,
    };

    /// Create ValidationError with allocated message
    pub fn withAllocatedMessage(message: []const u8, location: SourceLocation, kind: ValidationErrorKind) ValidationError {
        return ValidationError{
            .message = message,
            .location = location,
            .kind = kind,
            .allocated = true,
        };
    }

    /// Create ValidationError with static message
    pub fn withStaticMessage(message: []const u8, location: SourceLocation, kind: ValidationErrorKind) ValidationError {
        return ValidationError{
            .message = message,
            .location = location,
            .kind = kind,
            .allocated = false,
        };
    }
};

/// Symbol table for tracking variables and functions in scope
const SymbolTable = struct {
    allocator: Allocator,
    scopes: ArrayList(Scope),

    const Scope = struct {
        symbols: std.HashMap([]const u8, Symbol, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

        pub fn init(allocator: Allocator) Scope {
            return Scope{
                .symbols = std.HashMap([]const u8, Symbol, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            };
        }

        pub fn deinit(self: *Scope) void {
            self.symbols.deinit();
        }
    };

    const Symbol = struct {
        name: []const u8,
        type: Type,
        region: Region,
        mutable: bool,
        initialized: bool,
        kind: SymbolKind,
        location: SourceLocation,

        const SymbolKind = enum {
            variable,
            parameter,
            function,
            storage_variable,
        };
    };

    pub fn init(allocator: Allocator) SymbolTable {
        return SymbolTable{
            .allocator = allocator,
            .scopes = ArrayList(Scope).init(allocator),
        };
    }

    pub fn deinit(self: *SymbolTable) void {
        for (self.scopes.items) |*scope| {
            scope.deinit();
        }
        self.scopes.deinit();
    }

    pub fn pushScope(self: *SymbolTable) !void {
        try self.scopes.append(Scope.init(self.allocator));
    }

    pub fn popScope(self: *SymbolTable) void {
        if (self.scopes.items.len > 0) {
            if (self.scopes.pop()) |scope| {
                var scoped = scope;
                scoped.deinit();
            }
        }
    }

    pub fn addSymbol(self: *SymbolTable, symbol: Symbol) !void {
        if (self.scopes.items.len == 0) {
            try self.pushScope();
        }

        const current_scope = &self.scopes.items[self.scopes.items.len - 1];

        // Check for duplicate in current scope
        if (current_scope.symbols.contains(symbol.name)) {
            return error.DuplicateSymbol;
        }

        try current_scope.symbols.put(symbol.name, symbol);
    }

    pub fn lookupSymbol(self: *const SymbolTable, name: []const u8) ?Symbol {
        // Search scopes from innermost to outermost
        var i: usize = self.scopes.items.len;
        while (i > 0) {
            i -= 1;
            const scope = &self.scopes.items[i];
            if (scope.symbols.get(name)) |symbol| {
                return symbol;
            }
        }
        return null;
    }

    pub fn markInitialized(self: *SymbolTable, name: []const u8) !void {
        // Find and mark symbol as initialized
        var i: usize = self.scopes.items.len;
        while (i > 0) {
            i -= 1;
            const scope = &self.scopes.items[i];
            if (scope.symbols.getPtr(name)) |symbol| {
                symbol.initialized = true;
                return;
            }
        }
        return error.SymbolNotFound;
    }
};

/// Validation functions implementing the formal rules
pub const Validator = struct {
    context: ValidationContext,
    symbol_table: SymbolTable,
    current_function: ?*const Function,

    pub fn init(allocator: Allocator) Validator {
        return Validator{
            .context = ValidationContext.init(allocator),
            .symbol_table = SymbolTable.init(allocator),
            .current_function = null,
        };
    }

    pub fn deinit(self: *Validator) void {
        self.context.deinit();
        self.symbol_table.deinit();
    }

    /// Validate a complete HIR program
    pub fn validateProgram(self: *Validator, program: *const HIRProgram) !ValidationResult {
        for (program.contracts) |*contract| {
            try self.validateContract(contract);
        }

        return ValidationResult{
            .valid = !self.context.hasErrors(),
            .errors = self.context.errors.items,
            .warnings = self.context.warnings.items,
        };
    }

    /// Validate a contract
    fn validateContract(self: *Validator, contract: *const Contract) anyerror!void {
        // Push contract scope
        try self.symbol_table.pushScope();
        defer self.symbol_table.popScope();

        // Contract-level validation
        try self.validateContractStructure(contract);

        // Add storage variables to symbol table
        for (contract.storage) |*storage| {
            try self.validateStorageVariable(storage);

            // Add to symbol table
            self.symbol_table.addSymbol(SymbolTable.Symbol{
                .name = storage.name,
                .type = storage.type,
                .region = storage.region,
                .mutable = storage.mutable,
                .initialized = storage.value != null,
                .kind = .storage_variable,
                .location = storage.location,
            }) catch |err| {
                if (err == error.DuplicateSymbol) {
                    try self.context.addError(ValidationError.withStaticMessage("Duplicate storage variable name", storage.location, .symbol_error));
                } else {
                    return err;
                }
            };
        }

        // Add function signatures to symbol table
        for (contract.functions) |*function| {
            self.symbol_table.addSymbol(SymbolTable.Symbol{
                .name = function.name,
                .type = Type{ .primitive = .bool }, // Placeholder - functions need proper type system
                .region = .stack,
                .mutable = false,
                .initialized = true,
                .kind = .function,
                .location = function.location,
            }) catch |err| {
                if (err == error.DuplicateSymbol) {
                    try self.context.addError(ValidationError.withStaticMessage("Duplicate function name", function.location, .symbol_error));
                } else {
                    return err;
                }
            };
        }

        // Validate functions with symbol table populated
        for (contract.functions) |*function| {
            try self.validateFunction(function);
        }

        // Validate events
        for (contract.events) |*event| {
            try self.validateEvent(event);
        }
    }

    /// Validate contract structure and required elements
    fn validateContractStructure(self: *Validator, contract: *const Contract) anyerror!void {
        // Check for required init function
        var has_init = false;
        for (contract.functions) |*function| {
            if (std.mem.eql(u8, function.name, "init")) {
                has_init = true;
                break;
            }
        }

        if (!has_init) {
            try self.context.addError(ValidationError.withStaticMessage("Contract must have an 'init' function", SourceLocation{ .line = 1, .column = 1, .length = 1 }, // Contract doesn't have location
                .contract_error));
        }

        // Check for duplicate event names
        for (contract.events, 0..) |*event1, i| {
            for (contract.events[(i + 1)..]) |*event2| {
                if (std.mem.eql(u8, event1.name, event2.name)) {
                    try self.context.addError(ValidationError.withStaticMessage("Duplicate event name", event2.location, .contract_error));
                }
            }
        }
    }

    /// Validate storage variable (Region consistency rule)
    fn validateStorageVariable(self: *Validator, storage: *const StorageVariable) anyerror!void {
        switch (storage.type) {
            .mapping => {
                if (storage.region != .storage) {
                    try self.context.addError(ValidationError.withStaticMessage("Mapping types can only be used in storage region", storage.location, .region_error));
                }
            },
            else => {
                // Non-mapping types can be in any region
            },
        }
    }

    /// Validate function
    fn validateFunction(self: *Validator, function: *const Function) anyerror!void {
        // Set current function for control flow validation
        self.current_function = function;
        defer self.current_function = null;

        // Push function scope
        try self.symbol_table.pushScope();
        defer self.symbol_table.popScope();

        // Add function parameters to symbol table
        for (function.parameters) |*param| {
            try self.symbol_table.addSymbol(SymbolTable.Symbol{
                .name = param.name,
                .type = param.type,
                .region = .stack, // Parameters are typically on stack
                .mutable = true, // Parameters are mutable within function
                .initialized = true, // Parameters are initialized when function is called
                .kind = .parameter,
                .location = param.location,
            });
        }

        // Validate function signature
        try self.validateFunctionSignature(function);

        // Validate effects don't conflict
        if (function.state_effects.hasConflict() or function.observable_effects.hasConflict()) {
            try self.context.addError(ValidationError.withStaticMessage("Function has conflicting effects", function.location, .effect_error));
        }

        // Validate function body
        try self.validateBlock(&function.body);

        // Check for missing return statement if function has return type
        if (function.return_type != null) {
            if (!self.blockHasReturn(&function.body)) {
                try self.context.addError(ValidationError.withStaticMessage("Function with return type must have return statement", function.location, .control_flow_error));
            }
        }
    }

    /// Validate function signature
    fn validateFunctionSignature(self: *Validator, function: *const Function) anyerror!void {
        // Check for duplicate parameter names
        for (function.parameters, 0..) |*param1, i| {
            for (function.parameters[(i + 1)..]) |*param2| {
                if (std.mem.eql(u8, param1.name, param2.name)) {
                    try self.context.addError(ValidationError.withStaticMessage("Duplicate parameter name", param2.location, .symbol_error));
                }
            }
        }

        // Validate requires clauses
        for (function.requires) |*req| {
            try self.validateExpression(req);
        }

        // Validate ensures clauses
        for (function.ensures) |*ens| {
            try self.validateExpression(ens);
        }
    }

    /// Check if block has a return statement
    fn blockHasReturn(self: *Validator, block: *const Block) bool {
        for (block.statements) |*stmt| {
            switch (stmt.*) {
                .return_statement => return true,
                .if_statement => |*if_stmt| {
                    if (if_stmt.else_branch != null) {
                        // Both branches must have return
                        if (self.blockHasReturn(&if_stmt.then_branch) and
                            self.blockHasReturn(&if_stmt.else_branch.?))
                        {
                            return true;
                        }
                    }
                },
                else => {},
            }
        }
        return false;
    }

    /// Validate block
    fn validateBlock(self: *Validator, block: *const Block) anyerror!void {
        // Push new scope for block
        try self.symbol_table.pushScope();
        defer self.symbol_table.popScope();

        for (block.statements) |*stmt| {
            try self.validateStatement(stmt);
        }
    }

    /// Validate statement
    fn validateStatement(self: *Validator, stmt: *const Statement) anyerror!void {
        switch (stmt.*) {
            .variable_decl => |*vd| try self.validateVariableDecl(vd),
            .assignment => |*a| try self.validateAssignment(a),
            .compound_assignment => |*ca| try self.validateCompoundAssignment(ca),
            .if_statement => |*is| try self.validateIfStatement(is),
            .while_statement => |*ws| try self.validateWhileStatement(ws),
            .return_statement => |*rs| try self.validateReturnStatement(rs),
            .expression_statement => |*es| try self.validateExpression(es.expression),
            .lock_statement => |*ls| try self.validateLockStatement(ls),
            .unlock_statement => |*us| try self.validateUnlockStatement(us),
            .error_decl => |*ed| try self.validateErrorDecl(ed),
            .try_statement => |*ts| try self.validateTryStatement(ts),
            .error_return => |*er| try self.validateErrorReturn(er),
        }
    }

    /// Validate compound assignment
    fn validateCompoundAssignment(self: *Validator, comp_assign: *const CompoundAssignment) anyerror!void {
        try self.validateExpression(comp_assign.target);
        try self.validateExpression(comp_assign.value);

        // Check if target is mutable
        if (!self.isExpressionMutable(comp_assign.target)) {
            try self.context.addError(ValidationError.withStaticMessage("Cannot assign to immutable expression", comp_assign.target.getLocation(), .mutability_error));
        }

        // Check type compatibility for compound assignment
        const target_type = self.getExpressionType(comp_assign.target);
        const value_type = self.getExpressionType(comp_assign.value);

        if (target_type != null and value_type != null) {
            // For compound assignments, both operands should be numeric for arithmetic operations
            switch (comp_assign.operator) {
                .plus_equal, .minus_equal, .star_equal, .slash_equal, .percent_equal => {
                    if (!self.isNumericType(target_type.?) or !self.isNumericType(value_type.?)) {
                        try self.context.addError(ValidationError.withStaticMessage("Arithmetic compound assignment requires numeric types", comp_assign.target.getLocation(), .type_error));
                    }
                },
            }
        }
    }

    /// Validate return statement
    fn validateReturnStatement(self: *Validator, ret_stmt: *const ReturnStatement) anyerror!void {
        // Check if we're in a function that expects a return value
        if (self.current_function) |func| {
            const has_return_type = func.return_type != null;
            const has_return_value = ret_stmt.value != null;

            if (has_return_type and !has_return_value) {
                try self.context.addError(ValidationError.withStaticMessage("Function with return type must return a value", ret_stmt.location, .type_error));
            } else if (!has_return_type and has_return_value) {
                try self.context.addError(ValidationError.withStaticMessage("Function without return type cannot return a value", ret_stmt.location, .type_error));
            }

            // Validate return expression if present
            if (ret_stmt.value) |value| {
                try self.validateExpression(value);

                // Check type compatibility
                if (func.return_type) |expected_type| {
                    const return_type = self.getExpressionType(value);
                    if (return_type != null) {
                        if (!expected_type.isCompatibleWith(&return_type.?)) {
                            try self.context.addError(ValidationError.withStaticMessage("Return type mismatch", ret_stmt.location, .type_error));
                        }
                    }
                }
            }
        }
    }

    /// Validate lock statement
    fn validateLockStatement(self: *Validator, lock_stmt: *const LockStatement) anyerror!void {
        _ = self;
        _ = lock_stmt;
        // Lock statement validation - check that the path is valid
        // TODO: Implement proper lock path validation
    }

    /// Validate unlock statement
    fn validateUnlockStatement(self: *Validator, unlock_stmt: *const UnlockStatement) anyerror!void {
        _ = self;
        _ = unlock_stmt;
        // Unlock statement validation - check that the path is valid
        // TODO: Implement proper unlock path validation
    }

    /// Validate variable declaration
    fn validateVariableDecl(self: *Validator, decl: *const VariableDecl) anyerror!void {
        // Check region consistency
        switch (decl.type) {
            .mapping => {
                if (decl.region != .storage) {
                    try self.context.addError(ValidationError.withStaticMessage("Mapping types can only be declared in storage region", decl.location, .region_error));
                }
            },
            else => {},
        }

        // If there's an initializer, validate it
        if (decl.value) |value| {
            try self.validateExpression(value);

            // Check type compatibility
            const decl_type = decl.type;
            const value_type = self.getExpressionType(value);

            if (value_type != null) {
                if (!decl_type.isCompatibleWith(&value_type.?)) {
                    try self.context.addError(ValidationError.withStaticMessage("Type mismatch in variable initialization", decl.location, .type_error));
                }
            }
        }

        // Add variable to symbol table
        self.symbol_table.addSymbol(SymbolTable.Symbol{
            .name = decl.name,
            .type = decl.type,
            .region = decl.region,
            .mutable = decl.mutable,
            .initialized = decl.value != null,
            .kind = .variable,
            .location = decl.location,
        }) catch |err| {
            if (err == error.DuplicateSymbol) {
                try self.context.addError(ValidationError.withStaticMessage("Variable already declared in this scope", decl.location, .symbol_error));
            } else {
                return err;
            }
        };
    }

    /// Validate assignment
    fn validateAssignment(self: *Validator, assignment: *const Assignment) anyerror!void {
        try self.validateExpression(assignment.target);
        try self.validateExpression(assignment.value);

        // Check if target is mutable
        if (!self.isExpressionMutable(assignment.target)) {
            try self.context.addError(ValidationError.withStaticMessage("Cannot assign to immutable expression", assignment.target.getLocation(), .mutability_error));
        }

        // Check type compatibility
        const target_type = self.getExpressionType(assignment.target);
        const value_type = self.getExpressionType(assignment.value);

        if (target_type != null and value_type != null) {
            if (!target_type.?.isCompatibleWith(&value_type.?)) {
                try self.context.addError(ValidationError.withStaticMessage("Type mismatch in assignment", assignment.target.getLocation(), .type_error));
            }
        }
    }

    /// Check if an expression is mutable (can be assigned to)
    fn isExpressionMutable(self: *const Validator, expr: *const Expression) bool {
        switch (expr.*) {
            .identifier => |*ident| {
                if (self.symbol_table.lookupSymbol(ident.name)) |symbol| {
                    return symbol.mutable;
                }
                return false;
            },
            .field => |*field| {
                // Field access is mutable if the target is mutable
                return self.isExpressionMutable(field.target);
            },
            .index => |*index| {
                // Index access is mutable if the target is mutable
                return self.isExpressionMutable(index.target);
            },
            else => {
                // Literals and other expressions are not mutable
                return false;
            },
        }
    }

    /// Get the type of an expression with enhanced inference
    fn getExpressionType(self: *const Validator, expr: *const Expression) ?Type {
        switch (expr.*) {
            .identifier => |*ident| {
                if (self.symbol_table.lookupSymbol(ident.name)) |symbol| {
                    return symbol.type;
                }
                return null;
            },
            .literal => |*lit| {
                return self.getLiteralType(lit);
            },
            .binary => |*binary| {
                return self.inferBinaryExpressionType(binary);
            },
            .unary => |*unary| {
                return self.inferUnaryExpressionType(unary);
            },
            .call => |*call| {
                return self.inferCallExpressionType(call);
            },
            .field => |*field| {
                return self.inferFieldExpressionType(field);
            },
            .index => |*index| {
                return self.inferIndexExpressionType(index);
            },
            .transfer => |*transfer| {
                _ = transfer;
                return Type{ .primitive = .bool }; // Transfer operations return success/failure
            },
            .shift => |*shift| {
                _ = shift;
                return Type{ .primitive = .bool }; // Shift operations return success/failure
            },
            .old => |*old| {
                return self.getExpressionType(old.expression);
            },
            .try_expr => |*try_expr| {
                // Try expression unwraps error union
                const expr_type = self.getExpressionType(try_expr.expression);
                if (expr_type) |typ| {
                    return switch (typ) {
                        .error_union => |eu| eu.success_type.*,
                        else => typ,
                    };
                }
                return null;
            },
            .error_value => |*error_val| {
                _ = error_val;
                return Type{ .primitive = .u8 }; // Error codes are u8
            },
            .error_cast => |*error_cast| {
                return error_cast.target_type;
            },
            .struct_instantiation => |*struct_inst| {
                return struct_inst.struct_type;
            },
        }
    }

    /// Infer the type of a binary expression
    fn inferBinaryExpressionType(self: *const Validator, binary: *const BinaryExpression) ?Type {
        const left_type = self.getExpressionType(binary.left);
        const right_type = self.getExpressionType(binary.right);

        if (left_type == null or right_type == null) return null;

        switch (binary.operator) {
            // Arithmetic operations return the common type of operands
            .plus, .minus, .star, .slash, .percent => {
                return left_type.?.getCommonType(&right_type.?, self.context.allocator);
            },
            // Comparison operations return boolean
            .equal_equal, .bang_equal, .less, .less_equal, .greater, .greater_equal => {
                return Type{ .primitive = .bool };
            },
            // Logical operations return boolean
            .and_, .or_ => {
                return Type{ .primitive = .bool };
            },
            // Bitwise operations return the common type of operands
            .bit_and, .bit_or, .bit_xor, .shift_left, .shift_right => {
                return left_type.?.getCommonType(&right_type.?, self.context.allocator);
            },
        }
    }

    /// Infer the type of a unary expression
    fn inferUnaryExpressionType(self: *const Validator, unary: *const UnaryExpression) ?Type {
        const operand_type = self.getExpressionType(unary.operand);
        if (operand_type == null) return null;

        return switch (unary.operator) {
            .minus => operand_type, // Negation preserves numeric type
            .bang => Type{ .primitive = .bool }, // Logical not returns boolean
            .bit_not => operand_type, // Bitwise not preserves type
        };
    }

    /// Infer the type of a call expression
    fn inferCallExpressionType(self: *const Validator, call: *const CallExpression) ?Type {
        // For now, try to look up function return type
        if (call.callee.* == .identifier) {
            const func_name = call.callee.identifier.name;

            // Check for built-in functions
            if (std.mem.eql(u8, func_name, "require") or std.mem.eql(u8, func_name, "assert")) {
                return Type{ .primitive = .bool };
            }

            if (std.mem.eql(u8, func_name, "hash")) {
                return Type{ .primitive = .u256 };
            }

            // Look up user-defined function
            if (self.symbol_table.lookupSymbol(func_name)) |symbol| {
                _ = symbol; // Mark as used - TODO: Get actual return type from function signature
                return Type{ .primitive = .u256 }; // Placeholder
            }
        }

        return null;
    }

    /// Infer the type of a field expression
    fn inferFieldExpressionType(self: *const Validator, field: *const FieldExpression) ?Type {
        const target_type = self.getExpressionType(field.target);
        if (target_type == null) return null;

        // Handle standard library access
        if (target_type.? == .custom) {
            if (std.mem.eql(u8, target_type.?.custom.name, "std")) {
                // std library module access
                if (std.mem.eql(u8, field.field, "require") or std.mem.eql(u8, field.field, "assert")) {
                    return Type{ .primitive = .bool };
                }
                if (std.mem.eql(u8, field.field, "hash")) {
                    return Type{ .primitive = .u256 };
                }
            }
        }

        // TODO: Implement struct field type lookup
        return null;
    }

    /// Infer the type of an index expression
    fn inferIndexExpressionType(self: *const Validator, index: *const IndexExpression) ?Type {
        const target_type = self.getExpressionType(index.target);
        if (target_type == null) return null;

        return switch (target_type.?) {
            .slice => |slice| slice.element_type.*,
            .mapping => |mapping| mapping.value_type.*,
            else => null,
        };
    }

    /// Get the type of a literal
    fn getLiteralType(self: *const Validator, lit: *const Literal) Type {
        _ = self;
        return switch (lit.*) {
            .integer => Type.inferLiteralType(lit), // Use enhanced literal type inference
            .string => Type{ .primitive = .string },
            .boolean => Type{ .primitive = .bool },
            .address => Type{ .primitive = .address },
        };
    }

    /// Validate if statement
    fn validateIfStatement(self: *Validator, if_stmt: *const IfStatement) anyerror!void {
        try self.validateExpression(if_stmt.condition);
        try self.validateBlock(&if_stmt.then_branch);

        if (if_stmt.else_branch) |*else_branch| {
            try self.validateBlock(else_branch);
        }
    }

    /// Validate while statement
    fn validateWhileStatement(self: *Validator, while_stmt: *const WhileStatement) anyerror!void {
        try self.validateExpression(while_stmt.condition);
        try self.validateBlock(&while_stmt.body);

        // Validate invariants
        for (while_stmt.invariants) |*inv| {
            try self.validateExpression(inv);
        }
    }

    /// Validate error declaration
    fn validateErrorDecl(self: *Validator, error_decl: *const ErrorDecl) anyerror!void {
        // Check if error name is already declared
        if (self.symbol_table.lookupSymbol(error_decl.name) != null) {
            const error_msg = try std.fmt.allocPrint(self.context.allocator, "Error '{s}' already declared", .{error_decl.name});
            try self.context.addError(ValidationError.withAllocatedMessage(error_msg, error_decl.location, .symbol_error));
        }

        // Add error to symbol table
        try self.symbol_table.addSymbol(SymbolTable.Symbol{
            .name = error_decl.name,
            .type = Type{ .primitive = .u8 }, // Error codes are u8
            .region = .const_,
            .mutable = false,
            .initialized = true,
            .kind = .variable,
            .location = error_decl.location,
        });
    }

    /// Validate try statement
    fn validateTryStatement(self: *Validator, try_stmt: *const TryStatement) anyerror!void {
        try self.validateBlock(&try_stmt.try_block);

        if (try_stmt.catch_block) |*catch_block| {
            // If there's an error variable, add it to the symbol table in the catch block scope
            if (catch_block.error_variable) |error_var| {
                try self.symbol_table.addSymbol(SymbolTable.Symbol{
                    .name = error_var,
                    .type = Type{ .primitive = .u8 }, // Error codes are u8
                    .region = .stack,
                    .mutable = false,
                    .initialized = true,
                    .kind = .variable,
                    .location = catch_block.location,
                });
            }

            try self.validateBlock(&catch_block.block);
        }
    }

    /// Validate error return
    fn validateErrorReturn(self: *Validator, error_return: *const ErrorReturn) anyerror!void {
        // Check if error name is declared
        if (self.symbol_table.lookupSymbol(error_return.error_name) == null) {
            const error_msg = try std.fmt.allocPrint(self.context.allocator, "Undefined error '{s}'", .{error_return.error_name});
            try self.context.addError(ValidationError.withAllocatedMessage(error_msg, error_return.location, .symbol_error));
        }
    }

    /// Validate expression
    fn validateExpression(self: *Validator, expr: *const Expression) anyerror!void {
        switch (expr.*) {
            .identifier => |*ident| {
                // Check if identifier exists in symbol table
                if (self.symbol_table.lookupSymbol(ident.name) == null) {
                    // Create a more helpful error message
                    const error_msg = try std.fmt.allocPrint(self.context.allocator, "Undefined identifier '{s}'", .{ident.name});
                    try self.context.addError(ValidationError.withAllocatedMessage(error_msg, ident.location, .symbol_error));
                }
            },
            .binary => |*be| {
                try self.validateExpression(be.left);
                try self.validateExpression(be.right);

                // Check type compatibility for binary operations
                const left_type = self.getExpressionType(be.left);
                const right_type = self.getExpressionType(be.right);

                if (left_type != null and right_type != null) {
                    if (!self.areTypesCompatibleForBinaryOp(left_type.?, right_type.?, be.operator)) {
                        try self.context.addError(ValidationError.withStaticMessage("Type mismatch in binary operation", be.location, .type_error));
                    }
                }
            },
            .unary => |*ue| {
                try self.validateExpression(ue.operand);
            },
            .call => |*ce| {
                try self.validateExpression(ce.callee);

                // Validate function call
                try self.validateFunctionCall(ce);

                for (ce.arguments) |*arg| {
                    try self.validateExpression(arg);
                }
            },
            .index => |*ie| {
                try self.validateExpression(ie.target);
                try self.validateExpression(ie.index);

                // Check that target is indexable
                const target_type = self.getExpressionType(ie.target);
                if (target_type != null) {
                    if (!self.isTypeIndexable(target_type.?)) {
                        try self.context.addError(ValidationError.withStaticMessage("Cannot index non-indexable type", ie.target.getLocation(), .type_error));
                    }
                }
            },
            .field => |*fe| {
                try self.validateExpression(fe.target);

                // Check that target has the field
                const target_type = self.getExpressionType(fe.target);
                if (target_type != null) {
                    if (!self.typeHasField(target_type.?, fe.field)) {
                        try self.context.addError(ValidationError.withStaticMessage("Type does not have field", fe.location, .type_error));
                    }
                }
            },
            .transfer => |*te| {
                try self.validateExpression(te.to);
                try self.validateExpression(te.amount);

                // Check types
                const to_type = self.getExpressionType(te.to);
                const amount_type = self.getExpressionType(te.amount);

                if (to_type != null and !self.isAddressType(to_type.?)) {
                    try self.context.addError(ValidationError.withStaticMessage("Transfer 'to' must be address type", te.to.getLocation(), .type_error));
                }

                if (amount_type != null and !self.isNumericType(amount_type.?)) {
                    try self.context.addError(ValidationError.withStaticMessage("Transfer amount must be numeric type", te.amount.getLocation(), .type_error));
                }
            },
            .shift => |*se| {
                try self.validateExpression(se.mapping);
                try self.validateExpression(se.source);
                try self.validateExpression(se.dest);
                try self.validateExpression(se.amount);

                // Check types
                const source_type = self.getExpressionType(se.source);
                const dest_type = self.getExpressionType(se.dest);
                const amount_type = self.getExpressionType(se.amount);

                if (source_type != null and !self.isAddressType(source_type.?)) {
                    try self.context.addError(ValidationError.withStaticMessage("Shift source must be address type", se.source.getLocation(), .type_error));
                }

                if (dest_type != null and !self.isAddressType(dest_type.?)) {
                    try self.context.addError(ValidationError.withStaticMessage("Shift destination must be address type", se.dest.getLocation(), .type_error));
                }

                if (amount_type != null and !self.isNumericType(amount_type.?)) {
                    try self.context.addError(ValidationError.withStaticMessage("Shift amount must be numeric type", se.amount.getLocation(), .type_error));
                }
            },
            .old => |*oe| {
                try self.validateExpression(oe.expression);
            },
            .literal => {
                // Literals are always valid
            },
            .try_expr => |*te| {
                try self.validateExpression(te.expression);
            },
            .error_value => |*ev| {
                // Check if error name is declared
                if (self.symbol_table.lookupSymbol(ev.error_name) == null) {
                    const error_msg = try std.fmt.allocPrint(self.context.allocator, "Undefined error '{s}'", .{ev.error_name});
                    try self.context.addError(ValidationError.withAllocatedMessage(error_msg, ev.location, .symbol_error));
                }
            },
            .error_cast => |*ec| {
                try self.validateExpression(ec.operand);
                // TODO: Validate that the target type is an error union type
            },
            .struct_instantiation => |*si| {
                // Validate all field values
                for (si.field_values) |*field_value| {
                    try self.validateExpression(field_value.value);
                }
                // TODO: Validate that all required fields are provided and field types match
            },
        }
    }

    /// Check if two types are compatible for a binary operation
    fn areTypesCompatibleForBinaryOp(self: *const Validator, left: Type, right: Type, op: BinaryExpression.BinaryOp) bool {
        // Arithmetic operations require numeric types
        switch (op) {
            .plus, .minus, .star, .slash, .percent => {
                return self.isNumericType(left) and self.isNumericType(right);
            },
            .equal_equal, .bang_equal => {
                return left.isCompatibleWith(&right);
            },
            .less, .less_equal, .greater, .greater_equal => {
                return self.isNumericType(left) and self.isNumericType(right);
            },
            .and_, .or_ => {
                return self.isBooleanType(left) and self.isBooleanType(right);
            },
            .bit_and, .bit_or, .bit_xor, .shift_left, .shift_right => {
                return self.isNumericType(left) and self.isNumericType(right);
            },
        }
    }

    /// Check if a type is numeric
    fn isNumericType(self: *const Validator, typ: Type) bool {
        _ = self;
        return switch (typ) {
            .primitive => |p| switch (p) {
                .u8, .u16, .u32, .u64, .u128, .u256 => true,
                else => false,
            },
            else => false,
        };
    }

    /// Check if a type is boolean
    fn isBooleanType(self: *const Validator, typ: Type) bool {
        _ = self;
        return switch (typ) {
            .primitive => |p| p == .bool,
            else => false,
        };
    }

    /// Check if a type is address
    fn isAddressType(self: *const Validator, typ: Type) bool {
        _ = self;
        return switch (typ) {
            .primitive => |p| p == .address,
            else => false,
        };
    }

    /// Check if a type is indexable (array, mapping, etc.)
    fn isTypeIndexable(self: *const Validator, typ: Type) bool {
        _ = self;
        return switch (typ) {
            .slice, .mapping => true,
            else => false,
        };
    }

    /// Check if a type has a specific field
    fn typeHasField(self: *const Validator, typ: Type, field: []const u8) bool {
        _ = self;
        return typ.hasField(field);
    }

    /// Validate function call
    fn validateFunctionCall(self: *Validator, call: *const CallExpression) anyerror!void {
        // Check if function exists
        if (call.callee.* == .identifier) {
            const func_name = call.callee.identifier.name;
            if (self.symbol_table.lookupSymbol(func_name) == null) {
                const error_msg = try std.fmt.allocPrint(self.context.allocator, "Undefined function '{s}'", .{func_name});
                try self.context.addError(ValidationError.withAllocatedMessage(error_msg, call.callee.getLocation(), .symbol_error));
                return;
            }
        }

        // TODO: Validate argument types match parameter types
        // This would require function signature information in symbol table
    }

    /// Validate event
    fn validateEvent(self: *Validator, event: *const Event) anyerror!void {
        _ = self;
        _ = event;
        // Events are generally always valid in their structure
    }
};

pub const ValidationResult = struct {
    valid: bool,
    errors: []ValidationError,
    warnings: []ValidationError,
};

/// Optimization pass interface
pub const OptimizationPass = struct {
    name: []const u8,
    run: *const fn (*HIRProgram, Allocator) anyerror!void,
};

/// Optimization pipeline for HIR transformations
pub const OptimizationPipeline = struct {
    passes: ArrayList(OptimizationPass),
    allocator: Allocator,

    pub fn init(allocator: Allocator) OptimizationPipeline {
        return OptimizationPipeline{
            .passes = ArrayList(OptimizationPass).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *OptimizationPipeline) void {
        self.passes.deinit();
    }

    pub fn addPass(self: *OptimizationPipeline, pass: OptimizationPass) !void {
        try self.passes.append(pass);
    }

    pub fn run(self: *OptimizationPipeline, program: *HIRProgram) !void {
        for (self.passes.items) |pass| {
            try pass.run(program, self.allocator);
        }
    }
};

/// IR Builder for constructing HIR from AST
pub const IRBuilder = struct {
    allocator: Allocator,
    program: HIRProgram,
    optimization_pipeline: OptimizationPipeline,

    pub fn init(allocator: Allocator) IRBuilder {
        return IRBuilder{
            .allocator = allocator,
            .program = HIRProgram.init(allocator),
            .optimization_pipeline = OptimizationPipeline.init(allocator),
        };
    }

    pub fn deinit(self: *IRBuilder) void {
        self.program.deinit();
        self.optimization_pipeline.deinit();
    }

    pub fn getProgram(self: *IRBuilder) HIRProgram {
        return self.program;
    }

    /// Build HIR from AST nodes
    pub fn buildFromAST(self: *IRBuilder, ast_nodes: []@import("ast.zig").AstNode) anyerror!void {
        var converter = ASTToHIRConverter.init(self.allocator, &self.program);
        defer converter.deinit();
        try converter.convertAST(ast_nodes);
    }

    /// Get a pointer to the HIR program for validation
    pub fn getProgramPtr(self: *IRBuilder) *HIRProgram {
        return &self.program;
    }

    /// Add an optimization pass to the pipeline
    pub fn addOptimizationPass(self: *IRBuilder, pass: OptimizationPass) !void {
        try self.optimization_pipeline.addPass(pass);
    }

    /// Run all optimization passes on the HIR
    pub fn optimize(self: *IRBuilder) !void {
        try self.optimization_pipeline.run(&self.program);
    }

    /// Build HIR from AST with optional optimization
    pub fn buildOptimized(self: *IRBuilder, ast_nodes: []@import("ast.zig").AstNode) anyerror!void {
        try self.buildFromAST(ast_nodes);
        try self.optimize();
    }
};

/// Standard optimization passes
pub const StandardOptimizations = struct {
    /// Dead code elimination pass
    pub fn deadCodeElimination(program: *HIRProgram, allocator: Allocator) anyerror!void {
        _ = allocator; // For future use

        for (program.contracts) |*contract| {
            for (contract.functions) |*function| {
                try eliminateDeadCodeInFunction(function);
            }
        }
    }

    /// Constant folding pass
    pub fn constantFolding(program: *HIRProgram, allocator: Allocator) anyerror!void {
        _ = allocator; // For future use

        for (program.contracts) |*contract| {
            for (contract.functions) |*function| {
                try foldConstantsInFunction(function);
            }
        }
    }

    /// Effect optimization pass
    pub fn effectOptimization(program: *HIRProgram, allocator: Allocator) anyerror!void {
        _ = allocator; // For future use

        for (program.contracts) |*contract| {
            for (contract.functions) |*function| {
                try optimizeEffectsInFunction(function);
            }
        }
    }

    /// Gas optimization pass for smart contracts
    pub fn gasOptimization(program: *HIRProgram, allocator: Allocator) anyerror!void {
        _ = allocator; // For future use

        for (program.contracts) |*contract| {
            try optimizeStorageLayout(contract);
            for (contract.functions) |*function| {
                try optimizeGasUsage(function);
            }
        }
    }

    /// Struct layout optimization pass
    pub fn structLayoutOptimization(program: *HIRProgram, allocator: Allocator) anyerror!void {
        _ = allocator; // For future use

        for (program.contracts) |*contract| {
            try optimizeStructLayoutInContract(contract);
        }
    }

    /// Struct field access optimization pass
    pub fn structFieldAccessOptimization(program: *HIRProgram, allocator: Allocator) anyerror!void {
        _ = allocator; // For future use

        for (program.contracts) |*contract| {
            for (contract.functions) |*function| {
                try optimizeStructFieldAccessInFunction(function);
            }
        }
    }

    /// Helper function to optimize struct layout in a contract
    fn optimizeStructLayoutInContract(contract: *Contract) anyerror!void {
        // Analyze struct usage patterns and optimize layout
        for (contract.storage) |*storage_var| {
            if (storage_var.type == .struct_type) {
                try optimizeStructTypeLayout(&storage_var.type);
            }
        }
    }

    /// Helper function to optimize struct field access in a function
    fn optimizeStructFieldAccessInFunction(function: *Function) anyerror!void {
        // Analyze struct field access patterns and optimize
        try optimizeStructAccessInBlock(&function.body);
    }

    /// Helper function to optimize struct type layout
    fn optimizeStructTypeLayout(struct_type: *Type) anyerror!void {
        switch (struct_type.*) {
            .struct_type => |*st| {
                // Analyze field access patterns and reorder if beneficial
                // This is a placeholder for actual optimization logic
                if (st.layout) |*layout| {
                    // Mark as optimized
                    layout.packed_efficiently = true;
                }
            },
            else => {},
        }
    }

    /// Helper function to optimize struct access patterns in a block
    fn optimizeStructAccessInBlock(block: *Block) anyerror!void {
        for (block.statements) |*stmt| {
            switch (stmt.*) {
                .if_statement => |*if_stmt| {
                    if (if_stmt.else_branch) |*else_branch| {
                        try optimizeStructAccessInBlock(else_branch);
                    }
                    try optimizeStructAccessInBlock(&if_stmt.then_branch);
                },
                .while_statement => |*while_stmt| {
                    try optimizeStructAccessInBlock(&while_stmt.body);
                },
                else => {
                    // Analyze other statements for struct field access
                },
            }
        }
    }

    /// Helper function to eliminate dead code in a function
    fn eliminateDeadCodeInFunction(function: *Function) anyerror!void {
        // Analyze the function body for unreachable code
        try analyzeReachability(&function.body);
    }

    /// Helper function to analyze reachability in a block
    fn analyzeReachability(block: *Block) anyerror!void {
        var reachable = true;

        for (block.statements, 0..) |*stmt, i| {
            if (!reachable) {
                // Mark statement as dead code (would need a flag in Statement)
                continue;
            }

            // Check if this statement makes subsequent code unreachable
            switch (stmt.*) {
                .return_statement => reachable = false,
                .if_statement => |*if_stmt| {
                    try analyzeReachability(&if_stmt.then_branch);
                    if (if_stmt.else_branch) |*else_branch| {
                        try analyzeReachability(else_branch);
                    }
                },
                .while_statement => |*while_stmt| {
                    try analyzeReachability(&while_stmt.body);
                },
                else => {},
            }

            _ = i; // Suppress unused variable warning
        }
    }

    /// Helper function to fold constants in a function
    fn foldConstantsInFunction(function: *Function) anyerror!void {
        try foldConstantsInBlock(&function.body);
    }

    /// Helper function to fold constants in a block
    fn foldConstantsInBlock(block: *Block) anyerror!void {
        for (block.statements) |*stmt| {
            switch (stmt.*) {
                .expression_statement => |*expr_stmt| {
                    try foldConstantsInExpression(expr_stmt.expression);
                },
                .if_statement => |*if_stmt| {
                    try foldConstantsInExpression(if_stmt.condition);
                    try foldConstantsInBlock(&if_stmt.then_branch);
                    if (if_stmt.else_branch) |*else_branch| {
                        try foldConstantsInBlock(else_branch);
                    }
                },
                .while_statement => |*while_stmt| {
                    try foldConstantsInExpression(while_stmt.condition);
                    try foldConstantsInBlock(&while_stmt.body);
                },
                .return_statement => |*ret_stmt| {
                    if (ret_stmt.value) |value| {
                        try foldConstantsInExpression(value);
                    }
                },
                else => {},
            }
        }
    }

    /// Helper function to fold constants in an expression
    fn foldConstantsInExpression(expr: *Expression) anyerror!void {
        switch (expr.*) {
            .binary => |*binary| {
                try foldConstantsInExpression(binary.left);
                try foldConstantsInExpression(binary.right);

                // Check if both operands are now literals and can be folded
                if (binary.left.* == .literal and binary.right.* == .literal) {
                    // Attempt constant folding based on operator
                    try attemptBinaryConstantFold(binary);
                }
            },
            .unary => |*unary| {
                try foldConstantsInExpression(unary.operand);

                // Check if operand is now a literal
                if (unary.operand.* == .literal) {
                    try attemptUnaryConstantFold(unary);
                }
            },
            .call => |*call| {
                try foldConstantsInExpression(call.callee);
                for (call.arguments) |*arg| {
                    try foldConstantsInExpression(arg);
                }
            },
            .index => |*index| {
                try foldConstantsInExpression(index.target);
                try foldConstantsInExpression(index.index);
            },
            .field => |*field| {
                try foldConstantsInExpression(field.target);
            },
            else => {},
        }
    }

    /// Attempt to fold a binary operation with constant operands
    fn attemptBinaryConstantFold(binary: *BinaryExpression) anyerror!void {
        // Simplified constant folding - would need full implementation
        _ = binary;
        // TODO: Implement actual constant folding based on operator and literal types
    }

    /// Attempt to fold a unary operation with constant operand
    fn attemptUnaryConstantFold(unary: *UnaryExpression) anyerror!void {
        // Simplified constant folding - would need full implementation
        _ = unary;
        // TODO: Implement actual unary constant folding
    }

    /// Helper function to optimize effects in a function
    fn optimizeEffectsInFunction(function: *Function) anyerror!void {
        // Remove redundant effects
        try removeRedundantEffects(&function.state_effects);
        try removeRedundantEffects(&function.observable_effects);
    }

    /// Remove redundant effects from an effect set
    fn removeRedundantEffects(effect_set: *EffectSet) anyerror!void {
        // TODO: Implement redundant effect removal
        _ = effect_set;
    }

    /// Helper function to optimize storage layout
    fn optimizeStorageLayout(contract: *Contract) anyerror!void {
        // Pack storage variables efficiently
        try packStorageVariables(contract.storage);
    }

    /// Pack storage variables for optimal gas usage
    fn packStorageVariables(storage_vars: []StorageVariable) anyerror!void {
        // TODO: Implement storage packing optimization
        _ = storage_vars;
    }

    /// Helper function to optimize gas usage in a function
    fn optimizeGasUsage(function: *Function) anyerror!void {
        // Optimize function for minimal gas consumption
        try optimizeForGas(&function.body);
    }

    /// Optimize a block for gas efficiency
    fn optimizeForGas(block: *Block) anyerror!void {
        // TODO: Implement gas optimization strategies
        _ = block;
    }

    /// Get all standard optimization passes
    pub fn getAllStandardPasses(allocator: Allocator) ![]OptimizationPass {
        const passes = [_]OptimizationPass{
            OptimizationPass{ .name = "dead-code-elimination", .run = deadCodeElimination },
            OptimizationPass{ .name = "constant-folding", .run = constantFolding },
            OptimizationPass{ .name = "effect-optimization", .run = effectOptimization },
            OptimizationPass{ .name = "gas-optimization", .run = gasOptimization },
        };

        return try allocator.dupe(OptimizationPass, &passes);
    }
};

/// JSON serialization support
pub const JSONSerializer = struct {
    pub fn serializeProgram(program: *const HIRProgram, writer: anytype) !void {
        try writer.writeAll("{\n");
        try writer.print("  \"version\": \"{s}\",\n", .{program.version});
        try writer.writeAll("  \"contracts\": [\n");

        for (program.contracts, 0..) |*contract, i| {
            if (i > 0) try writer.writeAll(",\n");
            try serializeContract(contract, writer, 2);
        }

        try writer.writeAll("\n  ]\n");
        try writer.writeAll("}\n");
    }

    fn serializeContract(contract: *const Contract, writer: anytype, indent: u32) !void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        try writeIndent(writer, indent + 1);
        try writer.print("\"name\": \"{s}\",\n", .{contract.name});

        // Serialize storage variables
        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"storage\": [\n");
        for (contract.storage, 0..) |*storage, i| {
            if (i > 0) try writer.writeAll(",\n");
            try serializeStorageVariable(storage, writer, indent + 2);
        }
        try writer.writeAll("\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("],\n");

        // Serialize functions
        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"functions\": [\n");
        for (contract.functions, 0..) |*function, i| {
            if (i > 0) try writer.writeAll(",\n");
            try serializeFunction(function, writer, indent + 2);
        }
        try writer.writeAll("\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("],\n");

        // Serialize events
        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"events\": [\n");
        for (contract.events, 0..) |*event, i| {
            if (i > 0) try writer.writeAll(",\n");
            try serializeEvent(event, writer, indent + 2);
        }
        try writer.writeAll("\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("]\n");

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeStorageVariable(storage: *const StorageVariable, writer: anytype, indent: u32) !void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        try writeIndent(writer, indent + 1);
        try writer.print("\"name\": \"{s}\",\n", .{storage.name});

        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"type\": ");
        try serializeType(&storage.type, writer);
        try writer.writeAll(",\n");

        try writeIndent(writer, indent + 1);
        try writer.print("\"region\": \"{s}\",\n", .{storage.region.toString()});

        try writeIndent(writer, indent + 1);
        try writer.print("\"mutable\": {}\n", .{storage.mutable});

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeFunction(function: *const Function, writer: anytype, indent: u32) !void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        try writeIndent(writer, indent + 1);
        try writer.print("\"name\": \"{s}\",\n", .{function.name});

        try writeIndent(writer, indent + 1);
        try writer.print("\"visibility\": \"{s}\",\n", .{function.visibility.toString()});

        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"parameters\": [\n");
        for (function.parameters, 0..) |*param, i| {
            if (i > 0) try writer.writeAll(",\n");
            try serializeParameter(param, writer, indent + 2);
        }
        try writer.writeAll("\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("],\n");

        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"return_type\": ");
        if (function.return_type) |*ret_type| {
            try serializeType(ret_type, writer);
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n");

        try writeIndent(writer, indent + 1);
        try writer.print("\"state_effects_count\": {},\n", .{function.state_effects.effects.items.len});

        try writeIndent(writer, indent + 1);
        try writer.print("\"observable_effects_count\": {},\n", .{function.observable_effects.effects.items.len});

        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"effects\": {\n");
        try writeIndent(writer, indent + 2);
        try writer.print("\"writes_storage\": {},\n", .{function.effects.writes_storage});
        try writeIndent(writer, indent + 2);
        try writer.print("\"reads_storage\": {},\n", .{function.effects.reads_storage});
        try writeIndent(writer, indent + 2);
        try writer.print("\"writes_transient\": {},\n", .{function.effects.writes_transient});
        try writeIndent(writer, indent + 2);
        try writer.print("\"reads_transient\": {},\n", .{function.effects.reads_transient});
        try writeIndent(writer, indent + 2);
        try writer.print("\"emits_logs\": {},\n", .{function.effects.emits_logs});
        try writeIndent(writer, indent + 2);
        try writer.print("\"calls_other\": {},\n", .{function.effects.calls_other});
        try writeIndent(writer, indent + 2);
        try writer.print("\"modifies_state\": {},\n", .{function.effects.modifies_state});
        try writeIndent(writer, indent + 2);
        try writer.print("\"is_pure\": {}\n", .{function.effects.is_pure});
        try writeIndent(writer, indent + 1);
        try writer.writeAll("}\n");

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeParameter(param: *const Parameter, writer: anytype, indent: u32) !void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        try writeIndent(writer, indent + 1);
        try writer.print("\"name\": \"{s}\",\n", .{param.name});

        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"type\": ");
        try serializeType(&param.type, writer);
        try writer.writeAll("\n");

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeEvent(event: *const Event, writer: anytype, indent: u32) !void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        try writeIndent(writer, indent + 1);
        try writer.print("\"name\": \"{s}\",\n", .{event.name});

        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"fields\": [\n");
        for (event.fields, 0..) |*field, i| {
            if (i > 0) try writer.writeAll(",\n");
            try serializeEventField(field, writer, indent + 2);
        }
        try writer.writeAll("\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("]\n");

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeEventField(field: *const EventField, writer: anytype, indent: u32) !void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        try writeIndent(writer, indent + 1);
        try writer.print("\"name\": \"{s}\",\n", .{field.name});

        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"type\": ");
        try serializeType(&field.type, writer);
        try writer.writeAll(",\n");

        try writeIndent(writer, indent + 1);
        try writer.print("\"indexed\": {}\n", .{field.indexed});

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeType(typ: *const Type, writer: anytype) !void {
        switch (typ.*) {
            .primitive => |prim| {
                try writer.print("\"{s}\"", .{prim.toString()});
            },
            .mapping => |mapping| {
                try writer.writeAll("{\"type\": \"mapping\", \"key\": ");
                try serializeType(mapping.key_type, writer);
                try writer.writeAll(", \"value\": ");
                try serializeType(mapping.value_type, writer);
                try writer.writeAll("}");
            },
            .slice => |slice| {
                try writer.writeAll("{\"type\": \"slice\", \"element\": ");
                try serializeType(slice.element_type, writer);
                try writer.writeAll("}");
            },
            .custom => |custom| {
                try writer.writeAll("{\"type\": \"custom\", \"name\": \"");
                try writer.writeAll(custom.name);
                try writer.writeAll("\"}");
            },
            .error_union => |error_union| {
                try writer.writeAll("{\"type\": \"error_union\", \"success_type\": ");
                try serializeType(error_union.success_type, writer);
                try writer.writeAll("}");
            },
            .result => |result| {
                try writer.writeAll("{\"type\": \"result\", \"ok_type\": ");
                try serializeType(result.ok_type, writer);
                try writer.writeAll(", \"error_type\": ");
                try serializeType(result.error_type, writer);
                try writer.writeAll("}");
            },
            .struct_type => |struct_type| {
                try writer.writeAll("{\"type\": \"struct\", \"name\": \"");
                try writer.writeAll(struct_type.name);
                try writer.writeAll("\", \"fields\": [");
                for (struct_type.fields, 0..) |field, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.writeAll("{\"name\": \"");
                    try writer.writeAll(field.name);
                    try writer.writeAll("\", \"type\": ");
                    try serializeType(field.field_type, writer);
                    try writer.writeAll("}");
                }
                try writer.writeAll("]}");
            },
            .enum_type => |enum_type| {
                try writer.writeAll("{\"type\": \"enum\", \"name\": \"");
                try writer.writeAll(enum_type.name);
                try writer.writeAll("\", \"variants\": [");
                for (enum_type.variants, 0..) |variant, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.writeAll("{\"name\": \"");
                    try writer.writeAll(variant.name);
                    try writer.writeAll("\"}");
                }
                try writer.writeAll("]}");
            },
        }
    }

    fn writeIndent(writer: anytype, indent: u32) !void {
        var i: u32 = 0;
        while (i < indent) : (i += 1) {
            try writer.writeAll("  ");
        }
    }
};

/// AST to HIR Converter
pub const ASTToHIRConverter = struct {
    allocator: Allocator,
    program: *HIRProgram,
    storage_variables: std.HashMap([]const u8, void, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    transient_variables: std.HashMap([]const u8, void, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

    // Import AST types
    const AstNode = @import("ast.zig").AstNode;
    const ContractNode = @import("ast.zig").ContractNode;
    const FunctionNode = @import("ast.zig").FunctionNode;
    const VariableDeclNode = @import("ast.zig").VariableDeclNode;
    const LogDeclNode = @import("ast.zig").LogDeclNode;
    const ExprNode = @import("ast.zig").ExprNode;
    const StmtNode = @import("ast.zig").StmtNode;
    const BlockNode = @import("ast.zig").BlockNode;
    const ASTTypeRef = @import("ast.zig").TypeRef;
    const ASTMemoryRegion = @import("ast.zig").MemoryRegion;
    const SourceSpan = @import("ast.zig").SourceSpan;

    pub fn init(allocator: Allocator, program: *HIRProgram) ASTToHIRConverter {
        return ASTToHIRConverter{
            .allocator = allocator,
            .program = program,
            .storage_variables = std.HashMap([]const u8, void, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .transient_variables = std.HashMap([]const u8, void, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *ASTToHIRConverter) void {
        self.storage_variables.deinit();
        self.transient_variables.deinit();
    }

    /// Convert a list of AST nodes to HIR
    pub fn convertAST(self: *ASTToHIRConverter, ast_nodes: []AstNode) anyerror!void {
        for (ast_nodes) |*node| {
            try self.convertASTNode(node);
        }
    }

    /// Convert a single AST node
    fn convertASTNode(self: *ASTToHIRConverter, node: *AstNode) anyerror!void {
        switch (node.*) {
            .Contract => |*contract_node| {
                try self.convertContract(contract_node);
            },
            else => {
                // Skip non-contract top-level nodes for now
            },
        }
    }

    /// Convert a contract AST node to HIR Contract
    fn convertContract(self: *ASTToHIRConverter, ast_contract: *ContractNode) anyerror!void {
        var hir_contract = Contract.init(self.allocator, ast_contract.name);

        // Clear previous contract's variable tracking
        self.storage_variables.clearRetainingCapacity();
        self.transient_variables.clearRetainingCapacity();

        // Convert contract members
        var storage_vars = std.ArrayList(StorageVariable).init(self.allocator);
        defer storage_vars.deinit();

        var functions = std.ArrayList(Function).init(self.allocator);
        defer functions.deinit();

        var events = std.ArrayList(Event).init(self.allocator);
        defer events.deinit();

        for (ast_contract.body) |*member| {
            switch (member.*) {
                .VariableDecl => |*var_decl| {
                    if (var_decl.region == .Storage) {
                        const storage_var = try self.convertStorageVariable(var_decl);
                        try storage_vars.append(storage_var);
                        // Track storage variables
                        try self.storage_variables.put(var_decl.name, {});
                    } else if (var_decl.region == .TStore) {
                        // Track transient variables
                        try self.transient_variables.put(var_decl.name, {});
                    }
                },
                .Function => |*function| {
                    const hir_function = try self.convertFunction(function);
                    try functions.append(hir_function);
                },
                .LogDecl => |*log_decl| {
                    const hir_event = try self.convertEvent(log_decl);
                    try events.append(hir_event);
                },
                .EnumDecl => |*enum_decl| {
                    // Register enum type in HIR
                    _ = enum_decl; // For now, just suppress unused variable warning
                    // TODO: Add proper enum type registration to HIR
                },
                else => {
                    // Skip other member types for now
                },
            }
        }

        // Allocate and copy arrays
        hir_contract.storage = try self.allocator.dupe(StorageVariable, storage_vars.items);
        hir_contract.functions = try self.allocator.dupe(Function, functions.items);
        hir_contract.events = try self.allocator.dupe(Event, events.items);

        // Add contract to program
        try self.program.addContract(hir_contract);
    }

    /// Convert AST variable declaration to HIR storage variable
    fn convertStorageVariable(self: *ASTToHIRConverter, ast_var: *VariableDeclNode) anyerror!StorageVariable {
        return StorageVariable{
            .name = ast_var.name,
            .type = try self.convertType(&ast_var.typ),
            .region = self.convertRegion(ast_var.region),
            .mutable = ast_var.mutable,
            .slot = null, // Will be assigned during layout
            .value = if (ast_var.value) |*val| try self.convertExpression(val) else null,
            .location = self.convertSourceLocation(ast_var.span),
            .allocator = self.allocator,
        };
    }

    /// Convert AST function to HIR function
    fn convertFunction(self: *ASTToHIRConverter, ast_func: *FunctionNode) anyerror!Function {
        // Convert parameters
        var hir_params = std.ArrayList(Parameter).init(self.allocator);
        defer hir_params.deinit();

        for (ast_func.parameters) |*param| {
            try hir_params.append(Parameter{
                .name = param.name,
                .type = try self.convertType(&param.typ),
                .location = self.convertSourceLocation(param.span),
            });
        }

        // Convert requires/ensures clauses
        var requires = std.ArrayList(Expression).init(self.allocator);
        defer requires.deinit();

        var ensures = std.ArrayList(Expression).init(self.allocator);
        defer ensures.deinit();

        for (ast_func.requires_clauses) |*req| {
            const hir_expr = try self.convertExpression(req);
            try requires.append(hir_expr.*);
            // Free the allocated expression since we copied it
            self.allocator.destroy(hir_expr);
        }

        for (ast_func.ensures_clauses) |*ens| {
            const hir_expr = try self.convertExpression(ens);
            try ensures.append(hir_expr.*);
            // Free the allocated expression since we copied it
            self.allocator.destroy(hir_expr);
        }

        // Convert function body
        const hir_body = try self.convertBlock(&ast_func.body);

        return Function{
            .name = ast_func.name,
            .visibility = if (ast_func.pub_) .public else .internal,
            .parameters = try self.allocator.dupe(Parameter, hir_params.items),
            .return_type = if (ast_func.return_type) |*ret_type| try self.convertType(ret_type) else null,
            .requires = try self.allocator.dupe(Expression, requires.items),
            .ensures = try self.allocator.dupe(Expression, ensures.items),
            .body = hir_body,
            .state_effects = try self.computeStateEffects(ast_func), // Compute state effects
            .observable_effects = try self.computeObservableEffects(ast_func), // Compute observable effects
            .effects = try self.computeFunctionEffectMetadata(ast_func), // Compute effect metadata
            .location = self.convertSourceLocation(ast_func.span),
            .allocator = self.allocator,
        };
    }

    /// Compute state effects for a function by analyzing its body
    fn computeStateEffects(self: *ASTToHIRConverter, ast_func: *FunctionNode) anyerror!EffectSet {
        var effects = EffectSet.init(self.allocator);

        // Analyze function body to determine state effects
        try self.analyzeBlockForStateEffects(&ast_func.body, &effects);

        return effects;
    }

    /// Compute observable effects for a function by analyzing its body
    fn computeObservableEffects(self: *ASTToHIRConverter, ast_func: *FunctionNode) anyerror!EffectSet {
        var effects = EffectSet.init(self.allocator);

        // Analyze function body to determine observable effects
        try self.analyzeBlockForObservableEffects(&ast_func.body, &effects);

        return effects;
    }

    /// Compute function effect metadata for tooling
    fn computeFunctionEffectMetadata(self: *ASTToHIRConverter, ast_func: *FunctionNode) anyerror!FunctionEffects {
        var effects = FunctionEffects.init();

        // Analyze function body to determine effects
        try self.analyzeFunctionBodyForEffects(&ast_func.body, &effects);

        return effects;
    }

    /// Analyze function body for effect metadata
    fn analyzeFunctionBodyForEffects(self: *ASTToHIRConverter, block: *BlockNode, effects: *FunctionEffects) anyerror!void {
        for (block.statements) |*stmt| {
            try self.analyzeStatementForEffectMetadata(stmt, effects);
        }
    }

    /// Analyze a statement for effect metadata
    fn analyzeStatementForEffectMetadata(self: *ASTToHIRConverter, stmt: *StmtNode, effects: *FunctionEffects) anyerror!void {
        switch (stmt.*) {
            .VariableDecl => |*var_decl| {
                if (var_decl.region == .Storage) {
                    effects.markStorageWrite();
                } else if (var_decl.region == .TStore) {
                    effects.markTransientWrite();
                }
            },
            .Expr => |*expr| {
                try self.analyzeExpressionForEffectMetadata(expr, effects);
            },
            .If => |*if_stmt| {
                try self.analyzeFunctionBodyForEffects(&if_stmt.then_branch, effects);
                if (if_stmt.else_branch) |*else_branch| {
                    try self.analyzeFunctionBodyForEffects(else_branch, effects);
                }
            },
            .While => |*while_stmt| {
                try self.analyzeFunctionBodyForEffects(&while_stmt.body, effects);
            },
            .Log => |_| {
                effects.markLogEmission();
            },
            .Return => |*return_stmt| {
                if (return_stmt.value) |*value| {
                    try self.analyzeExpressionForEffectMetadata(value, effects);
                }
            },
            else => {
                // Other statement types don't affect our metadata
            },
        }
    }

    /// Analyze an expression for effect metadata
    fn analyzeExpressionForEffectMetadata(self: *ASTToHIRConverter, expr: *ExprNode, effects: *FunctionEffects) anyerror!void {
        switch (expr.*) {
            .Assignment => |*assign| {
                // Check if assignment target is storage or transient
                try self.analyzeAssignmentTargetForEffects(assign.target, effects);
                // Recursively analyze the value expression
                try self.analyzeExpressionForEffectMetadata(assign.value, effects);
            },
            .CompoundAssignment => |*comp_assign| {
                // Compound assignments read and write
                try self.analyzeAssignmentTargetForEffects(comp_assign.target, effects);
                try self.analyzeExpressionForEffectMetadata(comp_assign.value, effects);
            },
            .Call => |*call| {
                // Function calls might call other functions
                effects.markExternalCall();
                // Recursively analyze arguments
                for (call.arguments) |*arg| {
                    try self.analyzeExpressionForEffectMetadata(arg, effects);
                }
            },
            .Index => |*index| {
                // Index access might be storage read
                try self.analyzeStorageAccessForEffects(index.target, effects, false);
                try self.analyzeExpressionForEffectMetadata(index.index, effects);
            },
            .FieldAccess => |*field| {
                // Field access might be storage read
                try self.analyzeStorageAccessForEffects(field.target, effects, false);
            },
            .Identifier => |*ident| {
                // Check if this is a storage variable access
                try self.analyzeIdentifierForEffects(ident, effects);
            },
            .Binary => |*binary| {
                try self.analyzeExpressionForEffectMetadata(binary.lhs, effects);
                try self.analyzeExpressionForEffectMetadata(binary.rhs, effects);
            },
            .Unary => |*unary| {
                try self.analyzeExpressionForEffectMetadata(unary.operand, effects);
            },
            .Literal => |_| {
                // Literals don't have effects
            },
            else => {
                // Handle other expression types as needed
            },
        }
    }

    /// Analyze assignment target to determine if it's storage/transient write
    fn analyzeAssignmentTargetForEffects(self: *ASTToHIRConverter, target: *ExprNode, effects: *FunctionEffects) anyerror!void {
        switch (target.*) {
            .Identifier => |*ident| {
                // Check if this identifier refers to a storage variable
                if (self.isStorageVariable(ident.name)) {
                    effects.markStorageWrite();
                } else if (self.isTransientVariable(ident.name)) {
                    effects.markTransientWrite();
                }
            },
            .Index => |*index| {
                // Index into storage mapping/array
                try self.analyzeStorageAccessForEffects(index.target, effects, true);
            },
            .FieldAccess => |*field| {
                // Field access on storage
                try self.analyzeStorageAccessForEffects(field.target, effects, true);
            },
            else => {
                // Other assignment targets
            },
        }
    }

    /// Analyze storage access (read or write)
    fn analyzeStorageAccessForEffects(self: *ASTToHIRConverter, expr: *ExprNode, effects: *FunctionEffects, is_write: bool) anyerror!void {
        switch (expr.*) {
            .Identifier => |*ident| {
                if (self.isStorageVariable(ident.name)) {
                    if (is_write) {
                        effects.markStorageWrite();
                    } else {
                        effects.markStorageRead();
                    }
                } else if (self.isTransientVariable(ident.name)) {
                    if (is_write) {
                        effects.markTransientWrite();
                    } else {
                        effects.markTransientRead();
                    }
                }
            },
            else => {
                // For complex expressions, assume storage access
                if (is_write) {
                    effects.markStorageWrite();
                } else {
                    effects.markStorageRead();
                }
            },
        }
    }

    /// Analyze identifier for storage read effects
    fn analyzeIdentifierForEffects(self: *ASTToHIRConverter, ident: *@import("ast.zig").IdentifierExpr, effects: *FunctionEffects) anyerror!void {
        if (self.isStorageVariable(ident.name)) {
            effects.markStorageRead();
        } else if (self.isTransientVariable(ident.name)) {
            effects.markTransientRead();
        }
    }

    /// Check if an identifier refers to a storage variable
    fn isStorageVariable(self: *ASTToHIRConverter, name: []const u8) bool {
        return self.storage_variables.contains(name);
    }

    /// Check if an identifier refers to a transient variable
    fn isTransientVariable(self: *ASTToHIRConverter, name: []const u8) bool {
        return self.transient_variables.contains(name);
    }

    /// Analyze a block for state effects
    fn analyzeBlockForStateEffects(self: *ASTToHIRConverter, block: *BlockNode, effects: *EffectSet) anyerror!void {
        for (block.statements) |*stmt| {
            try self.analyzeStatementForStateEffects(stmt, effects);
        }
    }

    /// Analyze a block for observable effects
    fn analyzeBlockForObservableEffects(self: *ASTToHIRConverter, block: *BlockNode, effects: *EffectSet) anyerror!void {
        for (block.statements) |*stmt| {
            try self.analyzeStatementForObservableEffects(stmt, effects);
        }
    }

    /// Analyze a block for side effects (backwards compatibility)
    fn analyzeBlockForEffects(self: *ASTToHIRConverter, block: *BlockNode, effects: *EffectSet) anyerror!void {
        for (block.statements) |*stmt| {
            try self.analyzeStatementForEffects(stmt, effects);
        }
    }

    /// Analyze a statement for state effects
    fn analyzeStatementForStateEffects(self: *ASTToHIRConverter, stmt: *StmtNode, effects: *EffectSet) anyerror!void {
        switch (stmt.*) {
            .VariableDecl => |*var_decl| {
                if (var_decl.region == .Storage) {
                    // Storage variable declarations are state-modifying
                    try effects.add(Effect{
                        .type = .write,
                        .path = AccessPath{
                            .base = var_decl.name,
                            .region = Region.storage,
                            .selectors = &[_]AccessPath.PathSelector{},
                        },
                        .condition = null,
                    });
                }
            },
            .Expr => |*expr| {
                try self.analyzeExpressionForStateEffects(expr, effects);
            },
            .If => |*if_stmt| {
                try self.analyzeBlockForStateEffects(&if_stmt.then_branch, effects);
                if (if_stmt.else_branch) |*else_branch| {
                    try self.analyzeBlockForStateEffects(else_branch, effects);
                }
            },
            .While => |*while_stmt| {
                try self.analyzeBlockForStateEffects(&while_stmt.body, effects);
            },
            else => {
                // Other statement types don't modify state by default
            },
        }
    }

    /// Analyze a statement for observable effects
    fn analyzeStatementForObservableEffects(self: *ASTToHIRConverter, stmt: *StmtNode, effects: *EffectSet) anyerror!void {
        switch (stmt.*) {
            .Log => |_| {
                // Log statements are observable (emit events)
                try effects.add(Effect{
                    .type = .emit,
                    .path = AccessPath{
                        .base = "logs",
                        .region = Region.memory, // Logs don't have a specific region, use memory
                        .selectors = &[_]AccessPath.PathSelector{},
                    },
                    .condition = null,
                });
            },
            .Expr => |*expr| {
                try self.analyzeExpressionForObservableEffects(expr, effects);
            },
            .If => |*if_stmt| {
                try self.analyzeBlockForObservableEffects(&if_stmt.then_branch, effects);
                if (if_stmt.else_branch) |*else_branch| {
                    try self.analyzeBlockForObservableEffects(else_branch, effects);
                }
            },
            .While => |*while_stmt| {
                try self.analyzeBlockForObservableEffects(&while_stmt.body, effects);
            },
            else => {
                // Other statement types don't have observable effects by default
            },
        }
    }

    /// Analyze a statement for side effects (backwards compatibility)
    fn analyzeStatementForEffects(self: *ASTToHIRConverter, stmt: *StmtNode, effects: *EffectSet) anyerror!void {
        switch (stmt.*) {
            .VariableDecl => |*var_decl| {
                if (var_decl.region == .Storage) {
                    // Storage variable declarations are state-modifying
                    try effects.add(Effect{
                        .type = .write,
                        .path = AccessPath{
                            .base = var_decl.name,
                            .region = Region.storage,
                            .selectors = &[_]AccessPath.PathSelector{},
                        },
                        .condition = null,
                    });
                }
            },
            .Log => |_| {
                // Log statements are state-modifying (emit events)
                try effects.add(Effect{
                    .type = .emit,
                    .path = AccessPath{
                        .base = "logs",
                        .region = Region.memory, // Logs don't have a specific region, use memory
                        .selectors = &[_]AccessPath.PathSelector{},
                    },
                    .condition = null,
                });
            },
            .Expr => |*expr| {
                try self.analyzeExpressionForEffects(expr, effects);
            },
            .If => |*if_stmt| {
                try self.analyzeBlockForEffects(&if_stmt.then_branch, effects);
                if (if_stmt.else_branch) |*else_branch| {
                    try self.analyzeBlockForEffects(else_branch, effects);
                }
            },
            .While => |*while_stmt| {
                try self.analyzeBlockForEffects(&while_stmt.body, effects);
            },
            else => {
                // Other statement types don't modify state by default
            },
        }
    }

    /// Analyze an expression for state effects
    fn analyzeExpressionForStateEffects(self: *ASTToHIRConverter, expr: *ExprNode, effects: *EffectSet) anyerror!void {
        switch (expr.*) {
            .Assignment => |_| {
                // Assignments are state-modifying
                try effects.add(Effect{
                    .type = .write,
                    .path = AccessPath{
                        .base = "unknown",
                        .region = Region.stack, // Use stack as default for unknown assignments
                        .selectors = &[_]AccessPath.PathSelector{},
                    },
                    .condition = null,
                });
            },
            .CompoundAssignment => |_| {
                // Compound assignments are state-modifying
                try effects.add(Effect{
                    .type = .write,
                    .path = AccessPath{
                        .base = "unknown",
                        .region = Region.stack, // Use stack as default for unknown assignments
                        .selectors = &[_]AccessPath.PathSelector{},
                    },
                    .condition = null,
                });
            },
            .Call => |*call| {
                // Function calls might modify state - for now assume they do
                try effects.add(Effect{
                    .type = .write,
                    .path = AccessPath{
                        .base = "function_call",
                        .region = Region.stack, // Use stack as default for function calls
                        .selectors = &[_]AccessPath.PathSelector{},
                    },
                    .condition = null,
                });

                // Recursively analyze arguments
                for (call.arguments) |*arg| {
                    try self.analyzeExpressionForStateEffects(arg, effects);
                }
            },
            .Binary => |*binary| {
                try self.analyzeExpressionForStateEffects(binary.lhs, effects);
                try self.analyzeExpressionForStateEffects(binary.rhs, effects);
            },
            .Unary => |*unary| {
                try self.analyzeExpressionForStateEffects(unary.operand, effects);
            },
            else => {
                // Other expressions don't modify state by default
            },
        }
    }

    /// Analyze an expression for observable effects
    fn analyzeExpressionForObservableEffects(self: *ASTToHIRConverter, expr: *ExprNode, effects: *EffectSet) anyerror!void {
        switch (expr.*) {
            .Call => |*call| {
                // Function calls might have observable effects - for now assume they don't
                // In a full implementation, this would analyze the called function

                // Recursively analyze arguments
                for (call.arguments) |*arg| {
                    try self.analyzeExpressionForObservableEffects(arg, effects);
                }
            },
            .Binary => |*binary| {
                try self.analyzeExpressionForObservableEffects(binary.lhs, effects);
                try self.analyzeExpressionForObservableEffects(binary.rhs, effects);
            },
            .Unary => |*unary| {
                try self.analyzeExpressionForObservableEffects(unary.operand, effects);
            },
            else => {
                // Other expressions don't have observable effects by default
            },
        }
    }

    /// Analyze an expression for side effects (backwards compatibility)
    fn analyzeExpressionForEffects(self: *ASTToHIRConverter, expr: *ExprNode, effects: *EffectSet) anyerror!void {
        switch (expr.*) {
            .Assignment => |_| {
                // Assignments are state-modifying
                try effects.add(Effect{
                    .type = .write,
                    .path = AccessPath{
                        .base = "unknown",
                        .region = Region.stack, // Use stack as default for unknown assignments
                        .selectors = &[_]AccessPath.PathSelector{},
                    },
                    .condition = null,
                });
            },
            .CompoundAssignment => |_| {
                // Compound assignments are state-modifying
                try effects.add(Effect{
                    .type = .write,
                    .path = AccessPath{
                        .base = "unknown",
                        .region = Region.stack, // Use stack as default for unknown assignments
                        .selectors = &[_]AccessPath.PathSelector{},
                    },
                    .condition = null,
                });
            },
            .Call => |*call| {
                // Function calls might modify state - for now assume they do
                try effects.add(Effect{
                    .type = .write,
                    .path = AccessPath{
                        .base = "function_call",
                        .region = Region.stack, // Use stack as default for function calls
                        .selectors = &[_]AccessPath.PathSelector{},
                    },
                    .condition = null,
                });

                // Recursively analyze arguments
                for (call.arguments) |*arg| {
                    try self.analyzeExpressionForEffects(arg, effects);
                }
            },
            .Binary => |*binary| {
                try self.analyzeExpressionForEffects(binary.lhs, effects);
                try self.analyzeExpressionForEffects(binary.rhs, effects);
            },
            .Unary => |*unary| {
                try self.analyzeExpressionForEffects(unary.operand, effects);
            },
            else => {
                // Other expressions don't modify state by default
            },
        }
    }

    /// Get source location from an expression
    fn getExpressionSourceLocation(self: *ASTToHIRConverter, expr: *ExprNode) SourceLocation {
        // Extract source location from different expression types
        return switch (expr.*) {
            .Identifier => |*ident| self.convertSourceLocation(ident.span),
            .Literal => |*lit| self.getLiteralSourceLocation(lit),
            .Binary => |*bin| self.convertSourceLocation(bin.span),
            .Unary => |*un| self.convertSourceLocation(un.span),
            .Assignment => |*assign| self.convertSourceLocation(assign.span),
            .CompoundAssignment => |*comp| self.convertSourceLocation(comp.span),
            .Call => |*call| self.convertSourceLocation(call.span),
            .Index => |*index| self.convertSourceLocation(index.span),
            .FieldAccess => |*field| self.convertSourceLocation(field.span),
            .Cast => |*cast| self.convertSourceLocation(cast.span),
            else => SourceLocation{ .line = 0, .column = 0, .length = 0 }, // Fallback
        };
    }

    /// Get source location from a literal
    fn getLiteralSourceLocation(self: *ASTToHIRConverter, lit: *@import("ast.zig").LiteralNode) SourceLocation {
        return switch (lit.*) {
            .Integer => |*int| self.convertSourceLocation(int.span),
            .String => |*str| self.convertSourceLocation(str.span),
            .Bool => |*bool_lit| self.convertSourceLocation(bool_lit.span),
            .Address => |*addr| self.convertSourceLocation(addr.span),
            .Hex => |*hex| self.convertSourceLocation(hex.span),
        };
    }

    /// Determine if an event field should be indexed (basic heuristic)
    fn shouldIndexEventField(self: *ASTToHIRConverter, field: *@import("ast.zig").LogField) bool {
        _ = self;
        // Basic heuristic: index address and small integer types
        // In a full implementation, this would be configurable via annotations
        return switch (field.typ) {
            .Address => true,
            .U8, .U16, .U32, .U64 => true,
            .Bool => true,
            else => false,
        };
    }

    /// Convert AST log declaration to HIR event
    fn convertEvent(self: *ASTToHIRConverter, ast_log: *LogDeclNode) anyerror!Event {
        var hir_fields = std.ArrayList(EventField).init(self.allocator);
        defer hir_fields.deinit();

        for (ast_log.fields) |*field| {
            try hir_fields.append(EventField{
                .name = field.name,
                .type = try self.convertType(&field.typ),
                .indexed = self.shouldIndexEventField(field), // Check if field should be indexed
                .location = self.convertSourceLocation(field.span),
            });
        }

        return Event{
            .name = ast_log.name,
            .fields = try self.allocator.dupe(EventField, hir_fields.items),
            .location = self.convertSourceLocation(ast_log.span),
            .allocator = self.allocator,
        };
    }

    /// Convert AST block to HIR block
    fn convertBlock(self: *ASTToHIRConverter, ast_block: *BlockNode) anyerror!Block {
        var hir_statements = std.ArrayList(Statement).init(self.allocator);
        defer hir_statements.deinit();

        for (ast_block.statements) |*stmt| {
            const hir_stmt = try self.convertStatement(stmt);
            try hir_statements.append(hir_stmt);
        }

        return Block{
            .statements = try self.allocator.dupe(Statement, hir_statements.items),
            .location = self.convertSourceLocation(ast_block.span),
            .allocator = self.allocator,
        };
    }

    /// Convert AST statement to HIR statement
    fn convertStatement(self: *ASTToHIRConverter, ast_stmt: *StmtNode) anyerror!Statement {
        switch (ast_stmt.*) {
            .VariableDecl => |*var_decl| {
                return Statement{
                    .variable_decl = VariableDecl{
                        .name = var_decl.name,
                        .type = try self.convertType(&var_decl.typ),
                        .region = self.convertRegion(var_decl.region),
                        .mutable = var_decl.mutable,
                        .value = if (var_decl.value) |*val| try self.convertExpression(val) else null,
                        .location = self.convertSourceLocation(var_decl.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .Return => |*ret| {
                return Statement{
                    .return_statement = ReturnStatement{
                        .value = if (ret.value) |*val| try self.convertExpression(val) else null,
                        .location = self.convertSourceLocation(ret.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .If => |*if_stmt| {
                return Statement{
                    .if_statement = IfStatement{
                        .condition = try self.convertExpression(&if_stmt.condition),
                        .then_branch = try self.convertBlock(&if_stmt.then_branch),
                        .else_branch = if (if_stmt.else_branch) |*else_b| try self.convertBlock(else_b) else null,
                        .location = self.convertSourceLocation(if_stmt.span),
                    },
                };
            },
            .While => |*while_stmt| {
                var invariants = std.ArrayList(Expression).init(self.allocator);
                defer invariants.deinit();

                for (while_stmt.invariants) |*inv| {
                    const hir_expr = try self.convertExpression(inv);
                    try invariants.append(hir_expr.*);
                    // Free the allocated expression since we copied it
                    self.allocator.destroy(hir_expr);
                }

                return Statement{
                    .while_statement = WhileStatement{
                        .condition = try self.convertExpression(&while_stmt.condition),
                        .body = try self.convertBlock(&while_stmt.body),
                        .invariants = try self.allocator.dupe(Expression, invariants.items),
                        .location = self.convertSourceLocation(while_stmt.span),
                    },
                };
            },
            .Log => |*log| {
                var args = std.ArrayList(Expression).init(self.allocator);
                defer args.deinit();

                for (log.args) |*arg| {
                    const hir_expr = try self.convertExpression(arg);
                    try args.append(hir_expr.*);
                    // Free the allocated expression since we copied it
                    self.allocator.destroy(hir_expr);
                }

                // Create a proper call expression for log
                const log_expr = try self.allocator.create(Expression);
                log_expr.* = Expression{
                    .call = CallExpression{
                        .callee = try self.allocator.create(Expression),
                        .arguments = try self.allocator.dupe(Expression, args.items),
                        .location = self.convertSourceLocation(log.span),
                        .allocator = self.allocator,
                    },
                };

                // Initialize the callee as an identifier
                log_expr.call.callee.* = Expression{
                    .identifier = Identifier{
                        .name = log.event_name,
                        .location = self.convertSourceLocation(log.span),
                    },
                };

                return Statement{
                    .expression_statement = ExpressionStatement{
                        .expression = log_expr,
                        .location = self.convertSourceLocation(log.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .Expr => |*expr| {
                return Statement{
                    .expression_statement = ExpressionStatement{
                        .expression = try self.convertExpression(expr),
                        .location = self.getExpressionSourceLocation(expr), // Get actual source location from expression
                        .allocator = self.allocator,
                    },
                };
            },
            else => {
                // Return a placeholder for unsupported statement types
                const placeholder_expr = try self.allocator.create(Expression);
                placeholder_expr.* = Expression{
                    .literal = Literal{ .integer = "0" },
                };

                return Statement{
                    .expression_statement = ExpressionStatement{
                        .expression = placeholder_expr,
                        .location = SourceLocation{ .line = 0, .column = 0, .length = 0 },
                        .allocator = self.allocator,
                    },
                };
            },
        }
    }

    /// Convert AST expression to HIR expression
    fn convertExpression(self: *ASTToHIRConverter, ast_expr: *ExprNode) anyerror!*Expression {
        const hir_expr = try self.allocator.create(Expression);

        switch (ast_expr.*) {
            .Identifier => |*ident| {
                hir_expr.* = Expression{
                    .identifier = Identifier{
                        .name = ident.name,
                        .location = self.convertSourceLocation(ident.span),
                    },
                };
            },
            .Literal => |*lit| {
                hir_expr.* = Expression{
                    .literal = try self.convertLiteral(lit),
                };
            },
            .Binary => |*bin| {
                hir_expr.* = Expression{
                    .binary = BinaryExpression{
                        .left = try self.convertExpression(bin.lhs),
                        .operator = self.convertBinaryOp(bin.operator),
                        .right = try self.convertExpression(bin.rhs),
                        .location = self.convertSourceLocation(bin.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .Unary => |*un| {
                hir_expr.* = Expression{
                    .unary = UnaryExpression{
                        .operator = self.convertUnaryOp(un.operator),
                        .operand = try self.convertExpression(un.operand),
                        .location = self.convertSourceLocation(un.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .Assignment => |assign| {
                // For assignments, we should create a call expression to an assignment operation
                // or handle it as a separate statement type. For now, create a placeholder.
                _ = assign; // Suppress unused variable warning
                hir_expr.* = Expression{
                    .literal = Literal{ .integer = "0" },
                };
            },
            .Call => |*call| {
                var args = std.ArrayList(Expression).init(self.allocator);
                defer args.deinit();

                for (call.arguments) |*arg| {
                    const hir_arg = try self.convertExpression(arg);
                    try args.append(hir_arg.*);
                    // Free the allocated expression since we copied it
                    self.allocator.destroy(hir_arg);
                }

                hir_expr.* = Expression{
                    .call = CallExpression{
                        .callee = try self.convertExpression(call.callee),
                        .arguments = try self.allocator.dupe(Expression, args.items),
                        .location = self.convertSourceLocation(call.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .Index => |*idx| {
                hir_expr.* = Expression{
                    .index = IndexExpression{
                        .target = try self.convertExpression(idx.target),
                        .index = try self.convertExpression(idx.index),
                        .location = self.convertSourceLocation(idx.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .FieldAccess => |*field| {
                hir_expr.* = Expression{
                    .field = FieldExpression{
                        .target = try self.convertExpression(field.target),
                        .field = field.field,
                        .location = self.convertSourceLocation(field.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .Old => |*old| {
                hir_expr.* = Expression{
                    .old = OldExpression{
                        .expression = try self.convertExpression(old.expr),
                        .location = self.convertSourceLocation(old.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .Shift => |*shift| {
                hir_expr.* = Expression{
                    .shift = ShiftExpression{
                        .mapping = try self.convertExpression(shift.mapping),
                        .source = try self.convertExpression(shift.source),
                        .dest = try self.convertExpression(shift.dest),
                        .amount = try self.convertExpression(shift.amount),
                        .location = self.convertSourceLocation(shift.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .StructInstantiation => |*struct_inst| {
                // Convert struct instantiation to HIR with proper type resolution
                var field_values = std.ArrayList(StructFieldValue).init(self.allocator);
                defer field_values.deinit();

                for (struct_inst.fields) |*field| {
                    const hir_value = try self.convertExpression(field.value);
                    try field_values.append(StructFieldValue{
                        .field_name = field.name,
                        .value = hir_value,
                        .allocator = self.allocator,
                    });
                }

                // Try to resolve the struct type from the AST
                const struct_type = if (struct_inst.struct_name.* == .Identifier) blk: {
                    const name = struct_inst.struct_name.Identifier.name;
                    break :blk self.resolveStructType(name) catch Type{ .custom = .{ .name = name } };
                } else Type{ .custom = .{ .name = "unknown" } };

                hir_expr.* = Expression{
                    .struct_instantiation = StructInstantiationExpression{
                        .struct_type = struct_type,
                        .field_values = try field_values.toOwnedSlice(),
                        .location = self.convertSourceLocation(struct_inst.span),
                        .allocator = self.allocator,
                    },
                };
            },
            .EnumLiteral => |*enum_literal| {
                // Convert enum literal to HIR
                // For now, represent enum literals as their discriminant values
                _ = enum_literal; // Suppress unused variable warning

                hir_expr.* = Expression{
                    .literal = Literal{ .integer = "0" }, // For now, represent as discriminant value
                };
            },
            else => {
                // Create a placeholder literal for unsupported expressions
                hir_expr.* = Expression{
                    .literal = Literal{ .integer = "0" },
                };
            },
        }

        return hir_expr;
    }

    /// Convert AST literal to HIR literal
    fn convertLiteral(self: *ASTToHIRConverter, ast_lit: *@import("ast.zig").LiteralNode) anyerror!Literal {
        _ = self;
        return switch (ast_lit.*) {
            .Integer => |*int| Literal{ .integer = int.value },
            .String => |*str| Literal{ .string = str.value },
            .Bool => |*bool_lit| Literal{ .boolean = bool_lit.value },
            .Address => |*addr| Literal{ .address = addr.value },
            .Hex => |*hex| Literal{ .integer = hex.value },
        };
    }

    /// Convert AST type to HIR type
    fn convertType(self: *ASTToHIRConverter, ast_type: *ASTTypeRef) anyerror!Type {
        return switch (ast_type.*) {
            .Bool => Type{ .primitive = .bool },
            .Address => Type{ .primitive = .address },
            .U8 => Type{ .primitive = .u8 },
            .U16 => Type{ .primitive = .u16 },
            .U32 => Type{ .primitive = .u32 },
            .U64 => Type{ .primitive = .u64 },
            .U128 => Type{ .primitive = .u128 },
            .U256 => Type{ .primitive = .u256 },
            .I8 => Type{ .primitive = .i8 },
            .I16 => Type{ .primitive = .i16 },
            .I32 => Type{ .primitive = .i32 },
            .I64 => Type{ .primitive = .i64 },
            .I128 => Type{ .primitive = .i128 },
            .I256 => Type{ .primitive = .i256 },
            .String => Type{ .primitive = .string },
            .Bytes => Type{ .primitive = .bytes },
            .Mapping => |*mapping| {
                const key_type = try self.allocator.create(Type);
                const value_type = try self.allocator.create(Type);
                key_type.* = try self.convertType(@constCast(mapping.*.key));
                value_type.* = try self.convertType(@constCast(mapping.*.value));

                return Type{
                    .mapping = Type.MappingType{
                        .key_type = key_type,
                        .value_type = value_type,
                    },
                };
            },
            .Slice => |slice_element_type| {
                const element_type = try self.allocator.create(Type);
                element_type.* = try self.convertType(@constCast(slice_element_type));

                return Type{
                    .slice = Type.SliceType{
                        .element_type = element_type,
                    },
                };
            },
            .Identifier => |name| {
                // Try to resolve as struct type first
                if (self.resolveStructType(name)) |struct_type| {
                    return struct_type;
                } else |_| {
                    // Try to resolve as enum type
                    if (self.resolveEnumType(name)) |enum_type| {
                        return enum_type;
                    } else |_| {
                        // Fall back to custom type
                        return Type{
                            .custom = Type.CustomType{ .name = name },
                        };
                    }
                }
            },
            else => Type{ .primitive = .u256 }, // Default fallback
        };
    }

    /// Resolve a struct type by name with memory layout information
    fn resolveStructType(self: *ASTToHIRConverter, name: []const u8) anyerror!Type {
        // This is a placeholder implementation
        // In a full implementation, this would look up the struct type
        // from a type registry or symbol table

        // For now, create a basic struct type
        // This would be replaced with actual struct type resolution
        const fields = try self.allocator.alloc(Type.StructField, 0);

        return Type{
            .struct_type = Type.StructType{
                .name = name,
                .fields = fields,
                .layout = Type.StructType.StructLayout{
                    .total_size = 64, // Placeholder
                    .storage_slots = 2, // Placeholder
                    .alignment = 32,
                    .packed_efficiently = true,
                },
                .origin_type = null,
            },
        };
    }

    /// Resolve an enum type by name
    fn resolveEnumType(self: *ASTToHIRConverter, name: []const u8) anyerror!Type {
        // This is a placeholder implementation
        // In a full implementation, this would look up the enum type
        // from a type registry or symbol table

        // For now, create a basic enum type
        const variants = try self.allocator.alloc(Type.EnumType.EnumVariant, 0);

        return Type{
            .enum_type = Type.EnumType{
                .name = name,
                .variants = variants,
            },
        };
    }

    /// Convert AST memory region to HIR region
    fn convertRegion(self: *ASTToHIRConverter, ast_region: ASTMemoryRegion) Region {
        _ = self;
        return switch (ast_region) {
            .Stack => .stack,
            .Memory => .memory,
            .Storage => .storage,
            .TStore => .tstore,
            .Const => .const_,
            .Immutable => .immutable,
        };
    }

    /// Convert AST binary operator to HIR binary operator
    fn convertBinaryOp(self: *ASTToHIRConverter, ast_op: @import("ast.zig").BinaryOp) BinaryExpression.BinaryOp {
        _ = self;
        return switch (ast_op) {
            .Plus => .plus,
            .Minus => .minus,
            .Star => .star,
            .Slash => .slash,
            .Percent => .percent,
            .EqualEqual => .equal_equal,
            .BangEqual => .bang_equal,
            .Less => .less,
            .LessEqual => .less_equal,
            .Greater => .greater,
            .GreaterEqual => .greater_equal,
            .And => .and_,
            .Or => .or_,
            .BitAnd => .bit_and,
            .BitOr => .bit_or,
            .BitXor => .bit_xor,
            .ShiftLeft => .shift_left,
            .ShiftRight => .shift_right,
        };
    }

    /// Convert AST unary operator to HIR unary operator
    fn convertUnaryOp(self: *ASTToHIRConverter, ast_op: @import("ast.zig").UnaryOp) UnaryExpression.UnaryOp {
        _ = self;
        return switch (ast_op) {
            .Minus => .minus,
            .Bang => .bang,
            .BitNot => .bit_not,
        };
    }

    /// Convert AST source span to HIR source location
    fn convertSourceLocation(self: *ASTToHIRConverter, span: SourceSpan) SourceLocation {
        _ = self;
        return SourceLocation{
            .line = span.line,
            .column = span.column,
            .length = span.length,
        };
    }

    /// Compute all effects for a function by analyzing its body (kept for backwards compatibility)
    fn computeFunctionEffects(self: *ASTToHIRConverter, ast_func: *FunctionNode) anyerror!EffectSet {
        var effects = EffectSet.init(self.allocator);

        // Analyze function body to determine effects
        try self.analyzeBlockForEffects(&ast_func.body, &effects);

        return effects;
    }
};

// Tests
test "IR type compatibility" {
    const testing = std.testing;

    var u256_type = Type{ .primitive = .u256 };
    var u128_type = Type{ .primitive = .u128 };

    try testing.expect(u256_type.isCompatibleWith(&u256_type));
    try testing.expect(!u256_type.isCompatibleWith(&u128_type));
}

test "Effect conflict detection" {
    const testing = std.testing;

    var effect_set = EffectSet.init(testing.allocator);
    defer effect_set.deinit();

    const path = AccessPath{
        .base = "balance",
        .selectors = &[_]AccessPath.PathSelector{},
        .region = .storage,
    };

    const read_effect = Effect{
        .type = .read,
        .path = path,
        .condition = null,
    };

    const write_effect = Effect{
        .type = .write,
        .path = path,
        .condition = null,
    };

    try effect_set.add(read_effect);
    try effect_set.add(write_effect);

    try testing.expect(effect_set.hasConflict());
}

test "Validation context" {
    const testing = std.testing;

    var context = ValidationContext.init(testing.allocator);
    defer context.deinit();

    try testing.expect(!context.hasErrors());

    try context.addError(ValidationError.withStaticMessage("Test error", SourceLocation{ .line = 1, .column = 1, .length = 1 }, .type_error));

    try testing.expect(context.hasErrors());
}

test "Type promotion and inference" {
    const testing = std.testing;

    var u8_type = Type{ .primitive = .u8 };
    var u16_type = Type{ .primitive = .u16 };
    var u256_type = Type{ .primitive = .u256 };

    // Test type promotion
    try testing.expect(u8_type.isPromotableToType(&u16_type));
    try testing.expect(u8_type.isPromotableToType(&u256_type));
    try testing.expect(!u16_type.isPromotableToType(&u8_type));

    // Test common type detection
    const common_type = u8_type.getCommonType(&u16_type, testing.allocator);
    try testing.expect(common_type != null);
    try testing.expect(common_type.? == .primitive);
    try testing.expect(common_type.?.primitive == .u16);

    // Test literal type inference
    const int_literal = Literal{ .integer = "42" };
    const inferred_type = Type.inferLiteralType(&int_literal);
    try testing.expect(inferred_type == .primitive);
    try testing.expect(inferred_type.primitive == .u8);

    const large_literal = Literal{ .integer = "999999999999999999999999999999999999999999999999999999999999999999999999999999" };
    const large_inferred_type = Type.inferLiteralType(&large_literal);
    try testing.expect(large_inferred_type == .primitive);
    try testing.expect(large_inferred_type.primitive == .u256);
}

test "Optimization pipeline" {
    const testing = std.testing;

    var pipeline = OptimizationPipeline.init(testing.allocator);
    defer pipeline.deinit();

    // Add a test optimization pass
    const test_pass = OptimizationPass{
        .name = "test-pass",
        .run = struct {
            fn run(program: *HIRProgram, allocator: Allocator) anyerror!void {
                _ = program;
                _ = allocator;
                // Test pass - does nothing
            }
        }.run,
    };

    try pipeline.addPass(test_pass);
    try testing.expect(pipeline.passes.items.len == 1);
    try testing.expect(std.mem.eql(u8, pipeline.passes.items[0].name, "test-pass"));
}

test "Standard optimization passes" {
    const testing = std.testing;

    const passes = try StandardOptimizations.getAllStandardPasses(testing.allocator);
    defer testing.allocator.free(passes);

    try testing.expect(passes.len == 4);
    try testing.expect(std.mem.eql(u8, passes[0].name, "dead-code-elimination"));
    try testing.expect(std.mem.eql(u8, passes[1].name, "constant-folding"));
    try testing.expect(std.mem.eql(u8, passes[2].name, "effect-optimization"));
    try testing.expect(std.mem.eql(u8, passes[3].name, "gas-optimization"));
}

test "Enhanced expression type inference" {
    const testing = std.testing;

    var validator = Validator.init(testing.allocator);
    defer validator.deinit();

    // Create a simple binary expression: 1 + 2
    const left_expr = try testing.allocator.create(Expression);
    defer testing.allocator.destroy(left_expr);
    left_expr.* = Expression{ .literal = Literal{ .integer = "1" } };

    const right_expr = try testing.allocator.create(Expression);
    defer testing.allocator.destroy(right_expr);
    right_expr.* = Expression{ .literal = Literal{ .integer = "2" } };

    const binary_expr = BinaryExpression{
        .left = left_expr,
        .operator = .plus,
        .right = right_expr,
        .location = SourceLocation{ .line = 1, .column = 1, .length = 5 },
        .allocator = testing.allocator,
    };

    const inferred_type = validator.inferBinaryExpressionType(&binary_expr);
    try testing.expect(inferred_type != null);
    try testing.expect(inferred_type.? == .primitive);
    // Both operands are u8, so result should be u8
    try testing.expect(inferred_type.?.primitive == .u8);
}

test "Effect set filtering" {
    const testing = std.testing;

    var effect_set = EffectSet.init(testing.allocator);
    defer effect_set.deinit();

    const state_effect = Effect{
        .type = .write,
        .path = AccessPath{
            .base = "balance",
            .selectors = &[_]AccessPath.PathSelector{},
            .region = .storage,
        },
        .condition = null,
    };

    const observable_effect = Effect{
        .type = .emit,
        .path = AccessPath{
            .base = "event",
            .selectors = &[_]AccessPath.PathSelector{},
            .region = .memory,
        },
        .condition = null,
    };

    try effect_set.add(state_effect);
    try effect_set.add(observable_effect);

    // Test state effect filtering
    var state_effects = try effect_set.filterStateEffects(testing.allocator);
    defer state_effects.deinit();

    try testing.expect(state_effects.effects.items.len == 1);
    try testing.expect(state_effects.effects.items[0].type == .write);

    // Test observable effect filtering
    var observable_effects = try effect_set.filterObservableEffects(testing.allocator);
    defer observable_effects.deinit();

    try testing.expect(observable_effects.effects.items.len == 1);
    try testing.expect(observable_effects.effects.items[0].type == .emit);
}
