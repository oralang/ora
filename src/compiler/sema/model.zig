const std = @import("std");
const ast = @import("../ast/mod.zig");
const diagnostics = @import("../diagnostics/mod.zig");
const source = @import("../source/mod.zig");
const BigInt = std.math.big.int.Managed;

pub const ModuleImport = struct {
    range: source.TextRange,
    path: []const u8,
    alias: ?[]const u8,
    target_module_id: ?source.ModuleId = null,
};

pub const ModuleGraphInput = struct {
    module_id: source.ModuleId,
    file_id: source.FileId,
    path: []const u8,
    ast_file: *const ast.AstFile,
};

pub const ModuleSummary = struct {
    module_id: source.ModuleId,
    file_id: source.FileId,
    path: []const u8,
    imports: []ModuleImport,
    dependencies: []const source.ModuleId = &.{},
};

pub const NamedItem = struct {
    name: []const u8,
    item_id: ast.ItemId,
};

pub const ImplEntry = struct {
    trait_name: []const u8,
    target_name: []const u8,
    item_id: ast.ItemId,
};

pub const TraitMethodSignature = struct {
    name: []const u8,
    has_self: bool,
    is_comptime: bool = false,
    extern_call_kind: ast.ExternCallKind = .none,
    param_types: []const Type = &.{},
    return_type: Type = .{ .void = {} },
};

pub const TraitInterface = struct {
    trait_item_id: ast.ItemId,
    name: []const u8,
    is_extern: bool = false,
    methods: []const TraitMethodSignature,
};

pub const ImplInterface = struct {
    impl_item_id: ast.ItemId,
    trait_item_id: ast.ItemId,
    target_item_id: ast.ItemId,
    trait_name: []const u8,
    target_name: []const u8,
    methods: []const TraitMethodSignature,
};

pub const InstantiatedStructField = struct {
    name: []const u8,
    ty: Type,
};

pub const InstantiatedStruct = struct {
    template_item_id: ast.ItemId,
    mangled_name: []const u8,
    fields: []const InstantiatedStructField,
};

pub const InstantiatedEnum = struct {
    template_item_id: ast.ItemId,
    mangled_name: []const u8,
};

pub const InstantiatedBitfieldField = struct {
    name: []const u8,
    ty: Type,
    offset: ?u32,
    width: ?u32,
};

pub const InstantiatedBitfield = struct {
    template_item_id: ast.ItemId,
    mangled_name: []const u8,
    base_type: ?Type,
    fields: []const InstantiatedBitfieldField,
};

pub const Binding = union(enum) {
    item: ast.ItemId,
    pattern: ast.PatternId,
};

pub const ResolvedBinding = Binding;

pub const TypeKind = enum {
    unknown,
    void,
    bool,
    integer,
    string,
    address,
    bytes,
    external_proxy,
    named,
    function,
    contract,
    struct_,
    bitfield,
    enum_,
    tuple,
    array,
    slice,
    map,
    error_union,
    refinement,
};

pub const Region = enum {
    none,
    storage,
    memory,
    transient,
    calldata,
};

pub const NamedType = struct {
    name: []const u8,
};

pub const ExternalProxyType = struct {
    trait_name: []const u8,
};

pub const IntegerType = struct {
    bits: ?u16 = null,
    signed: ?bool = null,
    spelling: ?[]const u8 = null,
};

pub const FunctionType = struct {
    name: ?[]const u8 = null,
    param_types: []const Type = &.{},
    return_types: []const Type = &.{},
};

pub const ArrayType = struct {
    element_type: *const Type,
    len: ?u32 = null,
};

pub const SliceType = struct {
    element_type: *const Type,
};

pub const MapType = struct {
    key_type: ?*const Type = null,
    value_type: ?*const Type = null,
};

pub const ErrorUnionType = struct {
    payload_type: *const Type,
    error_types: []const Type = &.{},
};

pub const RefinementType = struct {
    name: []const u8,
    base_type: *const Type,
    args: []const ast.TypeArg = &.{},
};

pub const Type = union(TypeKind) {
    unknown: void,
    void: void,
    bool: void,
    integer: IntegerType,
    string: void,
    address: void,
    bytes: void,
    external_proxy: ExternalProxyType,
    named: NamedType,
    function: FunctionType,
    contract: NamedType,
    struct_: NamedType,
    bitfield: NamedType,
    enum_: NamedType,
    tuple: []const Type,
    array: ArrayType,
    slice: SliceType,
    map: MapType,
    error_union: ErrorUnionType,
    refinement: RefinementType,

    pub fn kind(self: Type) TypeKind {
        return std.meta.activeTag(self);
    }

    pub fn name(self: *const Type) ?[]const u8 {
        return switch (self.*) {
            .integer => |integer| integer.spelling,
            .external_proxy => |proxy| proxy.trait_name,
            .named => |named| named.name,
            .function => |function| function.name,
            .contract => |named| named.name,
            .struct_ => |named| named.name,
            .bitfield => |named| named.name,
            .enum_ => |named| named.name,
            .refinement => |refinement| refinement.name,
            else => null,
        };
    }

    pub fn elementType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .array => |array| array.element_type,
            .slice => |slice| slice.element_type,
            else => null,
        };
    }

    pub fn keyType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .map => |map| map.key_type,
            else => null,
        };
    }

    pub fn valueType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .map => |map| map.value_type,
            else => null,
        };
    }

    pub fn payloadType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .error_union => |error_union| error_union.payload_type,
            else => null,
        };
    }

    pub fn refinementBaseType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .refinement => |refinement| refinement.base_type,
            else => null,
        };
    }

    pub fn arrayLen(self: *const Type) ?u32 {
        return switch (self.*) {
            .array => |array| array.len,
            else => null,
        };
    }

    pub fn tupleTypes(self: *const Type) []const Type {
        return switch (self.*) {
            .tuple => |tuple| tuple,
            else => &.{},
        };
    }

    pub fn errorTypes(self: *const Type) []const Type {
        return switch (self.*) {
            .error_union => |error_union| error_union.error_types,
            else => &.{},
        };
    }

    pub fn paramTypes(self: *const Type) []const Type {
        return switch (self.*) {
            .function => |function| function.param_types,
            else => &.{},
        };
    }

    pub fn returnTypes(self: *const Type) []const Type {
        return switch (self.*) {
            .function => |function| function.return_types,
            else => &.{},
        };
    }
};

pub fn appendTypeMangleName(allocator: std.mem.Allocator, buffer: *std.ArrayList(u8), ty: Type) !void {
    switch (ty) {
        .bool => try buffer.appendSlice(allocator, "bool"),
        .address => try buffer.appendSlice(allocator, "address"),
        .string => try buffer.appendSlice(allocator, "string"),
        .bytes => try buffer.appendSlice(allocator, "bytes"),
        .external_proxy => |proxy| try buffer.writer(allocator).print("external_{s}", .{proxy.trait_name}),
        .void => try buffer.appendSlice(allocator, "void"),
        .integer => |integer| try buffer.appendSlice(allocator, integer.spelling orelse "int"),
        .named => |named| try buffer.appendSlice(allocator, named.name),
        .struct_ => |named| try buffer.appendSlice(allocator, named.name),
        .contract => |named| try buffer.appendSlice(allocator, named.name),
        .bitfield => |named| try buffer.appendSlice(allocator, named.name),
        .enum_ => |named| try buffer.appendSlice(allocator, named.name),
        .refinement => |refinement| try buffer.appendSlice(allocator, refinement.name),
        .slice => |slice| {
            try buffer.appendSlice(allocator, "slice_");
            try appendTypeMangleName(allocator, buffer, slice.element_type.*);
        },
        .array => |array| {
            try buffer.appendSlice(allocator, "array_");
            try appendTypeMangleName(allocator, buffer, array.element_type.*);
            if (array.len) |len| {
                try buffer.append(allocator, '_');
                try buffer.writer(allocator).print("{d}", .{len});
            }
        },
        .map => |map| {
            try buffer.appendSlice(allocator, "map");
            if (map.key_type) |key| {
                try buffer.append(allocator, '_');
                try appendTypeMangleName(allocator, buffer, key.*);
            }
            if (map.value_type) |value| {
                try buffer.append(allocator, '_');
                try appendTypeMangleName(allocator, buffer, value.*);
            }
        },
        .tuple => |elements| {
            try buffer.appendSlice(allocator, "tuple");
            for (elements) |element| {
                try buffer.append(allocator, '_');
                try appendTypeMangleName(allocator, buffer, element);
            }
        },
        .error_union => |error_union| {
            try buffer.appendSlice(allocator, "error_union_");
            try appendTypeMangleName(allocator, buffer, error_union.payload_type.*);
        },
        .function => |function| try buffer.appendSlice(allocator, function.name orelse "fn"),
        .unknown => try buffer.appendSlice(allocator, "type"),
    }
}

pub const LocatedType = struct {
    type: Type,
    region: Region = .none,

    pub fn unlocated(ty: Type) LocatedType {
        return .{
            .type = ty,
            .region = .none,
        };
    }

    pub fn withRegion(ty: Type, region: Region) LocatedType {
        return .{
            .type = ty,
            .region = region,
        };
    }

    pub fn kind(self: LocatedType) TypeKind {
        return self.type.kind();
    }

    pub fn name(self: *const LocatedType) ?[]const u8 {
        return self.type.name();
    }

    pub fn elementType(self: *const LocatedType) ?*const Type {
        return self.type.elementType();
    }

    pub fn keyType(self: *const LocatedType) ?*const Type {
        return self.type.keyType();
    }

    pub fn valueType(self: *const LocatedType) ?*const Type {
        return self.type.valueType();
    }

    pub fn payloadType(self: *const LocatedType) ?*const Type {
        return self.type.payloadType();
    }

    pub fn arrayLen(self: *const LocatedType) ?u32 {
        return self.type.arrayLen();
    }

    pub fn tupleTypes(self: *const LocatedType) []const Type {
        return self.type.tupleTypes();
    }

    pub fn errorTypes(self: *const LocatedType) []const Type {
        return self.type.errorTypes();
    }

    pub fn paramTypes(self: *const LocatedType) []const Type {
        return self.type.paramTypes();
    }

    pub fn returnTypes(self: *const LocatedType) []const Type {
        return self.type.returnTypes();
    }
};

pub const ConstValue = union(enum) {
    integer: BigInt,
    boolean: bool,
    string: []const u8,
};

pub const VerificationFact = struct {
    kind: ast.SpecClauseKind,
    expr: ast.ExprId,
    range: source.TextRange,
};

pub const EffectSlot = struct {
    name: []const u8,
    region: Region,
    key_path: ?[]const KeySegment = null,
};

pub const KeySegment = union(enum) {
    parameter: u32,
    constant: []const u8,
    self_ref,
    unknown,
};

pub const Effect = union(enum) {
    pure,
    external,
    side_effects: struct {
        has_external: bool = false,
        has_log: bool = false,
        has_havoc: bool = false,
        has_lock: bool = false,
        has_unlock: bool = false,
    },
    writes: struct {
        slots: []const EffectSlot,
        has_external: bool = false,
        has_log: bool = false,
        has_havoc: bool = false,
        has_lock: bool = false,
        has_unlock: bool = false,
    },
    reads: struct {
        slots: []const EffectSlot,
        has_external: bool = false,
        has_log: bool = false,
        has_havoc: bool = false,
        has_lock: bool = false,
        has_unlock: bool = false,
    },
    reads_writes: struct {
        reads: []const EffectSlot,
        writes: []const EffectSlot,
        has_external: bool = false,
        has_log: bool = false,
        has_havoc: bool = false,
        has_lock: bool = false,
        has_unlock: bool = false,
    },
};

pub const TypeCheckKey = union(enum) {
    item: ast.ItemId,
    body: ast.BodyId,
};

pub const VerificationFactsKey = union(enum) {
    item: ast.ItemId,
    body: ast.BodyId,
};

pub const ModuleGraphResult = struct {
    arena: std.heap.ArenaAllocator,
    package_id: source.PackageId,
    modules: []ModuleSummary,
    topo_order: []const source.ModuleId,
    has_cycles: bool,

    pub fn deinit(self: *ModuleGraphResult) void {
        self.arena.deinit();
    }
};

pub const ItemIndexResult = struct {
    arena: std.heap.ArenaAllocator,
    entries: []NamedItem,
    impl_entries: []ImplEntry,

    pub fn deinit(self: *ItemIndexResult) void {
        self.arena.deinit();
    }

    pub fn lookup(self: *const ItemIndexResult, name: []const u8) ?ast.ItemId {
        var left: usize = 0;
        var right: usize = self.entries.len;
        while (left < right) {
            const mid = left + (right - left) / 2;
            switch (std.mem.order(u8, name, self.entries[mid].name)) {
                .lt => right = mid,
                .gt => left = mid + 1,
                .eq => return self.entries[mid].item_id,
            }
        }
        return null;
    }

    pub fn lookupImpl(self: *const ItemIndexResult, trait_name: []const u8, target_name: []const u8) ?ast.ItemId {
        for (self.impl_entries) |entry| {
            if (std.mem.eql(u8, entry.trait_name, trait_name) and std.mem.eql(u8, entry.target_name, target_name)) {
                return entry.item_id;
            }
        }
        return null;
    }
};

pub const NameResolutionResult = struct {
    arena: std.heap.ArenaAllocator,
    expr_bindings: []?ResolvedBinding,
    diagnostics: diagnostics.DiagnosticList,

    pub fn deinit(self: *NameResolutionResult) void {
        self.diagnostics.deinit();
        self.arena.deinit();
    }
};

pub const TypeCheckResult = struct {
    arena: std.heap.ArenaAllocator,
    key: TypeCheckKey,
    item_types: []Type,
    item_regions: []Region,
    item_effects: []Effect,
    pattern_types: []LocatedType,
    expr_types: []Type,
    expr_effects: []Effect,
    body_types: []Type,
    instantiated_structs: []const InstantiatedStruct,
    instantiated_enums: []const InstantiatedEnum,
    instantiated_bitfields: []const InstantiatedBitfield,
    trait_interfaces: []const TraitInterface,
    impl_interfaces: []const ImplInterface,
    diagnostics: diagnostics.DiagnosticList,

    pub fn deinit(self: *TypeCheckResult) void {
        self.diagnostics.deinit();
        self.arena.deinit();
    }

    pub fn exprType(self: *const TypeCheckResult, id: ast.ExprId) Type {
        return self.expr_types[id.index()];
    }

    pub fn itemLocatedType(self: *const TypeCheckResult, id: ast.ItemId) LocatedType {
        return .{
            .type = self.item_types[id.index()],
            .region = self.item_regions[id.index()],
        };
    }

    pub fn exprEffect(self: *const TypeCheckResult, id: ast.ExprId) Effect {
        return self.expr_effects[id.index()];
    }

    pub fn itemEffect(self: *const TypeCheckResult, id: ast.ItemId) Effect {
        return self.item_effects[id.index()];
    }

    pub fn instantiatedStructByName(self: *const TypeCheckResult, name: []const u8) ?InstantiatedStruct {
        for (self.instantiated_structs) |instantiated| {
            if (std.mem.eql(u8, instantiated.mangled_name, name)) return instantiated;
        }
        return null;
    }

    pub fn instantiatedEnumByName(self: *const TypeCheckResult, name: []const u8) ?InstantiatedEnum {
        for (self.instantiated_enums) |instantiated| {
            if (std.mem.eql(u8, instantiated.mangled_name, name)) return instantiated;
        }
        return null;
    }

    pub fn instantiatedBitfieldByName(self: *const TypeCheckResult, name: []const u8) ?InstantiatedBitfield {
        for (self.instantiated_bitfields) |instantiated| {
            if (std.mem.eql(u8, instantiated.mangled_name, name)) return instantiated;
        }
        return null;
    }

    pub fn traitInterfaceByName(self: *const TypeCheckResult, name: []const u8) ?TraitInterface {
        for (self.trait_interfaces) |trait_interface| {
            if (std.mem.eql(u8, trait_interface.name, name)) return trait_interface;
        }
        return null;
    }

    pub fn implInterfaceByNames(self: *const TypeCheckResult, trait_name: []const u8, target_name: []const u8) ?ImplInterface {
        for (self.impl_interfaces) |impl_interface| {
            if (std.mem.eql(u8, impl_interface.trait_name, trait_name) and
                std.mem.eql(u8, impl_interface.target_name, target_name))
            {
                return impl_interface;
            }
        }
        return null;
    }
};

pub const ConstEvalResult = struct {
    arena: std.heap.ArenaAllocator,
    values: []?ConstValue,
    diagnostics: diagnostics.DiagnosticList,

    pub fn deinit(self: *ConstEvalResult) void {
        self.diagnostics.deinit();
        self.arena.deinit();
    }
};

pub const VerificationFactsResult = struct {
    arena: std.heap.ArenaAllocator,
    key: VerificationFactsKey,
    facts: []VerificationFact,

    pub fn deinit(self: *VerificationFactsResult) void {
        self.arena.deinit();
    }
};

pub const ModuleVerificationFactsResult = struct {
    arena: std.heap.ArenaAllocator,
    facts: []VerificationFact,

    pub fn deinit(self: *ModuleVerificationFactsResult) void {
        self.arena.deinit();
    }
};

test "LocatedType defaults to none region" {
    const ty = LocatedType.unlocated(.{ .integer = .{ .bits = 256, .signed = false } });
    try std.testing.expectEqual(Region.none, ty.region);
    try std.testing.expectEqual(TypeKind.integer, ty.type.kind());
}

test "LocatedType supports explicit regions" {
    const located = LocatedType.withRegion(.{ .bool = {} }, .storage);
    try std.testing.expectEqual(Region.storage, located.region);
    try std.testing.expectEqual(TypeKind.bool, located.type.kind());
}

test "LocatedType equality includes region" {
    const lhs = LocatedType.withRegion(.{ .address = {} }, .memory);
    const rhs_same = LocatedType.withRegion(.{ .address = {} }, .memory);
    const rhs_other_region = LocatedType.withRegion(.{ .address = {} }, .storage);

    try std.testing.expect(std.meta.eql(lhs, rhs_same));
    try std.testing.expect(!std.meta.eql(lhs, rhs_other_region));
}

test "refinement type preserves base type" {
    const base: Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const refinement: Type = .{ .refinement = .{
        .name = "MinValue",
        .base_type = &base,
        .args = &.{},
    } };

    try std.testing.expectEqual(TypeKind.refinement, refinement.kind());
    try std.testing.expectEqualStrings("MinValue", refinement.name().?);
    try std.testing.expect(refinement.refinementBaseType() != null);
    try std.testing.expectEqual(TypeKind.integer, refinement.refinementBaseType().?.kind());
}

test "Effect stub supports read and write slot sets" {
    const slots = [_]EffectSlot{
        .{ .name = "balances", .region = .storage },
        .{ .name = "pending", .region = .transient },
    };
    const read_effect: Effect = .{ .reads = .{ .slots = &slots } };
    const write_effect: Effect = .{ .writes = .{ .slots = &slots } };
    const both_effect: Effect = .{ .reads_writes = .{ .reads = &slots, .writes = &slots } };

    try std.testing.expectEqualStrings("balances", read_effect.reads.slots[0].name);
    try std.testing.expectEqual(.storage, read_effect.reads.slots[0].region);
    try std.testing.expectEqualStrings("pending", write_effect.writes.slots[1].name);
    try std.testing.expectEqual(.transient, write_effect.writes.slots[1].region);
    try std.testing.expectEqualStrings("balances", both_effect.reads_writes.reads[0].name);
    try std.testing.expectEqualStrings("pending", both_effect.reads_writes.writes[1].name);
}

test "Effect supports external call marker" {
    const slots = [_]EffectSlot{
        .{ .name = "total", .region = .storage },
    };
    const external_only: Effect = .external;
    const reads_external: Effect = .{ .reads = .{
        .slots = &slots,
        .has_external = true,
    } };
    const writes_external: Effect = .{ .writes = .{
        .slots = &slots,
        .has_external = true,
    } };
    const mixed_external: Effect = .{ .reads_writes = .{
        .reads = &slots,
        .writes = &slots,
        .has_external = true,
    } };

    try std.testing.expect(external_only == .external);
    try std.testing.expect(reads_external.reads.has_external);
    try std.testing.expect(writes_external.writes.has_external);
    try std.testing.expect(mixed_external.reads_writes.has_external);
}

test "Effect supports log and havoc markers" {
    const slots = [_]EffectSlot{
        .{ .name = "total", .region = .storage },
    };
    const reads_log: Effect = .{ .reads = .{
        .slots = &slots,
        .has_log = true,
    } };
    const writes_havoc: Effect = .{ .writes = .{
        .slots = &slots,
        .has_havoc = true,
    } };
    const mixed: Effect = .{ .reads_writes = .{
        .reads = &slots,
        .writes = &slots,
        .has_log = true,
        .has_havoc = true,
    } };

    try std.testing.expect(reads_log.reads.has_log);
    try std.testing.expect(writes_havoc.writes.has_havoc);
    try std.testing.expect(mixed.reads_writes.has_log);
    try std.testing.expect(mixed.reads_writes.has_havoc);
}

test "Effect supports side-effect-only marker" {
    const effect: Effect = .{ .side_effects = .{
        .has_log = true,
        .has_havoc = true,
        .has_lock = true,
        .has_unlock = true,
    } };

    try std.testing.expect(effect.side_effects.has_log);
    try std.testing.expect(effect.side_effects.has_havoc);
    try std.testing.expect(effect.side_effects.has_lock);
    try std.testing.expect(effect.side_effects.has_unlock);
}
