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

pub const Type = union(TypeKind) {
    unknown: void,
    void: void,
    bool: void,
    integer: IntegerType,
    string: void,
    address: void,
    bytes: void,
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

    pub fn kind(self: Type) TypeKind {
        return std.meta.activeTag(self);
    }

    pub fn name(self: *const Type) ?[]const u8 {
        return switch (self.*) {
            .integer => |integer| integer.spelling,
            .named => |named| named.name,
            .function => |function| function.name,
            .contract => |named| named.name,
            .struct_ => |named| named.name,
            .bitfield => |named| named.name,
            .enum_ => |named| named.name,
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

pub const Effect = union(enum) {
    pure,
    writes: struct { slots: []const []const u8 },
    reads: struct { slots: []const []const u8 },
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
    pattern_types: []LocatedType,
    expr_types: []Type,
    body_types: []Type,
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

test "Effect stub supports read and write slot sets" {
    const slots = [_][]const u8{ "balances", "pending" };
    const read_effect: Effect = .{ .reads = .{ .slots = &slots } };
    const write_effect: Effect = .{ .writes = .{ .slots = &slots } };

    try std.testing.expectEqualStrings("balances", read_effect.reads.slots[0]);
    try std.testing.expectEqualStrings("pending", write_effect.writes.slots[1]);
}
