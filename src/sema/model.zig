const std = @import("std");
const ast = @import("../ast/mod.zig");
const diagnostics = @import("../diagnostics/mod.zig");
const lookup_index = @import("lookup.zig");
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
    file_path: []const u8,
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

pub fn functionRuntimeSelfParameterIndex(file: *const ast.AstFile, function: ast.FunctionItem) ?usize {
    for (function.parameters, 0..) |parameter, index| {
        if (parameter.is_comptime) continue;
        const name = switch (file.pattern(parameter.pattern).*) {
            .Name => |pattern| pattern.name,
            else => null,
        };
        return if (std.mem.eql(u8, name orelse "", "self")) index else null;
    }
    return null;
}

pub fn functionHasRuntimeSelf(file: *const ast.AstFile, function: ast.FunctionItem) bool {
    return functionRuntimeSelfParameterIndex(file, function) != null;
}

pub const ContractMemberRole = enum {
    field,
    constant,
    function,
    struct_,
    bitfield,
    enum_,
    resource,
    type_alias,
    trait_,
    log_decl,
    error_decl,
};

pub const ContractMemberRoles = packed struct(u16) {
    field: bool = false,
    constant: bool = false,
    function: bool = false,
    struct_: bool = false,
    bitfield: bool = false,
    enum_: bool = false,
    resource: bool = false,
    type_alias: bool = false,
    trait_: bool = false,
    log_decl: bool = false,
    error_decl: bool = false,
    _padding: u5 = 0,

    pub fn contains(self: ContractMemberRoles, role: ContractMemberRole) bool {
        return switch (role) {
            .field => self.field,
            .constant => self.constant,
            .function => self.function,
            .struct_ => self.struct_,
            .bitfield => self.bitfield,
            .enum_ => self.enum_,
            .resource => self.resource,
            .type_alias => self.type_alias,
            .trait_ => self.trait_,
            .log_decl => self.log_decl,
            .error_decl => self.error_decl,
        };
    }
};

pub const TraitMethodSignature = struct {
    name: []const u8,
    receiver_kind: ast.ReceiverKind,
    is_comptime: bool = false,
    extern_call_kind: ast.ExternCallKind = .none,
    errors: []const []const u8 = &.{},
    param_types: []const Type = &.{},
    return_type: Type = .{ .void = {} },
};

pub const TraitInterface = struct {
    trait_item_id: ast.ItemId,
    name: []const u8,
    is_extern: bool = false,
    methods: []const TraitMethodSignature,
    method_lookup: []lookup_index.NamedEntry,

    pub fn methodByName(self: TraitInterface, name: []const u8) ?TraitMethodSignature {
        const index = findMethodIndexByName(self.methods, self.method_lookup, name) orelse return null;
        return self.methods[index];
    }

    pub fn methodByNameAndReceiver(self: TraitInterface, name: []const u8, receiver_kind: ast.ReceiverKind) ?TraitMethodSignature {
        const index = findMethodIndexByNameAndReceiver(self.methods, self.method_lookup, name, receiver_kind) orelse return null;
        return self.methods[index];
    }
};

pub const ImplInterface = struct {
    impl_item_id: ast.ItemId,
    trait_item_id: ast.ItemId,
    target_item_id: ast.ItemId,
    trait_name: []const u8,
    target_name: []const u8,
    methods: []const TraitMethodSignature,
    method_lookup: []lookup_index.NamedEntry,

    pub fn methodByNameAndReceiver(self: ImplInterface, name: []const u8, receiver_kind: ast.ReceiverKind) ?TraitMethodSignature {
        const index = findMethodIndexByNameAndReceiver(self.methods, self.method_lookup, name, receiver_kind) orelse return null;
        return self.methods[index];
    }

    pub fn methodIndexByNameAndReceiver(self: ImplInterface, name: []const u8, receiver_kind: ast.ReceiverKind) ?usize {
        return findMethodIndexByNameAndReceiver(self.methods, self.method_lookup, name, receiver_kind);
    }

    pub fn hasMethodByNameAndReceiver(self: ImplInterface, name: []const u8, receiver_kind: ast.ReceiverKind) bool {
        return self.methodIndexByNameAndReceiver(name, receiver_kind) != null;
    }

    pub fn methodCountByNameAndReceiver(self: ImplInterface, name: []const u8, receiver_kind: ast.ReceiverKind) usize {
        return countMethodsByNameAndReceiver(self.methods, self.method_lookup, name, receiver_kind);
    }
};

fn findMethodIndexByName(
    methods: []const TraitMethodSignature,
    method_lookup: []const lookup_index.NamedEntry,
    name: []const u8,
) ?usize {
    const index = lookup_index.findNamed(method_lookup, name) orelse return null;
    if (index >= methods.len) return null;
    return index;
}

fn findMethodIndexByNameAndReceiver(
    methods: []const TraitMethodSignature,
    method_lookup: []const lookup_index.NamedEntry,
    name: []const u8,
    receiver_kind: ast.ReceiverKind,
) ?usize {
    const range = lookup_index.findNamedRange(method_lookup, name) orelse return null;
    for (method_lookup[range.start..range.end]) |entry| {
        if (entry.index >= methods.len) return null;
        if (methods[entry.index].receiver_kind == receiver_kind) return entry.index;
    }
    return null;
}

fn countMethodsByNameAndReceiver(
    methods: []const TraitMethodSignature,
    method_lookup: []const lookup_index.NamedEntry,
    name: []const u8,
    receiver_kind: ast.ReceiverKind,
) usize {
    const range = lookup_index.findNamedRange(method_lookup, name) orelse return 0;
    var count: usize = 0;
    for (method_lookup[range.start..range.end]) |entry| {
        if (entry.index >= methods.len) return count;
        if (methods[entry.index].receiver_kind == receiver_kind) count += 1;
    }
    return count;
}

pub const InstantiatedStructField = struct {
    name: []const u8,
    ty: Type,
};

pub const InstantiatedStruct = struct {
    template_item_id: ast.ItemId,
    mangled_name: []const u8,
    fields: []const InstantiatedStructField,
    field_lookup: []lookup_index.NamedEntry = &.{},

    pub fn fieldIndex(self: InstantiatedStruct, name: []const u8) ?usize {
        return lookup_index.findNamed(self.field_lookup, name);
    }

    pub fn fieldByName(self: InstantiatedStruct, name: []const u8) ?InstantiatedStructField {
        const index = self.fieldIndex(name) orelse return null;
        return self.fields[index];
    }
};

pub const InstantiatedEnum = struct {
    template_item_id: ast.ItemId,
    mangled_name: []const u8,
    repr_type: ?Type = null,
    variants: []const InstantiatedEnumVariant = &.{},
    variant_lookup: []lookup_index.NamedEntry = &.{},

    pub fn variantIndex(self: InstantiatedEnum, name: []const u8) ?usize {
        return lookup_index.findNamed(self.variant_lookup, name);
    }

    pub fn variantByName(self: InstantiatedEnum, name: []const u8) ?InstantiatedEnumVariant {
        const index = self.variantIndex(name) orelse return null;
        return self.variants[index];
    }
};

pub const InstantiatedEnumVariant = struct {
    name: []const u8,
    payload_type: ?Type = null,
    explicit_value: ?ExplicitEnumValue = null,
};

pub const ExplicitEnumValue = union(enum) {
    integer: BigInt,
    string: []const u8,
    bytes: []const u8,
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
    field_lookup: []lookup_index.NamedEntry = &.{},

    pub fn fieldIndex(self: InstantiatedBitfield, name: []const u8) ?usize {
        return lookup_index.findNamed(self.field_lookup, name);
    }

    pub fn fieldByName(self: InstantiatedBitfield, name: []const u8) ?InstantiatedBitfieldField {
        const index = self.fieldIndex(name) orelse return null;
        return self.fields[index];
    }
};

pub const Binding = union(enum) {
    item: ast.ItemId,
    pattern: ast.PatternId,
};

pub const ResolvedBinding = Binding;

pub const semantic_types = @import("ora_types").semantic;
pub const semantic_values = @import("ora_types").value;

pub const TypeKind = semantic_types.TypeKind;
pub const Region = semantic_types.Region;
pub const Provenance = semantic_types.Provenance;
pub const NamedType = semantic_types.NamedType;
pub const ExternalProxyType = semantic_types.ExternalProxyType;
pub const IntegerType = semantic_types.IntegerType;
pub const ComptimeIntegerType = semantic_types.ComptimeIntegerType;
pub const FixedBytesType = semantic_types.FixedBytesType;
pub const ResourceDomainType = semantic_types.ResourceDomainType;
pub const ResourcePlaceType = semantic_types.ResourcePlaceType;
pub const FunctionType = semantic_types.FunctionType;
pub const ArrayType = semantic_types.ArrayType;
pub const SliceType = semantic_types.SliceType;
pub const MapType = semantic_types.MapType;
pub const AnonymousStructField = semantic_types.AnonymousStructField;
pub const AnonymousStructType = semantic_types.AnonymousStructType;
pub const anonymousStructFieldIndex = semantic_types.anonymousStructFieldIndex;
pub const anonymousStructFieldByName = semantic_types.anonymousStructFieldByName;
pub const ErrorUnionType = semantic_types.ErrorUnionType;
pub const RefinementArg = semantic_types.RefinementArg;
pub const RefinementIntegerArg = semantic_types.RefinementIntegerArg;
pub const RefinementType = semantic_types.RefinementType;
pub const Type = semantic_types.Type;
pub const GenericBindingValue = semantic_types.GenericBindingValue;
pub const GenericTypeBinding = semantic_types.GenericTypeBinding;
pub const LocatedType = semantic_types.LocatedType;
pub const appendTypeMangleName = semantic_types.appendTypeMangleName;
pub const typeMangleNameLen = semantic_types.typeMangleNameLen;

pub const ResolvedCall = struct {
    module_id: source.ModuleId,
    item_id: ast.ItemId,
    generic_bindings: []const GenericTypeBinding = &.{},
    runtime_parameter_types: []const Type = &.{},
    return_type: Type = .{ .unknown = {} },
};

pub const ConstValue = semantic_values.ConstValue;

pub const VerificationFactKind = enum {
    requires,
    guard,
    ensures,
    ensures_ok,
    ensures_err,
    modifies,
    contract_invariant,
    loop_invariant,
    assert,
    assume,
    havoc,
    ghost_function,
    ghost_field,
    ghost_constant,
    ghost_block,
    ghost_axiom,
    old,
    quantified,

    pub fn fromSpecClause(kind: ast.SpecClauseKind) ?VerificationFactKind {
        return switch (kind) {
            .requires => .requires,
            .guard => .guard,
            .ensures => .ensures,
            .ensures_ok => .ensures_ok,
            .ensures_err => .ensures_err,
            .modifies => .modifies,
            .invariant => null,
        };
    }

    pub fn specClauseKind(self: VerificationFactKind) ?ast.SpecClauseKind {
        return switch (self) {
            .requires => .requires,
            .guard => .guard,
            .ensures => .ensures,
            .ensures_ok => .ensures_ok,
            .ensures_err => .ensures_err,
            .modifies => .modifies,
            .contract_invariant => .invariant,
            else => null,
        };
    }
};

pub const VerificationContext = enum {
    source,
    contract,
    loop,
    ghost_block,
    trait_ghost_block,
    ghost_declaration,
    trait_method_contract,
};

pub const VerificationTraitMethodOwner = struct {
    trait_item: ast.ItemId,
    method_index: usize,
};

pub const VerificationStatementOwner = struct {
    item: ast.ItemId,
    stmt: ast.StmtId,
};

pub const VerificationFactOwner = union(enum) {
    none,
    item: ast.ItemId,
    trait_method: VerificationTraitMethodOwner,
    statement: VerificationStatementOwner,

    pub fn itemId(self: VerificationFactOwner) ?ast.ItemId {
        return switch (self) {
            .item => |item_id| item_id,
            else => null,
        };
    }

    pub fn traitMethod(self: VerificationFactOwner) ?VerificationTraitMethodOwner {
        return switch (self) {
            .trait_method => |owner| owner,
            else => null,
        };
    }

    pub fn statementOwner(self: VerificationFactOwner) ?VerificationStatementOwner {
        return switch (self) {
            .statement => |owner| owner,
            else => null,
        };
    }
};

pub const VerificationFact = struct {
    kind: VerificationFactKind,
    owner: VerificationFactOwner = .none,
    expr: ?ast.ExprId = null,
    label: ?[]const u8 = null,
    target_name: ?[]const u8 = null,
    range: source.TextRange,
    context: VerificationContext = .source,
};

pub const EffectSlot = struct {
    name: []const u8,
    region: Region,
    field_path: ?[]const []const u8 = null,
    key_path: ?[]const KeySegment = null,
};

pub const KeySegment = union(enum) {
    parameter: u32,
    comptime_parameter: u32,
    comptime_range_parameter: u32,
    constant: []const u8,
    msg_sender,
    tx_origin,
    unknown,
};

pub fn formatEffectSlotPath(allocator: std.mem.Allocator, slot: EffectSlot) ![]u8 {
    var buffer = std.Io.Writer.Allocating.init(allocator);
    errdefer buffer.deinit();

    const writer = &buffer.writer;
    try writer.writeAll(slot.name);
    if (slot.field_path) |field_path| {
        for (field_path) |field_name| {
            try writer.writeByte('.');
            try writer.writeAll(field_name);
        }
    }
    if (slot.key_path) |key_path| {
        for (key_path) |segment| {
            try writer.writeByte('[');
            switch (segment) {
                .parameter => |index| try writer.print("param#{d}", .{index}),
                .comptime_parameter => |index| try writer.print("comptime_param#{d}", .{index}),
                .comptime_range_parameter => |index| try writer.print("comptime_range_param#{d}", .{index}),
                .constant => |value| try writer.writeAll(value),
                .msg_sender => try writer.writeAll("msg.sender"),
                .tx_origin => try writer.writeAll("tx.origin"),
                .unknown => try writer.writeAll("?"),
            }
            try writer.writeByte(']');
        }
    }

    return try buffer.toOwnedSlice();
}

pub fn effectSlotPathRoot(path: []const u8) []const u8 {
    // Keep this parser aligned with z3.encoder.effectSlotPathRoot. The encoder
    // has a local copy because it is compiled as a standalone z3 module in
    // encoder-only tests.
    for (path, 0..) |byte, idx| {
        if (byte == '.' or byte == '[') return path[0..idx];
    }
    return path;
}

pub const EffectFlags = struct {
    has_external: bool = false,
    has_log: bool = false,
    has_havoc: bool = false,
    has_lock: bool = false,
    has_unlock: bool = false,

    pub fn any(self: EffectFlags) bool {
        return self.has_external or self.has_log or self.has_havoc or self.has_lock or self.has_unlock;
    }

    pub fn externalOnly(self: EffectFlags) bool {
        return self.has_external and !self.has_log and !self.has_havoc and !self.has_lock and !self.has_unlock;
    }

    pub fn merge(self: *EffectFlags, other: EffectFlags) void {
        self.has_external = self.has_external or other.has_external;
        self.has_log = self.has_log or other.has_log;
        self.has_havoc = self.has_havoc or other.has_havoc;
        self.has_lock = self.has_lock or other.has_lock;
        self.has_unlock = self.has_unlock or other.has_unlock;
    }
};

pub const Effect = union(enum) {
    pure,
    external,
    side_effects: EffectFlags,
    writes: struct {
        slots: []const EffectSlot,
        flags: EffectFlags = .{},
    },
    reads: struct {
        slots: []const EffectSlot,
        flags: EffectFlags = .{},
    },
    reads_writes: struct {
        reads: []const EffectSlot,
        writes: []const EffectSlot,
        flags: EffectFlags = .{},
    },

    pub fn readSlots(self: Effect) []const EffectSlot {
        return switch (self) {
            .pure, .external, .side_effects, .writes => &.{},
            .reads => |read_effect| read_effect.slots,
            .reads_writes => |read_write| read_write.reads,
        };
    }

    pub fn writeSlots(self: Effect) []const EffectSlot {
        return switch (self) {
            .pure, .external, .side_effects, .reads => &.{},
            .writes => |write_effect| write_effect.slots,
            .reads_writes => |read_write| read_write.writes,
        };
    }

    pub fn flags(self: Effect) EffectFlags {
        return switch (self) {
            .pure => .{},
            .external => .{ .has_external = true },
            .side_effects => |effect_flags| effect_flags,
            .reads => |read_effect| read_effect.flags,
            .writes => |write_effect| write_effect.flags,
            .reads_writes => |read_write| read_write.flags,
        };
    }

    pub fn hasExternal(self: Effect) bool {
        return self.flags().has_external;
    }

    pub fn hasLog(self: Effect) bool {
        return self.flags().has_log;
    }

    pub fn hasHavoc(self: Effect) bool {
        return self.flags().has_havoc;
    }

    pub fn hasLock(self: Effect) bool {
        return self.flags().has_lock;
    }

    pub fn hasUnlock(self: Effect) bool {
        return self.flags().has_unlock;
    }
};

pub const ExprEffect = struct {
    expr_id: ast.ExprId,
    effect: Effect,
};

pub const TypeCheckKey = union(enum) {
    item: ast.ItemId,
    body: ast.BodyId,
};

pub const VerificationFactsKey = union(enum) {
    item: ast.ItemId,
    body: ast.BodyId,
    trait_method: VerificationTraitMethodOwner,
    statement: VerificationStatementOwner,
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
    impl_lookup: []lookup_index.PairEntry,
    trait_method_lookup: []lookup_index.MemberEntry,
    impl_method_lookup: []lookup_index.MemberEntry,
    impl_method_owner_lookup: []lookup_index.IndexEntry,
    struct_field_lookup: []lookup_index.MemberEntry,
    bitfield_field_lookup: []lookup_index.MemberEntry,
    enum_variant_lookup: []lookup_index.MemberEntry,
    contract_member_lookup: []lookup_index.MemberEntry,

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
        const index = lookup_index.findPair(self.impl_lookup, trait_name, target_name) orelse return null;
        return self.impl_entries[index].item_id;
    }

    pub fn lookupTraitMethod(self: *const ItemIndexResult, file: *const ast.AstFile, trait_item_id: ast.ItemId, name: []const u8) ?ast.nodes.TraitMethod {
        const method_index = self.lookupTraitMethodIndex(trait_item_id, name) orelse return null;
        const trait_item = switch (file.item(trait_item_id).*) {
            .Trait => |trait_item| trait_item,
            else => return null,
        };
        if (method_index >= trait_item.methods.len) return null;
        return trait_item.methods[method_index];
    }

    pub fn lookupTraitMethodIndex(self: *const ItemIndexResult, trait_item_id: ast.ItemId, name: []const u8) ?usize {
        return lookup_index.findMember(self.trait_method_lookup, trait_item_id.index(), name);
    }

    pub fn lookupImplMethod(self: *const ItemIndexResult, file: *const ast.AstFile, impl_item_id: ast.ItemId, name: []const u8) ?ast.ItemId {
        const method_index = lookup_index.findMember(self.impl_method_lookup, impl_item_id.index(), name) orelse return null;
        return implMethodAt(file, impl_item_id, method_index);
    }

    pub fn lookupImplContainingMethod(self: *const ItemIndexResult, method_item_id: ast.ItemId) ?ast.ItemId {
        const impl_index = lookup_index.findIndex(self.impl_method_owner_lookup, method_item_id.index()) orelse return null;
        return ast.ItemId.fromIndex(impl_index);
    }

    pub fn lookupStructFieldIndex(self: *const ItemIndexResult, struct_item_id: ast.ItemId, name: []const u8) ?usize {
        return lookup_index.findMember(self.struct_field_lookup, struct_item_id.index(), name);
    }

    pub fn lookupStructField(self: *const ItemIndexResult, file: *const ast.AstFile, struct_item_id: ast.ItemId, name: []const u8) ?ast.StructField {
        const field_index = self.lookupStructFieldIndex(struct_item_id, name) orelse return null;
        const struct_item = switch (file.item(struct_item_id).*) {
            .Struct => |struct_item| struct_item,
            else => return null,
        };
        if (field_index >= struct_item.fields.len) return null;
        return struct_item.fields[field_index];
    }

    pub fn lookupBitfieldFieldIndex(self: *const ItemIndexResult, bitfield_item_id: ast.ItemId, name: []const u8) ?usize {
        return lookup_index.findMember(self.bitfield_field_lookup, bitfield_item_id.index(), name);
    }

    pub fn lookupBitfieldField(self: *const ItemIndexResult, file: *const ast.AstFile, bitfield_item_id: ast.ItemId, name: []const u8) ?ast.BitfieldField {
        const field_index = self.lookupBitfieldFieldIndex(bitfield_item_id, name) orelse return null;
        const bitfield_item = switch (file.item(bitfield_item_id).*) {
            .Bitfield => |bitfield_item| bitfield_item,
            else => return null,
        };
        if (field_index >= bitfield_item.fields.len) return null;
        return bitfield_item.fields[field_index];
    }

    pub fn countImplMethods(self: *const ItemIndexResult, file: *const ast.AstFile, impl_item_id: ast.ItemId, name: []const u8) usize {
        const range = lookup_index.findMemberRange(self.impl_method_lookup, impl_item_id.index(), name) orelse return 0;
        var count: usize = 0;
        for (self.impl_method_lookup[range.start..range.end]) |entry| {
            if (implMethodAt(file, impl_item_id, entry.index) != null) count += 1;
        }
        return count;
    }

    pub fn lookupImplMethodByReceiver(
        self: *const ItemIndexResult,
        file: *const ast.AstFile,
        impl_item_id: ast.ItemId,
        name: []const u8,
        receiver_kind: ast.ReceiverKind,
    ) ?ast.ItemId {
        const range = lookup_index.findMemberRange(self.impl_method_lookup, impl_item_id.index(), name) orelse return null;
        for (self.impl_method_lookup[range.start..range.end]) |entry| {
            const method_id = implMethodAt(file, impl_item_id, entry.index) orelse continue;
            const function = switch (file.item(method_id).*) {
                .Function => |function| function,
                else => continue,
            };
            if (functionReceiverKind(file, function) == receiver_kind) return method_id;
        }
        return null;
    }

    pub fn countImplMethodsByReceiver(
        self: *const ItemIndexResult,
        file: *const ast.AstFile,
        impl_item_id: ast.ItemId,
        name: []const u8,
        receiver_kind: ast.ReceiverKind,
    ) usize {
        const range = lookup_index.findMemberRange(self.impl_method_lookup, impl_item_id.index(), name) orelse return 0;
        var count: usize = 0;
        for (self.impl_method_lookup[range.start..range.end]) |entry| {
            const method_id = implMethodAt(file, impl_item_id, entry.index) orelse continue;
            const function = switch (file.item(method_id).*) {
                .Function => |function| function,
                else => continue,
            };
            if (functionReceiverKind(file, function) == receiver_kind) count += 1;
        }
        return count;
    }

    pub fn lookupEnumVariantIndex(self: *const ItemIndexResult, enum_item_id: ast.ItemId, name: []const u8) ?usize {
        return lookup_index.findMember(self.enum_variant_lookup, enum_item_id.index(), name);
    }

    pub fn lookupContractMemberWithRoles(
        self: *const ItemIndexResult,
        file: *const ast.AstFile,
        contract_item_id: ast.ItemId,
        name: []const u8,
        roles: ContractMemberRoles,
    ) ?ast.ItemId {
        const range = lookup_index.findMemberRange(self.contract_member_lookup, contract_item_id.index(), name) orelse return null;
        const contract = switch (file.item(contract_item_id).*) {
            .Contract => |contract| contract,
            else => return null,
        };
        for (self.contract_member_lookup[range.start..range.end]) |entry| {
            if (entry.index >= contract.members.len) return null;
            const member_id = contract.members[entry.index];
            const role = contractMemberRole(file.item(member_id).*) orelse continue;
            if (roles.contains(role)) return member_id;
        }
        return null;
    }
};

fn implMethodAt(file: *const ast.AstFile, impl_item_id: ast.ItemId, method_index: usize) ?ast.ItemId {
    const impl_item = switch (file.item(impl_item_id).*) {
        .Impl => |impl_item| impl_item,
        else => return null,
    };
    if (method_index >= impl_item.methods.len) return null;
    return impl_item.methods[method_index];
}

fn functionReceiverKind(file: *const ast.AstFile, function: ast.FunctionItem) ast.ReceiverKind {
    return if (functionHasRuntimeSelf(file, function)) .value_self else .none;
}

fn contractMemberRole(item: ast.Item) ?ContractMemberRole {
    return switch (item) {
        .Field => .field,
        .Constant => .constant,
        .Function => .function,
        .Struct => .struct_,
        .Bitfield => .bitfield,
        .Enum => .enum_,
        .Resource => .resource,
        .TypeAlias => .type_alias,
        .Trait => .trait_,
        .LogDecl => .log_decl,
        .ErrorDecl => .error_decl,
        else => null,
    };
}

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
    item_modifies: []?[]EffectSlot,
    pattern_types: []LocatedType,
    pattern_initializers: []?ast.ExprId,
    pattern_binding_kinds: []?ast.BindingKind,
    expr_types: []Type,
    call_resolutions: []?ResolvedCall,
    expr_effects: []const ExprEffect,
    body_types: []Type,
    instantiated_structs: []const InstantiatedStruct,
    instantiated_struct_lookup: []lookup_index.NamedEntry,
    instantiated_enums: []const InstantiatedEnum,
    instantiated_enum_lookup: []lookup_index.NamedEntry,
    instantiated_bitfields: []const InstantiatedBitfield,
    instantiated_bitfield_lookup: []lookup_index.NamedEntry,
    trait_interfaces: []const TraitInterface,
    trait_interface_lookup: []lookup_index.NamedEntry,
    impl_interfaces: []const ImplInterface,
    impl_interface_lookup: []lookup_index.PairEntry,
    diagnostics: diagnostics.DiagnosticList,

    pub fn deinit(self: *TypeCheckResult) void {
        self.diagnostics.deinit();
        self.arena.deinit();
    }

    pub fn exprType(self: *const TypeCheckResult, id: ast.ExprId) Type {
        return self.expr_types[id.index()];
    }

    pub fn patternInitializer(self: *const TypeCheckResult, id: ast.PatternId) ?ast.ExprId {
        if (id.index() >= self.pattern_initializers.len) return null;
        return self.pattern_initializers[id.index()];
    }

    pub fn patternBindingKind(self: *const TypeCheckResult, id: ast.PatternId) ?ast.BindingKind {
        if (id.index() >= self.pattern_binding_kinds.len) return null;
        return self.pattern_binding_kinds[id.index()];
    }

    pub fn exprCallResolution(self: *const TypeCheckResult, id: ast.ExprId) ?ResolvedCall {
        if (id.index() >= self.call_resolutions.len) return null;
        return self.call_resolutions[id.index()];
    }

    pub fn itemLocatedType(self: *const TypeCheckResult, id: ast.ItemId) LocatedType {
        return .{
            .type = self.item_types[id.index()],
            .region = self.item_regions[id.index()],
        };
    }

    pub fn exprEffect(self: *const TypeCheckResult, id: ast.ExprId) Effect {
        for (self.expr_effects) |entry| {
            if (entry.expr_id == id) return entry.effect;
        }
        return .pure;
    }

    pub fn itemEffect(self: *const TypeCheckResult, id: ast.ItemId) Effect {
        return self.item_effects[id.index()];
    }

    pub fn itemModifies(self: *const TypeCheckResult, id: ast.ItemId) ?[]EffectSlot {
        return self.item_modifies[id.index()];
    }

    pub fn instantiatedStructByName(self: *const TypeCheckResult, name: []const u8) ?InstantiatedStruct {
        const index = lookup_index.findNamed(self.instantiated_struct_lookup, name) orelse return null;
        return self.instantiated_structs[index];
    }

    pub fn instantiatedEnumByName(self: *const TypeCheckResult, name: []const u8) ?InstantiatedEnum {
        const index = lookup_index.findNamed(self.instantiated_enum_lookup, name) orelse return null;
        return self.instantiated_enums[index];
    }

    pub fn instantiatedBitfieldByName(self: *const TypeCheckResult, name: []const u8) ?InstantiatedBitfield {
        const index = lookup_index.findNamed(self.instantiated_bitfield_lookup, name) orelse return null;
        return self.instantiated_bitfields[index];
    }

    pub fn traitInterfaceByName(self: *const TypeCheckResult, name: []const u8) ?TraitInterface {
        const index = lookup_index.findNamed(self.trait_interface_lookup, name) orelse return null;
        return self.trait_interfaces[index];
    }

    pub fn implInterfaceByNames(self: *const TypeCheckResult, trait_name: []const u8, target_name: []const u8) ?ImplInterface {
        const index = lookup_index.findPair(self.impl_interface_lookup, trait_name, target_name) orelse return null;
        return self.impl_interfaces[index];
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

test "IntegerType is resolved by construction" {
    const u256_integer = IntegerType{ .bits = 256, .signed = false, .spelling = "u256" };

    try std.testing.expectEqual(@as(u16, 256), u256_integer.bits);
    try std.testing.expect(!u256_integer.signed);
    try std.testing.expect(u256_integer.isUnsignedBits(256));
    try std.testing.expect(u256_integer.builtinSpec() != null);
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

test "EffectSlot path formatting includes fields and keys" {
    const fields = [_][]const u8{"owner"};
    const keys = [_]KeySegment{
        .{ .parameter = 0 },
        .msg_sender,
        .{ .constant = "42" },
    };
    const slot = EffectSlot{
        .name = "allowances",
        .region = .storage,
        .field_path = &fields,
        .key_path = &keys,
    };

    const path = try formatEffectSlotPath(std.testing.allocator, slot);
    defer std.testing.allocator.free(path);

    try std.testing.expectEqualStrings("allowances.owner[param#0][msg.sender][42]", path);
    try std.testing.expectEqualStrings("allowances", effectSlotPathRoot(path));
}

test "Effect supports external call marker" {
    const slots = [_]EffectSlot{
        .{ .name = "total", .region = .storage },
    };
    const external_only: Effect = .external;
    const reads_external: Effect = .{ .reads = .{
        .slots = &slots,
        .flags = .{ .has_external = true },
    } };
    const writes_external: Effect = .{ .writes = .{
        .slots = &slots,
        .flags = .{ .has_external = true },
    } };
    const mixed_external: Effect = .{ .reads_writes = .{
        .reads = &slots,
        .writes = &slots,
        .flags = .{ .has_external = true },
    } };

    try std.testing.expect(external_only == .external);
    try std.testing.expect(external_only.hasExternal());
    try std.testing.expect(reads_external.hasExternal());
    try std.testing.expect(writes_external.hasExternal());
    try std.testing.expect(mixed_external.hasExternal());
}

test "Effect supports log and havoc markers" {
    const slots = [_]EffectSlot{
        .{ .name = "total", .region = .storage },
    };
    const reads_log: Effect = .{ .reads = .{
        .slots = &slots,
        .flags = .{ .has_log = true },
    } };
    const writes_havoc: Effect = .{ .writes = .{
        .slots = &slots,
        .flags = .{ .has_havoc = true },
    } };
    const mixed: Effect = .{ .reads_writes = .{
        .reads = &slots,
        .writes = &slots,
        .flags = .{
            .has_log = true,
            .has_havoc = true,
        },
    } };

    try std.testing.expect(reads_log.hasLog());
    try std.testing.expect(writes_havoc.hasHavoc());
    try std.testing.expect(mixed.hasLog());
    try std.testing.expect(mixed.hasHavoc());
}

test "Effect supports side-effect-only marker" {
    const effect: Effect = .{ .side_effects = .{
        .has_log = true,
        .has_havoc = true,
        .has_lock = true,
        .has_unlock = true,
    } };

    try std.testing.expect(effect.hasLog());
    try std.testing.expect(effect.hasHavoc());
    try std.testing.expect(effect.hasLock());
    try std.testing.expect(effect.hasUnlock());
}
