// ============================================================================
// ABI Support (Ora ABI v0.1 + Solidity projection)
// ============================================================================

const std = @import("std");
const crypto = std.crypto;
const compiler = @import("compiler.zig");
const compiler_abi = @import("hir/abi.zig");
const compiler_type_descriptors = @import("sema/type_descriptors.zig");

const ProfileId = "evm-default";
const SchemaVersion = "ora-abi-0.1";
const ExtrasSchemaVersion = "ora-abi-extras-0.1";
const CompilerName = "ora";
const CompilerVersion = "0.1.0";

pub const AbiError = std.mem.Allocator.Error || error{
    MissingContract,
    DuplicateCallableId,
    TypeHashCollision,
    UnsupportedAbiType,
    UnknownStructType,
    UnknownEnumType,
    RecursiveType,
    InvalidEnumRepr,
    UnresolvedType,
};

pub const AbiEffectKind = enum {
    reads,
    writes,
    emits,
    calls,
    value,
};

pub const AbiEffect = struct {
    kind: AbiEffectKind,
    path: ?[]const u8 = null,
    event_id: ?[]const u8 = null,
};

pub const AbiInput = struct {
    name: []const u8,
    type_id: []const u8,
    indexed: ?bool = null,
};

pub const CallableKind = enum {
    function,
    constructor,
    @"error",
    event,
};

pub const AbiCallable = struct {
    id: []const u8,
    kind: CallableKind,
    name: []const u8,
    signature: []const u8,
    selector: ?[]const u8 = null,
    inputs: []AbiInput,
    outputs: []AbiInput,
    effects: []AbiEffect,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *AbiCallable) void {
        self.allocator.free(self.id);
        self.allocator.free(self.signature);
        if (self.selector) |selector| {
            self.allocator.free(selector);
        }

        for (self.effects) |effect| {
            if (effect.path) |path| self.allocator.free(path);
            if (effect.event_id) |event_id| self.allocator.free(event_id);
        }

        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.free(self.effects);
    }
};

const TypeNodeKind = enum {
    primitive,
    array,
    slice,
    tuple,
    struct_,
    enum_,
    alias,
    refinement,
};

const AbiFieldRef = struct {
    name: []const u8,
    type_id: []const u8,
};

const AbiEnumVariant = struct {
    name: []const u8,
    value: i64,
};

pub const AbiTypeNode = struct {
    kind: TypeNodeKind,
    allocator: std.mem.Allocator,

    type_id: ?[]const u8 = null,
    canonical_payload: ?[]const u8 = null,

    // common
    name: ?[]const u8 = null,
    wire_type: ?[]const u8 = null,

    // array/slice
    element_type: ?[]const u8 = null,
    size: ?u64 = null,

    // tuple
    components: []const []const u8 = &.{},

    // struct
    fields: []AbiFieldRef = &.{},

    // enum
    repr_type: ?[]const u8 = null,
    variants: []AbiEnumVariant = &.{},

    // refinement
    base: ?[]const u8 = null,
    predicate_json: ?[]const u8 = null,

    // frontend-oriented metadata hints (emitted under meta.ui)
    ui_label: ?[]const u8 = null,
    ui_widget: ?[]const u8 = null,
    ui_min: ?[]const u8 = null,
    ui_max: ?[]const u8 = null,
    ui_decimals: ?u32 = null,
    ui_unit: ?[]const u8 = null,

    pub fn deinit(self: *AbiTypeNode) void {
        if (self.type_id) |type_id| self.allocator.free(type_id);
        if (self.canonical_payload) |payload| self.allocator.free(payload);
        if (self.wire_type) |wire_type| self.allocator.free(wire_type);
        if (self.predicate_json) |predicate| self.allocator.free(predicate);
        if (self.ui_min) |ui_min| self.allocator.free(ui_min);
        if (self.ui_max) |ui_max| self.allocator.free(ui_max);

        if (self.components.len > 0) self.allocator.free(self.components);
        if (self.fields.len > 0) self.allocator.free(self.fields);
        if (self.variants.len > 0) self.allocator.free(self.variants);
    }
};

pub const ContractAbi = struct {
    allocator: std.mem.Allocator,
    contract_name: []const u8,
    contract_count: usize,
    callables: []AbiCallable,
    types: []AbiTypeNode,
    type_lookup: std.StringHashMap(usize),

    pub fn deinit(self: *ContractAbi) void {
        for (self.callables) |*callable| {
            callable.deinit();
        }
        self.allocator.free(self.callables);

        for (self.types) |*typ| {
            typ.deinit();
        }
        self.allocator.free(self.types);

        self.type_lookup.deinit();
    }

    pub fn findType(self: *const ContractAbi, type_id: []const u8) ?*const AbiTypeNode {
        if (self.type_lookup.get(type_id)) |idx| {
            return &self.types[idx];
        }
        return null;
    }

    fn writeTypeRef(writer: anytype, type_id: []const u8) !void {
        try writer.writeByte('{');
        try writeJsonString(writer, "name");
        try writer.writeByte(':');
        try writeJsonString(writer, "");
        try writer.writeByte(',');
        try writeJsonString(writer, "typeId");
        try writer.writeByte(':');
        try writeJsonString(writer, type_id);
        try writer.writeByte('}');
    }

    fn writeCallableInput(writer: anytype, input: AbiInput, include_indexed: bool) !void {
        try writer.writeByte('{');
        try writeJsonString(writer, "name");
        try writer.writeByte(':');
        try writeJsonString(writer, input.name);
        try writer.writeByte(',');
        try writeJsonString(writer, "typeId");
        try writer.writeByte(':');
        try writeJsonString(writer, input.type_id);
        if (include_indexed) {
            try writer.writeByte(',');
            try writeJsonString(writer, "indexed");
            try writer.writeByte(':');
            try writer.writeAll(if (input.indexed orelse false) "true" else "false");
        }
        try writer.writeByte('}');
    }

    fn writeCallable(self: *const ContractAbi, writer: anytype, callable: AbiCallable) !void {
        _ = self;
        try writer.writeByte('{');

        var first = true;
        try writeObjectStringField(writer, &first, "id", callable.id);
        try writeObjectStringField(writer, &first, "kind", callableKindString(callable.kind));
        try writeObjectStringField(writer, &first, "name", callable.name);
        try writeObjectStringField(writer, &first, "signature", callable.signature);

        if (callable.selector) |selector| {
            try writeObjectKey(writer, &first, "wire");
            try writer.writeByte('{');
            try writeJsonString(writer, ProfileId);
            try writer.writeByte(':');
            try writer.writeByte('{');
            try writeJsonString(writer, "selector");
            try writer.writeByte(':');
            try writeJsonString(writer, selector);
            try writer.writeByte('}');
            try writer.writeByte('}');
        }

        try writeObjectKey(writer, &first, "inputs");
        try writer.writeByte('[');
        for (callable.inputs, 0..) |input, i| {
            if (i > 0) try writer.writeByte(',');
            try writeCallableInput(writer, input, callable.kind == .event);
        }
        try writer.writeByte(']');

        if (callable.kind == .function) {
            try writeObjectKey(writer, &first, "outputs");
            try writer.writeByte('[');
            for (callable.outputs, 0..) |output, i| {
                if (i > 0) try writer.writeByte(',');
                try ContractAbi.writeTypeRef(writer, output.type_id);
            }
            try writer.writeByte(']');
        }

        if (callable.effects.len > 0) {
            try writeObjectKey(writer, &first, "meta");
            try writer.writeByte('{');
            try writeJsonString(writer, "effects");
            try writer.writeByte(':');
            try writer.writeByte('[');
            for (callable.effects, 0..) |effect, i| {
                if (i > 0) try writer.writeByte(',');
                try writer.writeByte('{');
                var effect_first = true;
                try writeObjectStringField(writer, &effect_first, "kind", effectKindString(effect.kind));
                if (effect.path) |path| {
                    try writeObjectStringField(writer, &effect_first, "path", path);
                }
                if (effect.event_id) |event_id| {
                    try writeObjectStringField(writer, &effect_first, "eventId", event_id);
                }
                try writer.writeByte('}');
            }
            try writer.writeByte(']');
            try writer.writeByte('}');
        }

        try writer.writeByte('}');
    }

    fn callableHasUiMeta(self: *const ContractAbi, callable: AbiCallable) bool {
        _ = self;
        return switch (callable.kind) {
            .function => true,
            .constructor => true,
            .event => true,
            .@"error" => true,
        };
    }

    fn writeCallableUi(self: *const ContractAbi, writer: anytype, callable: AbiCallable) !void {
        try writer.writeByte('{');
        var ui_first = true;

        switch (callable.kind) {
            .function => {
                const mutability = projectStateMutability(callable);
                const group = if (std.mem.eql(u8, mutability, "pure") or std.mem.eql(u8, mutability, "view"))
                    "Read"
                else
                    "Write";
                const danger_level = if (std.mem.eql(u8, mutability, "pure") or std.mem.eql(u8, mutability, "view"))
                    "info"
                else if (std.mem.eql(u8, mutability, "payable"))
                    "warning"
                else
                    "normal";

                try writeObjectStringField(writer, &ui_first, "group", group);
                try writeObjectStringField(writer, &ui_first, "dangerLevel", danger_level);
                try self.writeCallableUiForms(writer, &ui_first, callable.inputs);
            },
            .constructor => {
                const mutability = projectStateMutability(callable);
                const danger_level = if (std.mem.eql(u8, mutability, "payable"))
                    "warning"
                else
                    "normal";

                try writeObjectStringField(writer, &ui_first, "group", "Deploy");
                try writeObjectStringField(writer, &ui_first, "dangerLevel", danger_level);
                try self.writeCallableUiForms(writer, &ui_first, callable.inputs);
            },
            .event => {
                try writeObjectStringField(writer, &ui_first, "group", "Events");
            },
            .@"error" => {
                try writeObjectStringField(writer, &ui_first, "group", "Errors");
                try writeObjectStringField(writer, &ui_first, "dangerLevel", "warning");
                const template = try self.defaultErrorMessageTemplate(self.allocator, callable);
                defer self.allocator.free(template);
                try writeObjectStringField(writer, &ui_first, "messageTemplate", template);
            },
        }

        try writer.writeByte('}');
    }

    fn writeCallableUiForms(
        self: *const ContractAbi,
        writer: anytype,
        ui_first: *bool,
        inputs: []AbiInput,
    ) !void {
        var has_forms = false;
        for (inputs) |input| {
            const hint = self.uiHintForType(input.type_id);
            if (hint.hasAny()) {
                has_forms = true;
                break;
            }
        }

        if (!has_forms) return;

        try writeObjectKey(writer, ui_first, "forms");
        try writer.writeByte('{');
        var first_form = true;
        for (inputs) |input| {
            const hint = self.uiHintForType(input.type_id);
            if (!hint.hasAny()) continue;

            if (!first_form) try writer.writeByte(',');
            first_form = false;
            try writeJsonString(writer, input.name);
            try writer.writeByte(':');
            try writeUiFieldHint(writer, hint);
        }
        try writer.writeByte('}');
    }

    fn defaultErrorMessageTemplate(self: *const ContractAbi, allocator: std.mem.Allocator, callable: AbiCallable) ![]const u8 {
        _ = self;
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);

        try writer.writeAll(callable.name);

        if (callable.inputs.len > 0) {
            try writer.writeAll(": ");
        }

        for (callable.inputs, 0..) |input, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.writeAll(input.name);
            try writer.writeAll("={");
            try writer.writeAll(input.name);
            try writer.writeAll("}");
        }

        return buffer.toOwnedSlice(allocator);
    }

    fn uiHintForType(self: *const ContractAbi, type_id: []const u8) UiFieldHint {
        const node = self.findType(type_id) orelse return .{};

        switch (node.kind) {
            .primitive => {
                return .{
                    .widget = node.ui_widget,
                    .min = node.ui_min,
                    .max = node.ui_max,
                    .decimals = node.ui_decimals,
                    .unit = node.ui_unit,
                };
            },
            .enum_ => {
                return .{
                    .widget = node.ui_widget orelse "select",
                };
            },
            .refinement => {
                var hint = UiFieldHint{
                    .widget = node.ui_widget,
                    .min = node.ui_min,
                    .max = node.ui_max,
                    .decimals = node.ui_decimals,
                    .unit = node.ui_unit,
                };

                if (hint.widget == null) {
                    if (node.base) |base_id| {
                        const base_hint = self.uiHintForType(base_id);
                        hint.widget = base_hint.widget;
                    }
                }
                return hint;
            },
            .array, .slice, .tuple, .struct_ => {
                return .{ .widget = "json" };
            },
            else => return .{},
        }
    }

    pub fn toJson(self: *const ContractAbi, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = try std.ArrayList(u8).initCapacity(allocator, 4096);
        defer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);

        try writer.writeByte('{');

        var first = true;
        try writeObjectStringField(writer, &first, "schemaVersion", SchemaVersion);

        try writeObjectKey(writer, &first, "contract");
        try writer.writeByte('{');
        var contract_first = true;
        try writeObjectStringField(writer, &contract_first, "name", self.contract_name);
        try writeObjectKey(writer, &contract_first, "build");
        try writer.writeByte('{');
        var build_first = true;
        try writeObjectStringField(writer, &build_first, "compiler", CompilerName);
        try writeObjectStringField(writer, &build_first, "version", CompilerVersion);
        try writer.writeByte('}');
        try writer.writeByte('}');

        try writeObjectKey(writer, &first, "wireProfiles");
        try writer.writeByte('[');
        try writer.writeByte('{');
        var profile_first = true;
        try writeObjectStringField(writer, &profile_first, "id", ProfileId);
        try writeObjectStringField(writer, &profile_first, "kind", "evm");
        try writeObjectStringField(writer, &profile_first, "encoding", "abi-v2");
        try writer.writeByte('}');
        try writer.writeByte(']');

        try writeObjectKey(writer, &first, "types");
        try writer.writeByte('{');
        if (self.types.len > 0) {
            var sorted_ids = try allocator.alloc([]const u8, self.types.len);
            defer allocator.free(sorted_ids);
            for (self.types, 0..) |typ, i| {
                sorted_ids[i] = typ.type_id.?;
            }
            sortStringSlices(sorted_ids);

            for (sorted_ids, 0..) |type_id, i| {
                if (i > 0) try writer.writeByte(',');
                try writeJsonString(writer, type_id);
                try writer.writeByte(':');
                const idx = self.type_lookup.get(type_id).?;
                try writeTypeNodeObject(writer, &self.types[idx], true);
            }
        }
        try writer.writeByte('}');

        try writeObjectKey(writer, &first, "callables");
        try writer.writeByte('[');
        for (self.callables, 0..) |callable, i| {
            if (i > 0) try writer.writeByte(',');
            try self.writeCallable(writer, callable);
        }
        try writer.writeByte(']');

        try writer.writeByte('}');
        return buffer.toOwnedSlice(allocator);
    }

    fn writeCallableExtras(self: *const ContractAbi, writer: anytype, callable: AbiCallable) !void {
        try writer.writeByte('{');
        var first = true;
        try writeObjectKey(writer, &first, "ui");
        try self.writeCallableUi(writer, callable);
        try writer.writeByte('}');
    }

    fn writeTypeExtras(writer: anytype, node: *const AbiTypeNode) !void {
        try writer.writeByte('{');
        var first = true;
        try writeObjectKey(writer, &first, "ui");
        try writeTypeUiObject(writer, node);
        try writer.writeByte('}');
    }

    pub fn toExtrasJson(self: *const ContractAbi, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = try std.ArrayList(u8).initCapacity(allocator, 2048);
        defer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);

        try writer.writeByte('{');
        var first = true;

        try writeObjectStringField(writer, &first, "schemaVersion", ExtrasSchemaVersion);
        try writeObjectStringField(writer, &first, "baseSchemaVersion", SchemaVersion);

        try writeObjectKey(writer, &first, "contract");
        try writer.writeByte('{');
        var contract_first = true;
        try writeObjectStringField(writer, &contract_first, "name", self.contract_name);
        try writer.writeByte('}');

        try writeObjectKey(writer, &first, "types");
        try writer.writeByte('{');
        if (self.types.len > 0) {
            var sorted_ids = try allocator.alloc([]const u8, self.types.len);
            defer allocator.free(sorted_ids);
            for (self.types, 0..) |typ, i| {
                sorted_ids[i] = typ.type_id.?;
            }
            sortStringSlices(sorted_ids);

            var emitted_types: usize = 0;
            for (sorted_ids) |type_id| {
                const idx = self.type_lookup.get(type_id).?;
                const typ = &self.types[idx];
                if (!typeHasUiMeta(typ)) continue;

                if (emitted_types > 0) try writer.writeByte(',');
                emitted_types += 1;

                try writeJsonString(writer, type_id);
                try writer.writeByte(':');
                try ContractAbi.writeTypeExtras(writer, typ);
            }
        }
        try writer.writeByte('}');

        try writeObjectKey(writer, &first, "callables");
        try writer.writeByte('{');
        var emitted_callables: usize = 0;
        for (self.callables) |callable| {
            if (!self.callableHasUiMeta(callable)) continue;
            if (emitted_callables > 0) try writer.writeByte(',');
            emitted_callables += 1;
            try writeJsonString(writer, callable.id);
            try writer.writeByte(':');
            try self.writeCallableExtras(writer, callable);
        }
        try writer.writeByte('}');

        try writer.writeByte('}');
        return buffer.toOwnedSlice(allocator);
    }

    pub fn toSolidityJson(self: *const ContractAbi, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = try std.ArrayList(u8).initCapacity(allocator, 2048);
        defer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);

        try writer.writeByte('[');
        var emitted: usize = 0;

        for (self.callables) |callable| {
            if (emitted > 0) try writer.writeByte(',');
            emitted += 1;

            try writer.writeByte('{');
            var first = true;

            try writeObjectStringField(writer, &first, "type", callableKindString(callable.kind));
            if (callable.kind != .constructor) {
                try writeObjectStringField(writer, &first, "name", callable.name);
            }

            try writeObjectKey(writer, &first, "inputs");
            try writer.writeByte('[');
            for (callable.inputs, 0..) |input, i| {
                if (i > 0) try writer.writeByte(',');
                try writer.writeByte('{');
                var input_first = true;
                try writeObjectStringField(writer, &input_first, "name", input.name);
                try writeObjectStringField(writer, &input_first, "type", self.findType(input.type_id).?.wire_type.?);
                if (callable.kind == .event) {
                    try writeObjectBoolField(writer, &input_first, "indexed", input.indexed orelse false);
                }
                try writer.writeByte('}');
            }
            try writer.writeByte(']');

            switch (callable.kind) {
                .function => {
                    try writeObjectKey(writer, &first, "outputs");
                    try writer.writeByte('[');
                    for (callable.outputs, 0..) |output, i| {
                        if (i > 0) try writer.writeByte(',');
                        try writer.writeByte('{');
                        var output_first = true;
                        try writeObjectStringField(writer, &output_first, "name", output.name);
                        try writeObjectStringField(writer, &output_first, "type", self.findType(output.type_id).?.wire_type.?);
                        try writer.writeByte('}');
                    }
                    try writer.writeByte(']');

                    try writeObjectStringField(writer, &first, "stateMutability", projectStateMutability(callable));
                },
                .constructor => {
                    try writeObjectStringField(writer, &first, "stateMutability", projectStateMutability(callable));
                },
                .event => {
                    try writeObjectBoolField(writer, &first, "anonymous", false);
                },
                .@"error" => {},
            }

            try writer.writeByte('}');
        }

        try writer.writeByte(']');
        return buffer.toOwnedSlice(allocator);
    }
};

const ResolvedType = struct {
    type_id: []const u8,
    wire_type: []const u8,
};

const UiHints = struct {
    label: ?[]const u8 = null,
    widget: ?[]const u8 = null,
    min: ?[]const u8 = null,
    max: ?[]const u8 = null,
    decimals: ?u32 = null,
    unit: ?[]const u8 = null,
};

const UiFieldHint = struct {
    widget: ?[]const u8 = null,
    min: ?[]const u8 = null,
    max: ?[]const u8 = null,
    decimals: ?u32 = null,
    unit: ?[]const u8 = null,

    fn hasAny(self: UiFieldHint) bool {
        return self.widget != null or self.min != null or self.max != null or self.decimals != null or self.unit != null;
    }
};

const CompilerNamedTypeRef = struct {
    module_id: compiler.ModuleId,
    item_id: compiler.ItemId,
};

const CompilerModuleContext = struct {
    module_id: compiler.ModuleId,
    file: *const compiler.AstFile,
    item_index: *const compiler.sema.ItemIndexResult,
    typecheck: *const compiler.sema.TypeCheckResult,
};

pub fn generateCompilerAbi(
    allocator: std.mem.Allocator,
    compilation: *compiler.driver.Compilation,
) anyerror!ContractAbi {
    var generator = try CompilerAbiGenerator.init(allocator, compilation);
    defer generator.deinit();
    return try generator.generate();
}

const CompilerAbiGenerator = struct {
    allocator: std.mem.Allocator,
    compilation: *compiler.driver.Compilation,

    callables: std.ArrayList(AbiCallable),
    types: std.ArrayList(AbiTypeNode),
    type_lookup: std.StringHashMap(usize),
    callable_ids: std.StringHashMap(void),

    global_structs: std.StringHashMap(CompilerNamedTypeRef),
    global_bitfields: std.StringHashMap(CompilerNamedTypeRef),
    global_enums: std.StringHashMap(CompilerNamedTypeRef),
    global_errors: std.StringHashMap(CompilerNamedTypeRef),

    contract_count: usize,
    primary_contract_name: ?[]const u8,

    fn init(allocator: std.mem.Allocator, compilation: *compiler.driver.Compilation) !CompilerAbiGenerator {
        return .{
            .allocator = allocator,
            .compilation = compilation,
            .callables = .{},
            .types = .{},
            .type_lookup = std.StringHashMap(usize).init(allocator),
            .callable_ids = std.StringHashMap(void).init(allocator),
            .global_structs = std.StringHashMap(CompilerNamedTypeRef).init(allocator),
            .global_bitfields = std.StringHashMap(CompilerNamedTypeRef).init(allocator),
            .global_enums = std.StringHashMap(CompilerNamedTypeRef).init(allocator),
            .global_errors = std.StringHashMap(CompilerNamedTypeRef).init(allocator),
            .contract_count = 0,
            .primary_contract_name = null,
        };
    }

    fn deinit(self: *CompilerAbiGenerator) void {
        for (self.callables.items) |*callable| {
            callable.deinit();
        }
        self.callables.deinit(self.allocator);

        for (self.types.items) |*typ| {
            typ.deinit();
        }
        self.types.deinit(self.allocator);

        self.type_lookup.deinit();
        self.callable_ids.deinit();
        self.global_structs.deinit();
        self.global_bitfields.deinit();
        self.global_enums.deinit();
        self.global_errors.deinit();
    }

    fn generate(self: *CompilerAbiGenerator) anyerror!ContractAbi {
        try self.collectGlobals();

        const package = self.compilation.db.sources.package(self.compilation.package_id);
        for (package.modules.items) |module_id| {
            const ctx = try self.moduleContext(module_id);
            for (ctx.file.root_items) |item_id| {
                switch (ctx.file.item(item_id).*) {
                    .Contract => |contract| {
                        self.contract_count += 1;
                        if (self.primary_contract_name == null) self.primary_contract_name = contract.name;
                    },
                    else => {},
                }
            }
        }

        if (self.contract_count == 0) return error.MissingContract;

        for (package.modules.items) |module_id| {
            const ctx = try self.moduleContext(module_id);
            for (ctx.file.root_items) |item_id| {
                switch (ctx.file.item(item_id).*) {
                    .Contract => |contract| try self.processContract(ctx, item_id, contract),
                    else => {},
                }
            }
        }

        const callables = try self.callables.toOwnedSlice(self.allocator);
        self.callables = .{};

        const types = try self.types.toOwnedSlice(self.allocator);
        self.types = .{};

        const lookup = self.type_lookup;
        self.type_lookup = std.StringHashMap(usize).init(self.allocator);

        self.callable_ids.deinit();
        self.callable_ids = std.StringHashMap(void).init(self.allocator);

        return .{
            .allocator = self.allocator,
            .contract_name = if (self.contract_count == 1) self.primary_contract_name.? else "bundle",
            .contract_count = self.contract_count,
            .callables = callables,
            .types = types,
            .type_lookup = lookup,
        };
    }

    fn collectGlobals(self: *CompilerAbiGenerator) anyerror!void {
        const package = self.compilation.db.sources.package(self.compilation.package_id);
        for (package.modules.items) |module_id| {
            const ctx = try self.moduleContext(module_id);
            for (ctx.file.root_items) |item_id| {
                switch (ctx.file.item(item_id).*) {
                    .Struct => |struct_item| {
                        if (!self.global_structs.contains(struct_item.name)) {
                            try self.global_structs.put(struct_item.name, .{ .module_id = module_id, .item_id = item_id });
                        }
                    },
                    .Bitfield => |bitfield_item| {
                        if (!self.global_bitfields.contains(bitfield_item.name)) {
                            try self.global_bitfields.put(bitfield_item.name, .{ .module_id = module_id, .item_id = item_id });
                        }
                    },
                    .Enum => |enum_item| {
                        if (!self.global_enums.contains(enum_item.name)) {
                            try self.global_enums.put(enum_item.name, .{ .module_id = module_id, .item_id = item_id });
                        }
                    },
                    .ErrorDecl => |error_item| {
                        if (!self.global_errors.contains(error_item.name)) {
                            try self.global_errors.put(error_item.name, .{ .module_id = module_id, .item_id = item_id });
                        }
                    },
                    else => {},
                }
            }
        }
    }

    fn moduleContext(self: *CompilerAbiGenerator, module_id: compiler.ModuleId) anyerror!CompilerModuleContext {
        const module = self.compilation.db.sources.module(module_id);
        return .{
            .module_id = module_id,
            .file = try self.compilation.db.astFile(module.file_id),
            .item_index = try self.compilation.db.itemIndex(module_id),
            .typecheck = try self.compilation.db.moduleTypeCheck(module_id),
        };
    }

    fn processContract(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        contract_item_id: compiler.ItemId,
        contract: compiler.ast.ContractItem,
    ) anyerror!void {
        _ = contract_item_id;
        for (contract.members) |member_id| {
            switch (ctx.file.item(member_id).*) {
                .Function => |function| {
                    if (function.visibility == .public) {
                        if (std.mem.eql(u8, function.name, "init")) {
                            try self.addConstructorCallable(ctx, contract.name, member_id, function);
                        } else {
                            try self.addFunctionCallable(ctx, contract.name, member_id, function);
                        }
                    }
                },
                .ErrorDecl => |error_decl| try self.addErrorCallable(ctx, contract.name, error_decl),
                .LogDecl => |log_decl| try self.addEventCallable(ctx, contract.name, log_decl),
                else => {},
            }
        }
    }

    fn addFunctionCallable(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        contract_name: []const u8,
        function_item_id: compiler.ItemId,
        function: compiler.ast.FunctionItem,
    ) anyerror!void {
        const function_type = switch (ctx.typecheck.item_types[function_item_id.index()]) {
            .function => |fn_type| fn_type,
            else => return error.UnresolvedType,
        };
        const has_self = functionHasBareSelf(ctx.file, function);
        const offset: usize = if (has_self) 1 else 0;

        var signature_types: std.ArrayList([]const u8) = .{};
        defer {
            for (signature_types.items) |item| self.allocator.free(item);
            signature_types.deinit(self.allocator);
        }
        var inputs: std.ArrayList(AbiInput) = .{};
        defer inputs.deinit(self.allocator);

        for (function.parameters[offset..], 0..) |parameter, index| {
            const param_type = function_type.param_types[index + offset];
            const resolved = switch (param_type) {
                .error_union => try self.resolvePublicResultInputType(ctx, param_type),
                else => try self.resolveSemaType(ctx, param_type, &.{}),
            };
            try signature_types.append(self.allocator, try self.allocator.dupe(u8, resolved.wire_type));
            try inputs.append(self.allocator, .{
                .name = patternName(ctx.file, parameter.pattern),
                .type_id = resolved.type_id,
                .indexed = null,
            });
        }

        var outputs: std.ArrayList(AbiInput) = .{};
        defer outputs.deinit(self.allocator);
        if (function_type.return_types.len != 0) {
            const raw_return = function_type.return_types[0];
            const payload_type = switch (raw_return) {
                .error_union => |error_union| error_union.payload_type.*,
                else => raw_return,
            };
            if (payload_type.kind() != .void) {
                const resolved_return = try self.resolveSemaType(ctx, payload_type, &.{});
                try outputs.append(self.allocator, .{
                    .name = "result",
                    .type_id = resolved_return.type_id,
                    .indexed = null,
                });
            }
        }

        const signature = try self.buildSignature(function.name, signature_types.items);
        const selector = try compiler_abi.keccakSelectorHex(self.allocator, signature);
        const id = try self.buildCallableId(contract_name, signature);
        try self.ensureCallableIdUnique(id);

        var effects: std.ArrayList(AbiEffect) = .{};
        defer effects.deinit(self.allocator);
        try self.buildEffects(ctx.typecheck.itemEffect(function_item_id), &effects);

        try self.callables.append(self.allocator, .{
            .id = id,
            .kind = .function,
            .name = function.name,
            .signature = signature,
            .selector = selector,
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .outputs = try outputs.toOwnedSlice(self.allocator),
            .effects = try effects.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        });
    }

    fn addConstructorCallable(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        contract_name: []const u8,
        function_item_id: compiler.ItemId,
        function: compiler.ast.FunctionItem,
    ) anyerror!void {
        var signature_types: std.ArrayList([]const u8) = .{};
        defer {
            for (signature_types.items) |item| self.allocator.free(item);
            signature_types.deinit(self.allocator);
        }
        var inputs: std.ArrayList(AbiInput) = .{};
        defer inputs.deinit(self.allocator);

        for (function.parameters) |parameter| {
            const resolved = try self.resolveSemaType(ctx, ctx.typecheck.pattern_types[parameter.pattern.index()].type, &.{});
            try signature_types.append(self.allocator, try self.allocator.dupe(u8, resolved.wire_type));
            try inputs.append(self.allocator, .{
                .name = patternName(ctx.file, parameter.pattern),
                .type_id = resolved.type_id,
                .indexed = null,
            });
        }

        const signature = try self.buildSignature(function.name, signature_types.items);
        const id = try self.buildCallableId(contract_name, signature);
        try self.ensureCallableIdUnique(id);

        var effects: std.ArrayList(AbiEffect) = .{};
        defer effects.deinit(self.allocator);
        try self.buildEffects(ctx.typecheck.itemEffect(function_item_id), &effects);

        try self.callables.append(self.allocator, .{
            .id = id,
            .kind = .constructor,
            .name = function.name,
            .signature = signature,
            .selector = null,
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .outputs = &.{},
            .effects = try effects.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        });
    }

    fn addErrorCallable(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        contract_name: []const u8,
        err: compiler.ast.ErrorDeclItem,
    ) anyerror!void {
        var signature_types: std.ArrayList([]const u8) = .{};
        defer {
            for (signature_types.items) |item| self.allocator.free(item);
            signature_types.deinit(self.allocator);
        }
        var inputs: std.ArrayList(AbiInput) = .{};
        defer inputs.deinit(self.allocator);

        for (err.parameters) |parameter| {
            const resolved = try self.resolveSemaType(ctx, ctx.typecheck.pattern_types[parameter.pattern.index()].type, &.{});
            try signature_types.append(self.allocator, try self.allocator.dupe(u8, resolved.wire_type));
            try inputs.append(self.allocator, .{
                .name = patternName(ctx.file, parameter.pattern),
                .type_id = resolved.type_id,
                .indexed = null,
            });
        }

        const signature = try self.buildSignature(err.name, signature_types.items);
        const selector = try compiler_abi.keccakSelectorHex(self.allocator, signature);
        const id = try self.buildCallableId(contract_name, signature);
        try self.ensureCallableIdUnique(id);

        try self.callables.append(self.allocator, .{
            .id = id,
            .kind = .@"error",
            .name = err.name,
            .signature = signature,
            .selector = selector,
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .outputs = &.{},
            .effects = &.{},
            .allocator = self.allocator,
        });
    }

    fn addEventCallable(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        contract_name: []const u8,
        log_decl: compiler.ast.LogDeclItem,
    ) anyerror!void {
        var signature_types: std.ArrayList([]const u8) = .{};
        defer {
            for (signature_types.items) |item| self.allocator.free(item);
            signature_types.deinit(self.allocator);
        }
        var inputs: std.ArrayList(AbiInput) = .{};
        defer inputs.deinit(self.allocator);

        for (log_decl.fields) |field| {
            const field_type = try compiler_type_descriptors.descriptorFromTypeExpr(self.allocator, ctx.file, ctx.item_index, field.type_expr);
            const resolved = try self.resolveSemaType(ctx, field_type, &.{});
            try signature_types.append(self.allocator, try self.allocator.dupe(u8, resolved.wire_type));
            try inputs.append(self.allocator, .{
                .name = field.name,
                .type_id = resolved.type_id,
                .indexed = field.indexed,
            });
        }

        const signature = try self.buildSignature(log_decl.name, signature_types.items);
        const id = try self.buildCallableId(contract_name, signature);
        try self.ensureCallableIdUnique(id);

        try self.callables.append(self.allocator, .{
            .id = id,
            .kind = .event,
            .name = log_decl.name,
            .signature = signature,
            .selector = null,
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .outputs = &.{},
            .effects = &.{},
            .allocator = self.allocator,
        });
    }

    fn buildEffects(self: *CompilerAbiGenerator, effect: compiler.sema.Effect, effects: *std.ArrayList(AbiEffect)) anyerror!void {
        switch (effect) {
            .pure => {},
            .external => try effects.append(self.allocator, .{ .kind = .calls }),
            .side_effects => |side_effects| {
                try self.appendEffectFlags(side_effects.has_external, side_effects.has_log, effects);
            },
            .reads => |reads| {
                try self.appendEffectSlots(.reads, reads.slots, effects);
                try self.appendEffectFlags(reads.has_external, reads.has_log, effects);
            },
            .writes => |writes| {
                try self.appendEffectSlots(.writes, writes.slots, effects);
                try self.appendEffectFlags(writes.has_external, writes.has_log, effects);
            },
            .reads_writes => |rw| {
                try self.appendEffectSlots(.reads, rw.reads, effects);
                try self.appendEffectSlots(.writes, rw.writes, effects);
                try self.appendEffectFlags(rw.has_external, rw.has_log, effects);
            },
        }
    }

    fn resolvePublicResultInputType(self: *CompilerAbiGenerator, ctx: CompilerModuleContext, ty: compiler.sema.Type) !ResolvedType {
        const error_union = switch (ty) {
            .error_union => |error_union| error_union,
            else => return error.UnsupportedAbiType,
        };
        if (error_union.error_types.len != 1) return error.UnsupportedAbiType;
        if (!self.publicAbiSupportsResultCarrierType(ctx, error_union.payload_type.*)) return error.UnsupportedAbiType;
        const payload_words = self.publicAbiStaticWordCount(ctx, error_union.payload_type.*);

        const payload = error_union.payload_type.*;
        if (!self.publicAbiErrorTypeHasPayload(ctx, error_union.error_types[0])) {
            return self.resolveTupleType(ctx, &.{ .bool, payload });
        }
        if (!self.publicAbiSupportsResultCarrierType(ctx, error_union.error_types[0])) return error.UnsupportedAbiType;
        const error_words = self.publicAbiStaticWordCount(ctx, error_union.error_types[0]);
        if (payload_words == 1 and (error_words == null or error_words.? > 1)) return error.UnsupportedAbiType;
        return self.resolveTupleType(ctx, &.{ .bool, payload, error_union.error_types[0] });
    }

    fn publicAbiSupportsResultCarrierType(self: *CompilerAbiGenerator, ctx: CompilerModuleContext, ty: compiler.sema.Type) bool {
        if (self.publicAbiStaticWordCount(ctx, ty) != null) return true;
        return switch (ty) {
            .bytes, .string => true,
            .slice => |slice| self.publicAbiSupportsResultDynamicArrayElement(ctx, slice.element_type.*),
            .array => |array| array.len == null and self.publicAbiSupportsResultDynamicArrayElement(ctx, array.element_type.*),
            .refinement => |refinement| self.publicAbiSupportsResultCarrierType(ctx, refinement.base_type.*),
            else => false,
        };
    }

    fn publicAbiSupportsResultDynamicArrayElement(self: *CompilerAbiGenerator, ctx: CompilerModuleContext, ty: compiler.sema.Type) bool {
        return switch (ty) {
            .bool, .address, .integer, .enum_, .bitfield => true,
            .refinement => |refinement| self.publicAbiSupportsResultDynamicArrayElement(ctx, refinement.base_type.*),
            .named => |named| blk: {
                const item_id = ctx.item_index.lookup(named.name) orelse break :blk false;
                break :blk switch (ctx.file.item(item_id).*) {
                    .Enum, .Bitfield => true,
                    else => false,
                };
            },
            else => false,
        };
    }

    fn publicAbiErrorTypeHasPayload(self: *CompilerAbiGenerator, ctx: CompilerModuleContext, ty: compiler.sema.Type) bool {
        _ = self;
        const name = ty.name() orelse return true;
        const item_id = ctx.item_index.lookup(name) orelse return true;
        return switch (ctx.file.item(item_id).*) {
            .ErrorDecl => |error_decl| error_decl.parameters.len != 0,
            else => true,
        };
    }

    fn publicAbiStaticWordCount(self: *CompilerAbiGenerator, ctx: CompilerModuleContext, ty: compiler.sema.Type) ?usize {
        return switch (ty) {
            .bool, .address, .integer, .enum_, .bitfield => 1,
            .refinement => |refinement| self.publicAbiStaticWordCount(ctx, refinement.base_type.*),
            .tuple => |elements| blk: {
                var total: usize = 0;
                for (elements) |element| {
                    total += self.publicAbiStaticWordCount(ctx, element) orelse break :blk null;
                }
                break :blk total;
            },
            .array => |array| blk: {
                const len = array.len orelse break :blk null;
                const element_words = self.publicAbiStaticWordCount(ctx, array.element_type.*) orelse break :blk null;
                break :blk element_words * len;
            },
            .anonymous_struct => |struct_type| blk: {
                var total: usize = 0;
                for (struct_type.fields) |field| {
                    total += self.publicAbiStaticWordCount(ctx, field.ty) orelse break :blk null;
                }
                break :blk total;
            },
            .struct_ => |named| self.publicAbiStaticWordCountForNamedStruct(ctx, named.name),
            .contract => |named| self.publicAbiStaticWordCountForNamedStruct(ctx, named.name),
            .named => |named| blk: {
                if (std.mem.eql(u8, named.name, "bool") or std.mem.eql(u8, named.name, "address")) break :blk 1;
                if (parseIntegerSpelling(named.name) != null) break :blk 1;
                const item_id = ctx.item_index.lookup(named.name) orelse break :blk null;
                break :blk switch (ctx.file.item(item_id).*) {
                    .Enum, .Bitfield => 1,
                    .Struct => self.publicAbiStaticWordCountForStructDecl(ctx, named.name),
                    .Contract => self.publicAbiStaticWordCountForContractDecl(ctx, named.name),
                    .ErrorDecl => |error_decl| blk2: {
                        var total: usize = 0;
                        for (error_decl.parameters) |parameter| {
                            total += self.publicAbiStaticWordCount(ctx, ctx.typecheck.pattern_types[parameter.pattern.index()].type) orelse break :blk2 null;
                        }
                        break :blk2 total;
                    },
                    else => null,
                };
            },
            else => null,
        };
    }

    fn publicAbiStaticWordCountForNamedStruct(self: *CompilerAbiGenerator, ctx: CompilerModuleContext, name: []const u8) ?usize {
        const item_id = ctx.item_index.lookup(name) orelse return null;
        return switch (ctx.file.item(item_id).*) {
            .Struct => self.publicAbiStaticWordCountForStructDecl(ctx, name),
            .Contract => self.publicAbiStaticWordCountForContractDecl(ctx, name),
            else => null,
        };
    }

    fn publicAbiStaticWordCountForStructDecl(self: *CompilerAbiGenerator, ctx: CompilerModuleContext, name: []const u8) ?usize {
        const item_id = ctx.item_index.lookup(name) orelse return null;
        const struct_item = switch (ctx.file.item(item_id).*) {
            .Struct => |struct_item| struct_item,
            else => return null,
        };
        var total: usize = 0;
        for (struct_item.fields) |field| {
            total += self.publicAbiStaticWordCount(ctx, compiler_type_descriptors.descriptorFromTypeExpr(self.allocator, ctx.file, ctx.item_index, field.type_expr) catch return null) orelse return null;
        }
        return total;
    }

    fn publicAbiStaticWordCountForContractDecl(self: *CompilerAbiGenerator, ctx: CompilerModuleContext, name: []const u8) ?usize {
        const item_id = ctx.item_index.lookup(name) orelse return null;
        const contract_item = switch (ctx.file.item(item_id).*) {
            .Contract => |contract_item| contract_item,
            else => return null,
        };
        var total: usize = 0;
        for (contract_item.members) |member_id| {
            switch (ctx.file.item(member_id).*) {
                .Field => |field| {
                    if (field.type_expr) |type_expr| {
                        total += self.publicAbiStaticWordCount(ctx, compiler_type_descriptors.descriptorFromTypeExpr(self.allocator, ctx.file, ctx.item_index, type_expr) catch return null) orelse return null;
                    } else return null;
                },
                else => {},
            }
        }
        return total;
    }

    fn appendEffectFlags(
        self: *CompilerAbiGenerator,
        has_external: bool,
        has_log: bool,
        effects: *std.ArrayList(AbiEffect),
    ) !void {
        if (has_external) try effects.append(self.allocator, .{ .kind = .calls });
        if (has_log) try effects.append(self.allocator, .{ .kind = .emits });
    }

    fn appendEffectSlots(
        self: *CompilerAbiGenerator,
        kind: AbiEffectKind,
        slots: []const compiler.sema.EffectSlot,
        effects: *std.ArrayList(AbiEffect),
    ) !void {
        var paths: std.ArrayList([]const u8) = .{};
        defer {
            for (paths.items) |path| self.allocator.free(path);
            paths.deinit(self.allocator);
        }
        for (slots) |slot| {
            try paths.append(self.allocator, try self.allocator.dupe(u8, slot.name));
        }
        sortStringArrayList(&paths);
        for (paths.items) |path| {
            try effects.append(self.allocator, .{
                .kind = kind,
                .path = try self.allocator.dupe(u8, path),
                .event_id = null,
            });
        }
    }

    fn resolveSemaType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        ty: compiler.sema.Type,
        stack: []const []const u8,
    ) anyerror!ResolvedType {
        _ = stack;
        switch (ty) {
            .bool, .address, .string, .bytes, .integer => {
                const wire = try compiler_abi.canonicalAbiType(self.allocator, ty);
                errdefer self.allocator.free(wire);
                var node = AbiTypeNode{
                    .kind = .primitive,
                    .allocator = self.allocator,
                    .name = switch (ty) {
                        .bool => "bool",
                        .address => "address",
                        .string => "string",
                        .bytes => "bytes",
                        .integer => |integer| integer.spelling orelse "u256",
                        else => unreachable,
                    },
                    .wire_type = wire,
                    .ui_widget = defaultWidgetForWireType(wire),
                };
                const type_id = try self.ensureTypeNode(&node);
                const idx = self.type_lookup.get(type_id).?;
                return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
            },
            .refinement => |refinement| return self.resolveRefinementType(ctx, refinement),
            .array => |array| return self.resolveArrayType(ctx, array),
            .slice => |slice| return self.resolveSliceType(ctx, slice),
            .tuple => |elements| return self.resolveTupleType(ctx, elements),
            .anonymous_struct => |struct_type| return self.resolveAnonymousStructType(ctx, struct_type.fields),
            .struct_ => |named| return self.resolveNamedStructType(ctx, named.name),
            .bitfield => |named| return self.resolveNamedBitfieldType(ctx, named.name),
            .enum_ => |named| return self.resolveNamedEnumType(ctx, named.name),
            .error_union => |error_union| return self.resolveSemaType(ctx, error_union.payload_type.*, &.{}),
            .named => |named| {
                if (self.global_structs.contains(named.name)) return self.resolveNamedStructType(ctx, named.name);
                if (self.global_bitfields.contains(named.name)) return self.resolveNamedBitfieldType(ctx, named.name);
                if (self.global_enums.contains(named.name)) return self.resolveNamedEnumType(ctx, named.name);
                if (self.global_errors.contains(named.name)) return self.resolveNamedErrorType(ctx, named.name);
                if (std.mem.eql(u8, named.name, "bool")) return self.resolveSemaType(ctx, .bool, &.{});
                if (std.mem.eql(u8, named.name, "address")) return self.resolveSemaType(ctx, .address, &.{});
                if (std.mem.eql(u8, named.name, "string")) return self.resolveSemaType(ctx, .string, &.{});
                if (std.mem.eql(u8, named.name, "bytes")) return self.resolveSemaType(ctx, .bytes, &.{});
                if (std.mem.eql(u8, named.name, "NonZeroAddress")) return self.resolveSemaType(ctx, .address, &.{});
                if (parseIntegerSpelling(named.name)) |integer_ty| return self.resolveSemaType(ctx, integer_ty, &.{});
                return error.UnsupportedAbiType;
            },
            else => return error.UnsupportedAbiType,
        }
    }

    fn parseIntegerSpelling(name: []const u8) ?compiler.sema.Type {
        if (name.len < 2) return null;
        const signed = switch (name[0]) {
            'u' => false,
            'i' => true,
            else => return null,
        };
        const bits = std.fmt.parseUnsigned(u16, name[1..], 10) catch return null;
        return .{ .integer = .{ .bits = bits, .signed = signed, .spelling = name } };
    }

    fn resolveArrayType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        array: compiler.sema.ArrayType,
    ) anyerror!ResolvedType {
        const elem = try self.resolveSemaType(ctx, array.element_type.*, &.{});
        const len = array.len orelse return error.UnsupportedAbiType;
        const wire = try std.fmt.allocPrint(self.allocator, "{s}[{d}]", .{ elem.wire_type, len });
        var node = AbiTypeNode{
            .kind = .array,
            .allocator = self.allocator,
            .element_type = elem.type_id,
            .size = len,
            .wire_type = wire,
            .ui_widget = "json",
        };
        const type_id = try self.ensureTypeNode(&node);
        const idx = self.type_lookup.get(type_id).?;
        return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
    }

    fn resolveSliceType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        slice: compiler.sema.SliceType,
    ) anyerror!ResolvedType {
        const elem = try self.resolveSemaType(ctx, slice.element_type.*, &.{});
        const wire = try std.fmt.allocPrint(self.allocator, "{s}[]", .{elem.wire_type});
        var node = AbiTypeNode{
            .kind = .slice,
            .allocator = self.allocator,
            .element_type = elem.type_id,
            .wire_type = wire,
            .ui_widget = "json",
        };
        const type_id = try self.ensureTypeNode(&node);
        const idx = self.type_lookup.get(type_id).?;
        return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
    }

    fn resolveTupleType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        elements: []const compiler.sema.Type,
    ) anyerror!ResolvedType {
        var component_ids = try self.allocator.alloc([]const u8, elements.len);
        errdefer self.allocator.free(component_ids);
        var wire_parts: std.ArrayList([]const u8) = .{};
        defer {
            for (wire_parts.items) |part| self.allocator.free(part);
            wire_parts.deinit(self.allocator);
        }
        for (elements, 0..) |element, index| {
            const resolved = try self.resolveSemaType(ctx, element, &.{});
            component_ids[index] = resolved.type_id;
            try wire_parts.append(self.allocator, try self.allocator.dupe(u8, resolved.wire_type));
        }
        const joined = try std.mem.join(self.allocator, ",", wire_parts.items);
        defer self.allocator.free(joined);
        const wire = try std.fmt.allocPrint(self.allocator, "({s})", .{joined});
        var node = AbiTypeNode{
            .kind = .tuple,
            .allocator = self.allocator,
            .components = component_ids,
            .wire_type = wire,
            .ui_widget = "json",
        };
        const type_id = try self.ensureTypeNode(&node);
        const idx = self.type_lookup.get(type_id).?;
        return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
    }

    fn resolveAnonymousStructType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        fields_input: []const compiler.sema.AnonymousStructField,
    ) anyerror!ResolvedType {
        var fields = try self.allocator.alloc(AbiFieldRef, fields_input.len);
        errdefer self.allocator.free(fields);
        var wire_parts: std.ArrayList([]const u8) = .{};
        defer {
            for (wire_parts.items) |part| self.allocator.free(part);
            wire_parts.deinit(self.allocator);
        }
        for (fields_input, 0..) |field, index| {
            const resolved = try self.resolveSemaType(ctx, field.ty, &.{});
            fields[index] = .{ .name = field.name, .type_id = resolved.type_id };
            try wire_parts.append(self.allocator, try self.allocator.dupe(u8, resolved.wire_type));
        }
        const joined = try std.mem.join(self.allocator, ",", wire_parts.items);
        defer self.allocator.free(joined);
        const wire = try std.fmt.allocPrint(self.allocator, "({s})", .{joined});
        var node = AbiTypeNode{
            .kind = .struct_,
            .allocator = self.allocator,
            .name = "__anon_struct__",
            .wire_type = wire,
            .fields = fields,
            .ui_label = "anonymous struct",
            .ui_widget = "json",
        };
        const type_id = try self.ensureTypeNode(&node);
        const idx = self.type_lookup.get(type_id).?;
        return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
    }

    fn resolveNamedStructType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        name: []const u8,
    ) anyerror!ResolvedType {
        if (ctx.typecheck.instantiatedStructByName(name)) |instantiated| {
            var fields = try self.allocator.alloc(AbiFieldRef, instantiated.fields.len);
            errdefer self.allocator.free(fields);
            var wire_parts: std.ArrayList([]const u8) = .{};
            defer {
                for (wire_parts.items) |part| self.allocator.free(part);
                wire_parts.deinit(self.allocator);
            }
            for (instantiated.fields, 0..) |field, index| {
                const resolved = try self.resolveSemaType(ctx, field.ty, &.{});
                fields[index] = .{ .name = field.name, .type_id = resolved.type_id };
                try wire_parts.append(self.allocator, try self.allocator.dupe(u8, resolved.wire_type));
            }
            const joined = try std.mem.join(self.allocator, ",", wire_parts.items);
            defer self.allocator.free(joined);
            const wire = try std.fmt.allocPrint(self.allocator, "({s})", .{joined});
            const template_name = switch (ctx.file.item(instantiated.template_item_id).*) {
                .Struct => |struct_item| struct_item.name,
                else => name,
            };
            var node = AbiTypeNode{
                .kind = .struct_,
                .allocator = self.allocator,
                .name = name,
                .wire_type = wire,
                .fields = fields,
                .ui_label = template_name,
                .ui_widget = "json",
            };
            const type_id = try self.ensureTypeNode(&node);
            const idx = self.type_lookup.get(type_id).?;
            return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
        }

        const ref = self.global_structs.get(name) orelse return error.UnknownStructType;
        const owner_ctx = try self.moduleContext(ref.module_id);
        const struct_item = owner_ctx.file.item(ref.item_id).Struct;
        var fields = try self.allocator.alloc(AbiFieldRef, struct_item.fields.len);
        errdefer self.allocator.free(fields);
        var wire_parts: std.ArrayList([]const u8) = .{};
        defer {
            for (wire_parts.items) |part| self.allocator.free(part);
            wire_parts.deinit(self.allocator);
        }
        for (struct_item.fields, 0..) |field, index| {
            const field_type = try compiler_type_descriptors.descriptorFromTypeExpr(self.allocator, owner_ctx.file, owner_ctx.item_index, field.type_expr);
            const resolved = try self.resolveSemaType(owner_ctx, field_type, &.{});
            fields[index] = .{ .name = field.name, .type_id = resolved.type_id };
            try wire_parts.append(self.allocator, try self.allocator.dupe(u8, resolved.wire_type));
        }
        const joined = try std.mem.join(self.allocator, ",", wire_parts.items);
        defer self.allocator.free(joined);
        const wire = try std.fmt.allocPrint(self.allocator, "({s})", .{joined});
        var node = AbiTypeNode{
            .kind = .struct_,
            .allocator = self.allocator,
            .name = name,
            .wire_type = wire,
            .fields = fields,
            .ui_label = name,
            .ui_widget = "json",
        };
        const type_id = try self.ensureTypeNode(&node);
        const idx = self.type_lookup.get(type_id).?;
        return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
    }

    fn resolveNamedEnumType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        name: []const u8,
    ) anyerror!ResolvedType {
        if (ctx.typecheck.instantiatedEnumByName(name) != null) {
            const repr = try self.resolveSemaType(ctx, .{ .integer = .{ .bits = 32, .signed = false, .spelling = "u32" } }, &.{});
            var node = AbiTypeNode{
                .kind = .enum_,
                .allocator = self.allocator,
                .name = name,
                .wire_type = try self.allocator.dupe(u8, repr.wire_type),
                .repr_type = repr.type_id,
                .variants = &.{},
                .ui_label = name,
                .ui_widget = "select",
            };
            const type_id = try self.ensureTypeNode(&node);
            const idx = self.type_lookup.get(type_id).?;
            return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
        }

        const ref = self.global_enums.get(name) orelse return error.UnknownEnumType;
        const owner_ctx = try self.moduleContext(ref.module_id);
        const enum_item = owner_ctx.file.item(ref.item_id).Enum;
        const repr = try self.resolveSemaType(owner_ctx, .{ .integer = .{ .bits = 32, .signed = false, .spelling = "u32" } }, &.{});
        var variants = try self.allocator.alloc(AbiEnumVariant, enum_item.variants.len);
        errdefer self.allocator.free(variants);
        for (enum_item.variants, 0..) |variant, index| {
            variants[index] = .{ .name = variant.name, .value = @intCast(index) };
        }
        var node = AbiTypeNode{
            .kind = .enum_,
            .allocator = self.allocator,
            .name = name,
            .wire_type = try self.allocator.dupe(u8, repr.wire_type),
            .repr_type = repr.type_id,
            .variants = variants,
            .ui_label = name,
            .ui_widget = "select",
        };
        const type_id = try self.ensureTypeNode(&node);
        const idx = self.type_lookup.get(type_id).?;
        return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
    }

    fn resolveNamedBitfieldType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        name: []const u8,
    ) anyerror!ResolvedType {
        const base = try self.resolveBitfieldBaseType(ctx, name);

        var node = AbiTypeNode{
            .kind = .primitive,
            .allocator = self.allocator,
            .name = name,
            .wire_type = try self.allocator.dupe(u8, base.wire_type),
            .ui_label = name,
            .ui_widget = defaultWidgetForWireType(base.wire_type),
        };
        const type_id = try self.ensureTypeNode(&node);
        const idx = self.type_lookup.get(type_id).?;
        return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
    }

    fn resolveNamedErrorType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        name: []const u8,
    ) anyerror!ResolvedType {
        _ = ctx;
        const ref = self.global_errors.get(name) orelse return error.UnsupportedAbiType;
        const owner_ctx = try self.moduleContext(ref.module_id);
        const error_item = owner_ctx.file.item(ref.item_id).ErrorDecl;

        if (error_item.parameters.len == 0) {
            return self.resolveSemaType(owner_ctx, .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } }, &.{});
        }

        var elements = try self.allocator.alloc(compiler.sema.Type, error_item.parameters.len);
        defer self.allocator.free(elements);
        for (error_item.parameters, 0..) |parameter, index| {
            elements[index] = owner_ctx.typecheck.pattern_types[parameter.pattern.index()].type;
        }

        if (elements.len == 1) {
            return self.resolveSemaType(owner_ctx, elements[0], &.{});
        }
        return self.resolveTupleType(owner_ctx, elements);
    }

    fn resolveBitfieldBaseType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        name: []const u8,
    ) anyerror!ResolvedType {
        if (ctx.typecheck.instantiatedBitfieldByName(name)) |instantiated| {
            if (instantiated.base_type) |base_type| return self.resolveSemaType(ctx, base_type, &.{});
            return self.resolveSemaType(ctx, .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } }, &.{});
        }

        const ref = self.global_bitfields.get(name) orelse return error.UnknownStructType;
        const owner_ctx = try self.moduleContext(ref.module_id);
        const bitfield_item = owner_ctx.file.item(ref.item_id).Bitfield;
        if (bitfield_item.base_type) |type_expr| {
            const field_type = try compiler_type_descriptors.descriptorFromTypeExpr(self.allocator, owner_ctx.file, owner_ctx.item_index, type_expr);
            return self.resolveSemaType(owner_ctx, field_type, &.{});
        }
        return self.resolveSemaType(owner_ctx, .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } }, &.{});
    }

    fn resolveRefinementType(
        self: *CompilerAbiGenerator,
        ctx: CompilerModuleContext,
        refinement: compiler.sema.RefinementType,
    ) anyerror!ResolvedType {
        const base = try self.resolveSemaType(ctx, refinement.base_type.*, &.{});

        var predicate_json: []const u8 = try self.buildExactPredicate();
        var ui_hints = UiHints{};

        if (std.mem.eql(u8, refinement.name, "NonZeroAddress")) {
            self.allocator.free(predicate_json);
            predicate_json = try self.buildNonZeroAddressPredicate();
            ui_hints.widget = "address";
        } else if (std.mem.eql(u8, refinement.name, "Scaled")) {
            if (refinement.args.len >= 2 and refinement.args[1] == .Integer) {
                const decimals = parseUnsignedIntLiteral(refinement.args[1].Integer.text) orelse 0;
                self.allocator.free(predicate_json);
                predicate_json = try self.buildScaledPredicate(@intCast(decimals));
                ui_hints.widget = "number";
                ui_hints.decimals = @intCast(decimals);
            }
        } else if (std.mem.eql(u8, refinement.name, "MinValue")) {
            if (refinement.args.len >= 2 and refinement.args[1] == .Integer) {
                const min_value = parseUnsignedIntLiteral(refinement.args[1].Integer.text) orelse 0;
                self.allocator.free(predicate_json);
                predicate_json = try self.buildMinPredicate(min_value);
                ui_hints.widget = "number";
                ui_hints.min = try std.fmt.allocPrint(self.allocator, "{d}", .{min_value});
            }
        } else if (std.mem.eql(u8, refinement.name, "MaxValue")) {
            if (refinement.args.len >= 2 and refinement.args[1] == .Integer) {
                const max_value = parseUnsignedIntLiteral(refinement.args[1].Integer.text) orelse 0;
                self.allocator.free(predicate_json);
                predicate_json = try self.buildMaxPredicate(max_value);
                ui_hints.widget = "number";
                ui_hints.max = try std.fmt.allocPrint(self.allocator, "{d}", .{max_value});
            }
        } else if (std.mem.eql(u8, refinement.name, "InRange")) {
            if (refinement.args.len >= 3 and refinement.args[1] == .Integer and refinement.args[2] == .Integer) {
                const min_value = parseUnsignedIntLiteral(refinement.args[1].Integer.text) orelse 0;
                const max_value = parseUnsignedIntLiteral(refinement.args[2].Integer.text) orelse 0;
                self.allocator.free(predicate_json);
                predicate_json = try self.buildRangePredicate(min_value, max_value);
                ui_hints.widget = "number";
                ui_hints.min = try std.fmt.allocPrint(self.allocator, "{d}", .{min_value});
                ui_hints.max = try std.fmt.allocPrint(self.allocator, "{d}", .{max_value});
            }
        } else if (std.mem.eql(u8, refinement.name, "Exact")) {
            ui_hints.widget = "number";
        }

        var node = AbiTypeNode{
            .kind = .refinement,
            .allocator = self.allocator,
            .base = base.type_id,
            .wire_type = try self.allocator.dupe(u8, base.wire_type),
            .predicate_json = predicate_json,
            .ui_widget = ui_hints.widget orelse defaultWidgetForWireType(base.wire_type),
            .ui_min = ui_hints.min,
            .ui_max = ui_hints.max,
            .ui_decimals = ui_hints.decimals,
            .ui_unit = ui_hints.unit,
        };
        const type_id = try self.ensureTypeNode(&node);
        const idx = self.type_lookup.get(type_id).?;
        return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
    }

    fn buildSignature(self: *CompilerAbiGenerator, name: []const u8, types: []const []const u8) anyerror![]const u8 {
        const joined = try std.mem.join(self.allocator, ",", types);
        defer self.allocator.free(joined);
        return std.fmt.allocPrint(self.allocator, "{s}({s})", .{ name, joined });
    }

    fn buildCallableId(self: *CompilerAbiGenerator, contract_name: []const u8, signature: []const u8) anyerror![]const u8 {
        return std.fmt.allocPrint(self.allocator, "c:{s}.{s}", .{ contract_name, signature });
    }

    fn ensureCallableIdUnique(self: *CompilerAbiGenerator, callable_id: []const u8) anyerror!void {
        if (self.callable_ids.contains(callable_id)) return error.DuplicateCallableId;
        try self.callable_ids.put(callable_id, {});
    }

    fn ensureTypeNode(self: *CompilerAbiGenerator, node: *AbiTypeNode) anyerror![]const u8 {
        const payload = try buildCanonicalTypePayload(self.allocator, node);
        errdefer self.allocator.free(payload);
        const type_id = try hashedTypeId(self.allocator, payload);
        errdefer self.allocator.free(type_id);
        if (self.type_lookup.get(type_id)) |idx| {
            const existing = &self.types.items[idx];
            if (!std.mem.eql(u8, existing.canonical_payload.?, payload)) {
                node.deinit();
                return error.TypeHashCollision;
            }
            self.allocator.free(payload);
            self.allocator.free(type_id);
            node.deinit();
            return existing.type_id.?;
        }
        node.canonical_payload = payload;
        node.type_id = type_id;
        try self.types.append(self.allocator, node.*);
        const new_index = self.types.items.len - 1;
        try self.type_lookup.put(self.types.items[new_index].type_id.?, new_index);
        return self.types.items[new_index].type_id.?;
    }

    fn buildMinPredicate(self: *CompilerAbiGenerator, min_value: u256) anyerror![]const u8 {
        return std.fmt.allocPrint(self.allocator, "{{\"kind\":\"min\",\"value\":\"{d}\"}}", .{min_value});
    }

    fn buildMaxPredicate(self: *CompilerAbiGenerator, max_value: u256) anyerror![]const u8 {
        return std.fmt.allocPrint(self.allocator, "{{\"kind\":\"max\",\"value\":\"{d}\"}}", .{max_value});
    }

    fn buildRangePredicate(self: *CompilerAbiGenerator, min_value: u256, max_value: u256) anyerror![]const u8 {
        return std.fmt.allocPrint(self.allocator, "{{\"kind\":\"range\",\"min\":\"{d}\",\"max\":\"{d}\"}}", .{ min_value, max_value });
    }

    fn buildScaledPredicate(self: *CompilerAbiGenerator, decimals: u32) anyerror![]const u8 {
        return std.fmt.allocPrint(self.allocator, "{{\"kind\":\"scaled\",\"decimals\":{d}}}", .{decimals});
    }

    fn buildExactPredicate(self: *CompilerAbiGenerator) anyerror![]const u8 {
        return self.allocator.dupe(u8, "{\"kind\":\"exact\"}");
    }

    fn buildNonZeroAddressPredicate(self: *CompilerAbiGenerator) anyerror![]const u8 {
        return self.allocator.dupe(u8, "{\"kind\":\"nonZeroAddress\"}");
    }
};

fn functionHasBareSelf(file: *const compiler.AstFile, function: compiler.ast.FunctionItem) bool {
    if (function.parameters.len == 0) return false;
    return std.mem.eql(u8, patternName(file, function.parameters[0].pattern), "self");
}

fn patternName(file: *const compiler.AstFile, pattern_id: compiler.PatternId) []const u8 {
    return switch (file.pattern(pattern_id).*) {
        .Name => |name| name.name,
        else => "_",
    };
}

fn callableKindString(kind: CallableKind) []const u8 {
    return switch (kind) {
        .function => "function",
        .constructor => "constructor",
        .@"error" => "error",
        .event => "event",
    };
}

fn effectKindString(kind: AbiEffectKind) []const u8 {
    return switch (kind) {
        .reads => "reads",
        .writes => "writes",
        .emits => "emits",
        .calls => "calls",
        .value => "value",
    };
}

fn projectStateMutability(callable: AbiCallable) []const u8 {
    var has_reads = false;
    var has_writes = false;
    var has_calls = false;
    var has_value = false;

    for (callable.effects) |effect| {
        switch (effect.kind) {
            .reads => has_reads = true,
            .writes => has_writes = true,
            .calls => has_calls = true,
            .value => has_value = true,
            .emits => {},
        }
    }

    if (has_value) return "payable";
    if (has_writes or has_calls) return "nonpayable";
    if (has_reads) return "view";
    return "pure";
}

fn parseUnsignedIntLiteral(raw: []const u8) ?u256 {
    var cleaned_storage = std.ArrayList(u8){};
    defer cleaned_storage.deinit(std.heap.page_allocator);

    for (raw) |ch| {
        if (ch != '_') {
            cleaned_storage.append(std.heap.page_allocator, ch) catch return null;
        }
    }

    const cleaned = cleaned_storage.items;
    return std.fmt.parseInt(u256, cleaned, 0) catch null;
}

fn sortStringArrayList(values: *std.ArrayList([]const u8)) void {
    sortStringSlices(values.items);
}

fn sortStringSlices(values: [][]const u8) void {
    var i: usize = 0;
    while (i < values.len) : (i += 1) {
        var j: usize = i + 1;
        while (j < values.len) : (j += 1) {
            if (std.mem.lessThan(u8, values[j], values[i])) {
                const tmp = values[i];
                values[i] = values[j];
                values[j] = tmp;
            }
        }
    }
}

fn keccakSelectorHex(allocator: std.mem.Allocator, signature: []const u8) ![]const u8 {
    var hash: [32]u8 = undefined;
    crypto.hash.sha3.Keccak256.hash(signature, &hash, .{});
    const selector = hash[0..4];

    var hex: [8]u8 = undefined;
    for (selector, 0..) |byte, i| {
        hex[i * 2] = std.fmt.hex_charset[byte >> 4];
        hex[i * 2 + 1] = std.fmt.hex_charset[byte & 0x0f];
    }

    return std.fmt.allocPrint(allocator, "0x{s}", .{hex[0..]});
}

fn hashedTypeId(allocator: std.mem.Allocator, canonical_payload: []const u8) ![]const u8 {
    var digest: [crypto.hash.Blake3.digest_length]u8 = undefined;
    crypto.hash.Blake3.hash(canonical_payload, digest[0..], .{});

    var hex = try allocator.alloc(u8, digest.len * 2);
    defer allocator.free(hex);

    for (digest, 0..) |byte, i| {
        hex[i * 2] = std.fmt.hex_charset[byte >> 4];
        hex[i * 2 + 1] = std.fmt.hex_charset[byte & 0x0f];
    }

    return std.fmt.allocPrint(allocator, "t:{s}", .{hex});
}

fn buildCanonicalTypePayload(allocator: std.mem.Allocator, node: *const AbiTypeNode) ![]const u8 {
    var buffer = try std.ArrayList(u8).initCapacity(allocator, 256);
    defer buffer.deinit(allocator);

    const writer = buffer.writer(allocator);
    try writeTypeNodeObject(writer, node, false);

    return buffer.toOwnedSlice(allocator);
}

fn writeTypeNodeObject(writer: anytype, node: *const AbiTypeNode, include_type_id: bool) !void {
    try writer.writeByte('{');

    var first = true;

    if (include_type_id) {
        try writeObjectStringField(writer, &first, "typeId", node.type_id.?);
    }

    try writeObjectStringField(writer, &first, "kind", typeKindString(node.kind));

    switch (node.kind) {
        .primitive => {
            try writeObjectStringField(writer, &first, "name", node.name.?);
            try writeObjectKey(writer, &first, "wire");
            try writeWireType(writer, node.wire_type.?);
        },
        .array => {
            try writeObjectStringField(writer, &first, "elementType", node.element_type.?);
            try writeObjectIntField(writer, &first, "size", node.size.?);
            try writeObjectKey(writer, &first, "wire");
            try writeWireType(writer, node.wire_type.?);
        },
        .slice => {
            try writeObjectStringField(writer, &first, "elementType", node.element_type.?);
            try writeObjectKey(writer, &first, "wire");
            try writeWireType(writer, node.wire_type.?);
        },
        .tuple => {
            try writeObjectKey(writer, &first, "components");
            try writer.writeByte('[');
            for (node.components, 0..) |component, i| {
                if (i > 0) try writer.writeByte(',');
                try writeJsonString(writer, component);
            }
            try writer.writeByte(']');

            try writeObjectKey(writer, &first, "wire");
            try writeWireAsTuple(writer);
        },
        .struct_ => {
            try writeObjectStringField(writer, &first, "name", node.name.?);

            try writeObjectKey(writer, &first, "fields");
            try writer.writeByte('[');
            for (node.fields, 0..) |field, i| {
                if (i > 0) try writer.writeByte(',');
                try writer.writeByte('{');
                var field_first = true;
                try writeObjectStringField(writer, &field_first, "name", field.name);
                try writeObjectStringField(writer, &field_first, "typeId", field.type_id);
                try writer.writeByte('}');
            }
            try writer.writeByte(']');

            try writeObjectKey(writer, &first, "wire");
            try writeWireAsTuple(writer);
        },
        .enum_ => {
            try writeObjectStringField(writer, &first, "name", node.name.?);

            try writeObjectKey(writer, &first, "repr");
            try writer.writeByte('{');
            var repr_first = true;
            try writeObjectStringField(writer, &repr_first, "typeId", node.repr_type.?);
            try writer.writeByte('}');

            try writeObjectKey(writer, &first, "variants");
            try writer.writeByte('[');
            for (node.variants, 0..) |variant, i| {
                if (i > 0) try writer.writeByte(',');
                try writer.writeByte('{');
                var variant_first = true;
                try writeObjectStringField(writer, &variant_first, "name", variant.name);
                try writeObjectIntField(writer, &variant_first, "value", variant.value);
                try writer.writeByte('}');
            }
            try writer.writeByte(']');

            try writeObjectKey(writer, &first, "wire");
            try writeWireType(writer, node.wire_type.?);
        },
        .alias => {
            if (node.name) |name| {
                try writeObjectStringField(writer, &first, "name", name);
            }
        },
        .refinement => {
            try writeObjectStringField(writer, &first, "base", node.base.?);

            try writeObjectKey(writer, &first, "predicate");
            try writer.writeAll(node.predicate_json.?);

            try writeObjectKey(writer, &first, "wire");
            try writeWireType(writer, node.wire_type.?);
        },
    }

    try writer.writeByte('}');
}

fn typeHasUiMeta(node: *const AbiTypeNode) bool {
    return node.ui_label != null or
        node.ui_widget != null or
        node.ui_min != null or
        node.ui_max != null or
        node.ui_decimals != null or
        node.ui_unit != null;
}

fn writeTypeUiObject(writer: anytype, node: *const AbiTypeNode) !void {
    try writer.writeByte('{');
    var first = true;
    if (node.ui_label) |label| try writeObjectStringField(writer, &first, "label", label);
    if (node.ui_widget) |widget| try writeObjectStringField(writer, &first, "widget", widget);
    if (node.ui_min) |min| try writeObjectStringField(writer, &first, "min", min);
    if (node.ui_max) |max| try writeObjectStringField(writer, &first, "max", max);
    if (node.ui_decimals) |decimals| try writeObjectIntField(writer, &first, "decimals", decimals);
    if (node.ui_unit) |unit| try writeObjectStringField(writer, &first, "unit", unit);
    try writer.writeByte('}');
}

fn typeKindString(kind: TypeNodeKind) []const u8 {
    return switch (kind) {
        .primitive => "primitive",
        .array => "array",
        .slice => "slice",
        .tuple => "tuple",
        .struct_ => "struct",
        .enum_ => "enum",
        .alias => "alias",
        .refinement => "refinement",
    };
}

fn writeUiFieldHint(writer: anytype, hint: UiFieldHint) !void {
    try writer.writeByte('{');
    var first = true;
    if (hint.widget) |widget| try writeObjectStringField(writer, &first, "widget", widget);
    if (hint.min) |min| try writeObjectStringField(writer, &first, "min", min);
    if (hint.max) |max| try writeObjectStringField(writer, &first, "max", max);
    if (hint.decimals) |decimals| try writeObjectIntField(writer, &first, "decimals", decimals);
    if (hint.unit) |unit| try writeObjectStringField(writer, &first, "unit", unit);
    try writer.writeByte('}');
}

fn defaultWidgetForWireType(wire_type: []const u8) ?[]const u8 {
    if (std.mem.startsWith(u8, wire_type, "uint")) return "number";
    if (std.mem.startsWith(u8, wire_type, "int")) return "number";
    if (std.mem.eql(u8, wire_type, "address")) return "address";
    if (std.mem.eql(u8, wire_type, "bytes")) return "bytes";
    if (std.mem.eql(u8, wire_type, "string")) return "text";
    if (std.mem.eql(u8, wire_type, "bool")) return "select";
    return null;
}

fn writeWireType(writer: anytype, wire_type: []const u8) !void {
    try writer.writeByte('{');
    try writeJsonString(writer, ProfileId);
    try writer.writeByte(':');
    try writer.writeByte('{');
    try writeJsonString(writer, "type");
    try writer.writeByte(':');
    try writeJsonString(writer, wire_type);
    try writer.writeByte('}');
    try writer.writeByte('}');
}

fn writeWireAsTuple(writer: anytype) !void {
    try writer.writeByte('{');
    try writeJsonString(writer, ProfileId);
    try writer.writeByte(':');
    try writer.writeByte('{');
    try writeJsonString(writer, "as");
    try writer.writeByte(':');
    try writeJsonString(writer, "tuple");
    try writer.writeByte('}');
    try writer.writeByte('}');
}

fn writeObjectKey(writer: anytype, first: *bool, key: []const u8) !void {
    if (!first.*) {
        try writer.writeByte(',');
    }
    first.* = false;
    try writeJsonString(writer, key);
    try writer.writeByte(':');
}

fn writeObjectStringField(writer: anytype, first: *bool, key: []const u8, value: []const u8) !void {
    try writeObjectKey(writer, first, key);
    try writeJsonString(writer, value);
}

fn writeObjectBoolField(writer: anytype, first: *bool, key: []const u8, value: bool) !void {
    try writeObjectKey(writer, first, key);
    try writer.writeAll(if (value) "true" else "false");
}

fn writeObjectIntField(writer: anytype, first: *bool, key: []const u8, value: anytype) !void {
    try writeObjectKey(writer, first, key);
    try writer.print("{d}", .{value});
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |ch| {
        switch (ch) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (ch < 0x20) {
                    try writer.print("\\u00{x:0>2}", .{ch});
                } else {
                    try writer.writeByte(ch);
                }
            },
        }
    }
    try writer.writeByte('"');
}
