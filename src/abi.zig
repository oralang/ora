// ============================================================================
// ABI Support (Ora ABI v0.1 + Solidity projection)
// ============================================================================

const std = @import("std");
const crypto = std.crypto;
const lib = @import("root.zig");
const state_tracker = @import("analysis/state_tracker.zig");

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

    fn findType(self: *const ContractAbi, type_id: []const u8) ?*const AbiTypeNode {
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
            try writeObjectStringField(writer, &first, "name", callable.name);

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

pub const AbiGenerator = struct {
    allocator: std.mem.Allocator,

    callables: std.ArrayList(AbiCallable),
    types: std.ArrayList(AbiTypeNode),

    type_lookup: std.StringHashMap(usize),
    callable_ids: std.StringHashMap(void),

    global_structs: std.StringHashMap(*const lib.ast.StructDeclNode),
    global_enums: std.StringHashMap(*const lib.ast.EnumDeclNode),

    contract_count: usize,
    primary_contract_name: ?[]const u8,

    pub fn init(allocator: std.mem.Allocator) !AbiGenerator {
        return .{
            .allocator = allocator,
            .callables = std.ArrayList(AbiCallable){},
            .types = std.ArrayList(AbiTypeNode){},
            .type_lookup = std.StringHashMap(usize).init(allocator),
            .callable_ids = std.StringHashMap(void).init(allocator),
            .global_structs = std.StringHashMap(*const lib.ast.StructDeclNode).init(allocator),
            .global_enums = std.StringHashMap(*const lib.ast.EnumDeclNode).init(allocator),
            .contract_count = 0,
            .primary_contract_name = null,
        };
    }

    pub fn deinit(self: *AbiGenerator) void {
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
        self.global_enums.deinit();
    }

    pub fn generate(self: *AbiGenerator, ast_nodes: []lib.AstNode) AbiError!ContractAbi {
        self.collectGlobals(ast_nodes);

        for (ast_nodes) |node| {
            if (node == .Contract) {
                self.contract_count += 1;
                if (self.primary_contract_name == null) {
                    self.primary_contract_name = node.Contract.name;
                }
            }
        }

        if (self.contract_count == 0) {
            return error.MissingContract;
        }

        for (ast_nodes) |node| {
            if (node == .Contract) {
                try self.processContract(&node.Contract);
            }
        }

        const callables = try self.callables.toOwnedSlice(self.allocator);
        self.callables = std.ArrayList(AbiCallable){};

        const types = try self.types.toOwnedSlice(self.allocator);
        self.types = std.ArrayList(AbiTypeNode){};

        const lookup = self.type_lookup;
        self.type_lookup = std.StringHashMap(usize).init(self.allocator);

        self.callable_ids.deinit();
        self.callable_ids = std.StringHashMap(void).init(self.allocator);

        const contract_name = if (self.contract_count == 1)
            self.primary_contract_name.?
        else
            "bundle";

        return .{
            .allocator = self.allocator,
            .contract_name = contract_name,
            .contract_count = self.contract_count,
            .callables = callables,
            .types = types,
            .type_lookup = lookup,
        };
    }

    fn collectGlobals(self: *AbiGenerator, ast_nodes: []lib.AstNode) void {
        for (ast_nodes) |*node| {
            switch (node.*) {
                .StructDecl => |*struct_decl| {
                    if (!self.global_structs.contains(struct_decl.name)) {
                        self.global_structs.put(struct_decl.name, struct_decl) catch {};
                    }
                },
                .EnumDecl => |*enum_decl| {
                    if (!self.global_enums.contains(enum_decl.name)) {
                        self.global_enums.put(enum_decl.name, enum_decl) catch {};
                    }
                },
                else => {},
            }
        }
    }

    fn processContract(self: *AbiGenerator, contract: *const lib.ContractNode) AbiError!void {
        var analysis = state_tracker.analyzeContract(self.allocator, contract) catch {
            // ABI emission should stay resilient; emit without effect metadata if analysis fails.
            var empty = state_tracker.ContractStateAnalysis.init(self.allocator, contract.name);
            defer empty.deinit();
            try self.processContractBody(contract, &empty);
            return;
        };
        defer analysis.deinit();

        try self.processContractBody(contract, &analysis);
    }

    fn processContractBody(
        self: *AbiGenerator,
        contract: *const lib.ContractNode,
        analysis: *const state_tracker.ContractStateAnalysis,
    ) AbiError!void {
        for (contract.body) |*member| {
            switch (member.*) {
                .Function => |func| {
                    if (func.visibility == .Public) {
                        const function_analysis = analysis.functions.get(func.name);
                        try self.addFunctionCallable(contract, &func, function_analysis);
                    }
                },
                .ErrorDecl => |error_decl| {
                    try self.addErrorCallable(contract, &error_decl);
                },
                .LogDecl => |log_decl| {
                    try self.addEventCallable(contract, &log_decl);
                },
                else => {},
            }
        }
    }

    fn addFunctionCallable(
        self: *AbiGenerator,
        contract: *const lib.ContractNode,
        func: *const lib.FunctionNode,
        function_analysis: ?state_tracker.FunctionStateAnalysis,
    ) AbiError!void {
        var signature_types = std.ArrayList([]const u8){};
        defer signature_types.deinit(self.allocator);

        var inputs = std.ArrayList(AbiInput){};
        defer inputs.deinit(self.allocator);

        for (func.parameters) |param| {
            const resolved = try self.resolveTypeInfo(param.type_info, contract);
            try signature_types.append(self.allocator, resolved.wire_type);
            try inputs.append(self.allocator, .{
                .name = param.name,
                .type_id = resolved.type_id,
                .indexed = null,
            });
        }

        var outputs = std.ArrayList(AbiInput){};
        defer outputs.deinit(self.allocator);

        if (func.return_type_info) |return_type| {
            if (!isVoidType(return_type)) {
                const resolved_return = try self.resolveTypeInfo(return_type, contract);
                try outputs.append(self.allocator, .{
                    .name = "result",
                    .type_id = resolved_return.type_id,
                    .indexed = null,
                });
            }
        }

        const signature = try self.buildSignature(func.name, signature_types.items);
        const selector = try keccakSelectorHex(self.allocator, signature);
        const id = try self.buildCallableId(contract.name, signature);
        try self.ensureCallableIdUnique(id);

        var effects = std.ArrayList(AbiEffect){};
        defer effects.deinit(self.allocator);
        try self.buildEffects(function_analysis, &effects);

        const callable = AbiCallable{
            .id = id,
            .kind = .function,
            .name = func.name,
            .signature = signature,
            .selector = selector,
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .outputs = try outputs.toOwnedSlice(self.allocator),
            .effects = try effects.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };

        try self.callables.append(self.allocator, callable);
    }

    fn addErrorCallable(self: *AbiGenerator, contract: *const lib.ContractNode, err: *const lib.ast.Statements.ErrorDeclNode) AbiError!void {
        var signature_types = std.ArrayList([]const u8){};
        defer signature_types.deinit(self.allocator);

        var inputs = std.ArrayList(AbiInput){};
        defer inputs.deinit(self.allocator);

        if (err.parameters) |params| {
            for (params) |param| {
                const resolved = try self.resolveTypeInfo(param.type_info, contract);
                try signature_types.append(self.allocator, resolved.wire_type);
                try inputs.append(self.allocator, .{
                    .name = param.name,
                    .type_id = resolved.type_id,
                    .indexed = null,
                });
            }
        }

        const signature = try self.buildSignature(err.name, signature_types.items);
        const selector = try keccakSelectorHex(self.allocator, signature);
        const id = try self.buildCallableId(contract.name, signature);
        try self.ensureCallableIdUnique(id);

        const callable = AbiCallable{
            .id = id,
            .kind = .@"error",
            .name = err.name,
            .signature = signature,
            .selector = selector,
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .outputs = &.{},
            .effects = &.{},
            .allocator = self.allocator,
        };

        try self.callables.append(self.allocator, callable);
    }

    fn addEventCallable(self: *AbiGenerator, contract: *const lib.ContractNode, log_decl: *const lib.ast.LogDeclNode) AbiError!void {
        var signature_types = std.ArrayList([]const u8){};
        defer signature_types.deinit(self.allocator);

        var inputs = std.ArrayList(AbiInput){};
        defer inputs.deinit(self.allocator);

        for (log_decl.fields) |field| {
            const resolved = try self.resolveTypeInfo(field.type_info, contract);
            try signature_types.append(self.allocator, resolved.wire_type);
            try inputs.append(self.allocator, .{
                .name = field.name,
                .type_id = resolved.type_id,
                .indexed = field.indexed,
            });
        }

        const signature = try self.buildSignature(log_decl.name, signature_types.items);
        const id = try self.buildCallableId(contract.name, signature);
        try self.ensureCallableIdUnique(id);

        const callable = AbiCallable{
            .id = id,
            .kind = .event,
            .name = log_decl.name,
            .signature = signature,
            .selector = null,
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .outputs = &.{},
            .effects = &.{},
            .allocator = self.allocator,
        };

        try self.callables.append(self.allocator, callable);
    }

    fn buildEffects(
        self: *AbiGenerator,
        function_analysis: ?state_tracker.FunctionStateAnalysis,
        effects: *std.ArrayList(AbiEffect),
    ) AbiError!void {
        if (function_analysis == null) {
            return;
        }

        var read_paths = std.ArrayList([]const u8){};
        defer read_paths.deinit(self.allocator);

        var write_paths = std.ArrayList([]const u8){};
        defer write_paths.deinit(self.allocator);

        var reads_it = function_analysis.?.reads_set.iterator();
        while (reads_it.next()) |entry| {
            const dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try read_paths.append(self.allocator, dup);
        }

        var writes_it = function_analysis.?.writes_set.iterator();
        while (writes_it.next()) |entry| {
            const dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try write_paths.append(self.allocator, dup);
        }

        sortStringArrayList(&read_paths);
        sortStringArrayList(&write_paths);

        for (read_paths.items) |path| {
            try effects.append(self.allocator, .{
                .kind = .reads,
                .path = path,
                .event_id = null,
            });
        }

        for (write_paths.items) |path| {
            try effects.append(self.allocator, .{
                .kind = .writes,
                .path = path,
                .event_id = null,
            });
        }
    }

    fn buildSignature(self: *AbiGenerator, name: []const u8, types: []const []const u8) AbiError![]const u8 {
        const joined = try std.mem.join(self.allocator, ",", types);
        defer self.allocator.free(joined);
        return std.fmt.allocPrint(self.allocator, "{s}({s})", .{ name, joined });
    }

    fn buildCallableId(self: *AbiGenerator, contract_name: []const u8, signature: []const u8) AbiError![]const u8 {
        if (self.contract_count > 1) {
            return std.fmt.allocPrint(self.allocator, "c:{s}.{s}", .{ contract_name, signature });
        }
        return std.fmt.allocPrint(self.allocator, "c:{s}", .{signature});
    }

    fn ensureCallableIdUnique(self: *AbiGenerator, callable_id: []const u8) AbiError!void {
        if (self.callable_ids.contains(callable_id)) {
            return error.DuplicateCallableId;
        }
        try self.callable_ids.put(callable_id, {});
    }

    fn resolveTypeInfo(self: *AbiGenerator, type_info: lib.ast.type_info.TypeInfo, contract: *const lib.ContractNode) AbiError!ResolvedType {
        const ora_type = type_info.ora_type orelse return error.UnresolvedType;

        var stack = std.ArrayList([]const u8){};
        defer stack.deinit(self.allocator);

        return self.resolveOraType(ora_type, contract, &stack);
    }

    fn resolveOraType(
        self: *AbiGenerator,
        ora_type: lib.ast.type_info.OraType,
        contract: *const lib.ContractNode,
        stack: *std.ArrayList([]const u8),
    ) AbiError!ResolvedType {
        if (primitiveInfo(ora_type)) |info| {
            var node = AbiTypeNode{
                .kind = .primitive,
                .allocator = self.allocator,
                .name = info.name,
                .wire_type = try self.allocator.dupe(u8, info.wire_type),
                .ui_widget = defaultWidgetForWireType(info.wire_type),
            };
            const type_id = try self.ensureTypeNode(&node);
            const idx = self.type_lookup.get(type_id).?;
            return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
        }

        return switch (ora_type) {
            .array => |arr| self.resolveArrayType(arr.elem.*, arr.len, contract, stack),
            .slice => |elem| self.resolveSliceType(elem.*, contract, stack),
            .tuple => |members| self.resolveTupleType(members, contract, stack),
            .anonymous_struct => |fields| self.resolveAnonymousStructType(fields, contract, stack),
            .struct_type => |name| self.resolveNamedStructType(name, contract, stack),
            .enum_type => |name| self.resolveNamedEnumType(name, contract, stack),
            .error_union => |succ| self.resolveOraType(succ.*, contract, stack),
            ._union => |members| self.resolveUnionType(members, contract, stack),
            .min_value => |mv| blk: {
                const min_str = try std.fmt.allocPrint(self.allocator, "{d}", .{mv.min});
                errdefer self.allocator.free(min_str);
                const predicate = try self.buildMinPredicate(mv.min);
                break :blk self.resolveRefinementType(mv.base.*, contract, stack, predicate, .{
                    .min = min_str,
                    .widget = "number",
                });
            },
            .max_value => |mv| blk: {
                const max_str = try std.fmt.allocPrint(self.allocator, "{d}", .{mv.max});
                errdefer self.allocator.free(max_str);
                const predicate = try self.buildMaxPredicate(mv.max);
                break :blk self.resolveRefinementType(mv.base.*, contract, stack, predicate, .{
                    .max = max_str,
                    .widget = "number",
                });
            },
            .in_range => |ir| blk: {
                const min_str = try std.fmt.allocPrint(self.allocator, "{d}", .{ir.min});
                errdefer self.allocator.free(min_str);
                const max_str = try std.fmt.allocPrint(self.allocator, "{d}", .{ir.max});
                errdefer self.allocator.free(max_str);
                const predicate = try self.buildRangePredicate(ir.min, ir.max);
                break :blk self.resolveRefinementType(ir.base.*, contract, stack, predicate, .{
                    .min = min_str,
                    .max = max_str,
                    .widget = "number",
                });
            },
            .scaled => |s| self.resolveRefinementType(s.base.*, contract, stack, try self.buildScaledPredicate(s.decimals), .{
                .decimals = s.decimals,
                .widget = "number",
            }),
            .exact => |base| self.resolveRefinementType(base.*, contract, stack, try self.buildExactPredicate(), .{
                .widget = "number",
            }),
            .non_zero_address => self.resolveRefinementType(.address, contract, stack, try self.buildNonZeroAddressPredicate(), .{
                .widget = "address",
            }),
            .void => error.UnsupportedAbiType,
            .map => error.UnsupportedAbiType,
            .function => error.UnsupportedAbiType,
            .contract_type => error.UnsupportedAbiType,
            .module => error.UnsupportedAbiType,
            else => error.UnsupportedAbiType,
        };
    }

    fn resolveArrayType(
        self: *AbiGenerator,
        element: lib.ast.type_info.OraType,
        len: u64,
        contract: *const lib.ContractNode,
        stack: *std.ArrayList([]const u8),
    ) AbiError!ResolvedType {
        const elem = try self.resolveOraType(element, contract, stack);
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
        self: *AbiGenerator,
        element: lib.ast.type_info.OraType,
        contract: *const lib.ContractNode,
        stack: *std.ArrayList([]const u8),
    ) AbiError!ResolvedType {
        const elem = try self.resolveOraType(element, contract, stack);
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
        self: *AbiGenerator,
        members: []const lib.ast.type_info.OraType,
        contract: *const lib.ContractNode,
        stack: *std.ArrayList([]const u8),
    ) AbiError!ResolvedType {
        var component_ids = try self.allocator.alloc([]const u8, members.len);
        errdefer self.allocator.free(component_ids);

        var wire_parts = std.ArrayList([]const u8){};
        defer wire_parts.deinit(self.allocator);

        for (members, 0..) |member, i| {
            const resolved = try self.resolveOraType(member, contract, stack);
            component_ids[i] = resolved.type_id;
            try wire_parts.append(self.allocator, resolved.wire_type);
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
        self: *AbiGenerator,
        fields: []const lib.ast.type_info.AnonymousStructFieldType,
        contract: *const lib.ContractNode,
        stack: *std.ArrayList([]const u8),
    ) AbiError!ResolvedType {
        var members = try self.allocator.alloc(lib.ast.type_info.OraType, fields.len);
        defer self.allocator.free(members);

        for (fields, 0..) |field, i| {
            members[i] = field.typ.*;
        }

        return self.resolveTupleType(members, contract, stack);
    }

    fn resolveNamedStructType(
        self: *AbiGenerator,
        name: []const u8,
        contract: *const lib.ContractNode,
        stack: *std.ArrayList([]const u8),
    ) AbiError!ResolvedType {
        if (containsString(stack.items, name)) return error.RecursiveType;
        try stack.append(self.allocator, name);
        defer _ = stack.pop();

        const struct_decl = self.findStructDecl(contract, name) orelse return error.UnknownStructType;

        var fields = try self.allocator.alloc(AbiFieldRef, struct_decl.fields.len);
        errdefer self.allocator.free(fields);

        var wire_parts = std.ArrayList([]const u8){};
        defer wire_parts.deinit(self.allocator);

        for (struct_decl.fields, 0..) |field, i| {
            const resolved = try self.resolveTypeInfo(field.type_info, contract);
            fields[i] = .{
                .name = field.name,
                .type_id = resolved.type_id,
            };
            try wire_parts.append(self.allocator, resolved.wire_type);
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
        self: *AbiGenerator,
        name: []const u8,
        contract: *const lib.ContractNode,
        stack: *std.ArrayList([]const u8),
    ) AbiError!ResolvedType {
        _ = stack;

        const enum_decl = self.findEnumDecl(contract, name) orelse return error.UnknownEnumType;
        const repr_type = try self.resolveEnumReprType(enum_decl, contract);

        var variants = try self.allocator.alloc(AbiEnumVariant, enum_decl.variants.len);
        errdefer self.allocator.free(variants);

        for (enum_decl.variants, 0..) |variant, i| {
            variants[i] = .{
                .name = variant.name,
                .value = extractEnumVariantValue(&variant, @intCast(i)),
            };
        }

        var node = AbiTypeNode{
            .kind = .enum_,
            .allocator = self.allocator,
            .name = name,
            .wire_type = try self.allocator.dupe(u8, repr_type.wire_type),
            .repr_type = repr_type.type_id,
            .variants = variants,
            .ui_label = name,
            .ui_widget = "select",
        };

        const type_id = try self.ensureTypeNode(&node);
        const idx = self.type_lookup.get(type_id).?;
        return .{ .type_id = self.types.items[idx].type_id.?, .wire_type = self.types.items[idx].wire_type.? };
    }

    fn resolveEnumReprType(self: *AbiGenerator, enum_decl: *const lib.ast.EnumDeclNode, contract: *const lib.ContractNode) AbiError!ResolvedType {
        if (enum_decl.underlying_type_info) |underlying| {
            if (underlying.ora_type) |ot| {
                const repr_ora: lib.ast.type_info.OraType = switch (ot) {
                    .u8, .i8 => .u8,
                    .u16, .i16 => .u16,
                    .u32, .i32 => .u32,
                    .u64, .i64 => .u64,
                    .u128, .i128 => .u128,
                    .u256, .i256 => .u256,
                    else => return error.InvalidEnumRepr,
                };
                var stack = std.ArrayList([]const u8){};
                defer stack.deinit(self.allocator);
                return self.resolveOraType(repr_ora, contract, &stack);
            }
        }

        var stack = std.ArrayList([]const u8){};
        defer stack.deinit(self.allocator);
        return self.resolveOraType(.u32, contract, &stack);
    }

    fn resolveUnionType(
        self: *AbiGenerator,
        members: []const lib.ast.type_info.OraType,
        contract: *const lib.ContractNode,
        stack: *std.ArrayList([]const u8),
    ) AbiError!ResolvedType {
        if (members.len == 0) return error.UnsupportedAbiType;

        if (members[0] == .error_union) {
            return self.resolveOraType(members[0].error_union.*, contract, stack);
        }

        return error.UnsupportedAbiType;
    }

    fn resolveRefinementType(
        self: *AbiGenerator,
        base_ora_type: lib.ast.type_info.OraType,
        contract: *const lib.ContractNode,
        stack: *std.ArrayList([]const u8),
        predicate_json: []const u8,
        ui_hints: UiHints,
    ) AbiError!ResolvedType {
        errdefer self.allocator.free(predicate_json);
        errdefer if (ui_hints.min) |min| self.allocator.free(min);
        errdefer if (ui_hints.max) |max| self.allocator.free(max);

        const base = try self.resolveOraType(base_ora_type, contract, stack);

        var node = AbiTypeNode{
            .kind = .refinement,
            .allocator = self.allocator,
            .base = base.type_id,
            .wire_type = try self.allocator.dupe(u8, base.wire_type),
            .predicate_json = predicate_json,
            .ui_label = ui_hints.label,
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

    fn ensureTypeNode(self: *AbiGenerator, node: *AbiTypeNode) AbiError![]const u8 {
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

        node.type_id = type_id;
        node.canonical_payload = payload;

        try self.types.append(self.allocator, node.*);
        const new_index = self.types.items.len - 1;
        try self.type_lookup.put(self.types.items[new_index].type_id.?, new_index);

        return self.types.items[new_index].type_id.?;
    }

    fn findStructDecl(self: *AbiGenerator, contract: *const lib.ContractNode, name: []const u8) ?*const lib.ast.StructDeclNode {
        for (contract.body) |*member| {
            if (member.* == .StructDecl and std.mem.eql(u8, member.StructDecl.name, name)) {
                return &member.StructDecl;
            }
        }
        return self.global_structs.get(name);
    }

    fn findEnumDecl(self: *AbiGenerator, contract: *const lib.ContractNode, name: []const u8) ?*const lib.ast.EnumDeclNode {
        for (contract.body) |*member| {
            if (member.* == .EnumDecl and std.mem.eql(u8, member.EnumDecl.name, name)) {
                return &member.EnumDecl;
            }
        }
        return self.global_enums.get(name);
    }

    fn buildMinPredicate(self: *AbiGenerator, min_value: u256) AbiError![]const u8 {
        const min_str = try std.fmt.allocPrint(self.allocator, "{d}", .{min_value});
        defer self.allocator.free(min_str);
        return std.fmt.allocPrint(
            self.allocator,
            "{{\"op\":\">=\",\"lhs\":{{\"var\":\"x\"}},\"rhs\":{{\"const\":\"{s}\"}}}}",
            .{min_str},
        );
    }

    fn buildMaxPredicate(self: *AbiGenerator, max_value: u256) AbiError![]const u8 {
        const max_str = try std.fmt.allocPrint(self.allocator, "{d}", .{max_value});
        defer self.allocator.free(max_str);
        return std.fmt.allocPrint(
            self.allocator,
            "{{\"op\":\"<=\",\"lhs\":{{\"var\":\"x\"}},\"rhs\":{{\"const\":\"{s}\"}}}}",
            .{max_str},
        );
    }

    fn buildRangePredicate(self: *AbiGenerator, min_value: u256, max_value: u256) AbiError![]const u8 {
        const min_str = try std.fmt.allocPrint(self.allocator, "{d}", .{min_value});
        defer self.allocator.free(min_str);
        const max_str = try std.fmt.allocPrint(self.allocator, "{d}", .{max_value});
        defer self.allocator.free(max_str);

        return std.fmt.allocPrint(
            self.allocator,
            "{{\"op\":\"&&\",\"lhs\":{{\"op\":\">=\",\"lhs\":{{\"var\":\"x\"}},\"rhs\":{{\"const\":\"{s}\"}}}},\"rhs\":{{\"op\":\"<=\",\"lhs\":{{\"var\":\"x\"}},\"rhs\":{{\"const\":\"{s}\"}}}}}}",
            .{ min_str, max_str },
        );
    }

    fn buildScaledPredicate(self: *AbiGenerator, decimals: u32) AbiError![]const u8 {
        const scale = try pow10String(self.allocator, decimals);
        defer self.allocator.free(scale);

        return std.fmt.allocPrint(
            self.allocator,
            "{{\"op\":\"==\",\"lhs\":{{\"op\":\"%\",\"lhs\":{{\"var\":\"x\"}},\"rhs\":{{\"const\":\"{s}\"}}}},\"rhs\":{{\"const\":\"0\"}}}}",
            .{scale},
        );
    }

    fn buildExactPredicate(self: *AbiGenerator) AbiError![]const u8 {
        return self.allocator.dupe(u8, "{\"op\":\"==\",\"lhs\":{\"var\":\"x\"},\"rhs\":{\"var\":\"x\"}}");
    }

    fn buildNonZeroAddressPredicate(self: *AbiGenerator) AbiError![]const u8 {
        return self.allocator.dupe(u8, "{\"op\":\"!=\",\"lhs\":{\"var\":\"x\"},\"rhs\":{\"const\":\"0\"}}");
    }
};

fn primitiveInfo(ora_type: lib.ast.type_info.OraType) ?struct { name: []const u8, wire_type: []const u8 } {
    return switch (ora_type) {
        .u8 => .{ .name = "u8", .wire_type = "uint8" },
        .u16 => .{ .name = "u16", .wire_type = "uint16" },
        .u32 => .{ .name = "u32", .wire_type = "uint32" },
        .u64 => .{ .name = "u64", .wire_type = "uint64" },
        .u128 => .{ .name = "u128", .wire_type = "uint128" },
        .u256 => .{ .name = "u256", .wire_type = "uint256" },
        .i8 => .{ .name = "i8", .wire_type = "int8" },
        .i16 => .{ .name = "i16", .wire_type = "int16" },
        .i32 => .{ .name = "i32", .wire_type = "int32" },
        .i64 => .{ .name = "i64", .wire_type = "int64" },
        .i128 => .{ .name = "i128", .wire_type = "int128" },
        .i256 => .{ .name = "i256", .wire_type = "int256" },
        .bool => .{ .name = "bool", .wire_type = "bool" },
        .address => .{ .name = "address", .wire_type = "address" },
        .bytes => .{ .name = "bytes", .wire_type = "bytes" },
        .string => .{ .name = "string", .wire_type = "string" },
        else => null,
    };
}

fn isVoidType(type_info: lib.ast.type_info.TypeInfo) bool {
    if (type_info.ora_type) |ot| {
        return ot == .void;
    }
    return false;
}

fn containsString(values: []const []const u8, needle: []const u8) bool {
    for (values) |value| {
        if (std.mem.eql(u8, value, needle)) return true;
    }
    return false;
}

fn callableKindString(kind: CallableKind) []const u8 {
    return switch (kind) {
        .function => "function",
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

fn extractEnumVariantValue(variant: *const lib.ast.EnumVariant, fallback: i64) i64 {
    if (variant.resolved_value) |resolved| {
        return @intCast(resolved);
    }

    if (variant.value) |value_expr| {
        if (value_expr == .Literal and value_expr.Literal == .Integer) {
            const raw = value_expr.Literal.Integer.value;
            if (parseSignedIntLiteral(raw)) |value| {
                return value;
            }
        }
    }

    return fallback;
}

fn parseSignedIntLiteral(raw: []const u8) ?i64 {
    var cleaned_storage = std.ArrayList(u8){};
    defer cleaned_storage.deinit(std.heap.page_allocator);

    for (raw) |ch| {
        if (ch != '_') {
            cleaned_storage.append(std.heap.page_allocator, ch) catch return null;
        }
    }

    const cleaned = cleaned_storage.items;
    return std.fmt.parseInt(i64, cleaned, 0) catch null;
}

fn pow10String(allocator: std.mem.Allocator, decimals: u32) ![]const u8 {
    const len: usize = @intCast(decimals + 1);
    var out = try allocator.alloc(u8, len);
    out[0] = '1';
    var i: usize = 1;
    while (i < out.len) : (i += 1) {
        out[i] = '0';
    }
    return out;
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
