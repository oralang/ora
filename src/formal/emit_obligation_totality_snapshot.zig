//! Emits `formal/Ora/Generated/ObligationTotalitySnapshot.lean` — data-only
//! rows for the obligation Lean-support totality fixture. The trusted Lean
//! check lives in `formal/Ora/ObligationTotalitySync.lean`.

const std = @import("std");
const formal = @import("ora_formal");

const obligation = formal.obligation;
const obligation_to_lean = formal.obligation_to_lean;
const type_builtin = formal.builtin;

const header =
    \\/-
    \\GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
    \\`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
    \\`import` to this file. It contains only primitive fixture rows emitted from
    \\the compiler. The TRUSTED checks live in `Ora/ObligationTotalitySync.lean`.
    \\
    \\Regenerate with `scripts/check-formal-sync.sh`. Source:
    \\src/formal/emit_obligation_totality_snapshot.zig.
    \\-/
    \\
    \\namespace Ora.Generated
    \\
    \\
;

const row_type =
    \\def obligationTotalityRows :
    \\    List (String ×
    \\      List (String × Bool × String × String × String × Nat × Nat × Option Nat × List Nat ×
    \\        (String × String × List String × List (String × String))) ×
    \\      List (Nat × String × Option Nat) ×
    \\      Bool × Bool × String × Nat ×
    \\      (String ×
    \\        List (String × String × List String × List (String × String)) ×
    \\        List (String × String × List String × List (String × String)) ×
    \\        List (String × Nat × Nat × Nat × Nat × Nat ×
    \\          (String × String × List String × List (String × String)) ×
    \\          (String × String × List String × List (String × String)) × Nat)) ×
    \\      (String × String ×
    \\        Option (String × String × List String × List (String × String)) ×
    \\        Option (String × String × List String × List (String × String)) ×
    \\        Option Nat × String)) :=
    \\  [
;

const Target = union(enum) {
    formula: obligation.TermId,
    effect: obligation.EffectFrameGoal,
    resource: obligation.ResourceGoal,
};

const empty_place: obligation.PlaceRef = .{
    .root = "",
    .region = .none,
};

const default_effect: obligation.EffectFrameGoal = .{
    .relation = .write_covered_by_modifies,
};

const default_resource: obligation.ResourceGoal = .{
    .op = .move,
    .domain = "",
    .property = .amount_non_negative,
};

fn boolText(value: bool) []const u8 {
    return if (value) "true" else "false";
}

fn writeLeanString(out: anytype, value: []const u8) !void {
    try out.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '\\' => try out.writeAll("\\\\"),
            '"' => try out.writeAll("\\\""),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            else => try out.writeByte(byte),
        }
    }
    try out.writeByte('"');
}

fn writeOptionalTermId(out: anytype, value: ?obligation.TermId) !void {
    if (value) |actual| {
        try out.print("some {d}", .{actual});
    } else {
        try out.writeAll("none");
    }
}

fn writeNatList(out: anytype, values: []const obligation.TermId) !void {
    if (values.len == 0) return out.writeAll("[]");
    try out.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try out.writeAll(", ");
        try out.print("{d}", .{value});
    }
    try out.writeByte(']');
}

fn writeTypeName(out: anytype, ty: ?obligation.TypeRef) !void {
    const value = ty orelse return writeLeanString(out, "");
    switch (value) {
        .spelling => |name| try writeLeanString(out, name),
        .compiler_type_id => |id| {
            try out.writeByte('"');
            try out.print("compiler:{d}", .{id});
            try out.writeByte('"');
        },
    }
}

fn regionName(region: obligation.RegionRef) []const u8 {
    return switch (region) {
        .none => "none",
        .storage => "storage",
        .memory => "memory",
        .transient => "transient",
        .calldata => "calldata",
    };
}

fn writeFreeVarIdString(out: anytype, id: obligation.FreeVarId) !void {
    try out.print("\"file:{d}:pattern:{d}\"", .{ id.file_id, id.pattern_id });
}

fn writeStringList(out: anytype, values: []const []const u8) !void {
    if (values.len == 0) return out.writeAll("[]");
    try out.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try out.writeAll(", ");
        try writeLeanString(out, value);
    }
    try out.writeByte(']');
}

fn writePlaceKey(out: anytype, key: obligation.PlaceKey) !void {
    try out.writeByte('(');
    switch (key) {
        .parameter => |id| {
            try writeLeanString(out, "parameter");
            try out.writeAll(", ");
            try writeFreeVarIdString(out, id);
        },
        .comptime_parameter => |index| {
            try writeLeanString(out, "comptime_parameter");
            try out.print(", \"{d}\"", .{index});
        },
        .comptime_range_parameter => |index| {
            try writeLeanString(out, "comptime_range_parameter");
            try out.print(", \"{d}\"", .{index});
        },
        .constant => |value| {
            try writeLeanString(out, "constant");
            try out.writeAll(", ");
            try writeLeanString(out, value);
        },
        .msg_sender => {
            try writeLeanString(out, "msg_sender");
            try out.writeAll(", \"\"");
        },
        .tx_origin => {
            try writeLeanString(out, "tx_origin");
            try out.writeAll(", \"\"");
        },
        .unknown => {
            try writeLeanString(out, "unknown");
            try out.writeAll(", \"\"");
        },
    }
    try out.writeByte(')');
}

fn writePlaceKeyList(out: anytype, keys: []const obligation.PlaceKey) !void {
    if (keys.len == 0) return out.writeAll("[]");
    try out.writeByte('[');
    for (keys, 0..) |key, index| {
        if (index != 0) try out.writeAll(", ");
        try writePlaceKey(out, key);
    }
    try out.writeByte(']');
}

fn writePlace(out: anytype, place: obligation.PlaceRef) !void {
    try out.writeByte('(');
    try writeLeanString(out, place.root);
    try out.writeAll(", ");
    try writeLeanString(out, regionName(place.region));
    try out.writeAll(", ");
    try writeStringList(out, place.fields);
    try out.writeAll(", ");
    try writePlaceKeyList(out, place.keys);
    try out.writeByte(')');
}

fn writeOptionalPlace(out: anytype, place: ?obligation.PlaceRef) !void {
    if (place) |actual| {
        try out.writeAll("some ");
        try writePlace(out, actual);
    } else {
        try out.writeAll("none");
    }
}

fn unaryName(op: obligation.UnaryOp) []const u8 {
    return switch (op) {
        .not => "not",
        .neg => "neg",
    };
}

fn binaryName(op: obligation.BinaryOp) []const u8 {
    return switch (op) {
        .eq => "eq",
        .ne => "ne",
        .lt => "lt",
        .le => "le",
        .gt => "gt",
        .ge => "ge",
        .slt => "slt",
        .sle => "sle",
        .sgt => "sgt",
        .sge => "sge",
        .add => "add",
        .sub => "sub",
        .mul => "mul",
        .div => "div",
        .mod => "mod",
        .and_ => "and",
        .or_ => "or",
        .implies => "implies",
    };
}

fn quantifierName(quantifier: obligation.Quantifier) []const u8 {
    return switch (quantifier) {
        .forall => "forall",
        .exists => "exists",
    };
}

fn writeRawTerm(
    out: anytype,
    tag: []const u8,
    bool_value: bool,
    text: []const u8,
    ty: ?obligation.TypeRef,
    name: []const u8,
    lhs: usize,
    rhs: usize,
    condition: ?obligation.TermId,
    args: []const obligation.TermId,
    place: obligation.PlaceRef,
) !void {
    try out.writeByte('(');
    try writeLeanString(out, tag);
    try out.writeAll(", ");
    try out.writeAll(boolText(bool_value));
    try out.writeAll(", ");
    try writeLeanString(out, text);
    try out.writeAll(", ");
    try writeTypeName(out, ty);
    try out.writeAll(", ");
    try writeLeanString(out, name);
    try out.print(", {d}, {d}, ", .{ lhs, rhs });
    try writeOptionalTermId(out, condition);
    try out.writeAll(", ");
    try writeNatList(out, args);
    try out.writeAll(", ");
    try writePlace(out, place);
    try out.writeByte(')');
}

fn writeTerm(out: anytype, term: obligation.Term) !void {
    switch (term) {
        .bool_lit => |value| try writeRawTerm(out, "bool_lit", value, "", null, "", 0, 0, null, &.{}, empty_place),
        .int_lit => |literal| try writeRawTerm(out, "int_lit", false, literal.value, literal.ty, "", 0, 0, null, &.{}, empty_place),
        .variable => |variable| switch (variable) {
            .free => |free| try writeRawTerm(out, "free_var", false, "", free.ty, free.name, free.id.file_id, free.id.pattern_id, null, &.{}, empty_place),
            .bound => |bound| try writeRawTerm(out, "bound_var", false, "", bound.ty, bound.name, bound.index, 0, null, &.{}, empty_place),
        },
        .old => |operand| try writeRawTerm(out, "old", false, "", null, "", operand, 0, null, &.{}, empty_place),
        .result => try writeRawTerm(out, "result", false, "", null, "", 0, 0, null, &.{}, empty_place),
        .place_read => |place| try writeRawTerm(out, "place_read", false, "", null, "", 0, 0, null, &.{}, place),
        .unary => |unary| try writeRawTerm(out, "unary", false, unaryName(unary.op), null, "", unary.operand, 0, null, &.{}, empty_place),
        .binary => |binary| try writeRawTerm(out, "binary", false, binaryName(binary.op), binary.ty, "", binary.lhs, binary.rhs, null, &.{}, empty_place),
        .refinement_predicate => |predicate| try writeRawTerm(out, "refinement", false, predicate.name, null, "", predicate.value, 0, null, predicate.args, empty_place),
        .quantified => |quantified| try writeRawTerm(out, "quantified", false, quantifierName(quantified.quantifier), quantified.binder.ty, quantified.binder.name, quantified.body, 0, quantified.condition, &.{}, empty_place),
    }
}

fn writeTermList(out: anytype, terms: []const obligation.Term) !void {
    if (terms.len == 0) return out.writeAll("[]");
    try out.writeByte('[');
    for (terms, 0..) |term, index| {
        if (index != 0) try out.writeAll(", ");
        try writeTerm(out, term);
    }
    try out.writeByte(']');
}

fn assumptionKindName(kind: obligation.AssumptionKind) []const u8 {
    return switch (kind) {
        .requires => "requires",
        .assume => "assume",
        .path_assume => "path_assume",
        .env_assume => "env_assume",
        .binding => "binding",
        .two_state_linkage => "two_state_linkage",
        .frame => "frame",
        .loop_invariant => "loop_invariant",
        .callee_obligation => "callee_obligation",
        .callee_ensures => "callee_ensures",
        .ghost_axiom => "ghost_axiom",
        .goal => "goal",
    };
}

fn writeAssumptions(out: anytype, assumptions: []const obligation.Assumption) !void {
    if (assumptions.len == 0) return out.writeAll("[]");
    try out.writeByte('[');
    for (assumptions, 0..) |assumption, index| {
        if (index != 0) try out.writeAll(", ");
        try out.writeByte('(');
        try out.print("{d}, ", .{assumption.id});
        try writeLeanString(out, assumptionKindName(assumption.kind));
        try out.writeAll(", ");
        if (assumption.formula) |formula| {
            switch (formula) {
                .term => |term_id| try out.print("some {d}", .{term_id}),
                .origin_value => return error.UnsupportedFixtureAssumptionFormula,
            }
        } else {
            try out.writeAll("none");
        }
        try out.writeByte(')');
    }
    try out.writeByte(']');
}

fn effectRelationName(relation: obligation.EffectFrameRelation) []const u8 {
    return switch (relation) {
        .write_covered_by_modifies => "write_covered_by_modifies",
        .read_preserved_by_frame => "read_preserved_by_frame",
        .read_preserved_by_key_evidence => "read_preserved_by_key_evidence",
        .lock_covers_write => "lock_covers_write",
        .external_call_frame => "external_call_frame",
    };
}

fn keyEvidenceKindName(kind: obligation.KeyDisjointEvidenceKind) []const u8 {
    return switch (kind) {
        .free_var_disequality => "free_var_disequality",
    };
}

fn writePlaceList(out: anytype, places: []const obligation.PlaceRef) !void {
    if (places.len == 0) return out.writeAll("[]");
    try out.writeByte('[');
    for (places, 0..) |place, index| {
        if (index != 0) try out.writeAll(", ");
        try writePlace(out, place);
    }
    try out.writeByte(']');
}

fn writeEvidenceList(out: anytype, evidence: []const obligation.KeyDisjointEvidence) !void {
    if (evidence.len == 0) return out.writeAll("[]");
    try out.writeByte('[');
    for (evidence, 0..) |item, index| {
        if (index != 0) try out.writeAll(", ");
        try out.writeByte('(');
        try writeLeanString(out, keyEvidenceKindName(item.kind));
        try out.print(", {d}, {d}, {d}, {d}, {d}, ", .{
            item.assumption_id,
            item.lhs.file_id,
            item.lhs.pattern_id,
            item.rhs.file_id,
            item.rhs.pattern_id,
        });
        try writePlace(out, item.read);
        try out.writeAll(", ");
        try writePlace(out, item.write);
        try out.print(", {d})", .{item.key_index});
    }
    try out.writeByte(']');
}

fn writeEffect(out: anytype, effect: obligation.EffectFrameGoal) !void {
    try out.writeByte('(');
    try writeLeanString(out, effectRelationName(effect.relation));
    try out.writeAll(", ");
    try writePlaceList(out, effect.declared);
    try out.writeAll(", ");
    try writePlaceList(out, effect.actual);
    try out.writeAll(", ");
    try writeEvidenceList(out, effect.evidence);
    try out.writeByte(')');
}

fn resourceOperationName(op: obligation.ResourceOperation) []const u8 {
    return switch (op) {
        .move => "move",
        .create => "create",
        .destroy => "destroy",
    };
}

fn resourcePropertyName(property: obligation.ResourceProperty) []const u8 {
    return switch (property) {
        .amount_non_negative => "amount_non_negative",
        .source_sufficient => "source_sufficient",
        .destination_no_overflow => "destination_no_overflow",
        .same_place_identity => "same_place_identity",
        .conservation => "conservation",
    };
}

fn writeResource(out: anytype, resource: obligation.ResourceGoal) !void {
    try out.writeByte('(');
    try writeLeanString(out, resourceOperationName(resource.op));
    try out.writeAll(", ");
    try writeLeanString(out, resource.domain);
    try out.writeAll(", ");
    try writeOptionalPlace(out, resource.source);
    try out.writeAll(", ");
    try writeOptionalPlace(out, resource.destination);
    try out.writeAll(", ");
    if (resource.amount) |formula| {
        try out.print("some {d}", .{formula.term});
    } else {
        try out.writeAll("none");
    }
    try out.writeAll(", ");
    try writeLeanString(out, resourcePropertyName(resource.property));
    try out.writeByte(')');
}

fn targetKindName(target: Target) []const u8 {
    return switch (target) {
        .formula => "formula",
        .effect => "effect",
        .resource => "resource",
    };
}

fn targetTerm(target: Target) obligation.TermId {
    return switch (target) {
        .formula => |id| id,
        else => 0,
    };
}

fn targetEffect(target: Target) obligation.EffectFrameGoal {
    return switch (target) {
        .effect => |effect| effect,
        else => default_effect,
    };
}

fn targetResource(target: Target) obligation.ResourceGoal {
    return switch (target) {
        .resource => |resource| resource,
        else => default_resource,
    };
}

fn writeFixture(
    out: anytype,
    first: *bool,
    label: []const u8,
    terms: []const obligation.Term,
    assumptions: []const obligation.Assumption,
    target: Target,
) !void {
    try writeFixtureWithExpectation(out, first, label, terms, assumptions, true, target);
}

fn writeFixtureWithExpectation(
    out: anytype,
    first: *bool,
    label: []const u8,
    terms: []const obligation.Term,
    assumptions: []const obligation.Assumption,
    must_support: bool,
    target: Target,
) !void {
    const owner: obligation.Owner = .{ .function = .{ .name = "obligation_totality_fixture" } };
    const obligation_ids = [_]obligation.Id{1};
    var assumption_ids_buffer: [8]obligation.Id = undefined;
    if (assumptions.len > assumption_ids_buffer.len) return error.TooManyFixtureAssumptions;
    for (assumptions, 0..) |assumption, index| assumption_ids_buffer[index] = assumption.id;

    const kind: obligation.Kind = switch (target) {
        .formula => |id| .{ .logical = .{ .role = .ensures, .formula = .{ .term = id } } },
        .effect => |effect| .{ .effect_frame = effect },
        .resource => |resource| .{ .resource = resource },
    };
    const obligation_row = obligation.Obligation{
        .id = 1,
        .owner = owner,
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = kind,
    };
    const obligations = [_]obligation.Obligation{obligation_row};
    const query = obligation.VerificationQuery{
        .id = 2,
        .owner = owner,
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .backend = .lean,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .assumption_ids = assumption_ids_buffer[0..assumptions.len],
    };
    const queries = [_]obligation.VerificationQuery{query};
    const set = obligation.ObligationSet{
        .obligations = &obligations,
        .assumptions = assumptions,
        .queries = &queries,
        .terms = terms,
    };
    const zig_supported = switch (obligation_to_lean.querySemanticSupport(set, query)) {
        .supported => true,
        .unsupported => false,
    };

    if (!first.*) try out.writeAll(",\n   ");
    first.* = false;

    try out.writeByte('(');
    try writeLeanString(out, label);
    try out.writeAll(", ");
    try writeTermList(out, terms);
    try out.writeAll(", ");
    try writeAssumptions(out, assumptions);
    try out.writeAll(", ");
    try out.writeAll(boolText(must_support));
    try out.writeAll(", ");
    try out.writeAll(boolText(zig_supported));
    try out.writeAll(", ");
    try writeLeanString(out, targetKindName(target));
    try out.print(", {d}, ", .{targetTerm(target)});
    try writeEffect(out, targetEffect(target));
    try out.writeAll(", ");
    try writeResource(out, targetResource(target));
    try out.writeByte(')');
}

fn writeAssumptionFormulaFixture(
    out: anytype,
    first: *bool,
    label: []const u8,
    terms: []const obligation.Term,
    formula_id: obligation.TermId,
) !void {
    const assumptions = [_]obligation.Assumption{.{
        .id = 10,
        .owner = .{ .function = .{ .name = "obligation_totality_fixture" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .requires,
        .formula = .{ .term = formula_id },
    }};
    try writeFixture(out, first, label, terms, &assumptions, .{ .formula = formula_id });
}

fn writeNullAssumptionFixture(out: anytype, first: *bool) !void {
    const terms = [_]obligation.Term{.{ .bool_lit = true }};
    const assumptions = [_]obligation.Assumption{.{
        .id = 10,
        .owner = .{ .function = .{ .name = "obligation_totality_fixture" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .requires,
        .formula = null,
    }};
    try writeFixtureWithExpectation(
        out,
        first,
        "assumption_null_formula_is_unsupported",
        &terms,
        &assumptions,
        false,
        .{ .formula = 0 },
    );
}

const ty_bool: obligation.TypeRef = .{ .compiler_type_id = type_builtin.lookupBuiltinById(.bool).comptime_type_id };
const ty_u256: obligation.TypeRef = .{ .compiler_type_id = type_builtin.lookupBuiltinById(.u256).comptime_type_id };
const ty_i256: obligation.TypeRef = .{ .compiler_type_id = type_builtin.lookupBuiltinById(.i256).comptime_type_id };

const user_id: obligation.FreeVarId = .{ .file_id = 0, .pattern_id = 1 };
const other_id: obligation.FreeVarId = .{ .file_id = 0, .pattern_id = 2 };
const user_key = [_]obligation.PlaceKey{.{ .parameter = user_id }};
const other_key = [_]obligation.PlaceKey{.{ .parameter = other_id }};
const balance_user: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &user_key };
const balance_other: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &other_key };
const balance_root: obligation.PlaceRef = .{ .root = "balance", .region = .storage };
const reserve_root: obligation.PlaceRef = .{ .root = "reserve", .region = .storage };
const allowance_root: obligation.PlaceRef = .{ .root = "allowance", .region = .storage };

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var out_buffer: [1 << 17]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(header);
    try out.writeAll(row_type);
    var first = true;

    {
        const terms = [_]obligation.Term{.{ .bool_lit = true }};
        try writeFixture(out, &first, "bool_literal_formula", &terms, &.{}, .{ .formula = 0 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_bool_literal_formula", &terms, 0);
        try writeNullAssumptionFixture(out, &first);
    }
    {
        const terms = [_]obligation.Term{.{ .variable = .{ .free = .{ .id = user_id, .name = "flag", .ty = ty_bool } } }};
        try writeFixture(out, &first, "bool_free_variable_formula", &terms, &.{}, .{ .formula = 0 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_bool_free_variable_formula", &terms, 0);
    }
    {
        const args = [_]obligation.TermId{ 1, 2 };
        const terms = [_]obligation.Term{
            .{ .int_lit = .{ .value = "1", .ty = ty_u256 } },
            .{ .int_lit = .{ .value = "0", .ty = ty_u256 } },
            .{ .int_lit = .{ .value = "1", .ty = ty_u256 } },
            .{ .refinement_predicate = .{ .name = "InRange", .value = 0, .args = &args } },
        };
        try writeFixture(out, &first, "refinement_predicate_formula", &terms, &.{}, .{ .formula = 3 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_refinement_predicate_formula", &terms, 3);
    }
    {
        const terms = [_]obligation.Term{
            .{ .bool_lit = true },
            .{ .bool_lit = false },
            .{ .unary = .{ .op = .not, .operand = 1 } },
            .{ .binary = .{ .op = .and_, .lhs = 0, .rhs = 2 } },
            .{ .binary = .{ .op = .implies, .lhs = 3, .rhs = 0 } },
        };
        try writeFixture(out, &first, "boolean_connectives_formula", &terms, &.{}, .{ .formula = 4 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_boolean_connectives_formula", &terms, 4);
    }
    {
        const terms = [_]obligation.Term{
            .{ .int_lit = .{ .value = "1", .ty = ty_u256 } },
            .{ .int_lit = .{ .value = "1", .ty = ty_u256 } },
            .{ .binary = .{ .op = .add, .lhs = 0, .rhs = 1, .ty = ty_u256 } },
            .{ .binary = .{ .op = .sub, .lhs = 1, .rhs = 0, .ty = ty_u256 } },
            .{ .binary = .{ .op = .mul, .lhs = 0, .rhs = 1, .ty = ty_u256 } },
            .{ .binary = .{ .op = .div, .lhs = 1, .rhs = 0, .ty = ty_u256 } },
            .{ .binary = .{ .op = .mod, .lhs = 1, .rhs = 0, .ty = ty_u256 } },
            .{ .binary = .{ .op = .eq, .lhs = 2, .rhs = 3 } },
            .{ .binary = .{ .op = .eq, .lhs = 4, .rhs = 5 } },
            .{ .binary = .{ .op = .le, .lhs = 6, .rhs = 1 } },
            .{ .binary = .{ .op = .and_, .lhs = 7, .rhs = 8 } },
            .{ .binary = .{ .op = .and_, .lhs = 10, .rhs = 9 } },
        };
        try writeFixture(out, &first, "u256_arithmetic_formula", &terms, &.{}, .{ .formula = 11 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_u256_arithmetic_formula", &terms, 11);
    }
    {
        const terms = [_]obligation.Term{
            .{ .int_lit = .{ .value = "1", .ty = ty_i256 } },
            .{ .int_lit = .{ .value = "1", .ty = ty_i256 } },
            .{ .binary = .{ .op = .add, .lhs = 0, .rhs = 1, .ty = ty_i256 } },
            .{ .binary = .{ .op = .sub, .lhs = 1, .rhs = 0, .ty = ty_i256 } },
            .{ .binary = .{ .op = .mul, .lhs = 0, .rhs = 1, .ty = ty_i256 } },
            .{ .binary = .{ .op = .div, .lhs = 1, .rhs = 0, .ty = ty_i256 } },
            .{ .binary = .{ .op = .mod, .lhs = 1, .rhs = 0, .ty = ty_i256 } },
            .{ .binary = .{ .op = .sle, .lhs = 2, .rhs = 3, .ty = ty_i256 } },
            .{ .binary = .{ .op = .eq, .lhs = 4, .rhs = 5 } },
            .{ .binary = .{ .op = .sge, .lhs = 6, .rhs = 0, .ty = ty_i256 } },
            .{ .binary = .{ .op = .and_, .lhs = 7, .rhs = 8 } },
            .{ .binary = .{ .op = .and_, .lhs = 10, .rhs = 9 } },
        };
        try writeFixture(out, &first, "i256_arithmetic_formula", &terms, &.{}, .{ .formula = 11 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_i256_arithmetic_formula", &terms, 11);
    }
    {
        const terms = [_]obligation.Term{
            .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = ty_u256 } } },
            .{ .int_lit = .{ .value = "0", .ty = ty_u256 } },
            .{ .binary = .{ .op = .le, .lhs = 1, .rhs = 0 } },
            .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 0 } },
            .{ .quantified = .{ .quantifier = .forall, .binder = .{ .name = "i", .ty = ty_u256 }, .condition = 2, .body = 3 } },
        };
        try writeFixture(out, &first, "forall_with_condition_formula", &terms, &.{}, .{ .formula = 4 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_forall_with_condition_formula", &terms, 4);
    }
    {
        const terms = [_]obligation.Term{
            .{ .variable = .{ .bound = .{ .index = 0, .name = "i", .ty = ty_u256 } } },
            .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 0 } },
            .{ .quantified = .{ .quantifier = .exists, .binder = .{ .name = "i", .ty = ty_u256 }, .body = 1 } },
        };
        try writeFixture(out, &first, "exists_formula", &terms, &.{}, .{ .formula = 2 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_exists_formula", &terms, 2);
    }
    {
        const terms = [_]obligation.Term{
            .{ .place_read = balance_root },
            .{ .old = 0 },
            .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
        };
        try writeFixture(out, &first, "place_read_old_formula", &terms, &.{}, .{ .formula = 2 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_place_read_old_formula", &terms, 2);
    }
    {
        const terms = [_]obligation.Term{
            .result,
            .{ .int_lit = .{ .value = "0", .ty = ty_u256 } },
            .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
        };
        try writeFixture(out, &first, "result_formula_with_canonical_result_env", &terms, &.{}, .{ .formula = 2 });
        try writeAssumptionFormulaFixture(out, &first, "assumption_result_formula_with_canonical_result_env", &terms, 2);
    }
    {
        const declared = [_]obligation.PlaceRef{balance_root};
        const actual = [_]obligation.PlaceRef{balance_root};
        try writeFixture(out, &first, "effect_write_covered", &.{}, &.{}, .{ .effect = .{
            .relation = .write_covered_by_modifies,
            .declared = &declared,
            .actual = &actual,
        } });
    }
    {
        const declared = [_]obligation.PlaceRef{reserve_root};
        const actual = [_]obligation.PlaceRef{allowance_root};
        try writeFixture(out, &first, "effect_read_preserved_static_disjoint", &.{}, &.{}, .{ .effect = .{
            .relation = .read_preserved_by_frame,
            .declared = &declared,
            .actual = &actual,
        } });
    }
    {
        const terms = [_]obligation.Term{
            .{ .variable = .{ .free = .{ .id = user_id, .name = "user", .ty = ty_u256 } } },
            .{ .variable = .{ .free = .{ .id = other_id, .name = "other", .ty = ty_u256 } } },
            .{ .binary = .{ .op = .ne, .lhs = 0, .rhs = 1 } },
        };
        const assumptions = [_]obligation.Assumption{.{
            .id = 10,
            .owner = .{ .function = .{ .name = "obligation_totality_fixture" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .source,
            .kind = .requires,
            .formula = .{ .term = 2 },
        }};
        const declared = [_]obligation.PlaceRef{balance_user};
        const actual = [_]obligation.PlaceRef{balance_other};
        const evidence = [_]obligation.KeyDisjointEvidence{.{
            .kind = .free_var_disequality,
            .assumption_id = 10,
            .lhs = user_id,
            .rhs = other_id,
            .read = balance_other,
            .write = balance_user,
            .key_index = 0,
        }};
        try writeFixture(out, &first, "effect_read_preserved_key_evidence", &terms, &assumptions, .{ .effect = .{
            .relation = .read_preserved_by_key_evidence,
            .declared = &declared,
            .actual = &actual,
            .evidence = &evidence,
        } });
    }
    {
        const terms = [_]obligation.Term{.{ .int_lit = .{ .value = "1", .ty = ty_u256 } }};
        try writeFixture(out, &first, "resource_move_amount_non_negative", &terms, &.{}, .{ .resource = .{
            .op = .move,
            .domain = "TokenUnit",
            .source = balance_user,
            .destination = balance_other,
            .amount = .{ .term = 0 },
            .property = .amount_non_negative,
        } });
        try writeFixture(out, &first, "resource_move_source_sufficient", &terms, &.{}, .{ .resource = .{
            .op = .move,
            .domain = "TokenUnit",
            .source = balance_user,
            .destination = balance_other,
            .amount = .{ .term = 0 },
            .property = .source_sufficient,
        } });
        try writeFixture(out, &first, "resource_move_destination_no_overflow", &terms, &.{}, .{ .resource = .{
            .op = .move,
            .domain = "TokenUnit",
            .source = balance_user,
            .destination = balance_other,
            .amount = .{ .term = 0 },
            .property = .destination_no_overflow,
        } });
        try writeFixture(out, &first, "resource_move_same_place_identity", &terms, &.{}, .{ .resource = .{
            .op = .move,
            .domain = "TokenUnit",
            .source = balance_user,
            .destination = balance_other,
            .amount = .{ .term = 0 },
            .property = .same_place_identity,
        } });
        try writeFixture(out, &first, "resource_move_conservation", &terms, &.{}, .{ .resource = .{
            .op = .move,
            .domain = "TokenUnit",
            .source = balance_user,
            .destination = balance_other,
            .amount = .{ .term = 0 },
            .property = .conservation,
        } });
        try writeFixture(out, &first, "resource_create_destination_no_overflow", &terms, &.{}, .{ .resource = .{
            .op = .create,
            .domain = "TokenUnit",
            .destination = balance_other,
            .amount = .{ .term = 0 },
            .property = .destination_no_overflow,
        } });
        try writeFixture(out, &first, "resource_destroy_source_sufficient", &terms, &.{}, .{ .resource = .{
            .op = .destroy,
            .domain = "TokenUnit",
            .source = balance_user,
            .amount = .{ .term = 0 },
            .property = .source_sufficient,
        } });
    }

    try out.writeAll("]\n\nend Ora.Generated\n");
    try out.flush();
}
