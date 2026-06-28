//! Canonical obligation manifest to Lean emitter.
//!
//! This emitter consumes the same `obligation.ObligationSet` used by the first
//! Z3 adapter and writes a Lean module containing manifest data plus a kernel
//! checked structural well-formedness theorem. It does not claim semantic proof
//! of user contracts yet; that belongs to later slices that assign meaning to
//! these rows.

const std = @import("std");
const obligation = @import("obligation.zig");

pub const Options = struct {
    namespace: []const u8 = "Ora.Generated.ObligationSmoke",
};

pub fn writeModule(writer: anytype, set: obligation.ObligationSet, options: Options) !void {
    try set.validateTermReferences();
    try set.validateIdReferences();

    try writer.writeAll("import Ora.Obligation.Manifest\n\n");
    try writer.writeAll("namespace ");
    try writer.writeAll(options.namespace);
    try writer.writeAll("\n\n");
    try writer.writeAll("open Ora.Obligation\n\n");

    try writeTerms(writer, set.terms);
    try writer.writeByte('\n');
    try writeAssumptions(writer, set.assumptions);
    try writer.writeByte('\n');
    try writeObligations(writer, set.obligations);
    try writer.writeByte('\n');
    try writeProofArtifacts(writer, set.proof_artifacts);
    try writer.writeByte('\n');

    try writer.writeAll("def emittedManifest : Manifest := {\n");
    try writer.writeAll("  terms := emittedTerms,\n");
    try writer.writeAll("  assumptions := emittedAssumptions,\n");
    try writer.writeAll("  obligations := emittedObligations,\n");
    try writer.writeAll("  proofArtifacts := emittedProofArtifacts\n");
    try writer.writeAll("}\n\n");
    try writer.writeAll("theorem emitted_manifest_wf : emittedManifest.wf = true := by decide\n\n");
    try writer.writeAll("end ");
    try writer.writeAll(options.namespace);
    try writer.writeByte('\n');
}

fn writeTerms(writer: anytype, terms: []const obligation.Term) !void {
    try writer.writeAll("def emittedTerms : List Term := ");
    if (terms.len == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    for (terms, 0..) |term, index| {
        try writer.writeAll("  ");
        try writeTerm(writer, term);
        try writeListSeparator(writer, index, terms.len);
    }
    try writer.writeAll("]\n");
}

fn writeAssumptions(writer: anytype, rows: []const obligation.Assumption) !void {
    try writer.writeAll("def emittedAssumptions : List AssumptionRow := ");
    if (rows.len == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    for (rows, 0..) |row, index| {
        try writer.writeAll("  { id := ");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(", owner := ");
        try writeLeanString(writer, ownerName(row.owner));
        try writer.writeAll(", kind := .");
        try writer.writeAll(assumptionKindName(row.kind));
        try writer.writeAll(", formula := ");
        if (row.formula) |formula| {
            try writer.writeAll("some (");
            try writeFormula(writer, formula);
            try writer.writeByte(')');
        } else {
            try writer.writeAll("none");
        }
        try writer.writeAll(" }");
        try writeListSeparator(writer, index, rows.len);
    }
    try writer.writeAll("]\n");
}

fn writeObligations(writer: anytype, rows: []const obligation.Obligation) !void {
    try writer.writeAll("def emittedObligations : List ObligationRow := ");
    if (rows.len == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    for (rows, 0..) |row, index| {
        try writer.writeAll("  { id := ");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(", owner := ");
        try writeLeanString(writer, ownerName(row.owner));
        try writer.writeAll(", kind := ");
        try writeObligationKind(writer, row.kind);
        try writer.writeAll(" }");
        try writeListSeparator(writer, index, rows.len);
    }
    try writer.writeAll("]\n");
}

fn writeProofArtifacts(writer: anytype, rows: []const obligation.ProofArtifact) !void {
    try writer.writeAll("def emittedProofArtifacts : List ProofArtifactRow := ");
    if (rows.len == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    for (rows, 0..) |row, index| {
        try writer.writeAll("  { id := ");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(", owner := ");
        try writeLeanString(writer, ownerName(row.owner));
        try writer.writeAll(", kind := .");
        try writer.writeAll(proofArtifactKindName(row.kind));
        try writer.writeAll(", moduleName := ");
        try writeLeanString(writer, row.module_name);
        try writer.writeAll(", theoremName := ");
        try writeLeanString(writer, row.theorem_name);
        try writer.writeAll(", path := ");
        try writeOptionalLeanString(writer, row.path);
        try writer.writeAll(", contentHash := ");
        if (row.content_hash) |hash| {
            try writer.writeAll("some ");
            try writer.print("{d}", .{hash});
        } else {
            try writer.writeAll("none");
        }
        try writer.writeAll(", obligationIds := ");
        try writeIdList(writer, row.obligation_ids);
        try writer.writeAll(" }");
        try writeListSeparator(writer, index, rows.len);
    }
    try writer.writeAll("]\n");
}

fn writeTerm(writer: anytype, term: obligation.Term) !void {
    switch (term) {
        .bool_lit => |value| {
            try writer.writeAll(".boolLit ");
            try writer.writeAll(if (value) "true" else "false");
        },
        .int_lit => |literal| {
            try writer.writeAll(".intLit { value := ");
            try writeLeanString(writer, literal.value);
            try writer.writeAll(", ty := ");
            try writeOptionalTypeRef(writer, literal.ty);
            try writer.writeAll(" }");
        },
        .variable => |variable| {
            try writer.writeAll(".variable ");
            try writeVarRef(writer, variable);
        },
        .old => |id| {
            try writer.writeAll(".old ");
            try writer.print("{d}", .{id});
        },
        .result => try writer.writeAll(".result"),
        .place_read => return error.UnsupportedPlaceReadTerm,
        .unary => |unary| {
            try writer.writeAll(".unary { op := .");
            try writer.writeAll(unaryOpName(unary.op));
            try writer.writeAll(", operand := ");
            try writer.print("{d}", .{unary.operand});
            try writer.writeAll(" }");
        },
        .binary => |binary| {
            try writer.writeAll(".binary { op := .");
            try writer.writeAll(binaryOpName(binary.op));
            try writer.writeAll(", lhs := ");
            try writer.print("{d}", .{binary.lhs});
            try writer.writeAll(", rhs := ");
            try writer.print("{d}", .{binary.rhs});
            try writer.writeAll(" }");
        },
        .refinement_predicate => |predicate| {
            try writer.writeAll(".refinementPredicate { name := ");
            try writeLeanString(writer, predicate.name);
            try writer.writeAll(", value := ");
            try writer.print("{d}", .{predicate.value});
            try writer.writeAll(", args := ");
            try writeTermIdList(writer, predicate.args);
            try writer.writeAll(" }");
        },
        .quantified => |quantified| {
            try writer.writeAll(".quantified { quantifier := .");
            try writer.writeAll(quantifierName(quantified.quantifier));
            try writer.writeAll(", binder := ");
            try writeVarRef(writer, quantified.binder);
            try writer.writeAll(", condition := ");
            if (quantified.condition) |condition| {
                try writer.writeAll("some ");
                try writer.print("{d}", .{condition});
            } else {
                try writer.writeAll("none");
            }
            try writer.writeAll(", body := ");
            try writer.print("{d}", .{quantified.body});
            try writer.writeAll(" }");
        },
    }
}

fn writeFormula(writer: anytype, formula: obligation.FormulaRef) !void {
    switch (formula) {
        .term => |id| {
            try writer.writeAll(".term ");
            try writer.print("{d}", .{id});
        },
        .origin_value => return error.UnsupportedOriginValue,
    }
}

fn writeObligationKind(writer: anytype, kind: obligation.Kind) !void {
    switch (kind) {
        .logical => |logical| {
            try writer.writeAll(".logical .");
            try writer.writeAll(logicalRoleName(logical.role));
            try writer.writeAll(" (");
            try writeFormula(writer, logical.formula);
            try writer.writeByte(')');
        },
        .runtime_guard => |guard| {
            try writer.writeAll(".runtimeGuard ");
            try writeLeanString(writer, guard.guard_id);
            try writer.writeAll(" (");
            try writeFormula(writer, guard.formula);
            try writer.writeByte(')');
        },
        .resource => |resource| {
            try writer.writeAll(".resource ");
            try writeResourceGoal(writer, resource);
        },
        .quantifier => |quantifier| {
            try writer.writeAll(".quantifier ");
            try writeQuantifierGoal(writer, quantifier);
        },
        .backend_fact => |fact| {
            try writer.writeAll(".backendFact ");
            try writeBackendFactGoal(writer, fact);
        },
        .type_wf,
        .type_relation,
        .region_relation,
        .effect_frame,
        .filtered_input,
        => return error.UnsupportedObligationKind,
    }
}

fn writeBackendFactGoal(writer: anytype, fact: obligation.BackendFactGoal) !void {
    try writer.writeAll("{ component := .");
    try writer.writeAll(backendComponentName(fact.component));
    try writer.writeAll(", property := .");
    try writer.writeAll(backendPropertyName(fact.property));
    try writer.writeAll(" }");
}

fn writeQuantifierGoal(writer: anytype, quantifier: obligation.QuantifierGoal) !void {
    try writer.writeAll("{ quantifier := .");
    try writer.writeAll(quantifierName(quantifier.quantifier));
    try writer.writeAll(", binderName := ");
    try writeLeanString(writer, quantifier.variable);
    try writer.writeAll(", binderType := ");
    try writeTypeRef(writer, quantifier.binder_type);
    try writer.writeAll(", binderSort := .");
    try writer.writeAll(quantifierBinderSortName(quantifier.binder_sort));
    try writer.writeAll(", fragment := .");
    try writer.writeAll(queryFragmentName(quantifier.fragment));
    try writer.writeAll(", patternStatus := .");
    try writer.writeAll(quantifierPatternStatusName(quantifier.pattern_status));
    try writer.writeAll(", degradation := ");
    if (quantifier.degradation) |degradation| {
        try writer.writeAll("some .");
        try writer.writeAll(quantifierDegradationName(degradation));
    } else {
        try writer.writeAll("none");
    }
    try writer.writeAll(" }");
}

fn writeResourceGoal(writer: anytype, resource: obligation.ResourceGoal) !void {
    try writer.writeAll("{ op := .");
    try writer.writeAll(resourceOperationName(resource.op));
    try writer.writeAll(", domain := ");
    try writeLeanString(writer, resource.domain);
    try writer.writeAll(", source := ");
    try writeOptionalPlaceRef(writer, resource.source);
    try writer.writeAll(", destination := ");
    try writeOptionalPlaceRef(writer, resource.destination);
    try writer.writeAll(", amount := ");
    if (resource.amount) |amount| {
        try writer.writeAll("some (");
        try writeFormula(writer, amount);
        try writer.writeByte(')');
    } else {
        try writer.writeAll("none");
    }
    try writer.writeAll(", property := .");
    try writer.writeAll(resourcePropertyName(resource.property));
    try writer.writeAll(" }");
}

fn writeOptionalPlaceRef(writer: anytype, place: ?obligation.PlaceRef) !void {
    if (place) |value| {
        try writer.writeAll("some ");
        try writePlaceRef(writer, value);
    } else {
        try writer.writeAll("none");
    }
}

fn writePlaceRef(writer: anytype, place: obligation.PlaceRef) !void {
    try writer.writeAll("{ root := ");
    try writeLeanString(writer, place.root);
    try writer.writeAll(", region := .");
    try writer.writeAll(regionName(place.region));
    try writer.writeAll(", fields := ");
    try writeStringList(writer, place.fields);
    try writer.writeAll(", keys := ");
    try writePlaceKeyList(writer, place.keys);
    try writer.writeAll(" }");
}

fn writeStringList(writer: anytype, values: []const []const u8) !void {
    if (values.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeLeanString(writer, value);
    }
    try writer.writeByte(']');
}

fn writePlaceKeyList(writer: anytype, keys: []const obligation.PlaceKey) !void {
    if (keys.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (keys, 0..) |key, index| {
        if (index != 0) try writer.writeAll(", ");
        try writePlaceKey(writer, key);
    }
    try writer.writeByte(']');
}

fn writePlaceKey(writer: anytype, key: obligation.PlaceKey) !void {
    switch (key) {
        .parameter => |index| try writer.print(".parameter {d}", .{index}),
        .comptime_parameter => |index| try writer.print(".comptimeParameter {d}", .{index}),
        .comptime_range_parameter => |index| try writer.print(".comptimeRangeParameter {d}", .{index}),
        .constant => |value| {
            try writer.writeAll(".constant ");
            try writeLeanString(writer, value);
        },
        .msg_sender => try writer.writeAll(".msgSender"),
        .tx_origin => try writer.writeAll(".txOrigin"),
        .unknown => try writer.writeAll(".unknown"),
    }
}

fn writeVarRef(writer: anytype, variable: obligation.VarRef) !void {
    try writer.writeAll("{ name := ");
    try writeLeanString(writer, variable.name);
    try writer.writeAll(", ty := ");
    try writeOptionalTypeRef(writer, variable.ty);
    try writer.writeAll(", region := ");
    if (variable.region) |region| {
        try writer.writeAll("some .");
        try writer.writeAll(regionName(region));
    } else {
        try writer.writeAll("none");
    }
    try writer.writeAll(" }");
}

fn writeOptionalTypeRef(writer: anytype, ty: ?obligation.TypeRef) !void {
    if (ty) |value| {
        try writer.writeAll("some (");
        try writeTypeRef(writer, value);
        try writer.writeByte(')');
    } else {
        try writer.writeAll("none");
    }
}

fn writeTypeRef(writer: anytype, ty: obligation.TypeRef) !void {
    switch (ty) {
        .spelling => |text| {
            try writer.writeAll(".spelling ");
            try writeLeanString(writer, text);
        },
        .compiler_type_id => |id| {
            try writer.writeAll(".compilerTypeId ");
            try writer.print("{d}", .{id});
        },
    }
}

fn writeTermIdList(writer: anytype, ids: []const obligation.TermId) !void {
    try writeIntegerList(writer, ids);
}

fn writeIdList(writer: anytype, ids: []const obligation.Id) !void {
    try writeIntegerList(writer, ids);
}

fn writeIntegerList(writer: anytype, ids: anytype) !void {
    if (ids.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{id});
    }
    try writer.writeByte(']');
}

fn writeOptionalLeanString(writer: anytype, value: ?[]const u8) !void {
    if (value) |text| {
        try writer.writeAll("some ");
        try writeLeanString(writer, text);
    } else {
        try writer.writeAll("none");
    }
}

fn writeListSeparator(writer: anytype, index: usize, len: usize) !void {
    if (index + 1 != len) try writer.writeByte(',');
    try writer.writeByte('\n');
}

fn writeLeanString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    var start: usize = 0;
    for (value, 0..) |byte, index| {
        switch (byte) {
            '"', '\\', '\n', '\r', '\t', 0x00...0x08, 0x0b...0x0c, 0x0e...0x1f => {
                if (start < index) try writer.writeAll(value[start..index]);
                switch (byte) {
                    '"' => try writer.writeAll("\\\""),
                    '\\' => try writer.writeAll("\\\\"),
                    '\n' => try writer.writeAll("\\n"),
                    '\r' => try writer.writeAll("\\r"),
                    '\t' => try writer.writeAll("\\t"),
                    else => try writer.print("\\u{{{x}}}", .{byte}),
                }
                start = index + 1;
            },
            else => {},
        }
    }
    if (start < value.len) try writer.writeAll(value[start..]);
    try writer.writeByte('"');
}

fn ownerName(owner: obligation.Owner) []const u8 {
    return switch (owner) {
        .module => |name| name,
        .contract => |name| name,
        .function => |function| function.name,
        .trait_method => |method| method.method_name,
        .statement => |statement| statement.function_name,
        .backend => |backend| backend.name,
    };
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

fn unaryOpName(op: obligation.UnaryOp) []const u8 {
    return switch (op) {
        .not => "not_",
        .neg => "neg",
    };
}

fn binaryOpName(op: obligation.BinaryOp) []const u8 {
    return switch (op) {
        .eq => "eq",
        .ne => "ne",
        .lt => "lt",
        .le => "le",
        .gt => "gt",
        .ge => "ge",
        .add => "add",
        .sub => "sub",
        .mul => "mul",
        .div => "div",
        .mod => "mod_",
        .and_ => "and_",
        .or_ => "or_",
        .implies => "implies",
    };
}

fn quantifierName(quantifier: obligation.Quantifier) []const u8 {
    return switch (quantifier) {
        .forall => "forall_",
        .exists => "exists_",
    };
}

fn queryFragmentName(fragment: obligation.VerificationQueryFragment) []const u8 {
    return switch (fragment) {
        .unknown => "unknown",
        .qf_bv => "qfBv",
        .qf_bv_array => "qfBvArray",
        .aufbv => "aufbv",
        .aufbv_quantifiers => "aufbvQuantifiers",
        .other => "other",
    };
}

fn quantifierBinderSortName(sort: obligation.QuantifierBinderSort) []const u8 {
    return switch (sort) {
        .bool => "bool",
        .bit_vector => "bitVector",
        .byte_sequence => "byteSequence",
        .opaque_unknown => "opaqueUnknown",
    };
}

fn quantifierPatternStatusName(status: obligation.QuantifierPatternStatus) []const u8 {
    return switch (status) {
        .explicit => "explicit",
        .synthesized => "synthesized",
        .absent => "absent",
    };
}

fn quantifierDegradationName(degradation: obligation.QuantifierDegradation) []const u8 {
    return switch (degradation) {
        .unsupported_binder_type => "unsupportedBinderType",
        .malformed_binder_width => "malformedBinderWidth",
    };
}

fn proofArtifactKindName(kind: obligation.ProofArtifactKind) []const u8 {
    return switch (kind) {
        .userland_lean => "userlandLean",
    };
}

fn backendComponentName(component: obligation.BackendComponent) []const u8 {
    return switch (component) {
        .dispatcher => "dispatcher",
        .oratosir => "oratosir",
        .sinora => "sinora",
        .artifact_policy => "artifactPolicy",
    };
}

fn backendPropertyName(property: obligation.BackendProperty) []const u8 {
    return switch (property) {
        .complete => "complete",
        .disjoint => "disjoint",
        .ordered => "ordered",
        .preserves_selector_behavior => "preservesSelectorBehavior",
        .no_unknown_strategy => "noUnknownStrategy",
        .dependency_valid => "dependencyValid",
    };
}

fn logicalRoleName(role: obligation.LogicalRole) []const u8 {
    return switch (role) {
        .invariant => "invariant",
        .requires => "requires",
        .callee_precondition => "calleePrecondition",
        .ensures => "ensures",
        .ensures_ok => "ensuresOk",
        .ensures_err => "ensuresErr",
        .assert => "assert_",
        .guard => "guard",
        .loop_invariant => "loopInvariant",
        .contract_invariant => "contractInvariant",
        .arithmetic_safety => "arithmeticSafety",
        .refinement => "refinement",
        .imported_callee_obligation => "importedCalleeObligation",
        .imported_callee_ensures => "importedCalleeEnsures",
    };
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
        .amount_non_negative => "amountNonNegative",
        .source_sufficient => "sourceSufficient",
        .destination_no_overflow => "destinationNoOverflow",
        .same_place_net_zero => "samePlaceNetZero",
        .conservation => "conservation",
        .modifies_covered => "modifiesCovered",
    };
}

fn assumptionKindName(kind: obligation.AssumptionKind) []const u8 {
    return switch (kind) {
        .requires => "requires",
        .assume => "assume",
        .path_assume => "pathAssume",
        .env_assume => "envAssume",
        .binding => "binding",
        .two_state_linkage => "twoStateLinkage",
        .frame => "frame",
        .loop_invariant => "loopInvariant",
        .callee_obligation => "calleeObligation",
        .callee_ensures => "calleeEnsures",
        .ghost_axiom => "ghostAxiom",
        .goal => "goal",
    };
}
