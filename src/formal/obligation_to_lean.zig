//! Canonical obligation manifest to Lean emitter.
//!
//! This emitter consumes the same `obligation.ObligationSet` used by the first
//! Z3 adapter and writes a Lean module containing manifest data, semantic
//! proposition names, and a kernel-checked structural well-formedness theorem.

const std = @import("std");
const obligation = @import("obligation.zig");
const type_builtin = @import("ora_types").builtin;

pub const Options = struct {
    namespace: []const u8 = "Ora.Generated.ObligationSmoke",
    proof_surface: bool = false,
};

pub const SemanticSupport = union(enum) {
    supported,
    unsupported: SemanticUnsupportedReason,
};

pub const SemanticUnsupportedReason = union(enum) {
    empty_query,
    invalid_dependency,
    unsupported_obligation_kind,
    unsupported_effect_frame_relation: obligation.EffectFrameRelation,
    unsupported_origin_value,
    unsupported_term_kind,
    missing_type,
    unsupported_type: obligation.TypeRef,
    unsupported_comparison_width,
    unknown_signedness,
    mixed_signedness,
    unsupported_arithmetic_width,
    unknown_arithmetic_signedness,
    mixed_arithmetic_signedness,
    missing_key_disjoint_evidence,
    unsupported_key_disjoint_evidence_formula,
    unsupported_key_disjoint_evidence_kind,
    key_disjoint_evidence_type_unsupported,
    key_disjoint_evidence_owner_mismatch,
    key_disjoint_evidence_path_mismatch,
    loop_summary_missing,
    loop_summary_query_mismatch,
    unsupported_loop_summary: obligation.LoopUnsupportedReason,
};

pub fn writeDataModule(writer: anytype, set: obligation.ObligationSet, options: Options) !void {
    try set.validateTermReferences();
    try set.validateIdReferences();

    try writeModulePreamble(writer, "Ora.Obligation.Manifest", options.namespace, false);
    try writeManifestRows(writer, set, false);
    try writeManifestDefinition(writer);
    try writeRowDefinitions(writer, set, false);
    try writeNamespaceEnd(writer, options.namespace);
}

pub fn writeModule(writer: anytype, set: obligation.ObligationSet, options: Options) !void {
    try set.validateTermReferences();
    try set.validateIdReferences();

    try writeModulePreamble(writer, "Ora.Obligation.Theorems", options.namespace, true);
    try writeManifestRows(writer, set, options.proof_surface);
    try writeManifestDefinition(writer);
    try writeTermLookupTheorems(writer, set.terms);
    try writeManifestWfTheorem(writer, set.terms);
    try writeTermsTopologicalTheorem(writer, set.terms);
    try writeLoopSummaryDefinitions(writer, set);
    try writeSemanticDefinitions(writer, set, options.proof_surface);
    try writeNamespaceEnd(writer, options.namespace);
}

fn writeTermLookupTheorems(writer: anytype, terms: []const obligation.Term) !void {
    if (terms.len > lean_term_chunk_size) {
        const chunk_count = leanTermChunkCount(terms.len);
        for (0..chunk_count) |chunk_index| {
            const start = chunk_index * lean_term_chunk_size;
            const end = @min(start + lean_term_chunk_size, terms.len);
            try writer.print(
                "theorem emitted_terms_chunk_{d}_length : emittedTermsChunk_{d}.length = {d} := by rfl\n",
                .{ chunk_index, chunk_index, end - start },
            );
        }
        try writer.print("\ntheorem emitted_terms_raw_length : emittedTerms.length = {d} := by\n", .{terms.len});
        try writer.writeAll("  simp only [emittedTerms, List.length_append");
        for (0..chunk_count) |chunk_index| {
            try writer.print(", emitted_terms_chunk_{d}_length", .{chunk_index});
        }
        try writer.writeAll("]\n\n");
    }
    try writer.print("theorem emitted_terms_length : emittedManifest.terms.length = {d} := by ", .{terms.len});
    if (terms.len <= lean_term_chunk_size) {
        try writer.writeAll("rfl\n\n");
    } else {
        try writer.writeAll("simpa [emittedManifest] using emitted_terms_raw_length\n\n");
    }
    for (terms, 0..) |term, index| {
        try writer.print(
            "theorem emitted_term_{d}_lookup : emittedManifest.terms[{d}]? = some (",
            .{ index, index },
        );
        try writeTerm(writer, term);
        try writer.writeAll(") := by ");
        if (terms.len <= lean_term_chunk_size) {
            try writer.writeAll("rfl\n");
        } else {
            const chunk_count = leanTermChunkCount(terms.len);
            const chunk_index = index / lean_term_chunk_size;
            try writer.writeAll("\n  simp only [emittedManifest, emittedTerms]\n");
            for (0..chunk_index) |_| {
                try writer.writeAll("  rw [List.getElem?_append_right (by decide)]\n");
            }
            if (chunk_index + 1 != chunk_count) {
                try writer.writeAll("  rw [List.getElem?_append_left (by decide)]\n");
            }
            try writer.writeAll("  simp [");
            for (0..chunk_index + 1) |prior_chunk| {
                if (prior_chunk != 0) try writer.writeAll(", ");
                try writer.print("emittedTermsChunk_{d}", .{prior_chunk});
            }
            try writer.writeAll("]\n");
        }
    }
    if (terms.len != 0) try writer.writeByte('\n');
}

fn writeManifestWfTheorem(writer: anytype, terms: []const obligation.Term) !void {
    if (terms.len <= lean_term_chunk_size) {
        return writer.writeAll("theorem emitted_manifest_wf : emittedManifest.wf = true := by decide\n\n");
    }
    const chunk_count = leanTermChunkCount(terms.len);
    for (0..chunk_count) |chunk_index| {
        try writer.print(
            "theorem emitted_terms_chunk_{d}_wf :\n" ++
                "    emittedTermsChunk_{d}.all (Term.wf emittedTerms) = true := by\n" ++
                "  simp [emittedTermsChunk_{d}, Term.wf, termRefInBounds, optionalTermRefInBounds,\n" ++
                "    emitted_terms_raw_length]\n\n",
            .{ chunk_index, chunk_index, chunk_index },
        );
    }
    try writer.writeAll(
        "theorem emitted_terms_wf : emittedTerms.all (Term.wf emittedTerms) = true := by\n" ++
            "  change (",
    );
    try writeLeanTermAppendExpression(writer, chunk_count);
    try writer.writeAll(").all (Term.wf emittedTerms) = true\n  simp only [List.all_append");
    for (0..chunk_count) |chunk_index| {
        try writer.print(", emitted_terms_chunk_{d}_wf", .{chunk_index});
    }
    try writer.writeAll(", Bool.true_and]\n\n");
    try writer.writeAll(
        "theorem emitted_manifest_wf : emittedManifest.wf = true := by\n" ++
            "  simp [Manifest.wf, emittedManifest, emitted_terms_wf",
    );
    try writer.writeAll(
        ", emittedAssumptions, emittedObligations, emittedProofArtifacts,\n" ++
            "    Term.wf, AssumptionRow.wf, ObligationRow.wf, ObligationKind.wf,\n" ++
            "    FormulaRef.wf, termRefInBounds, emitted_terms_raw_length,\n" ++
            "    ProofArtifactRow.wf, obligationIdExists]\n\n",
    );
}

fn writeTermsTopologicalTheorem(writer: anytype, terms: []const obligation.Term) !void {
    if (terms.len <= lean_term_chunk_size) {
        return writer.writeAll(
            "theorem emitted_terms_topological : emittedManifest.termsTopological = true := by decide\n\n",
        );
    }

    const chunk_count = leanTermChunkCount(terms.len);
    for (0..chunk_count) |chunk_index| {
        try writer.print(
            "theorem emitted_terms_chunk_{d}_topological :\n" ++
                "    termsTopologicalFrom {d} emittedTermsChunk_{d} = true := by decide\n\n",
            .{ chunk_index, chunk_index * lean_term_chunk_size, chunk_index },
        );
    }
    try writer.writeAll(
        "theorem emitted_terms_topological : emittedManifest.termsTopological = true := by\n" ++
            "  simp only [Manifest.termsTopological, emittedManifest, emittedTerms,\n" ++
            "    termsTopologicalFrom_append, Nat.reduceAdd",
    );
    for (0..chunk_count) |chunk_index| {
        try writer.print(
            ", emitted_terms_chunk_{d}_length, emitted_terms_chunk_{d}_topological",
            .{ chunk_index, chunk_index },
        );
    }
    try writer.writeAll(", Nat.zero_add, Bool.true_and]\n\n");
}

fn writeModulePreamble(
    writer: anytype,
    import_name: []const u8,
    namespace: []const u8,
    include_loops: bool,
) !void {
    try writer.writeAll("import ");
    try writer.writeAll(import_name);
    try writer.writeByte('\n');
    if (include_loops) try writer.writeAll("import Ora.Loop.StorageCertificate\n");
    try writer.writeAll("\nnamespace ");
    try writer.writeAll(namespace);
    try writer.writeAll("\n\nopen Ora.Obligation\n\n");
}

fn writeLoopSummaryDefinitions(writer: anytype, set: obligation.ObligationSet) !void {
    var emitted_count: usize = 0;
    for (set.loop_summaries) |row| {
        if (!row.projectionSupported()) continue;
        emitted_count += 1;
        try writer.print("def emittedLoopSummary_{d} : Ora.Loop.SummaryRow :=\n  ", .{row.id});
        try writeLoopSummaryRow(writer, row);
        const digest = try loopVcSha256(set, row.id);
        const digest_hex = std.fmt.bytesToHex(digest, .lower);
        try writer.print("\n\ndef emittedLoopVcSha256_{d} : String := \"{s}\"\n\n", .{ row.id, &digest_hex });
    }
    if (emitted_count == 0) return;
    try writer.writeAll("def emittedLoopSummaries : List Ora.Loop.SummaryRow := [");
    var emitted_index: usize = 0;
    for (set.loop_summaries) |row| {
        if (!row.projectionSupported()) continue;
        if (emitted_index != 0) try writer.writeAll(", ");
        try writer.print("emittedLoopSummary_{d}", .{row.id});
        emitted_index += 1;
    }
    try writer.writeAll("]\n\n");
}

/// Deterministic identity for the exact LoopVC proof input. The complete term
/// table is included deliberately: it may over-invalidate a certificate when
/// an unrelated term changes, but it cannot retain a certificate after a
/// referenced term changes.
pub fn loopVcSha256(set: obligation.ObligationSet, summary_id: obligation.Id) ![32]u8 {
    return loopVcSha256InDomain(set, summary_id, "ora-loop-vc-v3\n");
}

fn loopVcSha256InDomain(
    set: obligation.ObligationSet,
    summary_id: obligation.Id,
    domain: []const u8,
) ![32]u8 {
    const summary = obligation.findById(set.loop_summaries, summary_id) orelse
        return error.InvalidDependency;
    var induction_query: ?obligation.VerificationQuery = null;
    for (set.queries) |query| {
        if (query.kind != .loop_induction or query.loop_summary_id != summary_id) continue;
        if (induction_query != null) return error.InvalidDependency;
        induction_query = query;
    }
    const query = induction_query orelse return error.InvalidDependency;

    var canonical = std.Io.Writer.Allocating.init(std.heap.page_allocator);
    defer canonical.deinit();
    try canonical.writer.writeAll(domain);
    try writeCanonicalTerms(&canonical.writer, set.terms);
    try writeAssumptionRowsByIds(&canonical.writer, set, query.assumption_ids);
    try canonical.writer.writeByte('\n');
    try writeLoopSummaryRow(&canonical.writer, summary);
    try canonical.writer.writeByte('\n');

    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(canonical.written(), &digest, .{});
    return digest;
}

fn writeLoopSummaryRow(writer: anytype, row: obligation.LoopSummaryRow) !void {
    try writer.writeAll("{ id := ");
    try writer.print("{d}, owner := ", .{row.id});
    try writeLeanString(writer, ownerName(row.owner));
    try writer.writeAll(", kind := .");
    try writer.writeAll(switch (row.loop_kind) {
        .scf_while => "scfWhile",
        .scf_for => "scfFor",
        .other => "other",
    });
    try writer.writeAll(", contextVariables := ");
    try writeLoopVariables(writer, row.context_variables);
    try writer.writeAll(", variables := ");
    try writeLoopVariables(writer, row.variables);
    try writer.writeAll(", init := ");
    try writeFormulaList(writer, row.init_formulas);
    try writer.writeAll(", invariants := ");
    try writeFormulaList(writer, row.invariant_formulas);
    try writer.writeAll(", headerFacts := ");
    try writeLoopHeaderFacts(writer, row.header_facts);
    try writer.writeAll(", preheaderTimeline := ");
    try writeLoopTimeline(writer, row.preheader_timeline);
    try writer.writeAll(", preheaderProjectionComplete := ");
    try writer.writeAll(if (row.preheader_projection_complete) "true" else "false");
    try writer.writeAll(", edges := ");
    try writeLoopEdges(writer, row.edges);
    try writer.writeAll(", projectionComplete := ");
    try writer.writeAll(if (row.projection_completeness == .complete) "true" else "false");
    try writer.writeAll(", post := ");
    try writeFormulaList(writer, row.post_formulas);
    try writer.writeAll(", unsupportedReasons := []");
    try writer.writeAll(" }");
}

fn writeLoopHeaderFacts(writer: anytype, rows: []const obligation.LoopHeaderFactRow) !void {
    try writer.writeByte('[');
    for (rows, 0..) |row, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{{ id := {d}, structuralKey := ", .{row.id});
        try writeLeanString(writer, row.structural_key);
        try writer.writeAll(", formula := ");
        try writeFormula(writer, row.formula);
        try writer.writeAll(", sourceKind := .");
        try writer.writeAll(switch (row.source_kind) {
            .path_assume => "pathAssume",
            .function_requires => "functionRequires",
            .unresolved => "unresolved",
        });
        try writer.writeAll(", sourceStructuralKey := ");
        try writeLeanString(writer, row.source.structural_key);
        try writer.writeAll(", sourceBlockKey := ");
        try writeLeanString(writer, row.source.block_key);
        try writer.print(", sourcePosition := {d}, targetStructuralKey := ", .{row.source.position});
        try writeLeanString(writer, row.target.structural_key);
        try writer.writeAll(", targetBlockKey := ");
        try writeLeanString(writer, row.target.block_key);
        try writer.print(", targetPosition := {d}, stableValues := [", .{row.target.position});
        for (row.stable_values, 0..) |value, value_index| {
            if (value_index != 0) try writer.writeAll(", ");
            try writer.writeAll("{ id := ");
            try writeFreeVarId(writer, value.id);
            try writer.writeAll(", ty := ");
            try writeTypeRef(writer, value.ty);
            try writer.writeAll(" }");
        }
        try writer.writeAll("], assumptionId := ");
        if (row.assumption_id) |id| try writer.print("some {d}", .{id}) else try writer.writeAll("none");
        try writer.writeAll(" }");
    }
    try writer.writeByte(']');
}

fn writeLoopVariables(writer: anytype, variables: []const obligation.LoopVariable) !void {
    try writer.writeByte('[');
    for (variables, 0..) |variable, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.writeAll("{ index := ");
        try writer.print("{d}, id := ", .{variable.index});
        try writeFreeVarId(writer, variable.id orelse return error.UnsupportedLoopIdentity);
        try writer.writeAll(", name := ");
        try writeLeanString(writer, variable.name orelse "");
        try writer.writeAll(", ty := ");
        try writeTypeRef(writer, variable.ty);
        try writer.writeAll(" }");
    }
    try writer.writeByte(']');
}

fn writeFormulaList(writer: anytype, formulas: []const obligation.FormulaRef) !void {
    try writer.writeByte('[');
    for (formulas, 0..) |formula, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeFormula(writer, formula);
    }
    try writer.writeByte(']');
}

fn writeLoopStepAssignments(
    writer: anytype,
    assignments: []const obligation.LoopStepAssignment,
) !void {
    try writer.writeByte('[');
    for (assignments, 0..) |assignment, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.writeAll("{ variableIndex := ");
        try writer.print("{d}, target := ", .{assignment.variable_index});
        try writeFreeVarId(writer, assignment.target orelse return error.UnsupportedLoopIdentity);
        try writer.writeAll(", sourceKey := ");
        try writeLeanString(writer, assignment.source_key);
        try writer.writeAll(", value := ");
        try writeFormula(writer, assignment.value);
        try writer.writeAll(", readResults := ");
        try writeFreeVarIdList(writer, assignment.read_results);
        try writer.writeAll(" }");
    }
    try writer.writeByte(']');
}

fn writeLoopEdges(writer: anytype, edges: []const obligation.LoopEdgeRow) !void {
    try writer.writeByte('[');
    for (edges, 0..) |edge, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.writeAll("{ id := ");
        try writer.print("{d}, structuralKey := ", .{edge.id});
        try writeLeanString(writer, edge.structural_key);
        try writer.writeAll(", timeline := ");
        try writeLoopTimeline(writer, edge.timeline);
        try writer.writeAll(", nextState := ");
        try writeLoopStepAssignments(writer, edge.next_state);
        try writer.writeAll(", target := .");
        try writer.writeAll(if (edge.target) |target| switch (target) {
            .backedge => "backedge",
            .successor => "successor",
            .function_return => "functionReturn",
            .error_return => "errorReturn",
            .explicit_revert => "explicitRevert",
        } else "unresolved");
        try writer.writeAll(", targetFormulas := ");
        try writeFormulaList(writer, edge.target_formulas);
        try writer.writeAll(" }");
    }
    try writer.writeByte(']');
}

fn writeFreeVarIdList(writer: anytype, ids: []const obligation.FreeVarId) !void {
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeFreeVarId(writer, id);
    }
    try writer.writeByte(']');
}

fn writeLoopFormulaExprRow(writer: anytype, expression: obligation.LoopFormulaExpr) !void {
    try writer.writeAll("{ formula := ");
    try writeFormula(writer, expression.formula);
    try writer.writeAll(", readResults := ");
    try writeFreeVarIdList(writer, expression.read_results);
    try writer.writeAll(" }");
}

fn writeLoopFormulaExprRowList(
    writer: anytype,
    expressions: []const obligation.LoopFormulaExpr,
) !void {
    try writer.writeByte('[');
    for (expressions, 0..) |expression, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeLoopFormulaExprRow(writer, expression);
    }
    try writer.writeByte(']');
}

fn writeLoopTypedFormulaRow(writer: anytype, expression: obligation.LoopTypedFormula) !void {
    try writer.writeAll("{ value := ");
    try writeLoopFormulaExprRow(writer, expression.value);
    try writer.writeAll(", ty := ");
    try writeTypeRef(writer, expression.ty);
    try writer.writeAll(" }");
}

fn writeLoopStorageLocationRow(
    writer: anytype,
    location: obligation.LoopStorageLocationExpr,
) !void {
    try writer.writeAll("{ root := ");
    try writeLeanString(writer, location.root);
    try writer.writeAll(", projections := [");
    for (location.projections, 0..) |projection, index| {
        if (index != 0) try writer.writeAll(", ");
        switch (projection) {
            .field => |field| {
                try writer.writeAll(".field ");
                try writeLeanString(writer, field);
            },
            .fixed_index => |fixed| {
                try writer.writeAll(".fixedIndex { index := ");
                try writeLoopTypedFormulaRow(writer, fixed.index);
                try writer.print(", extent := {d} }}", .{fixed.extent});
            },
            .mapping_key => |value| {
                try writer.writeAll(".mappingKey ");
                try writeLoopTypedFormulaRow(writer, value);
            },
        }
    }
    try writer.writeAll("], valueTy := ");
    try writeTypeRef(writer, location.value_ty);
    try writer.writeAll(" }");
}

fn writeLoopSummaryDependency(
    writer: anytype,
    dependency: obligation.LoopSummaryDependency,
) !void {
    try writer.writeAll("{ id := ");
    try writer.print("{d}, structuralKey := ", .{dependency.id});
    try writeLeanString(writer, dependency.structural_key);
    try writer.print(", summaryId := {d}, loopVcSha256 := \"", .{dependency.summary_id});
    const digest_hex = std.fmt.bytesToHex(dependency.loop_vc_sha256, .lower);
    try writer.writeAll(&digest_hex);
    try writer.writeAll("\", target := .");
    try writer.writeAll(switch (dependency.target) {
        .backedge => "backedge",
        .successor => "successor",
        .function_return => "functionReturn",
        .error_return => "errorReturn",
        .explicit_revert => "explicitRevert",
    });
    try writer.writeAll(", writeFootprint := ");
    try writePlaceRefList(writer, dependency.write_footprint);
    try writer.writeAll(", writeFootprintComplete := ");
    try writer.writeAll(if (dependency.write_footprint_complete) "true" else "false");
    try writer.writeAll(" }");
}

fn writeLoopTimeline(writer: anytype, events: []const obligation.LoopTimelineEvent) !void {
    try writer.writeByte('[');
    for (events, 0..) |event, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.writeAll("{ id := ");
        try writer.print("{d}, structuralKey := ", .{event.id});
        try writeLeanString(writer, event.structural_key);
        try writer.writeAll(", payload := .");
        switch (event.payload) {
            .path_assumption => |path| {
                try writer.writeAll("pathAssumption { formulas := ");
                try writeLoopFormulaExprRowList(writer, path.formulas);
                try writer.writeAll(" }");
            },
            .safety_checkpoint => |checkpoint| {
                try writer.writeAll("safetyCheckpoint { formulas := ");
                try writeLoopFormulaExprRowList(writer, checkpoint.formulas);
                if (checkpoint.dynamic_index_bounds) try writer.writeAll(", dynamicIndexBounds := true");
                try writer.writeAll(" }");
            },
            .storage_read => |read| {
                try writer.writeAll("storageRead { result := ");
                try writeFreeVarId(writer, read.result);
                try writer.writeAll(", location := ");
                try writeLoopStorageLocationRow(writer, read.location);
                try writer.writeAll(" }");
            },
            .indexed_region_read => |read| {
                try writer.writeAll("indexedRegionRead { result := ");
                try writeFreeVarId(writer, read.result);
                try writer.writeAll(", region := .");
                try writer.writeAll(@tagName(read.region));
                try writer.writeAll(", owner := ");
                try writeFreeVarId(writer, read.owner);
                try writer.writeAll(", index := ");
                try writeLoopFormulaExprRow(writer, read.index);
                try writer.writeAll(", elementTy := ");
                try writeTypeRef(writer, read.element_ty);
                try writer.writeAll(", checkpointStructuralKey := ");
                try writeLeanString(writer, read.checkpoint_structural_key);
                try writer.writeAll(" }");
            },
            .storage_write => |write| {
                try writer.writeAll("storageWrite { location := ");
                try writeLoopStorageLocationRow(writer, write.location);
                try writer.writeAll(", value := ");
                try writeLoopFormulaExprRow(writer, write.value);
                try writer.writeAll(" }");
            },
            .summary_dependency => |dependency| {
                try writer.writeAll("summaryDependency ");
                try writeLoopSummaryDependency(writer, dependency);
            },
        }
        try writer.writeAll(" }");
    }
    try writer.writeByte(']');
}

fn writeManifestRows(writer: anytype, set: obligation.ObligationSet, proof_surface: bool) !void {
    try writeLeanTerms(writer, set.terms);
    try writer.writeByte('\n');
    try writeAssumptions(writer, set, proof_surface);
    try writer.writeByte('\n');
    try writeObligations(writer, set, proof_surface);
    try writer.writeByte('\n');
    try writeProofArtifacts(writer, set.proof_artifacts);
    try writer.writeByte('\n');
}

fn writeManifestDefinition(writer: anytype) !void {
    try writer.writeAll("def emittedManifest : Manifest := {\n");
    try writer.writeAll("  terms := emittedTerms,\n");
    try writer.writeAll("  assumptions := emittedAssumptions,\n");
    try writer.writeAll("  obligations := emittedObligations,\n");
    try writer.writeAll("  proofArtifacts := emittedProofArtifacts\n");
    try writer.writeAll("}\n\n");
}

fn writeNamespaceEnd(writer: anytype, namespace: []const u8) !void {
    try writer.writeAll("end ");
    try writer.writeAll(namespace);
    try writer.writeByte('\n');
}

pub const lean_term_chunk_size: usize = 32;

fn leanTermChunkCount(term_count: usize) usize {
    return std.math.divCeil(usize, term_count, lean_term_chunk_size) catch unreachable;
}

fn writeLeanTermChunkNames(writer: anytype, chunk_count: usize) !void {
    for (0..chunk_count) |chunk_index| {
        try writer.print(", emittedTermsChunk_{d}", .{chunk_index});
    }
}

fn writeLeanTermAppendExpression(writer: anytype, chunk_count: usize) !void {
    for (0..chunk_count) |chunk_index| {
        try writer.print("emittedTermsChunk_{d}", .{chunk_index});
        if (chunk_index + 1 != chunk_count) try writer.writeAll(" ++ (");
    }
    if (chunk_count > 1) try writer.splatByteAll(')', chunk_count - 1);
}

fn writeLeanTerms(writer: anytype, terms: []const obligation.Term) !void {
    if (terms.len <= lean_term_chunk_size) {
        return writeNamedTermList(writer, "emittedTerms", terms);
    }

    const chunk_count = leanTermChunkCount(terms.len);
    for (0..chunk_count) |chunk_index| {
        const start = chunk_index * lean_term_chunk_size;
        const end = @min(start + lean_term_chunk_size, terms.len);
        var name_buffer: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buffer, "emittedTermsChunk_{d}", .{chunk_index});
        try writeNamedTermList(writer, name, terms[start..end]);
        try writer.writeByte('\n');
    }

    try writer.writeAll("def emittedTerms : List Term :=\n");
    for (0..chunk_count) |chunk_index| {
        try writer.print("  emittedTermsChunk_{d}", .{chunk_index});
        if (chunk_index + 1 != chunk_count) {
            try writer.writeAll(" ++ (\n");
        } else {
            try writer.splatByteAll(')', chunk_count - 1);
            try writer.writeByte('\n');
        }
    }
}

fn writeNamedTermList(
    writer: anytype,
    name: []const u8,
    terms: []const obligation.Term,
) !void {
    try writer.print("def {s} : List Term := ", .{name});
    if (terms.len == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    for (terms, 0..) |term, index| {
        try writer.writeAll("  ");
        try writeTerm(writer, term);
        try writeListSeparator(writer, index, terms.len);
    }
    try writer.writeAll("]\n");
}

/// Certificate identity uses one stable serialization independent of how the
/// generated Lean module groups declarations for elaboration.
fn writeCanonicalTerms(writer: anytype, terms: []const obligation.Term) !void {
    return writeNamedTermList(writer, "emittedTerms", terms);
}

fn writeAssumptions(writer: anytype, set: obligation.ObligationSet, proof_surface: bool) !void {
    try writer.writeAll("def emittedAssumptions : List AssumptionRow := ");
    const count = countWritableAssumptions(set, proof_surface);
    if (count == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    var emitted_index: usize = 0;
    for (set.assumptions) |row| {
        if (proof_surface and !assumptionWritableInProofSurface(row)) continue;
        try writer.writeAll("  ");
        try writeAssumptionRow(writer, row);
        try writeListSeparator(writer, emitted_index, count);
        emitted_index += 1;
    }
    try writer.writeAll("]\n");
}

fn writeObligations(writer: anytype, set: obligation.ObligationSet, proof_surface: bool) !void {
    try writer.writeAll("def emittedObligations : List ObligationRow := ");
    const count = countWritableObligations(set, proof_surface);
    if (count == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    var emitted_index: usize = 0;
    for (set.obligations) |row| {
        if (proof_surface and !obligationWritableInProofSurface(row)) continue;
        try writer.writeAll("  ");
        try writeObligationRow(writer, row);
        try writeListSeparator(writer, emitted_index, count);
        emitted_index += 1;
    }
    try writer.writeAll("]\n");
}

fn countWritableAssumptions(set: obligation.ObligationSet, proof_surface: bool) usize {
    var count: usize = 0;
    for (set.assumptions) |row| {
        if (proof_surface and !assumptionWritableInProofSurface(row)) continue;
        count += 1;
    }
    return count;
}

fn countWritableObligations(set: obligation.ObligationSet, proof_surface: bool) usize {
    var count: usize = 0;
    for (set.obligations) |row| {
        if (proof_surface and !obligationWritableInProofSurface(row)) continue;
        count += 1;
    }
    return count;
}

fn assumptionWritableInProofSurface(row: obligation.Assumption) bool {
    return if (row.formula) |formula| formulaWritableInLeanModule(formula) else false;
}

fn obligationWritableInProofSurface(row: obligation.Obligation) bool {
    return switch (row.kind) {
        .logical => |logical| formulaWritableInLeanModule(logical.formula),
        .runtime_guard => |guard| formulaWritableInLeanModule(guard.formula),
        else => true,
    };
}

fn formulaWritableInLeanModule(formula: obligation.FormulaRef) bool {
    return switch (formula) {
        .term => true,
        .origin_value => false,
    };
}

fn writeAssumptionRow(writer: anytype, row: obligation.Assumption) !void {
    try writer.writeAll("{ id := ");
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
}

fn writeObligationRow(writer: anytype, row: obligation.Obligation) !void {
    try writer.writeAll("{ id := ");
    try writer.print("{d}", .{row.id});
    try writer.writeAll(", owner := ");
    try writeLeanString(writer, ownerName(row.owner));
    try writer.writeAll(", kind := ");
    try writeObligationKind(writer, row.kind);
    try writer.writeAll(" }");
}

fn writeSemanticDefinitions(writer: anytype, set: obligation.ObligationSet, proof_surface: bool) !void {
    var emitted_any = false;

    try writeRowDefinitions(writer, set, proof_surface);

    for (set.obligations) |row| {
        switch (kindSemanticSupport(set, row.owner, row.kind)) {
            .supported => {},
            .unsupported => continue,
        }
        if (!emitted_any) {
            try writer.writeAll("-- Semantic proposition names for proof attachment.\n");
            emitted_any = true;
        }
        try writer.writeAll("def emittedObligation_");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(" : Prop :=\n  obligationDenotes emittedManifest ");
        try writeObligationRow(writer, row);
        try writer.writeAll("\n\n");
    }

    for (set.queries) |query| {
        if (!try querySupportsSemantics(set, query)) continue;
        if (!emitted_any) {
            try writer.writeAll("-- Semantic proposition names for proof attachment.\n");
            emitted_any = true;
        }
        try writer.writeAll("def emittedQuery_");
        try writer.print("{d}", .{query.id});
        try writer.writeAll(" : Prop :=\n");
        if (query.kind == .loop_induction) {
            const summary_id = query.loop_summary_id orelse return error.InvalidDependency;
            const summary = obligation.findById(set.loop_summaries, summary_id) orelse
                return error.InvalidDependency;
            const has_dependencies = loopSummaryHasDependencies(summary);
            const direct_storage = loopSummaryUsesDirectStorageCertificate(summary);
            try writer.writeAll(if (direct_storage)
                "  Ora.Loop.directStorageLoopProofAuthorizationFromAssumptions emittedManifest\n    "
            else if (has_dependencies)
                "  Ora.Loop.loopProofAuthorizationFromAssumptionsIn emittedManifest\n    "
            else
                "  Ora.Loop.loopProofAuthorizationFromAssumptions emittedManifest\n    ");
            try writeAssumptionRowsByIds(writer, set, query.assumption_ids);
            if (has_dependencies) try writer.writeAll("\n    emittedLoopSummaries");
            try writer.print("\n    emittedLoopSummary_{d}\n\n", .{summary_id});
            try writer.print("def emittedVerifiedQuery_{d} : Prop :=\n", .{query.id});
            try writer.writeAll(if (direct_storage)
                "  Ora.Loop.directStorageLoopVerifiedFromAssumptions emittedManifest\n    "
            else if (has_dependencies)
                "  Ora.Loop.loopVerifiedFromAssumptionsIn emittedManifest\n    "
            else
                "  Ora.Loop.loopVerifiedFromAssumptions emittedManifest\n    ");
            try writeAssumptionRowsByIds(writer, set, query.assumption_ids);
            if (has_dependencies) try writer.writeAll("\n    emittedLoopSummaries");
            try writer.print("\n    emittedLoopSummary_{d}\n\n", .{summary_id});
            if (set.terms.len > lean_term_chunk_size) {
                try writer.writeAll(
                    "section\n" ++
                        "-- Do not delta-reduce the complete package table while applying the\n" ++
                        "-- generic soundness theorem. This local attribute ends before clients\n" ++
                        "-- import the module and evaluate the canonical manifest.\n" ++
                        "attribute [local irreducible] emittedManifest\n\n",
                );
            }
            try writer.print(
                "theorem emittedQuery_{d}_sound : emittedQuery_{d} -> emittedVerifiedQuery_{d} := by\n" ++
                    "  intro proof\n" ++
                    "  exact Ora.Loop.{s} emittedManifest ",
                .{
                    query.id,
                    query.id,
                    query.id,
                    if (direct_storage)
                        "direct_storage_loop_authorization_sound"
                    else if (has_dependencies)
                        "loop_authorization_sound_in"
                    else
                        "loop_authorization_sound",
                },
            );
            try writeAssumptionRowsByIds(writer, set, query.assumption_ids);
            if (has_dependencies) try writer.writeAll(" emittedLoopSummaries");
            try writer.print(" emittedLoopSummary_{d} proof\n\n", .{summary_id});
            if (set.terms.len > lean_term_chunk_size) try writer.writeAll("end\n\n");
            continue;
        }
        for (query.obligation_ids, 0..) |obligation_id, index| {
            const row = obligation.findById(set.obligations, obligation_id) orelse return error.InvalidDependency;
            if (index != 0) try writer.writeAll(" /\\\n");
            try writer.writeAll("  obligationFollowsFromAssumptions emittedManifest\n    ");
            try writeAssumptionRowsByIds(writer, set, query.assumption_ids);
            try writer.writeAll("\n    ");
            try writeObligationRow(writer, row);
        }
        try writer.writeAll("\n\n");
    }
}

fn loopSummaryHasDependencies(summary: obligation.LoopSummaryRow) bool {
    for (summary.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
        .summary_dependency => return true,
        else => {},
    };
    return false;
}

pub fn loopSummaryUsesDirectStorageCertificate(summary: obligation.LoopSummaryRow) bool {
    var has_storage = false;
    for (summary.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
        .storage_read, .storage_write => has_storage = true,
        .summary_dependency => return false,
        else => {},
    };
    return has_storage;
}

fn writeRowDefinitions(writer: anytype, set: obligation.ObligationSet, proof_surface: bool) !void {
    for (set.assumptions) |row| {
        if (proof_surface and !assumptionWritableInProofSurface(row)) continue;
        try writer.writeAll("def emittedAssumptionRow_");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(" : AssumptionRow :=\n  ");
        try writeAssumptionRow(writer, row);
        try writer.writeAll("\n\n");
    }

    for (set.obligations) |row| {
        if (proof_surface and !obligationWritableInProofSurface(row)) continue;
        try writer.writeAll("def emittedObligationRow_");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(" : ObligationRow :=\n  ");
        try writeObligationRow(writer, row);
        try writer.writeAll("\n\n");
    }
}

fn writeAssumptionRowsByIds(writer: anytype, set: obligation.ObligationSet, ids: []const obligation.Id) !void {
    if (ids.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(", ");
        const row = obligation.findById(set.assumptions, id) orelse return error.InvalidDependency;
        try writeAssumptionRow(writer, row);
    }
    try writer.writeByte(']');
}

pub fn querySemanticSupport(set: obligation.ObligationSet, query: obligation.VerificationQuery) SemanticSupport {
    switch (query.kind) {
        .loop_induction => return loopInductionQuerySemanticSupport(set, query),
        .loop_invariant_step,
        .loop_body_safety,
        .loop_invariant_post,
        => return .{ .unsupported = .unsupported_obligation_kind },
        else => {},
    }
    if (query.obligation_ids.len == 0) return .{ .unsupported = .empty_query };

    switch (queryAssumptionsSemanticSupport(set, query)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = reason },
    }

    for (query.obligation_ids) |obligation_id| {
        const row = obligation.findById(set.obligations, obligation_id) orelse return .{ .unsupported = .invalid_dependency };
        switch (kindSemanticSupport(set, query.owner, row.kind)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }

    return .supported;
}

fn queryAssumptionsSemanticSupport(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) SemanticSupport {
    for (query.assumption_ids) |assumption_id| {
        const row = obligation.findById(set.assumptions, assumption_id) orelse return .{ .unsupported = .invalid_dependency };
        if (row.formula) |formula| {
            switch (formulaSemanticSupport(set, formula)) {
                .supported => {},
                .unsupported => |reason| return .{ .unsupported = reason },
            }
        } else {
            return .{ .unsupported = .unsupported_origin_value };
        }
    }
    return .supported;
}

fn loopInductionQuerySemanticSupport(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
) SemanticSupport {
    const summary_id = query.loop_summary_id orelse
        return .{ .unsupported = .loop_summary_missing };
    const summary = obligation.findById(set.loop_summaries, summary_id) orelse
        return .{ .unsupported = .loop_summary_missing };
    if (!loopSummaryOwnsQuery(summary, query)) {
        return .{ .unsupported = .loop_summary_query_mismatch };
    }
    if (summary.unsupported_reasons.len != 0) {
        return .{ .unsupported = .{ .unsupported_loop_summary = summary.unsupported_reasons[0] } };
    }
    if (summary.validationReason()) |reason| {
        return .{ .unsupported = .{ .unsupported_loop_summary = reason } };
    }
    if (set.fixedIndexBoundsReason(summary.preheader_timeline)) |reason| {
        return .{ .unsupported = .{ .unsupported_loop_summary = reason } };
    }
    for (summary.edges) |edge| if (set.fixedIndexBoundsReason(edge.timeline)) |reason| {
        return .{ .unsupported = .{ .unsupported_loop_summary = reason } };
    };
    if (summary.variables.len == 0 or
        summary.init_formulas.len != summary.variables.len or
        summary.invariant_formulas.len == 0 or
        summary.edges.len == 0)
    {
        return .{ .unsupported = .{ .unsupported_loop_summary = .loop_update_not_scalar_assignment } };
    }
    switch (queryAssumptionsSemanticSupport(set, query)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = reason },
    }

    for (summary.context_variables) |variable| {
        if (variable.id == null) return .{ .unsupported = .{ .unsupported_loop_summary = .loop_identity_missing } };
        switch (optionalTypeSupportsInteger(variable.ty)) {
            .supported => {},
            .unsupported => return .{ .unsupported = .{ .unsupported_loop_summary = .loop_variable_not_registered_integer } },
        }
        for (summary.variables) |loop_variable| {
            if (loop_variable.id != null and obligation.freeVarIdEql(variable.id.?, loop_variable.id.?)) {
                return .{ .unsupported = .{ .unsupported_loop_summary = .loop_identity_missing } };
            }
        }
    }
    for (summary.variables, 0..) |variable, index| {
        if (variable.index != index or variable.id == null) {
            return .{ .unsupported = .{ .unsupported_loop_summary = .loop_identity_missing } };
        }
        switch (optionalTypeSupportsInteger(variable.ty)) {
            .supported => {},
            .unsupported => return .{ .unsupported = .{ .unsupported_loop_summary = .loop_variable_not_registered_integer } },
        }
    }

    for (summary.init_formulas, summary.variables) |formula, variable| switch (valueFormulaSemanticSupportForType(set, formula, variable.ty)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = .{ .unsupported_loop_summary = loopValueUnsupportedReason(reason) } },
    };
    for (summary.init_formulas) |formula| if (formulaReferencesSummaryDependencyVariable(set, summary, formula)) {
        return .{ .unsupported = .{ .unsupported_loop_summary = .loop_summary_dependency_value_out_of_scope } };
    };
    for (summary.invariant_formulas) |formula| {
        if (formulaReferencesSummaryDependencyVariable(set, summary, formula)) {
            return .{ .unsupported = .{ .unsupported_loop_summary = .loop_summary_dependency_value_out_of_scope } };
        }
        switch (formulaSemanticSupport(set, formula)) {
            .supported => {},
            .unsupported => return .{ .unsupported = .{ .unsupported_loop_summary = .loop_formula_unsupported } },
        }
    }
    for (summary.edges) |edge| {
        switch (edge.target orelse
            return .{ .unsupported = .{ .unsupported_loop_summary = .loop_edge_target_unresolved } }) {
            .backedge, .successor => if (edge.target_formulas.len != 0) return .{
                .unsupported = .{ .unsupported_loop_summary = .loop_formula_unsupported },
            },
            .function_return, .error_return => {},
            .explicit_revert => return .{
                .unsupported = .{ .unsupported_loop_summary = .loop_control_target_unsupported },
            },
        }
        var dependency_count: u32 = 0;
        for (edge.timeline) |event| switch (event.payload) {
            .path_assumption => |path| for (path.formulas) |expression| {
                const formula = expression.formula;
                if (formulaReferencesSummaryDependencyVariable(set, summary, formula)) {
                    return .{ .unsupported = .{ .unsupported_loop_summary = .loop_summary_dependency_value_out_of_scope } };
                }
                switch (formulaSemanticSupport(set, formula)) {
                    .supported => {},
                    .unsupported => return .{ .unsupported = .{ .unsupported_loop_summary = .loop_formula_unsupported } },
                }
            },
            .safety_checkpoint => |checkpoint| for (checkpoint.formulas) |expression| {
                const formula = expression.formula;
                if (formulaReferencesUnavailableCheckpointDependency(
                    set,
                    summary,
                    edge,
                    dependency_count,
                    formula,
                )) {
                    return .{ .unsupported = .{ .unsupported_loop_summary = .loop_summary_dependency_value_out_of_scope } };
                }
                switch (formulaSemanticSupport(set, formula)) {
                    .supported => {},
                    .unsupported => return .{ .unsupported = .{ .unsupported_loop_summary = .loop_formula_unsupported } },
                }
            },
            .summary_dependency => dependency_count += 1,
            .storage_read, .indexed_region_read, .storage_write => {},
        };
        for (edge.target_formulas) |formula| {
            if (formulaReferencesSummaryDependencyVariable(set, summary, formula)) {
                return .{ .unsupported = .{ .unsupported_loop_summary = .loop_summary_dependency_value_out_of_scope } };
            }
            switch (formulaSemanticSupport(set, formula)) {
                .supported => {},
                .unsupported => return .{ .unsupported = .{ .unsupported_loop_summary = .loop_formula_unsupported } },
            }
        }
        for (edge.next_state, summary.variables) |binding, variable| switch (valueFormulaSemanticSupportForType(set, binding.value, variable.ty)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = .{ .unsupported_loop_summary = loopValueUnsupportedReason(reason) } },
        };
    }
    for (summary.post_formulas) |formula| {
        if (formulaReferencesSummaryDependencyVariable(set, summary, formula)) {
            return .{ .unsupported = .{ .unsupported_loop_summary = .loop_summary_dependency_value_out_of_scope } };
        }
        switch (formulaSemanticSupport(set, formula)) {
            .supported => {},
            .unsupported => return .{ .unsupported = .{ .unsupported_loop_summary = .loop_formula_unsupported } },
        }
    }
    return .supported;
}

fn formulaReferencesSummaryDependencyVariable(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    formula: obligation.FormulaRef,
) bool {
    return switch (formula) {
        .term => |id| termReferencesSummaryDependencyVariable(set, summary, id, set.terms.len + 1),
        .origin_value => false,
    };
}

fn formulaReferencesUnavailableCheckpointDependency(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    edge: obligation.LoopEdgeRow,
    after_dependency_count: u32,
    formula: obligation.FormulaRef,
) bool {
    return switch (formula) {
        .term => |id| termReferencesUnavailableCheckpointDependency(
            set,
            summary,
            edge,
            after_dependency_count,
            id,
            set.terms.len + 1,
        ),
        .origin_value => false,
    };
}

fn termReferencesUnavailableCheckpointDependency(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    edge: obligation.LoopEdgeRow,
    after_dependency_count: u32,
    id: obligation.TermId,
    fuel: usize,
) bool {
    if (fuel == 0) return true;
    const term = termById(set, id) orelse return true;
    return switch (term) {
        .variable => |variable| switch (variable) {
            .free => |free| !summaryDependencyVariableAvailableAtCheckpoint(
                set,
                summary,
                edge,
                after_dependency_count,
                free.id,
            ),
            .bound => false,
        },
        .old => |operand| termReferencesUnavailableCheckpointDependency(
            set,
            summary,
            edge,
            after_dependency_count,
            operand,
            fuel - 1,
        ),
        .unary => |unary| termReferencesUnavailableCheckpointDependency(
            set,
            summary,
            edge,
            after_dependency_count,
            unary.operand,
            fuel - 1,
        ),
        .binary => |binary| termReferencesUnavailableCheckpointDependency(
            set,
            summary,
            edge,
            after_dependency_count,
            binary.lhs,
            fuel - 1,
        ) or termReferencesUnavailableCheckpointDependency(
            set,
            summary,
            edge,
            after_dependency_count,
            binary.rhs,
            fuel - 1,
        ),
        .refinement_predicate => |predicate| blk: {
            if (termReferencesUnavailableCheckpointDependency(
                set,
                summary,
                edge,
                after_dependency_count,
                predicate.value,
                fuel - 1,
            )) break :blk true;
            for (predicate.args) |arg| {
                if (termReferencesUnavailableCheckpointDependency(
                    set,
                    summary,
                    edge,
                    after_dependency_count,
                    arg,
                    fuel - 1,
                )) break :blk true;
            }
            break :blk false;
        },
        .quantified => |quantified| blk: {
            if (quantified.condition) |condition| {
                if (termReferencesUnavailableCheckpointDependency(
                    set,
                    summary,
                    edge,
                    after_dependency_count,
                    condition,
                    fuel - 1,
                )) break :blk true;
            }
            break :blk termReferencesUnavailableCheckpointDependency(
                set,
                summary,
                edge,
                after_dependency_count,
                quantified.body,
                fuel - 1,
            );
        },
        .bool_lit,
        .int_lit,
        .result,
        .place_read,
        => false,
    };
}

fn summaryDependencyVariableAvailableAtCheckpoint(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    edge: obligation.LoopEdgeRow,
    after_dependency_count: u32,
    id: obligation.FreeVarId,
) bool {
    var owned_by_dependency = false;
    for (summary.edges) |candidate_edge| {
        var dependency_index: u32 = 0;
        for (candidate_edge.timeline) |event| switch (event.payload) {
            .summary_dependency => |dependency| {
                const inner = obligation.findById(set.loop_summaries, dependency.summary_id) orelse
                    return false;
                for (inner.variables) |variable| {
                    if (variable.id == null or !obligation.freeVarIdEql(variable.id.?, id)) continue;
                    owned_by_dependency = true;
                    if (candidate_edge.id == edge.id and dependency_index < after_dependency_count) {
                        return true;
                    }
                }
                dependency_index += 1;
            },
            else => {},
        };
    }
    return !owned_by_dependency;
}

fn termReferencesSummaryDependencyVariable(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    id: obligation.TermId,
    fuel: usize,
) bool {
    if (fuel == 0) return true;
    const term = termById(set, id) orelse return true;
    return switch (term) {
        .variable => |variable| switch (variable) {
            .free => |free| summaryDependencyOwnsFreeVariable(set, summary, free.id),
            .bound => false,
        },
        .old => |operand| termReferencesSummaryDependencyVariable(set, summary, operand, fuel - 1),
        .unary => |unary| termReferencesSummaryDependencyVariable(set, summary, unary.operand, fuel - 1),
        .binary => |binary| termReferencesSummaryDependencyVariable(set, summary, binary.lhs, fuel - 1) or
            termReferencesSummaryDependencyVariable(set, summary, binary.rhs, fuel - 1),
        .refinement_predicate => |predicate| blk: {
            if (termReferencesSummaryDependencyVariable(set, summary, predicate.value, fuel - 1)) break :blk true;
            for (predicate.args) |arg| {
                if (termReferencesSummaryDependencyVariable(set, summary, arg, fuel - 1)) break :blk true;
            }
            break :blk false;
        },
        .quantified => |quantified| blk: {
            if (quantified.condition) |condition| {
                if (termReferencesSummaryDependencyVariable(set, summary, condition, fuel - 1)) break :blk true;
            }
            break :blk termReferencesSummaryDependencyVariable(set, summary, quantified.body, fuel - 1);
        },
        .bool_lit,
        .int_lit,
        .result,
        .place_read,
        => false,
    };
}

fn summaryDependencyOwnsFreeVariable(
    set: obligation.ObligationSet,
    summary: obligation.LoopSummaryRow,
    id: obligation.FreeVarId,
) bool {
    for (summary.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
        .summary_dependency => |dependency| {
            const inner = obligation.findById(set.loop_summaries, dependency.summary_id) orelse return true;
            for (inner.variables) |variable| {
                if (variable.id != null and obligation.freeVarIdEql(variable.id.?, id)) return true;
            }
        },
        else => {},
    };
    return false;
}

fn loopValueUnsupportedReason(reason: SemanticUnsupportedReason) obligation.LoopUnsupportedReason {
    return switch (reason) {
        .missing_type => .loop_value_type_unresolved,
        .unsupported_type => .loop_value_type_unresolved,
        .unsupported_comparison_width,
        .unsupported_arithmetic_width,
        => .loop_value_width_mismatch,
        .unknown_signedness,
        .mixed_signedness,
        .unknown_arithmetic_signedness,
        .mixed_arithmetic_signedness,
        => .loop_value_signedness_mismatch,
        else => .loop_formula_unsupported,
    };
}

fn valueFormulaSemanticSupport(set: obligation.ObligationSet, formula: obligation.FormulaRef) SemanticSupport {
    return switch (formula) {
        .term => |id| integerValueTermSemanticSupport(set, id, set.terms.len + 1),
        .origin_value => .{ .unsupported = .unsupported_origin_value },
    };
}

fn valueFormulaSemanticSupportForType(
    set: obligation.ObligationSet,
    formula: obligation.FormulaRef,
    ty: obligation.TypeRef,
) SemanticSupport {
    const expected = compilerIntegerInfo(ty) orelse
        return .{ .unsupported = .{ .unsupported_type = ty } };
    return switch (formula) {
        .term => |id| integerValueTermSemanticSupportForInfo(set, id, set.terms.len + 1, expected),
        .origin_value => .{ .unsupported = .unsupported_origin_value },
    };
}

fn loopSummaryOwnsQuery(
    summary: obligation.LoopSummaryRow,
    query: obligation.VerificationQuery,
) bool {
    var has_query = false;
    for (summary.query_ids.induction) |id| {
        if (id == query.id) has_query = true;
    }
    if (!has_query or query.owner != .function or summary.owner != .statement) return false;
    return std.mem.eql(u8, query.owner.function.name, summary.owner.statement.function_name);
}

fn querySupportsSemantics(set: obligation.ObligationSet, query: obligation.VerificationQuery) !bool {
    return switch (querySemanticSupport(set, query)) {
        .supported => true,
        .unsupported => |reason| switch (reason) {
            .invalid_dependency => error.InvalidDependency,
            else => false,
        },
    };
}

fn kindSemanticSupport(set: obligation.ObligationSet, owner: obligation.Owner, kind: obligation.Kind) SemanticSupport {
    return switch (kind) {
        .logical => |logical| formulaSemanticSupport(set, logical.formula),
        .runtime_guard => |guard| formulaSemanticSupport(set, guard.formula),
        .effect_frame => |effect| effectFrameSemanticSupport(set, owner, effect),
        .resource => |resource| resourceSemanticSupport(set, resource),
        .type_wf,
        .type_relation,
        .region_relation,
        .quantifier,
        .filtered_input,
        .backend_fact,
        => .{ .unsupported = .unsupported_obligation_kind },
    };
}

fn resourceSemanticSupport(set: obligation.ObligationSet, resource: obligation.ResourceGoal) SemanticSupport {
    const amount = resource.amount orelse return .{ .unsupported = .unsupported_origin_value };
    switch (amount) {
        .term => |id| switch (u256ValueTermSemanticSupport(set, id, set.terms.len + 1)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        },
        .origin_value => return .{ .unsupported = .unsupported_origin_value },
    }

    return switch (resource.property) {
        .amount_non_negative => .supported,
        .source_sufficient => switch (resource.op) {
            .move => if (resource.source != null and resource.destination != null)
                .supported
            else
                .{ .unsupported = .invalid_dependency },
            .destroy => if (resource.source != null)
                .supported
            else
                .{ .unsupported = .invalid_dependency },
            .create => .{ .unsupported = .invalid_dependency },
        },
        .destination_no_overflow => switch (resource.op) {
            .move => if (resource.source != null and resource.destination != null)
                .supported
            else
                .{ .unsupported = .invalid_dependency },
            .create => if (resource.destination != null)
                .supported
            else
                .{ .unsupported = .invalid_dependency },
            .destroy => .{ .unsupported = .invalid_dependency },
        },
        .same_place_identity,
        .conservation,
        => if (resource.op == .move and resource.source != null and resource.destination != null)
            .supported
        else
            .{ .unsupported = .invalid_dependency },
    };
}

fn effectFrameSemanticSupport(
    set: obligation.ObligationSet,
    owner: obligation.Owner,
    effect: obligation.EffectFrameGoal,
) SemanticSupport {
    return switch (effect.relation) {
        .write_covered_by_modifies,
        .read_preserved_by_frame,
        => .supported,
        .read_preserved_by_key_evidence => keyEvidenceFrameSemanticSupport(set, owner, effect),
        .lock_covers_write,
        .external_call_frame,
        => .{ .unsupported = .{ .unsupported_effect_frame_relation = effect.relation } },
    };
}

fn keyEvidenceFrameSemanticSupport(
    set: obligation.ObligationSet,
    owner: obligation.Owner,
    effect: obligation.EffectFrameGoal,
) SemanticSupport {
    if (effect.evidence.len == 0) return .{ .unsupported = .missing_key_disjoint_evidence };
    for (effect.evidence) |evidence| {
        switch (keyDisjointEvidenceSemanticSupport(set, owner, evidence)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }
    return .supported;
}

fn keyDisjointEvidenceSemanticSupport(
    set: obligation.ObligationSet,
    owner: obligation.Owner,
    evidence: obligation.KeyDisjointEvidence,
) SemanticSupport {
    switch (evidence.kind) {
        .free_var_disequality => {},
    }
    if (!keyEvidencePathMatches(evidence.read, evidence.write, evidence.key_index, evidence.lhs, evidence.rhs)) {
        return .{ .unsupported = .key_disjoint_evidence_path_mismatch };
    }
    const assumption = obligation.findById(set.assumptions, evidence.assumption_id) orelse
        return .{ .unsupported = .unsupported_key_disjoint_evidence_formula };
    if (!ownerEqual(assumption.owner, owner)) {
        return .{ .unsupported = .key_disjoint_evidence_owner_mismatch };
    }
    if (assumption.kind != .requires) {
        return .{ .unsupported = .unsupported_key_disjoint_evidence_formula };
    }
    const formula = assumption.formula orelse return .{ .unsupported = .unsupported_key_disjoint_evidence_formula };
    if (formula != .term) return .{ .unsupported = .unsupported_key_disjoint_evidence_formula };
    const term = termById(set, formula.term) orelse return .{ .unsupported = .invalid_dependency };
    if (term != .binary or term.binary.op != .ne) {
        return .{ .unsupported = .unsupported_key_disjoint_evidence_formula };
    }
    const lhs = freeVariableByTermId(set, term.binary.lhs) orelse
        return .{ .unsupported = .unsupported_key_disjoint_evidence_formula };
    const rhs = freeVariableByTermId(set, term.binary.rhs) orelse
        return .{ .unsupported = .unsupported_key_disjoint_evidence_formula };
    if (!freeVarPairMatches(lhs.id, rhs.id, evidence.lhs, evidence.rhs)) {
        return .{ .unsupported = .key_disjoint_evidence_path_mismatch };
    }
    const lhs_info = compilerIntegerInfo(lhs.ty orelse
        return .{ .unsupported = .key_disjoint_evidence_type_unsupported }) orelse
        return .{ .unsupported = .key_disjoint_evidence_type_unsupported };
    const rhs_info = compilerIntegerInfo(rhs.ty orelse
        return .{ .unsupported = .key_disjoint_evidence_type_unsupported }) orelse
        return .{ .unsupported = .key_disjoint_evidence_type_unsupported };
    if (!compilerIntegerInfoEql(lhs_info, rhs_info)) {
        return .{ .unsupported = .key_disjoint_evidence_type_unsupported };
    }
    return .supported;
}

fn keyEvidencePathMatches(
    read: obligation.PlaceRef,
    write: obligation.PlaceRef,
    key_index: u32,
    lhs: obligation.FreeVarId,
    rhs: obligation.FreeVarId,
) bool {
    if (read.region == .none or write.region == .none) return false;
    if (std.mem.eql(u8, read.root, "$computed_storage") or std.mem.eql(u8, write.root, "$computed_storage")) return false;
    if (read.region != write.region) return false;
    if (!std.mem.eql(u8, read.root, write.root)) return false;
    if (!stringSlicesEql(read.fields, write.fields)) return false;
    if (read.projection_domain != write.projection_domain) return false;
    if (read.keys.len != write.keys.len) return false;
    const index: usize = @intCast(key_index);
    if (index >= read.keys.len) return false;
    for (read.keys[0..index], write.keys[0..index]) |read_key, write_key| {
        if (!obligation.placeKeyEql(read_key, write_key)) return false;
    }
    if (read.keys[index] != .parameter or write.keys[index] != .parameter) return false;
    const read_id = read.keys[index].parameter;
    const write_id = write.keys[index].parameter;
    if (obligation.freeVarIdEql(read_id, write_id)) return false;
    return freeVarPairMatches(lhs, rhs, read_id, write_id);
}

fn freeVarPairMatches(
    lhs: obligation.FreeVarId,
    rhs: obligation.FreeVarId,
    first: obligation.FreeVarId,
    second: obligation.FreeVarId,
) bool {
    return (obligation.freeVarIdEql(lhs, first) and obligation.freeVarIdEql(rhs, second)) or
        (obligation.freeVarIdEql(lhs, second) and obligation.freeVarIdEql(rhs, first));
}

fn freeVariableByTermId(set: obligation.ObligationSet, id: obligation.TermId) ?obligation.FreeVarRef {
    const term = termById(set, id) orelse return null;
    if (term != .variable or term.variable != .free) return null;
    return term.variable.free;
}

fn stringSlicesEql(lhs: []const []const u8, rhs: []const []const u8) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!std.mem.eql(u8, left, right)) return false;
    }
    return true;
}

fn formulaSemanticSupport(set: obligation.ObligationSet, formula: obligation.FormulaRef) SemanticSupport {
    return switch (formula) {
        .term => |id| formulaTermSemanticSupport(set, id, set.terms.len + 1),
        .origin_value => .{ .unsupported = .unsupported_origin_value },
    };
}

fn termById(set: obligation.ObligationSet, id: obligation.TermId) ?obligation.Term {
    if (id >= set.terms.len) return null;
    return set.terms[id];
}

fn formulaTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .bool_lit => .supported,
        .variable => |variable| varRefSupportsBoolFormula(variable),
        .refinement_predicate => |predicate| refinementPredicateSemanticSupport(set, predicate, fuel - 1),
        .unary => |unary| switch (unary.op) {
            .not => formulaTermSemanticSupport(set, unary.operand, fuel - 1),
            .neg => .{ .unsupported = .unsupported_term_kind },
        },
        .binary => |binary| binaryFormulaSemanticSupport(set, binary, fuel - 1),
        .quantified => |quantified| quantifiedSemanticSupport(set, quantified, fuel - 1),
        .int_lit,
        .old,
        .result,
        .place_read,
        => .{ .unsupported = .unsupported_term_kind },
    };
}

fn valueTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .bool_lit => .supported,
        .int_lit => |literal| optionalTypeSupportsInteger(literal.ty),
        .variable => |variable| varRefSupportsBoolOrIntegerValue(variable),
        .result => .supported,
        .place_read => .supported,
        .binary => |binary| switch (binary.op) {
            .add, .sub, .mul, .pow, .shl, .shr, .div, .mod, .bit_and, .bit_xor => binaryCarrierArithmeticValueSemanticSupport(set, binary, fuel - 1),
            else => .{ .unsupported = .unsupported_term_kind },
        },
        .old => |operand| oldPlaceReadSemanticSupport(set, operand),
        .unary,
        .refinement_predicate,
        .quantified,
        => .{ .unsupported = .unsupported_term_kind },
    };
}

fn integerValueTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .int_lit => |literal| optionalTypeSupportsInteger(literal.ty),
        .variable => |variable| varRefSupportsIntegerValue(variable),
        .result => .supported,
        .place_read => .supported,
        .binary => |binary| switch (binary.op) {
            .add, .sub, .mul, .pow, .shl, .shr, .div, .mod, .bit_and, .bit_xor => binaryCarrierArithmeticValueSemanticSupport(set, binary, fuel - 1),
            else => .{ .unsupported = .unsupported_term_kind },
        },
        .old => |operand| oldPlaceReadSemanticSupport(set, operand),
        else => .{ .unsupported = .unsupported_term_kind },
    };
}

fn integerValueTermSemanticSupportForInfo(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
    expected: CompilerIntegerInfo,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .int_lit => |literal| optionalTypeMatchesIntegerInfo(literal.ty, expected),
        .variable => |variable| varRefMatchesIntegerInfo(variable, expected),
        .result, .place_read => .supported,
        .binary => |binary| blk: {
            const actual = switch (arithmeticTypeSupport(binary.ty)) {
                .supported => |info| info,
                .unsupported => |reason| break :blk .{ .unsupported = reason },
            };
            if (!compilerIntegerInfoEql(actual, expected)) {
                break :blk .{ .unsupported = if (actual.signed != expected.signed)
                    .mixed_arithmetic_signedness
                else
                    .unsupported_arithmetic_width };
            }
            break :blk firstUnsupported(.{
                integerValueTermSemanticSupportForInfo(set, binary.lhs, fuel - 1, expected),
                integerValueTermSemanticSupportForInfo(set, binary.rhs, fuel - 1, expected),
            });
        },
        .old => |operand| oldPlaceReadSemanticSupport(set, operand),
        else => .{ .unsupported = .unsupported_term_kind },
    };
}

fn unsignedIntegerValueTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .int_lit => |literal| optionalTypeSupportsUnsignedInteger(literal.ty),
        .variable => |variable| varRefSupportsUnsignedIntegerValue(variable),
        .result => .supported,
        .place_read => .supported,
        .binary => |binary| switch (binary.op) {
            .add, .sub, .mul, .pow, .shl, .shr, .div, .mod, .bit_and, .bit_xor => binaryUnsignedArithmeticValueSemanticSupport(set, binary, fuel - 1),
            else => .{ .unsupported = .unsupported_term_kind },
        },
        .old => |operand| oldPlaceReadSemanticSupport(set, operand),
        else => .{ .unsupported = .unsupported_term_kind },
    };
}

fn u256ValueTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .int_lit => |literal| optionalTypeSupportsU256(literal.ty),
        .variable => |variable| varRefSupportsU256Value(variable),
        .result, .place_read => .supported,
        .binary => |binary| blk: {
            switch (optionalTypeSupportsU256(binary.ty)) {
                .supported => {},
                .unsupported => |reason| break :blk .{ .unsupported = reason },
            }
            break :blk firstUnsupported(.{
                u256ValueTermSemanticSupport(set, binary.lhs, fuel - 1),
                u256ValueTermSemanticSupport(set, binary.rhs, fuel - 1),
            });
        },
        .old => |operand| oldPlaceReadSemanticSupport(set, operand),
        else => .{ .unsupported = .unsupported_term_kind },
    };
}

fn oldPlaceReadSemanticSupport(set: obligation.ObligationSet, operand: obligation.TermId) SemanticSupport {
    const term = termById(set, operand) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .place_read => .supported,
        else => .{ .unsupported = .unsupported_term_kind },
    };
}

fn binaryFormulaSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    return switch (binary.op) {
        .eq, .ne => firstUnsupported(.{
            valueTermSemanticSupport(set, binary.lhs, fuel),
            valueTermSemanticSupport(set, binary.rhs, fuel),
        }),
        .lt, .le, .gt, .ge => unsignedComparisonSemanticSupport(set, binary, fuel),
        .slt, .sle, .sgt, .sge => signedComparisonSemanticSupport(set, binary, fuel),
        .and_, .or_, .implies => firstUnsupported(.{
            formulaTermSemanticSupport(set, binary.lhs, fuel),
            formulaTermSemanticSupport(set, binary.rhs, fuel),
        }),
        .add,
        .sub,
        .mul,
        .pow,
        .shl,
        .shr,
        .div,
        .mod,
        .bit_and,
        .bit_xor,
        => .{ .unsupported = .unsupported_term_kind },
    };
}

fn signedIntegerValueTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .int_lit => |literal| optionalTypeSupportsSignedInteger(literal.ty),
        .variable => |variable| varRefSupportsSignedIntegerValue(variable),
        .result => .supported,
        .place_read => .supported,
        .binary => |binary| switch (binary.op) {
            .add, .sub, .mul, .pow, .shl, .shr, .div, .mod, .bit_and, .bit_xor => binarySignedArithmeticValueSemanticSupport(set, binary, fuel - 1),
            else => .{ .unsupported = .unsupported_term_kind },
        },
        .old => |operand| oldPlaceReadSemanticSupport(set, operand),
        else => .{ .unsupported = .unsupported_term_kind },
    };
}

fn signedComparisonSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    const ty = binary.ty orelse return .{ .unsupported = .unknown_signedness };
    const info = compilerIntegerInfo(ty) orelse
        return .{ .unsupported = .unknown_signedness };
    if (!info.signed) return .{ .unsupported = .mixed_signedness };
    return firstUnsupported(.{
        integerValueTermSemanticSupportForInfo(set, binary.lhs, fuel, info),
        integerValueTermSemanticSupportForInfo(set, binary.rhs, fuel, info),
    });
}

fn unsignedComparisonSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    const ty = binary.ty orelse return .{ .unsupported = .unknown_signedness };
    const info = compilerIntegerInfo(ty) orelse
        return .{ .unsupported = .unknown_signedness };
    if (info.signed) return .{ .unsupported = .mixed_signedness };
    return firstUnsupported(.{
        integerValueTermSemanticSupportForInfo(set, binary.lhs, fuel, info),
        integerValueTermSemanticSupportForInfo(set, binary.rhs, fuel, info),
    });
}

fn binarySignedArithmeticValueSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    const info = switch (arithmeticTypeSupport(binary.ty)) {
        .supported => |info| info,
        .unsupported => |reason| return .{ .unsupported = reason },
    };
    if (!info.signed) return .{ .unsupported = .mixed_arithmetic_signedness };
    return firstUnsupported(.{
        integerValueTermSemanticSupportForInfo(set, binary.lhs, fuel, info),
        integerValueTermSemanticSupportForInfo(set, binary.rhs, fuel, info),
    });
}

fn binaryUnsignedArithmeticValueSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    const info = switch (arithmeticTypeSupport(binary.ty)) {
        .supported => |info| info,
        .unsupported => |reason| return .{ .unsupported = reason },
    };
    if (info.signed) return .{ .unsupported = .mixed_arithmetic_signedness };
    return firstUnsupported(.{
        integerValueTermSemanticSupportForInfo(set, binary.lhs, fuel, info),
        integerValueTermSemanticSupportForInfo(set, binary.rhs, fuel, info),
    });
}

fn binaryCarrierArithmeticValueSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    const info = switch (arithmeticTypeSupport(binary.ty)) {
        .supported => |info| info,
        .unsupported => |reason| return .{ .unsupported = reason },
    };
    return firstUnsupported(.{
        integerValueTermSemanticSupportForInfo(set, binary.lhs, fuel, info),
        integerValueTermSemanticSupportForInfo(set, binary.rhs, fuel, info),
    });
}

fn refinementPredicateSemanticSupport(
    set: obligation.ObligationSet,
    predicate: obligation.RefinementPredicateTerm,
    fuel: usize,
) SemanticSupport {
    switch (integerValueTermSemanticSupport(set, predicate.value, fuel)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = reason },
    }
    for (predicate.args) |arg| {
        switch (integerValueTermSemanticSupport(set, arg, fuel)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }
    return .supported;
}

fn quantifiedSemanticSupport(
    set: obligation.ObligationSet,
    quantified: obligation.QuantifiedTerm,
    fuel: usize,
) SemanticSupport {
    var binder_reason: ?SemanticUnsupportedReason = null;
    switch (optionalTypeSupportsInteger(quantified.binder.ty)) {
        .supported => {},
        .unsupported => |reason| binder_reason = reason,
    }
    if (quantified.condition) |condition| {
        switch (formulaTermSemanticSupport(set, condition, fuel)) {
            .supported => {},
            .unsupported => |reason| {
                if (reasonIsTypedOperationGate(reason)) return .{ .unsupported = reason };
                return .{ .unsupported = binder_reason orelse reason };
            },
        }
    }
    switch (formulaTermSemanticSupport(set, quantified.body, fuel)) {
        .supported => {},
        .unsupported => |reason| {
            if (reasonIsTypedOperationGate(reason)) return .{ .unsupported = reason };
            return .{ .unsupported = binder_reason orelse reason };
        },
    }
    if (binder_reason) |reason| return .{ .unsupported = reason };
    return .supported;
}

fn firstUnsupported(results: anytype) SemanticSupport {
    inline for (results) |result| {
        switch (result) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }
    return .supported;
}

fn reasonIsTypedOperationGate(reason: SemanticUnsupportedReason) bool {
    return switch (reason) {
        .unsupported_comparison_width,
        .unknown_signedness,
        .mixed_signedness,
        .unsupported_arithmetic_width,
        .unknown_arithmetic_signedness,
        .mixed_arithmetic_signedness,
        => true,
        else => false,
    };
}

fn varRefSupportsBoolFormula(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsBool(free.ty),
        .bound => |bound| optionalTypeSupportsBool(bound.ty),
    };
}

fn varRefSupportsBoolOrIntegerValue(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsBoolOrInteger(free.ty),
        .bound => |bound| optionalTypeSupportsBoolOrInteger(bound.ty),
    };
}

fn varRefSupportsIntegerValue(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsInteger(free.ty),
        .bound => |bound| optionalTypeSupportsInteger(bound.ty),
    };
}

fn varRefSupportsSignedIntegerValue(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsSignedInteger(free.ty),
        .bound => |bound| optionalTypeSupportsSignedInteger(bound.ty),
    };
}

fn varRefSupportsUnsignedIntegerValue(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsUnsignedInteger(free.ty),
        .bound => |bound| optionalTypeSupportsUnsignedInteger(bound.ty),
    };
}

fn varRefSupportsU256Value(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsU256(free.ty),
        .bound => |bound| optionalTypeSupportsU256(bound.ty),
    };
}

fn varRefMatchesIntegerInfo(
    variable: obligation.VarRef,
    expected: CompilerIntegerInfo,
) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeMatchesIntegerInfo(free.ty, expected),
        .bound => |bound| optionalTypeMatchesIntegerInfo(bound.ty, expected),
    };
}

fn optionalTypeSupportsBool(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    if (typeRefIsBool(value)) return .supported;
    return .{ .unsupported = .{ .unsupported_type = value } };
}

fn optionalTypeSupportsU256(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    if (typeRefIsU256(value)) return .supported;
    return .{ .unsupported = .{ .unsupported_type = value } };
}

fn optionalTypeSupportsInteger(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    if (compilerIntegerInfo(value) != null) return .supported;
    return .{ .unsupported = .{ .unsupported_type = value } };
}

fn optionalTypeMatchesIntegerInfo(
    ty: ?obligation.TypeRef,
    expected: CompilerIntegerInfo,
) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    const actual = compilerIntegerInfo(value) orelse
        return .{ .unsupported = .{ .unsupported_type = value } };
    if (actual.signed != expected.signed) {
        return .{ .unsupported = .mixed_arithmetic_signedness };
    }
    if (actual.width != expected.width) {
        return .{ .unsupported = .unsupported_arithmetic_width };
    }
    return .supported;
}

fn optionalTypeSupportsUnsignedInteger(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .unknown_signedness };
    const info = compilerIntegerInfo(value) orelse return .{ .unsupported = .unknown_signedness };
    if (info.signed) return .{ .unsupported = .mixed_signedness };
    return .supported;
}

fn optionalTypeSupportsSignedInteger(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .unknown_signedness };
    const info = compilerIntegerInfo(value) orelse return .{ .unsupported = .unknown_signedness };
    if (!info.signed) return .{ .unsupported = .mixed_signedness };
    return .supported;
}

fn optionalTypeSupportsU256Carrier(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    if (typeRefIsU256(value) or typeRefIsI256(value)) return .supported;
    return .{ .unsupported = .{ .unsupported_type = value } };
}

fn optionalTypeSupportsBoolOrInteger(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    if (typeRefIsBool(value) or compilerIntegerInfo(value) != null) return .supported;
    return .{ .unsupported = .{ .unsupported_type = value } };
}

const ArithmeticTypeSupport = union(enum) {
    supported: CompilerIntegerInfo,
    unsupported: SemanticUnsupportedReason,
};

fn arithmeticTypeSupport(ty: ?obligation.TypeRef) ArithmeticTypeSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    const info = compilerIntegerInfo(value) orelse return .{ .unsupported = .unknown_arithmetic_signedness };
    return .{ .supported = info };
}

fn typeRefIsBool(ty: obligation.TypeRef) bool {
    return switch (ty) {
        .spelling => |name| std.mem.eql(u8, name, "bool") or std.mem.eql(u8, name, "i1"),
        .compiler_type_id => |id| compilerTypeIdIsBuiltin(id, .bool),
    };
}

fn typeRefIsU256(ty: obligation.TypeRef) bool {
    return switch (ty) {
        .spelling => |name| std.mem.eql(u8, name, "u256") or std.mem.eql(u8, name, "uint256"),
        .compiler_type_id => |id| compilerTypeIdIsBuiltin(id, .u256),
    };
}

fn typeRefIsI256(ty: obligation.TypeRef) bool {
    return switch (ty) {
        .spelling => |name| std.mem.eql(u8, name, "i256") or std.mem.eql(u8, name, "int256"),
        .compiler_type_id => |id| compilerTypeIdIsBuiltin(id, .i256),
    };
}

const CompilerIntegerInfo = struct {
    width: u32,
    signed: bool,
};

fn compilerIntegerInfoEql(lhs: CompilerIntegerInfo, rhs: CompilerIntegerInfo) bool {
    return lhs.width == rhs.width and lhs.signed == rhs.signed;
}

fn compilerIntegerInfo(ty: obligation.TypeRef) ?CompilerIntegerInfo {
    return switch (ty) {
        .compiler_type_id => |id| blk: {
            const info = type_builtin.integerInfoByComptimeTypeId(id) orelse break :blk null;
            break :blk .{
                .width = info.width,
                .signed = info.signed,
            };
        },
        .spelling => |name| blk: {
            const canonical = type_builtin.integerInfoByName(name) orelse
                abiIntegerInfo(name) orelse break :blk null;
            break :blk .{
                .width = canonical.width,
                .signed = canonical.signed,
            };
        },
    };
}

fn abiIntegerInfo(name: []const u8) ?type_builtin.BuiltinIntegerInfo {
    const builtin_id: type_builtin.BuiltinTypeId = if (std.mem.eql(u8, name, "uint8"))
        .u8
    else if (std.mem.eql(u8, name, "uint16"))
        .u16
    else if (std.mem.eql(u8, name, "uint32"))
        .u32
    else if (std.mem.eql(u8, name, "uint64"))
        .u64
    else if (std.mem.eql(u8, name, "uint128"))
        .u128
    else if (std.mem.eql(u8, name, "uint160"))
        .u160
    else if (std.mem.eql(u8, name, "uint256"))
        .u256
    else if (std.mem.eql(u8, name, "int8"))
        .i8
    else if (std.mem.eql(u8, name, "int16"))
        .i16
    else if (std.mem.eql(u8, name, "int32"))
        .i32
    else if (std.mem.eql(u8, name, "int64"))
        .i64
    else if (std.mem.eql(u8, name, "int128"))
        .i128
    else if (std.mem.eql(u8, name, "int256"))
        .i256
    else
        return null;
    const spec = type_builtin.lookupBuiltinById(builtin_id);
    return type_builtin.integerInfoByComptimeTypeId(spec.comptime_type_id);
}

fn compilerTypeIdIsBuiltin(id: u32, builtin: type_builtin.BuiltinTypeId) bool {
    return id == type_builtin.lookupBuiltinById(builtin).comptime_type_id;
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
            try writer.writeAll(".variable (");
            try writeVarRef(writer, variable);
            try writer.writeByte(')');
        },
        .old => |id| {
            try writer.writeAll(".old ");
            try writer.print("{d}", .{id});
        },
        .result => try writer.writeAll(".result"),
        .place_read => |place| {
            try writer.writeAll(".placeRead ");
            try writePlaceRef(writer, place);
        },
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
            try writer.writeAll(", ty := ");
            try writeOptionalTypeRef(writer, binary.ty);
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
            try writeBinderRef(writer, quantified.binder);
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
        .effect_frame => |effect| {
            try writer.writeAll(".effectFrame ");
            try writeEffectFrameGoal(writer, effect);
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

fn writeEffectFrameGoal(writer: anytype, effect: obligation.EffectFrameGoal) !void {
    try writer.writeAll("{ relation := .");
    try writer.writeAll(effectFrameRelationName(effect.relation));
    try writer.writeAll(", declared := ");
    try writePlaceRefList(writer, effect.declared);
    try writer.writeAll(", actual := ");
    try writePlaceRefList(writer, effect.actual);
    try writer.writeAll(", evidence := ");
    try writeKeyDisjointEvidenceList(writer, effect.evidence);
    try writer.writeAll(" }");
}

fn writeKeyDisjointEvidenceList(writer: anytype, evidence: []const obligation.KeyDisjointEvidence) !void {
    if (evidence.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (evidence, 0..) |item, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeKeyDisjointEvidence(writer, item);
    }
    try writer.writeByte(']');
}

fn writeKeyDisjointEvidence(writer: anytype, evidence: obligation.KeyDisjointEvidence) !void {
    try writer.writeAll("{ kind := .");
    try writer.writeAll(keyDisjointEvidenceKindName(evidence.kind));
    try writer.writeAll(", assumptionId := ");
    try writer.print("{d}", .{evidence.assumption_id});
    try writer.writeAll(", lhs := ");
    try writeFreeVarId(writer, evidence.lhs);
    try writer.writeAll(", rhs := ");
    try writeFreeVarId(writer, evidence.rhs);
    try writer.writeAll(", read := ");
    try writePlaceRef(writer, evidence.read);
    try writer.writeAll(", write := ");
    try writePlaceRef(writer, evidence.write);
    try writer.writeAll(", keyIndex := ");
    try writer.print("{d}", .{evidence.key_index});
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

fn writePlaceRefList(writer: anytype, places: []const obligation.PlaceRef) !void {
    if (places.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (places, 0..) |place, index| {
        if (index != 0) try writer.writeAll(", ");
        try writePlaceRef(writer, place);
    }
    try writer.writeByte(']');
}

fn writePlaceRef(writer: anytype, place: obligation.PlaceRef) !void {
    try writer.writeAll("{ root := ");
    try writeLeanString(writer, place.root);
    try writer.writeAll(", region := .");
    try writer.writeAll(regionName(place.region));
    try writer.writeAll(", fields := ");
    try writeStringList(writer, place.fields);
    if (place.projection_domain != .mapping) {
        try writer.writeAll(", projectionDomain := .fixedIndex");
    }
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
        .parameter => |id| {
            try writer.writeAll(".parameter ");
            try writeFreeVarId(writer, id);
        },
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

fn writeFreeVarId(writer: anytype, id: obligation.FreeVarId) !void {
    try writer.writeAll("{ file_id := ");
    try writer.print("{d}", .{id.file_id});
    try writer.writeAll(", pattern_id := ");
    try writer.print("{d}", .{id.pattern_id});
    try writer.writeAll(" }");
}

fn writeVarRef(writer: anytype, variable: obligation.VarRef) !void {
    switch (variable) {
        .free => |free| {
            try writer.writeAll(".free ");
            try writeFreeVarRef(writer, free);
        },
        .bound => |bound| {
            try writer.writeAll(".bound ");
            try writeBoundVarRef(writer, bound);
        },
    }
}

fn writeFreeVarRef(writer: anytype, variable: obligation.FreeVarRef) !void {
    try writer.writeAll("{ id := ");
    try writeFreeVarId(writer, variable.id);
    try writer.writeAll(", name := ");
    try writeLeanString(writer, variable.name);
    try writer.writeAll(", ty := ");
    try writeOptionalTypeRef(writer, variable.ty);
    try writer.writeAll(", region := ");
    try writeOptionalRegionRef(writer, variable.region);
    try writer.writeAll(" }");
}

fn writeBoundVarRef(writer: anytype, variable: obligation.BoundVarRef) !void {
    try writer.writeAll("{ index := ");
    try writer.print("{d}", .{variable.index});
    try writer.writeAll(", name := ");
    try writeLeanString(writer, variable.name);
    try writer.writeAll(", ty := ");
    try writeOptionalTypeRef(writer, variable.ty);
    try writer.writeAll(", region := ");
    try writeOptionalRegionRef(writer, variable.region);
    try writer.writeAll(" }");
}

fn writeBinderRef(writer: anytype, variable: obligation.BinderRef) !void {
    try writer.writeAll("{ name := ");
    try writeLeanString(writer, variable.name);
    try writer.writeAll(", ty := ");
    try writeOptionalTypeRef(writer, variable.ty);
    try writer.writeAll(", region := ");
    try writeOptionalRegionRef(writer, variable.region);
    try writer.writeAll(" }");
}

fn writeOptionalRegionRef(writer: anytype, region: ?obligation.RegionRef) !void {
    if (region) |value| {
        try writer.writeAll("some .");
        try writer.writeAll(regionName(value));
    } else {
        try writer.writeAll("none");
    }
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

fn ownerEqual(lhs: obligation.Owner, rhs: obligation.Owner) bool {
    return switch (lhs) {
        .module => |left| switch (rhs) {
            .module => |right| std.mem.eql(u8, left, right),
            else => false,
        },
        .contract => |left| switch (rhs) {
            .contract => |right| std.mem.eql(u8, left, right),
            else => false,
        },
        .function => |left| switch (rhs) {
            .function => |right| std.mem.eql(u8, left.name, right.name),
            else => false,
        },
        .trait_method => |left| switch (rhs) {
            .trait_method => |right| std.mem.eql(u8, left.trait_name, right.trait_name) and
                std.mem.eql(u8, left.method_name, right.method_name),
            else => false,
        },
        .statement => |left| switch (rhs) {
            .statement => |right| std.mem.eql(u8, left.function_name, right.function_name) and left.ordinal == right.ordinal,
            else => false,
        },
        .backend => |left| switch (rhs) {
            .backend => |right| left.component == right.component and std.mem.eql(u8, left.name, right.name),
            else => false,
        },
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
        .slt => "slt",
        .sle => "sle",
        .sgt => "sgt",
        .sge => "sge",
        .add => "add",
        .sub => "sub",
        .mul => "mul",
        .pow => "pow",
        .shl => "shl",
        .shr => "shr",
        .div => "div",
        .mod => "mod_",
        .bit_and => "bitAnd",
        .bit_xor => "bitXor",
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
        .same_place_identity => "samePlaceIdentity",
        .conservation => "conservation",
    };
}

fn effectFrameRelationName(relation: obligation.EffectFrameRelation) []const u8 {
    return switch (relation) {
        .write_covered_by_modifies => "writeCoveredByModifies",
        .read_preserved_by_frame => "readPreservedByFrame",
        .read_preserved_by_key_evidence => "readPreservedByKeyEvidence",
        .lock_covers_write => "lockCoversWrite",
        .external_call_frame => "externalCallFrame",
    };
}

fn keyDisjointEvidenceKindName(kind: obligation.KeyDisjointEvidenceKind) []const u8 {
    return switch (kind) {
        .free_var_disequality => "freeVarDisequality",
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

test "Lean emitter writes locally nameless variable references" {
    const terms = [_]obligation.Term{
        .{ .variable = .{
            .free = .{
                .id = .{ .file_id = 0, .pattern_id = 7 },
                .name = "x",
                .ty = .{ .spelling = "u256" },
            },
        } },
        .{ .variable = .{
            .bound = .{
                .index = 0,
                .name = "i",
                .ty = .{ .spelling = "u256" },
            },
        } },
    };
    const set: obligation.ObligationSet = .{ .terms = &terms };

    var buffer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer buffer.deinit();
    try writeDataModule(&buffer.writer, set, .{ .namespace = "Ora.Generated.VarRefSmoke" });

    const rendered = buffer.written();
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".free { id := { file_id := 0, pattern_id := 7 }"));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".bound { index := 0"));
}

test "Lean emitter writes effect frame obligations" {
    const declared_keys = [_]obligation.PlaceKey{.{ .parameter = .{ .file_id = 0, .pattern_id = 0 } }};
    const declared = [_]obligation.PlaceRef{
        .{
            .root = "balances",
            .region = .storage,
            .projection_domain = .fixed_index,
            .keys = &declared_keys,
        },
    };
    const actual = declared;
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .{},
            .phase = .ora_mlir,
            .origin = .source,
            .kind = .{ .effect_frame = .{
                .relation = .write_covered_by_modifies,
                .declared = &declared,
                .actual = &actual,
            } },
        },
    };
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .{},
            .phase = .ora_mlir,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &.{1},
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations, .queries = &queries };

    var buffer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer buffer.deinit();
    try writeModule(&buffer.writer, set, .{ .namespace = "Ora.Generated.EffectFrameSmoke" });

    const rendered = buffer.written();
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".effectFrame { relation := .writeCoveredByModifies"));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "{ root := \"balances\", region := .storage"));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "projectionDomain := .fixedIndex"));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "keys := [.parameter { file_id := 0, pattern_id := 0 }]"));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "def emittedQuery_2 : Prop :="));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "obligationFollowsFromAssumptions emittedManifest"));
}

test "Lean emitter refuses origin-owned header facts" {
    const fact: obligation.LoopHeaderFactRow = .{
        .id = 0,
        .structural_key = "header-fact:0",
        .formula = .{ .origin_value = .{ .origin = .source } },
        .source_kind = .path_assume,
        .source = .{ .structural_key = "assume:0", .block_key = "block:0", .position = 1 },
        .target = .{ .structural_key = "loop:0", .block_key = "block:0", .position = 2 },
    };
    var buffer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer buffer.deinit();
    try std.testing.expectError(error.UnsupportedOriginValue, writeLoopHeaderFacts(&buffer.writer, &.{fact}));
}

test "Lean emitter pins indexed-region read timeline data" {
    const events = [_]obligation.LoopTimelineEvent{
        .{
            .id = 0,
            .structural_key = "body/check",
            .payload = .{ .safety_checkpoint = .{
                .formulas = &.{.{ .formula = .{ .term = 2 } }},
                .dynamic_index_bounds = true,
            } },
        },
        .{
            .id = 1,
            .structural_key = "body/read",
            .payload = .{ .indexed_region_read = .{
                .result = .{ .file_id = 7, .pattern_id = 8 },
                .region = .calldata,
                .owner = .{ .file_id = 7, .pattern_id = 3 },
                .index = .{ .formula = .{ .term = 4 } },
                .element_ty = .{ .compiler_type_id = 6 },
                .checkpoint_structural_key = "body/check",
            } },
        },
    };
    var buffer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer buffer.deinit();
    try writeLoopTimeline(&buffer.writer, &events);
    try std.testing.expect(std.mem.containsAtLeast(
        u8,
        buffer.written(),
        1,
        ".indexedRegionRead { result := { file_id := 7, pattern_id := 8 }, " ++
            "region := .calldata, owner := { file_id := 7, pattern_id := 3 }, " ++
            "index := { formula := .term 4, readResults := [] }, " ++
            "elementTy := .compilerTypeId 6, checkpointStructuralKey := \"body/check\" }",
    ));
}

test "LoopVC digest covers every certificate-authority component class" {
    const int_ty: obligation.TypeRef = .{ .spelling = "u256" };
    const loop_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 1 };
    const read_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 2 };
    const region_read_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 3 };
    const region_owner_id: obligation.FreeVarId = .{ .file_id = 1, .pattern_id = 4 };
    const place_keys = [_]obligation.PlaceKey{.{ .parameter = read_id }};
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = loop_id, .name = "i", .ty = int_ty } } },
        .{ .int_lit = .{ .value = "0", .ty = int_ty } },
        .{ .int_lit = .{ .value = "1", .ty = int_ty } },
        .{ .int_lit = .{ .value = "3", .ty = int_ty } },
        .{ .binary = .{ .op = .le, .lhs = 0, .rhs = 3, .ty = int_ty } },
        .{ .binary = .{ .op = .lt, .lhs = 0, .rhs = 3, .ty = int_ty } },
        .{ .bool_lit = true },
        .{ .binary = .{ .op = .add, .lhs = 0, .rhs = 2, .ty = int_ty } },
        .{ .unary = .{ .op = .not, .operand = 5 } },
        .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 3, .ty = int_ty } },
        .{ .variable = .{ .free = .{ .id = read_id, .name = "read_result", .ty = int_ty } } },
        .{ .place_read = .{
            .root = "values",
            .region = .storage,
            .projection_domain = .fixed_index,
            .keys = &place_keys,
        } },
    };
    const variables = [_]obligation.LoopVariable{.{ .index = 0, .id = loop_id, .name = "i", .ty = int_ty }};
    const header_values = [_]obligation.LoopHeaderFactValue{.{ .id = loop_id, .ty = int_ty }};
    const header_facts = [_]obligation.LoopHeaderFactRow{.{
        .id = 0,
        .structural_key = "header-fact:0",
        .formula = .{ .term = 6 },
        .source_kind = .function_requires,
        .source = .{ .structural_key = "requires:0", .block_key = "block:0", .position = 1 },
        .target = .{ .structural_key = "loop:0", .block_key = "block:0", .position = 3 },
        .stable_values = &header_values,
        .assumption_id = 50,
    }};
    const backedge_bindings = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "body/straight-line-yield/operand:0",
        .value = .{ .term = 7 },
    }};
    const exit_bindings = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop-state:0",
        .value = .{ .term = 0 },
    }};
    const path_expr = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 5 } }};
    const exit_path_expr = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 8 } }};
    const safety_expr = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 6 } }};
    const read_refs = [_]obligation.FreeVarId{read_id};
    const index_projection = [_]obligation.LoopStorageProjection{.{
        .fixed_index = .{
            .index = .{ .value = .{ .formula = .{ .term = 2 } }, .ty = int_ty },
            .extent = 4,
        },
    }};
    const location: obligation.LoopStorageLocationExpr = .{
        .root = "values",
        .projections = &index_projection,
        .value_ty = int_ty,
    };
    const backedge_timeline = [_]obligation.LoopTimelineEvent{
        .{ .id = 0, .structural_key = "body/path", .payload = .{ .path_assumption = .{ .formulas = &path_expr } } },
        .{ .id = 1, .structural_key = "body/read", .payload = .{ .storage_read = .{ .result = read_id, .location = location } } },
        .{ .id = 2, .structural_key = "body/write", .payload = .{ .storage_write = .{
            .location = location,
            .value = .{ .formula = .{ .term = 10 }, .read_results = &read_refs },
        } } },
        .{ .id = 3, .structural_key = "body/ora.assert:0", .payload = .{ .safety_checkpoint = .{
            .formulas = &safety_expr,
            .dynamic_index_bounds = true,
        } } },
        .{ .id = 4, .structural_key = "body/region-read", .payload = .{ .indexed_region_read = .{
            .result = region_read_id,
            .region = .calldata,
            .owner = region_owner_id,
            .index = .{ .formula = .{ .term = 0 } },
            .element_ty = int_ty,
            .checkpoint_structural_key = "body/ora.assert:0",
        } } },
    };
    const exit_timeline = [_]obligation.LoopTimelineEvent{
        .{ .id = 0, .structural_key = "guard/false", .payload = .{ .path_assumption = .{ .formulas = &exit_path_expr } } },
    };
    const edges = [_]obligation.LoopEdgeRow{
        .{ .id = 0, .structural_key = "body/straight-line-yield", .timeline = &backedge_timeline, .next_state = &backedge_bindings, .target = .backedge },
        .{ .id = 1, .structural_key = "guard/false-successor", .timeline = &exit_timeline, .next_state = &exit_bindings, .target = .successor },
    };
    const summary: obligation.LoopSummaryRow = .{
        .id = 1,
        .owner = .{ .statement = .{ .function_name = "count", .ordinal = 0 } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 0,
        .loop_kind = .scf_while,
        .variables = &variables,
        .init_formulas = &.{.{ .term = 1 }},
        .invariant_formulas = &.{.{ .term = 4 }},
        .header_facts = &header_facts,
        .edges = &edges,
        .projection_completeness = .complete,
        .post_formulas = &.{.{ .term = 9 }},
        .query_ids = .{ .induction = &.{60} },
    };
    const assumptions = [_]obligation.Assumption{.{
        .id = 50,
        .owner = .{ .function = .{ .name = "count" } },
        .source = .generated(),
        .phase = .sema,
        .origin = .source,
        .kind = .requires,
        .formula = .{ .term = 6 },
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 60,
        .owner = .{ .function = .{ .name = "count" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .loop_induction,
        .assumption_ids = &.{50},
        .loop_summary_id = 1,
    }};
    const summaries = [_]obligation.LoopSummaryRow{summary};
    const set: obligation.ObligationSet = .{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &summaries,
    };
    const baseline = try loopVcSha256(set, 1);
    const old_domain_digest = try loopVcSha256InDomain(set, 1, "ora-loop-vc-v2\n");
    try std.testing.expect(!std.mem.eql(u8, &baseline, &old_domain_digest));

    const HeaderFactMutation = enum {
        id,
        structural_key,
        formula,
        source_kind,
        source_structural_key,
        source_block_key,
        source_position,
        target_structural_key,
        target_block_key,
        target_position,
        stable_value_id,
        stable_value_type,
        assumption_id,
    };
    inline for (std.meta.tags(HeaderFactMutation)) |mutation| {
        var changed_fact = header_facts[0];
        var changed_values = header_values;
        switch (mutation) {
            .id => changed_fact.id = 1,
            .structural_key => changed_fact.structural_key = "header-fact:changed",
            .formula => changed_fact.formula = .{ .term = 5 },
            .source_kind => changed_fact.source_kind = .path_assume,
            .source_structural_key => changed_fact.source.structural_key = "requires:changed",
            .source_block_key => changed_fact.source.block_key = "block:changed",
            .source_position => changed_fact.source.position = 2,
            .target_structural_key => changed_fact.target.structural_key = "loop:changed",
            .target_block_key => changed_fact.target.block_key = "block:changed",
            .target_position => changed_fact.target.position = 4,
            .stable_value_id => changed_values[0].id.pattern_id = 9,
            .stable_value_type => changed_values[0].ty = .{ .spelling = "u128" },
            .assumption_id => changed_fact.assumption_id = 51,
        }
        changed_fact.stable_values = &changed_values;
        const changed_facts = [_]obligation.LoopHeaderFactRow{changed_fact};
        var changed_summary = summary;
        changed_summary.header_facts = &changed_facts;
        const changed_summaries = [_]obligation.LoopSummaryRow{changed_summary};
        const changed_digest = try loopVcSha256(.{
            .terms = &terms,
            .assumptions = &assumptions,
            .queries = &queries,
            .loop_summaries = &changed_summaries,
        }, 1);
        try std.testing.expect(!std.mem.eql(u8, &baseline, &changed_digest));
    }

    var preheader_summary = summary;
    preheader_summary.preheader_timeline = &.{backedge_timeline[0]};
    preheader_summary.preheader_projection_complete = true;
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &.{preheader_summary},
    }, 1))));

    const DigestCheck = struct {
        fn differs(
            expected: [32]u8,
            term_rows: []const obligation.Term,
            assumption_rows: []const obligation.Assumption,
            query_rows: []const obligation.VerificationQuery,
            changed: obligation.LoopSummaryRow,
        ) !void {
            const rows = [_]obligation.LoopSummaryRow{changed};
            const actual = try loopVcSha256(.{
                .terms = term_rows,
                .assumptions = assumption_rows,
                .queries = query_rows,
                .loop_summaries = &rows,
            }, changed.id);
            try std.testing.expect(!std.mem.eql(u8, &expected, &actual));
        }
    };

    const IndexedRegionMutation = enum {
        result,
        region,
        owner,
        index_formula,
        index_read_results,
        element_type,
        checkpoint_structural_key,
    };
    inline for (std.meta.tags(IndexedRegionMutation)) |mutation| {
        var changed_timeline = backedge_timeline;
        var changed_read = changed_timeline[4].payload.indexed_region_read;
        const changed_read_results = [_]obligation.FreeVarId{read_id};
        switch (mutation) {
            .result => changed_read.result.pattern_id = 30,
            .region => changed_read.region = .memory,
            .owner => changed_read.owner.pattern_id = 40,
            .index_formula => changed_read.index.formula = .{ .term = 1 },
            .index_read_results => changed_read.index.read_results = &changed_read_results,
            .element_type => changed_read.element_ty = .{ .spelling = "u128" },
            .checkpoint_structural_key => changed_read.checkpoint_structural_key = "body/ora.assert:changed",
        }
        changed_timeline[4].payload.indexed_region_read = changed_read;
        var changed_edges = edges;
        changed_edges[0].timeline = &changed_timeline;
        var changed_summary = summary;
        changed_summary.edges = &changed_edges;
        try DigestCheck.differs(baseline, &terms, &assumptions, &queries, changed_summary);
    }

    var access_timeline = backedge_timeline;
    access_timeline[3].payload.safety_checkpoint.dynamic_index_bounds = false;
    var access_edges = edges;
    access_edges[0].timeline = &access_timeline;
    var access_summary = summary;
    access_summary.edges = &access_edges;
    try DigestCheck.differs(baseline, &terms, &assumptions, &queries, access_summary);

    var projection_domain_terms = terms;
    projection_domain_terms[11].place_read.projection_domain = .mapping;
    try DigestCheck.differs(
        baseline,
        &projection_domain_terms,
        &assumptions,
        &queries,
        summary,
    );

    var timeline_mutation = backedge_timeline;
    timeline_mutation[1].structural_key = "body/read:changed";
    var timeline_edges = edges;
    timeline_edges[0].timeline = &timeline_mutation;
    var timeline_summary = summary;
    timeline_summary.edges = &timeline_edges;
    try DigestCheck.differs(baseline, &terms, &assumptions, &queries, timeline_summary);

    timeline_mutation = backedge_timeline;
    timeline_mutation[1].id = 7;
    timeline_edges = edges;
    timeline_edges[0].timeline = &timeline_mutation;
    timeline_summary = summary;
    timeline_summary.edges = &timeline_edges;
    try DigestCheck.differs(baseline, &terms, &assumptions, &queries, timeline_summary);

    timeline_mutation = backedge_timeline;
    timeline_mutation[1].payload = .{ .storage_read = .{
        .result = .{ .file_id = 1, .pattern_id = 3 },
        .location = location,
    } };
    timeline_edges = edges;
    timeline_edges[0].timeline = &timeline_mutation;
    timeline_summary = summary;
    timeline_summary.edges = &timeline_edges;
    try DigestCheck.differs(baseline, &terms, &assumptions, &queries, timeline_summary);

    var changed_location = location;
    changed_location.root = "other_values";
    timeline_mutation = backedge_timeline;
    timeline_mutation[1].payload = .{ .storage_read = .{ .result = read_id, .location = changed_location } };
    timeline_edges = edges;
    timeline_edges[0].timeline = &timeline_mutation;
    timeline_summary = summary;
    timeline_summary.edges = &timeline_edges;
    try DigestCheck.differs(baseline, &terms, &assumptions, &queries, timeline_summary);

    var changed_extent_projection = index_projection;
    changed_extent_projection[0].fixed_index.extent = 5;
    changed_location = location;
    changed_location.projections = &changed_extent_projection;
    timeline_mutation = backedge_timeline;
    timeline_mutation[1].payload = .{ .storage_read = .{
        .result = read_id,
        .location = changed_location,
    } };
    timeline_edges = edges;
    timeline_edges[0].timeline = &timeline_mutation;
    timeline_summary = summary;
    timeline_summary.edges = &timeline_edges;
    try DigestCheck.differs(baseline, &terms, &assumptions, &queries, timeline_summary);

    var changed_projection = index_projection;
    changed_projection[0] = .{ .mapping_key = .{
        .value = .{ .formula = .{ .term = 2 } },
        .ty = .{ .spelling = "u128" },
    } };
    changed_location = location;
    changed_location.projections = &changed_projection;
    changed_location.value_ty = .{ .spelling = "u128" };
    timeline_mutation = backedge_timeline;
    timeline_mutation[1].payload = .{ .storage_read = .{ .result = read_id, .location = changed_location } };
    timeline_edges = edges;
    timeline_edges[0].timeline = &timeline_mutation;
    timeline_summary = summary;
    timeline_summary.edges = &timeline_edges;
    try DigestCheck.differs(baseline, &terms, &assumptions, &queries, timeline_summary);

    const no_read_refs = [_]obligation.FreeVarId{};
    timeline_mutation = backedge_timeline;
    timeline_mutation[2].payload = .{ .storage_write = .{
        .location = location,
        .value = .{ .formula = .{ .term = 2 }, .read_results = &no_read_refs },
    } };
    timeline_edges = edges;
    timeline_edges[0].timeline = &timeline_mutation;
    timeline_summary = summary;
    timeline_summary.edges = &timeline_edges;
    try DigestCheck.differs(baseline, &terms, &assumptions, &queries, timeline_summary);

    var read_binding = backedge_bindings;
    read_binding[0].read_results = &read_refs;
    timeline_edges = edges;
    timeline_edges[0].next_state = &read_binding;
    timeline_summary = summary;
    timeline_summary.edges = &timeline_edges;
    try DigestCheck.differs(baseline, &terms, &assumptions, &queries, timeline_summary);

    var changed_summary = summary;
    changed_summary.invariant_formulas = &.{.{ .term = 6 }};
    var changed_summaries = [_]obligation.LoopSummaryRow{changed_summary};
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    changed_summary = summary;
    changed_summary.init_formulas = &.{.{ .term = 2 }};
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    changed_summary = summary;
    changed_summary.post_formulas = &.{.{ .term = 6 }};
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    var changed_timeline = backedge_timeline;
    changed_timeline[0].payload.path_assumption.formulas = &.{.{ .formula = .{ .term = 6 } }};
    var changed_edges = edges;
    changed_edges[0].timeline = &changed_timeline;
    changed_summary = summary;
    changed_summary.edges = &changed_edges;
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    changed_timeline = backedge_timeline;
    changed_timeline[3].payload.safety_checkpoint.formulas = &.{.{ .formula = .{ .term = 4 } }};
    changed_edges = edges;
    changed_edges[0].timeline = &changed_timeline;
    changed_summary = summary;
    changed_summary.edges = &changed_edges;
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    changed_timeline = backedge_timeline;
    changed_timeline[3].structural_key = "body/ora.assert:changed";
    changed_edges = edges;
    changed_edges[0].timeline = &changed_timeline;
    changed_summary = summary;
    changed_summary.edges = &changed_edges;
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    var changed_bindings = backedge_bindings;
    changed_bindings[0].value = .{ .term = 2 };
    changed_edges = edges;
    changed_edges[0].next_state = &changed_bindings;
    changed_summary = summary;
    changed_summary.edges = &changed_edges;
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    changed_edges = edges;
    changed_edges[0].target = .successor;
    changed_summary = summary;
    changed_summary.edges = &changed_edges;
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    const footprint = [_]obligation.PlaceRef{.{ .root = "values", .region = .storage }};
    const dependency: obligation.LoopSummaryDependency = .{
        .id = 0,
        .structural_key = "body/op:4/scf.while",
        .summary_id = 2,
        .loop_vc_sha256 = [_]u8{9} ** 32,
        .write_footprint = &footprint,
        .write_footprint_complete = true,
        .target = .successor,
    };
    const dependency_event: obligation.LoopTimelineEvent = .{
        .id = backedge_timeline.len,
        .structural_key = dependency.structural_key,
        .payload = .{ .summary_dependency = dependency },
    };
    const dependency_timeline = backedge_timeline ++ [_]obligation.LoopTimelineEvent{dependency_event};
    changed_edges = edges;
    changed_edges[0].timeline = &dependency_timeline;
    changed_summary = summary;
    changed_summary.edges = &changed_edges;
    changed_summaries[0] = changed_summary;
    const dependency_baseline = try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1);
    try std.testing.expect(!std.mem.eql(u8, &baseline, &dependency_baseline));

    var changed_dependency = dependency;
    changed_dependency.write_footprint_complete = false;
    var changed_dependency_timeline = dependency_timeline;
    changed_dependency_timeline[backedge_timeline.len].payload.summary_dependency = changed_dependency;
    changed_edges[0].timeline = &changed_dependency_timeline;
    changed_summary.edges = &changed_edges;
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &dependency_baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    const other_footprint = [_]obligation.PlaceRef{.{ .root = "other", .region = .storage }};
    changed_dependency = dependency;
    changed_dependency.write_footprint = &other_footprint;
    changed_dependency_timeline = dependency_timeline;
    changed_dependency_timeline[backedge_timeline.len].payload.summary_dependency = changed_dependency;
    changed_edges[0].timeline = &changed_dependency_timeline;
    changed_summary.edges = &changed_edges;
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &dependency_baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    changed_edges = edges;
    changed_edges[1].target = .function_return;
    changed_edges[1].target_formulas = &.{.{ .term = 9 }};
    changed_summary = summary;
    changed_summary.edges = &changed_edges;
    changed_summaries[0] = changed_summary;
    const return_target_baseline = try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1);
    changed_edges[1].target_formulas = &.{.{ .term = 6 }};
    try std.testing.expect(!std.mem.eql(u8, &return_target_baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    var changed_variables = variables;
    changed_variables[0].ty = .{ .spelling = "i256" };
    changed_summary = summary;
    changed_summary.variables = &changed_variables;
    changed_summaries[0] = changed_summary;
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &changed_summaries,
    }, 1))));

    var changed_assumptions = assumptions;
    changed_assumptions[0].formula = .{ .term = 4 };
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &terms,
        .assumptions = &changed_assumptions,
        .queries = &queries,
        .loop_summaries = &summaries,
    }, 1))));

    var changed_terms = terms;
    changed_terms[3] = .{ .int_lit = .{ .value = "4", .ty = int_ty } };
    try std.testing.expect(!std.mem.eql(u8, &baseline, &(try loopVcSha256(.{
        .terms = &changed_terms,
        .assumptions = &assumptions,
        .queries = &queries,
        .loop_summaries = &summaries,
    }, 1))));
}
