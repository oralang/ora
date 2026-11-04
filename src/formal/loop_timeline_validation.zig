//! Shape validation for canonical loop timelines.
//!
//! `Validator` is instantiated with the obligation schema namespace.  The
//! explicit namespace contract keeps this module independent of the schema
//! file without introducing a second set of timeline types or an `anytype`
//! adapter seam.

const std = @import("std");

pub fn Validator(comptime Schema: type) type {
    return struct {
        const Reason = Schema.LoopUnsupportedReason;

        const FixedIndexBoundsMatch = enum(u8) {
            matched,
            formula_shape_mismatch,
            index_identity_mismatch,
            type_mismatch,
            extent_literal_mismatch,

            fn rank(self: FixedIndexBoundsMatch) u8 {
                return @backingInt(self);
            }

            fn reason(self: FixedIndexBoundsMatch) Reason {
                return switch (self) {
                    .matched => unreachable,
                    .formula_shape_mismatch => .loop_fixed_index_bounds_formula_shape_mismatch,
                    .index_identity_mismatch => .loop_fixed_index_bounds_index_identity_mismatch,
                    .type_mismatch => .loop_fixed_index_bounds_type_mismatch,
                    .extent_literal_mismatch => .loop_fixed_index_bounds_extent_literal_mismatch,
                };
            }
        };

        fn typeRefEql(lhs: Schema.TypeRef, rhs: Schema.TypeRef) bool {
            return switch (lhs) {
                .compiler_type_id => |id| switch (rhs) {
                    .compiler_type_id => |other| id == other,
                    .spelling => false,
                },
                .spelling => |name| switch (rhs) {
                    .spelling => |other| std.mem.eql(u8, name, other),
                    .compiler_type_id => false,
                },
            };
        }

        fn optionalTypeRefsEql(lhs: ?Schema.TypeRef, rhs: ?Schema.TypeRef) bool {
            if (lhs == null or rhs == null) return lhs == null and rhs == null;
            return typeRefEql(lhs.?, rhs.?);
        }

        fn freeVarIdEql(lhs: Schema.FreeVarId, rhs: Schema.FreeVarId) bool {
            return lhs.file_id == rhs.file_id and lhs.pattern_id == rhs.pattern_id;
        }

        fn timelineContainsReadIdentity(events: []const Schema.LoopTimelineEvent, id: Schema.FreeVarId) bool {
            for (events) |event| for (0..Schema.loopTimelineReadCount(event.payload)) |result_index| if (Schema.loopTimelineReadId(event.payload, result_index)) |read_id| {
                if (freeVarIdEql(read_id, id)) return true;
            };
            return false;
        }

        fn timelineReadReferencesReason(
            events: []const Schema.LoopTimelineEvent,
            event_index: usize,
            references: []const Schema.FreeVarId,
        ) ?Reason {
            for (references, 0..) |reference, index| {
                for (references[0..index]) |previous| {
                    if (freeVarIdEql(previous, reference)) return .loop_storage_read_identity_duplicate;
                }
                var found_before = false;
                for (events[0..event_index]) |event| {
                    if (timelineContainsReadIdentity(&.{event}, reference)) {
                        found_before = true;
                        break;
                    }
                }
                if (found_before) continue;
                if (timelineContainsReadIdentity(events[event_index..], reference)) {
                    return .loop_storage_read_identity_forward;
                }
                return .loop_storage_read_identity_foreign;
            }
            return null;
        }

        fn fixedIndexExtentValid(extent: u64) bool {
            return extent != 0;
        }

        fn storageLocationReason(
            events: []const Schema.LoopTimelineEvent,
            event_index: usize,
            location: Schema.LoopStorageLocationExpr,
        ) ?Reason {
            if (location.root.len == 0) return .loop_storage_root_missing;
            if (!Schema.typeRefExplicit(location.value_ty)) return .loop_storage_type_invalid;
            for (location.projections) |projection| switch (projection) {
                .field => |field| if (field.len == 0) return .loop_storage_projection_invalid,
                .fixed_index => |fixed| {
                    if (!fixedIndexExtentValid(fixed.extent)) return .loop_fixed_index_extent_invalid;
                    if (!Schema.typeRefExplicit(fixed.index.ty)) return .loop_storage_type_invalid;
                    if (timelineReadReferencesReason(events, event_index, fixed.index.value.read_results)) |reason| {
                        return reason;
                    }
                },
                .mapping_key => |key| {
                    if (!Schema.typeRefExplicit(key.ty)) return .loop_storage_type_invalid;
                    if (timelineReadReferencesReason(events, event_index, key.value.read_results)) |reason| {
                        return reason;
                    }
                },
            };
            return null;
        }

        fn formulaEventReason(
            events: []const Schema.LoopTimelineEvent,
            event_index: usize,
            formulas: []const Schema.LoopFormulaExpr,
            missing_reason: Reason,
        ) ?Reason {
            if (formulas.len == 0) return missing_reason;
            for (formulas) |formula| {
                if (timelineReadReferencesReason(events, event_index, formula.read_results)) |reason| return reason;
            }
            return null;
        }

        fn indexedRegionCheckpointReason(
            events: []const Schema.LoopTimelineEvent,
            event_index: usize,
            structural_key: []const u8,
        ) ?Reason {
            if (structural_key.len == 0) return .loop_indexed_region_checkpoint_missing;
            for (events[0..event_index]) |event| {
                if (!std.mem.eql(u8, event.structural_key, structural_key)) continue;
                return switch (event.payload) {
                    .safety_checkpoint => |check| if (check.dynamic_index_bounds)
                        null
                    else
                        .loop_indexed_region_checkpoint_invalid,
                    else => .loop_indexed_region_checkpoint_invalid,
                };
            }
            for (events[event_index + 1 ..]) |event| {
                if (std.mem.eql(u8, event.structural_key, structural_key))
                    return .loop_indexed_region_checkpoint_after_read;
            }
            return .loop_indexed_region_checkpoint_missing;
        }

        fn readIdentityDuplicate(
            variables: []const Schema.LoopVariable,
            context: []const Schema.LoopVariable,
            events: []const Schema.LoopTimelineEvent,
            event_index: usize,
            result: Schema.FreeVarId,
        ) bool {
            for (context) |variable| if (variable.id) |id|
                if (freeVarIdEql(id, result)) return true;
            for (variables) |variable| if (variable.id) |id|
                if (freeVarIdEql(id, result)) return true;
            return timelineContainsReadIdentity(events[0..event_index], result);
        }

        pub fn timelineReason(
            summary: Schema.LoopSummaryRow,
            events: []const Schema.LoopTimelineEvent,
            require_path: bool,
            allow_dependencies: bool,
        ) ?Reason {
            return scopedTimelineReason(summary.variables, summary.context_variables, events, require_path, allow_dependencies);
        }

        pub fn scopedTimelineReason(
            variables: []const Schema.LoopVariable,
            context: []const Schema.LoopVariable,
            events: []const Schema.LoopTimelineEvent,
            require_path: bool,
            allow_dependencies: bool,
        ) ?Reason {
            var has_path = false;
            var dependency_index: u32 = 0;
            for (events, 0..) |event, event_index| {
                if (event.id != event_index) return .loop_timeline_event_order_invalid;
                if (event.structural_key.len == 0) return .loop_timeline_event_identity_missing;
                for (events[0..event_index]) |previous| {
                    if (std.mem.eql(u8, previous.structural_key, event.structural_key)) {
                        return .loop_timeline_event_identity_duplicate;
                    }
                }
                switch (event.payload) {
                    .path_assumption => |path| {
                        has_path = true;
                        if (formulaEventReason(events, event_index, path.formulas, .loop_timeline_path_missing)) |reason| {
                            return reason;
                        }
                    },
                    .safety_checkpoint => |checkpoint| {
                        if (formulaEventReason(events, event_index, checkpoint.formulas, .loop_body_safety_missing)) |reason| {
                            return reason;
                        }
                    },
                    .storage_read => |read| {
                        if (storageLocationReason(events, event_index, read.location)) |reason| return reason;
                        if (readIdentityDuplicate(variables, context, events, event_index, read.result))
                            return .loop_storage_read_identity_duplicate;
                    },
                    .indexed_region_read => |read| {
                        if (!Schema.typeRefExplicit(read.element_ty)) return .loop_storage_type_invalid;
                        if (timelineReadReferencesReason(events, event_index, read.index.read_results)) |reason|
                            return reason;
                        if (readIdentityDuplicate(variables, context, events, event_index, read.result))
                            return .loop_storage_read_identity_duplicate;
                        if (indexedRegionCheckpointReason(
                            events,
                            event_index,
                            read.checkpoint_structural_key,
                        )) |reason| return reason;
                    },
                    .storage_write => |write| {
                        if (storageLocationReason(events, event_index, write.location)) |reason| return reason;
                        if (timelineReadReferencesReason(events, event_index, write.value.read_results)) |reason| {
                            return reason;
                        }
                    },
                    .call => |invocation| {
                        if (std.mem.allEqual(u8, &invocation.callee_vc_sha256, 0)) return .loop_call_digest_missing;
                        for (invocation.arguments) |argument| {
                            if (!Schema.typeRefExplicit(argument.ty)) return .loop_call_binding_invalid;
                            if (timelineReadReferencesReason(events, event_index, argument.value.read_results)) |reason| return reason;
                        }
                        for (invocation.results, 0..) |result, index| {
                            const id = result.id orelse return .loop_call_binding_invalid;
                            if (result.index != index or !Schema.typeRefExplicit(result.ty)) return .loop_call_binding_invalid;
                            if (readIdentityDuplicate(variables, context, events, event_index, id)) return .loop_storage_read_identity_duplicate;
                            for (invocation.results[0..index]) |previous| {
                                if (freeVarIdEql(previous.id.?, id)) return .loop_storage_read_identity_duplicate;
                            }
                        }
                    },
                    .summary_dependency => |dependency| {
                        if (!allow_dependencies) return .loop_preheader_storage_relation_unsupported;
                        if (dependency.id != dependency_index) return .loop_summary_dependency_order_invalid;
                        dependency_index += 1;
                        if (!dependency.targetSupported()) return .loop_summary_dependency_target_unsupported;
                        if (!dependency.write_footprint_complete) return .loop_storage_child_footprint_missing;
                        if (dependency.result_bindings) |bindings| for (bindings, 0..) |binding, index| {
                            const id = binding.id orelse return .loop_call_binding_invalid;
                            if (!Schema.typeRefExplicit(binding.ty)) return .loop_call_binding_invalid;
                            if (readIdentityDuplicate(variables, context, events, event_index, id)) return .loop_storage_read_identity_duplicate;
                            for (bindings[0..index]) |previous| {
                                if (freeVarIdEql(previous.id.?, id)) return .loop_storage_read_identity_duplicate;
                                if (previous.index == binding.index) return .loop_call_binding_invalid;
                            }
                        };
                        for (dependency.write_footprint) |place| {
                            if (place.region != .storage or place.root.len == 0) {
                                return .loop_storage_projection_invalid;
                            }
                        }
                    },
                }
            }
            if (require_path and !has_path) return .loop_timeline_path_missing;
            return null;
        }

        pub fn edgeReason(summary: Schema.LoopSummaryRow, edge: Schema.LoopEdgeRow) ?Reason {
            if (timelineReason(summary, edge.timeline, true, true)) |reason| return reason;
            for (edge.next_state) |binding| {
                if (timelineReadReferencesReason(edge.timeline, edge.timeline.len, binding.read_results)) |reason| {
                    return reason;
                }
            }
            if (edge.return_value) |value| {
                if (edge.target != .function_return and edge.target != .error_return) return .loop_call_return_invalid;
                if (returnReason(edge.timeline, value)) |reason| return reason;
                if (value == .direct and (value.direct.outcome == .success) != (edge.target == .function_return)) return .loop_call_return_invalid;
            }
            return null;
        }

        pub fn returnReason(events: []const Schema.LoopTimelineEvent, value: Schema.CallReturnExpr) ?Reason {
            switch (value) {
                .direct => |direct| for (direct.values) |item| {
                    if (!Schema.typeRefExplicit(item.ty)) return .loop_call_return_invalid;
                    if (timelineReadReferencesReason(events, events.len, item.value.read_results)) |reason| return reason;
                },
                .loop_exit => {
                    if (events.len == 0) return .loop_call_return_invalid;
                    const dependency = switch (events[events.len - 1].payload) {
                        .summary_dependency => |dependency| dependency,
                        else => return .loop_call_return_invalid,
                    };
                    if (dependency.target != .function_return and dependency.target != .error_return) return .loop_call_return_invalid;
                },
            }
            return null;
        }

        fn fixedIndexBoundsFormulaMatch(
            terms: []const Schema.Term,
            formula: Schema.FormulaRef,
            projection: Schema.LoopFixedIndexProjection,
        ) FixedIndexBoundsMatch {
            const formula_id = switch (formula) {
                .term => |id| id,
                .origin_value => return .formula_shape_mismatch,
            };
            const index_id = switch (projection.index.value.formula) {
                .term => |id| id,
                .origin_value => return .index_identity_mismatch,
            };
            if (formula_id >= terms.len) return .formula_shape_mismatch;
            const binary = switch (terms[formula_id]) {
                .binary => |binary| binary,
                else => return .formula_shape_mismatch,
            };
            if (binary.op != .lt) return .formula_shape_mismatch;
            if (binary.lhs != index_id) return .index_identity_mismatch;
            if (!optionalTypeRefsEql(binary.ty, projection.index.ty)) return .type_mismatch;
            if (binary.rhs >= terms.len) return .formula_shape_mismatch;
            const literal = switch (terms[binary.rhs]) {
                .int_lit => |literal| literal,
                else => return .formula_shape_mismatch,
            };
            if (!optionalTypeRefsEql(literal.ty, projection.index.ty)) return .type_mismatch;
            const extent_decimal_capacity = comptime std.fmt.count("{d}", .{
                std.math.maxInt(@TypeOf(projection.extent)),
            });
            var extent_buffer: [extent_decimal_capacity]u8 = undefined;
            // Capacity is derived from the schema's exact extent carrier, so
            // formatting cannot fail; a future carrier change is compile-time
            // visible instead of degrading to a semantic mismatch.
            const canonical_extent = std.fmt.bufPrint(&extent_buffer, "{d}", .{projection.extent}) catch unreachable;
            if (!std.mem.eql(u8, literal.value, canonical_extent)) return .extent_literal_mismatch;
            return .matched;
        }

        fn fixedIndexBoundsLocationReason(
            terms: []const Schema.Term,
            prior: []const Schema.LoopTimelineEvent,
            location: Schema.LoopStorageLocationExpr,
        ) ?Reason {
            for (location.projections) |projection| switch (projection) {
                .fixed_index => |fixed| {
                    if (!fixedIndexExtentValid(fixed.extent)) return .loop_fixed_index_extent_invalid;
                    var best_mismatch: ?FixedIndexBoundsMatch = null;
                    for (prior) |event| switch (event.payload) {
                        .safety_checkpoint => |checkpoint| for (checkpoint.formulas) |candidate| {
                            const result = fixedIndexBoundsFormulaMatch(terms, candidate.formula, fixed);
                            if (result == .matched) return null;
                            if (best_mismatch == null or result.rank() > best_mismatch.?.rank()) {
                                best_mismatch = result;
                            }
                        },
                        else => {},
                    };
                    if (best_mismatch) |mismatch| return mismatch.reason();
                    return .loop_fixed_index_bounds_checkpoint_missing;
                },
                .field, .mapping_key => {},
            };
            return null;
        }

        pub fn fixedIndexBoundsReason(
            terms: []const Schema.Term,
            events: []const Schema.LoopTimelineEvent,
        ) ?Reason {
            for (events, 0..) |event, event_index| switch (event.payload) {
                .storage_read => |read| if (fixedIndexBoundsLocationReason(
                    terms,
                    events[0..event_index],
                    read.location,
                )) |reason| return reason,
                .storage_write => |write| if (fixedIndexBoundsLocationReason(
                    terms,
                    events[0..event_index],
                    write.location,
                )) |reason| return reason,
                .path_assumption, .safety_checkpoint, .indexed_region_read, .summary_dependency, .call => {},
            };
            return null;
        }
    };
}
