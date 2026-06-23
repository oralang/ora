const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/sinora.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("sinora", lib_mod);

    const exe = b.addExecutable(.{
        .name = "sinora",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the Sinora SIR parser/checker");
    run_step.dependOn(&run_cmd.step);

    const tests = b.addTest(.{
        .root_module = lib_mod,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run Sinora tests");
    test_step.dependOn(&run_tests.step);

    const release_fixture_paths = [_][]const u8{
        "fixtures/init_runtime_offsets.sir",
        "fixtures/revert_runtime.sir",
        "fixtures/invalid_runtime.sir",
        "fixtures/selfdestruct_runtime.sir",
        "fixtures/data_offset_return.sir",
        "fixtures/mallocany_runtime_return.sir",
        "fixtures/salloc_runtime_return.sir",
        "fixtures/sallocany_runtime_return.sir",
        "fixtures/sallocany_two_runtime_return.sir",
        "fixtures/static_dynamic_runtime_return.sir",
        "fixtures/copy_noop_runtime_return.sir",
        "fixtures/mcopy_runtime_return.sir",
        "fixtures/flat_runtime_return.sir",
        "fixtures/sub_runtime_return.sir",
        "fixtures/mload_runtime_return.sir",
        "fixtures/narrow_memory_runtime_return.sir",
        "fixtures/narrow_memory16_runtime_return.sir",
        "fixtures/branch_terminal_runtime_return.sir",
        "fixtures/branch_terminal_zero_live_return.sir",
        "fixtures/deep_stack_spill_runtime_return.sir",
        "fixtures/constructor_no_arg_user_init_empty.sir",
        "fixtures/constructor_no_arg_user_init_storage.sir",
        "fixtures/constructor_one_arg_user_init_storage.sir",
        "fixtures/default_dispatcher_runtime_revert.sir",
        "fixtures/selector_case_terminal_return.sir",
        "fixtures/selector_no_arg_icall_return.sir",
        "fixtures/selector_no_arg_icall_codesize20_return.sir",
        "fixtures/selector_no_arg_icall_branch_return.sir",
        "fixtures/selector_no_arg_icall_branch_jump_bool_return.sir",
        "fixtures/selector_no_arg_icall_branch_jump_word_return.sir",
        "fixtures/selector_no_arg_icall_runtime_caller_guard_void_return.sir",
        "fixtures/selector_no_arg_icall_void_return.sir",
        "fixtures/selector_no_arg_icall_void_jump_chain_return.sir",
        "fixtures/selector_no_arg_icall_void_branch_join_tstore.sir",
        "fixtures/selector_no_arg_icall_void_tload_guard_tstore_return.sir",
        "fixtures/selector_no_arg_icall_void_compact_constant_branch_return.sir",
        "fixtures/selector_no_arg_icall_aggregate_pair_return.sir",
        "fixtures/selector_no_arg_icall_aggregate_triple_return.sir",
        "fixtures/selector_no_arg_icall_nested_aggregate_return.sir",
        "fixtures/selector_no_arg_icall_storage_tag_switch_return.sir",
        "fixtures/selector_no_arg_icall_constant_guard_chain_return.sir",
        "fixtures/selector_no_arg_icall_constant_guard_chain_distinct_return.sir",
        "fixtures/selector_no_arg_icall_guard_fallback_return.sir",
        "fixtures/selector_no_arg_icall_interleaved_error_guard_chain_return.sir",
        "fixtures/selector_no_arg_icall_compact_interleaved_error_guard_chain_return.sir",
        "fixtures/selector_no_arg_icall_builtins_return.sir",
        "fixtures/selector_no_arg_icall_error_union_return.sir",
        "fixtures/selector_no_arg_icall_dynamic_abi_return.sir",
        "fixtures/selector_no_arg_icall_internal_word_discard_return.sir",
        "fixtures/selector_no_arg_icall_internal_void_shared_empty_return.sir",
        "fixtures/selector_no_arg_icall_counted_loop_return.sir",
        "fixtures/selector_no_arg_icall_checked_storage_update_return.sir",
        "fixtures/selector_one_arg_icall_counted_loop_return.sir",
        "fixtures/selector_one_arg_icall_void_return.sir",
        "fixtures/selector_one_arg_icall_void_noop_return.sir",
        "fixtures/selector_one_arg_icall_void_storage_tstore_return.sir",
        "fixtures/selector_one_arg_icall_void_constant_branch_return.sir",
        "fixtures/selector_one_arg_icall_void_constant_branch_sstore_return.sir",
        "fixtures/selector_one_arg_icall_void_lock_guard_return.sir",
        "fixtures/selector_one_arg_icall_void_owner_guard_sstore_return.sir",
        "fixtures/selector_one_arg_icall_void_switch_sstore_return.sir",
        "fixtures/selector_one_arg_icall_void_guarded_switch_sstore_return.sir",
        "fixtures/selector_one_arg_icall_void_direct_bounded_proof_return.sir",
        "fixtures/selector_two_arg_icall_void_return.sir",
        "fixtures/selector_two_arg_icall_void_noop_return.sir",
        "fixtures/selector_two_arg_icall_first_masked_void_storage_tstore_return.sir",
        "fixtures/selector_mixed_icall_log_return.sir",
        "fixtures/selector_three_arg_first_two_masked_icall_log_return.sir",
        "fixtures/selector_three_arg_raw_bool_masked_void_return.sir",
        "fixtures/selector_three_arg_raw_bool_masked_void_storage_return.sir",
        "fixtures/selector_three_arg_named_payload_storage_return.sir",
        "fixtures/selector_three_arg_named_payload_field_return.sir",
        "fixtures/selector_three_arg_raw_bool_masked_word_return.sir",
        "fixtures/selector_one_arg_icall_word_return.sir",
        "fixtures/selector_one_arg_icall_nested_identity_chain_return.sir",
        "fixtures/selector_one_arg_icall_computed_word_return.sir",
        "fixtures/selector_one_arg_icall_arg_computed_word_return.sir",
        "fixtures/selector_one_arg_icall_arg_bool_comparison_return.sir",
        "fixtures/selector_one_arg_icall_masked_bit_not_return.sir",
        "fixtures/selector_one_arg_masked_void_ignored_return.sir",
        "fixtures/selector_one_arg_icall_storage_read_return.sir",
        "fixtures/selector_one_arg_icall_return_internal_void_call.sir",
        "fixtures/selector_one_arg_icall_masked_internal_storage_reads_return.sir",
        "fixtures/selector_one_arg_icall_static_memory_internal_word_return.sir",
        "fixtures/selector_one_arg_icall_static_memory_struct_return.sir",
        "fixtures/selector_one_arg_icall_static_memory_array_return.sir",
        "fixtures/selector_one_arg_icall_runtime_guard_return.sir",
        "fixtures/selector_one_arg_icall_memory_switch_return.sir",
        "fixtures/selector_one_arg_icall_lower_bound_return.sir",
        "fixtures/selector_one_arg_icall_lower_bound_branch_join_return.sir",
        "fixtures/selector_one_arg_icall_bounded_void_proof_return.sir",
        "fixtures/selector_one_arg_icall_bounded_void_strict_proof_return.sir",
        "fixtures/selector_one_arg_icall_bounded_void_multiple_proof_return.sir",
        "fixtures/selector_one_arg_icall_checked_add_proof_return.sir",
        "fixtures/selector_one_arg_icall_checked_add_memory_join_return.sir",
        "fixtures/selector_one_arg_icall_checked_storage_add_return.sir",
        "fixtures/selector_one_arg_icall_bounded_mul_proof_return.sir",
        "fixtures/selector_one_arg_icall_address_nonzero_proof_return.sir",
        "fixtures/selector_one_arg_icall_bool_guard_memory_join_return.sir",
        "fixtures/selector_one_arg_icall_signed_negate_overflow_return.sir",
        "fixtures/selector_one_arg_icall_signed_widening_return.sir",
        "fixtures/selector_two_arg_icall_word_return.sir",
        "fixtures/selector_two_arg_icall_first_arg_return.sir",
        "fixtures/selector_two_arg_icall_aggregate_pair_return.sir",
        "fixtures/selector_two_arg_icall_input_select_branch_jump_return.sir",
        "fixtures/selector_masked_void_supported_paths_return.sir",
        "fixtures/selector_two_arg_icall_signed_overflow_add_return.sir",
        "fixtures/selector_two_arg_icall_signed_overflow_sub_return.sir",
        "fixtures/selector_two_arg_icall_signed_mul_overflow_return.sir",
        "fixtures/selector_two_arg_icall_percent_overflow_proof_return.sir",
        "fixtures/selector_two_arg_icall_masked_word_return.sir",
        "fixtures/selector_two_arg_icall_mapping_copy_return.sir",
        "fixtures/selector_two_arg_icall_nested_generic_helper_return.sir",
        "fixtures/selector_two_arg_icall_first_bool_guard_second_word_return.sir",
        "fixtures/selector_two_arg_icall_mixed_masked_unmasked_return.sir",
        "fixtures/selector_two_arg_masked_void_noop_return.sir",
        "fixtures/selector_two_arg_second_bool_guard_void_noop_return.sir",
        "fixtures/selector_two_arg_selected_payload_branch_join_return.sir",
        "fixtures/selector_two_arg_selected_stateful_payload_branch_join_return.sir",
        "fixtures/selector_two_arg_selected_static_pointer_branch_join_return.sir",
        "fixtures/selector_two_arg_icall_all_width_add_return.sir",
        "fixtures/selector_two_arg_icall_all_width_sub_return.sir",
        "fixtures/selector_two_arg_icall_all_width_mul_return.sir",
        "fixtures/selector_two_arg_icall_all_width_shift_return.sir",
        "fixtures/selector_two_arg_icall_all_width_bitwise_return.sir",
        "fixtures/selector_two_arg_icall_all_width_div_mod_return.sir",
        "fixtures/selector_two_arg_icall_all_width_comparison_return.sir",
        "fixtures/selector_two_arg_icall_eq_ne_all_types_return.sir",
        "fixtures/selector_two_arg_icall_signed_wrapping_return.sir",
        "fixtures/selector_two_arg_icall_signed_comparison_return.sir",
        "fixtures/selector_two_arg_icall_word_many_ops_return.sir",
        "fixtures/selector_mixed_icall_math_storage_return.sir",
    };

    const release_fixtures_step = b.step("test-fixtures-release", "Compare focused release fixtures against Rust Plank bytecode");
    for (release_fixture_paths) |path| {
        const fixture_cmd = b.addRunArtifact(exe);
        fixture_cmd.addArgs(&.{
            "compare",
            "--release",
            "--json",
            "--expect",
            "both-accept-bytecode-equal",
            path,
        });
        release_fixtures_step.dependOn(&fixture_cmd.step);
    }

    const corpus_debug_cmd = b.addRunArtifact(exe);
    corpus_debug_cmd.addArgs(&.{ "compare-corpus", "--debug", "--json", "../artifacts" });
    const corpus_debug_step = b.step("test-corpus-debug", "Compare artifact SIR corpus against Rust Plank debug bytecode");
    corpus_debug_step.dependOn(&corpus_debug_cmd.step);

    const corpus_release_cmd = b.addRunArtifact(exe);
    corpus_release_cmd.addArgs(&.{ "compare-corpus", "--release", "--json", "../artifacts" });
    const corpus_release_step = b.step("test-corpus-release", "Compare artifact SIR corpus against Rust Plank release acceptance");
    corpus_release_step.dependOn(&corpus_release_cmd.step);

    // Generic Plank-port release backend gate. The migration is COMPLETE: the
    // generic backend reached full corpus byte parity (586/586) and is now the
    // backend behind `--release` too (legacy `release_codegen.zig` deleted), so
    // this `--release-generic` target is an explicit alias kept for clarity. It
    // stays out of the aggregate `test-corpus` step only to avoid running the
    // same 586-file comparison twice.
    const corpus_release_generic_cmd = b.addRunArtifact(exe);
    corpus_release_generic_cmd.addArgs(&.{ "compare-corpus", "--release-generic", "--json", "../artifacts" });
    const corpus_release_generic_step = b.step("test-corpus-release-generic", "Compare artifact SIR corpus against Rust Plank release bytecode via the generic backend");
    corpus_release_generic_step.dependOn(&corpus_release_generic_cmd.step);

    const corpus_step = b.step("test-corpus", "Run Sinora artifact corpus gates");
    corpus_step.dependOn(corpus_debug_step);
    corpus_step.dependOn(corpus_release_step);
}
