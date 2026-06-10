const std = @import("std");
const cache_stats_response = @import("cache_stats_response.zig");

test "lsp cache stats response: builds expected integer fields" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const object = try cache_stats_response.build(arena_state.allocator(), .{
        .open_documents = 2,
        .core_builds = 3,
        .semantic_index_builds = 14,
        .diagnostic_cache_builds = 3,
        .diagnostic_fast_builds = 4,
        .diagnostic_full_builds = 5,
        .workspace_discovery_cache_hits = 4,
        .cache_builder_capacity_requested = 30,
        .cache_builder_items_built = 24,
        .cache_builder_unused_capacity = 6,
        .cache_builder_growth_events = 0,
        .cache_side_map_capacity_requested = 11,
        .cache_side_map_items_built = 9,
        .cache_side_map_unused_capacity = 2,
        .cache_side_map_growth_events = 0,
        .response_builder_capacity_requested = 13,
        .response_builder_items_built = 10,
        .response_builder_unused_capacity = 3,
        .response_builder_growth_events = 1,
        .response_builder_capacity_bytes = 416,
        .response_builder_item_bytes = 320,
        .response_location_capacity_bytes = 96,
        .response_text_edit_capacity_bytes = 64,
        .response_completion_item_capacity_bytes = 128,
        .response_semantic_token_data_capacity_bytes = 80,
        .response_workspace_symbol_capacity_bytes = 32,
        .response_call_hierarchy_capacity_bytes = 16,
        .response_hover_capacity_bytes = 24,
        .response_definition_capacity_bytes = 48,
        .response_document_symbol_capacity_bytes = 56,
        .response_document_highlight_capacity_bytes = 64,
        .response_inlay_hint_capacity_bytes = 72,
        .response_code_lens_capacity_bytes = 88,
        .response_formatting_edit_capacity_bytes = 104,
        .response_selection_range_capacity_bytes = 112,
        .response_folding_range_capacity_bytes = 120,
        .response_document_link_capacity_bytes = 136,
        .response_code_action_capacity_bytes = 144,
        .response_signature_help_capacity_bytes = 152,
        .response_prepare_rename_capacity_bytes = 160,
        .response_string_bytes = 4096,
        .response_markdown_bytes = 1024,
        .response_location_string_bytes = 384,
        .response_text_edit_string_bytes = 192,
        .response_completion_string_bytes = 512,
        .response_completion_markdown_bytes = 256,
        .response_hover_string_bytes = 128,
        .response_hover_markdown_bytes = 128,
        .response_definition_string_bytes = 112,
        .response_signature_string_bytes = 96,
        .response_signature_markdown_bytes = 64,
        .response_document_symbol_string_bytes = 80,
        .response_workspace_symbol_string_bytes = 72,
        .response_call_hierarchy_string_bytes = 48,
        .response_inlay_hint_string_bytes = 40,
        .response_code_lens_string_bytes = 36,
        .response_formatting_edit_string_bytes = 34,
        .response_document_link_string_bytes = 33,
        .response_code_action_string_bytes = 32,
        .response_prepare_rename_string_bytes = 31,
        .dependent_diagnostic_publish_runs = 2,
        .dependent_diagnostic_published_documents = 3,
        .dependent_diagnostic_publish_skips = 4,
        .diagnostic_debounce_pending = 1,
        .diagnostic_debounce_scheduled = 2,
        .diagnostic_debounce_canceled = 1,
        .diagnostic_debounce_flushed = 1,
        .diagnostic_debounce_cleared = 1,
        .stale_document_change_skips = 5,
        .edit_diagnostic_fast_publishes = 6,
        .edit_diagnostic_full_skips = 7,
        .cold_documents = 6,
        .cold_document_max_count = 512,
        .cold_document_evictions = 7,
        .cold_document_refresh_checks = 8,
        .cold_document_refreshes = 9,
        .cold_document_stale_removals = 10,
        .workspace_index_bytes = 1024,
        .open_source_bytes = 2048,
        .cold_source_bytes = 4096,
        .cold_source_max_bytes = 67108864,
        .open_token_count = 11,
        .open_token_diagnostic_count = 1,
        .open_symbol_count = 7,
        .open_occurrence_count = 5,
        .open_imported_member_count = 4,
        .open_call_edge_count = 3,
        .workspace_index_symbol_count = 13,
        .workspace_index_root_symbol_count = 2,
        .workspace_index_callable_symbol_count = 8,
        .workspace_index_import_count = 6,
        .workspace_index_occurrence_count = 21,
        .workspace_index_imported_member_count = 9,
        .workspace_index_call_edge_count = 12,
        .workspace_index_interned_string_bytes = 768,
        .workspace_index_interned_string_count = 18,
        .workspace_index_duplicate_string_bytes_saved = 320,
        .workspace_index_interned_string_capacity_requested = 32,
        .workspace_index_interned_string_items_built = 18,
        .workspace_index_interned_string_unused_capacity = 14,
        .workspace_index_interned_string_growth_events = 2,
        .formatting_cache_entries = 1,
        .formatting_cache_bytes = 456,
        .cold_workspace_index_entries = 3,
        .cold_workspace_index_bytes = 640,
        .cold_workspace_index_interned_string_bytes = 192,
        .cold_workspace_index_interned_string_count = 7,
        .cold_workspace_index_duplicate_string_bytes_saved = 80,
        .cold_workspace_index_interned_string_capacity_requested = 12,
        .cold_workspace_index_interned_string_items_built = 7,
        .cold_workspace_index_interned_string_unused_capacity = 5,
        .cold_workspace_index_interned_string_growth_events = 1,
        .token_cache_bytes = 512,
        .import_resolution_bytes = 384,
        .semantic_index_bytes = 256,
        .diagnostic_cache_bytes = 128,
        .lex_builds = 4,
        .parse_builds = 5,
        .ast_lower_builds = 6,
        .item_index_builds = 7,
        .resolve_builds = 8,
        .const_eval_builds = 9,
        .type_check_builds = 10,
        .formatter_builds = 11,
        .server_allocator_bytes_peak = 4096,
        .server_allocator_unscoped_alloc_calls = 12,
        .server_allocator_unscoped_bytes_allocated = 13,
        .server_allocator_request_alloc_calls = 14,
        .server_allocator_request_bytes_allocated = 15,
        .server_allocator_response_alloc_calls = 16,
        .server_allocator_response_bytes_allocated = 17,
        .server_allocator_cache_build_alloc_calls = 18,
        .server_allocator_cache_build_bytes_allocated = 19,
        .server_allocator_temp_analysis_alloc_calls = 20,
        .server_allocator_temp_analysis_bytes_allocated = 21,
    });

    try expectInt(object, "openDocuments", 2);
    try expectInt(object, "coreBuilds", 3);
    try expectInt(object, "diagnosticCacheBuilds", 3);
    try expectInt(object, "diagnosticFastBuilds", 4);
    try expectInt(object, "diagnosticFullBuilds", 5);
    try expectInt(object, "workspaceDiscoveryCacheHits", 4);
    try expectInt(object, "cacheBuilderCapacityRequested", 30);
    try expectInt(object, "cacheBuilderItemsBuilt", 24);
    try expectInt(object, "cacheBuilderUnusedCapacity", 6);
    try expectInt(object, "cacheBuilderGrowthEvents", 0);
    try expectInt(object, "cacheSideMapCapacityRequested", 11);
    try expectInt(object, "cacheSideMapItemsBuilt", 9);
    try expectInt(object, "cacheSideMapUnusedCapacity", 2);
    try expectInt(object, "cacheSideMapGrowthEvents", 0);
    try expectInt(object, "responseBuilderCapacityRequested", 13);
    try expectInt(object, "responseBuilderItemsBuilt", 10);
    try expectInt(object, "responseBuilderUnusedCapacity", 3);
    try expectInt(object, "responseBuilderGrowthEvents", 1);
    try expectInt(object, "responseBuilderCapacityBytes", 416);
    try expectInt(object, "responseBuilderItemBytes", 320);
    try expectInt(object, "responseLocationCapacityBytes", 96);
    try expectInt(object, "responseTextEditCapacityBytes", 64);
    try expectInt(object, "responseCompletionItemCapacityBytes", 128);
    try expectInt(object, "responseSemanticTokenDataCapacityBytes", 80);
    try expectInt(object, "responseWorkspaceSymbolCapacityBytes", 32);
    try expectInt(object, "responseCallHierarchyCapacityBytes", 16);
    try expectInt(object, "responseHoverCapacityBytes", 24);
    try expectInt(object, "responseDefinitionCapacityBytes", 48);
    try expectInt(object, "responseDocumentSymbolCapacityBytes", 56);
    try expectInt(object, "responseDocumentHighlightCapacityBytes", 64);
    try expectInt(object, "responseInlayHintCapacityBytes", 72);
    try expectInt(object, "responseCodeLensCapacityBytes", 88);
    try expectInt(object, "responseFormattingEditCapacityBytes", 104);
    try expectInt(object, "responseSelectionRangeCapacityBytes", 112);
    try expectInt(object, "responseFoldingRangeCapacityBytes", 120);
    try expectInt(object, "responseDocumentLinkCapacityBytes", 136);
    try expectInt(object, "responseCodeActionCapacityBytes", 144);
    try expectInt(object, "responseSignatureHelpCapacityBytes", 152);
    try expectInt(object, "responsePrepareRenameCapacityBytes", 160);
    try expectInt(object, "responseStringBytes", 4096);
    try expectInt(object, "responseMarkdownBytes", 1024);
    try expectInt(object, "responseLocationStringBytes", 384);
    try expectInt(object, "responseTextEditStringBytes", 192);
    try expectInt(object, "responseCompletionStringBytes", 512);
    try expectInt(object, "responseCompletionMarkdownBytes", 256);
    try expectInt(object, "responseHoverStringBytes", 128);
    try expectInt(object, "responseHoverMarkdownBytes", 128);
    try expectInt(object, "responseDefinitionStringBytes", 112);
    try expectInt(object, "responseSignatureStringBytes", 96);
    try expectInt(object, "responseSignatureMarkdownBytes", 64);
    try expectInt(object, "responseDocumentSymbolStringBytes", 80);
    try expectInt(object, "responseWorkspaceSymbolStringBytes", 72);
    try expectInt(object, "responseCallHierarchyStringBytes", 48);
    try expectInt(object, "responseInlayHintStringBytes", 40);
    try expectInt(object, "responseCodeLensStringBytes", 36);
    try expectInt(object, "responseFormattingEditStringBytes", 34);
    try expectInt(object, "responseDocumentLinkStringBytes", 33);
    try expectInt(object, "responseCodeActionStringBytes", 32);
    try expectInt(object, "responsePrepareRenameStringBytes", 31);
    try expectInt(object, "dependentDiagnosticPublishRuns", 2);
    try expectInt(object, "dependentDiagnosticPublishedDocuments", 3);
    try expectInt(object, "dependentDiagnosticPublishSkips", 4);
    try expectInt(object, "diagnosticDebouncePending", 1);
    try expectInt(object, "diagnosticDebounceScheduled", 2);
    try expectInt(object, "diagnosticDebounceCanceled", 1);
    try expectInt(object, "diagnosticDebounceFlushed", 1);
    try expectInt(object, "diagnosticDebounceCleared", 1);
    try expectInt(object, "staleDocumentChangeSkips", 5);
    try expectInt(object, "editDiagnosticFastPublishes", 6);
    try expectInt(object, "editDiagnosticFullSkips", 7);
    try expectInt(object, "coldDocuments", 6);
    try expectInt(object, "coldDocumentMaxCount", 512);
    try expectInt(object, "coldDocumentEvictions", 7);
    try expectInt(object, "coldDocumentRefreshChecks", 8);
    try expectInt(object, "coldDocumentRefreshes", 9);
    try expectInt(object, "coldDocumentStaleRemovals", 10);
    try expectInt(object, "workspaceIndexBytes", 1024);
    try expectInt(object, "openSourceBytes", 2048);
    try expectInt(object, "coldSourceBytes", 4096);
    try expectInt(object, "coldSourceMaxBytes", 67108864);
    try expectInt(object, "openTokenCount", 11);
    try expectInt(object, "openTokenDiagnosticCount", 1);
    try expectInt(object, "openSymbolCount", 7);
    try expectInt(object, "openOccurrenceCount", 5);
    try expectInt(object, "openImportedMemberCount", 4);
    try expectInt(object, "openCallEdgeCount", 3);
    try expectInt(object, "workspaceIndexSymbolCount", 13);
    try expectInt(object, "workspaceIndexRootSymbolCount", 2);
    try expectInt(object, "workspaceIndexCallableSymbolCount", 8);
    try expectInt(object, "workspaceIndexImportCount", 6);
    try expectInt(object, "workspaceIndexOccurrenceCount", 21);
    try expectInt(object, "workspaceIndexImportedMemberCount", 9);
    try expectInt(object, "workspaceIndexCallEdgeCount", 12);
    try expectInt(object, "workspaceIndexInternedStringBytes", 768);
    try expectInt(object, "workspaceIndexInternedStringCount", 18);
    try expectInt(object, "workspaceIndexDuplicateStringBytesSaved", 320);
    try expectInt(object, "workspaceIndexInternedStringCapacityRequested", 32);
    try expectInt(object, "workspaceIndexInternedStringItemsBuilt", 18);
    try expectInt(object, "workspaceIndexInternedStringUnusedCapacity", 14);
    try expectInt(object, "workspaceIndexInternedStringGrowthEvents", 2);
    try expectInt(object, "formattingCacheEntries", 1);
    try expectInt(object, "formattingCacheBytes", 456);
    try expectInt(object, "coldWorkspaceIndexEntries", 3);
    try expectInt(object, "coldWorkspaceIndexBytes", 640);
    try expectInt(object, "coldWorkspaceIndexInternedStringBytes", 192);
    try expectInt(object, "coldWorkspaceIndexInternedStringCount", 7);
    try expectInt(object, "coldWorkspaceIndexDuplicateStringBytesSaved", 80);
    try expectInt(object, "coldWorkspaceIndexInternedStringCapacityRequested", 12);
    try expectInt(object, "coldWorkspaceIndexInternedStringItemsBuilt", 7);
    try expectInt(object, "coldWorkspaceIndexInternedStringUnusedCapacity", 5);
    try expectInt(object, "coldWorkspaceIndexInternedStringGrowthEvents", 1);
    try expectInt(object, "tokenCacheBytes", 512);
    try expectInt(object, "importResolutionBytes", 384);
    try expectInt(object, "semanticIndexBytes", 256);
    try expectInt(object, "diagnosticCacheBytes", 128);
    try expectInt(object, "lexBuilds", 4);
    try expectInt(object, "parseBuilds", 5);
    try expectInt(object, "astLowerBuilds", 6);
    try expectInt(object, "itemIndexBuilds", 7);
    try expectInt(object, "resolveBuilds", 8);
    try expectInt(object, "constEvalBuilds", 9);
    try expectInt(object, "typeCheckBuilds", 10);
    try expectInt(object, "formatterBuilds", 11);
    try expectInt(object, "serverAllocatorBytesPeak", 4096);
    try expectInt(object, "serverAllocatorUnscopedAllocCalls", 12);
    try expectInt(object, "serverAllocatorUnscopedBytesAllocated", 13);
    try expectInt(object, "serverAllocatorRequestAllocCalls", 14);
    try expectInt(object, "serverAllocatorRequestBytesAllocated", 15);
    try expectInt(object, "serverAllocatorResponseAllocCalls", 16);
    try expectInt(object, "serverAllocatorResponseBytesAllocated", 17);
    try expectInt(object, "serverAllocatorCacheBuildAllocCalls", 18);
    try expectInt(object, "serverAllocatorCacheBuildBytesAllocated", 19);
    try expectInt(object, "serverAllocatorTempAnalysisAllocCalls", 20);
    try expectInt(object, "serverAllocatorTempAnalysisBytesAllocated", 21);
    try expectInt(object, "semanticIndexBuilds", 14);
}

test "lsp cache stats response: saturates values above json integer range" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const object = try cache_stats_response.build(arena_state.allocator(), .{
        .server_allocator_bytes_peak = std.math.maxInt(usize),
    });

    const expected: i64 = if (std.math.maxInt(usize) > std.math.maxInt(i64))
        std.math.maxInt(i64)
    else
        @intCast(std.math.maxInt(usize));
    try expectInt(object, "serverAllocatorBytesPeak", expected);
}

fn expectInt(object: std.json.ObjectMap, key: []const u8, expected: i64) !void {
    const value = object.get(key) orelse return error.ExpectedKey;
    try std.testing.expectEqual(expected, value.integer);
}
