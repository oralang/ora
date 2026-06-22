const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Snapshot = struct {
    open_documents: usize = 0,
    core_builds: usize = 0,
    semantic_index_builds: usize = 0,
    occurrence_index_builds: usize = 0,
    imported_member_index_builds: usize = 0,
    call_edge_index_builds: usize = 0,
    workspace_index_builds: usize = 0,
    cold_workspace_index_builds: usize = 0,
    incoming_call_target_index_builds: usize = 0,
    workspace_discovery_runs: usize = 0,
    workspace_discovery_files_seen: usize = 0,
    workspace_discovery_files_enqueued: usize = 0,
    workspace_discovery_skipped: usize = 0,
    workspace_discovery_limit_hits: usize = 0,
    workspace_discovery_cache_hits: usize = 0,
    workspace_discovery_cache_rebuilds: usize = 0,
    workspace_discovery_max_files: usize = 0,
    diagnostic_cache_builds: usize = 0,
    diagnostic_fast_builds: usize = 0,
    diagnostic_full_builds: usize = 0,
    cache_builder_capacity_requested: usize = 0,
    cache_builder_items_built: usize = 0,
    cache_builder_unused_capacity: usize = 0,
    cache_builder_growth_events: usize = 0,
    cache_side_map_capacity_requested: usize = 0,
    cache_side_map_items_built: usize = 0,
    cache_side_map_unused_capacity: usize = 0,
    cache_side_map_growth_events: usize = 0,
    response_builder_capacity_requested: usize = 0,
    response_builder_items_built: usize = 0,
    response_builder_unused_capacity: usize = 0,
    response_builder_growth_events: usize = 0,
    response_builder_capacity_bytes: usize = 0,
    response_builder_item_bytes: usize = 0,
    response_location_capacity_bytes: usize = 0,
    response_text_edit_capacity_bytes: usize = 0,
    response_completion_item_capacity_bytes: usize = 0,
    response_semantic_token_data_capacity_bytes: usize = 0,
    response_workspace_symbol_capacity_bytes: usize = 0,
    response_call_hierarchy_capacity_bytes: usize = 0,
    response_hover_capacity_bytes: usize = 0,
    response_definition_capacity_bytes: usize = 0,
    response_document_symbol_capacity_bytes: usize = 0,
    response_document_highlight_capacity_bytes: usize = 0,
    response_inlay_hint_capacity_bytes: usize = 0,
    response_code_lens_capacity_bytes: usize = 0,
    response_formatting_edit_capacity_bytes: usize = 0,
    response_selection_range_capacity_bytes: usize = 0,
    response_folding_range_capacity_bytes: usize = 0,
    response_document_link_capacity_bytes: usize = 0,
    response_code_action_capacity_bytes: usize = 0,
    response_signature_help_capacity_bytes: usize = 0,
    response_prepare_rename_capacity_bytes: usize = 0,
    response_string_bytes: usize = 0,
    response_markdown_bytes: usize = 0,
    response_location_string_bytes: usize = 0,
    response_text_edit_string_bytes: usize = 0,
    response_completion_string_bytes: usize = 0,
    response_completion_markdown_bytes: usize = 0,
    response_hover_string_bytes: usize = 0,
    response_hover_markdown_bytes: usize = 0,
    response_definition_string_bytes: usize = 0,
    response_signature_string_bytes: usize = 0,
    response_signature_markdown_bytes: usize = 0,
    response_document_symbol_string_bytes: usize = 0,
    response_workspace_symbol_string_bytes: usize = 0,
    response_call_hierarchy_string_bytes: usize = 0,
    response_inlay_hint_string_bytes: usize = 0,
    response_code_lens_string_bytes: usize = 0,
    response_formatting_edit_string_bytes: usize = 0,
    response_document_link_string_bytes: usize = 0,
    response_code_action_string_bytes: usize = 0,
    response_prepare_rename_string_bytes: usize = 0,
    dependent_diagnostic_publish_runs: usize = 0,
    dependent_diagnostic_published_documents: usize = 0,
    dependent_diagnostic_publish_skips: usize = 0,
    diagnostic_debounce_pending: usize = 0,
    diagnostic_debounce_scheduled: usize = 0,
    diagnostic_debounce_canceled: usize = 0,
    diagnostic_debounce_flushed: usize = 0,
    diagnostic_debounce_cleared: usize = 0,
    stale_document_change_skips: usize = 0,
    edit_diagnostic_fast_publishes: usize = 0,
    edit_diagnostic_full_skips: usize = 0,
    cold_documents: usize = 0,
    cold_document_max_count: usize = 0,
    cold_document_evictions: usize = 0,
    cold_document_refresh_checks: usize = 0,
    cold_document_refreshes: usize = 0,
    cold_document_stale_removals: usize = 0,
    workspace_index_entries: usize = 0,
    workspace_index_bytes: usize = 0,
    workspace_index_max_bytes: usize = 0,
    workspace_index_evictions: usize = 0,
    open_source_bytes: usize = 0,
    cold_source_bytes: usize = 0,
    cold_source_max_bytes: usize = 0,
    open_token_count: usize = 0,
    open_token_diagnostic_count: usize = 0,
    open_symbol_count: usize = 0,
    open_occurrence_count: usize = 0,
    open_imported_member_count: usize = 0,
    open_call_edge_count: usize = 0,
    workspace_index_symbol_count: usize = 0,
    workspace_index_root_symbol_count: usize = 0,
    workspace_index_callable_symbol_count: usize = 0,
    workspace_index_import_count: usize = 0,
    workspace_index_occurrence_count: usize = 0,
    workspace_index_imported_member_count: usize = 0,
    workspace_index_call_edge_count: usize = 0,
    incoming_call_target_name_count: usize = 0,
    incoming_call_target_uri_count: usize = 0,
    workspace_index_interned_string_bytes: usize = 0,
    workspace_index_interned_string_count: usize = 0,
    workspace_index_duplicate_string_bytes_saved: usize = 0,
    workspace_index_interned_string_capacity_requested: usize = 0,
    workspace_index_interned_string_items_built: usize = 0,
    workspace_index_interned_string_unused_capacity: usize = 0,
    workspace_index_interned_string_growth_events: usize = 0,
    cold_workspace_index_entries: usize = 0,
    cold_workspace_index_bytes: usize = 0,
    cold_workspace_index_interned_string_bytes: usize = 0,
    cold_workspace_index_interned_string_count: usize = 0,
    cold_workspace_index_duplicate_string_bytes_saved: usize = 0,
    cold_workspace_index_interned_string_capacity_requested: usize = 0,
    cold_workspace_index_interned_string_items_built: usize = 0,
    cold_workspace_index_interned_string_unused_capacity: usize = 0,
    cold_workspace_index_interned_string_growth_events: usize = 0,
    document_state_bytes: usize = 0,
    line_index_bytes: usize = 0,
    token_cache_bytes: usize = 0,
    std_import_alias_bytes: usize = 0,
    import_resolution_bytes: usize = 0,
    semantic_index_bytes: usize = 0,
    occurrence_index_bytes: usize = 0,
    imported_member_index_bytes: usize = 0,
    call_edge_index_bytes: usize = 0,
    incoming_call_target_index_bytes: usize = 0,
    formatting_cache_entries: usize = 0,
    formatting_cache_bytes: usize = 0,
    diagnostic_cache_bytes: usize = 0,
    lex_builds: usize = 0,
    parse_builds: usize = 0,
    ast_lower_builds: usize = 0,
    item_index_builds: usize = 0,
    resolve_builds: usize = 0,
    const_eval_builds: usize = 0,
    type_check_builds: usize = 0,
    formatter_builds: usize = 0,
    server_allocator_alloc_calls: usize = 0,
    server_allocator_resize_calls: usize = 0,
    server_allocator_remap_calls: usize = 0,
    server_allocator_free_calls: usize = 0,
    server_allocator_bytes_allocated: usize = 0,
    server_allocator_bytes_freed: usize = 0,
    server_allocator_bytes_live: usize = 0,
    server_allocator_bytes_peak: usize = 0,
    server_allocator_unscoped_alloc_calls: usize = 0,
    server_allocator_unscoped_bytes_allocated: usize = 0,
    server_allocator_request_alloc_calls: usize = 0,
    server_allocator_request_bytes_allocated: usize = 0,
    server_allocator_response_alloc_calls: usize = 0,
    server_allocator_response_bytes_allocated: usize = 0,
    server_allocator_cache_build_alloc_calls: usize = 0,
    server_allocator_cache_build_bytes_allocated: usize = 0,
    server_allocator_temp_analysis_alloc_calls: usize = 0,
    server_allocator_temp_analysis_bytes_allocated: usize = 0,
};

pub fn build(arena: Allocator, snapshot: Snapshot) !std.json.ObjectMap {
    var object = try std.json.ObjectMap.init(arena, &.{}, &.{});
    try putJsonInt(arena, &object, "openDocuments", snapshot.open_documents);
    try putJsonInt(arena, &object, "coreBuilds", snapshot.core_builds);
    try putJsonInt(arena, &object, "semanticIndexBuilds", snapshot.semantic_index_builds);
    try putJsonInt(arena, &object, "occurrenceIndexBuilds", snapshot.occurrence_index_builds);
    try putJsonInt(arena, &object, "importedMemberIndexBuilds", snapshot.imported_member_index_builds);
    try putJsonInt(arena, &object, "callEdgeIndexBuilds", snapshot.call_edge_index_builds);
    try putJsonInt(arena, &object, "workspaceIndexBuilds", snapshot.workspace_index_builds);
    try putJsonInt(arena, &object, "coldWorkspaceIndexBuilds", snapshot.cold_workspace_index_builds);
    try putJsonInt(arena, &object, "incomingCallTargetIndexBuilds", snapshot.incoming_call_target_index_builds);
    try putJsonInt(arena, &object, "workspaceDiscoveryRuns", snapshot.workspace_discovery_runs);
    try putJsonInt(arena, &object, "workspaceDiscoveryFilesSeen", snapshot.workspace_discovery_files_seen);
    try putJsonInt(arena, &object, "workspaceDiscoveryFilesEnqueued", snapshot.workspace_discovery_files_enqueued);
    try putJsonInt(arena, &object, "workspaceDiscoverySkipped", snapshot.workspace_discovery_skipped);
    try putJsonInt(arena, &object, "workspaceDiscoveryLimitHits", snapshot.workspace_discovery_limit_hits);
    try putJsonInt(arena, &object, "workspaceDiscoveryCacheHits", snapshot.workspace_discovery_cache_hits);
    try putJsonInt(arena, &object, "workspaceDiscoveryCacheRebuilds", snapshot.workspace_discovery_cache_rebuilds);
    try putJsonInt(arena, &object, "workspaceDiscoveryMaxFiles", snapshot.workspace_discovery_max_files);
    try putJsonInt(arena, &object, "diagnosticCacheBuilds", snapshot.diagnostic_cache_builds);
    try putJsonInt(arena, &object, "diagnosticFastBuilds", snapshot.diagnostic_fast_builds);
    try putJsonInt(arena, &object, "diagnosticFullBuilds", snapshot.diagnostic_full_builds);
    try putJsonInt(arena, &object, "cacheBuilderCapacityRequested", snapshot.cache_builder_capacity_requested);
    try putJsonInt(arena, &object, "cacheBuilderItemsBuilt", snapshot.cache_builder_items_built);
    try putJsonInt(arena, &object, "cacheBuilderUnusedCapacity", snapshot.cache_builder_unused_capacity);
    try putJsonInt(arena, &object, "cacheBuilderGrowthEvents", snapshot.cache_builder_growth_events);
    try putJsonInt(arena, &object, "cacheSideMapCapacityRequested", snapshot.cache_side_map_capacity_requested);
    try putJsonInt(arena, &object, "cacheSideMapItemsBuilt", snapshot.cache_side_map_items_built);
    try putJsonInt(arena, &object, "cacheSideMapUnusedCapacity", snapshot.cache_side_map_unused_capacity);
    try putJsonInt(arena, &object, "cacheSideMapGrowthEvents", snapshot.cache_side_map_growth_events);
    try putJsonInt(arena, &object, "responseBuilderCapacityRequested", snapshot.response_builder_capacity_requested);
    try putJsonInt(arena, &object, "responseBuilderItemsBuilt", snapshot.response_builder_items_built);
    try putJsonInt(arena, &object, "responseBuilderUnusedCapacity", snapshot.response_builder_unused_capacity);
    try putJsonInt(arena, &object, "responseBuilderGrowthEvents", snapshot.response_builder_growth_events);
    try putJsonInt(arena, &object, "responseBuilderCapacityBytes", snapshot.response_builder_capacity_bytes);
    try putJsonInt(arena, &object, "responseBuilderItemBytes", snapshot.response_builder_item_bytes);
    try putJsonInt(arena, &object, "responseLocationCapacityBytes", snapshot.response_location_capacity_bytes);
    try putJsonInt(arena, &object, "responseTextEditCapacityBytes", snapshot.response_text_edit_capacity_bytes);
    try putJsonInt(arena, &object, "responseCompletionItemCapacityBytes", snapshot.response_completion_item_capacity_bytes);
    try putJsonInt(arena, &object, "responseSemanticTokenDataCapacityBytes", snapshot.response_semantic_token_data_capacity_bytes);
    try putJsonInt(arena, &object, "responseWorkspaceSymbolCapacityBytes", snapshot.response_workspace_symbol_capacity_bytes);
    try putJsonInt(arena, &object, "responseCallHierarchyCapacityBytes", snapshot.response_call_hierarchy_capacity_bytes);
    try putJsonInt(arena, &object, "responseHoverCapacityBytes", snapshot.response_hover_capacity_bytes);
    try putJsonInt(arena, &object, "responseDefinitionCapacityBytes", snapshot.response_definition_capacity_bytes);
    try putJsonInt(arena, &object, "responseDocumentSymbolCapacityBytes", snapshot.response_document_symbol_capacity_bytes);
    try putJsonInt(arena, &object, "responseDocumentHighlightCapacityBytes", snapshot.response_document_highlight_capacity_bytes);
    try putJsonInt(arena, &object, "responseInlayHintCapacityBytes", snapshot.response_inlay_hint_capacity_bytes);
    try putJsonInt(arena, &object, "responseCodeLensCapacityBytes", snapshot.response_code_lens_capacity_bytes);
    try putJsonInt(arena, &object, "responseFormattingEditCapacityBytes", snapshot.response_formatting_edit_capacity_bytes);
    try putJsonInt(arena, &object, "responseSelectionRangeCapacityBytes", snapshot.response_selection_range_capacity_bytes);
    try putJsonInt(arena, &object, "responseFoldingRangeCapacityBytes", snapshot.response_folding_range_capacity_bytes);
    try putJsonInt(arena, &object, "responseDocumentLinkCapacityBytes", snapshot.response_document_link_capacity_bytes);
    try putJsonInt(arena, &object, "responseCodeActionCapacityBytes", snapshot.response_code_action_capacity_bytes);
    try putJsonInt(arena, &object, "responseSignatureHelpCapacityBytes", snapshot.response_signature_help_capacity_bytes);
    try putJsonInt(arena, &object, "responsePrepareRenameCapacityBytes", snapshot.response_prepare_rename_capacity_bytes);
    try putJsonInt(arena, &object, "responseStringBytes", snapshot.response_string_bytes);
    try putJsonInt(arena, &object, "responseMarkdownBytes", snapshot.response_markdown_bytes);
    try putJsonInt(arena, &object, "responseLocationStringBytes", snapshot.response_location_string_bytes);
    try putJsonInt(arena, &object, "responseTextEditStringBytes", snapshot.response_text_edit_string_bytes);
    try putJsonInt(arena, &object, "responseCompletionStringBytes", snapshot.response_completion_string_bytes);
    try putJsonInt(arena, &object, "responseCompletionMarkdownBytes", snapshot.response_completion_markdown_bytes);
    try putJsonInt(arena, &object, "responseHoverStringBytes", snapshot.response_hover_string_bytes);
    try putJsonInt(arena, &object, "responseHoverMarkdownBytes", snapshot.response_hover_markdown_bytes);
    try putJsonInt(arena, &object, "responseDefinitionStringBytes", snapshot.response_definition_string_bytes);
    try putJsonInt(arena, &object, "responseSignatureStringBytes", snapshot.response_signature_string_bytes);
    try putJsonInt(arena, &object, "responseSignatureMarkdownBytes", snapshot.response_signature_markdown_bytes);
    try putJsonInt(arena, &object, "responseDocumentSymbolStringBytes", snapshot.response_document_symbol_string_bytes);
    try putJsonInt(arena, &object, "responseWorkspaceSymbolStringBytes", snapshot.response_workspace_symbol_string_bytes);
    try putJsonInt(arena, &object, "responseCallHierarchyStringBytes", snapshot.response_call_hierarchy_string_bytes);
    try putJsonInt(arena, &object, "responseInlayHintStringBytes", snapshot.response_inlay_hint_string_bytes);
    try putJsonInt(arena, &object, "responseCodeLensStringBytes", snapshot.response_code_lens_string_bytes);
    try putJsonInt(arena, &object, "responseFormattingEditStringBytes", snapshot.response_formatting_edit_string_bytes);
    try putJsonInt(arena, &object, "responseDocumentLinkStringBytes", snapshot.response_document_link_string_bytes);
    try putJsonInt(arena, &object, "responseCodeActionStringBytes", snapshot.response_code_action_string_bytes);
    try putJsonInt(arena, &object, "responsePrepareRenameStringBytes", snapshot.response_prepare_rename_string_bytes);
    try putJsonInt(arena, &object, "dependentDiagnosticPublishRuns", snapshot.dependent_diagnostic_publish_runs);
    try putJsonInt(arena, &object, "dependentDiagnosticPublishedDocuments", snapshot.dependent_diagnostic_published_documents);
    try putJsonInt(arena, &object, "dependentDiagnosticPublishSkips", snapshot.dependent_diagnostic_publish_skips);
    try putJsonInt(arena, &object, "diagnosticDebouncePending", snapshot.diagnostic_debounce_pending);
    try putJsonInt(arena, &object, "diagnosticDebounceScheduled", snapshot.diagnostic_debounce_scheduled);
    try putJsonInt(arena, &object, "diagnosticDebounceCanceled", snapshot.diagnostic_debounce_canceled);
    try putJsonInt(arena, &object, "diagnosticDebounceFlushed", snapshot.diagnostic_debounce_flushed);
    try putJsonInt(arena, &object, "diagnosticDebounceCleared", snapshot.diagnostic_debounce_cleared);
    try putJsonInt(arena, &object, "staleDocumentChangeSkips", snapshot.stale_document_change_skips);
    try putJsonInt(arena, &object, "editDiagnosticFastPublishes", snapshot.edit_diagnostic_fast_publishes);
    try putJsonInt(arena, &object, "editDiagnosticFullSkips", snapshot.edit_diagnostic_full_skips);
    try putJsonInt(arena, &object, "coldDocuments", snapshot.cold_documents);
    try putJsonInt(arena, &object, "coldDocumentMaxCount", snapshot.cold_document_max_count);
    try putJsonInt(arena, &object, "coldDocumentEvictions", snapshot.cold_document_evictions);
    try putJsonInt(arena, &object, "coldDocumentRefreshChecks", snapshot.cold_document_refresh_checks);
    try putJsonInt(arena, &object, "coldDocumentRefreshes", snapshot.cold_document_refreshes);
    try putJsonInt(arena, &object, "coldDocumentStaleRemovals", snapshot.cold_document_stale_removals);
    try putJsonInt(arena, &object, "workspaceIndexEntries", snapshot.workspace_index_entries);
    try putJsonInt(arena, &object, "workspaceIndexBytes", snapshot.workspace_index_bytes);
    try putJsonInt(arena, &object, "workspaceIndexMaxBytes", snapshot.workspace_index_max_bytes);
    try putJsonInt(arena, &object, "workspaceIndexEvictions", snapshot.workspace_index_evictions);
    try putJsonInt(arena, &object, "openSourceBytes", snapshot.open_source_bytes);
    try putJsonInt(arena, &object, "coldSourceBytes", snapshot.cold_source_bytes);
    try putJsonInt(arena, &object, "coldSourceMaxBytes", snapshot.cold_source_max_bytes);
    try putJsonInt(arena, &object, "openTokenCount", snapshot.open_token_count);
    try putJsonInt(arena, &object, "openTokenDiagnosticCount", snapshot.open_token_diagnostic_count);
    try putJsonInt(arena, &object, "openSymbolCount", snapshot.open_symbol_count);
    try putJsonInt(arena, &object, "openOccurrenceCount", snapshot.open_occurrence_count);
    try putJsonInt(arena, &object, "openImportedMemberCount", snapshot.open_imported_member_count);
    try putJsonInt(arena, &object, "openCallEdgeCount", snapshot.open_call_edge_count);
    try putJsonInt(arena, &object, "workspaceIndexSymbolCount", snapshot.workspace_index_symbol_count);
    try putJsonInt(arena, &object, "workspaceIndexRootSymbolCount", snapshot.workspace_index_root_symbol_count);
    try putJsonInt(arena, &object, "workspaceIndexCallableSymbolCount", snapshot.workspace_index_callable_symbol_count);
    try putJsonInt(arena, &object, "workspaceIndexImportCount", snapshot.workspace_index_import_count);
    try putJsonInt(arena, &object, "workspaceIndexOccurrenceCount", snapshot.workspace_index_occurrence_count);
    try putJsonInt(arena, &object, "workspaceIndexImportedMemberCount", snapshot.workspace_index_imported_member_count);
    try putJsonInt(arena, &object, "workspaceIndexCallEdgeCount", snapshot.workspace_index_call_edge_count);
    try putJsonInt(arena, &object, "incomingCallTargetNameCount", snapshot.incoming_call_target_name_count);
    try putJsonInt(arena, &object, "incomingCallTargetUriCount", snapshot.incoming_call_target_uri_count);
    try putJsonInt(arena, &object, "workspaceIndexInternedStringBytes", snapshot.workspace_index_interned_string_bytes);
    try putJsonInt(arena, &object, "workspaceIndexInternedStringCount", snapshot.workspace_index_interned_string_count);
    try putJsonInt(arena, &object, "workspaceIndexDuplicateStringBytesSaved", snapshot.workspace_index_duplicate_string_bytes_saved);
    try putJsonInt(arena, &object, "workspaceIndexInternedStringCapacityRequested", snapshot.workspace_index_interned_string_capacity_requested);
    try putJsonInt(arena, &object, "workspaceIndexInternedStringItemsBuilt", snapshot.workspace_index_interned_string_items_built);
    try putJsonInt(arena, &object, "workspaceIndexInternedStringUnusedCapacity", snapshot.workspace_index_interned_string_unused_capacity);
    try putJsonInt(arena, &object, "workspaceIndexInternedStringGrowthEvents", snapshot.workspace_index_interned_string_growth_events);
    try putJsonInt(arena, &object, "coldWorkspaceIndexEntries", snapshot.cold_workspace_index_entries);
    try putJsonInt(arena, &object, "coldWorkspaceIndexBytes", snapshot.cold_workspace_index_bytes);
    try putJsonInt(arena, &object, "coldWorkspaceIndexInternedStringBytes", snapshot.cold_workspace_index_interned_string_bytes);
    try putJsonInt(arena, &object, "coldWorkspaceIndexInternedStringCount", snapshot.cold_workspace_index_interned_string_count);
    try putJsonInt(arena, &object, "coldWorkspaceIndexDuplicateStringBytesSaved", snapshot.cold_workspace_index_duplicate_string_bytes_saved);
    try putJsonInt(arena, &object, "coldWorkspaceIndexInternedStringCapacityRequested", snapshot.cold_workspace_index_interned_string_capacity_requested);
    try putJsonInt(arena, &object, "coldWorkspaceIndexInternedStringItemsBuilt", snapshot.cold_workspace_index_interned_string_items_built);
    try putJsonInt(arena, &object, "coldWorkspaceIndexInternedStringUnusedCapacity", snapshot.cold_workspace_index_interned_string_unused_capacity);
    try putJsonInt(arena, &object, "coldWorkspaceIndexInternedStringGrowthEvents", snapshot.cold_workspace_index_interned_string_growth_events);
    try putJsonInt(arena, &object, "documentStateBytes", snapshot.document_state_bytes);
    try putJsonInt(arena, &object, "lineIndexBytes", snapshot.line_index_bytes);
    try putJsonInt(arena, &object, "tokenCacheBytes", snapshot.token_cache_bytes);
    try putJsonInt(arena, &object, "stdImportAliasBytes", snapshot.std_import_alias_bytes);
    try putJsonInt(arena, &object, "importResolutionBytes", snapshot.import_resolution_bytes);
    try putJsonInt(arena, &object, "semanticIndexBytes", snapshot.semantic_index_bytes);
    try putJsonInt(arena, &object, "occurrenceIndexBytes", snapshot.occurrence_index_bytes);
    try putJsonInt(arena, &object, "importedMemberIndexBytes", snapshot.imported_member_index_bytes);
    try putJsonInt(arena, &object, "callEdgeIndexBytes", snapshot.call_edge_index_bytes);
    try putJsonInt(arena, &object, "incomingCallTargetIndexBytes", snapshot.incoming_call_target_index_bytes);
    try putJsonInt(arena, &object, "formattingCacheEntries", snapshot.formatting_cache_entries);
    try putJsonInt(arena, &object, "formattingCacheBytes", snapshot.formatting_cache_bytes);
    try putJsonInt(arena, &object, "diagnosticCacheBytes", snapshot.diagnostic_cache_bytes);
    try putJsonInt(arena, &object, "lexBuilds", snapshot.lex_builds);
    try putJsonInt(arena, &object, "parseBuilds", snapshot.parse_builds);
    try putJsonInt(arena, &object, "astLowerBuilds", snapshot.ast_lower_builds);
    try putJsonInt(arena, &object, "itemIndexBuilds", snapshot.item_index_builds);
    try putJsonInt(arena, &object, "resolveBuilds", snapshot.resolve_builds);
    try putJsonInt(arena, &object, "constEvalBuilds", snapshot.const_eval_builds);
    try putJsonInt(arena, &object, "typeCheckBuilds", snapshot.type_check_builds);
    try putJsonInt(arena, &object, "formatterBuilds", snapshot.formatter_builds);
    try putJsonInt(arena, &object, "serverAllocatorAllocCalls", snapshot.server_allocator_alloc_calls);
    try putJsonInt(arena, &object, "serverAllocatorResizeCalls", snapshot.server_allocator_resize_calls);
    try putJsonInt(arena, &object, "serverAllocatorRemapCalls", snapshot.server_allocator_remap_calls);
    try putJsonInt(arena, &object, "serverAllocatorFreeCalls", snapshot.server_allocator_free_calls);
    try putJsonInt(arena, &object, "serverAllocatorBytesAllocated", snapshot.server_allocator_bytes_allocated);
    try putJsonInt(arena, &object, "serverAllocatorBytesFreed", snapshot.server_allocator_bytes_freed);
    try putJsonInt(arena, &object, "serverAllocatorBytesLive", snapshot.server_allocator_bytes_live);
    try putJsonInt(arena, &object, "serverAllocatorBytesPeak", snapshot.server_allocator_bytes_peak);
    try putJsonInt(arena, &object, "serverAllocatorUnscopedAllocCalls", snapshot.server_allocator_unscoped_alloc_calls);
    try putJsonInt(arena, &object, "serverAllocatorUnscopedBytesAllocated", snapshot.server_allocator_unscoped_bytes_allocated);
    try putJsonInt(arena, &object, "serverAllocatorRequestAllocCalls", snapshot.server_allocator_request_alloc_calls);
    try putJsonInt(arena, &object, "serverAllocatorRequestBytesAllocated", snapshot.server_allocator_request_bytes_allocated);
    try putJsonInt(arena, &object, "serverAllocatorResponseAllocCalls", snapshot.server_allocator_response_alloc_calls);
    try putJsonInt(arena, &object, "serverAllocatorResponseBytesAllocated", snapshot.server_allocator_response_bytes_allocated);
    try putJsonInt(arena, &object, "serverAllocatorCacheBuildAllocCalls", snapshot.server_allocator_cache_build_alloc_calls);
    try putJsonInt(arena, &object, "serverAllocatorCacheBuildBytesAllocated", snapshot.server_allocator_cache_build_bytes_allocated);
    try putJsonInt(arena, &object, "serverAllocatorTempAnalysisAllocCalls", snapshot.server_allocator_temp_analysis_alloc_calls);
    try putJsonInt(arena, &object, "serverAllocatorTempAnalysisBytesAllocated", snapshot.server_allocator_temp_analysis_bytes_allocated);
    return object;
}

fn putJsonInt(arena: std.mem.Allocator, object: *std.json.ObjectMap, key: []const u8, value: usize) !void {
    const integer_value: i64 = if (value > std.math.maxInt(i64)) std.math.maxInt(i64) else @intCast(value);
    try object.put(arena, key, .{ .integer = integer_value });
}
