#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

from lsp_jsonrpc import (
    JsonRpcClient,
    did_change_incremental,
    did_open,
    file_uri,
    full_range,
    initialize,
    position_of,
    wait_diagnostics,
)


TRACKED_CACHE_BYTE_KEYS = (
    "documentStateBytes",
    "lineIndexBytes",
    "tokenCacheBytes",
    "semanticIndexBytes",
    "occurrenceIndexBytes",
    "importedMemberIndexBytes",
    "callEdgeIndexBytes",
    "incomingCallTargetIndexBytes",
    "formattingCacheBytes",
    "workspaceIndexBytes",
    "diagnosticCacheBytes",
    "importResolutionBytes",
    "stdImportAliasBytes",
)

PHASE_COUNTER_KEYS = (
    "lexBuilds",
    "parseBuilds",
    "astLowerBuilds",
    "itemIndexBuilds",
    "resolveBuilds",
    "constEvalBuilds",
    "typeCheckBuilds",
)

FORMATTER_COUNTER_KEY = "formatterBuilds"

CACHE_BUILDER_COUNTER_KEYS = (
    "cacheBuilderCapacityRequested",
    "cacheBuilderItemsBuilt",
    "cacheBuilderUnusedCapacity",
    "cacheBuilderGrowthEvents",
)

CACHE_SIDE_MAP_COUNTER_KEYS = (
    "cacheSideMapCapacityRequested",
    "cacheSideMapItemsBuilt",
    "cacheSideMapUnusedCapacity",
    "cacheSideMapGrowthEvents",
)

WORKSPACE_INTERNER_COUNTER_KEYS = (
    "workspaceIndexInternedStringCapacityRequested",
    "workspaceIndexInternedStringItemsBuilt",
    "workspaceIndexInternedStringUnusedCapacity",
    "workspaceIndexInternedStringGrowthEvents",
)

COLD_WORKSPACE_INTERNER_COUNTER_KEYS = (
    "coldWorkspaceIndexEntries",
    "coldWorkspaceIndexBytes",
    "coldWorkspaceIndexInternedStringBytes",
    "coldWorkspaceIndexInternedStringCount",
    "coldWorkspaceIndexInternedStringCapacityRequested",
    "coldWorkspaceIndexInternedStringItemsBuilt",
    "coldWorkspaceIndexInternedStringUnusedCapacity",
    "coldWorkspaceIndexInternedStringGrowthEvents",
)

COLD_DOCUMENT_COUNTER_KEYS = (
    "coldDocuments",
    "coldDocumentMaxCount",
    "coldDocumentEvictions",
    "coldDocumentRefreshChecks",
    "coldDocumentRefreshes",
    "coldDocumentStaleRemovals",
    "coldSourceBytes",
    "coldSourceMaxBytes",
)

PROFILE_P95_BUDGET_MS = {
    "quick": {
        ("bench", "completion.cache_hit"): 1.0,
        ("bench", "definition.cache_hit"): 0.5,
        ("bench", "didChange.diagnostics"): 300.0,
        ("bench", "documentHighlight.cache_hit"): 2.0,
        ("bench", "documentSymbol.cache_hit"): 6.0,
        ("bench", "foldingRange.cache_hit"): 2.0,
        ("bench", "formatting.cache_hit"): 1.0,
        ("bench", "hover.cache_hit"): 1.0,
        ("bench", "inlayHint.cache_hit"): 8.0,
        ("bench", "prepareRename.cache_hit"): 1.0,
        ("bench", "references.cache_hit"): 2.0,
        ("bench", "rename.cache_hit"): 2.0,
        ("bench", "selectionRange.cache_hit"): 1.0,
        ("bench", "semanticTokens.cache_hit"): 4.0,
        ("bench", "signatureHelp.cache_hit"): 1.0,
        ("bench", "workspaceSymbol.cache_hit"): 1.0,
        ("code_action", "codeAction.quickfix.cache_hit"): 1.0,
        ("cold", "callHierarchy.incoming.cold_imported_member.first_build"): 150.0,
        ("cold", "callHierarchy.incoming.cold_imported_member.cache_hit"): 2.0,
        ("cold", "callHierarchy.incoming.recursive_cold_imported_member.first_build"): 150.0,
        ("cold", "callHierarchy.incoming.recursive_cold_imported_member.cache_hit"): 1.0,
        ("cold", "callHierarchy.outgoing.cold_imported_target.first_build"): 150.0,
        ("cold", "callHierarchy.outgoing.cold_imported_target.cache_hit"): 1.0,
        ("cold", "callHierarchy.outgoing.recursive_cold_imported_target.first_build"): 150.0,
        ("cold", "callHierarchy.outgoing.recursive_cold_imported_target.cache_hit"): 1.0,
        ("cold", "references.cold_imported_member.first_build"): 150.0,
        ("cold", "references.cold_imported_member.cache_hit"): 1.5,
        ("cold", "rename.cold_imported_member.cache_hit"): 1.5,
        ("contractsuite", "callHierarchy.incoming.cache_hit"): 1.0,
        ("contractsuite", "callHierarchy.outgoing.cache_hit"): 1.0,
        ("contractsuite", "callHierarchy.prepare.cache_hit"): 1.0,
        ("contractsuite", "codeLens.cache_hit"): 3.0,
        ("contractsuite", "completion.cache_hit"): 3.0,
        ("contractsuite", "documentLink.cache_hit"): 6.0,
        ("contractsuite", "documentSymbol.cache_hit"): 4.0,
        ("contractsuite", "definition.imported_member.cache_hit"): 1.0,
        ("contractsuite", "foldingRange.cache_hit"): 1.0,
        ("contractsuite", "hover.cache_hit"): 1.0,
        ("contractsuite", "references.imported_member.cache_hit"): 1.0,
        ("contractsuite", "rename.imported_member.cache_hit"): 1.0,
        ("contractsuite", "semanticTokens.cache_hit"): 3.0,
        ("contractsuite", "workspaceSymbol.cache_hit"): 1.0,
        ("importfanin", "references.imported_member.cache_hit"): 3.0,
        ("importfanin", "rename.imported_member.cache_hit"): 3.0,
        ("importfanin", "workspaceSymbol.cache_hit"): 1.0,
        ("importfanout", "documentSymbol.cache_hit"): 3.0,
        ("importfanout", "references.imported_member.cache_hit"): 2.0,
        ("importfanout", "workspaceSymbol.cache_hit"): 1.0,
        ("positionlarge", "positionLookup.bottom.cache_hit"): 1.0,
        ("positionlarge", "positionLookup.middle.cache_hit"): 1.0,
        ("positionlarge", "positionLookup.top.cache_hit"): 1.0,
        ("positionsmall", "positionLookup.bottom.cache_hit"): 1.0,
        ("positionsmall", "positionLookup.middle.cache_hit"): 1.0,
        ("positionsmall", "positionLookup.top.cache_hit"): 1.0,
        ("stress", "documentSymbol.cache_hit"): 4.0,
        ("stress", "references.cache_hit"): 4.0,
        ("stress", "semanticTokens.cache_hit"): 4.0,
        ("stress", "workspaceSymbol.cache_hit"): 2.0,
    },
    "release": {
        ("bench", "completion.cache_hit"): 0.2,
        ("bench", "definition.cache_hit"): 0.2,
        ("bench", "didChange.diagnostics"): 150.0,
        ("bench", "documentHighlight.cache_hit"): 0.75,
        ("bench", "documentSymbol.cache_hit"): 4.0,
        ("bench", "foldingRange.cache_hit"): 0.5,
        ("bench", "formatting.cache_hit"): 0.75,
        ("bench", "hover.cache_hit"): 0.25,
        ("bench", "inlayHint.cache_hit"): 2.0,
        ("bench", "prepareRename.cache_hit"): 0.2,
        ("bench", "references.cache_hit"): 0.75,
        ("bench", "rename.cache_hit"): 0.75,
        ("bench", "selectionRange.cache_hit"): 0.2,
        ("bench", "semanticTokens.cache_hit"): 3.0,
        ("bench", "signatureHelp.cache_hit"): 0.2,
        ("bench", "workspaceSymbol.cache_hit"): 0.2,
        ("code_action", "codeAction.quickfix.cache_hit"): 0.2,
        ("cold", "callHierarchy.incoming.cold_imported_member.first_build"): 50.0,
        ("cold", "callHierarchy.incoming.cold_imported_member.cache_hit"): 1.5,
        ("cold", "callHierarchy.incoming.recursive_cold_imported_member.first_build"): 50.0,
        ("cold", "callHierarchy.incoming.recursive_cold_imported_member.cache_hit"): 0.3,
        ("cold", "callHierarchy.outgoing.cold_imported_target.first_build"): 50.0,
        ("cold", "callHierarchy.outgoing.cold_imported_target.cache_hit"): 0.25,
        ("cold", "callHierarchy.outgoing.recursive_cold_imported_target.first_build"): 50.0,
        ("cold", "callHierarchy.outgoing.recursive_cold_imported_target.cache_hit"): 0.25,
        ("cold", "references.cold_imported_member.first_build"): 50.0,
        ("cold", "references.cold_imported_member.cache_hit"): 0.6,
        ("cold", "rename.cold_imported_member.cache_hit"): 0.6,
        ("contractsuite", "callHierarchy.incoming.cache_hit"): 0.2,
        ("contractsuite", "callHierarchy.outgoing.cache_hit"): 0.2,
        ("contractsuite", "callHierarchy.prepare.cache_hit"): 0.2,
        ("contractsuite", "codeLens.cache_hit"): 1.0,
        ("contractsuite", "completion.cache_hit"): 0.5,
        ("contractsuite", "documentLink.cache_hit"): 1.5,
        ("contractsuite", "documentSymbol.cache_hit"): 2.0,
        ("contractsuite", "definition.imported_member.cache_hit"): 0.25,
        ("contractsuite", "foldingRange.cache_hit"): 0.5,
        ("contractsuite", "hover.cache_hit"): 0.25,
        ("contractsuite", "references.imported_member.cache_hit"): 0.25,
        ("contractsuite", "rename.imported_member.cache_hit"): 0.25,
        ("contractsuite", "semanticTokens.cache_hit"): 1.5,
        ("contractsuite", "workspaceSymbol.cache_hit"): 0.2,
        ("importfanin", "references.imported_member.cache_hit"): 1.0,
        ("importfanin", "rename.imported_member.cache_hit"): 1.0,
        ("importfanin", "workspaceSymbol.cache_hit"): 0.5,
        ("importfanout", "documentSymbol.cache_hit"): 1.0,
        ("importfanout", "references.imported_member.cache_hit"): 0.75,
        ("importfanout", "workspaceSymbol.cache_hit"): 0.5,
        ("positionlarge", "positionLookup.bottom.cache_hit"): 0.2,
        ("positionlarge", "positionLookup.middle.cache_hit"): 0.2,
        ("positionlarge", "positionLookup.top.cache_hit"): 0.2,
        ("positionsmall", "positionLookup.bottom.cache_hit"): 0.2,
        ("positionsmall", "positionLookup.middle.cache_hit"): 0.2,
        ("positionsmall", "positionLookup.top.cache_hit"): 0.2,
        ("stress", "documentSymbol.cache_hit"): 3.0,
        ("stress", "references.cache_hit"): 3.0,
        ("stress", "semanticTokens.cache_hit"): 3.0,
        ("stress", "workspaceSymbol.cache_hit"): 1.5,
    },
}

PROFILE_BYTE_BUDGETS = {
    "quick": {
        "totalSourceBytes": 768 * 1024,
        "openSourceBytes": 768 * 1024,
        "coldSourceBytes": 16 * 1024,
        "trackedCacheBytes": 6 * 1024 * 1024,
        "trackedTotalBytes": 7 * 1024 * 1024,
        "peakRssKiB": 64 * 1024,
        "components": {
            "documentStateBytes": 3 * 1024 * 1024,
            "lineIndexBytes": 96 * 1024,
            "tokenCacheBytes": 2 * 1024 * 1024,
            "semanticIndexBytes": 256 * 1024,
            "occurrenceIndexBytes": 256 * 1024,
            "importedMemberIndexBytes": 64 * 1024,
            "callEdgeIndexBytes": 32 * 1024,
            "incomingCallTargetIndexBytes": 16 * 1024,
            "formattingCacheBytes": 64 * 1024,
            "workspaceIndexBytes": 512 * 1024,
            "diagnosticCacheBytes": 256 * 1024,
            "importResolutionBytes": 128 * 1024,
            "stdImportAliasBytes": 0,
        },
    },
    "release": {
        "totalSourceBytes": 3 * 1024 * 1024,
        "openSourceBytes": 3 * 1024 * 1024,
        "coldSourceBytes": 32 * 1024,
        "trackedCacheBytes": 16 * 1024 * 1024,
        "trackedTotalBytes": 20 * 1024 * 1024,
        "peakRssKiB": 80 * 1024,
        "components": {
            "documentStateBytes": 9 * 1024 * 1024,
            "lineIndexBytes": 256 * 1024,
            "tokenCacheBytes": 5 * 1024 * 1024,
            "semanticIndexBytes": 640 * 1024,
            "occurrenceIndexBytes": 768 * 1024,
            "importedMemberIndexBytes": 128 * 1024,
            "callEdgeIndexBytes": 96 * 1024,
            "incomingCallTargetIndexBytes": 24 * 1024,
            "formattingCacheBytes": 128 * 1024,
            "workspaceIndexBytes": 1536 * 1024,
            "diagnosticCacheBytes": 768 * 1024,
            "importResolutionBytes": 256 * 1024,
            "stdImportAliasBytes": 0,
        },
    },
}

DB_BACKED_PHASE_FLAT_ROWS = (
    ("bench", "hover.cache_hit"),
    ("bench", "completion.cache_hit"),
    ("bench", "definition.cache_hit"),
    ("bench", "prepareRename.cache_hit"),
    ("bench", "references.cache_hit"),
    ("bench", "documentSymbol.cache_hit"),
    ("bench", "workspaceSymbol.cache_hit"),
    ("bench", "documentHighlight.cache_hit"),
    ("bench", "rename.cache_hit"),
    ("bench", "semanticTokens.cache_hit"),
    ("bench", "inlayHint.cache_hit"),
    ("bench", "signatureHelp.cache_hit"),
    ("bench", "foldingRange.cache_hit"),
    ("code_action", "codeAction.quickfix.cache_hit"),
    ("contractsuite", "hover.cache_hit"),
    ("contractsuite", "completion.cache_hit"),
    ("contractsuite", "documentSymbol.cache_hit"),
    ("contractsuite", "definition.imported_member.cache_hit"),
    ("contractsuite", "documentLink.cache_hit"),
    ("contractsuite", "workspaceSymbol.cache_hit"),
    ("contractsuite", "callHierarchy.prepare.cache_hit"),
    ("contractsuite", "callHierarchy.incoming.cache_hit"),
    ("contractsuite", "callHierarchy.outgoing.cache_hit"),
    ("contractsuite", "codeLens.cache_hit"),
    ("contractsuite", "foldingRange.cache_hit"),
    ("cold", "callHierarchy.incoming.cold_imported_member.cache_hit"),
    ("cold", "callHierarchy.incoming.recursive_cold_imported_member.cache_hit"),
    ("cold", "callHierarchy.outgoing.cold_imported_target.cache_hit"),
    ("cold", "callHierarchy.outgoing.recursive_cold_imported_target.cache_hit"),
    ("contractsuite", "references.imported_member.cache_hit"),
    ("contractsuite", "rename.imported_member.cache_hit"),
    ("importfanin", "references.imported_member.cache_hit"),
    ("importfanin", "rename.imported_member.cache_hit"),
    ("importfanin", "workspaceSymbol.cache_hit"),
    ("importfanout", "documentSymbol.cache_hit"),
    ("importfanout", "references.imported_member.cache_hit"),
    ("importfanout", "workspaceSymbol.cache_hit"),
    ("cold", "references.cold_imported_member.cache_hit"),
    ("cold", "rename.cold_imported_member.cache_hit"),
    ("contractsuite", "semanticTokens.cache_hit"),
    ("stress", "documentSymbol.cache_hit"),
    ("stress", "workspaceSymbol.cache_hit"),
    ("stress", "references.cache_hit"),
    ("stress", "semanticTokens.cache_hit"),
)

ALLOCATION_ATTRIBUTION_ROWS = (
    *DB_BACKED_PHASE_FLAT_ROWS,
    ("bench", "formatting.cache_hit"),
)

PROFILE_ALLOCATION_ATTRIBUTION_BUDGET_BYTES = {
    "quick": {
        "default": 768 * 1024,
        ("bench", "definition.cache_hit"): 128 * 1024,
        ("bench", "documentSymbol.cache_hit"): 1 * 1024 * 1024,
        ("bench", "formatting.cache_hit"): 256 * 1024,
        ("bench", "inlayHint.cache_hit"): 256 * 1024,
    },
    "release": {
        "default": 1024 * 1024,
        ("bench", "definition.cache_hit"): 128 * 1024,
        ("bench", "documentSymbol.cache_hit"): 2 * 1024 * 1024,
        ("bench", "formatting.cache_hit"): 512 * 1024,
        ("bench", "inlayHint.cache_hit"): 512 * 1024,
    },
}

COLD_IMPORTED_MEMBER_ROWS = (
    ("cold", "references.cold_imported_member.cache_hit"),
    ("cold", "rename.cold_imported_member.cache_hit"),
    ("cold", "callHierarchy.incoming.cold_imported_member.cache_hit"),
    ("cold", "callHierarchy.incoming.recursive_cold_imported_member.cache_hit"),
)

COLD_FIRST_BUILD_ROWS = (
    ("cold", "references.cold_imported_member.first_build"),
    ("cold", "callHierarchy.incoming.cold_imported_member.first_build"),
    ("cold", "callHierarchy.incoming.recursive_cold_imported_member.first_build"),
    ("cold", "callHierarchy.outgoing.cold_imported_target.first_build"),
    ("cold", "callHierarchy.outgoing.recursive_cold_imported_target.first_build"),
)

COLD_DIRECT_IMPORTED_MEMBER_FIRST_BUILD_ROWS = (
    ("cold", "references.cold_imported_member.first_build"),
)

COLD_WORKSPACE_FIRST_BUILD_ROWS = tuple(
    row for row in COLD_FIRST_BUILD_ROWS
    if row not in COLD_DIRECT_IMPORTED_MEMBER_FIRST_BUILD_ROWS
)

POSITION_LOOKUP_ROWS = tuple(
    (fixture, "positionLookup.%s.cache_hit" % location)
    for fixture in ("positionsmall", "positionlarge")
    for location in ("top", "middle", "bottom")
)

CACHE_HIT_ZERO_DELTA_KEYS = (
    *PHASE_COUNTER_KEYS,
    "semanticIndexBuilds",
    "occurrenceIndexBuilds",
    "importedMemberIndexBuilds",
    "callEdgeIndexBuilds",
    "incomingCallTargetIndexBuilds",
    "workspaceIndexBuilds",
    "coldWorkspaceIndexBuilds",
    "workspaceDiscoveryRuns",
    "serverAllocatorCacheBuildBytesAllocated",
    FORMATTER_COUNTER_KEY,
)

SCOPED_ALLOCATOR_ALLOC_CALL_KEYS = (
    "serverAllocatorUnscopedAllocCalls",
    "serverAllocatorRequestAllocCalls",
    "serverAllocatorResponseAllocCalls",
    "serverAllocatorCacheBuildAllocCalls",
    "serverAllocatorTempAnalysisAllocCalls",
)

SCOPED_ALLOCATOR_BYTE_KEYS = (
    "serverAllocatorUnscopedBytesAllocated",
    "serverAllocatorRequestBytesAllocated",
    "serverAllocatorResponseBytesAllocated",
    "serverAllocatorCacheBuildBytesAllocated",
    "serverAllocatorTempAnalysisBytesAllocated",
)

SCOPED_ALLOCATOR_ACTIVE_BYTE_KEYS = (
    "serverAllocatorRequestBytesAllocated",
    "serverAllocatorResponseBytesAllocated",
    "serverAllocatorCacheBuildBytesAllocated",
    "serverAllocatorTempAnalysisBytesAllocated",
)

FORMATTER_ZERO_DELTA_KEYS = (
    *PHASE_COUNTER_KEYS,
    "semanticIndexBuilds",
    "occurrenceIndexBuilds",
    "importedMemberIndexBuilds",
    "callEdgeIndexBuilds",
    "incomingCallTargetIndexBuilds",
    "workspaceIndexBuilds",
    "coldWorkspaceIndexBuilds",
    "workspaceDiscoveryRuns",
    "workspaceDiscoveryCacheHits",
    "lineIndexBytes",
    "diagnosticCacheBuilds",
    FORMATTER_COUNTER_KEY,
)

RESPONSE_COVERAGE_COUNTERS = (
    ("completion", "responseCompletionItemCapacityBytes"),
    ("hover", "responseHoverCapacityBytes"),
    ("definition", "responseDefinitionCapacityBytes"),
    ("references", "responseLocationCapacityBytes"),
    ("documentSymbol", "responseDocumentSymbolCapacityBytes"),
    ("workspaceSymbol", "responseWorkspaceSymbolCapacityBytes"),
    ("semanticTokens", "responseSemanticTokenDataCapacityBytes"),
    ("documentLink", "responseDocumentLinkCapacityBytes"),
    ("documentHighlight", "responseDocumentHighlightCapacityBytes"),
    ("inlayHint", "responseInlayHintCapacityBytes"),
    ("codeLens", "responseCodeLensCapacityBytes"),
    ("foldingRange", "responseFoldingRangeCapacityBytes"),
    ("formatting", "responseFormattingEditCapacityBytes"),
    ("selectionRange", "responseSelectionRangeCapacityBytes"),
    ("signatureHelp", "responseSignatureHelpCapacityBytes"),
    ("prepareRename", "responsePrepareRenameCapacityBytes"),
    ("rename", "responseTextEditCapacityBytes"),
    ("callHierarchy", "responseCallHierarchyCapacityBytes"),
    ("codeAction", "responseCodeActionCapacityBytes"),
)

RESPONSE_COUNTER_BY_OPERATION = {
    "hover.cache_hit": "responseHoverCapacityBytes",
    "completion.cache_hit": "responseCompletionItemCapacityBytes",
    "definition.cache_hit": "responseDefinitionCapacityBytes",
    "definition.imported_member.cache_hit": "responseDefinitionCapacityBytes",
    "references.cache_hit": "responseLocationCapacityBytes",
    "references.imported_member.cache_hit": "responseLocationCapacityBytes",
    "references.cold_imported_member.cache_hit": "responseLocationCapacityBytes",
    "documentSymbol.cache_hit": "responseDocumentSymbolCapacityBytes",
    "workspaceSymbol.cache_hit": "responseWorkspaceSymbolCapacityBytes",
    "semanticTokens.cache_hit": "responseSemanticTokenDataCapacityBytes",
    "documentHighlight.cache_hit": "responseDocumentHighlightCapacityBytes",
    "signatureHelp.cache_hit": "responseSignatureHelpCapacityBytes",
    "inlayHint.cache_hit": "responseInlayHintCapacityBytes",
    "foldingRange.cache_hit": "responseFoldingRangeCapacityBytes",
    "selectionRange.cache_hit": "responseSelectionRangeCapacityBytes",
    "positionLookup.top.cache_hit": "responseSelectionRangeCapacityBytes",
    "positionLookup.middle.cache_hit": "responseSelectionRangeCapacityBytes",
    "positionLookup.bottom.cache_hit": "responseSelectionRangeCapacityBytes",
    "prepareRename.cache_hit": "responsePrepareRenameCapacityBytes",
    "rename.cache_hit": "responseTextEditCapacityBytes",
    "rename.imported_member.cache_hit": "responseTextEditCapacityBytes",
    "rename.cold_imported_member.cache_hit": "responseTextEditCapacityBytes",
    "formatting.cache_hit": "responseFormattingEditCapacityBytes",
    "documentLink.cache_hit": "responseDocumentLinkCapacityBytes",
    "codeLens.cache_hit": "responseCodeLensCapacityBytes",
    "callHierarchy.prepare.cache_hit": "responseCallHierarchyCapacityBytes",
    "callHierarchy.incoming.cache_hit": "responseCallHierarchyCapacityBytes",
    "callHierarchy.incoming.cold_imported_member.cache_hit": "responseCallHierarchyCapacityBytes",
    "callHierarchy.incoming.recursive_cold_imported_member.cache_hit": "responseCallHierarchyCapacityBytes",
    "callHierarchy.outgoing.cache_hit": "responseCallHierarchyCapacityBytes",
    "callHierarchy.outgoing.cold_imported_target.cache_hit": "responseCallHierarchyCapacityBytes",
    "callHierarchy.outgoing.recursive_cold_imported_target.cache_hit": "responseCallHierarchyCapacityBytes",
    "codeAction.quickfix.cache_hit": "responseCodeActionCapacityBytes",
}

RESPONSE_STRING_COUNTER_BY_OPERATION = {
    "hover.cache_hit": "responseHoverStringBytes",
    "completion.cache_hit": "responseCompletionStringBytes",
    "definition.cache_hit": "responseDefinitionStringBytes",
    "definition.imported_member.cache_hit": "responseDefinitionStringBytes",
    "references.cache_hit": "responseLocationStringBytes",
    "references.imported_member.cache_hit": "responseLocationStringBytes",
    "references.cold_imported_member.cache_hit": "responseLocationStringBytes",
    "documentSymbol.cache_hit": "responseDocumentSymbolStringBytes",
    "workspaceSymbol.cache_hit": "responseWorkspaceSymbolStringBytes",
    "signatureHelp.cache_hit": "responseSignatureStringBytes",
    "inlayHint.cache_hit": "responseInlayHintStringBytes",
    "prepareRename.cache_hit": "responsePrepareRenameStringBytes",
    "rename.cache_hit": "responseTextEditStringBytes",
    "rename.imported_member.cache_hit": "responseTextEditStringBytes",
    "rename.cold_imported_member.cache_hit": "responseTextEditStringBytes",
    "formatting.cache_hit": "responseFormattingEditStringBytes",
    "documentLink.cache_hit": "responseDocumentLinkStringBytes",
    "codeLens.cache_hit": "responseCodeLensStringBytes",
    "callHierarchy.prepare.cache_hit": "responseCallHierarchyStringBytes",
    "callHierarchy.incoming.cache_hit": "responseCallHierarchyStringBytes",
    "callHierarchy.incoming.cold_imported_member.cache_hit": "responseCallHierarchyStringBytes",
    "callHierarchy.incoming.recursive_cold_imported_member.cache_hit": "responseCallHierarchyStringBytes",
    "callHierarchy.outgoing.cache_hit": "responseCallHierarchyStringBytes",
    "callHierarchy.outgoing.cold_imported_target.cache_hit": "responseCallHierarchyStringBytes",
    "callHierarchy.outgoing.recursive_cold_imported_target.cache_hit": "responseCallHierarchyStringBytes",
    "codeAction.quickfix.cache_hit": "responseCodeActionStringBytes",
}

RESPONSE_MARKDOWN_COUNTER_BY_OPERATION = {
    "hover.cache_hit": "responseHoverMarkdownBytes",
    "completion.cache_hit": "responseCompletionMarkdownBytes",
    "signatureHelp.cache_hit": "responseSignatureMarkdownBytes",
}

expected_response_counters: set[str] = set()
expected_response_string_counters: set[str] = set()
expected_response_markdown_counters: set[str] = set()

DELTA_COUNTER_KEYS = (
    *PHASE_COUNTER_KEYS,
    "semanticIndexBuilds",
    "occurrenceIndexBuilds",
    "importedMemberIndexBuilds",
    "callEdgeIndexBuilds",
    "incomingCallTargetIndexBuilds",
    "workspaceIndexBuilds",
    "coldWorkspaceIndexBuilds",
    "workspaceDiscoveryRuns",
    "workspaceDiscoveryCacheHits",
    "lineIndexBytes",
    "diagnosticCacheBuilds",
    "diagnosticFastBuilds",
    "diagnosticFullBuilds",
    "editDiagnosticFastPublishes",
    "editDiagnosticFullSkips",
    "dependentDiagnosticPublishRuns",
    "dependentDiagnosticPublishedDocuments",
    "dependentDiagnosticPublishSkips",
    "diagnosticDebouncePending",
    "diagnosticDebounceScheduled",
    "diagnosticDebounceCanceled",
    "diagnosticDebounceFlushed",
    "diagnosticDebounceCleared",
    FORMATTER_COUNTER_KEY,
    "serverAllocatorAllocCalls",
    "serverAllocatorResizeCalls",
    "serverAllocatorRemapCalls",
    "serverAllocatorFreeCalls",
    "serverAllocatorBytesAllocated",
    "serverAllocatorBytesFreed",
    *SCOPED_ALLOCATOR_ALLOC_CALL_KEYS,
    *SCOPED_ALLOCATOR_BYTE_KEYS,
    "responseBuilderItemsBuilt",
    "responseBuilderCapacityBytes",
    "responseStringBytes",
)


def cache_stats(client: JsonRpcClient) -> dict:
    try:
        result = client.request("workspace/executeCommand", {"command": "ora.cacheStats", "arguments": []})
    except Exception:
        return {}
    return result if isinstance(result, dict) else {}


def stat_int(stats: dict, key: str) -> int:
    value = stats.get(key, 0)
    return value if isinstance(value, int) else 0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def row_key(row: tuple[str, str]) -> str:
    return "%s.%s" % row


def load_p95_baseline(path: str) -> dict[tuple[str, str], float]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise RuntimeError("baseline p95 file must contain a JSON object")

    result: dict[tuple[str, str], float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise RuntimeError("baseline p95 keys must be strings")
        if not isinstance(value, (int, float)):
            raise RuntimeError("baseline p95 value for %s must be a number" % key)
        fixture, sep, operation = key.partition(".")
        if not sep or not fixture or not operation:
            raise RuntimeError("baseline p95 key %s must be fixture.operation" % key)
        result[(fixture, operation)] = float(value)
    return result


def peak_rss_kib() -> int:
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        return int(rss / 1024)
    return int(rss)


def response_has_payload(result) -> bool:
    if result is None:
        return False
    if isinstance(result, (list, tuple, str, bytes)):
        return len(result) > 0
    if isinstance(result, dict):
        if "data" in result and isinstance(result["data"], list):
            return len(result["data"]) > 0
        if "changes" in result and isinstance(result["changes"], dict):
            return len(result["changes"]) > 0
        return len(result) > 0
    return True


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def has_location_uri(result, uri: str) -> bool:
    return isinstance(result, list) and any(isinstance(item, dict) and item.get("uri") == uri for item in result)


def has_definition_uri(result, uri: str) -> bool:
    return isinstance(result, dict) and result.get("uri") == uri


def has_change_uri(result, uri: str) -> bool:
    return isinstance(result, dict) and isinstance(result.get("changes"), dict) and uri in result["changes"]


def has_call_target_uri(result, uri: str) -> bool:
    return isinstance(result, list) and any(
        isinstance(item, dict) and isinstance(item.get("to"), dict) and item["to"].get("uri") == uri
        for item in result
    )


def call_target_item(result, uri: str):
    if not isinstance(result, list):
        return None
    for item in result:
        if isinstance(item, dict) and isinstance(item.get("to"), dict) and item["to"].get("uri") == uri:
            return item["to"]
    return None


def has_incoming_call_from_uri(result, uri: str) -> bool:
    return isinstance(result, list) and any(
        isinstance(item, dict) and isinstance(item.get("from"), dict) and item["from"].get("uri") == uri
        for item in result
    )


def incoming_call_from_item(result, uri: str, name=None):
    if not isinstance(result, list):
        return None
    for item in result:
        if not isinstance(item, dict) or not isinstance(item.get("from"), dict):
            continue
        from_item = item["from"]
        if from_item.get("uri") != uri:
            continue
        if name is not None and from_item.get("name") != name:
            continue
        return from_item
    return None


def has_selection_range(result) -> bool:
    return isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict) and "range" in result[0]


def timed(
    samples: dict[tuple[str, str], list[float]],
    counter_deltas: dict[tuple[str, str, str], list[int]],
    client: JsonRpcClient,
    fixture: str,
    operation: str,
    func,
):
    before = cache_stats(client)
    start = time.perf_counter()
    result = func()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    after = cache_stats(client)
    samples.setdefault((fixture, operation), []).append(elapsed_ms)
    response_counter = RESPONSE_COUNTER_BY_OPERATION.get(operation)
    if response_counter and response_has_payload(result):
        expected_response_counters.add(response_counter)
    response_string_counter = RESPONSE_STRING_COUNTER_BY_OPERATION.get(operation)
    if response_string_counter and response_has_payload(result):
        expected_response_string_counters.add(response_string_counter)
    response_markdown_counter = RESPONSE_MARKDOWN_COUNTER_BY_OPERATION.get(operation)
    if response_markdown_counter and response_has_payload(result):
        expected_response_markdown_counters.add(response_markdown_counter)
    if before and after:
        for key in DELTA_COUNTER_KEYS:
            delta = stat_int(after, key) - stat_int(before, key)
            counter_deltas.setdefault((fixture, operation, key), []).append(delta)
    return result


def operation_delta_values(
    counter_deltas: dict[tuple[str, str, str], list[int]],
    fixture: str,
    operation: str,
    key: str,
) -> list[int]:
    return counter_deltas.get((fixture, operation, key), [])


def operation_delta_at(values: list[int], index: int) -> int:
    return values[index] if index < len(values) else 0


def allocation_attribution_for_row(
    counter_deltas: dict[tuple[str, str, str], list[int]],
    fixture: str,
    operation: str,
) -> tuple[int, int, int, int, int]:
    allocated_values = operation_delta_values(counter_deltas, fixture, operation, "serverAllocatorBytesAllocated")
    response_capacity_values = operation_delta_values(counter_deltas, fixture, operation, "responseBuilderCapacityBytes")
    response_string_values = operation_delta_values(counter_deltas, fixture, operation, "responseStringBytes")
    count = max(len(allocated_values), len(response_capacity_values), len(response_string_values))
    allocated_total = 0
    response_payload_total = 0
    unattributed_total = 0
    max_unattributed = 0
    for index in range(count):
        allocated = operation_delta_at(allocated_values, index)
        response_payload = (
            operation_delta_at(response_capacity_values, index) +
            operation_delta_at(response_string_values, index)
        )
        unattributed = max(0, allocated - response_payload)
        allocated_total += allocated
        response_payload_total += response_payload
        unattributed_total += unattributed
        max_unattributed = max(max_unattributed, unattributed)
    return count, allocated_total, response_payload_total, unattributed_total, max_unattributed


def allocation_attribution_budget(profile: str, row: tuple[str, str]) -> int | None:
    budgets = PROFILE_ALLOCATION_ATTRIBUTION_BUDGET_BYTES[profile]
    value = budgets.get(row, budgets.get("default"))
    return value if isinstance(value, int) else None


def make_source(functions: int) -> str:
    lines = [
        "// Internal helper note.",
        "/// Returns `value` unchanged.",
        "pub fn helper(value: u256) -> u256 {",
        "    return value;",
        "}",
        "",
    ]
    for index in range(functions):
        lines.extend(
            [
                f"pub fn generated_{index}(balance: u256, allowance: u256, amount: u256) -> u256 {{",
                f"    let local_{index}: u256 = helper(amount);",
                f"    return balance + allowance + local_{index};",
                "}",
                "",
            ]
        )
    lines.extend(
        [
            "pub fn run(amount: u256) -> u256 {",
            "    return helper(amount);",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def make_contract_suite(libs: int, helpers: int, abi_methods: int) -> dict[str, str]:
    files: dict[str, str] = {}
    for lib_index in range(libs):
        lines = [f"/// Shared arithmetic helpers for library {lib_index}."]
        for helper_index in range(helpers):
            lines.extend(
                [
                    f"pub fn shared{lib_index}_{helper_index}(balance: u256, allowance: u256, amount: u256) -> u256 {{",
                    "    return balance + allowance + amount;",
                    "}",
                    "",
                ]
            )
        files[f"lib{lib_index}.ora"] = "\n".join(lines)

    primary = []
    for lib_index in range(libs):
        primary.append(f"comptime const lib{lib_index} = @import(\"./lib{lib_index}.ora\");")
    primary.extend(
        [
            "",
            "contract BenchmarkToken {",
            "    error NotOwner(owner: address, spender: address);",
            "    log Transfer(owner: address, spender: address, amount: u256);",
            "    storage var balance: u256;",
            "    storage var allowance: u256;",
            "    storage var owner: address;",
            "",
        ]
    )
    for method_index in range(abi_methods):
        lib_index = method_index % max(1, libs)
        helper_index = method_index % max(1, helpers)
        primary.extend(
                [
                    f"    /// Public ABI method {method_index}.",
                    f"    pub fn method{method_index}(owner: address, spender: address, amount: u256) -> u256",
                    "        requires(amount >= 0)",
                    "        ensures(result >= amount)",
                    "    {",
                    f"        let balance{method_index}: u256 = lib{lib_index}.shared{lib_index}_{helper_index}(amount, amount, amount);",
                    f"        let allowance{method_index}: u256 = balance{method_index} + amount;",
                    f"        return allowance{method_index};",
                "    }",
                "",
            ]
        )
    primary.append("}")
    files["contractsuite.ora"] = "\n".join(primary)
    return files


def make_cold_imported_member_suite(uses: int) -> dict[str, str]:
    target = "\n".join(
        [
            "/// Cold imported member used from an unopened file.",
            "pub fn cold_shared(value: u256) -> u256 {",
            "    return value;",
            "}",
            "",
        ]
    )
    importer = [
        'comptime const cold = @import("./cold_target.ora");',
        "",
        "contract ColdImporter {",
    ]
    for index in range(uses):
        importer.extend(
            [
                f"    pub fn use_cold_{index}(amount: u256) -> u256 {{",
                f"        let value{index}: u256 = cold.cold_shared(amount);",
                f"        return value{index};",
                "    }",
                "",
            ]
        )
    importer.append("}")
    upstream = "\n".join(
        [
            'comptime const coldimporter = @import("./cold_importer.ora");',
            "",
            "/// Recursive cold incoming caller.",
            "pub fn upstream_use_cold(amount: u256) -> u256 {",
            "    return coldimporter.use_cold_0(amount);",
            "}",
            "",
        ]
    )
    return {
        "cold_target.ora": target,
        "cold_importer.ora": "\n".join(importer),
        "cold_upstream.ora": upstream,
    }


def make_cold_call_hierarchy_suite() -> dict[str, str]:
    leaf = "\n".join(
        [
            "/// Recursive cold outgoing call leaf.",
            "pub fn cold_call_leaf(value: u256) -> u256 {",
            "    return value;",
            "}",
            "",
        ]
    )
    target = "\n".join(
        [
            'comptime const coldleaf = @import("./cold_call_leaf.ora");',
            "",
            "/// Cold outgoing call target used from an open importer.",
            "pub fn cold_call_target(value: u256) -> u256 {",
            "    return coldleaf.cold_call_leaf(value);",
            "}",
            "",
        ]
    )
    importer = "\n".join(
        [
            'comptime const coldcall = @import("./cold_call_target.ora");',
            "",
            "contract ColdCallImporter {",
            "    pub fn use_cold_call(amount: u256) -> u256 {",
            "        return coldcall.cold_call_target(amount);",
            "    }",
            "}",
            "",
        ]
    )
    return {
        "cold_call_leaf.ora": leaf,
        "cold_call_target.ora": target,
        "cold_call_importer.ora": importer,
    }


def make_import_fanout_suite(imports: int) -> dict[str, str]:
    files: dict[str, str] = {}
    for index in range(imports):
        files[f"leaf{index}.ora"] = "\n".join(
            [
                f"/// Fan-out leaf {index}.",
                f"pub fn fanout_leaf_{index}(value: u256) -> u256 {{",
                "    return value;",
                "}",
                "",
            ]
        )

    hub = []
    for index in range(imports):
        hub.append(f"comptime const leaf{index} = @import(\"./leaf{index}.ora\");")
    hub.extend(["", "contract FanOutHub {"])
    for index in range(imports):
        hub.extend(
            [
                f"    pub fn use_leaf_{index}(amount: u256) -> u256 {{",
                f"        return leaf{index}.fanout_leaf_{index}(amount);",
                "    }",
                "",
            ]
        )
    hub.append("}")
    files["hub.ora"] = "\n".join(hub)
    return files


def make_import_fanin_suite(importers: int) -> dict[str, str]:
    files = {
        "shared.ora": "\n".join(
            [
                "/// Shared fan-in target.",
                "pub fn fanin_shared(value: u256) -> u256 {",
                "    return value;",
                "}",
                "",
            ]
        )
    }
    for index in range(importers):
        files[f"user{index}.ora"] = "\n".join(
            [
                'comptime const shared = @import("./shared.ora");',
                "",
                f"contract FanInUser{index} {{",
                f"    pub fn call_shared_{index}(amount: u256) -> u256 {{",
                "        return shared.fanin_shared(amount);",
                "    }",
                "}",
                "",
            ]
        )
    return files


def make_position_lookup_source(lines: int) -> str:
    total_lines = max(4, lines)
    result = [
        "pub fn anchor() -> u256 { return 0; }",
    ]
    for index in range(1, total_lines):
        result.append("// position lookup filler line %06d" % index)
    return "\n".join(result)


def make_stress_source(symbols: int, occurrences: int) -> str:
    lines = [
        "/// Hot target used by repeated occurrence stress.",
        "pub fn hot_target(value: u256) -> u256 {",
        "    return value;",
        "}",
        "",
    ]
    for index in range(symbols):
        lines.extend(
            [
                f"pub fn symbol_{index}(value: u256) -> u256 {{",
                "    return hot_target(value);",
                "}",
                "",
            ]
        )
    lines.extend(["pub fn occurrence_driver(value: u256) -> u256 {", "    let total: u256 = value;"])
    for index in range(occurrences):
        lines.append(f"    let repeated_{index}: u256 = hot_target(total);")
    lines.append("    return total;")
    lines.append("}")
    return "\n".join(lines)


def write_fixture(root: Path, rel_path: str, source: str) -> str:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return file_uri(path)


def fixture_stats(name: str, sources: dict[str, str], public_functions: int, extra: int = 0) -> None:
    sizes = [len(source.encode("utf-8")) for source in sources.values()]
    print(
        "fixture_stats %s %d %d %d %d %d"
        % (name, len(sources), sum(sizes), max(sizes) if sizes else 0, public_functions, extra)
    )


def request_text_document(client: JsonRpcClient, method: str, uri: str, params: dict | None = None):
    payload = {"textDocument": {"uri": uri}}
    if params:
        payload.update(params)
    return client.request(method, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Ora LSP JSON-RPC benchmark.")
    parser.add_argument("server_positional", nargs="?")
    parser.add_argument("--server")
    parser.add_argument("--profile", choices=("quick", "release"), default="quick")
    parser.add_argument("--build-mode", default="")
    parser.add_argument("--require-build-mode")
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Timed samples per operation. The default keeps p95 from degenerating into the max sample.",
    )
    parser.add_argument("--functions", type=int, default=80)
    parser.add_argument("--contract-libs", type=int, default=3)
    parser.add_argument("--contract-helpers", type=int, default=8)
    parser.add_argument("--contract-abi-methods", type=int, default=24)
    parser.add_argument("--cold-import-uses", type=int, default=24)
    parser.add_argument("--import-fanout-files", type=int, default=8)
    parser.add_argument("--import-fanin-files", type=int, default=8)
    parser.add_argument("--position-small-lines", type=int, default=64)
    parser.add_argument("--position-large-lines", type=int, default=8192)
    parser.add_argument("--stress-symbols", type=int, default=80)
    parser.add_argument("--stress-occurrences", type=int, default=160)
    parser.add_argument("--max-request-p95-ms", type=float, default=500.0)
    parser.add_argument("--max-edit-diagnostics-p95-ms", type=float, default=1000.0)
    parser.add_argument("--max-total-source-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-open-source-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-cold-source-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-tracked-cache-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tracked-total-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-peak-rss-kib", type=int, default=256 * 1024)
    parser.add_argument("--max-position-lookup-p95-ms", type=float, default=10.0)
    parser.add_argument("--max-position-lookup-scale-ratio", type=float, default=8.0)
    parser.add_argument("--position-lookup-scale-slop-ms", type=float, default=1.0)
    parser.add_argument(
        "--baseline-p95-file",
        help="JSON object keyed by fixture.operation with baseline p95 milliseconds.",
    )
    parser.add_argument(
        "--min-p95-improvement",
        type=float,
        default=0.15,
        help="Required fractional p95 improvement when --baseline-p95-file is provided.",
    )
    parser.add_argument(
        "--require-baseline-for-all-rows",
        action="store_true",
        help="Fail if --baseline-p95-file omits any observed benchmark row.",
    )
    parser.add_argument(
        "--write-p95-file",
        help="Write observed p95 milliseconds as a fixture.operation JSON object.",
    )
    parser.add_argument(
        "--skip-profile-p95-budgets",
        action="store_true",
        help="Skip the per-fixture/operation p95 budget gate for ad hoc scaled benchmark runs.",
    )
    parser.add_argument(
        "--skip-profile-byte-budgets",
        action="store_true",
        help="Skip the profile-specific source/cache/RSS byte budget gate for ad hoc scaled benchmark runs.",
    )
    parser.add_argument(
        "--skip-profile-allocation-budgets",
        action="store_true",
        help="Skip the profile-specific allocator-attribution budget gate for ad hoc scaled benchmark runs.",
    )
    parser.add_argument(
        "--strict-future-gates",
        action="store_true",
        help="Fail gates for response attribution, scoped allocator attribution, and zero-delta cache-hit rows.",
    )
    args = parser.parse_args()

    server = args.server or args.server_positional
    if server is None:
        parser.error("missing LSP server path")
    if args.require_build_mode and args.build_mode != args.require_build_mode:
        parser.error("--require-build-mode expected %s, got %s" % (args.require_build_mode, args.build_mode or "<unset>"))
    if args.min_p95_improvement < 0.0 or args.min_p95_improvement >= 1.0:
        parser.error("--min-p95-improvement must be >= 0.0 and < 1.0")
    if args.profile == "release":
        args.functions = max(args.functions, 180)
        args.contract_libs = max(args.contract_libs, 6)
        args.contract_helpers = max(args.contract_helpers, 16)
        args.contract_abi_methods = max(args.contract_abi_methods, 72)
        args.cold_import_uses = max(args.cold_import_uses, 96)
        args.import_fanout_files = max(args.import_fanout_files, 24)
        args.import_fanin_files = max(args.import_fanin_files, 24)
        args.position_small_lines = max(args.position_small_lines, 64)
        args.position_large_lines = max(args.position_large_lines, 32768)
        args.stress_symbols = max(args.stress_symbols, 220)
        args.stress_occurrences = max(args.stress_occurrences, 640)
    for name in ("iterations", "functions", "contract_libs", "contract_helpers", "contract_abi_methods", "cold_import_uses", "import_fanout_files", "import_fanin_files", "position_small_lines", "position_large_lines", "stress_symbols", "stress_occurrences"):
        if getattr(args, name) < 1:
            parser.error("--%s must be positive" % name.replace("_", "-"))

    samples: dict[tuple[str, str], list[float]] = {}
    counter_deltas: dict[tuple[str, str, str], list[int]] = {}
    source_bytes = 0

    with tempfile.TemporaryDirectory(prefix="ora-lsp-bench-") as tmp:
        root = Path(tmp)
        bench_source = make_source(args.functions)
        contract_sources = make_contract_suite(args.contract_libs, args.contract_helpers, args.contract_abi_methods)
        cold_sources = make_cold_imported_member_suite(args.cold_import_uses)
        cold_call_sources = make_cold_call_hierarchy_suite()
        import_fanout_sources = make_import_fanout_suite(args.import_fanout_files)
        import_fanin_sources = make_import_fanin_suite(args.import_fanin_files)
        position_small_source = make_position_lookup_source(args.position_small_lines)
        position_large_source = make_position_lookup_source(args.position_large_lines)
        stress_source = make_stress_source(args.stress_symbols, args.stress_occurrences)
        broken_source = "pub fn broken() -> u256 {\n    return 1\n}\n"

        opened_sources = {
            "bench.ora": bench_source,
            **{f"contracts/{path}": src for path, src in contract_sources.items()},
            **{f"importfanout/{path}": src for path, src in import_fanout_sources.items()},
            **{f"importfanin/{path}": src for path, src in import_fanin_sources.items()},
            "cold/cold_target.ora": cold_sources["cold_target.ora"],
            "cold_call/cold_call_importer.ora": cold_call_sources["cold_call_importer.ora"],
            "positionsmall.ora": position_small_source,
            "positionlarge.ora": position_large_source,
            "stress.ora": stress_source,
            "broken.ora": broken_source,
        }
        cold_disk_sources = {
            "cold/cold_importer.ora": cold_sources["cold_importer.ora"],
            "cold/cold_upstream.ora": cold_sources["cold_upstream.ora"],
            "cold_call/cold_call_leaf.ora": cold_call_sources["cold_call_leaf.ora"],
            "cold_call/cold_call_target.ora": cold_call_sources["cold_call_target.ora"],
        }
        all_sources = {**opened_sources, **cold_disk_sources}
        uris = {rel: write_fixture(root, rel, source) for rel, source in all_sources.items()}
        source_bytes = sum(len(source.encode("utf-8")) for source in all_sources.values())

        client = JsonRpcClient(server)
        stats = {}
        try:
            initialize(client, file_uri(root))
            diagnostics_by_uri = {}
            for rel, source in opened_sources.items():
                did_open(client, uris[rel], source)
                diagnostics_by_uri[uris[rel]] = wait_diagnostics(client, uris[rel])

            bench_uri = uris["bench.ora"]
            contract_uri = uris["contracts/contractsuite.ora"]
            lib0_uri = uris["contracts/lib0.ora"]
            cold_target_uri = uris["cold/cold_target.ora"]
            cold_importer_uri = uris["cold/cold_importer.ora"]
            cold_upstream_uri = uris["cold/cold_upstream.ora"]
            cold_call_importer_uri = uris["cold_call/cold_call_importer.ora"]
            cold_call_leaf_uri = uris["cold_call/cold_call_leaf.ora"]
            cold_call_target_uri = uris["cold_call/cold_call_target.ora"]
            fanout_hub_uri = uris["importfanout/hub.ora"]
            fanout_leaf0_uri = uris["importfanout/leaf0.ora"]
            fanin_shared_uri = uris["importfanin/shared.ora"]
            position_small_uri = uris["positionsmall.ora"]
            position_large_uri = uris["positionlarge.ora"]
            stress_uri = uris["stress.ora"]
            broken_uri = uris["broken.ora"]
            broken_diagnostics = diagnostics_by_uri.get(broken_uri, [])
            current_bench_source = bench_source
            contract_imported_member_call = position_of(contract_sources["contractsuite.ora"], "lib0.shared0_0", 0)
            lib0_shared0_decl = position_of(contract_sources["lib0.ora"], "shared0_0", 0)
            cold_shared_decl = position_of(cold_sources["cold_target.ora"], "cold_shared", 0)
            cold_call_method_decl = position_of(cold_call_sources["cold_call_importer.ora"], "use_cold_call", 0)
            fanout_leaf0_decl = position_of(import_fanout_sources["leaf0.ora"], "fanout_leaf_0", 0)
            fanin_shared_decl = position_of(import_fanin_sources["shared.ora"], "fanin_shared", 0)
            position_lookup_cases = (
                ("positionsmall", position_small_uri, args.position_small_lines),
                ("positionlarge", position_large_uri, args.position_large_lines),
            )

            for iteration in range(args.iterations):
                bench_helper_decl = position_of(current_bench_source, "helper", 0)
                bench_helper_call = position_of(current_bench_source, "helper(amount)", 0)
                bench_helper_signature = position_of(current_bench_source, "helper(amount)", 0, len("helper("))

                request_text_document(client, "textDocument/hover", bench_uri, {"position": bench_helper_decl})
                timed(samples, counter_deltas, client, "bench", "hover.cache_hit", lambda: request_text_document(client, "textDocument/hover", bench_uri, {"position": bench_helper_decl}))
                request_text_document(client, "textDocument/completion", bench_uri, {"position": position_of(current_bench_source, "helper(amount)", 0, 3), "context": {"triggerKind": 1}})
                timed(samples, counter_deltas, client, "bench", "completion.cache_hit", lambda: request_text_document(client, "textDocument/completion", bench_uri, {"position": position_of(current_bench_source, "helper(amount)", 0, 3), "context": {"triggerKind": 1}}))
                request_text_document(client, "textDocument/definition", bench_uri, {"position": bench_helper_call})
                timed(samples, counter_deltas, client, "bench", "definition.cache_hit", lambda: request_text_document(client, "textDocument/definition", bench_uri, {"position": bench_helper_call}))
                request_text_document(client, "textDocument/references", bench_uri, {"position": bench_helper_call, "context": {"includeDeclaration": True}})
                timed(samples, counter_deltas, client, "bench", "references.cache_hit", lambda: request_text_document(client, "textDocument/references", bench_uri, {"position": bench_helper_call, "context": {"includeDeclaration": True}}))
                request_text_document(client, "textDocument/documentSymbol", bench_uri)
                timed(samples, counter_deltas, client, "bench", "documentSymbol.cache_hit", lambda: request_text_document(client, "textDocument/documentSymbol", bench_uri))
                client.request("workspace/symbol", {"query": "helper"})
                timed(samples, counter_deltas, client, "bench", "workspaceSymbol.cache_hit", lambda: client.request("workspace/symbol", {"query": "helper"}))
                request_text_document(client, "textDocument/semanticTokens/full", bench_uri)
                timed(samples, counter_deltas, client, "bench", "semanticTokens.cache_hit", lambda: request_text_document(client, "textDocument/semanticTokens/full", bench_uri))
                request_text_document(client, "textDocument/documentHighlight", bench_uri, {"position": bench_helper_call})
                timed(samples, counter_deltas, client, "bench", "documentHighlight.cache_hit", lambda: request_text_document(client, "textDocument/documentHighlight", bench_uri, {"position": bench_helper_call}))
                request_text_document(client, "textDocument/signatureHelp", bench_uri, {"position": bench_helper_signature})
                timed(samples, counter_deltas, client, "bench", "signatureHelp.cache_hit", lambda: request_text_document(client, "textDocument/signatureHelp", bench_uri, {"position": bench_helper_signature}))
                request_text_document(client, "textDocument/inlayHint", bench_uri, {"range": full_range(current_bench_source)})
                timed(samples, counter_deltas, client, "bench", "inlayHint.cache_hit", lambda: request_text_document(client, "textDocument/inlayHint", bench_uri, {"range": full_range(current_bench_source)}))
                request_text_document(client, "textDocument/foldingRange", bench_uri)
                timed(samples, counter_deltas, client, "bench", "foldingRange.cache_hit", lambda: request_text_document(client, "textDocument/foldingRange", bench_uri))
                request_text_document(client, "textDocument/selectionRange", bench_uri, {"positions": [bench_helper_call]})
                timed(samples, counter_deltas, client, "bench", "selectionRange.cache_hit", lambda: request_text_document(client, "textDocument/selectionRange", bench_uri, {"positions": [bench_helper_call]}))
                for position_fixture, position_uri, position_lines in position_lookup_cases:
                    position_rows = (
                        ("top", {"line": min(1, position_lines - 1), "character": 2}),
                        ("middle", {"line": max(1, position_lines // 2), "character": 2}),
                        ("bottom", {"line": position_lines - 1, "character": 2}),
                    )
                    for position_name, lookup_position in position_rows:
                        request_text_document(client, "textDocument/selectionRange", position_uri, {"positions": [lookup_position]})
                        position_result = timed(
                            samples,
                            counter_deltas,
                            client,
                            position_fixture,
                            "positionLookup.%s.cache_hit" % position_name,
                            lambda uri=position_uri, pos=lookup_position: request_text_document(client, "textDocument/selectionRange", uri, {"positions": [pos]}),
                        )
                        require(has_selection_range(position_result), "position lookup selectionRange returned no range")
                request_text_document(client, "textDocument/prepareRename", bench_uri, {"position": bench_helper_call})
                timed(samples, counter_deltas, client, "bench", "prepareRename.cache_hit", lambda: request_text_document(client, "textDocument/prepareRename", bench_uri, {"position": bench_helper_call}))
                request_text_document(client, "textDocument/rename", bench_uri, {"position": bench_helper_call, "newName": "helperRenamed"})
                timed(samples, counter_deltas, client, "bench", "rename.cache_hit", lambda: request_text_document(client, "textDocument/rename", bench_uri, {"position": bench_helper_call, "newName": "helperRenamed"}))
                request_text_document(client, "textDocument/formatting", bench_uri, {"options": {"tabSize": 4, "insertSpaces": True}})
                timed(samples, counter_deltas, client, "bench", "formatting.cache_hit", lambda: request_text_document(client, "textDocument/formatting", bench_uri, {"options": {"tabSize": 4, "insertSpaces": True}}))

                edit_text = f"\n// edit iteration {iteration}\n"
                edit_position = {"line": 0, "character": 0}
                timed(
                    samples,
                    counter_deltas,
                    client,
                    "bench",
                    "didChange.diagnostics",
                    lambda: (
                        did_change_incremental(
                            client,
                            bench_uri,
                            {"start": edit_position, "end": edit_position},
                            edit_text,
                            iteration + 2,
                        ),
                        wait_diagnostics(client, bench_uri),
                    ),
                )
                current_bench_source = edit_text + current_bench_source

                request_text_document(client, "textDocument/documentSymbol", contract_uri)
                timed(samples, counter_deltas, client, "contractsuite", "documentSymbol.cache_hit", lambda: request_text_document(client, "textDocument/documentSymbol", contract_uri))
                request_text_document(client, "textDocument/documentLink", contract_uri)
                timed(samples, counter_deltas, client, "contractsuite", "documentLink.cache_hit", lambda: request_text_document(client, "textDocument/documentLink", contract_uri))
                client.request("workspace/symbol", {"query": "method"})
                timed(samples, counter_deltas, client, "contractsuite", "workspaceSymbol.cache_hit", lambda: client.request("workspace/symbol", {"query": "method"}))
                request_text_document(client, "textDocument/semanticTokens/full", contract_uri)
                timed(samples, counter_deltas, client, "contractsuite", "semanticTokens.cache_hit", lambda: request_text_document(client, "textDocument/semanticTokens/full", contract_uri))
                request_text_document(client, "textDocument/hover", contract_uri, {"position": position_of(contract_sources["contractsuite.ora"], "method0", 0)})
                timed(samples, counter_deltas, client, "contractsuite", "hover.cache_hit", lambda: request_text_document(client, "textDocument/hover", contract_uri, {"position": position_of(contract_sources["contractsuite.ora"], "method0", 0)}))
                request_text_document(client, "textDocument/completion", contract_uri, {"position": position_of(contract_sources["contractsuite.ora"], "method0", 0, 3), "context": {"triggerKind": 1}})
                timed(samples, counter_deltas, client, "contractsuite", "completion.cache_hit", lambda: request_text_document(client, "textDocument/completion", contract_uri, {"position": position_of(contract_sources["contractsuite.ora"], "method0", 0, 3), "context": {"triggerKind": 1}}))
                request_text_document(client, "textDocument/definition", contract_uri, {"position": contract_imported_member_call})
                imported_definition = timed(samples, counter_deltas, client, "contractsuite", "definition.imported_member.cache_hit", lambda: request_text_document(client, "textDocument/definition", contract_uri, {"position": contract_imported_member_call}))
                require(has_definition_uri(imported_definition, lib0_uri), "timed imported-member definition did not resolve to imported library")
                request_text_document(client, "textDocument/codeLens", contract_uri)
                timed(samples, counter_deltas, client, "contractsuite", "codeLens.cache_hit", lambda: request_text_document(client, "textDocument/codeLens", contract_uri))
                request_text_document(client, "textDocument/foldingRange", contract_uri)
                timed(samples, counter_deltas, client, "contractsuite", "foldingRange.cache_hit", lambda: request_text_document(client, "textDocument/foldingRange", contract_uri))
                request_text_document(client, "textDocument/references", lib0_uri, {"position": lib0_shared0_decl, "context": {"includeDeclaration": True}})
                timed(samples, counter_deltas, client, "contractsuite", "references.imported_member.cache_hit", lambda: request_text_document(client, "textDocument/references", lib0_uri, {"position": lib0_shared0_decl, "context": {"includeDeclaration": True}}))
                request_text_document(client, "textDocument/rename", lib0_uri, {"position": lib0_shared0_decl, "newName": "shared0Renamed"})
                timed(samples, counter_deltas, client, "contractsuite", "rename.imported_member.cache_hit", lambda: request_text_document(client, "textDocument/rename", lib0_uri, {"position": lib0_shared0_decl, "newName": "shared0Renamed"}))
                request_text_document(client, "textDocument/documentSymbol", fanout_hub_uri)
                timed(samples, counter_deltas, client, "importfanout", "documentSymbol.cache_hit", lambda: request_text_document(client, "textDocument/documentSymbol", fanout_hub_uri))
                client.request("workspace/symbol", {"query": "FanOut"})
                timed(samples, counter_deltas, client, "importfanout", "workspaceSymbol.cache_hit", lambda: client.request("workspace/symbol", {"query": "FanOut"}))
                fanout_refs_warm = request_text_document(client, "textDocument/references", fanout_leaf0_uri, {"position": fanout_leaf0_decl, "context": {"includeDeclaration": True}})
                require(has_location_uri(fanout_refs_warm, fanout_hub_uri), "warm import fan-out references did not include hub use")
                fanout_refs = timed(samples, counter_deltas, client, "importfanout", "references.imported_member.cache_hit", lambda: request_text_document(client, "textDocument/references", fanout_leaf0_uri, {"position": fanout_leaf0_decl, "context": {"includeDeclaration": True}}))
                require(has_location_uri(fanout_refs, fanout_hub_uri), "timed import fan-out references did not include hub use")
                client.request("workspace/symbol", {"query": "FanIn"})
                timed(samples, counter_deltas, client, "importfanin", "workspaceSymbol.cache_hit", lambda: client.request("workspace/symbol", {"query": "FanIn"}))
                fanin_refs_warm = request_text_document(client, "textDocument/references", fanin_shared_uri, {"position": fanin_shared_decl, "context": {"includeDeclaration": True}})
                require(
                    isinstance(fanin_refs_warm, list) and len(fanin_refs_warm) >= args.import_fanin_files,
                    "warm import fan-in references did not include expected importer uses",
                )
                fanin_refs = timed(samples, counter_deltas, client, "importfanin", "references.imported_member.cache_hit", lambda: request_text_document(client, "textDocument/references", fanin_shared_uri, {"position": fanin_shared_decl, "context": {"includeDeclaration": True}}))
                require(
                    isinstance(fanin_refs, list) and len(fanin_refs) >= args.import_fanin_files,
                    "timed import fan-in references did not include expected importer uses",
                )
                fanin_rename_warm = request_text_document(client, "textDocument/rename", fanin_shared_uri, {"position": fanin_shared_decl, "newName": "fanin_shared_renamed"})
                require(
                    isinstance(fanin_rename_warm, dict) and isinstance(fanin_rename_warm.get("changes"), dict) and len(fanin_rename_warm["changes"]) >= args.import_fanin_files,
                    "warm import fan-in rename did not include expected importer edits",
                )
                fanin_rename = timed(samples, counter_deltas, client, "importfanin", "rename.imported_member.cache_hit", lambda: request_text_document(client, "textDocument/rename", fanin_shared_uri, {"position": fanin_shared_decl, "newName": "fanin_shared_renamed"}))
                require(
                    isinstance(fanin_rename, dict) and isinstance(fanin_rename.get("changes"), dict) and len(fanin_rename["changes"]) >= args.import_fanin_files,
                    "timed import fan-in rename did not include expected importer edits",
                )
                cold_incoming_items = request_text_document(client, "textDocument/prepareCallHierarchy", cold_target_uri, {"position": cold_shared_decl})
                require(bool(cold_incoming_items), "cold incoming prepare call hierarchy did not return target method")
                cold_incoming_item = cold_incoming_items[0]

                cold_refs_first = timed(samples, counter_deltas, client, "cold", "references.cold_imported_member.first_build", lambda: request_text_document(client, "textDocument/references", cold_target_uri, {"position": cold_shared_decl, "context": {"includeDeclaration": True}}))
                require(has_location_uri(cold_refs_first, cold_importer_uri), "first-build cold references did not include unopened importer")
                cold_refs = timed(samples, counter_deltas, client, "cold", "references.cold_imported_member.cache_hit", lambda: request_text_document(client, "textDocument/references", cold_target_uri, {"position": cold_shared_decl, "context": {"includeDeclaration": True}}))
                require(has_location_uri(cold_refs, cold_importer_uri), "timed cold references did not include unopened importer")
                cold_rename_warm = request_text_document(client, "textDocument/rename", cold_target_uri, {"position": cold_shared_decl, "newName": "coldSharedRenamed"})
                require(has_change_uri(cold_rename_warm, cold_importer_uri), "warm cold rename did not include unopened importer edit")
                cold_rename = timed(samples, counter_deltas, client, "cold", "rename.cold_imported_member.cache_hit", lambda: request_text_document(client, "textDocument/rename", cold_target_uri, {"position": cold_shared_decl, "newName": "coldSharedRenamed"}))
                require(has_change_uri(cold_rename, cold_importer_uri), "timed cold rename did not include unopened importer edit")

                cold_incoming_first = timed(samples, counter_deltas, client, "cold", "callHierarchy.incoming.cold_imported_member.first_build", lambda: client.request("callHierarchy/incomingCalls", {"item": cold_incoming_item}))
                require(has_incoming_call_from_uri(cold_incoming_first, cold_importer_uri), "first-build cold incoming call hierarchy did not include unopened importer")
                cold_importer_item = incoming_call_from_item(cold_incoming_first, cold_importer_uri, "use_cold_0")
                require(isinstance(cold_importer_item, dict), "first-build cold incoming call hierarchy did not return the recursive importer item")
                cold_recursive_incoming_first = timed(samples, counter_deltas, client, "cold", "callHierarchy.incoming.recursive_cold_imported_member.first_build", lambda: client.request("callHierarchy/incomingCalls", {"item": cold_importer_item}))
                require(has_incoming_call_from_uri(cold_recursive_incoming_first, cold_upstream_uri), "first-build recursive cold incoming call hierarchy did not include unopened upstream importer")
                cold_incoming = timed(samples, counter_deltas, client, "cold", "callHierarchy.incoming.cold_imported_member.cache_hit", lambda: client.request("callHierarchy/incomingCalls", {"item": cold_incoming_item}))
                require(has_incoming_call_from_uri(cold_incoming, cold_importer_uri), "timed cold incoming call hierarchy did not include unopened importer")
                cold_recursive_incoming = timed(samples, counter_deltas, client, "cold", "callHierarchy.incoming.recursive_cold_imported_member.cache_hit", lambda: client.request("callHierarchy/incomingCalls", {"item": cold_importer_item}))
                require(has_incoming_call_from_uri(cold_recursive_incoming, cold_upstream_uri), "timed recursive cold incoming call hierarchy did not include unopened upstream importer")

                cold_call_items = request_text_document(client, "textDocument/prepareCallHierarchy", cold_call_importer_uri, {"position": cold_call_method_decl})
                require(bool(cold_call_items), "cold outgoing prepare call hierarchy did not return importer method")
                cold_call_item = cold_call_items[0]
                cold_outgoing_first = timed(samples, counter_deltas, client, "cold", "callHierarchy.outgoing.cold_imported_target.first_build", lambda: client.request("callHierarchy/outgoingCalls", {"item": cold_call_item}))
                require(has_call_target_uri(cold_outgoing_first, cold_call_target_uri), "first-build cold outgoing call hierarchy did not include unopened target")
                cold_call_target_item = call_target_item(cold_outgoing_first, cold_call_target_uri)
                require(isinstance(cold_call_target_item, dict), "first-build cold outgoing call hierarchy did not return a target item")
                cold_recursive_outgoing_first = timed(samples, counter_deltas, client, "cold", "callHierarchy.outgoing.recursive_cold_imported_target.first_build", lambda: client.request("callHierarchy/outgoingCalls", {"item": cold_call_target_item}))
                require(has_call_target_uri(cold_recursive_outgoing_first, cold_call_leaf_uri), "first-build recursive cold outgoing call hierarchy did not include unopened leaf")
                cold_outgoing = timed(samples, counter_deltas, client, "cold", "callHierarchy.outgoing.cold_imported_target.cache_hit", lambda: client.request("callHierarchy/outgoingCalls", {"item": cold_call_item}))
                require(has_call_target_uri(cold_outgoing, cold_call_target_uri), "timed cold outgoing call hierarchy did not include unopened target")
                cold_recursive_outgoing = timed(samples, counter_deltas, client, "cold", "callHierarchy.outgoing.recursive_cold_imported_target.cache_hit", lambda: client.request("callHierarchy/outgoingCalls", {"item": cold_call_target_item}))
                require(has_call_target_uri(cold_recursive_outgoing, cold_call_leaf_uri), "timed recursive cold outgoing call hierarchy did not include unopened leaf")
                warm_call_items = request_text_document(client, "textDocument/prepareCallHierarchy", contract_uri, {"position": position_of(contract_sources["contractsuite.ora"], "method0", 0)})
                require(bool(warm_call_items), "warm prepare call hierarchy did not return method0")
                warm_call_item = warm_call_items[0]
                client.request("callHierarchy/incomingCalls", {"item": warm_call_item})
                warm_outgoing = client.request("callHierarchy/outgoingCalls", {"item": warm_call_item})
                require(has_call_target_uri(warm_outgoing, lib0_uri), "warm outgoing call hierarchy did not include imported library target")
                call_items = timed(samples, counter_deltas, client, "contractsuite", "callHierarchy.prepare.cache_hit", lambda: request_text_document(client, "textDocument/prepareCallHierarchy", contract_uri, {"position": position_of(contract_sources["contractsuite.ora"], "method0", 0)}))
                require(bool(call_items), "timed prepare call hierarchy did not return method0")
                call_item = call_items[0]
                timed(samples, counter_deltas, client, "contractsuite", "callHierarchy.incoming.cache_hit", lambda: client.request("callHierarchy/incomingCalls", {"item": call_item}))
                outgoing = timed(samples, counter_deltas, client, "contractsuite", "callHierarchy.outgoing.cache_hit", lambda: client.request("callHierarchy/outgoingCalls", {"item": call_item}))
                require(has_call_target_uri(outgoing, lib0_uri), "timed outgoing call hierarchy did not include imported library target")

                request_text_document(client, "textDocument/documentSymbol", stress_uri)
                timed(samples, counter_deltas, client, "stress", "documentSymbol.cache_hit", lambda: request_text_document(client, "textDocument/documentSymbol", stress_uri))
                client.request("workspace/symbol", {"query": "symbol"})
                timed(samples, counter_deltas, client, "stress", "workspaceSymbol.cache_hit", lambda: client.request("workspace/symbol", {"query": "symbol"}))
                request_text_document(client, "textDocument/references", stress_uri, {"position": position_of(stress_source, "hot_target(value)", 0), "context": {"includeDeclaration": True}})
                timed(samples, counter_deltas, client, "stress", "references.cache_hit", lambda: request_text_document(client, "textDocument/references", stress_uri, {"position": position_of(stress_source, "hot_target(value)", 0), "context": {"includeDeclaration": True}}))
                request_text_document(client, "textDocument/semanticTokens/full", stress_uri)
                timed(samples, counter_deltas, client, "stress", "semanticTokens.cache_hit", lambda: request_text_document(client, "textDocument/semanticTokens/full", stress_uri))

                request_text_document(client, "textDocument/codeAction", broken_uri, {"range": full_range(broken_source), "context": {"diagnostics": broken_diagnostics, "only": ["quickfix"]}})
                timed(samples, counter_deltas, client, "code_action", "codeAction.quickfix.cache_hit", lambda: request_text_document(client, "textDocument/codeAction", broken_uri, {"range": full_range(broken_source), "context": {"diagnostics": broken_diagnostics, "only": ["quickfix"]}}))

            stats = cache_stats(client)
        finally:
            client.close()

    print("fixture operation count p50_ms p95_ms max_ms")
    request_ok = True
    edit_ok = True
    p95_by_row: dict[tuple[str, str], float] = {}
    for (fixture, operation), values in sorted(samples.items()):
        p50 = statistics.median(values)
        p95 = percentile(values, 0.95)
        max_v = max(values)
        p95_by_row[(fixture, operation)] = p95
        print("%s %s %d %.3f %.3f %.3f" % (fixture, operation, len(values), p50, p95, max_v))
        if operation == "didChange.diagnostics":
            edit_ok = edit_ok and p95 <= args.max_edit_diagnostics_p95_ms
        else:
            request_ok = request_ok and p95 <= args.max_request_p95_ms

    if args.write_p95_file:
        observed_p95 = {row_key(row): p95_by_row[row] for row in sorted(p95_by_row.keys())}
        Path(args.write_p95_file).write_text(json.dumps(observed_p95, indent=2, sort_keys=True) + "\n")

    profile_budget_failures = []
    if not args.skip_profile_p95_budgets:
        profile_budgets = PROFILE_P95_BUDGET_MS[args.profile]
        for row, p95 in sorted(p95_by_row.items()):
            budget = profile_budgets.get(row)
            if budget is None:
                profile_budget_failures.append("%s.%s:missing-budget" % row)
            elif p95 > budget:
                profile_budget_failures.append("%s.%s %.3f > %.3f" % (row[0], row[1], p95, budget))
        for row in sorted(profile_budgets.keys()):
            if row not in p95_by_row:
                profile_budget_failures.append("%s.%s:missing-row" % row)
    profile_budget_ok = args.skip_profile_p95_budgets or len(profile_budget_failures) == 0

    baseline_p95_failures = []
    baseline_p95_uncovered_rows = []
    baseline_p95_count = 0
    if args.baseline_p95_file:
        baseline_p95 = load_p95_baseline(args.baseline_p95_file)
        baseline_p95_count = len(baseline_p95)
        for row, baseline in sorted(baseline_p95.items()):
            p95 = p95_by_row.get(row)
            if p95 is None:
                baseline_p95_failures.append("%s:missing-row" % row_key(row))
                continue
            target = baseline * (1.0 - args.min_p95_improvement)
            if p95 > target:
                baseline_p95_failures.append(
                    "%s %.3f > %.3f baseline %.3f"
                    % (row_key(row), p95, target, baseline)
                )
        baseline_p95_uncovered_rows = sorted(
            row_key(row) for row in p95_by_row.keys() if row not in baseline_p95
        )
        if args.require_baseline_for_all_rows:
            baseline_p95_failures.extend(
                "%s:missing-baseline" % row for row in baseline_p95_uncovered_rows
            )
    baseline_p95_ok = len(baseline_p95_failures) == 0

    printed_counter_delta = False
    for (fixture, operation, key), values in sorted(counter_deltas.items()):
        total = sum(values)
        max_v = max(values) if values else 0
        if total == 0 and max_v == 0:
            continue
        printed_counter_delta = True
        print("counter_delta %s %s %s %d %d %d" % (fixture, operation, key, len(values), total, max_v))
    print("info counter_delta_rows %d" % (1 if printed_counter_delta else 0))
    allocation_attribution_by_row: dict[tuple[str, str], tuple[int, int, int, int, int]] = {}
    for fixture, operation in ALLOCATION_ATTRIBUTION_ROWS:
        attribution = allocation_attribution_for_row(counter_deltas, fixture, operation)
        allocation_attribution_by_row[(fixture, operation)] = attribution
        count, allocated_total, response_payload_total, unattributed_total, max_unattributed = attribution
        print(
            "allocation_attribution %s %s %d %d %d %d %d"
            % (fixture, operation, count, allocated_total, response_payload_total, unattributed_total, max_unattributed)
        )

    print("benchmark_profile %s" % args.profile)
    print("benchmark_build_mode %s" % (args.build_mode or "-"))
    print("fixture_stats fixture files source_bytes max_file_bytes public_functions configured_scale")
    fixture_stats("bench", {"bench.ora": bench_source}, args.functions + 2)
    fixture_stats("contractsuite", contract_sources, args.contract_helpers * args.contract_libs + args.contract_abi_methods, args.contract_libs)
    fixture_stats("importfanout", import_fanout_sources, args.import_fanout_files * 2, args.import_fanout_files)
    fixture_stats("importfanin", import_fanin_sources, args.import_fanin_files + 1, args.import_fanin_files)
    fixture_stats("cold", cold_sources, args.cold_import_uses + 2, args.cold_import_uses)
    fixture_stats("cold_call", cold_call_sources, 3)
    fixture_stats("positionsmall", {"positionsmall.ora": position_small_source}, 1, args.position_small_lines)
    fixture_stats("positionlarge", {"positionlarge.ora": position_large_source}, 1, args.position_large_lines)
    fixture_stats("stress", {"stress.ora": stress_source}, args.stress_symbols + 2, args.stress_occurrences)
    print("source_byte_totals total_source_bytes %d" % source_bytes)
    observed_peak_rss_kib = peak_rss_kib()
    print("peak_rss_kib %d" % observed_peak_rss_kib)
    open_source_bytes = stat_int(stats, "openSourceBytes")
    cold_source_bytes = stat_int(stats, "coldSourceBytes")
    tracked_cache_bytes = sum(stat_int(stats, key) for key in TRACKED_CACHE_BYTE_KEYS)
    tracked_total_bytes = open_source_bytes + cold_source_bytes + tracked_cache_bytes
    print(
        "tracked_byte_totals open_source_bytes %d cold_source_bytes %d tracked_cache_bytes %d tracked_total_bytes %d"
        % (open_source_bytes, cold_source_bytes, tracked_cache_bytes, tracked_total_bytes)
    )
    print("tracked_cache_components " + " ".join("%s=%d" % (key, stat_int(stats, key)) for key in TRACKED_CACHE_BYTE_KEYS))
    for key in sorted(stats.keys()):
        if isinstance(stats[key], int):
            print("cache_stats_final %s %d" % (key, stats[key]))

    source_ok = source_bytes <= args.max_total_source_bytes
    peak_ok = observed_peak_rss_kib <= args.max_peak_rss_kib
    byte_failures = []
    if open_source_bytes > args.max_open_source_bytes:
        byte_failures.append("openSourceBytes %d > %d" % (open_source_bytes, args.max_open_source_bytes))
    if cold_source_bytes > args.max_cold_source_bytes:
        byte_failures.append("coldSourceBytes %d > %d" % (cold_source_bytes, args.max_cold_source_bytes))
    if tracked_cache_bytes > args.max_tracked_cache_bytes:
        byte_failures.append("trackedCacheBytes %d > %d" % (tracked_cache_bytes, args.max_tracked_cache_bytes))
    if tracked_total_bytes > args.max_tracked_total_bytes:
        byte_failures.append("trackedTotalBytes %d > %d" % (tracked_total_bytes, args.max_tracked_total_bytes))
    byte_ok = len(byte_failures) == 0
    cache_stats_available = len(stats) > 0
    cold_document_failures = []
    if cache_stats_available:
        for key in COLD_DOCUMENT_COUNTER_KEYS:
            if key not in stats:
                cold_document_failures.append("%s:missing" % key)
        cold_documents = stat_int(stats, "coldDocuments")
        cold_document_max_count = stat_int(stats, "coldDocumentMaxCount")
        cold_source_max_bytes = stat_int(stats, "coldSourceMaxBytes")
        cold_refresh_checks = stat_int(stats, "coldDocumentRefreshChecks")
        if cold_documents <= 0:
            cold_document_failures.append("coldDocuments<=0")
        if cold_document_max_count <= 0:
            cold_document_failures.append("coldDocumentMaxCount<=0")
        if cold_source_max_bytes <= 0:
            cold_document_failures.append("coldSourceMaxBytes<=0")
        if cold_document_max_count > 0 and cold_documents > cold_document_max_count:
            cold_document_failures.append("coldDocuments %d > %d" % (cold_documents, cold_document_max_count))
        if cold_source_max_bytes > 0 and cold_source_bytes > cold_source_max_bytes:
            cold_document_failures.append("coldSourceBytes %d > coldSourceMaxBytes %d" % (cold_source_bytes, cold_source_max_bytes))
        if cold_refresh_checks <= 0:
            cold_document_failures.append("coldDocumentRefreshChecks<=0")
    cold_document_ok = cache_stats_available and len(cold_document_failures) == 0
    coverage_fixtures = ("bench", "contractsuite", "importfanout", "importfanin", "cold", "positionsmall", "positionlarge", "stress", "code_action")
    coverage_ok = all(any(key[0] == fixture for key in samples.keys()) for fixture in coverage_fixtures)
    profile_byte_failures = []
    if not args.skip_profile_byte_budgets:
        profile_byte_budgets = PROFILE_BYTE_BUDGETS[args.profile]
        if not cache_stats_available:
            profile_byte_failures.append("cache_stats_unavailable")
        for key, observed in (
            ("totalSourceBytes", source_bytes),
            ("openSourceBytes", open_source_bytes),
            ("coldSourceBytes", cold_source_bytes),
            ("trackedCacheBytes", tracked_cache_bytes),
            ("trackedTotalBytes", tracked_total_bytes),
            ("peakRssKiB", observed_peak_rss_kib),
        ):
            budget = profile_byte_budgets[key]
            if observed > budget:
                profile_byte_failures.append("%s %d > %d" % (key, observed, budget))
        for key in TRACKED_CACHE_BYTE_KEYS:
            budget = profile_byte_budgets["components"].get(key)
            if budget is None:
                profile_byte_failures.append("%s:missing-component-budget" % key)
                continue
            observed = stat_int(stats, key)
            if observed > budget:
                profile_byte_failures.append("%s %d > %d" % (key, observed, budget))
        for key in sorted(profile_byte_budgets["components"].keys()):
            if key not in TRACKED_CACHE_BYTE_KEYS:
                profile_byte_failures.append("%s:unused-component-budget" % key)
    profile_byte_ok = args.skip_profile_byte_budgets or len(profile_byte_failures) == 0
    response_coverage_failures = []
    if cache_stats_available:
        response_coverage_failures = [
            key
            for key in sorted(expected_response_counters)
            if stat_int(stats, key) <= 0
        ]
    response_coverage_ok = (not cache_stats_available) or len(response_coverage_failures) == 0
    response_string_failures = []
    response_markdown_failures = []
    if cache_stats_available:
        response_string_failures = [
            key
            for key in sorted(expected_response_string_counters)
            if stat_int(stats, key) <= 0
        ]
        response_markdown_failures = [
            key
            for key in sorted(expected_response_markdown_counters)
            if stat_int(stats, key) <= 0
        ]
    response_string_ok = (not cache_stats_available) or len(response_string_failures) == 0
    response_markdown_ok = (not cache_stats_available) or len(response_markdown_failures) == 0
    phase_counter_failures = []
    if cache_stats_available:
        phase_counter_failures = [
            key
            for key in PHASE_COUNTER_KEYS
            if stat_int(stats, key) <= 0
        ]
    phase_counter_ok = cache_stats_available and len(phase_counter_failures) == 0
    formatter_counter_failures = []
    if cache_stats_available:
        if stat_int(stats, FORMATTER_COUNTER_KEY) <= 0:
            formatter_counter_failures.append(FORMATTER_COUNTER_KEY)
    formatter_counter_ok = cache_stats_available and len(formatter_counter_failures) == 0
    formatter_phase_failures = []
    if cache_stats_available:
        if ("bench", "formatting.cache_hit") not in samples:
            formatter_phase_failures.append("bench.formatting.cache_hit:missing-samples")
        for key in FORMATTER_ZERO_DELTA_KEYS:
            values = counter_deltas.get(("bench", "formatting.cache_hit", key), [])
            if sum(values) != 0 or (max(values) if values else 0) != 0:
                formatter_phase_failures.append("bench.formatting.cache_hit:%s" % key)
    formatter_phase_ok = cache_stats_available and len(formatter_phase_failures) == 0
    cache_builder_failures = []
    if cache_stats_available:
        for key in CACHE_BUILDER_COUNTER_KEYS:
            if key not in stats:
                cache_builder_failures.append(key)
        cache_builder_capacity = stat_int(stats, "cacheBuilderCapacityRequested")
        cache_builder_items = stat_int(stats, "cacheBuilderItemsBuilt")
        cache_builder_unused = stat_int(stats, "cacheBuilderUnusedCapacity")
        if cache_builder_capacity <= 0:
            cache_builder_failures.append("cacheBuilderCapacityRequested<=0")
        if cache_builder_items <= 0:
            cache_builder_failures.append("cacheBuilderItemsBuilt<=0")
        if cache_builder_capacity < cache_builder_items:
            cache_builder_failures.append("cacheBuilderCapacityRequested<cacheBuilderItemsBuilt")
        if cache_builder_capacity >= cache_builder_items and cache_builder_unused != cache_builder_capacity - cache_builder_items:
            cache_builder_failures.append("cacheBuilderUnusedCapacity")
    cache_builder_ok = cache_stats_available and len(cache_builder_failures) == 0
    cache_side_map_failures = []
    if cache_stats_available:
        for key in CACHE_SIDE_MAP_COUNTER_KEYS:
            if key not in stats:
                cache_side_map_failures.append(key)
        cache_side_map_capacity = stat_int(stats, "cacheSideMapCapacityRequested")
        cache_side_map_items = stat_int(stats, "cacheSideMapItemsBuilt")
        cache_side_map_unused = stat_int(stats, "cacheSideMapUnusedCapacity")
        if cache_side_map_capacity <= 0:
            cache_side_map_failures.append("cacheSideMapCapacityRequested<=0")
        if cache_side_map_items <= 0:
            cache_side_map_failures.append("cacheSideMapItemsBuilt<=0")
        if cache_side_map_capacity < cache_side_map_items:
            cache_side_map_failures.append("cacheSideMapCapacityRequested<cacheSideMapItemsBuilt")
        if cache_side_map_capacity >= cache_side_map_items and cache_side_map_unused != cache_side_map_capacity - cache_side_map_items:
            cache_side_map_failures.append("cacheSideMapUnusedCapacity")
    cache_side_map_ok = cache_stats_available and len(cache_side_map_failures) == 0
    workspace_interner_failures = []
    if cache_stats_available:
        for key in WORKSPACE_INTERNER_COUNTER_KEYS:
            if key not in stats:
                workspace_interner_failures.append(key)
        workspace_interner_capacity = stat_int(stats, "workspaceIndexInternedStringCapacityRequested")
        workspace_interner_items = stat_int(stats, "workspaceIndexInternedStringItemsBuilt")
        workspace_interner_unused = stat_int(stats, "workspaceIndexInternedStringUnusedCapacity")
        if workspace_interner_capacity <= 0:
            workspace_interner_failures.append("workspaceIndexInternedStringCapacityRequested<=0")
        if workspace_interner_items <= 0:
            workspace_interner_failures.append("workspaceIndexInternedStringItemsBuilt<=0")
        if workspace_interner_capacity < workspace_interner_items:
            workspace_interner_failures.append("workspaceIndexInternedStringCapacityRequested<workspaceIndexInternedStringItemsBuilt")
        if workspace_interner_capacity >= workspace_interner_items and workspace_interner_unused != workspace_interner_capacity - workspace_interner_items:
            workspace_interner_failures.append("workspaceIndexInternedStringUnusedCapacity")
    workspace_interner_ok = cache_stats_available and len(workspace_interner_failures) == 0
    cold_workspace_interner_failures = []
    if cache_stats_available:
        for key in COLD_WORKSPACE_INTERNER_COUNTER_KEYS:
            if key not in stats:
                cold_workspace_interner_failures.append(key)
        cold_workspace_entries = stat_int(stats, "coldWorkspaceIndexEntries")
        cold_workspace_bytes = stat_int(stats, "coldWorkspaceIndexBytes")
        cold_workspace_interner_bytes = stat_int(stats, "coldWorkspaceIndexInternedStringBytes")
        cold_workspace_interner_count = stat_int(stats, "coldWorkspaceIndexInternedStringCount")
        cold_workspace_interner_capacity = stat_int(stats, "coldWorkspaceIndexInternedStringCapacityRequested")
        cold_workspace_interner_items = stat_int(stats, "coldWorkspaceIndexInternedStringItemsBuilt")
        cold_workspace_interner_unused = stat_int(stats, "coldWorkspaceIndexInternedStringUnusedCapacity")
        if cold_workspace_entries <= 0:
            cold_workspace_interner_failures.append("coldWorkspaceIndexEntries<=0")
        if cold_workspace_bytes <= 0:
            cold_workspace_interner_failures.append("coldWorkspaceIndexBytes<=0")
        if cold_workspace_interner_bytes <= 0:
            cold_workspace_interner_failures.append("coldWorkspaceIndexInternedStringBytes<=0")
        if cold_workspace_interner_count <= 0:
            cold_workspace_interner_failures.append("coldWorkspaceIndexInternedStringCount<=0")
        if cold_workspace_interner_capacity <= 0:
            cold_workspace_interner_failures.append("coldWorkspaceIndexInternedStringCapacityRequested<=0")
        if cold_workspace_interner_items <= 0:
            cold_workspace_interner_failures.append("coldWorkspaceIndexInternedStringItemsBuilt<=0")
        if cold_workspace_interner_capacity < cold_workspace_interner_items:
            cold_workspace_interner_failures.append("coldWorkspaceIndexInternedStringCapacityRequested<coldWorkspaceIndexInternedStringItemsBuilt")
        if cold_workspace_interner_capacity >= cold_workspace_interner_items and cold_workspace_interner_unused != cold_workspace_interner_capacity - cold_workspace_interner_items:
            cold_workspace_interner_failures.append("coldWorkspaceIndexInternedStringUnusedCapacity")
        if cold_workspace_entries > stat_int(stats, "workspaceIndexEntries"):
            cold_workspace_interner_failures.append("coldWorkspaceIndexEntries>workspaceIndexEntries")
        if cold_workspace_bytes > stat_int(stats, "workspaceIndexBytes"):
            cold_workspace_interner_failures.append("coldWorkspaceIndexBytes>workspaceIndexBytes")
    cold_workspace_interner_ok = cache_stats_available and len(cold_workspace_interner_failures) == 0
    allocator_counter_failures = []
    if cache_stats_available:
        for key in (
            "serverAllocatorAllocCalls",
            "serverAllocatorBytesAllocated",
            "serverAllocatorBytesLive",
            "serverAllocatorBytesPeak",
        ):
            if stat_int(stats, key) <= 0:
                allocator_counter_failures.append(key)
    allocator_counter_ok = cache_stats_available and len(allocator_counter_failures) == 0
    scoped_allocator_failures = []
    if cache_stats_available:
        for key in (*SCOPED_ALLOCATOR_ALLOC_CALL_KEYS, *SCOPED_ALLOCATOR_BYTE_KEYS):
            if key not in stats:
                scoped_allocator_failures.append("%s:missing" % key)
        for key in SCOPED_ALLOCATOR_ACTIVE_BYTE_KEYS:
            if stat_int(stats, key) <= 0:
                scoped_allocator_failures.append("%s<=0" % key)

        scoped_alloc_calls = sum(stat_int(stats, key) for key in SCOPED_ALLOCATOR_ALLOC_CALL_KEYS)
        total_alloc_calls = stat_int(stats, "serverAllocatorAllocCalls")
        if scoped_alloc_calls != total_alloc_calls:
            scoped_allocator_failures.append(
                "scopedAllocCalls %d != serverAllocatorAllocCalls %d" % (scoped_alloc_calls, total_alloc_calls)
            )

        scoped_allocated_bytes = sum(stat_int(stats, key) for key in SCOPED_ALLOCATOR_BYTE_KEYS)
        total_allocated_bytes = stat_int(stats, "serverAllocatorBytesAllocated")
        if scoped_allocated_bytes != total_allocated_bytes:
            scoped_allocator_failures.append(
                "scopedBytesAllocated %d != serverAllocatorBytesAllocated %d" % (scoped_allocated_bytes, total_allocated_bytes)
            )
    scoped_allocator_ok = cache_stats_available and len(scoped_allocator_failures) == 0
    allocation_attribution_failures = []
    if not args.skip_profile_allocation_budgets:
        for fixture, operation in ALLOCATION_ATTRIBUTION_ROWS:
            row = (fixture, operation)
            if row not in samples:
                allocation_attribution_failures.append("%s.%s:missing-samples" % row)
                continue
            count, _, _, _, max_unattributed = allocation_attribution_by_row.get(row, (0, 0, 0, 0, 0))
            if count <= 0:
                allocation_attribution_failures.append("%s.%s:missing-allocator-deltas" % row)
                continue
            budget = allocation_attribution_budget(args.profile, row)
            if budget is None:
                allocation_attribution_failures.append("%s.%s:missing-budget" % row)
                continue
            if max_unattributed > budget:
                allocation_attribution_failures.append("%s.%s %d > %d" % (fixture, operation, max_unattributed, budget))
    allocation_attribution_ok = args.skip_profile_allocation_budgets or (
        cache_stats_available and len(allocation_attribution_failures) == 0
    )
    selection_phase_failures = []
    if cache_stats_available:
        for key in PHASE_COUNTER_KEYS:
            values = counter_deltas.get(("bench", "selectionRange.cache_hit", key), [])
            if sum(values) != 0 or (max(values) if values else 0) != 0:
                selection_phase_failures.append(key)
    selection_phase_ok = cache_stats_available and len(selection_phase_failures) == 0
    db_phase_failures = []
    if cache_stats_available:
        for fixture, operation in DB_BACKED_PHASE_FLAT_ROWS:
            for key in CACHE_HIT_ZERO_DELTA_KEYS:
                values = counter_deltas.get((fixture, operation, key), [])
                if sum(values) != 0 or (max(values) if values else 0) != 0:
                    db_phase_failures.append("%s.%s:%s" % (fixture, operation, key))
    db_phase_ok = cache_stats_available and len(db_phase_failures) == 0
    position_lookup_failures = []
    if cache_stats_available:
        for fixture, operation in POSITION_LOOKUP_ROWS:
            if (fixture, operation) not in samples:
                position_lookup_failures.append("%s.%s:missing-samples" % (fixture, operation))
                continue
            for key in (*PHASE_COUNTER_KEYS, "lineIndexBytes"):
                values = counter_deltas.get((fixture, operation, key), [])
                if sum(values) != 0 or (max(values) if values else 0) != 0:
                    position_lookup_failures.append("%s.%s:%s" % (fixture, operation, key))
        small_p95 = max((p95_by_row.get(row, 0.0) for row in POSITION_LOOKUP_ROWS if row[0] == "positionsmall"), default=0.0)
        large_p95 = max((p95_by_row.get(row, 0.0) for row in POSITION_LOOKUP_ROWS if row[0] == "positionlarge"), default=0.0)
        scale_budget = small_p95 * args.max_position_lookup_scale_ratio + args.position_lookup_scale_slop_ms
        if large_p95 > args.max_position_lookup_p95_ms:
            position_lookup_failures.append("positionlarge.p95 %.3f > %.3f" % (large_p95, args.max_position_lookup_p95_ms))
        if small_p95 > 0.0 and large_p95 > scale_budget:
            position_lookup_failures.append("positionlarge.p95 %.3f > scaled %.3f" % (large_p95, scale_budget))
    position_lookup_ok = cache_stats_available and len(position_lookup_failures) == 0
    cold_cache_failures = []
    if cache_stats_available:
        for fixture, operation in COLD_IMPORTED_MEMBER_ROWS:
            if (fixture, operation) not in samples:
                cold_cache_failures.append("%s.%s:missing-samples" % (fixture, operation))
                continue
            for key in ("coldWorkspaceIndexBuilds", "workspaceDiscoveryRuns"):
                values = counter_deltas.get((fixture, operation, key), [])
                if sum(values) != 0 or (max(values) if values else 0) != 0:
                    cold_cache_failures.append("%s.%s:%s" % (fixture, operation, key))
    cold_cache_ok = cache_stats_available and len(cold_cache_failures) == 0
    cold_first_build_failures = []
    if cache_stats_available:
        for fixture, operation in COLD_WORKSPACE_FIRST_BUILD_ROWS:
            if (fixture, operation) not in samples:
                cold_first_build_failures.append("%s.%s:missing-samples" % (fixture, operation))
                continue
            build_values = counter_deltas.get((fixture, operation, "coldWorkspaceIndexBuilds"), [])
            if sum(build_values) <= 0:
                cold_first_build_failures.append("%s.%s:coldWorkspaceIndexBuilds<=0" % (fixture, operation))
        for fixture, operation in COLD_DIRECT_IMPORTED_MEMBER_FIRST_BUILD_ROWS:
            if (fixture, operation) not in samples:
                cold_first_build_failures.append("%s.%s:missing-samples" % (fixture, operation))
                continue
            import_values = counter_deltas.get((fixture, operation, "importedMemberIndexBuilds"), [])
            if sum(import_values) <= 0:
                cold_first_build_failures.append("%s.%s:importedMemberIndexBuilds<=0" % (fixture, operation))
            workspace_values = counter_deltas.get((fixture, operation, "coldWorkspaceIndexBuilds"), [])
            if sum(workspace_values) != 0 or (max(workspace_values) if workspace_values else 0) != 0:
                cold_first_build_failures.append("%s.%s:coldWorkspaceIndexBuilds" % (fixture, operation))
    cold_first_build_ok = cache_stats_available and len(cold_first_build_failures) == 0
    edit_diagnostic_boundary_failures = []
    if cache_stats_available:
        edit_positive_keys = (
            "diagnosticCacheBuilds",
            "diagnosticFastBuilds",
            "editDiagnosticFastPublishes",
            "editDiagnosticFullSkips",
            "dependentDiagnosticPublishSkips",
            "diagnosticDebounceScheduled",
            "diagnosticDebouncePending",
        )
        for key in edit_positive_keys:
            values = counter_deltas.get(("bench", "didChange.diagnostics", key), [])
            if sum(values) <= 0:
                edit_diagnostic_boundary_failures.append("%s<=0" % key)

        edit_zero_keys = (
            "itemIndexBuilds",
            "resolveBuilds",
            "constEvalBuilds",
            "typeCheckBuilds",
            "semanticIndexBuilds",
            "occurrenceIndexBuilds",
            "importedMemberIndexBuilds",
            "callEdgeIndexBuilds",
            "workspaceIndexBuilds",
            "coldWorkspaceIndexBuilds",
            "incomingCallTargetIndexBuilds",
            "diagnosticFullBuilds",
            "dependentDiagnosticPublishRuns",
            "dependentDiagnosticPublishedDocuments",
            "diagnosticDebounceFlushed",
            "diagnosticDebounceCleared",
        )
        for key in edit_zero_keys:
            values = counter_deltas.get(("bench", "didChange.diagnostics", key), [])
            if sum(values) != 0 or (max(values) if values else 0) != 0:
                edit_diagnostic_boundary_failures.append("%s=%s" % (key, ",".join(str(v) for v in values)))
        cancel_values = counter_deltas.get(("bench", "didChange.diagnostics", "diagnosticDebounceCanceled"), [])
        if len(cancel_values) > 1 and sum(cancel_values) <= 0:
            edit_diagnostic_boundary_failures.append("diagnosticDebounceCanceled<=0")
    edit_diagnostic_boundary_ok = cache_stats_available and len(edit_diagnostic_boundary_failures) == 0

    print("gate request_p95_budget ok %d max_p95_ms %.3f" % (1 if request_ok else 0, args.max_request_p95_ms))
    print("gate edit_diagnostics_p95_budget ok %d max_p95_ms %.3f" % (1 if edit_ok else 0, args.max_edit_diagnostics_p95_ms))
    print("gate profile_p95_budget ok %d failures %s" % (
        1 if profile_budget_ok else 0,
        ",".join(profile_budget_failures) if profile_budget_failures else "-",
    ))
    if args.baseline_p95_file:
        print("gate p95_baseline_improvement ok %d min_improvement %.3f compared_rows %d uncovered_rows %d failures %s" % (
            1 if baseline_p95_ok else 0,
            args.min_p95_improvement,
            baseline_p95_count,
            len(baseline_p95_uncovered_rows),
            ",".join(baseline_p95_failures) if baseline_p95_failures else "-",
        ))
        if baseline_p95_uncovered_rows:
            print("info p95_baseline_uncovered %s" % ",".join(baseline_p95_uncovered_rows))
    else:
        print("info p95_baseline_improvement skipped no_baseline_p95_file")
    print("gate representative_fixture_coverage ok %d fixtures %s" % (1 if coverage_ok else 0, ",".join(coverage_fixtures)))
    print("info cache_stats_available %d" % (1 if cache_stats_available else 0))
    if cache_stats_available:
        response_gate_label = "gate" if args.strict_future_gates else "target_gate"
        print("%s response_builder_coverage ok %d missing %s" % (
            response_gate_label,
            1 if response_coverage_ok else 0,
            ",".join(response_coverage_failures) if response_coverage_failures else "-",
        ))
        print("info response_builder_expected %s" % (",".join(sorted(expected_response_counters)) if expected_response_counters else "-"))
        print("%s response_string_byte_coverage ok %d missing %s" % (
            response_gate_label,
            1 if response_string_ok else 0,
            ",".join(response_string_failures) if response_string_failures else "-",
        ))
        print("info response_string_expected %s" % (",".join(sorted(expected_response_string_counters)) if expected_response_string_counters else "-"))
        print("%s response_markdown_byte_coverage ok %d missing %s" % (
            response_gate_label,
            1 if response_markdown_ok else 0,
            ",".join(response_markdown_failures) if response_markdown_failures else "-",
        ))
        print("info response_markdown_expected %s" % (",".join(sorted(expected_response_markdown_counters)) if expected_response_markdown_counters else "-"))
        print("gate compiler_phase_counter_coverage ok %d missing %s" % (
            1 if phase_counter_ok else 0,
            ",".join(phase_counter_failures) if phase_counter_failures else "-",
        ))
        print("gate formatter_counter_coverage ok %d missing %s" % (
            1 if formatter_counter_ok else 0,
            ",".join(formatter_counter_failures) if formatter_counter_failures else "-",
        ))
        print("gate formatter_phase_boundary ok %d deltas %s" % (
            1 if formatter_phase_ok else 0,
            ",".join(formatter_phase_failures) if formatter_phase_failures else "-",
        ))
        print("gate cache_builder_telemetry ok %d failures %s" % (
            1 if cache_builder_ok else 0,
            ",".join(cache_builder_failures) if cache_builder_failures else "-",
        ))
        print("gate cache_side_map_telemetry ok %d failures %s" % (
            1 if cache_side_map_ok else 0,
            ",".join(cache_side_map_failures) if cache_side_map_failures else "-",
        ))
        print("gate workspace_interner_telemetry ok %d failures %s" % (
            1 if workspace_interner_ok else 0,
            ",".join(workspace_interner_failures) if workspace_interner_failures else "-",
        ))
        print("gate cold_workspace_interner_telemetry ok %d failures %s" % (
            1 if cold_workspace_interner_ok else 0,
            ",".join(cold_workspace_interner_failures) if cold_workspace_interner_failures else "-",
        ))
        print("gate cold_document_telemetry ok %d failures %s" % (
            1 if cold_document_ok else 0,
            ",".join(cold_document_failures) if cold_document_failures else "-",
        ))
        print("gate server_allocator_counter_coverage ok %d missing %s" % (
            1 if allocator_counter_ok else 0,
            ",".join(allocator_counter_failures) if allocator_counter_failures else "-",
        ))
        print("gate scoped_allocator_counter_coverage ok %d failures %s" % (
            1 if scoped_allocator_ok else 0,
            ",".join(scoped_allocator_failures) if scoped_allocator_failures else "-",
        ))
        print("gate allocator_attribution_budget ok %d failures %s" % (
            1 if allocation_attribution_ok else 0,
            ",".join(allocation_attribution_failures) if allocation_attribution_failures else "-",
        ))
        print("%s selection_range_phase_cache_hit ok %d deltas %s" % (
            response_gate_label,
            1 if selection_phase_ok else 0,
            ",".join(selection_phase_failures) if selection_phase_failures else "-",
        ))
        print("%s db_backed_read_phase_cache_hit ok %d deltas %s" % (
            response_gate_label,
            1 if db_phase_ok else 0,
            ",".join(db_phase_failures) if db_phase_failures else "-",
        ))
        print("%s position_lookup_scaling ok %d deltas %s" % (
            response_gate_label,
            1 if position_lookup_ok else 0,
            ",".join(position_lookup_failures) if position_lookup_failures else "-",
        ))
        print("%s cold_imported_member_cache_hit ok %d deltas %s" % (
            response_gate_label,
            1 if cold_cache_ok else 0,
            ",".join(cold_cache_failures) if cold_cache_failures else "-",
        ))
        print("%s cold_workspace_first_build ok %d deltas %s" % (
            response_gate_label,
            1 if cold_first_build_ok else 0,
            ",".join(cold_first_build_failures) if cold_first_build_failures else "-",
        ))
        print("gate edit_diagnostics_phase_boundary ok %d deltas %s" % (
            1 if edit_diagnostic_boundary_ok else 0,
            ",".join(edit_diagnostic_boundary_failures) if edit_diagnostic_boundary_failures else "-",
        ))
    else:
        print("info response_builder_coverage skipped cache_stats_unavailable")
        print("info response_string_byte_coverage skipped cache_stats_unavailable")
        print("info response_markdown_byte_coverage skipped cache_stats_unavailable")
        print("info compiler_phase_counter_coverage skipped cache_stats_unavailable")
        print("info formatter_counter_coverage skipped cache_stats_unavailable")
        print("info formatter_phase_boundary skipped cache_stats_unavailable")
        print("info cache_builder_telemetry skipped cache_stats_unavailable")
        print("info cache_side_map_telemetry skipped cache_stats_unavailable")
        print("info workspace_interner_telemetry skipped cache_stats_unavailable")
        print("info cold_workspace_interner_telemetry skipped cache_stats_unavailable")
        print("info cold_document_telemetry skipped cache_stats_unavailable")
        print("info server_allocator_counter_coverage skipped cache_stats_unavailable")
        print("info scoped_allocator_counter_coverage skipped cache_stats_unavailable")
        print("info allocator_attribution_budget skipped cache_stats_unavailable")
        print("info selection_range_phase_cache_hit skipped cache_stats_unavailable")
        print("info db_backed_read_phase_cache_hit skipped cache_stats_unavailable")
        print("info position_lookup_scaling skipped cache_stats_unavailable")
        print("info cold_imported_member_cache_hit skipped cache_stats_unavailable")
        print("info cold_workspace_first_build skipped cache_stats_unavailable")
        print("info edit_diagnostics_phase_boundary skipped cache_stats_unavailable")
    print("gate source_bytes_budget ok %d max_total_source_bytes %d observed_total_source_bytes %d" % (1 if source_ok else 0, args.max_total_source_bytes, source_bytes))
    print("gate tracked_cache_bytes ok %d max_open_source_bytes %d max_cold_source_bytes %d max_tracked_cache_bytes %d max_tracked_total_bytes %d failures %s" % (
        1 if byte_ok else 0,
        args.max_open_source_bytes,
        args.max_cold_source_bytes,
        args.max_tracked_cache_bytes,
        args.max_tracked_total_bytes,
        ";".join(byte_failures) if byte_failures else "-",
    ))
    print("gate peak_rss_budget ok %d max_peak_rss_kib %d observed_peak_rss_kib %d" % (1 if peak_ok else 0, args.max_peak_rss_kib, observed_peak_rss_kib))
    print("gate profile_byte_budget ok %d failures %s" % (
        1 if profile_byte_ok else 0,
        ";".join(profile_byte_failures) if profile_byte_failures else "-",
    ))
    future_gates_ok = response_coverage_ok and response_string_ok and response_markdown_ok and selection_phase_ok and db_phase_ok and position_lookup_ok and cold_cache_ok and cold_first_build_ok
    return 0 if (
        request_ok and
        edit_ok and
        profile_budget_ok and
        baseline_p95_ok and
        profile_byte_ok and
        source_ok and
        byte_ok and
        peak_ok and
        coverage_ok and
        phase_counter_ok and
        formatter_counter_ok and
        formatter_phase_ok and
        cache_builder_ok and
        cache_side_map_ok and
        workspace_interner_ok and
        cold_workspace_interner_ok and
        cold_document_ok and
        cold_first_build_ok and
        edit_diagnostic_boundary_ok and
        allocator_counter_ok and
        scoped_allocator_ok and
        allocation_attribution_ok and
        (future_gates_ok or not args.strict_future_gates)
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
