#!/usr/bin/env python3
import argparse
import sys
import tempfile
from pathlib import Path

from lsp_jsonrpc import (
    JsonRpcClient,
    did_change_full,
    did_open,
    file_uri,
    initialize,
    position_of,
    require,
    wait_diagnostics,
)


VALID_SOURCE = """// Internal note that must not become public docs.
/// Returns `value` unchanged.
pub fn helper(value: u256) -> u256 {
    return value;
}

pub fn run(amount: u256) -> u256 {
    return helper(amount);
}

contract Wallet {
    invariant(balance >= 0);
    storage var balance: u256;

    pub fn deposit(amount: u256) -> u256
        requires(amount > 0)
        ensures(result >= amount)
    {
        return amount;
    }
}
"""


LIB_SOURCE = """pub fn external(value: u256) -> u256 {
    return value;
}
"""


UNICODE_LIB_SOURCE = """/* é */ pub fn external(value: u256) -> u256 {
    return value;
}
"""


IMPORTING_SOURCE = """const Lib = @import("./lib.ora");

pub fn use_external(amount: u256) -> u256 {
    return Lib.external(amount);
}
"""


UNICODE_IMPORTING_SOURCE = """const U = @import("./unicode_lib.ora");

pub fn use_unicode(amount: u256) -> u256 {
    return U.external(amount);
}
"""


BROKEN_SOURCE = """pub fn broken() -> u256 {
    return 1
}
"""


def has_label(result, label: str) -> bool:
    if result is None:
        return False
    items = result.get("items") if isinstance(result, dict) else result
    return any(isinstance(item, dict) and item.get("label") == label for item in items or [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an Ora LSP JSON-RPC smoke test.")
    parser.add_argument("server_positional", nargs="?")
    parser.add_argument("--server")
    args = parser.parse_args()

    server = args.server or args.server_positional
    if server is None:
        parser.error("missing LSP server path")

    with tempfile.TemporaryDirectory(prefix="ora-lsp-smoke-") as tmp:
        root = Path(tmp)
        valid_path = root / "wallet.ora"
        lib_path = root / "lib.ora"
        unicode_lib_path = root / "unicode_lib.ora"
        importing_path = root / "importing.ora"
        unicode_importing_path = root / "unicode_importing.ora"
        broken_path = root / "broken.ora"
        valid_path.write_text(VALID_SOURCE)
        lib_path.write_text(LIB_SOURCE)
        unicode_lib_path.write_text(UNICODE_LIB_SOURCE)
        importing_path.write_text(IMPORTING_SOURCE)
        unicode_importing_path.write_text(UNICODE_IMPORTING_SOURCE)
        broken_path.write_text(BROKEN_SOURCE)

        valid_uri = file_uri(valid_path)
        lib_uri = file_uri(lib_path)
        unicode_lib_uri = file_uri(unicode_lib_path)
        importing_uri = file_uri(importing_path)
        unicode_importing_uri = file_uri(unicode_importing_path)
        broken_uri = file_uri(broken_path)

        client = JsonRpcClient(server)
        try:
            init_result = initialize(client, file_uri(root))
            capabilities = init_result.get("capabilities", {})
            require(capabilities.get("hoverProvider") is not None, "missing hoverProvider capability")
            require(capabilities.get("completionProvider") is not None, "missing completionProvider capability")
            require(capabilities.get("semanticTokensProvider") is not None, "missing semanticTokensProvider capability")
            execute_provider = capabilities.get("executeCommandProvider") or {}
            require("ora.cacheStats" in execute_provider.get("commands", []), "ora.cacheStats command not advertised")

            did_open(client, valid_uri, VALID_SOURCE)
            valid_diagnostics = wait_diagnostics(client, valid_uri)
            require(
                not any(diag.get("severity") == 1 for diag in valid_diagnostics),
                "valid smoke document produced error diagnostics",
            )

            did_change_full(client, valid_uri, VALID_SOURCE + "\n// stale edit that must be ignored\n", version=1)
            stale_cache_stats = client.request("workspace/executeCommand", {"command": "ora.cacheStats", "arguments": []})
            require(
                stale_cache_stats.get("staleDocumentChangeSkips", 0) > 0,
                "ora.cacheStats did not count stale didChange skips",
            )

            symbols = client.request("textDocument/documentSymbol", {"textDocument": {"uri": valid_uri}})
            require(symbols and len(symbols) >= 2, "documentSymbol returned no top-level symbols")

            hover = client.request(
                "textDocument/hover",
                {"textDocument": {"uri": valid_uri}, "position": position_of(VALID_SOURCE, "helper", 0)},
            )
            hover_text = hover.get("contents", {}).get("value", "") if isinstance(hover, dict) else ""
            require("Returns `value` unchanged." in hover_text, "/// public doc missing from hover markdown")
            require("Internal note" not in hover_text, "// internal comment leaked into hover markdown")

            completion = client.request(
                "textDocument/completion",
                {
                    "textDocument": {"uri": valid_uri},
                    "position": position_of(VALID_SOURCE, "helper(amount)", 0, 3),
                    "context": {"triggerKind": 1},
                },
            )
            require(has_label(completion, "helper"), "completion did not include helper symbol")

            definition = client.request(
                "textDocument/definition",
                {"textDocument": {"uri": valid_uri}, "position": position_of(VALID_SOURCE, "helper(amount)", 0)},
            )
            require(definition is not None, "definition returned null")

            references = client.request(
                "textDocument/references",
                {
                    "textDocument": {"uri": valid_uri},
                    "position": position_of(VALID_SOURCE, "helper(amount)", 0),
                    "context": {"includeDeclaration": True},
                },
            )
            require(references and len(references) >= 2, "references did not include declaration and call")

            highlights = client.request(
                "textDocument/documentHighlight",
                {"textDocument": {"uri": valid_uri}, "position": position_of(VALID_SOURCE, "helper(amount)", 0)},
            )
            require(highlights and len(highlights) >= 2, "documentHighlight did not use cached occurrences")

            signature = client.request(
                "textDocument/signatureHelp",
                {"textDocument": {"uri": valid_uri}, "position": position_of(VALID_SOURCE, "helper(", 0, len("helper("))},
            )
            require(isinstance(signature, dict), "signatureHelp returned no payload")
            signatures = signature.get("signatures", [])
            require(signatures and "fn helper(value: u256) -> u256" in signatures[0].get("label", ""), "signatureHelp missing helper signature")
            require(signature.get("activeSignature") == 0, "signatureHelp active signature should be zero")
            require(signature.get("activeParameter") == 0, "signatureHelp active parameter should be zero")

            semantic_tokens = client.request(
                "textDocument/semanticTokens/full",
                {"textDocument": {"uri": valid_uri}},
            )
            require(semantic_tokens and len(semantic_tokens.get("data", [])) > 0, "semanticTokens returned empty data")

            formatting = client.request(
                "textDocument/formatting",
                {"textDocument": {"uri": valid_uri}, "options": {"tabSize": 4, "insertSpaces": True}},
            )
            require(formatting is not None, "formatting returned null")

            code_lens = client.request("textDocument/codeLens", {"textDocument": {"uri": valid_uri}})
            require(code_lens and len(code_lens) >= 2, "codeLens returned no verification lenses")

            folding = client.request("textDocument/foldingRange", {"textDocument": {"uri": valid_uri}})
            require(folding and len(folding) > 0, "foldingRange returned empty result")

            did_open(client, unicode_importing_uri, UNICODE_IMPORTING_SOURCE)
            unopened_definition = client.request(
                "textDocument/definition",
                {
                    "textDocument": {"uri": unicode_importing_uri},
                    "position": position_of(UNICODE_IMPORTING_SOURCE, "U.external", 0),
                },
            )
            expected_external = position_of(UNICODE_LIB_SOURCE, "external", 0)
            require(unopened_definition is not None, "cross-file unopened definition returned null")
            require(
                unopened_definition.get("uri") == unicode_lib_uri,
                f"cross-file unopened definition returned wrong uri: {unopened_definition.get('uri')!r}",
            )
            actual_range = unopened_definition.get("range", {})
            actual_start = actual_range.get("start", {})
            require(
                actual_start.get("line") == expected_external["line"]
                and actual_start.get("character") == expected_external["character"],
                f"cross-file unopened definition did not convert target range through UTF-16 line index: {actual_start!r}",
            )

            did_open(client, lib_uri, LIB_SOURCE)
            cold_imported_references = client.request(
                "textDocument/references",
                {
                    "textDocument": {"uri": lib_uri},
                    "position": position_of(LIB_SOURCE, "external", 0),
                    "context": {"includeDeclaration": True},
                },
            )
            require(
                cold_imported_references
                and any(ref.get("uri") == importing_uri for ref in cold_imported_references),
                "references did not include cold unopened imported member use",
            )

            cold_rename = client.request(
                "textDocument/rename",
                {
                    "textDocument": {"uri": lib_uri},
                    "position": position_of(LIB_SOURCE, "external", 0),
                    "newName": "externalRenamed",
                },
            )
            cold_changes = cold_rename.get("changes", {}) if isinstance(cold_rename, dict) else {}
            require(importing_uri in cold_changes, "rename did not include cold unopened imported member edit")
            require(
                any(edit.get("newText") == "externalRenamed" for edit in cold_changes.get(importing_uri, [])),
                "rename cold unopened edit used wrong replacement text",
            )

            cold_cache_stats = client.request("workspace/executeCommand", {"command": "ora.cacheStats", "arguments": []})
            require(cold_cache_stats.get("coldDocuments", 0) > 0, "ora.cacheStats missing cold document count")
            require(cold_cache_stats.get("coldSourceBytes", 0) > 0, "ora.cacheStats missing cold source bytes")
            require(cold_cache_stats.get("coldWorkspaceIndexBuilds", 0) > 0, "ora.cacheStats missing cold workspace index builds")
            require(cold_cache_stats.get("coldWorkspaceIndexEntries", 0) > 0, "ora.cacheStats missing cold workspace index entries")
            require(cold_cache_stats.get("coldWorkspaceIndexBytes", 0) > 0, "ora.cacheStats missing cold workspace index bytes")
            require(
                cold_cache_stats.get("coldWorkspaceIndexInternedStringBytes", 0) > 0,
                "ora.cacheStats missing cold workspace interned string bytes",
            )
            cold_workspace_interner_capacity = cold_cache_stats.get("coldWorkspaceIndexInternedStringCapacityRequested", 0)
            cold_workspace_interner_items = cold_cache_stats.get("coldWorkspaceIndexInternedStringItemsBuilt", 0)
            require(
                cold_workspace_interner_capacity > 0,
                "ora.cacheStats missing cold workspace interned string capacity",
            )
            require(
                cold_workspace_interner_items > 0,
                "ora.cacheStats missing cold workspace interned string item count",
            )
            require(
                cold_workspace_interner_capacity >= cold_workspace_interner_items,
                "ora.cacheStats cold workspace interned string capacity is smaller than built items",
            )
            require(cold_cache_stats.get("workspaceDiscoveryRuns", 0) > 0, "ora.cacheStats missing workspace discovery runs")
            require(cold_cache_stats.get("workspaceDiscoveryCacheHits", 0) > 0, "ora.cacheStats missing workspace discovery cache hits")

            did_open(client, importing_uri, IMPORTING_SOURCE)
            imported_references = client.request(
                "textDocument/references",
                {
                    "textDocument": {"uri": lib_uri},
                    "position": position_of(LIB_SOURCE, "external", 0),
                    "context": {"includeDeclaration": True},
                },
            )
            require(
                imported_references and len(imported_references) >= 2,
                "references did not include cached imported member use",
            )

            did_open(client, broken_uri, BROKEN_SOURCE)
            broken_diagnostics = wait_diagnostics(client, broken_uri)
            require(len(broken_diagnostics) > 0, "broken smoke document produced no diagnostics")

            cache_stats = client.request("workspace/executeCommand", {"command": "ora.cacheStats", "arguments": []})
            require(isinstance(cache_stats, dict), "ora.cacheStats did not return an object")
            require(cache_stats.get("openDocuments", 0) >= 2, "ora.cacheStats missing open document count")
            require(cache_stats.get("openSourceBytes", 0) > 0, "ora.cacheStats missing open source bytes")
            require(cache_stats.get("documentStateBytes", 0) > 0, "ora.cacheStats missing document-state bytes")
            require(cache_stats.get("lineIndexBytes", 0) > 0, "ora.cacheStats missing line-index bytes")
            require(cache_stats.get("openTokenCount", 0) > 0, "ora.cacheStats missing open token count")
            require(cache_stats.get("tokenCacheBytes", 0) > 0, "ora.cacheStats missing token cache bytes")
            cache_builder_capacity = cache_stats.get("cacheBuilderCapacityRequested", 0)
            cache_builder_items = cache_stats.get("cacheBuilderItemsBuilt", 0)
            require(cache_builder_capacity > 0, "ora.cacheStats missing cache builder capacity")
            require(cache_builder_items > 0, "ora.cacheStats missing cache builder item count")
            require(
                cache_builder_capacity >= cache_builder_items,
                "ora.cacheStats cache builder capacity is smaller than built items",
            )
            cache_side_map_capacity = cache_stats.get("cacheSideMapCapacityRequested", 0)
            cache_side_map_items = cache_stats.get("cacheSideMapItemsBuilt", 0)
            require(cache_side_map_capacity > 0, "ora.cacheStats missing cache side-map capacity")
            require(cache_side_map_items > 0, "ora.cacheStats missing cache side-map item count")
            require(
                cache_side_map_capacity >= cache_side_map_items,
                "ora.cacheStats cache side-map capacity is smaller than built items",
            )
            workspace_interner_capacity = cache_stats.get("workspaceIndexInternedStringCapacityRequested", 0)
            workspace_interner_items = cache_stats.get("workspaceIndexInternedStringItemsBuilt", 0)
            require(workspace_interner_capacity > 0, "ora.cacheStats missing workspace interned string capacity")
            require(workspace_interner_items > 0, "ora.cacheStats missing workspace interned string item count")
            require(
                workspace_interner_capacity >= workspace_interner_items,
                "ora.cacheStats workspace interned string capacity is smaller than built items",
            )
            require(cache_stats.get("responseBuilderItemsBuilt", 0) > 0, "ora.cacheStats missing response builder item count")
            require(cache_stats.get("responseBuilderCapacityBytes", 0) > 0, "ora.cacheStats missing response builder capacity bytes")
            require(cache_stats.get("responseStringBytes", 0) > 0, "ora.cacheStats missing aggregate response string bytes")
            require(cache_stats.get("responseMarkdownBytes", 0) > 0, "ora.cacheStats missing aggregate response markdown bytes")
            require(cache_stats.get("responseHoverStringBytes", 0) > 0, "ora.cacheStats missing hover response string bytes")
            require(cache_stats.get("responseHoverMarkdownBytes", 0) > 0, "ora.cacheStats missing hover response markdown bytes")
            require(cache_stats.get("responseCompletionStringBytes", 0) > 0, "ora.cacheStats missing completion response string bytes")
            require(cache_stats.get("responseCompletionMarkdownBytes", 0) > 0, "ora.cacheStats missing completion response markdown bytes")
            require(cache_stats.get("responseSignatureStringBytes", 0) > 0, "ora.cacheStats missing signature response string bytes")
            require(cache_stats.get("responseSignatureMarkdownBytes", 0) > 0, "ora.cacheStats missing signature response markdown bytes")
            require(cache_stats.get("responseCodeLensCapacityBytes", 0) > 0, "ora.cacheStats missing codeLens response bytes")
            require(cache_stats.get("responseCodeLensStringBytes", 0) > 0, "ora.cacheStats missing codeLens string bytes")
            require(cache_stats.get("responseFoldingRangeCapacityBytes", 0) > 0, "ora.cacheStats missing folding response bytes")
            require(cache_stats.get("occurrenceIndexBuilds", 0) > 0, "ora.cacheStats missing occurrence index builds")
            require(cache_stats.get("openOccurrenceCount", 0) > 0, "ora.cacheStats missing occurrence count")
            require(cache_stats.get("occurrenceIndexBytes", 0) > 0, "ora.cacheStats missing occurrence bytes")
            require(cache_stats.get("importedMemberIndexBuilds", 0) > 0, "ora.cacheStats missing imported-member index builds")
            require(cache_stats.get("openImportedMemberCount", 0) > 0, "ora.cacheStats missing imported-member count")
            require(cache_stats.get("importedMemberIndexBytes", 0) > 0, "ora.cacheStats missing imported-member bytes")
            require(cache_stats.get("coldWorkspaceIndexBuilds", 0) > 0, "ora.cacheStats missing cold workspace index builds")
            require(cache_stats.get("workspaceDiscoveryRuns", 0) > 0, "ora.cacheStats missing workspace discovery runs")
            require(cache_stats.get("workspaceDiscoveryCacheHits", 0) > 0, "ora.cacheStats missing workspace discovery cache hits")
            require(cache_stats.get("importResolutionBytes", 0) > 0, "ora.cacheStats missing import-resolution cache bytes")
            require(cache_stats.get("staleDocumentChangeSkips", 0) > 0, "ora.cacheStats missing stale didChange skip count")
            phase_keys = (
                "lexBuilds",
                "parseBuilds",
                "astLowerBuilds",
                "itemIndexBuilds",
                "resolveBuilds",
                "constEvalBuilds",
                "typeCheckBuilds",
            )
            for key in phase_keys:
                require(cache_stats.get(key, 0) > 0, f"ora.cacheStats missing compiler phase counter {key}")
            require(cache_stats.get("formatterBuilds", 0) > 0, "ora.cacheStats missing formatter counter")
            require(cache_stats.get("formattingCacheEntries", 0) > 0, "ora.cacheStats missing formatting cache entries")
            require(cache_stats.get("formattingCacheBytes", 0) > 0, "ora.cacheStats missing formatting cache bytes")
            allocator_keys = (
                "serverAllocatorAllocCalls",
                "serverAllocatorBytesAllocated",
                "serverAllocatorBytesLive",
                "serverAllocatorBytesPeak",
            )
            for key in allocator_keys:
                require(cache_stats.get(key, 0) > 0, f"ora.cacheStats missing server allocator counter {key}")
            scoped_allocator_alloc_call_keys = (
                "serverAllocatorUnscopedAllocCalls",
                "serverAllocatorRequestAllocCalls",
                "serverAllocatorResponseAllocCalls",
                "serverAllocatorCacheBuildAllocCalls",
                "serverAllocatorTempAnalysisAllocCalls",
            )
            scoped_allocator_byte_keys = (
                "serverAllocatorUnscopedBytesAllocated",
                "serverAllocatorRequestBytesAllocated",
                "serverAllocatorResponseBytesAllocated",
                "serverAllocatorCacheBuildBytesAllocated",
                "serverAllocatorTempAnalysisBytesAllocated",
            )
            for key in (*scoped_allocator_alloc_call_keys, *scoped_allocator_byte_keys):
                require(key in cache_stats, f"ora.cacheStats missing scoped server allocator counter {key}")
            for key in (
                "serverAllocatorRequestBytesAllocated",
                "serverAllocatorResponseBytesAllocated",
                "serverAllocatorCacheBuildBytesAllocated",
                "serverAllocatorTempAnalysisBytesAllocated",
            ):
                require(cache_stats.get(key, 0) > 0, f"ora.cacheStats scoped server allocator counter did not move {key}")
            require(
                sum(cache_stats.get(key, 0) for key in scoped_allocator_alloc_call_keys) == cache_stats.get("serverAllocatorAllocCalls", 0),
                "ora.cacheStats scoped allocator alloc calls do not sum to aggregate alloc calls",
            )
            require(
                sum(cache_stats.get(key, 0) for key in scoped_allocator_byte_keys) == cache_stats.get("serverAllocatorBytesAllocated", 0),
                "ora.cacheStats scoped allocator bytes do not sum to aggregate allocated bytes",
            )
        finally:
            client.close()

    print(
        "ok: JSON-RPC smoke covered diagnostics, document symbols, hover docs, completion, "
        "definition, cross-file unopened definition, cold references/rename, references, "
        "document highlights, signature help, semantic tokens, formatting, code lens, folding, "
        "stale didChange skips, and cache stats"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"lsp-jsonrpc-smoke: {exc}", file=sys.stderr)
        raise
