const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const semantic_tokens_api = ora_root.lsp.semantic_tokens;

const protocol_helpers = @import("protocol_helpers.zig");

const types = lsp.types;

pub fn initialize(
    server: anytype,
    params: types.InitializeParams,
) !types.InitializeResult {
    try configureWorkspaceRoots(server, params);
    server.position_encoding = protocol_helpers.negotiatePositionEncoding(params);

    const capabilities: types.ServerCapabilities = .{
        .positionEncoding = protocol_helpers.toLspPositionEncoding(server.position_encoding),
        .textDocumentSync = .{
            .TextDocumentSyncOptions = .{
                .openClose = true,
                .change = .Incremental,
                .save = .{ .SaveOptions = .{ .includeText = false } },
            },
        },
        .hoverProvider = .{ .bool = true },
        .definitionProvider = .{ .bool = true },
        .referencesProvider = .{ .bool = true },
        .documentSymbolProvider = .{ .bool = true },
        .completionProvider = .{
            .triggerCharacters = &[_][]const u8{ ".", ":" },
            .resolveProvider = false,
        },
        .renameProvider = .{ .RenameOptions = .{ .prepareProvider = true } },
        .documentHighlightProvider = .{ .bool = true },
        .foldingRangeProvider = .{ .bool = true },
        .workspaceSymbolProvider = .{ .bool = true },
        .codeActionProvider = .{ .CodeActionOptions = .{
            .codeActionKinds = &.{ .quickfix, .@"source.organizeImports" },
        } },
        .selectionRangeProvider = .{ .bool = true },
        .documentLinkProvider = .{ .resolveProvider = false },
        .callHierarchyProvider = .{ .bool = true },
        .documentFormattingProvider = .{ .bool = true },
        .inlayHintProvider = .{ .InlayHintOptions = .{ .resolveProvider = false } },
        .codeLensProvider = .{ .resolveProvider = false },
        .executeCommandProvider = .{ .commands = &[_][]const u8{"ora.cacheStats"} },
        .signatureHelpProvider = .{
            .triggerCharacters = &[_][]const u8{ "(", "," },
            .retriggerCharacters = &[_][]const u8{","},
        },
        .semanticTokensProvider = .{
            .SemanticTokensOptions = .{
                .legend = .{
                    .tokenTypes = &semantic_tokens_api.SemanticTokenKind.legend,
                    .tokenModifiers = &semantic_tokens_api.SemanticTokenModifier.legend,
                },
                .full = .{ .bool = true },
            },
        },
    };

    if (@import("builtin").mode == .Debug) {
        lsp.basic_server.validateServerCapabilities(@TypeOf(server.*), capabilities);
    }

    return .{
        .capabilities = capabilities,
        .serverInfo = .{
            .name = "ora-lsp",
            .version = "0.1.0",
        },
    };
}

fn configureWorkspaceRoots(server: anytype, params: types.InitializeParams) !void {
    clearWorkspaceRoots(server);
    try server.workspace_roots.addInitializeRoots(params);
}

fn clearWorkspaceRoots(server: anytype) void {
    server.workspace_roots.clear();
    server.docs.clearImportResolutionCache();
    server.workspace_discovery.clear();
}
