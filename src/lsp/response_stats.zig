const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const cache_stats_response = @import("cache_stats_response.zig");
const code_action_response = @import("code_action.zig");
const semantic_index = ora_root.lsp.semantic_index;

const types = lsp.types;

pub const Kind = enum {
    location,
    text_edit,
    completion_item,
    semantic_token_data,
    workspace_symbol,
    call_hierarchy,
    hover,
    definition,
    document_symbol,
    document_highlight,
    inlay_hint,
    code_lens,
    formatting_edit,
    selection_range,
    folding_range,
    document_link,
    code_action,
    signature_help,
    prepare_rename,
};

pub const Stats = struct {
    builder_items_built: usize = 0,
    builder_capacity_bytes: usize = 0,
    location_capacity_bytes: usize = 0,
    text_edit_capacity_bytes: usize = 0,
    completion_item_capacity_bytes: usize = 0,
    semantic_token_data_capacity_bytes: usize = 0,
    workspace_symbol_capacity_bytes: usize = 0,
    call_hierarchy_capacity_bytes: usize = 0,
    hover_capacity_bytes: usize = 0,
    definition_capacity_bytes: usize = 0,
    document_symbol_capacity_bytes: usize = 0,
    document_highlight_capacity_bytes: usize = 0,
    inlay_hint_capacity_bytes: usize = 0,
    code_lens_capacity_bytes: usize = 0,
    formatting_edit_capacity_bytes: usize = 0,
    selection_range_capacity_bytes: usize = 0,
    folding_range_capacity_bytes: usize = 0,
    document_link_capacity_bytes: usize = 0,
    code_action_capacity_bytes: usize = 0,
    signature_help_capacity_bytes: usize = 0,
    prepare_rename_capacity_bytes: usize = 0,
    string_bytes: usize = 0,
    markdown_bytes: usize = 0,
    location_string_bytes: usize = 0,
    text_edit_string_bytes: usize = 0,
    completion_string_bytes: usize = 0,
    completion_markdown_bytes: usize = 0,
    hover_string_bytes: usize = 0,
    hover_markdown_bytes: usize = 0,
    definition_string_bytes: usize = 0,
    signature_string_bytes: usize = 0,
    signature_markdown_bytes: usize = 0,
    document_symbol_string_bytes: usize = 0,
    workspace_symbol_string_bytes: usize = 0,
    call_hierarchy_string_bytes: usize = 0,
    inlay_hint_string_bytes: usize = 0,
    code_lens_string_bytes: usize = 0,
    formatting_edit_string_bytes: usize = 0,
    document_link_string_bytes: usize = 0,
    code_action_string_bytes: usize = 0,
    prepare_rename_string_bytes: usize = 0,

    pub fn recordItems(self: *Stats, comptime kind: Kind, comptime Item: type, count: usize) void {
        const bytes = mulSat(count, @sizeOf(Item));
        self.builder_items_built = addSat(self.builder_items_built, count);
        self.builder_capacity_bytes = addSat(self.builder_capacity_bytes, bytes);
        switch (kind) {
            .location => self.location_capacity_bytes = addSat(self.location_capacity_bytes, bytes),
            .text_edit => self.text_edit_capacity_bytes = addSat(self.text_edit_capacity_bytes, bytes),
            .completion_item => self.completion_item_capacity_bytes = addSat(self.completion_item_capacity_bytes, bytes),
            .semantic_token_data => self.semantic_token_data_capacity_bytes = addSat(self.semantic_token_data_capacity_bytes, bytes),
            .workspace_symbol => self.workspace_symbol_capacity_bytes = addSat(self.workspace_symbol_capacity_bytes, bytes),
            .call_hierarchy => self.call_hierarchy_capacity_bytes = addSat(self.call_hierarchy_capacity_bytes, bytes),
            .hover => self.hover_capacity_bytes = addSat(self.hover_capacity_bytes, bytes),
            .definition => self.definition_capacity_bytes = addSat(self.definition_capacity_bytes, bytes),
            .document_symbol => self.document_symbol_capacity_bytes = addSat(self.document_symbol_capacity_bytes, bytes),
            .document_highlight => self.document_highlight_capacity_bytes = addSat(self.document_highlight_capacity_bytes, bytes),
            .inlay_hint => self.inlay_hint_capacity_bytes = addSat(self.inlay_hint_capacity_bytes, bytes),
            .code_lens => self.code_lens_capacity_bytes = addSat(self.code_lens_capacity_bytes, bytes),
            .formatting_edit => self.formatting_edit_capacity_bytes = addSat(self.formatting_edit_capacity_bytes, bytes),
            .selection_range => self.selection_range_capacity_bytes = addSat(self.selection_range_capacity_bytes, bytes),
            .folding_range => self.folding_range_capacity_bytes = addSat(self.folding_range_capacity_bytes, bytes),
            .document_link => self.document_link_capacity_bytes = addSat(self.document_link_capacity_bytes, bytes),
            .code_action => self.code_action_capacity_bytes = addSat(self.code_action_capacity_bytes, bytes),
            .signature_help => self.signature_help_capacity_bytes = addSat(self.signature_help_capacity_bytes, bytes),
            .prepare_rename => self.prepare_rename_capacity_bytes = addSat(self.prepare_rename_capacity_bytes, bytes),
        }
    }

    pub fn recordStringBytes(self: *Stats, comptime kind: Kind, bytes: usize) void {
        if (bytes == 0) return;
        self.string_bytes = addSat(self.string_bytes, bytes);
        switch (kind) {
            .location => self.location_string_bytes = addSat(self.location_string_bytes, bytes),
            .text_edit => self.text_edit_string_bytes = addSat(self.text_edit_string_bytes, bytes),
            .completion_item => self.completion_string_bytes = addSat(self.completion_string_bytes, bytes),
            .workspace_symbol => self.workspace_symbol_string_bytes = addSat(self.workspace_symbol_string_bytes, bytes),
            .call_hierarchy => self.call_hierarchy_string_bytes = addSat(self.call_hierarchy_string_bytes, bytes),
            .hover => self.hover_string_bytes = addSat(self.hover_string_bytes, bytes),
            .definition => self.definition_string_bytes = addSat(self.definition_string_bytes, bytes),
            .document_symbol => self.document_symbol_string_bytes = addSat(self.document_symbol_string_bytes, bytes),
            .inlay_hint => self.inlay_hint_string_bytes = addSat(self.inlay_hint_string_bytes, bytes),
            .code_lens => self.code_lens_string_bytes = addSat(self.code_lens_string_bytes, bytes),
            .formatting_edit => self.formatting_edit_string_bytes = addSat(self.formatting_edit_string_bytes, bytes),
            .document_link => self.document_link_string_bytes = addSat(self.document_link_string_bytes, bytes),
            .code_action => self.code_action_string_bytes = addSat(self.code_action_string_bytes, bytes),
            .prepare_rename => self.prepare_rename_string_bytes = addSat(self.prepare_rename_string_bytes, bytes),
            .signature_help => self.signature_string_bytes = addSat(self.signature_string_bytes, bytes),
            .semantic_token_data, .document_highlight, .selection_range, .folding_range => {},
        }
    }

    pub fn recordMarkdownBytes(self: *Stats, comptime kind: Kind, bytes: usize) void {
        if (bytes == 0) return;
        self.recordStringBytes(kind, bytes);
        self.markdown_bytes = addSat(self.markdown_bytes, bytes);
        switch (kind) {
            .completion_item => self.completion_markdown_bytes = addSat(self.completion_markdown_bytes, bytes),
            .hover => self.hover_markdown_bytes = addSat(self.hover_markdown_bytes, bytes),
            .signature_help => self.signature_markdown_bytes = addSat(self.signature_markdown_bytes, bytes),
            else => {},
        }
    }

    pub fn writeSnapshot(self: *const Stats, snapshot: *cache_stats_response.Snapshot) void {
        snapshot.response_builder_items_built = self.builder_items_built;
        snapshot.response_builder_capacity_bytes = self.builder_capacity_bytes;
        snapshot.response_builder_item_bytes = self.builder_capacity_bytes;
        snapshot.response_location_capacity_bytes = self.location_capacity_bytes;
        snapshot.response_text_edit_capacity_bytes = self.text_edit_capacity_bytes;
        snapshot.response_completion_item_capacity_bytes = self.completion_item_capacity_bytes;
        snapshot.response_semantic_token_data_capacity_bytes = self.semantic_token_data_capacity_bytes;
        snapshot.response_workspace_symbol_capacity_bytes = self.workspace_symbol_capacity_bytes;
        snapshot.response_call_hierarchy_capacity_bytes = self.call_hierarchy_capacity_bytes;
        snapshot.response_hover_capacity_bytes = self.hover_capacity_bytes;
        snapshot.response_definition_capacity_bytes = self.definition_capacity_bytes;
        snapshot.response_document_symbol_capacity_bytes = self.document_symbol_capacity_bytes;
        snapshot.response_document_highlight_capacity_bytes = self.document_highlight_capacity_bytes;
        snapshot.response_inlay_hint_capacity_bytes = self.inlay_hint_capacity_bytes;
        snapshot.response_code_lens_capacity_bytes = self.code_lens_capacity_bytes;
        snapshot.response_formatting_edit_capacity_bytes = self.formatting_edit_capacity_bytes;
        snapshot.response_selection_range_capacity_bytes = self.selection_range_capacity_bytes;
        snapshot.response_folding_range_capacity_bytes = self.folding_range_capacity_bytes;
        snapshot.response_document_link_capacity_bytes = self.document_link_capacity_bytes;
        snapshot.response_code_action_capacity_bytes = self.code_action_capacity_bytes;
        snapshot.response_signature_help_capacity_bytes = self.signature_help_capacity_bytes;
        snapshot.response_prepare_rename_capacity_bytes = self.prepare_rename_capacity_bytes;
        snapshot.response_string_bytes = self.string_bytes;
        snapshot.response_markdown_bytes = self.markdown_bytes;
        snapshot.response_location_string_bytes = self.location_string_bytes;
        snapshot.response_text_edit_string_bytes = self.text_edit_string_bytes;
        snapshot.response_completion_string_bytes = self.completion_string_bytes;
        snapshot.response_completion_markdown_bytes = self.completion_markdown_bytes;
        snapshot.response_hover_string_bytes = self.hover_string_bytes;
        snapshot.response_hover_markdown_bytes = self.hover_markdown_bytes;
        snapshot.response_definition_string_bytes = self.definition_string_bytes;
        snapshot.response_signature_string_bytes = self.signature_string_bytes;
        snapshot.response_signature_markdown_bytes = self.signature_markdown_bytes;
        snapshot.response_document_symbol_string_bytes = self.document_symbol_string_bytes;
        snapshot.response_workspace_symbol_string_bytes = self.workspace_symbol_string_bytes;
        snapshot.response_call_hierarchy_string_bytes = self.call_hierarchy_string_bytes;
        snapshot.response_inlay_hint_string_bytes = self.inlay_hint_string_bytes;
        snapshot.response_code_lens_string_bytes = self.code_lens_string_bytes;
        snapshot.response_formatting_edit_string_bytes = self.formatting_edit_string_bytes;
        snapshot.response_document_link_string_bytes = self.document_link_string_bytes;
        snapshot.response_code_action_string_bytes = self.code_action_string_bytes;
        snapshot.response_prepare_rename_string_bytes = self.prepare_rename_string_bytes;
    }
};

pub fn semanticSymbolStringBytes(symbols: []const semantic_index.Symbol) usize {
    var total: usize = 0;
    for (symbols) |symbol| {
        total = addSat(total, symbol.name.len);
        if (symbol.detail) |detail| total = addSat(total, detail.len);
    }
    return total;
}

pub fn callHierarchyItemStringBytes(items: []const types.CallHierarchyItem) usize {
    var total: usize = 0;
    for (items) |item| {
        total = addSat(total, item.name.len);
        total = addSat(total, item.uri.len);
        if (item.detail) |detail| total = addSat(total, detail.len);
    }
    return total;
}

pub fn incomingCallStringBytes(items: []const types.CallHierarchyIncomingCall) usize {
    var total: usize = 0;
    for (items) |item| {
        total = addSat(total, item.from.name.len);
        total = addSat(total, item.from.uri.len);
        if (item.from.detail) |detail| total = addSat(total, detail.len);
    }
    return total;
}

pub fn outgoingCallStringBytes(items: []const types.CallHierarchyOutgoingCall) usize {
    var total: usize = 0;
    for (items) |item| {
        total = addSat(total, item.to.name.len);
        total = addSat(total, item.to.uri.len);
        if (item.to.detail) |detail| total = addSat(total, detail.len);
    }
    return total;
}

pub fn locationUriBytes(locations: []const types.Location) usize {
    var total: usize = 0;
    for (locations) |location| {
        total = addSat(total, location.uri.len);
    }
    return total;
}

pub fn documentLinkStringBytes(links: []const types.DocumentLink) usize {
    var total: usize = 0;
    for (links) |link| {
        if (link.target) |target| total = addSat(total, target.len);
        if (link.tooltip) |tooltip| total = addSat(total, tooltip.len);
    }
    return total;
}

pub fn codeActionStringBytes(actions: []const code_action_response.CodeActionOrCommand) usize {
    var total: usize = 0;
    for (actions) |action_or_command| {
        switch (action_or_command) {
            .CodeAction => |action| {
                total = addSat(total, action.title.len);
                if (action.diagnostics) |diagnostics| {
                    for (diagnostics) |diagnostic| {
                        total = addSat(total, diagnostic.message.len);
                        if (diagnostic.source) |source| total = addSat(total, source.len);
                    }
                }
                if (action.edit) |edit| total = addSat(total, workspaceEditStringBytes(edit));
                if (action.command) |command| total = addSat(total, commandStringBytes(command));
            },
            .Command => |command| total = addSat(total, commandStringBytes(command)),
        }
    }
    return total;
}

pub fn selectionRangeNodeCount(ranges: []const types.SelectionRange) usize {
    var total: usize = 0;
    for (ranges) |range| {
        total = addSat(total, selectionRangeNodeCountOne(range));
    }
    return total;
}

fn workspaceEditStringBytes(edit: types.WorkspaceEdit) usize {
    var total: usize = 0;
    if (edit.changes) |changes| {
        var iterator = changes.map.iterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.key_ptr.*.len);
            for (entry.value_ptr.*) |text_edit| {
                total = addSat(total, text_edit.newText.len);
            }
        }
    }
    return total;
}

fn commandStringBytes(command: types.Command) usize {
    return addSat(command.title.len, command.command.len);
}

fn selectionRangeNodeCountOne(range: types.SelectionRange) usize {
    var total: usize = 1;
    var parent = range.parent;
    while (parent) |next| {
        total = addSat(total, 1);
        parent = next.parent;
    }
    return total;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

fn mulSat(a: usize, b: usize) usize {
    return std.math.mul(usize, a, b) catch std.math.maxInt(usize);
}
