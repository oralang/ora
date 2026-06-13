const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const signature_help = ora_root.lsp.signature_help;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub const BuildResult = struct {
    item: types.SignatureHelp,
    signature_count: usize,
    parameter_count: usize,
    string_bytes: usize,
    markdown_bytes: usize,
};

pub fn build(arena: Allocator, signature: signature_help.SignatureInfo) !types.SignatureHelp {
    const parameters = try arena.alloc(types.ParameterInformation, signature.parameters.len);
    for (signature.parameters, 0..) |param, index| {
        parameters[index] = .{
            .label = .{ .string = try arena.dupe(u8, param.label) },
            .documentation = null,
        };
    }

    const signatures = try arena.alloc(types.SignatureInformation, 1);
    signatures[0] = .{
        .label = try arena.dupe(u8, signature.label),
        .documentation = if (signature.documentation) |doc| .{ .MarkupContent = .{
            .kind = .markdown,
            .value = try arena.dupe(u8, doc),
        } } else null,
        .parameters = parameters,
        .activeParameter = signature.active_parameter,
    };

    return .{
        .signatures = signatures,
        .activeSignature = 0,
        .activeParameter = signature.active_parameter,
    };
}

pub fn buildFromView(arena: Allocator, view: signature_help.SignatureView) !?BuildResult {
    const detail = view.symbol.detail orelse return null;
    const parameter_count = countParameters(detail);
    const parameters = try arena.alloc(types.ParameterInformation, parameter_count);

    var string_bytes: usize = 0;
    fillParameters(parameters, detail, &string_bytes);

    const kind_prefix: []const u8 = switch (view.symbol.kind) {
        .function, .method => "fn ",
        .event => "log ",
        .error_decl => "error ",
        else => "",
    };
    const label = try std.fmt.allocPrint(arena, "{s}{s}{s}", .{ kind_prefix, view.symbol.name, detail });
    string_bytes = addSat(string_bytes, label.len);

    const markdown_bytes = if (view.symbol.doc_comment) |doc| doc.len else 0;
    const signatures = try arena.alloc(types.SignatureInformation, 1);
    signatures[0] = .{
        .label = label,
        .documentation = if (view.symbol.doc_comment) |doc| .{ .MarkupContent = .{
            .kind = .markdown,
            .value = doc,
        } } else null,
        .parameters = parameters,
        .activeParameter = view.active_parameter,
    };

    return .{
        .item = .{
            .signatures = signatures,
            .activeSignature = 0,
            .activeParameter = view.active_parameter,
        },
        .signature_count = 1,
        .parameter_count = parameter_count,
        .string_bytes = string_bytes,
        .markdown_bytes = markdown_bytes,
    };
}

fn countParameters(detail: []const u8) usize {
    if (detail.len == 0 or detail[0] != '(') return 0;

    var count: usize = 0;
    var i: usize = 1;
    while (i < detail.len and detail[i] != ')') {
        const param_start = i;
        var depth: u32 = 0;
        while (i < detail.len) : (i += 1) {
            if (detail[i] == '(' or detail[i] == '<') {
                depth += 1;
            } else if (detail[i] == ')' or detail[i] == '>') {
                if (depth == 0) break;
                depth -= 1;
            } else if (detail[i] == ',' and depth == 0) {
                break;
            }
        }
        if (std.mem.trim(u8, detail[param_start..i], " ").len > 0) count += 1;
        if (i < detail.len and detail[i] == ',') i += 1;
        while (i < detail.len and detail[i] == ' ') : (i += 1) {}
    }
    return count;
}

fn fillParameters(parameters: []types.ParameterInformation, detail: []const u8, string_bytes: *usize) void {
    if (parameters.len == 0 or detail.len == 0 or detail[0] != '(') return;

    var out: usize = 0;
    var i: usize = 1;
    while (i < detail.len and detail[i] != ')' and out < parameters.len) {
        const param_start = i;
        var depth: u32 = 0;
        while (i < detail.len) : (i += 1) {
            if (detail[i] == '(' or detail[i] == '<') {
                depth += 1;
            } else if (detail[i] == ')' or detail[i] == '>') {
                if (depth == 0) break;
                depth -= 1;
            } else if (detail[i] == ',' and depth == 0) {
                break;
            }
        }
        const param_text = std.mem.trim(u8, detail[param_start..i], " ");
        if (param_text.len > 0) {
            parameters[out] = .{
                .label = .{ .string = param_text },
                .documentation = null,
            };
            string_bytes.* = addSat(string_bytes.*, param_text.len);
            out += 1;
        }
        if (i < detail.len and detail[i] == ',') i += 1;
        while (i < detail.len and detail[i] == ' ') : (i += 1) {}
    }
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
