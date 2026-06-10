const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const signature_help = ora_root.lsp.signature_help;

const Allocator = std.mem.Allocator;
const types = lsp.types;

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
