const std = @import("std");
const ora_root = @import("ora_root");

const compiler = ora_root.compiler;
const definition = ora_root.lsp.definition;
const semantic_index = ora_root.lsp.semantic_index;

pub fn semanticIndex(allocator: std.mem.Allocator, source: []const u8) !semantic_index.SemanticIndex {
    var fixture: TestAnalysis = undefined;
    try fixture.init(allocator, source);
    defer fixture.deinit();

    return fixture.buildSemanticIndex(allocator, source);
}

pub const TestAnalysis = struct {
    allocator: std.mem.Allocator,
    db: compiler.CompilerDb,
    analysis: definition.Analysis = undefined,
    initialized: bool = false,

    pub fn init(self: *TestAnalysis, allocator: std.mem.Allocator, source: []const u8) !void {
        self.* = .{
            .allocator = allocator,
            .db = compiler.CompilerDb.init(allocator),
        };
        errdefer self.db.deinitFrontendOnly();

        const package_id = try self.db.addPackage("lsp-test");
        const file_id = try self.db.addSourceFile("<lsp-test>", source);
        const module_id = try self.db.addModule(package_id, file_id, "lsp-test");
        const ast_file = try self.db.astFile(file_id);
        const item_index = try self.db.itemIndex(module_id);
        const resolution = try self.db.resolveNames(module_id);
        self.analysis = definition.Analysis.initBorrowed(
            allocator,
            &self.db.sources,
            file_id,
            module_id,
            source,
            ast_file,
            item_index,
            resolution,
        );
        self.initialized = true;
    }

    pub fn deinit(self: *TestAnalysis) void {
        if (self.initialized) self.analysis.deinit();
        self.db.deinitFrontendOnly();
        self.* = undefined;
    }

    pub fn buildSemanticIndex(self: *TestAnalysis, allocator: std.mem.Allocator, source: []const u8) !semantic_index.SemanticIndex {
        const parse_succeeded =
            (try self.db.syntaxDiagnostics(self.analysis.file_id)).isEmpty() and
            (try self.db.astDiagnostics(self.analysis.file_id)).isEmpty();
        return semantic_index.indexAstFileWithSourceStoreAlloc(
            allocator,
            allocator,
            &self.db.sources,
            self.analysis.file_id,
            source,
            self.analysis.ast_file,
            parse_succeeded,
        );
    }
};
