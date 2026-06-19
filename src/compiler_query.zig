//! Compiler query capability views.
//!
//! These structs are restricted views over the compiler DB/query surface used by
//! frontend stages. Keep the definitions here so stages do not grow parallel
//! mini-interfaces with the same callbacks.

const ast = @import("ast/mod.zig");
const source = @import("source/mod.zig");
const sema_model = @import("sema/model.zig");

pub const SemaView = struct {
    context: *anyopaque,
    ast_file: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const ast.AstFile,
    module_path: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror![]const u8,
    item_index: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.ItemIndexResult,
    resolution: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.NameResolutionResult,
    module_typecheck: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.TypeCheckResult,
    lookup_item: *const fn (context: *anyopaque, module_id: source.ModuleId, name: []const u8) anyerror!?ast.ItemId,
    resolve_import_alias: *const fn (context: *anyopaque, module_id: source.ModuleId, alias: []const u8) anyerror!?source.ModuleId,

    pub fn astFile(self: *const SemaView, module_id: source.ModuleId) anyerror!*const ast.AstFile {
        return self.ast_file(self.context, module_id);
    }

    pub fn modulePath(self: *const SemaView, module_id: source.ModuleId) anyerror![]const u8 {
        return self.module_path(self.context, module_id);
    }

    pub fn itemIndex(self: *const SemaView, module_id: source.ModuleId) anyerror!*const sema_model.ItemIndexResult {
        return self.item_index(self.context, module_id);
    }

    pub fn nameResolution(self: *const SemaView, module_id: source.ModuleId) anyerror!*const sema_model.NameResolutionResult {
        return self.resolution(self.context, module_id);
    }

    pub fn moduleTypeCheck(self: *const SemaView, module_id: source.ModuleId) anyerror!*const sema_model.TypeCheckResult {
        return self.module_typecheck(self.context, module_id);
    }

    pub fn lookupItem(self: *const SemaView, module_id: source.ModuleId, name: []const u8) anyerror!?ast.ItemId {
        return self.lookup_item(self.context, module_id, name);
    }

    pub fn resolveImportAlias(self: *const SemaView, module_id: source.ModuleId, alias: []const u8) anyerror!?source.ModuleId {
        return self.resolve_import_alias(self.context, module_id, alias);
    }
};

pub const ComptimeView = struct {
    context: *anyopaque,
    ensure_typecheck: *const fn (context: *anyopaque, module_id: source.ModuleId, key: sema_model.TypeCheckKey) anyerror!*const sema_model.TypeCheckResult,
    module_typecheck: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.TypeCheckResult,
    item_index: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.ItemIndexResult,
    const_eval: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.ConstEvalResult,
    ast_file: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const ast.AstFile,
    lookup_item: *const fn (context: *anyopaque, module_id: source.ModuleId, name: []const u8) anyerror!?ast.ItemId,
    resolve_import_alias: *const fn (context: *anyopaque, module_id: source.ModuleId, alias: []const u8) anyerror!?source.ModuleId,

    pub fn ensureTypeCheck(self: *const ComptimeView, module_id: source.ModuleId, key: sema_model.TypeCheckKey) anyerror!*const sema_model.TypeCheckResult {
        return self.ensure_typecheck(self.context, module_id, key);
    }

    pub fn moduleTypeCheck(self: *const ComptimeView, module_id: source.ModuleId) anyerror!*const sema_model.TypeCheckResult {
        return self.module_typecheck(self.context, module_id);
    }

    pub fn itemIndex(self: *const ComptimeView, module_id: source.ModuleId) anyerror!*const sema_model.ItemIndexResult {
        return self.item_index(self.context, module_id);
    }

    pub fn constEval(self: *const ComptimeView, module_id: source.ModuleId) anyerror!*const sema_model.ConstEvalResult {
        return self.const_eval(self.context, module_id);
    }

    pub fn astFile(self: *const ComptimeView, module_id: source.ModuleId) anyerror!*const ast.AstFile {
        return self.ast_file(self.context, module_id);
    }

    pub fn lookupItem(self: *const ComptimeView, module_id: source.ModuleId, name: []const u8) anyerror!?ast.ItemId {
        return self.lookup_item(self.context, module_id, name);
    }

    pub fn resolveImportAlias(self: *const ComptimeView, module_id: source.ModuleId, alias: []const u8) anyerror!?source.ModuleId {
        return self.resolve_import_alias(self.context, module_id, alias);
    }
};

pub const HirView = struct {
    context: *anyopaque,
    ast_file: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const ast.AstFile,
    module_path: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror![]const u8,
    item_index: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.ItemIndexResult,
    resolution: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.NameResolutionResult,
    module_typecheck: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.TypeCheckResult,
    module_verification_facts: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.ModuleVerificationFactsResult,
    const_eval: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema_model.ConstEvalResult,
    lookup_item: *const fn (context: *anyopaque, module_id: source.ModuleId, name: []const u8) anyerror!?ast.ItemId,
    resolve_import_alias: *const fn (context: *anyopaque, module_id: source.ModuleId, alias: []const u8) anyerror!?source.ModuleId,

    pub fn astFile(self: *const HirView, module_id: source.ModuleId) anyerror!*const ast.AstFile {
        return self.ast_file(self.context, module_id);
    }

    pub fn modulePath(self: *const HirView, module_id: source.ModuleId) anyerror![]const u8 {
        return self.module_path(self.context, module_id);
    }

    pub fn itemIndex(self: *const HirView, module_id: source.ModuleId) anyerror!*const sema_model.ItemIndexResult {
        return self.item_index(self.context, module_id);
    }

    pub fn nameResolution(self: *const HirView, module_id: source.ModuleId) anyerror!*const sema_model.NameResolutionResult {
        return self.resolution(self.context, module_id);
    }

    pub fn moduleTypeCheck(self: *const HirView, module_id: source.ModuleId) anyerror!*const sema_model.TypeCheckResult {
        return self.module_typecheck(self.context, module_id);
    }

    pub fn moduleVerificationFacts(self: *const HirView, module_id: source.ModuleId) anyerror!*const sema_model.ModuleVerificationFactsResult {
        return self.module_verification_facts(self.context, module_id);
    }

    pub fn constEval(self: *const HirView, module_id: source.ModuleId) anyerror!*const sema_model.ConstEvalResult {
        return self.const_eval(self.context, module_id);
    }

    pub fn lookupItem(self: *const HirView, module_id: source.ModuleId, name: []const u8) anyerror!?ast.ItemId {
        return self.lookup_item(self.context, module_id, name);
    }

    pub fn resolveImportAlias(self: *const HirView, module_id: source.ModuleId, alias: []const u8) anyerror!?source.ModuleId {
        return self.resolve_import_alias(self.context, module_id, alias);
    }
};
