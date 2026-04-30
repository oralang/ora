//! Front-end-agnostic controller layer for the Ora EVM debugger.
//!
//! Exists to give the TUI and the DAP server a single shared
//! query surface for things they both need: binding resolution
//! by name, expression evaluation, eventually scope listing and
//! variable formatting. Today's scope is the eval path —
//! `evaluateExpr` is the gating capability for DAP's `evaluate`
//! request, and lifting it here means DAP no longer has to
//! import `debug_tui.zig` to reuse it.
//!
//! What's intentionally NOT yet on the controller (kept on Ui
//! while the boundary stabilises):
//!   - ABI document loading / per-frame ABI lookup
//!   - ABI-decoded value formatting (uses TUI render_scratch)
//!   - ABI param decoding for SSA function arguments
//!   - Decoded revert / log formatting
//! These are next on the lift-list once DAP needs `scopes` /
//! `variables` / decoded-call-frame names. The seam is in
//! place — adding fields here and routing existing call sites
//! through the controller is mechanical.

const std = @import("std");
const evm_mod = @import("evm.zig");
const evm_config = @import("evm_config.zig");
const debugger_mod = @import("debugger.zig");
const debug_eval = @import("debug_eval.zig");

/// Generic over an EvmConfig so the controller binds to the
/// same Debugger / Evm instantiation the front-end uses. The
/// front-end passes `Session(.{})` -> `Debugger(.{})` and the
/// controller picks the matching parameterised types.
pub fn DebugController(comptime config: evm_config.EvmConfig) type {
    return struct {
        const Self = @This();
        pub const Debugger = debugger_mod.Debugger(config);
        pub const Evm = evm_mod.Evm(config);

        allocator: std.mem.Allocator,
        debugger: *Debugger,

        pub fn init(allocator: std.mem.Allocator, debugger: *Debugger) Self {
            return .{ .allocator = allocator, .debugger = debugger };
        }

        /// Resolve a binding name to its numeric value at the
        /// current stop. Returns:
        ///   - `null` when the name isn't a visible binding at
        ///     this stop.
        ///   - `error.BindingUnavailable` when the name resolves
        ///     to a visible binding whose value can't be read
        ///     (e.g. an SSA local that's been optimised away).
        ///   - The numeric value otherwise. Storage / memory /
        ///     transient-storage roots are read directly; folded
        ///     bindings parse their literal text.
        pub fn resolveBindingNumeric(self: *Self, name: []const u8) debug_eval.EvalError!?u256 {
            const binding_opt = self.debugger.findVisibleBindingByName(self.allocator, name) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
            };
            const binding = binding_opt orelse return null;

            if (binding.folded_value) |text| {
                const parsed = std.fmt.parseUnsigned(u256, text, 0) catch return error.BindingUnavailable;
                return parsed;
            }

            const numeric_opt = try self.debugger.getVisibleBindingValueByName(self.allocator, name);
            if (numeric_opt) |value| return value;

            return error.BindingUnavailable;
        }

        /// Evaluate a side-effect-free expression against the
        /// debugger's visible bindings. Front-end-agnostic: same
        /// path the TUI's `:eval <expr>` and DAP's `evaluate`
        /// request both reach.
        ///
        /// The resolver layer only knows about numeric bindings.
        /// ABI-param decoding (function args read from calldata
        /// when the binding's runtime_kind is `ssa`) is currently
        /// only available through the TUI's bigger resolver and
        /// should be lifted here when DAP's `variables` request
        /// lands.
        pub fn evaluateExpr(self: *Self, expr: []const u8) debug_eval.EvalError!debug_eval.Value {
            const resolver = debug_eval.Resolver{
                .ctx = @ptrCast(self),
                .resolveFn = resolveForEval,
            };
            return debug_eval.evaluate(expr, resolver);
        }

        fn resolveForEval(ctx: *anyopaque, name: []const u8) debug_eval.EvalError!?debug_eval.Value {
            const self: *Self = @alignCast(@ptrCast(ctx));
            const numeric = try self.resolveBindingNumeric(name);
            if (numeric) |n| return debug_eval.Value{ .num = n };
            return null;
        }
    };
}
