/// Source-level debugger for the Ora EVM.
/// Wraps the EVM with breakpoints, statement-level stepping, and state inspection.
/// Does NOT modify evm.zig or frame.zig — it drives them externally via step().
const std = @import("std");
const source_map = @import("source_map.zig");
const SourceMap = source_map.SourceMap;
const debug_info_mod = @import("debug_info.zig");
const DebugInfo = debug_info_mod.DebugInfo;
const opcode_utils = @import("opcode.zig");
const evm_mod = @import("evm.zig");
const evm_config = @import("evm_config.zig");

pub fn Debugger(comptime config: evm_config.EvmConfig) type {
    const EvmType = evm_mod.Evm(config);

    return struct {
        const Self = @This();

        evm: *EvmType,
        src_map: SourceMap,
        debug_info: ?DebugInfo,
        source_text: []const u8,
        source_lines: []const LineSlice,
        breakpoints: std.AutoHashMap(u32, void),
        state: State,
        stop_reason: StopReason,
        allocator: std.mem.Allocator,
        /// Total opcodes executed since last user command
        steps_executed: u64,
        /// Safety limit to prevent infinite loops
        max_steps: u64,

        pub const State = enum {
            paused,
            running,
            halted,
        };

        pub const StopReason = enum {
            not_started,
            breakpoint_hit,
            step_complete,
            execution_finished,
            execution_reverted,
            execution_error,
            step_limit_reached,
        };

        pub const LineSlice = struct {
            start: u32,
            end: u32,
        };

        const StatementKey = struct {
            depth: usize,
            stmt_pc: u32,
        };

        pub fn init(
            allocator: std.mem.Allocator,
            evm: *EvmType,
            src_map: SourceMap,
            source_text: []const u8,
        ) !Self {
            const lines = try buildLineIndex(allocator, source_text);
            return Self{
                .evm = evm,
                .src_map = src_map,
                .debug_info = null,
                .source_text = source_text,
                .source_lines = lines,
                .breakpoints = std.AutoHashMap(u32, void).init(allocator),
                .state = .paused,
                .stop_reason = .not_started,
                .allocator = allocator,
                .steps_executed = 0,
                .max_steps = 10_000_000,
            };
        }

        pub fn deinit(self: *Self) void {
            self.breakpoints.deinit();
            if (self.debug_info) |*debug_info| debug_info.deinit();
            self.allocator.free(self.source_lines);
        }

        pub fn initWithDebugInfo(
            allocator: std.mem.Allocator,
            evm: *EvmType,
            src_map: SourceMap,
            debug_info: DebugInfo,
            source_text: []const u8,
        ) !Self {
            var self = try Self.init(allocator, evm, src_map, source_text);
            self.debug_info = debug_info;
            return self;
        }

        // ====================================================================
        // Breakpoints
        // ====================================================================

        /// Set a breakpoint on a source line. Resolves to the first statement PC for that line.
        pub fn setBreakpoint(self: *Self, file: []const u8, line: u32) bool {
            if (self.src_map.getFirstPcForLine(file, line)) |pc| {
                self.breakpoints.put(pc, {}) catch return false;
                return true;
            }
            return false;
        }

        /// Remove a breakpoint from a source line.
        pub fn removeBreakpoint(self: *Self, file: []const u8, line: u32) void {
            if (self.src_map.getFirstPcForLine(file, line)) |pc| {
                _ = self.breakpoints.remove(pc);
            }
        }

        /// Toggle a breakpoint on a source line. Returns true if now set, false if removed.
        pub fn toggleBreakpoint(self: *Self, file: []const u8, line: u32) bool {
            if (self.src_map.getFirstPcForLine(file, line)) |pc| {
                if (self.breakpoints.contains(pc)) {
                    _ = self.breakpoints.remove(pc);
                    return false;
                } else {
                    self.breakpoints.put(pc, {}) catch return false;
                    return true;
                }
            }
            return false;
        }

        /// Check if a source line has a breakpoint set.
        pub fn hasBreakpoint(self: *const Self, file: []const u8, line: u32) bool {
            if (self.src_map.getFirstPcForLine(file, line)) |pc| {
                return self.breakpoints.contains(pc);
            }
            return false;
        }

        // ====================================================================
        // Stepping commands
        // ====================================================================

        /// Step In: execute until the source line changes (at any call depth).
        /// Enters function calls.
        pub fn stepIn(self: *Self) !void {
            if (self.isHalted()) return;

            const start_statement = self.currentStatementKey();
            self.steps_executed = 0;

            while (true) {
                try self.executeOneOpcode();
                if (self.isHalted()) return;
                if (self.steps_executed >= self.max_steps) {
                    self.stop_reason = .step_limit_reached;
                    self.state = .paused;
                    return;
                }

                if (self.currentStatementKeyChanged(start_statement)) {
                    self.stop_reason = .step_complete;
                    self.state = .paused;
                    return;
                }
            }
        }

        /// Step Over: execute until the source line changes at the same or lower call depth.
        /// Skips over function calls.
        pub fn stepOver(self: *Self) !void {
            if (self.isHalted()) return;

            const start_statement = self.currentStatementKey();
            const start_depth = self.getCallDepth();
            self.steps_executed = 0;

            while (true) {
                try self.executeOneOpcode();
                if (self.isHalted()) return;
                if (self.steps_executed >= self.max_steps) {
                    self.stop_reason = .step_limit_reached;
                    self.state = .paused;
                    return;
                }

                const current_depth = self.getCallDepth();
                // Only consider stopping if we're at same or shallower depth
                if (current_depth <= start_depth) {
                    if (self.currentStatementKeyChanged(start_statement)) {
                        self.stop_reason = .step_complete;
                        self.state = .paused;
                        return;
                    }
                }
            }
        }

        /// Step Out: execute until the call depth is less than current.
        /// Returns from the current function.
        pub fn stepOut(self: *Self) !void {
            if (self.isHalted()) return;

            const start_depth = self.getCallDepth();
            if (start_depth == 0) {
                // Already at top level — run to completion
                return self.continue_();
            }

            self.steps_executed = 0;

            while (true) {
                try self.executeOneOpcode();
                if (self.isHalted()) return;
                if (self.steps_executed >= self.max_steps) {
                    self.stop_reason = .step_limit_reached;
                    self.state = .paused;
                    return;
                }

                if (self.getCallDepth() < start_depth) {
                    // Wait for the next statement boundary after returning
                    if (self.isAtStatementBoundary()) {
                        self.stop_reason = .step_complete;
                        self.state = .paused;
                        return;
                    }
                }
            }
        }

        /// Continue: run until a breakpoint is hit or execution finishes.
        pub fn continue_(self: *Self) !void {
            if (self.isHalted()) return;

            self.state = .running;
            self.steps_executed = 0;

            // Step past current PC first (so we don't immediately re-hit the same breakpoint)
            try self.executeOneOpcode();
            if (self.isHalted()) return;

            while (true) {
                if (self.steps_executed >= self.max_steps) {
                    self.stop_reason = .step_limit_reached;
                    self.state = .paused;
                    return;
                }

                // Check breakpoint before executing
                const pc = self.getPC();
                if (self.breakpoints.contains(pc) and self.isAtStatementBoundary()) {
                    self.stop_reason = .breakpoint_hit;
                    self.state = .paused;
                    return;
                }

                try self.executeOneOpcode();
                if (self.isHalted()) return;
            }
        }

        // ====================================================================
        // State queries
        // ====================================================================

        /// Get current source location for the PC.
        pub fn currentEntry(self: *const Self) ?*const SourceMap.Entry {
            const frame = self.evm.getCurrentFrame() orelse return null;
            return self.src_map.getEntry(frame.pc);
        }

        pub fn currentOpIndex(self: *const Self) ?u32 {
            const entry = self.currentEntry() orelse return null;
            return entry.idx;
        }

        pub fn currentVisibility(self: *const Self) ?*const DebugInfo.OpVisibility {
            const info = self.debug_info orelse return null;
            const idx = self.currentOpIndex() orelse return null;
            return info.getVisibilityForIdx(idx);
        }

        pub fn getVisibleLocals(self: *const Self, allocator: std.mem.Allocator) ![]DebugInfo.VisibleLocal {
            const info = self.debug_info orelse return &.{};
            const idx = self.currentOpIndex() orelse return &.{};
            return try info.collectVisibleLocals(allocator, idx);
        }

        pub fn getVisibleBindings(self: *const Self, allocator: std.mem.Allocator) ![]DebugInfo.VisibleBinding {
            const info = self.debug_info orelse return &.{};
            const idx = self.currentOpIndex() orelse return &.{};
            return try info.collectVisibleBindings(allocator, idx);
        }

        pub fn getWritableVisibleBindings(self: *const Self, allocator: std.mem.Allocator) ![]DebugInfo.VisibleBinding {
            const bindings = try self.getVisibleBindings(allocator);
            errdefer allocator.free(bindings);

            var writable: std.ArrayList(DebugInfo.VisibleBinding) = .{};
            defer writable.deinit(allocator);

            for (bindings) |binding| {
                if (self.bindingHasWritableRuntimeHome(binding)) {
                    try writable.append(allocator, binding);
                }
            }

            allocator.free(bindings);
            return try writable.toOwnedSlice(allocator);
        }

        pub fn findVisibleBindingByName(
            self: *const Self,
            allocator: std.mem.Allocator,
            name: []const u8,
        ) !?DebugInfo.VisibleBinding {
            const bindings = try self.getVisibleBindings(allocator);
            defer allocator.free(bindings);

            for (bindings) |binding| {
                if (std.mem.eql(u8, binding.name, name)) return binding;
            }
            return null;
        }

        pub fn getVisibleStorageRootValue(
            self: *const Self,
            binding: DebugInfo.VisibleBinding,
        ) !?u256 {
            const slot = self.getWritableRootSlot(binding, "storage_field", "storage_root") orelse return null;
            const frame = self.evm.getCurrentFrame() orelse return null;
            return try self.evm.storage.get(frame.address, slot);
        }

        pub fn setVisibleStorageRootValue(
            self: *Self,
            binding: DebugInfo.VisibleBinding,
            value: u256,
        ) !bool {
            const slot = self.getWritableRootSlot(binding, "storage_field", "storage_root") orelse return false;
            const frame = self.evm.getCurrentFrame() orelse return false;
            try self.evm.storage.set(frame.address, slot, value);
            return true;
        }

        pub fn getVisibleMemoryRootValue(
            self: *const Self,
            binding: DebugInfo.VisibleBinding,
        ) ?u256 {
            const slot = self.getWritableRootSlot(binding, "memory_field", "memory_root") orelse return null;
            const frame = self.evm.getCurrentFrame() orelse return null;
            const offset = self.memoryRootWordOffset(frame, slot) orelse return null;

            var result: u256 = 0;
            var idx: u32 = 0;
            while (idx < 32) : (idx += 1) {
                const addr = std.math.add(u32, offset, idx) catch return null;
                result = (result << 8) | frame.readMemory(addr);
            }
            return result;
        }

        pub fn setVisibleMemoryRootValue(
            self: *Self,
            binding: DebugInfo.VisibleBinding,
            value: u256,
        ) !bool {
            const slot = self.getWritableRootSlot(binding, "memory_field", "memory_root") orelse return false;
            const frame = self.evm.getCurrentFrame() orelse return false;
            const offset = self.memoryRootWordOffset(frame, slot) orelse return false;

            var idx: u32 = 0;
            while (idx < 32) : (idx += 1) {
                const byte = @as(u8, @truncate(value >> @intCast((31 - idx) * 8)));
                const addr = std.math.add(u32, offset, idx) catch return false;
                try frame.writeMemory(addr, byte);
            }
            return true;
        }

        pub fn getVisibleTStoreRootValue(
            self: *const Self,
            binding: DebugInfo.VisibleBinding,
        ) ?u256 {
            const slot = self.getWritableRootSlot(binding, "tstore_field", "tstore_root") orelse return null;
            const frame = self.evm.getCurrentFrame() orelse return null;
            return self.evm.storage.get_transient(frame.address, slot);
        }

        pub fn setVisibleTStoreRootValue(
            self: *Self,
            binding: DebugInfo.VisibleBinding,
            value: u256,
        ) !bool {
            const slot = self.getWritableRootSlot(binding, "tstore_field", "tstore_root") orelse return false;
            const frame = self.evm.getCurrentFrame() orelse return false;
            try self.evm.storage.set_transient(frame.address, slot, value);
            return true;
        }

        pub fn getVisibleBindingValueByName(
            self: *const Self,
            allocator: std.mem.Allocator,
            name: []const u8,
        ) !?u256 {
            const binding = try self.findVisibleBindingByName(allocator, name) orelse return null;
            if (std.mem.eql(u8, binding.runtime_kind, "storage_field")) {
                return try self.getVisibleStorageRootValue(binding);
            }
            if (std.mem.eql(u8, binding.runtime_kind, "memory_field")) {
                return self.getVisibleMemoryRootValue(binding);
            }
            if (std.mem.eql(u8, binding.runtime_kind, "tstore_field")) {
                return self.getVisibleTStoreRootValue(binding);
            }
            return null;
        }

        pub fn setVisibleBindingValueByName(
            self: *Self,
            allocator: std.mem.Allocator,
            name: []const u8,
            value: u256,
        ) !bool {
            const binding = try self.findVisibleBindingByName(allocator, name) orelse return false;
            if (std.mem.eql(u8, binding.runtime_kind, "storage_field")) {
                return try self.setVisibleStorageRootValue(binding, value);
            }
            if (std.mem.eql(u8, binding.runtime_kind, "memory_field")) {
                return try self.setVisibleMemoryRootValue(binding, value);
            }
            if (std.mem.eql(u8, binding.runtime_kind, "tstore_field")) {
                return try self.setVisibleTStoreRootValue(binding, value);
            }
            return false;
        }

        pub fn getVisibleScopes(self: *const Self, allocator: std.mem.Allocator) ![]*const DebugInfo.SourceScope {
            const info = self.debug_info orelse return &.{};
            const idx = self.currentOpIndex() orelse return &.{};
            return try info.collectVisibleScopes(allocator, idx);
        }

        /// Get current source line number (or null if in compiler-internal code).
        pub fn currentSourceLine(self: *const Self) ?u32 {
            const entry = self.currentEntry() orelse return null;
            return entry.line;
        }

        /// Get the text of a source line (1-based).
        pub fn getSourceLineText(self: *const Self, line: u32) ?[]const u8 {
            if (line == 0 or line > self.source_lines.len) return null;
            const sl = self.source_lines[line - 1];
            return self.source_text[sl.start..sl.end];
        }

        /// Get current PC.
        pub fn getPC(self: *const Self) u32 {
            return self.evm.getPC();
        }

        /// Get current opcode name.
        pub fn getCurrentOpcodeName(self: *const Self) []const u8 {
            const frame = self.evm.getCurrentFrame() orelse return "???";
            const opcode = frame.bytecode.getOpcode(frame.pc) orelse return "???";
            return opcode_utils.getOpName(opcode);
        }

        /// Get current call depth.
        pub fn getCallDepth(self: *const Self) usize {
            return self.evm.frames.items.len;
        }

        /// Get EVM stack contents.
        pub fn getStack(self: *const Self) []const u256 {
            const frame = self.evm.getCurrentFrame() orelse return &[_]u256{};
            return frame.stack.items;
        }

        /// Get gas remaining.
        pub fn getGasRemaining(self: *const Self) i64 {
            const frame = self.evm.getCurrentFrame() orelse return 0;
            return frame.gas_remaining;
        }

        /// Check if execution has finished (stopped, reverted, or no frames).
        pub fn isHalted(self: *const Self) bool {
            return self.state == .halted;
        }

        /// Check if execution completed successfully (not reverted).
        pub fn isSuccess(self: *const Self) bool {
            if (self.evm.getCurrentFrame()) |frame| {
                return frame.stopped and !frame.reverted;
            }
            // No frames = execution completed
            return self.evm.frames.items.len == 0;
        }

        /// Get total number of source lines.
        pub fn totalSourceLines(self: *const Self) u32 {
            return @intCast(self.source_lines.len);
        }

        // ====================================================================
        // Internal
        // ====================================================================

        fn executeOneOpcode(self: *Self) !void {
            const frame = self.evm.getCurrentFrame() orelse {
                self.state = .halted;
                self.stop_reason = .execution_finished;
                return;
            };

            if (frame.stopped) {
                self.state = .halted;
                self.stop_reason = if (frame.reverted) .execution_reverted else .execution_finished;
                return;
            }

            self.evm.step() catch |err| {
                self.state = .halted;
                self.stop_reason = .execution_error;
                return err;
            };

            self.steps_executed += 1;

            // Check if frame finished after step
            if (self.evm.getCurrentFrame()) |f| {
                if (f.stopped) {
                    self.state = .halted;
                    self.stop_reason = if (f.reverted) .execution_reverted else .execution_finished;
                }
            } else {
                self.state = .halted;
                self.stop_reason = .execution_finished;
            }
        }

        fn isAtStatementBoundary(self: *const Self) bool {
            const frame = self.evm.getCurrentFrame() orelse return false;
            const entry = self.src_map.getEntry(frame.pc) orelse return false;
            return entry.is_statement;
        }

        fn currentStatementKey(self: *const Self) ?StatementKey {
            const frame = self.evm.getCurrentFrame() orelse return null;
            const entry = self.src_map.getEntry(frame.pc) orelse return null;
            if (!entry.is_statement) return null;
            return .{
                .depth = self.getCallDepth(),
                .stmt_pc = entry.pc,
            };
        }

        fn currentStatementKeyChanged(self: *const Self, start: ?StatementKey) bool {
            const current = self.currentStatementKey() orelse return false;
            if (start == null) return true;
            return current.depth != start.?.depth or current.stmt_pc != start.?.stmt_pc;
        }

        fn bindingHasWritableRuntimeHome(self: *const Self, binding: DebugInfo.VisibleBinding) bool {
            _ = self;
            if (!binding.editable) return false;
            const location_kind = binding.runtime_location_kind orelse return false;
            if (binding.runtime_location_slot == null) return false;
            return std.mem.eql(u8, location_kind, "storage_root") or
                std.mem.eql(u8, location_kind, "memory_root") or
                std.mem.eql(u8, location_kind, "tstore_root");
        }

        fn getWritableRootSlot(
            self: *const Self,
            binding: DebugInfo.VisibleBinding,
            expected_runtime_kind: []const u8,
            expected_location_kind: []const u8,
        ) ?u64 {
            _ = self;
            if (!binding.editable) return null;
            if (!std.mem.eql(u8, binding.runtime_kind, expected_runtime_kind)) return null;
            const location_kind = binding.runtime_location_kind orelse return null;
            if (!std.mem.eql(u8, location_kind, expected_location_kind)) return null;
            return binding.runtime_location_slot;
        }

        fn memoryRootWordOffset(self: *const Self, frame: anytype, slot: u64) ?u32 {
            _ = self;
            const code_size = std.math.cast(u32, frame.bytecode.len()) orelse return null;
            const slot_bytes = std.math.cast(u32, slot * 32) orelse return null;
            return std.math.add(u32, code_size, slot_bytes) catch null;
        }

        fn buildLineIndex(allocator: std.mem.Allocator, text: []const u8) ![]LineSlice {
            var lines: std.ArrayList(LineSlice) = .{};
            var start: u32 = 0;

            for (text, 0..) |c, i| {
                if (c == '\n') {
                    try lines.append(allocator, .{ .start = start, .end = @intCast(i) });
                    start = @intCast(i + 1);
                }
            }
            // Last line (no trailing newline)
            if (start <= text.len) {
                try lines.append(allocator, .{ .start = start, .end = @intCast(text.len) });
            }

            return try lines.toOwnedSlice(allocator);
        }
    };
}
