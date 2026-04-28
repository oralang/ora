/// Source-level debugger for the Ora EVM.
/// Wraps the EVM with breakpoints, statement-level stepping, and state inspection.
/// Does NOT modify evm.zig or frame.zig — it drives them externally via step().
const std = @import("std");
const primitives = @import("voltaire");
const source_map = @import("source_map.zig");
const SourceMap = source_map.SourceMap;
const debug_info_mod = @import("debug_info.zig");
const DebugInfo = debug_info_mod.DebugInfo;
const opcode_utils = @import("opcode.zig");
const evm_mod = @import("evm.zig");
const evm_config = @import("evm_config.zig");
const debug_session = @import("debug_session.zig");

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
        /// Active storage watchpoints. Checked after every opcode; if any
        /// slot has changed since the last check, execution pauses with
        /// stop_reason == .watchpoint_hit and `last_watchpoint_id` is
        /// set to the triggering id.
        watchpoints: std.ArrayList(Watchpoint),
        next_watchpoint_id: u32,
        last_watchpoint_id: ?u32,
        /// Set of source-map entry idx values whose statement boundary the
        /// debugger should ignore: their op_meta is "invalid" / "sir.invalid"
        /// but another entry in the same statement_id surfaces a real op.
        /// Built once at the time debug_info is bound; queried in O(1) on
        /// every step. Empty when debug_info isn't set.
        ignored_invalid_idx: std.AutoHashMap(u32, void),
        /// Per-source-line statement-boundary hit counts. Bumped on every
        /// distinct statement-line transition; never reset across the
        /// session. Exposed via `getLineHits` / `getLineHitsTopN` for the
        /// `:cov` command. Lines that don't host any statement boundary
        /// (whitespace, comments) never appear here.
        line_hits: std.AutoHashMap(u32, u32),
        /// Per-source-line cumulative gas spent. Each opcode's gas cost
        /// is attributed to the source line of the most recent statement
        /// boundary (`last_statement_line`). Surfaces via `getLineGas`
        /// and the `:gascov` command. Negative deltas (refunds) are
        /// clamped to 0 to avoid wraparound.
        line_gas: std.AutoHashMap(u32, u64),
        state: State,
        stop_reason: StopReason,
        allocator: std.mem.Allocator,
        /// Total opcodes executed since last user command
        steps_executed: u64,
        /// Safety limit to prevent infinite loops
        max_steps: u64,
        last_statement_id: ?u32,
        last_statement_line: ?u32,
        last_statement_sir_line: ?u32,
        last_error_name: ?[]const u8,

        pub const State = enum {
            paused,
            running,
            halted,
        };

        pub const StopReason = enum {
            not_started,
            breakpoint_hit,
            watchpoint_hit,
            step_complete,
            execution_finished,
            execution_reverted,
            execution_error,
            step_limit_reached,
        };

        /// A storage watchpoint pauses execution when the value at
        /// `(address, slot)` changes since the watchpoint was last
        /// observed. `last_seen` is the value at the moment the
        /// watchpoint was added (or last serviced); `name` is the
        /// human-readable label rendered in `:info watch` and the
        /// watchpoint-hit status line.
        pub const Watchpoint = struct {
            id: u32,
            slot: u256,
            address: primitives.Address,
            last_seen: u256,
            name: []const u8,
        };

        pub const LineSlice = struct {
            start: u32,
            end: u32,
        };

        const StatementKey = struct {
            depth: usize,
            stmt_id: ?u32,
            execution_region_id: ?u32,
            statement_run_index: ?u32,
            stmt_pc: u32,
        };

        pub fn init(
            allocator: std.mem.Allocator,
            evm: *EvmType,
            src_map: SourceMap,
            source_text: []const u8,
        ) !Self {
            const lines = try buildLineIndex(allocator, source_text);
            var self = Self{
                .evm = evm,
                .src_map = src_map,
                .debug_info = null,
                .source_text = source_text,
                .source_lines = lines,
                .breakpoints = std.AutoHashMap(u32, void).init(allocator),
                .watchpoints = .{},
                .next_watchpoint_id = 1,
                .last_watchpoint_id = null,
                .ignored_invalid_idx = std.AutoHashMap(u32, void).init(allocator),
                .line_hits = std.AutoHashMap(u32, u32).init(allocator),
                .line_gas = std.AutoHashMap(u32, u64).init(allocator),
                .state = .paused,
                .stop_reason = .not_started,
                .allocator = allocator,
                .steps_executed = 0,
                .max_steps = debug_session.kDefaultMaxSteps,
                .last_statement_id = null,
                .last_statement_line = null,
                .last_statement_sir_line = null,
                .last_error_name = null,
            };
            self.updateLastStatementLine();
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.breakpoints.deinit();
            self.ignored_invalid_idx.deinit();
            self.line_hits.deinit();
            self.line_gas.deinit();
            for (self.watchpoints.items) |wp| self.allocator.free(wp.name);
            self.watchpoints.deinit(self.allocator);
            if (self.debug_info) |*debug_info| debug_info.deinit();
            self.src_map.deinit();
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
            try self.buildIgnoredInvalidIdxSet();
            return self;
        }

        /// Walks the source map once and records every entry whose op_meta
        /// is "invalid" / "sir.invalid" *and* shares its statement_id with
        /// at least one entry whose op is a real opcode. Step-time we then
        /// just check `ignored_invalid_idx.contains(idx)` — O(1) per step
        /// instead of the previous O(N) scan whose inner getOpMetaForIdx
        /// itself was O(N) (so O(N²) overall).
        fn buildIgnoredInvalidIdxSet(self: *Self) !void {
            const debug_info = self.debug_info orelse return;
            // Bucket entries by statement_id, splitting into invalid-op idx
            // and "has at least one real op" flag.
            const ScratchEntry = struct {
                invalid_idxs: std.ArrayList(u32) = .{},
                has_real_op: bool = false,
            };
            var by_stmt = std.AutoHashMap(u32, ScratchEntry).init(self.allocator);
            defer {
                var it = by_stmt.valueIterator();
                while (it.next()) |bucket| bucket.invalid_idxs.deinit(self.allocator);
                by_stmt.deinit();
            }

            for (self.src_map.entries) |entry| {
                const stmt_id = entry.statement_id orelse continue;
                const idx = entry.idx orelse continue;
                const op_meta = debug_info.getOpMetaForIdx(idx) orelse continue;
                const is_invalid = std.mem.eql(u8, op_meta.op, "invalid") or
                    std.mem.eql(u8, op_meta.op, "sir.invalid");

                const gop = try by_stmt.getOrPut(stmt_id);
                if (!gop.found_existing) gop.value_ptr.* = .{};
                const bucket = gop.value_ptr;
                if (is_invalid) {
                    try bucket.invalid_idxs.append(self.allocator, idx);
                } else {
                    bucket.has_real_op = true;
                }
            }

            var it = by_stmt.valueIterator();
            while (it.next()) |bucket| {
                if (!bucket.has_real_op) continue;
                for (bucket.invalid_idxs.items) |idx| {
                    try self.ignored_invalid_idx.put(idx, {});
                }
            }
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
        // Watchpoints
        // ====================================================================

        /// Watch a raw storage slot at the current frame's address.
        /// Returns the watchpoint id so callers can remove it later.
        pub fn addWatchpointBySlot(self: *Self, slot: u256) !u32 {
            const frame = self.evm.getCurrentFrame() orelse return error.NoActiveFrame;
            const current_value = try self.evm.storage.get(frame.address, slot);
            const id = self.next_watchpoint_id;
            self.next_watchpoint_id += 1;

            const buf = try std.fmt.allocPrint(self.allocator, "slot 0x{x}", .{slot});
            errdefer self.allocator.free(buf);
            try self.watchpoints.append(self.allocator, .{
                .id = id,
                .slot = slot,
                .address = frame.address,
                .last_seen = current_value,
                .name = buf,
            });
            return id;
        }

        /// Watch a source-level binding by name. Resolves to its
        /// runtime storage slot via the active scope's debug info.
        /// Returns null if the binding isn't visible at the current op
        /// or isn't backed by a writable storage slot.
        pub fn addWatchpointByBindingName(self: *Self, name: []const u8) !?u32 {
            const binding = try self.findVisibleBindingByName(self.allocator, name) orelse return null;
            if (!std.mem.eql(u8, binding.runtime_kind, "storage_field")) return null;
            const location_kind = binding.runtime_location_kind orelse return null;
            if (!std.mem.eql(u8, location_kind, "storage_root")) return null;
            const slot64 = binding.runtime_location_slot orelse return null;
            const slot: u256 = slot64;

            const frame = self.evm.getCurrentFrame() orelse return error.NoActiveFrame;
            const current_value = try self.evm.storage.get(frame.address, slot);
            const id = self.next_watchpoint_id;
            self.next_watchpoint_id += 1;

            const buf = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(buf);
            try self.watchpoints.append(self.allocator, .{
                .id = id,
                .slot = slot,
                .address = frame.address,
                .last_seen = current_value,
                .name = buf,
            });
            return id;
        }

        /// Remove a watchpoint by id. Returns true if a watchpoint with
        /// that id existed.
        pub fn removeWatchpoint(self: *Self, id: u32) bool {
            for (self.watchpoints.items, 0..) |wp, i| {
                if (wp.id != id) continue;
                self.allocator.free(wp.name);
                _ = self.watchpoints.orderedRemove(i);
                return true;
            }
            return false;
        }

        pub fn getWatchpoints(self: *const Self) []const Watchpoint {
            return self.watchpoints.items;
        }

        pub fn lastWatchpointId(self: *const Self) ?u32 {
            return self.last_watchpoint_id;
        }

        /// Hit count for a single source line; `null` if the line never
        /// had a statement boundary visited at runtime.
        pub fn getLineHits(self: *const Self, line: u32) ?u32 {
            return self.line_hits.get(line);
        }

        pub const LineHit = struct {
            line: u32,
            count: u32,
        };

        /// Returns the `n` hottest lines by hit count, sorted descending.
        /// Caller owns the returned slice.
        pub fn getLineHitsTopN(self: *const Self, allocator: std.mem.Allocator, n: usize) ![]LineHit {
            var entries: std.ArrayList(LineHit) = .{};
            errdefer entries.deinit(allocator);
            try entries.ensureTotalCapacity(allocator, self.line_hits.count());
            var it = self.line_hits.iterator();
            while (it.next()) |kv| {
                try entries.append(allocator, .{ .line = kv.key_ptr.*, .count = kv.value_ptr.* });
            }
            const items = entries.items;
            std.sort.heap(LineHit, items, {}, struct {
                fn lessThan(_: void, a: LineHit, b: LineHit) bool {
                    if (a.count != b.count) return a.count > b.count;
                    return a.line < b.line;
                }
            }.lessThan);
            if (items.len > n) {
                const trimmed = try allocator.dupe(LineHit, items[0..n]);
                entries.deinit(allocator);
                return trimmed;
            }
            return entries.toOwnedSlice(allocator);
        }

        /// Total distinct lines that have been hit at least once.
        pub fn lineHitsCount(self: *const Self) usize {
            return self.line_hits.count();
        }

        /// Cumulative gas spent on a single source line; `null` if no
        /// gas-spending opcode has been attributed to this line yet.
        pub fn getLineGas(self: *const Self, line: u32) ?u64 {
            return self.line_gas.get(line);
        }

        pub const LineGas = struct {
            line: u32,
            gas: u64,
        };

        /// Returns the `n` lines with the most cumulative gas spent,
        /// sorted descending. Caller owns the returned slice.
        pub fn getLineGasTopN(self: *const Self, allocator: std.mem.Allocator, n: usize) ![]LineGas {
            var entries: std.ArrayList(LineGas) = .{};
            errdefer entries.deinit(allocator);
            try entries.ensureTotalCapacity(allocator, self.line_gas.count());
            var it = self.line_gas.iterator();
            while (it.next()) |kv| {
                try entries.append(allocator, .{ .line = kv.key_ptr.*, .gas = kv.value_ptr.* });
            }
            const items = entries.items;
            std.sort.heap(LineGas, items, {}, struct {
                fn lessThan(_: void, a: LineGas, b: LineGas) bool {
                    if (a.gas != b.gas) return a.gas > b.gas;
                    return a.line < b.line;
                }
            }.lessThan);
            if (items.len > n) {
                const trimmed = try allocator.dupe(LineGas, items[0..n]);
                entries.deinit(allocator);
                return trimmed;
            }
            return entries.toOwnedSlice(allocator);
        }

        /// Total distinct lines that have been attributed any gas cost.
        pub fn lineGasCount(self: *const Self) usize {
            return self.line_gas.count();
        }

        /// True iff the last executeOneOpcode pausing was a watchpoint
        /// trigger. The stepping loops use this to short-circuit out of
        /// their drive loop and surface the pause to the caller.
        fn pausedOnWatchpoint(self: *const Self) bool {
            return self.state == .paused and self.stop_reason == .watchpoint_hit;
        }

        /// Sample every watchpoint and return the id of the first one
        /// whose value has changed since the last sample. Updates
        /// `last_seen` on every watchpoint regardless. Cost is one
        /// `storage.get` per watchpoint per call.
        fn pollWatchpoints(self: *Self) !?u32 {
            var triggered: ?u32 = null;
            for (self.watchpoints.items) |*wp| {
                const current_value = try self.evm.storage.get(wp.address, wp.slot);
                if (current_value != wp.last_seen) {
                    if (triggered == null) triggered = wp.id;
                    wp.last_seen = current_value;
                }
            }
            return triggered;
        }

        // ====================================================================
        // Stepping commands
        // ====================================================================

        /// Step In: execute until the source line changes (at any call depth).
        /// Enters function calls.
        pub fn stepIn(self: *Self) !void {
            if (self.isHalted()) return;
            // Resume from any prior pause (e.g. a watchpoint hit). The
            // post-step pausedOnWatchpoint() check catches NEW watchpoint
            // triggers during this step.
            self.state = .running;

            const start_statement = self.currentStatementKey();
            self.steps_executed = 0;

            while (true) {
                try self.executeOneOpcode();
                if (self.isHalted() or self.pausedOnWatchpoint()) return;
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

        /// Step Opcode: execute exactly one opcode and pause immediately.
        /// This is useful for inspecting the transient EVM operand stack between
        /// source-level statement boundaries.
        pub fn stepOpcode(self: *Self) !void {
            if (self.isHalted()) return;
            self.state = .running;

            self.steps_executed = 0;
            try self.executeOneOpcode();
            if (self.isHalted() or self.pausedOnWatchpoint()) return;
            self.stop_reason = .step_complete;
            self.state = .paused;
        }

        /// Step Over: execute until the source line changes at the same or lower call depth.
        /// Skips over function calls.
        pub fn stepOver(self: *Self) !void {
            if (self.isHalted()) return;
            self.state = .running;

            const start_statement = self.currentStatementKey();
            const start_depth = self.getCallDepth();
            self.steps_executed = 0;

            while (true) {
                try self.executeOneOpcode();
                if (self.isHalted() or self.pausedOnWatchpoint()) return;
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
            self.state = .running;

            const start_depth = self.getCallDepth();
            if (start_depth == 0) {
                // Already at top level — run to completion
                return self.continue_();
            }

            self.steps_executed = 0;

            while (true) {
                try self.executeOneOpcode();
                if (self.isHalted() or self.pausedOnWatchpoint()) return;
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
            if (self.isHalted() or self.pausedOnWatchpoint()) return;

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
                if (self.isHalted() or self.pausedOnWatchpoint()) return;
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

        pub fn currentOpMeta(self: *const Self) ?*const DebugInfo.OpMeta {
            const info = self.debug_info orelse return null;
            const idx = self.currentOpIndex() orelse return null;
            return info.getOpMetaForIdx(idx);
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

        pub fn lastStatementLine(self: *const Self) ?u32 {
            return self.last_statement_line;
        }

        pub fn lastStatementId(self: *const Self) ?u32 {
            return self.last_statement_id;
        }

        pub fn lastStatementSirLine(self: *const Self) ?u32 {
            return self.last_statement_sir_line;
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
                self.last_error_name = null;
                return;
            }

            // Snapshot the gas pre-step so we can attribute the cost to
            // whichever source line is currently active. Use the line we
            // most recently transitioned to (`last_statement_line`) so
            // multi-opcode statements accumulate their full cost.
            const gas_before = frame.gas_remaining;
            const attributed_line = self.last_statement_line;

            self.evm.step() catch |err| {
                self.state = .halted;
                self.stop_reason = .execution_error;
                self.last_error_name = @errorName(err);
                return err;
            };

            // Best-effort gas accounting. The post-step frame may differ
            // from the pre-step frame (CALL pushed a new one); in that
            // case the parent's snapshotted gas is no longer comparable,
            // so we skip the accounting for that step rather than
            // mis-attribute it.
            if (self.evm.getCurrentFrame()) |post_frame| {
                if (post_frame == frame) {
                    if (gas_before > post_frame.gas_remaining) {
                        const delta: u64 = @intCast(gas_before - post_frame.gas_remaining);
                        if (attributed_line) |line| {
                            const gop = self.line_gas.getOrPut(line) catch null;
                            if (gop) |entry| {
                                if (!entry.found_existing) entry.value_ptr.* = 0;
                                entry.value_ptr.* +%= delta;
                            }
                        }
                    }
                }
            }

            self.steps_executed += 1;
            self.updateLastStatementLine();

            // Watchpoints fire on the first step after any of their
            // slots changes. Sampling on every step (rather than only
            // on SSTORE) keeps the check uniform with cross-frame
            // calls — a callee's SSTORE on a delegate-shared address
            // surfaces here too.
            if (self.watchpoints.items.len != 0) {
                if (self.pollWatchpoints() catch null) |wp_id| {
                    self.last_watchpoint_id = wp_id;
                    self.state = .paused;
                    self.stop_reason = .watchpoint_hit;
                    return;
                }
            }

            // Check if frame finished after step
            if (self.evm.getCurrentFrame()) |f| {
                if (f.stopped) {
                    self.state = .halted;
                    self.stop_reason = if (f.reverted) .execution_reverted else .execution_finished;
                    self.last_error_name = null;
                }
            } else {
                self.state = .halted;
                self.stop_reason = .execution_finished;
                self.last_error_name = null;
            }
        }

        pub fn lastErrorName(self: *const Self) ?[]const u8 {
            return self.last_error_name;
        }

        fn isAtStatementBoundary(self: *const Self) bool {
            const frame = self.evm.getCurrentFrame() orelse return false;
            const entry = self.src_map.getEntry(frame.pc) orelse return false;
            return entry.is_statement;
        }

        fn updateLastStatementLine(self: *Self) void {
            const entry = self.currentEntry() orelse return;
            if (!entry.is_statement) return;
            // Bump the coverage counter on every distinct statement-line
            // transition. Re-entering the same statement (loop iteration,
            // recursion) counts as a fresh hit. OOM is non-fatal — the
            // counter just stops updating.
            const prev_line = self.last_statement_line;
            if (prev_line == null or prev_line.? != entry.line or self.last_statement_id != entry.statement_id) {
                const gop = self.line_hits.getOrPut(entry.line) catch return;
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* +%= 1;
            }
            self.last_statement_id = entry.statement_id;
            self.last_statement_line = entry.line;
            self.last_statement_sir_line = entry.sir_line;
        }

        fn currentStatementKey(self: *const Self) ?StatementKey {
            const frame = self.evm.getCurrentFrame() orelse return null;
            const entry = self.src_map.getEntry(frame.pc) orelse return null;
            if (!entry.is_statement) return null;
            if (self.shouldIgnoreStatementBoundary(entry)) return null;
            return .{
                .depth = self.getCallDepth(),
                .stmt_id = entry.statement_id,
                .execution_region_id = entry.execution_region_id,
                .statement_run_index = entry.statement_run_index,
                .stmt_pc = entry.pc,
            };
        }

        fn currentStatementKeyChanged(self: *const Self, start: ?StatementKey) bool {
            const current = self.currentStatementKey() orelse return false;
            if (start == null) return true;
            if (current.depth != start.?.depth) return true;
            if (current.execution_region_id != null and start.?.execution_region_id != null) {
                if (current.execution_region_id.? != start.?.execution_region_id.?) return true;
                if (current.statement_run_index != start.?.statement_run_index) return true;
                return false;
            }
            if (current.stmt_id != null and start.?.stmt_id != null) {
                return current.stmt_id.? != start.?.stmt_id.?;
            }
            return current.stmt_pc != start.?.stmt_pc;
        }

        fn shouldIgnoreStatementBoundary(self: *const Self, entry: *const SourceMap.Entry) bool {
            // Population is gated on having debug_info; without it the set is
            // empty and we fall through immediately.
            const idx = entry.idx orelse return false;
            return self.ignored_invalid_idx.contains(idx);
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
