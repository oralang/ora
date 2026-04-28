//! Shared debugger-session helpers consumed by both `debug_probe` and
//! `debug_tui`.
//!
//! Both binaries need the same artifact-loading, deployment-frame, and
//! source-map-rebase logic. This module is the single home for that code so
//! the two consumers can't drift.
const std = @import("std");
const primitives = @import("voltaire");
const evm_mod = @import("evm.zig");
const evm_config = @import("evm_config.zig");
const frame_mod = @import("frame.zig");
const source_map = @import("source_map.zig");
const SourceMap = source_map.SourceMap;

/// Maximum bytes any debugger artifact (.hex, .sourcemap.json, .debug.json,
/// .abi.json, .sir) is permitted to load. 16 MiB.
pub const kArtifactMaxBytes: usize = 16 * 1024 * 1024;

/// Default gas budget for both deployment and runtime frames in the debugger.
pub const kDefaultGasLimit: i64 = 5_000_000;

/// Step cap on the deployment phase. The deployer is expected to return a
/// runtime body well under this; a contract that doesn't halt is treated as a
/// bug rather than something to grind through.
pub const kDeploymentStepCap: usize = 200_000;

/// Read a debugger artifact from disk, capped at `kArtifactMaxBytes`.
pub fn loadArtifact(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fs.cwd().readFileAlloc(allocator, path, kArtifactMaxBytes);
}

/// Per-EVM-config helpers. The two binaries do `DebugSession(.{})` once at
/// the top of the file and then call the inner functions.
pub fn DebugSession(comptime config: evm_config.EvmConfig) type {
    const Evm = evm_mod.Evm(config);
    const Frame = frame_mod.Frame(config);

    return struct {
        pub const DeployOptions = struct {
            caller: primitives.Address,
            contract: primitives.Address,
            deployment_bytecode: []const u8,
            init_calldata: []const u8 = &.{},
            gas_limit: i64 = kDefaultGasLimit,
            step_cap: usize = kDeploymentStepCap,
            /// Strict mode raises an explicit error when the deployer fails to
            /// produce runtime bytes (revert, no output, or step-cap reached).
            /// Non-strict mode returns the deployment bytecode as a fallback —
            /// useful for the headless probe, which still wants to step
            /// *something* in that case.
            strict: bool = false,
        };

        /// Run the deployment frame to completion and return the resulting
        /// runtime bytecode. Cleans up `evm.frames` before returning.
        pub fn deployRuntimeBytecode(
            allocator: std.mem.Allocator,
            evm: *Evm,
            opts: DeployOptions,
        ) ![]u8 {
            try evm.frames.append(evm.arena.allocator(), try Frame.init(
                evm.arena.allocator(),
                opts.deployment_bytecode,
                opts.gas_limit,
                opts.caller,
                opts.contract,
                0,
                opts.init_calldata,
                @as(*anyopaque, @ptrCast(evm)),
                evm.hardfork,
                false,
            ));

            var steps: usize = 0;
            var step_cap_hit = false;
            while (evm.getCurrentFrame()) |frame| {
                if (frame.stopped or frame.reverted) break;
                if (steps >= opts.step_cap) {
                    step_cap_hit = true;
                    break;
                }
                try evm.step();
                steps += 1;
            }

            if (step_cap_hit and opts.strict) {
                return error.DeploymentDidNotHalt;
            }

            const frame = evm.getCurrentFrame() orelse {
                if (opts.strict) return error.DeploymentReturnedNoFrame;
                return try allocator.dupe(u8, opts.deployment_bytecode);
            };

            // Both consumers cleared frames on the way out; preserve that.
            defer {
                for (evm.frames.items) |*live_frame| {
                    live_frame.deinit();
                }
                evm.frames.clearRetainingCapacity();
            }

            if (frame.reverted or frame.output.len == 0) {
                if (opts.strict) return error.DeploymentRevertedWithNoRuntime;
                return try allocator.dupe(u8, opts.deployment_bytecode);
            }
            return try allocator.dupe(u8, frame.output);
        }

        /// Rebase a creation-time source map onto the runtime bytecode by
        /// subtracting `runtime_start_pc` from each entry's pc and dropping
        /// entries that fall outside the runtime body. Preserves every
        /// entry field so downstream consumers (the debugger UI, in
        /// particular) keep their statement-id / synthetic / hoist metadata.
        pub fn rebaseSourceMapForRuntime(
            allocator: std.mem.Allocator,
            creation_source_map: *const SourceMap,
            runtime_bytecode: []const u8,
        ) !SourceMap {
            const runtime_start_pc = creation_source_map.runtime_start_pc orelse {
                return try SourceMap.fromEntries(allocator, creation_source_map.entries);
            };
            if (runtime_bytecode.len == 0) {
                return try SourceMap.fromEntries(allocator, creation_source_map.entries);
            }

            var rebased: std.ArrayList(SourceMap.Entry) = .{};
            defer rebased.deinit(allocator);

            for (creation_source_map.entries) |entry| {
                if (entry.pc < runtime_start_pc) continue;
                const rebased_pc = entry.pc - runtime_start_pc;
                if (rebased_pc >= runtime_bytecode.len) continue;
                try rebased.append(allocator, .{
                    .idx = entry.idx,
                    .pc = @intCast(rebased_pc),
                    .file = entry.file,
                    .line = entry.line,
                    .col = entry.col,
                    .statement_id = entry.statement_id,
                    .origin_statement_id = entry.origin_statement_id,
                    .execution_region_id = entry.execution_region_id,
                    .statement_run_index = entry.statement_run_index,
                    .function = entry.function,
                    .sir_line = entry.sir_line,
                    .is_synthetic = entry.is_synthetic,
                    .synthetic_index = entry.synthetic_index,
                    .synthetic_count = entry.synthetic_count,
                    .synthetic_path = entry.synthetic_path,
                    .is_hoisted = entry.is_hoisted,
                    .is_duplicated = entry.is_duplicated,
                    .is_statement = entry.is_statement,
                    .kind = entry.kind,
                });
            }

            if (rebased.items.len == 0) {
                return try SourceMap.fromEntries(allocator, creation_source_map.entries);
            }
            var runtime_map = try SourceMap.fromEntries(allocator, rebased.items);
            runtime_map.runtime_start_pc = 0;
            return runtime_map;
        }
    };
}
