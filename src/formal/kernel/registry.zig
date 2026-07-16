//! Compiler-kernel formal gate registry.
//!
//! Kernel gates are listed here and exposed through phase-specific sessions.
//! The underlying gates retain ownership of proof checking and certificate
//! serialization; the registry only coordinates when each phase may run.

const std = @import("std");
const mlir = @import("mlir_c_api").c;
const dispatcher_gate = @import("../dispatcher_table_gate.zig");

pub const GateId = enum {
    dispatcher,
};

pub const GatePhase = enum {
    prepare_from_sir,
    bind_backend_output,
    finish_certificate,
};

pub const GateDefinition = struct {
    id: GateId,
    name: []const u8,
    blocking: bool,
    phases: []const GatePhase,
};

const dispatcher_phases = [_]GatePhase{
    .prepare_from_sir,
    .bind_backend_output,
    .finish_certificate,
};

pub const gates = [_]GateDefinition{
    .{
        .id = .dispatcher,
        .name = "dispatcher_userland",
        .blocking = true,
        .phases = &dispatcher_phases,
    },
};

pub fn definition(id: GateId) *const GateDefinition {
    return switch (id) {
        .dispatcher => &gates[0],
    };
}

pub const DispatcherSession = struct {
    check: dispatcher_gate.CheckResult,
    lifecycle: Lifecycle = .{},

    pub const CertificateKind = enum {
        sir,
        bytecode,
    };

    pub fn prepareFromSir(
        allocator: std.mem.Allocator,
        ctx: mlir.MlirContext,
        module: mlir.MlirModule,
        intent_json: []const u8,
        sir_text: []const u8,
        file_path: []const u8,
        process_environ: std.process.Environ,
        stdout: anytype,
    ) !DispatcherSession {
        return .{
            .check = try dispatcher_gate.checkCurrentModule(
                allocator,
                ctx,
                module,
                intent_json,
                sir_text,
                file_path,
                process_environ,
                stdout,
            ),
        };
    }

    pub fn deinit(self: *DispatcherSession) void {
        self.check.deinit();
        self.* = undefined;
    }

    pub fn bindBackendOutput(
        self: *DispatcherSession,
        allocator: std.mem.Allocator,
        report_json: []const u8,
        bytecode_hex: []const u8,
    ) !void {
        try self.lifecycle.requireBackendBinding();
        try self.check.validateAndBindBytecode(allocator, report_json, bytecode_hex);
        self.lifecycle.completeBackendBinding();
    }

    /// Completes the session and returns the certificate bytes produced by the
    /// underlying gate. No parsing or reserialization occurs in the registry.
    pub fn finishCertificate(
        self: *DispatcherSession,
        kind: CertificateKind,
    ) ![]const u8 {
        try self.lifecycle.finish(kind);
        return self.check.certificate_json;
    }

    pub fn writeVerificationSummary(
        self: *const DispatcherSession,
        writer: anytype,
    ) !void {
        const bytecode_bound = try self.lifecycle.finishedWithBytecode();
        try self.check.writeVerificationSummary(writer, bytecode_bound);
    }

    const State = enum {
        prepared,
        backend_bound,
        finished_sir,
        finished_bytecode,
    };

    const Lifecycle = struct {
        state: State = .prepared,

        fn requireBackendBinding(self: *const Lifecycle) !void {
            return switch (self.state) {
                .prepared => {},
                .backend_bound => error.DispatcherBackendAlreadyBound,
                .finished_sir, .finished_bytecode => error.DispatcherCertificateAlreadyFinished,
            };
        }

        fn completeBackendBinding(self: *Lifecycle) void {
            std.debug.assert(self.state == .prepared);
            self.state = .backend_bound;
        }

        fn finish(self: *Lifecycle, kind: CertificateKind) !void {
            switch (kind) {
                .sir => switch (self.state) {
                    .prepared => self.state = .finished_sir,
                    .backend_bound => return error.DispatcherBackendOutputRequiresBytecodeCertificate,
                    .finished_sir, .finished_bytecode => return error.DispatcherCertificateAlreadyFinished,
                },
                .bytecode => switch (self.state) {
                    .prepared => return error.DispatcherBackendOutputNotBound,
                    .backend_bound => self.state = .finished_bytecode,
                    .finished_sir, .finished_bytecode => return error.DispatcherCertificateAlreadyFinished,
                },
            }
        }

        fn finishedWithBytecode(self: *const Lifecycle) !bool {
            return switch (self.state) {
                .finished_sir => false,
                .finished_bytecode => true,
                .prepared, .backend_bound => error.DispatcherCertificateNotFinished,
            };
        }
    };
};

test "kernel registry describes the blocking dispatcher phases" {
    const dispatcher = definition(.dispatcher);
    try std.testing.expectEqual(GateId.dispatcher, dispatcher.id);
    try std.testing.expectEqualStrings("dispatcher_userland", dispatcher.name);
    try std.testing.expect(dispatcher.blocking);
    try std.testing.expectEqualSlices(GatePhase, &dispatcher_phases, dispatcher.phases);
}

test "dispatcher lifecycle requires backend binding before bytecode finish" {
    var lifecycle: DispatcherSession.Lifecycle = .{};
    try std.testing.expectError(
        error.DispatcherBackendOutputNotBound,
        lifecycle.finish(.bytecode),
    );
    try lifecycle.requireBackendBinding();
    lifecycle.completeBackendBinding();
    try std.testing.expectError(
        error.DispatcherBackendAlreadyBound,
        lifecycle.requireBackendBinding(),
    );
    try lifecycle.finish(.bytecode);
    try std.testing.expect(try lifecycle.finishedWithBytecode());
    try std.testing.expectError(
        error.DispatcherCertificateAlreadyFinished,
        lifecycle.finish(.bytecode),
    );
}

test "dispatcher lifecycle permits a SIR-only certificate" {
    var lifecycle: DispatcherSession.Lifecycle = .{};
    try lifecycle.finish(.sir);
    try std.testing.expect(!try lifecycle.finishedWithBytecode());
    try std.testing.expectError(
        error.DispatcherCertificateAlreadyFinished,
        lifecycle.requireBackendBinding(),
    );
}

test "dispatcher session returns the gate certificate without reserialization" {
    const certificate_json =
        \\{
        \\  "schema_version": 1,
        \\  "proof_surface": "dispatcher_userland"
        \\}
    ;
    var session: DispatcherSession = .{
        .check = .{
            .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
            .certificate_json = certificate_json,
            .switch_manifest_json = "{}",
            .switch_count = 0,
            .case_count = 0,
        },
    };
    defer session.deinit();

    const finished = try session.finishCertificate(.sir);
    try std.testing.expectEqualStrings(certificate_json, finished);
    try std.testing.expectEqual(@intFromPtr(certificate_json.ptr), @intFromPtr(finished.ptr));
}
