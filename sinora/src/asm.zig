const std = @import("std");

// Minimal EVM assembler support used by Sinora release codegen.
//
// This file intentionally stays lower-level than the SIR emitter:
// - `Bytecode` only appends raw bytes and minimal PUSH immediates.
// - `LabelBytecode` adds symbolic labels and resolves them at the end.
//
// Why labels and patches exist:
//
// EVM does not have symbolic block names. A jump is encoded as a number on the
// stack followed by `JUMP` or `JUMPI`:
//
//     PUSH <destination_pc>
//     JUMP
//
// During codegen we usually need to emit a jump before the destination block has
// been emitted, so the final byte offset is not known yet. A `Label` is the
// symbolic name for "some bytecode position that will be marked later". A
// `Patch` records "we emitted a placeholder here; after all labels are known,
// replace this placeholder with the target label offset".
//
// Why this is not a single-pass assembler:
//
// The simple implementation would reserve `PUSH4 00 00 00 00` for every label
// reference and patch only the four bytes. That works, but wastes bytes. If the
// target offset is 0x2a, the optimal encoding is `PUSH1 2a`, not
// `PUSH4 00 00 00 2a`.
//
// Shrinking one placeholder changes the byte offsets of everything after it.
// That can make later label references smaller too. `LabelBytecode` therefore
// computes patch widths to a fixed point:
//
//     current widths -> final label offsets -> needed widths
//
// and repeats until no width changes. Only then can it emit final bytes.

pub const op = struct {
    // Raw EVM opcode bytes. These constants are deliberately numeric and flat:
    // higher-level code should decide stack semantics, gas, legality, and fork
    // availability before it reaches this assembler.
    pub const STOP: u8 = 0x00;
    pub const ADD: u8 = 0x01;
    pub const MUL: u8 = 0x02;
    pub const SUB: u8 = 0x03;
    pub const DIV: u8 = 0x04;
    pub const SDIV: u8 = 0x05;
    pub const MOD: u8 = 0x06;
    pub const SMOD: u8 = 0x07;
    pub const ADDMOD: u8 = 0x08;
    pub const MULMOD: u8 = 0x09;
    pub const EXP: u8 = 0x0a;
    pub const SIGNEXTEND: u8 = 0x0b;
    pub const LT: u8 = 0x10;
    pub const GT: u8 = 0x11;
    pub const SLT: u8 = 0x12;
    pub const SGT: u8 = 0x13;
    pub const EQ: u8 = 0x14;
    pub const ISZERO: u8 = 0x15;
    pub const AND: u8 = 0x16;
    pub const OR: u8 = 0x17;
    pub const XOR: u8 = 0x18;
    pub const NOT: u8 = 0x19;
    pub const BYTE: u8 = 0x1a;
    pub const SHL: u8 = 0x1b;
    pub const SHR: u8 = 0x1c;
    pub const SAR: u8 = 0x1d;
    pub const KECCAK256: u8 = 0x20;
    pub const ADDRESS: u8 = 0x30;
    pub const BALANCE: u8 = 0x31;
    pub const ORIGIN: u8 = 0x32;
    pub const CALLER: u8 = 0x33;
    pub const CALLVALUE: u8 = 0x34;
    pub const CALLDATALOAD: u8 = 0x35;
    pub const CALLDATASIZE: u8 = 0x36;
    pub const CALLDATACOPY: u8 = 0x37;
    pub const CODESIZE: u8 = 0x38;
    pub const CODECOPY: u8 = 0x39;
    pub const GASPRICE: u8 = 0x3a;
    pub const EXTCODESIZE: u8 = 0x3b;
    pub const EXTCODECOPY: u8 = 0x3c;
    pub const RETURNDATASIZE: u8 = 0x3d;
    pub const RETURNDATACOPY: u8 = 0x3e;
    pub const EXTCODEHASH: u8 = 0x3f;
    pub const BLOCKHASH: u8 = 0x40;
    pub const COINBASE: u8 = 0x41;
    pub const TIMESTAMP: u8 = 0x42;
    pub const NUMBER: u8 = 0x43;
    pub const DIFFICULTY: u8 = 0x44;
    pub const GASLIMIT: u8 = 0x45;
    pub const CHAINID: u8 = 0x46;
    pub const SELFBALANCE: u8 = 0x47;
    pub const BASEFEE: u8 = 0x48;
    pub const BLOBHASH: u8 = 0x49;
    pub const BLOBBASEFEE: u8 = 0x4a;
    pub const POP: u8 = 0x50;
    pub const MLOAD: u8 = 0x51;
    pub const MSTORE: u8 = 0x52;
    pub const MSTORE8: u8 = 0x53;
    pub const SLOAD: u8 = 0x54;
    pub const SSTORE: u8 = 0x55;
    pub const JUMP: u8 = 0x56;
    pub const JUMPI: u8 = 0x57;
    pub const JUMPDEST: u8 = 0x5b;
    pub const TLOAD: u8 = 0x5c;
    pub const TSTORE: u8 = 0x5d;
    pub const MCOPY: u8 = 0x5e;
    pub const PUSH0: u8 = 0x5f;
    pub const PUSH1: u8 = 0x60;
    pub const DUP1: u8 = 0x80;
    pub const DUP2: u8 = 0x81;
    pub const DUP4: u8 = 0x83;
    pub const SWAP1: u8 = 0x90;
    pub const LOG0: u8 = 0xa0;
    pub const LOG1: u8 = 0xa1;
    pub const LOG2: u8 = 0xa2;
    pub const LOG3: u8 = 0xa3;
    pub const LOG4: u8 = 0xa4;
    pub const CREATE: u8 = 0xf0;
    pub const CALL: u8 = 0xf1;
    pub const CALLCODE: u8 = 0xf2;
    pub const RETURN: u8 = 0xf3;
    pub const DELEGATECALL: u8 = 0xf4;
    pub const CREATE2: u8 = 0xf5;
    pub const STATICCALL: u8 = 0xfa;
    pub const REVERT: u8 = 0xfd;
    pub const INVALID: u8 = 0xfe;
    pub const SELFDESTRUCT: u8 = 0xff;
};

pub const Bytecode = struct {
    // Raw append-only byte stream.
    //
    // This type knows nothing about control flow, labels, jumps, source maps, or
    // final program counters. It only offers small helpers for appending EVM
    // bytes and minimal PUSH constants.
    //
    // Use this when every byte is already known. Use `LabelBytecode` when a byte
    // depends on a future label position.
    allocator: std.mem.Allocator,
    bytes: std.ArrayList(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) Bytecode {
        // Keep the allocator with the stream so callers do not need to thread it
        // through every byte append.
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Bytecode) void {
        // Frees the backing ArrayList storage. Does not own any returned slice
        // produced by `toOwnedSlice`.
        self.bytes.deinit(self.allocator);
    }

    pub fn reserve(self: *Bytecode, byte_capacity: usize) !void {
        // Optional performance hint. Small callers can ignore it and keep lazy
        // allocation; large codegen paths can reserve once from their known IR
        // size to avoid repeated ArrayList growth.
        try self.bytes.ensureTotalCapacity(self.allocator, byte_capacity);
    }

    pub fn pushOp(self: *Bytecode, byte: u8) !void {
        // Append one raw opcode byte. This function does not validate stack
        // arity, fork availability, or whether the opcode is legal at this
        // point in the program.
        try self.bytes.append(self.allocator, byte);
    }

    pub fn pushData(self: *Bytecode, bytes: []const u8) !void {
        // Append uninterpreted bytes. Used for payloads such as runtime bytecode
        // data sections, not for PUSH immediates that need an opcode prefix.
        try self.bytes.appendSlice(self.allocator, bytes);
    }

    // Current raw byte offset. Before label patching this is the placeholder
    // stream offset, not necessarily the final EVM program counter.
    pub fn position(self: Bytecode) usize {
        return self.bytes.items.len;
    }

    pub fn pushU32(self: *Bytecode, value: u32) !void {
        // Convenience wrapper. The EVM stack word is still 256-bit; this only
        // documents that callers know the value fits in u32.
        try appendFixedWidthPush(&self.bytes, self.allocator, minimalPushWidth(value), value);
    }

    pub fn pushU256(self: *Bytecode, value: u256) !void {
        // EIP-3855 `PUSH0` is the canonical shortest encoding for zero.
        if (value == 0) {
            try self.pushOp(op.PUSH0);
            return;
        }

        // Count only the bytes that are needed. This avoids materializing a
        // full 32-byte word and scanning its leading zeroes for common small
        // constants.
        var width: usize = 0;
        var remaining = value;
        while (remaining != 0) : (remaining >>= 8) {
            width += 1;
        }

        try self.pushOp(op.PUSH1 + @as(u8, @intCast(width - 1)));
        try self.bytes.ensureUnusedCapacity(self.allocator, width);

        var index = width;
        while (index > 0) {
            index -= 1;
            const shift: u8 = @intCast(index * 8);
            self.bytes.appendAssumeCapacity(@intCast((value >> shift) & 0xff));
        }
    }

    pub fn toOwnedSlice(self: *Bytecode) ![]const u8 {
        // Transfers ownership of the accumulated bytes to the caller and resets
        // the internal ArrayList. Callers must free the returned slice.
        return self.bytes.toOwnedSlice(self.allocator);
    }
};

pub const Label = struct {
    // A symbolic bytecode position.
    //
    // A label starts unresolved:
    //
    //     const revert = try code.newLabel();
    //
    // Later, codegen marks the current raw byte position:
    //
    //     try code.markJumpDest(revert);
    //
    // Jumps emitted before the mark can still refer to `revert`; those uses
    // become `Patch` entries. The label ID is stable even though the final byte
    // offset is not.
    id: u32,
};

const Patch = struct {
    // One unresolved numeric label reference.
    //
    // Example absolute patch:
    //
    //     PUSH4 00 00 00 00
    //           ^^^^^^^^^^^ placeholder at `byte_offset`
    //
    //     target = revert_block
    //     base   = null
    //
    // Final value: final_offset(revert_block)
    //
    // Example delta patch:
    //
    //     target = some_runtime_label
    //     base   = runtime_start
    //
    // Final value: final_offset(some_runtime_label) - final_offset(runtime_start)
    //
    // Delta patches are needed when runtime bytecode wants offsets relative to
    // runtime start even though the deployment byte stream also contains
    // initcode before runtime code.
    target: Label,
    base: ?Label = null,
    // Offset of the 4-byte placeholder payload, not the PUSH opcode. The PUSH
    // opcode is at `byte_offset - 1`.
    byte_offset: usize,
};

const DataPatch = struct {
    target: Label,
    base: ?Label = null,
    byte_offset: usize,
    width: u8,
};

pub const LabelError = error{
    UndefinedLabel,
    DuplicateLabel,
    BytecodeTooLarge,
    NegativeDelta,
};

pub const LabelBytecode = struct {
    // Label-aware bytecode builder.
    //
    // The raw stream always reserves five bytes for a label reference:
    // `PUSH4 00 00 00 00`. Final assembly may shrink that to `PUSH0`, `PUSH1`,
    // `PUSH2`, `PUSH3`, or leave it as `PUSH4`.
    allocator: std.mem.Allocator,
    bytecode: Bytecode,
    labels: std.ArrayList(?u32) = .empty,
    patches: std.ArrayList(Patch) = .empty,
    data_patches: std.ArrayList(DataPatch) = .empty,

    pub fn init(allocator: std.mem.Allocator) LabelBytecode {
        return .{
            .allocator = allocator,
            .bytecode = Bytecode.init(allocator),
        };
    }

    pub fn deinit(self: *LabelBytecode) void {
        self.data_patches.deinit(self.allocator);
        self.patches.deinit(self.allocator);
        self.labels.deinit(self.allocator);
        self.bytecode.deinit();
    }

    pub fn reserve(
        self: *LabelBytecode,
        byte_capacity: usize,
        label_capacity: usize,
        patch_capacity: usize,
    ) !void {
        // Optional performance hint for codegen. It reserves the three buffers
        // this assembler grows independently: raw bytes, symbolic labels, and
        // placeholder patches.
        try self.bytecode.reserve(byte_capacity);
        try self.labels.ensureTotalCapacity(self.allocator, label_capacity);
        try self.patches.ensureTotalCapacity(self.allocator, patch_capacity);
        try self.data_patches.ensureTotalCapacity(self.allocator, patch_capacity);
    }

    pub fn newLabel(self: *LabelBytecode) !Label {
        // Allocate a symbolic label ID.
        //
        // The label exists immediately so earlier code can refer to it, but its
        // offset is `null` until `mark` is called. Using an unmarked label at
        // final assembly is an error because there is no target address to
        // encode.
        const id: u32 = @intCast(self.labels.items.len);
        try self.labels.append(self.allocator, null);
        return .{ .id = id };
    }

    pub fn mark(self: *LabelBytecode, label: Label) !void {
        // Resolve a label to the current raw byte position.
        //
        // This stores the position in the placeholder stream, not the final EVM
        // PC. Final PCs may be smaller because earlier label references can
        // shrink from `PUSH4` to `PUSH0..PUSH3`.
        //
        // A label may be marked exactly once. Marking the same label twice would
        // make a branch target ambiguous, so it is rejected.
        const index: usize = @intCast(label.id);
        if (index >= self.labels.items.len) return LabelError.UndefinedLabel;
        if (self.labels.items[index] != null) return LabelError.DuplicateLabel;
        const position = self.bytecode.position();
        if (position > std.math.maxInt(u32)) return LabelError.BytecodeTooLarge;
        self.labels.items[index] = @intCast(position);
    }

    pub fn pushOp(self: *LabelBytecode, byte: u8) !void {
        // Forward raw op emission to the inner byte stream. Use this for bytes
        // whose value does not depend on final label offsets.
        try self.bytecode.pushOp(byte);
    }

    pub fn pushU32(self: *LabelBytecode, value: u32) !void {
        // Emit a known numeric constant immediately. This is not patched later.
        try self.bytecode.pushU32(value);
    }

    pub fn pushU256(self: *LabelBytecode, value: u256) !void {
        // Emit a known 256-bit numeric constant immediately. This is not patched
        // later.
        try self.bytecode.pushU256(value);
    }

    pub fn pushData(self: *LabelBytecode, bytes: []const u8) !void {
        // Append raw bytes such as data sections. Labels may point at data too;
        // the tests cover that case.
        try self.bytecode.pushData(bytes);
    }

    pub fn pushLabelRef(self: *LabelBytecode, label: Label) !void {
        // Emit a placeholder for an absolute label address.
        //
        // Raw stream now:
        //
        //     PUSH4 00 00 00 00
        //
        // Patch table now:
        //
        //     "the 4 zero bytes just emitted should become label's final offset"
        //
        // Final stream may become:
        //
        //     PUSH0
        //     PUSH1 xx
        //     PUSH2 xxxx
        //     PUSH3 xxxxxx
        //     PUSH4 xxxxxxxx
        try self.bytecode.pushOp(op.PUSH1 + 3);
        const byte_offset = self.bytecode.bytes.items.len;
        try self.bytecode.bytes.appendNTimes(self.allocator, 0, 4);
        try self.patches.append(self.allocator, .{
            .target = label,
            .byte_offset = byte_offset,
        });
    }

    pub fn pushLabelDelta(self: *LabelBytecode, base: Label, target: Label) !void {
        // Emit a placeholder for `target - base`.
        //
        // This is used for runtime-relative offsets. Deployment bytecode has
        // initcode followed by runtime bytecode, but runtime code often needs a
        // position relative to the runtime start. The `base` label gives that
        // origin explicitly.
        //
        // Negative deltas are rejected during finalization; this assembler only
        // supports forward-or-equal relative references.
        try self.bytecode.pushOp(op.PUSH1 + 3);
        const byte_offset = self.bytecode.bytes.items.len;
        try self.bytecode.bytes.appendNTimes(self.allocator, 0, 4);
        try self.patches.append(self.allocator, .{
            .target = target,
            .base = base,
            .byte_offset = byte_offset,
        });
    }

    pub fn pushLabelData(self: *LabelBytecode, label: Label, width: u8) !void {
        try self.pushLabelDeltaData(null, label, width);
    }

    pub fn pushLabelDeltaData(self: *LabelBytecode, base: ?Label, target: Label, width: u8) !void {
        // Emit fixed-width raw data containing a label address or label delta.
        //
        // Unlike `pushLabelRef`, this is not an EVM PUSH instruction and must
        // never shrink. Jump tables need a stable entry width so code can
        // compute `table + index * width` at runtime.
        std.debug.assert(width > 0 and width <= 4);
        const byte_offset = self.bytecode.bytes.items.len;
        try self.bytecode.bytes.appendNTimes(self.allocator, 0, width);
        try self.data_patches.append(self.allocator, .{
            .target = target,
            .base = base,
            .byte_offset = byte_offset,
            .width = width,
        });
    }

    pub fn markJumpDest(self: *LabelBytecode, label: Label) !void {
        // Common basic-block entry helper: resolve the label and emit the EVM
        // `JUMPDEST` opcode at that exact byte position.
        try self.mark(label);
        try self.pushOp(op.JUMPDEST);
    }

    pub fn pushJumpRef(self: *LabelBytecode, label: Label) !void {
        // Emit:
        //
        //     PUSH <absolute label pc>
        //     JUMP
        //
        // The numeric PUSH is patched at final assembly.
        try self.pushLabelRef(label);
        try self.pushOp(op.JUMP);
    }

    pub fn pushJumpiRef(self: *LabelBytecode, label: Label) !void {
        // Emit:
        //
        //     PUSH <absolute label pc>
        //     JUMPI
        //
        // The branch condition must already be on the EVM stack; this helper
        // only emits the destination and conditional jump opcode.
        try self.pushLabelRef(label);
        try self.pushOp(op.JUMPI);
    }

    pub fn pushJumpDelta(self: *LabelBytecode, base: Label, target: Label) !void {
        // Emit:
        //
        //     PUSH <target pc - base pc>
        //     JUMP
        //
        // This is useful when the pushed jump target is interpreted relative to
        // another code section, such as runtime start.
        try self.pushLabelDelta(base, target);
        try self.pushOp(op.JUMP);
    }

    pub fn pushJumpiDelta(self: *LabelBytecode, base: Label, target: Label) !void {
        // Conditional version of `pushJumpDelta`; the condition must already be
        // on the stack.
        try self.pushLabelDelta(base, target);
        try self.pushOp(op.JUMPI);
    }

    pub const Finalized = struct {
        allocator: std.mem.Allocator,
        bytes: []const u8,
        label_offsets: []const u32,

        pub fn deinit(self: *Finalized) void {
            self.allocator.free(self.label_offsets);
            self.allocator.free(self.bytes);
            self.* = undefined;
        }
    };

    pub fn toOwnedSlice(self: *LabelBytecode) ![]const u8 {
        const finalized = try self.toOwnedFinalized();
        self.allocator.free(finalized.label_offsets);
        return finalized.bytes;
    }

    pub fn toOwnedFinalized(self: *LabelBytecode) !Finalized {
        // Two-phase finalization:
        // 1. find the minimal width required by every label reference;
        // 2. rewrite the raw placeholder stream into final bytecode.
        const layout = try self.computePatchLayout();
        defer self.allocator.free(layout.widths);
        errdefer self.allocator.free(layout.label_offsets);
        const bytes = try self.assembleWithLayout(layout);
        return .{
            .allocator = self.allocator,
            .bytes = bytes,
            .label_offsets = layout.label_offsets,
        };
    }

    pub fn labelOffset(finalized: Finalized, label: Label) ?u32 {
        const index: usize = @intCast(label.id);
        if (index >= finalized.label_offsets.len) return null;
        return finalized.label_offsets[index];
    }

    fn computePatchLayout(self: *LabelBytecode) !PatchLayout {
        // Fixed-point solver for patch widths.
        //
        // Each placeholder starts as a logical width of zero. Given the current
        // widths, we can compute final label offsets, then compute the minimal
        // width needed for each patch value. If any width changes, offsets may
        // change too, so we repeat until stable.
        const widths = try self.allocator.alloc(u8, self.patches.items.len);
        errdefer self.allocator.free(widths);
        @memset(widths, 0);

        const labels_by_raw = try self.sortedRawLabels();
        defer self.allocator.free(labels_by_raw);
        const label_offsets = try self.allocator.alloc(u32, self.labels.items.len);
        errdefer self.allocator.free(label_offsets);

        var changed = true;
        var iteration: usize = 0;
        while (changed) {
            changed = false;
            iteration += 1;
            // Widths are monotonic for this placeholder model in practice; this
            // assert guards against accidental future changes that could make
            // the solver oscillate forever.
            std.debug.assert(iteration <= self.patches.items.len * 5 + 8);

            self.computeFinalLabelOffsets(labels_by_raw, widths, label_offsets);
            for (self.patches.items, 0..) |patch, index| {
                // Recompute this patch's final value using the current width
                // guess for all patches, then choose the smallest PUSH width
                // that can encode that value.
                const value = try patchValueFromOffsets(patch, label_offsets);
                const width = minimalPushWidth(value);
                if (width != widths[index]) {
                    widths[index] = width;
                    changed = true;
                }
            }
        }

        return .{
            .widths = widths,
            .label_offsets = label_offsets,
        };
    }

    fn sortedRawLabels(self: *LabelBytecode) ![]RawLabel {
        // Resolve every DEFINED label once and sort by raw byte position. The
        // solver then walks this list with the patch list to compute final
        // offsets in one pass per iteration, instead of scanning all patches for
        // every label lookup.
        //
        // A label slot may be declared but never defined (and never referenced
        // by a patch) — e.g. a label for a block that is not emitted. Those are
        // harmless and must be skipped here, NOT treated as an error; only a
        // patch that actually references an undefined label is a real failure,
        // and that is the patch resolver's concern. Erroring eagerly on any
        // undefined slot rejects otherwise-valid contracts.
        var defined_count: usize = 0;
        for (self.labels.items) |maybe_raw| {
            if (maybe_raw != null) defined_count += 1;
        }
        const labels_by_raw = try self.allocator.alloc(RawLabel, defined_count);
        errdefer self.allocator.free(labels_by_raw);
        var next: usize = 0;
        for (self.labels.items, 0..) |maybe_raw, index| {
            const raw = maybe_raw orelse continue;
            labels_by_raw[next] = .{
                .id = @intCast(index),
                .raw = raw,
            };
            next += 1;
        }
        std.mem.sortUnstable(RawLabel, labels_by_raw, {}, RawLabel.lessThan);
        return labels_by_raw;
    }

    fn computeFinalLabelOffsets(
        self: *LabelBytecode,
        labels_by_raw: []const RawLabel,
        widths: []const u8,
        label_offsets: []u32,
    ) void {
        // Patches are appended in bytecode order. Walk them once while walking
        // sorted labels, accumulating bytes saved by every placeholder that
        // ends before the current raw label position.
        var patch_index: usize = 0;
        var savings: u32 = 0;
        for (labels_by_raw) |label| {
            while (patch_index < self.patches.items.len and self.patches.items[patch_index].byte_offset + 4 <= label.raw) {
                savings += 4 - @as(u32, widths[patch_index]);
                patch_index += 1;
            }
            label_offsets[label.id] = label.raw - savings;
        }
    }

    fn assembleWithLayout(self: *LabelBytecode, layout: PatchLayout) ![]const u8 {
        // Walk the raw stream once. Ordinary byte ranges are copied as slices.
        // Whenever we hit a placeholder `PUSH4`, write the final shortest PUSH
        // instead and skip the original 5-byte placeholder. Patch values use
        // the final label offsets already computed by the solver.
        var output: std.ArrayList(u8) = .empty;
        errdefer output.deinit(self.allocator);

        const raw = self.bytecode.bytes.items;
        var final_len = raw.len;
        for (layout.widths) |width| {
            final_len -= 4 - @as(usize, @intCast(width));
        }
        try output.ensureTotalCapacity(self.allocator, final_len);

        var raw_cursor: usize = 0;
        for (self.patches.items, 0..) |patch, patch_index| {
            const push_offset = self.patchPushOffset(patch);
            output.appendSliceAssumeCapacity(raw[raw_cursor..push_offset]);

            // Replace the placeholder `PUSH4 00000000` with the shortest final
            // encoding selected by `computePatchLayout`.
            const value = try patchValueFromOffsets(patch, layout.label_offsets);
            appendFixedWidthPushAssumeCapacity(&output, layout.widths[patch_index], value);

            raw_cursor = push_offset + 5;
        }
        output.appendSliceAssumeCapacity(raw[raw_cursor..]);
        std.debug.assert(output.items.len == final_len);

        try self.applyDataPatches(&output, layout);
        return output.toOwnedSlice(self.allocator);
    }

    fn applyDataPatches(self: *LabelBytecode, output: *std.ArrayList(u8), layout: PatchLayout) !void {
        for (self.data_patches.items) |patch| {
            const final_offset = self.finalOffsetForRawOffset(patch.byte_offset, layout.widths);
            const value = try dataPatchValueFromOffsets(patch, layout.label_offsets);
            if (value >= (@as(u64, 1) << @as(u6, @intCast(patch.width * 8)))) return LabelError.BytecodeTooLarge;
            if (final_offset + patch.width > output.items.len) return LabelError.BytecodeTooLarge;
            for (0..patch.width) |i| {
                const shift: u5 = @intCast((patch.width - 1 - i) * 8);
                output.items[final_offset + i] = @intCast((value >> shift) & 0xff);
            }
        }
    }

    fn finalOffsetForRawOffset(self: *LabelBytecode, raw_offset: usize, widths: []const u8) usize {
        var savings: usize = 0;
        for (self.patches.items, 0..) |patch, index| {
            if (patch.byte_offset + 4 > raw_offset) break;
            savings += 4 - @as(usize, @intCast(widths[index]));
        }
        return raw_offset - savings;
    }

    fn patchPushOffset(self: *LabelBytecode, patch: Patch) usize {
        // `Patch.byte_offset` points at the placeholder payload. The opcode byte
        // immediately before it is the `PUSH4` placeholder that will be replaced
        // along with the payload.
        _ = self;
        std.debug.assert(patch.byte_offset > 0);
        return patch.byte_offset - 1;
    }
};

pub const ReserveHint = struct {
    byte_capacity: usize,
    label_capacity: usize,
    patch_capacity: usize,
};

pub fn estimateReserveHint(stats: anytype) ReserveHint {
    // Shared codegen-side reservation formula. `stats` is intentionally
    // structural: callers can pass `ir.Stats` without making this low-level
    // assembler import the IR module.
    return .{
        .byte_capacity = 128 +
            stats.data_bytes +
            stats.instructions * 16 +
            stats.terminators * 8 +
            stats.switch_cases * 8 +
            stats.blocks * 4,
        .label_capacity = 8 + stats.blocks + stats.data_segments + stats.functions * 2,
        .patch_capacity = 8 + stats.terminators * 2 + stats.switch_cases + stats.data_segments * 2,
    };
}

const RawLabel = struct {
    id: usize,
    raw: u32,

    fn lessThan(_: void, lhs: RawLabel, rhs: RawLabel) bool {
        if (lhs.raw == rhs.raw) return lhs.id < rhs.id;
        return lhs.raw < rhs.raw;
    }
};

const PatchLayout = struct {
    widths: []u8,
    label_offsets: []u32,
};

fn patchValueFromOffsets(patch: Patch, label_offsets: []const u32) !u32 {
    // Fast path used by the fixed-point solver after it has cached all final
    // label offsets for the current width guess.
    const target = label_offsets[@intCast(patch.target.id)];
    const base = if (patch.base) |base_label| label_offsets[@intCast(base_label.id)] else 0;
    if (target < base) return LabelError.NegativeDelta;
    return target - base;
}

fn dataPatchValueFromOffsets(patch: DataPatch, label_offsets: []const u32) !u32 {
    const target = label_offsets[@intCast(patch.target.id)];
    const base = if (patch.base) |base_label| label_offsets[@intCast(base_label.id)] else 0;
    if (target < base) return LabelError.NegativeDelta;
    return target - base;
}

fn minimalPushWidth(value: u32) u8 {
    // Width 0 means `PUSH0`; otherwise the result is the number of immediate
    // bytes after `PUSH1..PUSH4`.
    if (value == 0) return 0;
    if (value <= 0xff) return 1;
    if (value <= 0xffff) return 2;
    if (value <= 0xffffff) return 3;
    return 4;
}

fn appendFixedWidthPush(output: *std.ArrayList(u8), allocator: std.mem.Allocator, width: u8, value: u32) !void {
    // Emit the exact-width PUSH selected by the caller. Used when the target
    // ArrayList may still need to grow.
    var bytes: [5]u8 = undefined;
    const len = encodeFixedWidthPush(&bytes, width, value);
    try output.appendSlice(allocator, bytes[0..len]);
}

fn appendFixedWidthPushAssumeCapacity(output: *std.ArrayList(u8), width: u8, value: u32) void {
    // Emit the exact-width PUSH selected by the fixed-point solver. Used by the
    // final assembly pass after it has reserved the exact output size.
    var bytes: [5]u8 = undefined;
    const len = encodeFixedWidthPush(&bytes, width, value);
    output.appendSliceAssumeCapacity(bytes[0..len]);
}

fn encodeFixedWidthPush(bytes: *[5]u8, width: u8, value: u32) usize {
    // Encode one `PUSH0` or `PUSH1..PUSH4` instruction into a stack buffer and
    // return the number of bytes written.
    if (width == 0) {
        std.debug.assert(value == 0);
        bytes[0] = op.PUSH0;
        return 1;
    }

    bytes[0] = op.PUSH1 + width - 1;
    var index = width;
    while (index > 0) {
        index -= 1;
        const shift: u5 = @intCast(index * 8);
        bytes[1 + @as(usize, @intCast(width - index - 1))] = @intCast((value >> shift) & 0xff);
    }
    return 1 + @as(usize, @intCast(width));
}

pub fn writeHex(writer: anytype, bytes: []const u8) !void {
    // CLI/debug helper; keeps bytecode display deterministic and lowercase.
    const hex = "0123456789abcdef";
    try writer.writeAll("0x");
    var buffer: [4096]u8 = undefined;
    var out: usize = 0;
    for (bytes) |byte| {
        if (out + 2 > buffer.len) {
            try writer.writeAll(buffer[0..out]);
            out = 0;
        }
        buffer[out] = hex[byte >> 4];
        buffer[out + 1] = hex[byte & 0x0f];
        out += 2;
    }
    if (out != 0) {
        try writer.writeAll(buffer[0..out]);
    }
}

const evm_opcode_map = std.StaticStringMap(u8).initComptime(.{
    .{ "add", op.ADD },
    .{ "mul", op.MUL },
    .{ "sub", op.SUB },
    .{ "div", op.DIV },
    .{ "sdiv", op.SDIV },
    .{ "mod", op.MOD },
    .{ "smod", op.SMOD },
    .{ "addmod", op.ADDMOD },
    .{ "mulmod", op.MULMOD },
    .{ "exp", op.EXP },
    .{ "signextend", op.SIGNEXTEND },
    .{ "lt", op.LT },
    .{ "gt", op.GT },
    .{ "slt", op.SLT },
    .{ "sgt", op.SGT },
    .{ "eq", op.EQ },
    .{ "iszero", op.ISZERO },
    .{ "and", op.AND },
    .{ "or", op.OR },
    .{ "xor", op.XOR },
    .{ "not", op.NOT },
    .{ "byte", op.BYTE },
    .{ "shl", op.SHL },
    .{ "shr", op.SHR },
    .{ "sar", op.SAR },
    .{ "keccak256", op.KECCAK256 },
    .{ "address", op.ADDRESS },
    .{ "balance", op.BALANCE },
    .{ "origin", op.ORIGIN },
    .{ "caller", op.CALLER },
    .{ "callvalue", op.CALLVALUE },
    .{ "calldataload", op.CALLDATALOAD },
    .{ "calldatasize", op.CALLDATASIZE },
    .{ "calldatacopy", op.CALLDATACOPY },
    .{ "codesize", op.CODESIZE },
    .{ "codecopy", op.CODECOPY },
    .{ "gasprice", op.GASPRICE },
    .{ "extcodesize", op.EXTCODESIZE },
    .{ "extcodecopy", op.EXTCODECOPY },
    .{ "returndatasize", op.RETURNDATASIZE },
    .{ "returndatacopy", op.RETURNDATACOPY },
    .{ "extcodehash", op.EXTCODEHASH },
    .{ "blockhash", op.BLOCKHASH },
    .{ "coinbase", op.COINBASE },
    .{ "timestamp", op.TIMESTAMP },
    .{ "number", op.NUMBER },
    .{ "difficulty", op.DIFFICULTY },
    .{ "gaslimit", op.GASLIMIT },
    .{ "chainid", op.CHAINID },
    .{ "selfbalance", op.SELFBALANCE },
    .{ "basefee", op.BASEFEE },
    .{ "blobhash", op.BLOBHASH },
    .{ "blobbasefee", op.BLOBBASEFEE },
    .{ "sload", op.SLOAD },
    .{ "sstore", op.SSTORE },
    .{ "tload", op.TLOAD },
    .{ "tstore", op.TSTORE },
    .{ "log0", op.LOG0 },
    .{ "log1", op.LOG1 },
    .{ "log2", op.LOG2 },
    .{ "log3", op.LOG3 },
    .{ "log4", op.LOG4 },
    .{ "create", op.CREATE },
    .{ "create2", op.CREATE2 },
    .{ "call", op.CALL },
    .{ "callcode", op.CALLCODE },
    .{ "delegatecall", op.DELEGATECALL },
    .{ "staticcall", op.STATICCALL },
    .{ "mcopy", op.MCOPY },
});

pub fn evmOpcode(mnemonic: []const u8) ?u8 {
    // SIR mnemonic -> literal EVM opcode mapping for operations that need no
    // special codegen. `StaticStringMap` keeps this fixed EVM table in static
    // memory and avoids the old linear scan over every opcode name.
    return evm_opcode_map.get(mnemonic);
}

test "pushes minimal constants" {
    var bytecode = Bytecode.init(std.testing.allocator);
    defer bytecode.deinit();

    try bytecode.pushU256(0);
    try bytecode.pushU256(0x1234);

    const bytes = try bytecode.toOwnedSlice();
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, &.{ op.PUSH0, op.PUSH1 + 1, 0x12, 0x34 }, bytes);
}

test "patches minimal label references" {
    var bytecode = LabelBytecode.init(std.testing.allocator);
    defer bytecode.deinit();

    const target = try bytecode.newLabel();
    try bytecode.pushJumpRef(target);
    try bytecode.pushJumpiRef(target);
    try bytecode.markJumpDest(target);
    try bytecode.pushOp(op.STOP);

    const bytes = try bytecode.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualSlices(u8, &.{ op.PUSH1, 0x06, op.JUMP, op.PUSH1, 0x06, op.JUMPI, op.JUMPDEST, op.STOP }, bytes);
}

test "patch width solver chooses the least fixed point at push-size boundaries" {
    var bytecode = LabelBytecode.init(std.testing.allocator);
    defer bytecode.deinit();

    const target = try bytecode.newLabel();
    try bytecode.pushLabelRef(target);
    try bytecode.bytecode.bytes.appendNTimes(std.testing.allocator, op.STOP, 0x102 - 5);
    try bytecode.mark(target);

    const bytes = try bytecode.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqual(op.PUSH1, bytes[0]);
    try std.testing.expectEqual(@as(u8, 0xff), bytes[1]);
    try std.testing.expectEqual(@as(usize, 0xff), bytes.len);
}

test "patches minimal label deltas" {
    var bytecode = LabelBytecode.init(std.testing.allocator);
    defer bytecode.deinit();

    const start = try bytecode.newLabel();
    const end = try bytecode.newLabel();
    try bytecode.mark(start);
    try bytecode.pushOp(op.JUMPDEST);
    try bytecode.pushLabelDelta(start, end);
    try bytecode.pushOp(op.STOP);
    try bytecode.mark(end);

    const bytes = try bytecode.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualSlices(u8, &.{ op.JUMPDEST, op.PUSH1, 0x04, op.STOP }, bytes);
}

test "label delta jump helpers include branch opcodes" {
    var bytecode = LabelBytecode.init(std.testing.allocator);
    defer bytecode.deinit();

    const start = try bytecode.newLabel();
    const end = try bytecode.newLabel();
    try bytecode.markJumpDest(start);
    try bytecode.pushJumpDelta(start, end);
    try bytecode.pushJumpiDelta(start, end);
    try bytecode.markJumpDest(end);

    const bytes = try bytecode.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualSlices(u8, &.{ op.JUMPDEST, op.PUSH1, 0x07, op.JUMP, op.PUSH1, 0x07, op.JUMPI, op.JUMPDEST }, bytes);
}

test "labels can point at appended data" {
    var bytecode = LabelBytecode.init(std.testing.allocator);
    defer bytecode.deinit();

    const data = try bytecode.newLabel();
    try bytecode.pushLabelRef(data);
    try bytecode.mark(data);
    try bytecode.pushData(&.{ 0x11, 0x22 });

    const bytes = try bytecode.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualSlices(u8, &.{ op.PUSH1, 0x02, 0x11, 0x22 }, bytes);
}

test "fixed-width label data patches account for shrunk push patches" {
    var bytecode = LabelBytecode.init(std.testing.allocator);
    defer bytecode.deinit();

    const start = try bytecode.newLabel();
    const target = try bytecode.newLabel();
    try bytecode.mark(start);
    try bytecode.pushLabelRef(target);
    try bytecode.pushLabelDeltaData(start, target, 2);
    try bytecode.mark(target);
    try bytecode.pushOp(op.STOP);

    const bytes = try bytecode.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualSlices(u8, &.{ op.PUSH1, 0x04, 0x00, 0x04, op.STOP }, bytes);
}
