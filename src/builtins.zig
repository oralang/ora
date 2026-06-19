const std = @import("std");

pub const Kind = enum {
    abi_decode,
    abi_decode_permissive,
    abi_encode,
    abi_signature,
    add_with_overflow,
    bit_cast,
    cast,
    chain_id,
    compile_error,
    concat,
    div_ceil,
    div_exact,
    div_floor,
    div_trunc,
    div_with_overflow,
    divmod,
    event_topic,
    keccak256,
    lock,
    mod_with_overflow,
    mul_with_overflow,
    neg_with_overflow,
    power_with_overflow,
    selector,
    shl_with_overflow,
    shr_with_overflow,
    size_of,
    slice,
    struct_fields,
    storage_derive,
    storage_range,
    storage_range_erase,
    storage_word_load,
    storage_word_store,
    sub_with_overflow,
    trait_methods,
    truncate,
    type_name,
    unlock,
};

pub const Entry = struct {
    name: []const u8,
    kind: Kind,
    signature: []const u8,
    documentation: []const u8,
    example: []const u8,
};

pub const entries = [_]Entry{
    .{ .name = "abiDecode", .kind = .abi_decode, .signature = "@abiDecode(T, bytes) -> Result<T, AbiDecodeError>", .documentation = "Decode ABI bytes into a typed value and report strict decoding errors.", .example = "const decoded = @abiDecode(u256, payload);" },
    .{ .name = "abiDecodePermissive", .kind = .abi_decode_permissive, .signature = "@abiDecodePermissive(T, bytes) -> Result<T, AbiDecodeError>", .documentation = "Decode ABI bytes using the permissive runtime decoder for supported target types.", .example = "const decoded = @abiDecodePermissive(u256, payload);" },
    .{ .name = "abiEncode", .kind = .abi_encode, .signature = "@abiEncode(value) -> bytes", .documentation = "Encode a value using its ABI representation.", .example = "const payload = @abiEncode(amount);" },
    .{ .name = "abiSignature", .kind = .abi_signature, .signature = "@abiSignature(function) -> string", .documentation = "Return the canonical ABI signature for a function reference.", .example = "const sig = @abiSignature(Token.transfer);" },
    .{ .name = "addWithOverflow", .kind = .add_with_overflow, .signature = "@addWithOverflow(lhs, rhs) -> (value, overflow)", .documentation = "Add two integers and return both the wrapped result and overflow flag.", .example = "const wrapped = @addWithOverflow(a, b);" },
    .{ .name = "bitCast", .kind = .bit_cast, .signature = "@bitCast(T, value) -> T", .documentation = "Reinterpret a value as another type with the same bit representation.", .example = "const raw = @bitCast(u256, value);" },
    .{ .name = "cast", .kind = .cast, .signature = "@cast(T, value) -> T", .documentation = "Convert a value to a target type using Ora's checked conversion rules.", .example = "const wide = @cast(u256, small);" },
    .{ .name = "chainId", .kind = .chain_id, .signature = "@chainId() -> u256", .documentation = "Return the current EVM chain id.", .example = "const id = @chainId();" },
    .{ .name = "compileError", .kind = .compile_error, .signature = "@compileError(message) -> noreturn", .documentation = "Emit a compile-time error with the provided message.", .example = "@compileError(\"unsupported type\");" },
    .{ .name = "concat", .kind = .concat, .signature = "@concat(lhs, rhs) -> string|bytes", .documentation = "Concatenate two strings or two byte buffers.", .example = "const joined = @concat(prefix, suffix);" },
    .{ .name = "divCeil", .kind = .div_ceil, .signature = "@divCeil(lhs, rhs) -> integer", .documentation = "Divide integers and round the quotient toward positive infinity.", .example = "const q = @divCeil(total, size);" },
    .{ .name = "divExact", .kind = .div_exact, .signature = "@divExact(lhs, rhs) -> integer", .documentation = "Divide integers and require that the division has no remainder.", .example = "const q = @divExact(total, size);" },
    .{ .name = "divFloor", .kind = .div_floor, .signature = "@divFloor(lhs, rhs) -> integer", .documentation = "Divide integers and round the quotient toward negative infinity.", .example = "const q = @divFloor(total, size);" },
    .{ .name = "divTrunc", .kind = .div_trunc, .signature = "@divTrunc(lhs, rhs) -> integer", .documentation = "Divide integers and truncate the quotient toward zero.", .example = "const q = @divTrunc(total, size);" },
    .{ .name = "divWithOverflow", .kind = .div_with_overflow, .signature = "@divWithOverflow(lhs, rhs) -> (value, overflow)", .documentation = "Divide two integers and return the result plus an overflow flag.", .example = "const wrapped = @divWithOverflow(a, b);" },
    .{ .name = "divmod", .kind = .divmod, .signature = "@divmod(lhs, rhs) -> (quotient, remainder)", .documentation = "Return both quotient and remainder for an integer division.", .example = "const qr = @divmod(total, size);" },
    .{ .name = "eventTopic", .kind = .event_topic, .signature = "@eventTopic(event) -> u256", .documentation = "Return the ABI event topic for an event reference.", .example = "const topic = @eventTopic(Token.Transfer);" },
    .{ .name = "keccak256", .kind = .keccak256, .signature = "@keccak256(value) -> u256", .documentation = "Hash a string or bytes value with EVM Keccak-256.", .example = "const hash = @keccak256(payload);" },
    .{ .name = "lock", .kind = .lock, .signature = "@lock(path) -> void", .documentation = "Acquire a verification/runtime lock for a storage path.", .example = "@lock(balances);" },
    .{ .name = "modWithOverflow", .kind = .mod_with_overflow, .signature = "@modWithOverflow(lhs, rhs) -> (value, overflow)", .documentation = "Compute integer remainder and return an overflow flag.", .example = "const wrapped = @modWithOverflow(a, b);" },
    .{ .name = "mulWithOverflow", .kind = .mul_with_overflow, .signature = "@mulWithOverflow(lhs, rhs) -> (value, overflow)", .documentation = "Multiply two integers and return both the wrapped result and overflow flag.", .example = "const wrapped = @mulWithOverflow(a, b);" },
    .{ .name = "negWithOverflow", .kind = .neg_with_overflow, .signature = "@negWithOverflow(value) -> (value, overflow)", .documentation = "Negate an integer and return both the wrapped result and overflow flag.", .example = "const wrapped = @negWithOverflow(value);" },
    .{ .name = "powerWithOverflow", .kind = .power_with_overflow, .signature = "@powerWithOverflow(base, exp) -> (value, overflow)", .documentation = "Exponentiate integers and return both the wrapped result and overflow flag.", .example = "const wrapped = @powerWithOverflow(base, exp);" },
    .{ .name = "selector", .kind = .selector, .signature = "@selector(function) -> u32", .documentation = "Return the four-byte ABI selector for a function reference.", .example = "const selector = @selector(Token.transfer);" },
    .{ .name = "shlWithOverflow", .kind = .shl_with_overflow, .signature = "@shlWithOverflow(lhs, rhs) -> (value, overflow)", .documentation = "Shift left and return both the wrapped result and overflow flag.", .example = "const wrapped = @shlWithOverflow(value, bits);" },
    .{ .name = "shrWithOverflow", .kind = .shr_with_overflow, .signature = "@shrWithOverflow(lhs, rhs) -> (value, overflow)", .documentation = "Shift right and return both the result and overflow flag.", .example = "const wrapped = @shrWithOverflow(value, bits);" },
    .{ .name = "sizeOf", .kind = .size_of, .signature = "@sizeOf(T) -> u256", .documentation = "Return the ABI/storage size of a type where supported.", .example = "const size = @sizeOf(u256);" },
    .{ .name = "slice", .kind = .slice, .signature = "@slice(value, start, length) -> string|bytes", .documentation = "Take a bounded slice from a string or bytes value.", .example = "const part = @slice(data, 0, 32);" },
    .{ .name = "structFields", .kind = .struct_fields, .signature = "@structFields(Struct) -> comptime metadata", .documentation = "Return compile-time metadata for the fields of a struct.", .example = "const fields = @structFields(VaultConfig);" },
    .{ .name = "storageDerive", .kind = .storage_derive, .signature = "@storageDerive(namespace, keys...) -> StorageSlot", .documentation = "Derive an opaque computed-storage slot from a compile-time namespace and scalar runtime keys.", .example = "const slot = @storageDerive(\"vault.position\", owner);" },
    .{ .name = "storageRange", .kind = .storage_range, .signature = "@storageRange(slot, len) -> StorageRange", .documentation = "Create an opaque bounded computed-storage range from a slot and word count.", .example = "const range = @storageRange(slot, 4);" },
    .{ .name = "storageRangeErase", .kind = .storage_range_erase, .signature = "@storageRangeErase(range) -> void", .documentation = "Erase every word in a bounded computed-storage range.", .example = "@storageRangeErase(range);" },
    .{ .name = "storageWordLoad", .kind = .storage_word_load, .signature = "@storageWordLoad(slot, offset) -> u256", .documentation = "Load a storage word through an opaque computed-storage capability.", .example = "const word = @storageWordLoad(slot, 0);" },
    .{ .name = "storageWordStore", .kind = .storage_word_store, .signature = "@storageWordStore(slot, offset, value) -> void", .documentation = "Store one word through an opaque computed-storage capability.", .example = "@storageWordStore(slot, 0, value);" },
    .{ .name = "subWithOverflow", .kind = .sub_with_overflow, .signature = "@subWithOverflow(lhs, rhs) -> (value, overflow)", .documentation = "Subtract two integers and return both the wrapped result and overflow flag.", .example = "const wrapped = @subWithOverflow(a, b);" },
    .{ .name = "traitMethods", .kind = .trait_methods, .signature = "@traitMethods(Trait) -> comptime metadata", .documentation = "Return compile-time metadata for methods required by a trait.", .example = "const methods = @traitMethods(Token);" },
    .{ .name = "truncate", .kind = .truncate, .signature = "@truncate(T, value) -> T", .documentation = "Truncate an integer value to a narrower target type.", .example = "const small = @truncate(u32, value);" },
    .{ .name = "typeName", .kind = .type_name, .signature = "@typeName(T) -> string", .documentation = "Return the compile-time display name for a type.", .example = "const name = @typeName(u256);" },
    .{ .name = "unlock", .kind = .unlock, .signature = "@unlock(path) -> void", .documentation = "Release a verification/runtime lock for a storage path.", .example = "@unlock(balances);" },
};

const NameEntry = struct { []const u8, Kind };

pub const name_entries = buildNameEntries();

fn buildNameEntries() [entries.len]NameEntry {
    var result: [entries.len]NameEntry = undefined;
    for (entries, 0..) |entry, index| {
        result[index] = .{ entry.name, entry.kind };
    }
    return result;
}

pub const name_map = std.StaticStringMap(Kind).initComptime(name_entries);

pub fn kindForName(name: []const u8) ?Kind {
    return name_map.get(name);
}

pub fn entryForName(name: []const u8) ?Entry {
    const kind = kindForName(name) orelse return null;
    return entryForKind(kind);
}

pub fn entryForKind(kind: Kind) Entry {
    for (entries) |entry| {
        if (entry.kind == kind) return entry;
    }
    unreachable;
}

comptime {
    @setEvalBranchQuota(5000);
    if (entries.len != @typeInfo(Kind).@"enum".fields.len) {
        @compileError("builtin function registry must cover every BuiltinKind");
    }
    for (entries, 0..) |entry, index| {
        if (entry.documentation.len == 0) @compileError("builtin entry is missing documentation");
        if (entry.example.len == 0) @compileError("builtin entry is missing an example");
        for (entries[index + 1 ..]) |other| {
            if (std.mem.eql(u8, entry.name, other.name)) {
                @compileError("duplicate builtin function name");
            }
            if (entry.kind == other.kind) {
                @compileError("duplicate builtin function kind");
            }
        }
    }
}
