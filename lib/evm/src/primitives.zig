//! Ora-owned EVM primitive boundary.
//!
//! This module keeps the public shape that the EVM already used
//! (`primitives.Address.Address`, `primitives.AccessList.AccessList`, etc.)
//! while keeping common value types and protocol constants inside lib/evm.

const std = @import("std");

pub const Address = struct {
    pub const Address = @This();
    const Self = @This();

    pub const ZERO_ADDRESS: Self = .{ .bytes = [_]u8{0} ** 20 };
    pub const ZERO = Self.ZERO_ADDRESS;

    bytes: [20]u8,

    pub fn zero() Self {
        return Self.ZERO_ADDRESS;
    }

    pub fn toU256(addr: Self) u256 {
        var result: u256 = 0;
        for (addr.bytes) |byte| result = (result << 8) | byte;
        return result;
    }

    pub fn fromU256(value: u256) Self {
        var addr: Self = undefined;
        var v = value;
        for (0..20) |i| {
            addr.bytes[19 - i] = @truncate(v & 0xff);
            v >>= 8;
        }
        return addr;
    }

    pub fn fromNumber(value: anytype) Self {
        return fromU256(@as(u256, @intCast(value)));
    }

    pub fn fromHex(hex_str: []const u8) !Self {
        var slice = hex_str;
        if (slice.len >= 2 and slice[0] == '0' and (slice[1] == 'x' or slice[1] == 'X')) {
            if (slice.len != 42) return error.InvalidHexFormat;
            slice = slice[2..];
        } else if (slice.len != 40) {
            return error.InvalidHexFormat;
        }

        var addr: Self = undefined;
        _ = std.fmt.hexToBytes(&addr.bytes, slice) catch return error.InvalidHexString;
        return addr;
    }

    pub const FromBytesError = error{InvalidAddressLength};

    pub fn fromBytes(bytes: []const u8) FromBytesError!Self {
        if (bytes.len != 20) return error.InvalidAddressLength;
        var addr: Self = undefined;
        @memcpy(&addr.bytes, bytes[0..20]);
        return addr;
    }

    pub fn toBytes(address: Self) [20]u8 {
        return address.bytes;
    }

    pub fn toHex(address: Self) [42]u8 {
        var result: [42]u8 = undefined;
        result[0] = '0';
        result[1] = 'x';
        _ = std.fmt.bytesToHex(result[2..], &address.bytes, .lower);
        return result;
    }

    pub fn toLowercase(address: Self) [42]u8 {
        return toHex(address);
    }

    pub fn toUppercase(address: Self) [42]u8 {
        var result: [42]u8 = undefined;
        result[0] = '0';
        result[1] = 'x';
        _ = std.fmt.bytesToHex(result[2..], &address.bytes, .upper);
        return result;
    }

    pub fn toAbiEncoded(address: Self) [32]u8 {
        var result = [_]u8{0} ** 32;
        @memcpy(result[12..32], &address.bytes);
        return result;
    }

    pub fn fromAbiEncoded(bytes: []const u8) !Self {
        if (bytes.len != 32) return error.InvalidAbiEncodedLength;
        return fromBytes(bytes[12..32]);
    }

    pub fn toShortHex(address: Self) [14]u8 {
        const hex = toHex(address);
        var result: [14]u8 = undefined;
        @memcpy(result[0..8], hex[0..8]);
        result[8] = '.';
        result[9] = '.';
        result[10] = '.';
        @memcpy(result[11..14], hex[39..42]);
        return result;
    }

    pub fn compare(a: Self, b: Self) i8 {
        for (a.bytes, b.bytes) |a_byte, b_byte| {
            if (a_byte < b_byte) return -1;
            if (a_byte > b_byte) return 1;
        }
        return 0;
    }

    pub fn lessThan(a: Self, b: Self) bool {
        return compare(a, b) < 0;
    }

    pub fn greaterThan(a: Self, b: Self) bool {
        return compare(a, b) > 0;
    }

    pub fn isZero(address: Self) bool {
        return std.mem.eql(u8, &address.bytes, &Self.ZERO_ADDRESS.bytes);
    }

    pub fn equals(a: Self, b: Self) bool {
        return std.mem.eql(u8, &a.bytes, &b.bytes);
    }

    pub fn eql(self: Self, other: Self) bool {
        return equals(self, other);
    }

    pub fn clone(address: Self) Self {
        return .{ .bytes = address.bytes };
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        const hex = toHex(self);
        try writer.writeAll(&hex);
    }

    pub fn formatNumber(self: Self, writer: anytype, options: std.fmt.Number) !void {
        _ = options;
        const hex = toHex(self);
        try writer.writeAll(&hex);
    }
};

pub const ZERO_ADDRESS = Address.ZERO_ADDRESS;

pub const Hash = struct {
    pub const SIZE: usize = 32;
    const Bytes = [SIZE]u8;
    pub const Hash = Bytes;
    pub const ZERO: Bytes = [_]u8{0} ** SIZE;
    pub const EMPTY_CODE_HASH: Bytes = .{
        0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
        0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
        0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
        0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
    };
    pub const EMPTY_TRIE_ROOT: Bytes = .{
        0x56, 0xe8, 0x1f, 0x17, 0x1b, 0xcc, 0x55, 0xa6,
        0xff, 0x83, 0x45, 0xe6, 0x92, 0xc0, 0xf8, 0x6e,
        0x5b, 0x48, 0xe0, 0x1b, 0x99, 0x6c, 0xad, 0xc0,
        0x01, 0x62, 0x2f, 0xb5, 0xe3, 0x63, 0xb4, 0x21,
    };

    pub fn fromBytes(bytes: []const u8) Bytes {
        std.debug.assert(bytes.len == SIZE);
        var result: Bytes = undefined;
        @memcpy(&result, bytes[0..SIZE]);
        return result;
    }

    pub fn fromHex(hex: []const u8) !Bytes {
        var slice = hex;
        if (slice.len >= 2 and slice[0] == '0' and (slice[1] == 'x' or slice[1] == 'X')) {
            slice = slice[2..];
        }
        if (slice.len != SIZE * 2) return error.InvalidHashLength;
        var result: Bytes = undefined;
        _ = std.fmt.hexToBytes(&result, slice) catch return error.InvalidHexCharacter;
        return result;
    }
};

pub const State = struct {
    pub const StorageKey = struct {
        const Self = @This();

        address: [20]u8,
        slot: u256,

        pub fn hash(self: Self, hasher: anytype) void {
            hasher.update(&self.address);
            var slot_bytes: [32]u8 = undefined;
            std.mem.writeInt(u256, &slot_bytes, self.slot, .big);
            hasher.update(&slot_bytes);
        }

        pub fn eql(a: Self, b: Self) bool {
            return std.mem.eql(u8, &a.address, &b.address) and a.slot == b.slot;
        }
    };
};

pub const StorageKey = State.StorageKey;

pub const logs = struct {
    const Self = @This();

    pub const Log = struct {
        address: Address.Address,
        topics: []const u256,
        data: []const u8,
    };

    pub const SENTINEL = Self.Log{
        .address = ZERO_ADDRESS,
        .topics = &[_]u256{},
        .data = &[_]u8{},
    };
};

pub const Log = logs.Log;

pub const AccessList = struct {
    pub const AccessListEntry = struct {
        address: Address.Address,
        storage_keys: []const Hash.Hash,
    };

    const List = []const AccessListEntry;
    pub const AccessList = List;

    pub const ACCESS_LIST_ADDRESS_COST: u64 = 2400;
    pub const ACCESS_LIST_STORAGE_KEY_COST: u64 = 1900;
    pub const COLD_ACCOUNT_ACCESS_COST: u64 = 2600;
    pub const COLD_STORAGE_ACCESS_COST: u64 = 2100;
    pub const WARM_STORAGE_ACCESS_COST: u64 = 100;

    pub fn calculateAccessListGasCost(access_list: List) u64 {
        var total: u64 = 0;
        for (access_list) |entry| {
            total += ACCESS_LIST_ADDRESS_COST;
            total += ACCESS_LIST_STORAGE_KEY_COST * entry.storage_keys.len;
        }
        return total;
    }

    pub fn isAddressInAccessList(access_list: List, addr: Address.Address) bool {
        for (access_list) |entry| {
            if (entry.address.eql(addr)) return true;
        }
        return false;
    }

    pub fn isStorageKeyInAccessList(access_list: List, addr: Address.Address, storage_key: Hash.Hash) bool {
        for (access_list) |entry| {
            if (!entry.address.eql(addr)) continue;
            for (entry.storage_keys) |key| {
                if (std.mem.eql(u8, &key, &storage_key)) return true;
            }
        }
        return false;
    }
};

pub const Authorization = struct {
    pub const PER_EMPTY_ACCOUNT_COST: u64 = 25000;
    pub const PER_AUTH_BASE_COST: u64 = 12500;
};

pub const GasConstants = struct {
    pub const GasQuickStep: u64 = 2;
    pub const GasFastestStep: u64 = 3;
    pub const GasFastStep: u64 = 5;
    pub const GasMidStep: u64 = 8;
    pub const GasSlowStep: u64 = 10;
    pub const GasExtStep: u64 = 20;

    pub const Keccak256Gas: u64 = 30;
    pub const Keccak256WordGas: u64 = 6;

    pub const SloadGas: u64 = 100;
    pub const ColdSloadCost: u64 = 2100;
    pub const ColdAccountAccessCost: u64 = 2600;
    pub const WarmStorageReadCost: u64 = 100;
    pub const SstoreSentryGas: u64 = 2300;
    pub const SstoreSetGas: u64 = 20000;
    pub const SstoreResetGas: u64 = 5000;
    pub const SstoreClearGas: u64 = 5000;
    pub const SstoreRefundGas: u64 = 4800;

    pub const JumpdestGas: u64 = 1;

    pub const LogGas: u64 = 375;
    pub const LogDataGas: u64 = 8;
    pub const LogTopicGas: u64 = 375;

    pub const CreateGas: u64 = 32000;
    pub const CallGas: u64 = 40;
    pub const CallStipend: u64 = 2300;
    pub const CallValueTransferGas: u64 = 9000;
    pub const CallNewAccountGas: u64 = 25000;

    pub const SelfdestructGas: u64 = 5000;
    pub const SelfdestructRefundGas: u64 = 24000;

    pub const MemoryGas: u64 = 3;
    pub const QuadCoeffDiv: u64 = 512;

    pub const CreateDataGas: u64 = 200;
    pub const InitcodeWordGas: u64 = 2;
    pub const MaxInitcodeSize: u64 = 49152;

    pub const TxGas: u64 = 21000;
    pub const TxGasContractCreation: u64 = 53000;
    pub const TxDataZeroGas: u64 = 4;
    pub const TxDataNonZeroGas: u64 = 16;

    pub const CopyGas: u64 = 3;
    pub const COPY_GAS: u64 = CopyGas;

    pub const TLoadGas: u64 = 100;
    pub const TStoreGas: u64 = 100;

    pub fn sstoreGasCostWithHardfork(
        current: u256,
        original: u256,
        new: u256,
        is_cold: bool,
        is_berlin_or_later: bool,
        is_istanbul_or_later: bool,
    ) u64 {
        var gas: u64 = 0;

        if (is_berlin_or_later and is_cold) gas += ColdSloadCost;

        if (is_istanbul_or_later) {
            if (original == current and current == new) {
                gas += if (is_berlin_or_later) WarmStorageReadCost else 200;
            } else if (original == current and current != new) {
                gas += if (original == 0) SstoreSetGas else SstoreResetGas;
            } else {
                gas += if (is_berlin_or_later) WarmStorageReadCost else 200;
            }
        } else {
            gas += if (current == 0 and new != 0) 20000 else 5000;
        }

        return gas;
    }
};

pub const TraceConfig = struct {
    disable_storage: bool = false,
    disable_stack: bool = false,
    disable_memory: bool = false,
    enable_memory: bool = false,
    enable_return_data: bool = false,
    tracer: ?[]const u8 = null,
    timeout: ?[]const u8 = null,

    pub fn from() TraceConfig {
        return .{};
    }

    pub fn enableAll() TraceConfig {
        return .{
            .enable_memory = true,
            .enable_return_data = true,
        };
    }

    pub fn disableAll() TraceConfig {
        return .{
            .disable_storage = true,
            .disable_stack = true,
            .disable_memory = true,
        };
    }

    pub fn tracksStorage(self: TraceConfig) bool {
        return !self.disable_storage;
    }

    pub fn tracksStack(self: TraceConfig) bool {
        return !self.disable_stack;
    }

    pub fn tracksMemory(self: TraceConfig) bool {
        return !self.disable_memory or self.enable_memory;
    }
};

pub const Hardfork = enum {
    FRONTIER,
    HOMESTEAD,
    DAO,
    TANGERINE_WHISTLE,
    SPURIOUS_DRAGON,
    BYZANTIUM,
    CONSTANTINOPLE,
    PETERSBURG,
    ISTANBUL,
    MUIR_GLACIER,
    BERLIN,
    LONDON,
    ARROW_GLACIER,
    GRAY_GLACIER,
    MERGE,
    SHANGHAI,
    CANCUN,
    PRAGUE,
    OSAKA,

    pub const DEFAULT = Hardfork.OSAKA;

    pub fn toInt(self: Hardfork) u32 {
        return @intFromEnum(self);
    }

    pub fn isAtLeast(self: Hardfork, target: Hardfork) bool {
        return self.toInt() >= target.toInt();
    }

    pub fn isBefore(self: Hardfork, target: Hardfork) bool {
        return self.toInt() < target.toInt();
    }

    pub fn isAfter(self: Hardfork, target: Hardfork) bool {
        return self.toInt() > target.toInt();
    }

    pub fn isEqual(self: Hardfork, other: Hardfork) bool {
        return self == other;
    }

    pub fn compare(a: Hardfork, b: Hardfork) i32 {
        const a_val: i32 = @intCast(a.toInt());
        const b_val: i32 = @intCast(b.toInt());
        return a_val - b_val;
    }

    pub fn gte(self: Hardfork, target: Hardfork) bool {
        return self.toInt() >= target.toInt();
    }

    pub fn lt(self: Hardfork, target: Hardfork) bool {
        return self.toInt() < target.toInt();
    }

    pub fn gt(self: Hardfork, target: Hardfork) bool {
        return self.toInt() > target.toInt();
    }

    pub fn lte(self: Hardfork, target: Hardfork) bool {
        return self.toInt() <= target.toInt();
    }

    pub fn equals(self: Hardfork, other: Hardfork) bool {
        return self.isEqual(other);
    }

    pub fn min(forks: []const Hardfork) ?Hardfork {
        if (forks.len == 0) return null;
        var minimum = forks[0];
        for (forks[1..]) |fork| {
            if (fork.toInt() < minimum.toInt()) minimum = fork;
        }
        return minimum;
    }

    pub fn max(forks: []const Hardfork) ?Hardfork {
        if (forks.len == 0) return null;
        var maximum = forks[0];
        for (forks[1..]) |fork| {
            if (fork.toInt() > maximum.toInt()) maximum = fork;
        }
        return maximum;
    }

    pub fn range(allocator: std.mem.Allocator, start: Hardfork, end: Hardfork) ![]Hardfork {
        const start_idx = start.toInt();
        const end_idx = end.toInt();
        const count = if (start_idx <= end_idx)
            end_idx - start_idx + 1
        else
            start_idx - end_idx + 1;

        const result = try allocator.alloc(Hardfork, count);
        for (result, 0..) |*fork, i| {
            const offset: u32 = @intCast(i);
            fork.* = if (start_idx <= end_idx)
                @enumFromInt(start_idx + offset)
            else
                @enumFromInt(start_idx - offset);
        }
        return result;
    }

    pub fn toString(self: Hardfork) []const u8 {
        return switch (self) {
            .FRONTIER => "frontier",
            .HOMESTEAD => "homestead",
            .DAO => "dao",
            .TANGERINE_WHISTLE => "tangerinewhistle",
            .SPURIOUS_DRAGON => "spuriousdragon",
            .BYZANTIUM => "byzantium",
            .CONSTANTINOPLE => "constantinople",
            .PETERSBURG => "petersburg",
            .ISTANBUL => "istanbul",
            .MUIR_GLACIER => "muirglacier",
            .BERLIN => "berlin",
            .LONDON => "london",
            .ARROW_GLACIER => "arrowglacier",
            .GRAY_GLACIER => "grayglacier",
            .MERGE => "merge",
            .SHANGHAI => "shanghai",
            .CANCUN => "cancun",
            .PRAGUE => "prague",
            .OSAKA => "osaka",
        };
    }

    pub fn isValidName(name: []const u8) bool {
        return fromString(name) != null;
    }

    pub fn allNames() [19][]const u8 {
        return .{
            "frontier",
            "homestead",
            "dao",
            "tangerinewhistle",
            "spuriousdragon",
            "byzantium",
            "constantinople",
            "petersburg",
            "istanbul",
            "muirglacier",
            "berlin",
            "london",
            "arrowglacier",
            "grayglacier",
            "merge",
            "shanghai",
            "cancun",
            "prague",
            "osaka",
        };
    }

    pub fn allIds() [19]Hardfork {
        return .{
            .FRONTIER,
            .HOMESTEAD,
            .DAO,
            .TANGERINE_WHISTLE,
            .SPURIOUS_DRAGON,
            .BYZANTIUM,
            .CONSTANTINOPLE,
            .PETERSBURG,
            .ISTANBUL,
            .MUIR_GLACIER,
            .BERLIN,
            .LONDON,
            .ARROW_GLACIER,
            .GRAY_GLACIER,
            .MERGE,
            .SHANGHAI,
            .CANCUN,
            .PRAGUE,
            .OSAKA,
        };
    }

    pub fn hasEIP1559(self: Hardfork) bool {
        return self.isAtLeast(.LONDON);
    }

    pub fn supportsEIP1559(self: Hardfork) bool {
        return self.hasEIP1559();
    }

    pub fn hasEIP3855(self: Hardfork) bool {
        return self.isAtLeast(.SHANGHAI);
    }

    pub fn supportsPUSH0(self: Hardfork) bool {
        return self.hasEIP3855();
    }

    pub fn hasEIP4844(self: Hardfork) bool {
        return self.isAtLeast(.CANCUN);
    }

    pub fn supportsBlobs(self: Hardfork) bool {
        return self.hasEIP4844();
    }

    pub fn hasEIP1153(self: Hardfork) bool {
        return self.isAtLeast(.CANCUN);
    }

    pub fn supportsTransientStorage(self: Hardfork) bool {
        return self.hasEIP1153();
    }

    pub fn isPostMerge(self: Hardfork) bool {
        return self.isAtLeast(.MERGE);
    }

    pub fn isPoS(self: Hardfork) bool {
        return self.isPostMerge();
    }

    pub fn mainnetActivationBlock(self: Hardfork) ?u64 {
        return switch (self) {
            .FRONTIER => 0,
            .HOMESTEAD => 1150000,
            .DAO => 1920000,
            .TANGERINE_WHISTLE => 2463000,
            .SPURIOUS_DRAGON => 2675000,
            .BYZANTIUM => 4370000,
            .CONSTANTINOPLE => 7280000,
            .PETERSBURG => 7280000,
            .ISTANBUL => 9069000,
            .MUIR_GLACIER => 9200000,
            .BERLIN => 12244000,
            .LONDON => 12965000,
            .ARROW_GLACIER => 13773000,
            .GRAY_GLACIER => 15050000,
            .MERGE => 15537394,
            .SHANGHAI => 17034870,
            .CANCUN => 19426587,
            .PRAGUE, .OSAKA => null,
        };
    }

    pub fn mainnetActivationTimestamp(self: Hardfork) ?u64 {
        return switch (self) {
            .FRONTIER,
            .HOMESTEAD,
            .DAO,
            .TANGERINE_WHISTLE,
            .SPURIOUS_DRAGON,
            .BYZANTIUM,
            .CONSTANTINOPLE,
            .PETERSBURG,
            .ISTANBUL,
            .MUIR_GLACIER,
            .BERLIN,
            .LONDON,
            .ARROW_GLACIER,
            .GRAY_GLACIER,
            => null,
            .MERGE => 1663224162,
            .SHANGHAI => 1681338455,
            .CANCUN => 1710338135,
            .PRAGUE, .OSAKA => null,
        };
    }

    pub fn isEipActive(self: Hardfork, eip: u16) bool {
        return switch (eip) {
            1559 => self.hasEIP1559(),
            3855 => self.hasEIP3855(),
            4844 => self.hasEIP4844(),
            1153 => self.hasEIP1153(),
            5656 => self.isAtLeast(.CANCUN),
            2537, 7002, 7251, 7702 => self.isAtLeast(.PRAGUE),
            7823, 7825, 7883, 7934 => self.isAtLeast(.OSAKA),
            else => false,
        };
    }

    pub fn fromString(name: []const u8) ?Hardfork {
        var clean_name = name;
        if (name.len > 0 and (name[0] == '>' or name[0] == '<')) {
            var start: usize = 1;
            if (name.len > 1 and name[1] == '=') start = 2;
            clean_name = name[start..];
        }

        if (std.ascii.eqlIgnoreCase(clean_name, "Frontier")) return .FRONTIER;
        if (std.ascii.eqlIgnoreCase(clean_name, "Homestead")) return .HOMESTEAD;
        if (std.ascii.eqlIgnoreCase(clean_name, "DAO")) return .DAO;
        if (std.ascii.eqlIgnoreCase(clean_name, "TangerineWhistle")) return .TANGERINE_WHISTLE;
        if (std.ascii.eqlIgnoreCase(clean_name, "SpuriousDragon")) return .SPURIOUS_DRAGON;
        if (std.ascii.eqlIgnoreCase(clean_name, "Byzantium")) return .BYZANTIUM;
        if (std.ascii.eqlIgnoreCase(clean_name, "Constantinople")) return .CONSTANTINOPLE;
        if (std.ascii.eqlIgnoreCase(clean_name, "Petersburg")) return .PETERSBURG;
        if (std.ascii.eqlIgnoreCase(clean_name, "ConstantinopleFix")) return .PETERSBURG;
        if (std.ascii.eqlIgnoreCase(clean_name, "Istanbul")) return .ISTANBUL;
        if (std.ascii.eqlIgnoreCase(clean_name, "MuirGlacier")) return .MUIR_GLACIER;
        if (std.ascii.eqlIgnoreCase(clean_name, "Berlin")) return .BERLIN;
        if (std.ascii.eqlIgnoreCase(clean_name, "London")) return .LONDON;
        if (std.ascii.eqlIgnoreCase(clean_name, "ArrowGlacier")) return .ARROW_GLACIER;
        if (std.ascii.eqlIgnoreCase(clean_name, "GrayGlacier")) return .GRAY_GLACIER;
        if (std.ascii.eqlIgnoreCase(clean_name, "Merge")) return .MERGE;
        if (std.ascii.eqlIgnoreCase(clean_name, "Paris")) return .MERGE;
        if (std.ascii.eqlIgnoreCase(clean_name, "Shanghai")) return .SHANGHAI;
        if (std.ascii.eqlIgnoreCase(clean_name, "Cancun")) return .CANCUN;
        if (std.ascii.eqlIgnoreCase(clean_name, "Prague")) return .PRAGUE;
        if (std.ascii.eqlIgnoreCase(clean_name, "Osaka")) return .OSAKA;
        return null;
    }
};

pub const ForkTransition = struct {
    from_fork: Hardfork,
    to_fork: Hardfork,
    at_block: ?u64,
    at_timestamp: ?u64,

    pub fn fromString(name: []const u8) ?ForkTransition {
        const to_index = std.mem.indexOf(u8, name, "To") orelse return null;
        const from_str = name[0..to_index];
        const from_fork = Hardfork.fromString(from_str) orelse return null;

        const at_index = std.mem.indexOf(u8, name[to_index..], "At") orelse return null;
        const at_pos = to_index + at_index;
        const to_str = name[to_index + 2 .. at_pos];
        const to_fork = Hardfork.fromString(to_str) orelse return null;
        const transition_str = name[at_pos + 2 ..];

        if (std.mem.indexOf(u8, transition_str, "Time") != null) {
            const time_index = std.mem.indexOf(u8, transition_str, "Time") orelse return null;
            const timestamp = parseTransitionNumber(transition_str[time_index + 4 ..]) catch return null;
            return .{
                .from_fork = from_fork,
                .to_fork = to_fork,
                .at_block = null,
                .at_timestamp = timestamp,
            };
        }

        const block = parseTransitionNumber(transition_str) catch return null;
        return .{
            .from_fork = from_fork,
            .to_fork = to_fork,
            .at_block = block,
            .at_timestamp = null,
        };
    }

    pub fn getActiveFork(self: ForkTransition, block_number: u64, timestamp: u64) Hardfork {
        if (self.at_block) |at_block| {
            return if (block_number >= at_block) self.to_fork else self.from_fork;
        }
        if (self.at_timestamp) |at_timestamp| {
            return if (timestamp >= at_timestamp) self.to_fork else self.from_fork;
        }
        return self.to_fork;
    }
};

const ParseTransitionNumberError = error{
    EmptyString,
    InvalidFormat,
};

fn parseTransitionNumber(str: []const u8) ParseTransitionNumberError!u64 {
    if (str.len == 0) return error.EmptyString;
    if (str[str.len - 1] == 'k') {
        const base = std.fmt.parseInt(u64, str[0 .. str.len - 1], 10) catch return error.InvalidFormat;
        return base * 1000;
    }
    return std.fmt.parseInt(u64, str, 10) catch error.InvalidFormat;
}

pub const Hex = struct {
    pub const HexError = error{
        InvalidHexFormat,
        InvalidHexLength,
        InvalidHexCharacter,
        OddLengthHex,
        ValueTooLarge,
        InvalidLength,
    };

    pub fn hexToBytes(allocator: std.mem.Allocator, hex: []const u8) ![]u8 {
        if (hex.len < 2 or !std.mem.eql(u8, hex[0..2], "0x")) {
            return HexError.InvalidHexFormat;
        }

        const hex_digits = hex[2..];
        if (hex_digits.len % 2 != 0) {
            return HexError.OddLengthHex;
        }

        const bytes = try allocator.alloc(u8, hex_digits.len / 2);
        errdefer allocator.free(bytes);

        var i: usize = 0;
        while (i < hex_digits.len) : (i += 2) {
            const high = hexCharToValue(hex_digits[i]) orelse return HexError.InvalidHexCharacter;
            const low = hexCharToValue(hex_digits[i + 1]) orelse return HexError.InvalidHexCharacter;
            bytes[i / 2] = high * 16 + low;
        }

        return bytes;
    }

    pub fn bytesToHex(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
        const hex_chars = "0123456789abcdef";
        const result = try allocator.alloc(u8, 2 + bytes.len * 2);
        result[0] = '0';
        result[1] = 'x';
        for (bytes, 0..) |byte, i| {
            result[2 + i * 2] = hex_chars[byte >> 4];
            result[2 + i * 2 + 1] = hex_chars[byte & 0x0f];
        }
        return result;
    }

    fn hexCharToValue(c: u8) ?u8 {
        return switch (c) {
            '0'...'9' => c - '0',
            'a'...'f' => c - 'a' + 10,
            'A'...'F' => c - 'A' + 10,
            else => null,
        };
    }
};

pub const Rlp = struct {
    pub const MAX_RLP_DEPTH: u32 = 32;

    pub const RlpError = error{
        InputTooShort,
        InputTooLong,
        LeadingZeros,
        NonCanonicalSize,
        InvalidLength,
        UnexpectedInput,
        InvalidRemainder,
        ExtraZeros,
        RecursionDepthExceeded,
    };

    pub const Decoded = struct {
        data: Data,
        remainder: []const u8,
    };

    pub const Data = union(enum) {
        List: []Data,
        String: []const u8,

        pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
            switch (self) {
                .List => |items| {
                    for (items) |item| {
                        item.deinit(allocator);
                    }
                    allocator.free(items);
                },
                .String => |value| allocator.free(value),
            }
        }
    };

    pub fn decode(allocator: std.mem.Allocator, input: []const u8, stream: bool) !Decoded {
        if (input.len == 0) {
            return .{
                .data = .{ .String = try allocator.dupe(u8, &.{}) },
                .remainder = &.{},
            };
        }

        const result = try decodeInner(allocator, input, 0);
        if (!stream and result.remainder.len > 0) {
            result.data.deinit(allocator);
            return RlpError.InvalidRemainder;
        }
        return result;
    }

    fn decodeInner(allocator: std.mem.Allocator, input: []const u8, depth: u32) !Decoded {
        if (input.len == 0) return RlpError.InputTooShort;
        if (depth >= MAX_RLP_DEPTH) return RlpError.RecursionDepthExceeded;

        const prefix = input[0];
        if (prefix <= 0x7f) {
            const result = try allocator.alloc(u8, 1);
            result[0] = prefix;
            return .{ .data = .{ .String = result }, .remainder = input[1..] };
        }

        if (prefix <= 0xb7) {
            const length: usize = prefix - 0x80;
            if (input.len - 1 < length) return RlpError.InputTooShort;
            if (length == 0) {
                return .{
                    .data = .{ .String = try allocator.dupe(u8, &.{}) },
                    .remainder = input[1..],
                };
            }
            if (length == 1 and input[1] < 0x80) return RlpError.NonCanonicalSize;

            const data = try allocator.alloc(u8, length);
            @memcpy(data, input[1 .. 1 + length]);
            return .{ .data = .{ .String = data }, .remainder = input[1 + length ..] };
        }

        if (prefix <= 0xbf) {
            const length_of_length: usize = prefix - 0xb7;
            if (input.len - 1 < length_of_length) return RlpError.InputTooShort;
            if (input[1] == 0) return RlpError.LeadingZeros;

            var total_length: usize = 0;
            for (input[1 .. 1 + length_of_length]) |byte| {
                total_length = (total_length << 8) + byte;
            }
            if (total_length < 56) return RlpError.NonCanonicalSize;
            if (input.len - 1 - length_of_length < total_length) return RlpError.InputTooShort;

            const data = try allocator.alloc(u8, total_length);
            @memcpy(data, input[1 + length_of_length .. 1 + length_of_length + total_length]);
            return .{
                .data = .{ .String = data },
                .remainder = input[1 + length_of_length + total_length ..],
            };
        }

        if (prefix <= 0xf7) {
            const length: usize = prefix - 0xc0;
            if (input.len - 1 < length) return RlpError.InputTooShort;
            if (length == 0) {
                return .{
                    .data = .{ .List = try allocator.alloc(Data, 0) },
                    .remainder = input[1..],
                };
            }
            const items = try decodeListPayload(allocator, input[1 .. 1 + length], depth);
            return .{ .data = .{ .List = items }, .remainder = input[1 + length ..] };
        }

        const length_of_length: usize = prefix - 0xf7;
        if (input.len - 1 < length_of_length) return RlpError.InputTooShort;
        if (input[1] == 0) return RlpError.LeadingZeros;

        var total_length: usize = 0;
        for (input[1 .. 1 + length_of_length]) |byte| {
            total_length = (total_length << 8) + byte;
        }
        if (total_length < 56) return RlpError.NonCanonicalSize;
        if (input.len - 1 - length_of_length < total_length) return RlpError.InputTooShort;

        const payload = input[1 + length_of_length .. 1 + length_of_length + total_length];
        const items = try decodeListPayload(allocator, payload, depth);
        return .{
            .data = .{ .List = items },
            .remainder = input[1 + length_of_length + total_length ..],
        };
    }

    fn decodeListPayload(allocator: std.mem.Allocator, payload: []const u8, depth: u32) ![]Data {
        var items: std.ArrayList(Data) = .{};
        errdefer {
            for (items.items) |item| item.deinit(allocator);
            items.deinit(allocator);
        }

        var remaining = payload;
        while (remaining.len > 0) {
            const decoded = try decodeInner(allocator, remaining, depth + 1);
            try items.append(allocator, decoded.data);
            remaining = decoded.remainder;
        }
        return try items.toOwnedSlice(allocator);
    }

    pub fn encodeLength(allocator: std.mem.Allocator, length: usize) ![]u8 {
        var len_bytes: std.ArrayList(u8) = .{};
        defer len_bytes.deinit(allocator);

        var temp = length;
        while (temp > 0) {
            try len_bytes.insert(allocator, 0, @as(u8, @intCast(temp & 0xff)));
            temp >>= 8;
        }
        return try len_bytes.toOwnedSlice(allocator);
    }

    pub fn encodeBytes(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
        if (bytes.len == 1 and bytes[0] < 0x80) {
            const result = try allocator.alloc(u8, 1);
            result[0] = bytes[0];
            return result;
        }

        if (bytes.len < 56) {
            const result = try allocator.alloc(u8, 1 + bytes.len);
            result[0] = 0x80 + @as(u8, @intCast(bytes.len));
            @memcpy(result[1..], bytes);
            return result;
        }

        const len_bytes = try encodeLength(allocator, bytes.len);
        defer allocator.free(len_bytes);
        const result = try allocator.alloc(u8, 1 + len_bytes.len + bytes.len);
        result[0] = 0xb7 + @as(u8, @intCast(len_bytes.len));
        @memcpy(result[1 .. 1 + len_bytes.len], len_bytes);
        @memcpy(result[1 + len_bytes.len ..], bytes);
        return result;
    }
};

pub const BlockHeader = struct {
    const Module = @This();

    pub const BLOOM_SIZE = 256;
    pub const NONCE_SIZE = 8;
    pub const MAX_EXTRA_DATA_SIZE = 32;

    pub const BlockHeader = struct {
        parent_hash: Hash.Hash = Hash.ZERO,
        ommers_hash: Hash.Hash = Hash.ZERO,
        beneficiary: Address.Address = Address.ZERO_ADDRESS,
        state_root: Hash.Hash = Hash.ZERO,
        transactions_root: Hash.Hash = Hash.ZERO,
        receipts_root: Hash.Hash = Hash.ZERO,
        logs_bloom: [BLOOM_SIZE]u8 = [_]u8{0} ** BLOOM_SIZE,
        difficulty: u256 = 0,
        number: u64 = 0,
        gas_limit: u64 = 0,
        gas_used: u64 = 0,
        timestamp: u64 = 0,
        extra_data: []const u8 = &.{},
        mix_hash: Hash.Hash = Hash.ZERO,
        nonce: [NONCE_SIZE]u8 = [_]u8{0} ** NONCE_SIZE,
        base_fee_per_gas: ?u256 = null,
        withdrawals_root: ?Hash.Hash = null,
        blob_gas_used: ?u64 = null,
        excess_blob_gas: ?u64 = null,
        parent_beacon_block_root: ?Hash.Hash = null,
    };

    pub fn init() Module.BlockHeader {
        return .{};
    }
};
