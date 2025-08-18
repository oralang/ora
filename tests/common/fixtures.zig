//! Test Fixtures - Test data management and loading system
//!
//! Provides a centralized system for managing test fixtures, loading test data,
//! and organizing test cases for different compiler components.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Test fixture categories
pub const FixtureCategory = enum {
    valid_tokens,
    error_cases,
    complex_programs,
    expressions,
    statements,
    declarations,
    contracts,

    pub fn getDirectory(self: FixtureCategory) []const u8 {
        return switch (self) {
            .valid_tokens => "tests/fixtures/lexer/valid_tokens",
            .error_cases => "tests/fixtures/lexer/error_cases",
            .complex_programs => "tests/fixtures/integration/complex_programs",
            .expressions => "tests/fixtures/parser/expressions",
            .statements => "tests/fixtures/parser/statements",
            .declarations => "tests/fixtures/parser/declarations",
            .contracts => "tests/fixtures/parser/contracts",
        };
    }
};

/// Test fixture metadata
pub const FixtureMetadata = struct {
    name: []const u8,
    description: []const u8,
    category: FixtureCategory,
    tags: []const []const u8,
    expected_result: ExpectedResult,

    pub const ExpectedResult = union(enum) {
        success: void,
        lexer_error: []const u8,
        parser_error: []const u8,
        semantic_error: []const u8,
        token_count: u32,
        node_count: u32,
    };
};

/// Test fixture with content and metadata
pub const TestFixture = struct {
    metadata: FixtureMetadata,
    content: []const u8,
    content_owned: bool = false,

    pub fn deinit(self: *TestFixture, allocator: Allocator) void {
        if (self.content_owned) {
            allocator.free(self.content);
        }
    }
};

/// Fixture manager for loading and caching test data
pub const FixtureManager = struct {
    allocator: Allocator,
    cache: std.HashMap([]const u8, TestFixture, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

    pub fn init(allocator: Allocator) FixtureManager {
        return FixtureManager{
            .allocator = allocator,
            .cache = std.HashMap([]const u8, TestFixture, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *FixtureManager) void {
        var iterator = self.cache.iterator();
        while (iterator.next()) |entry| {
            var fixture = entry.value_ptr;
            fixture.deinit(self.allocator);
            // Free the key (name_copy) that we allocated
            self.allocator.free(entry.key_ptr.*);
        }
        self.cache.deinit();
    }

    /// Load fixture by name
    pub fn loadFixture(self: *FixtureManager, name: []const u8) !TestFixture {
        // Check cache first
        if (self.cache.get(name)) |fixture| {
            return fixture;
        }

        // Load from embedded data
        const fixture = getEmbeddedFixture(name) orelse return error.FixtureNotFound;

        // Cache the fixture
        const name_copy = try self.allocator.dupe(u8, name);
        try self.cache.put(name_copy, fixture);

        return fixture;
    }

    /// Load all fixtures in a category
    pub fn loadCategory(self: *FixtureManager, category: FixtureCategory) ![]TestFixture {
        const fixtures = switch (category) {
            .valid_tokens => VALID_TOKEN_FIXTURES[0..],
            .error_cases => ERROR_CASE_FIXTURES[0..],
            .complex_programs => COMPLEX_PROGRAM_FIXTURES[0..],
            .expressions => EXPRESSION_FIXTURES[0..],
            .statements => STATEMENT_FIXTURES[0..],
            .declarations => DECLARATION_FIXTURES[0..],
            .contracts => CONTRACT_FIXTURES[0..],
        };

        return try self.allocator.dupe(TestFixture, fixtures);
    }
};

/// Get embedded fixture by name
fn getEmbeddedFixture(name: []const u8) ?TestFixture {
    // Check all embedded fixture arrays
    for (VALID_TOKEN_FIXTURES) |fixture| {
        if (std.mem.eql(u8, fixture.metadata.name, name)) {
            return fixture;
        }
    }

    for (ERROR_CASE_FIXTURES) |fixture| {
        if (std.mem.eql(u8, fixture.metadata.name, name)) {
            return fixture;
        }
    }

    return null;
}

// Embedded test fixtures (simplified)
const VALID_TOKEN_FIXTURES = [_]TestFixture{
    TestFixture{
        .metadata = FixtureMetadata{
            .name = "identifiers",
            .description = "Various valid identifier patterns",
            .category = .valid_tokens,
            .tags = &.{ "lexer", "identifiers" },
            .expected_result = .{ .token_count = 5 },
        },
        .content = "identifier _underscore camelCase UPPER_CASE snake_case",
    },
    TestFixture{
        .metadata = FixtureMetadata{
            .name = "numbers",
            .description = "Various number literal formats",
            .category = .valid_tokens,
            .tags = &.{ "lexer", "numbers" },
            .expected_result = .{ .token_count = 5 },
        },
        .content = "42 0x1A2B 0b1010 123.456 0xdeadbeef",
    },
};

const ERROR_CASE_FIXTURES = [_]TestFixture{
    TestFixture{
        .metadata = FixtureMetadata{
            .name = "unterminated_string",
            .description = "String literal missing closing quote",
            .category = .error_cases,
            .tags = &.{ "lexer", "errors", "strings" },
            .expected_result = .{ .lexer_error = "UnterminatedString" },
        },
        .content = "\"unterminated string",
    },
};

const COMPLEX_PROGRAM_FIXTURES = [_]TestFixture{
    TestFixture{
        .metadata = FixtureMetadata{
            .name = "simple_contract",
            .description = "Basic contract with functions",
            .category = .complex_programs,
            .tags = &.{ "integration", "contracts" },
            .expected_result = .success,
        },
        .content = "contract SimpleToken { storage { balances: map<address, u256> } }",
    },
};

const EXPRESSION_FIXTURES = [_]TestFixture{
    TestFixture{
        .metadata = FixtureMetadata{
            .name = "binary_expressions",
            .description = "Various binary expressions",
            .category = .expressions,
            .tags = &.{ "parser", "expressions", "binary" },
            .expected_result = .success,
        },
        .content = "a + b * c - d / e % f",
    },
};

const STATEMENT_FIXTURES = [_]TestFixture{
    TestFixture{
        .metadata = FixtureMetadata{
            .name = "variable_declarations",
            .description = "Various variable declaration patterns",
            .category = .statements,
            .tags = &.{ "parser", "statements", "declarations" },
            .expected_result = .success,
        },
        .content = "let x = 42; var y: u256 = 100; const z = \"hello\";",
    },
};

const DECLARATION_FIXTURES = [_]TestFixture{
    TestFixture{
        .metadata = FixtureMetadata{
            .name = "function_declarations",
            .description = "Various function declaration patterns",
            .category = .declarations,
            .tags = &.{ "parser", "declarations", "functions" },
            .expected_result = .success,
        },
        .content = "fn simple_function() {} pub fn public_function(param: u256) -> bool {}",
    },
};

const CONTRACT_FIXTURES = [_]TestFixture{
    TestFixture{
        .metadata = FixtureMetadata{
            .name = "basic_contract",
            .description = "Basic contract structure",
            .category = .contracts,
            .tags = &.{ "parser", "contracts", "basic" },
            .expected_result = .success,
        },
        .content = "contract BasicContract { storage { owner: address } }",
    },
};

// Tests
test "FixtureManager basic functionality" {
    var manager = FixtureManager.init(std.testing.allocator);
    defer manager.deinit();

    const fixture = try manager.loadFixture("identifiers");
    try std.testing.expect(std.mem.eql(u8, fixture.metadata.name, "identifiers"));
    try std.testing.expect(fixture.content.len > 0);
}

test "FixtureManager category loading" {
    var manager = FixtureManager.init(std.testing.allocator);
    defer manager.deinit();

    const fixtures = try manager.loadCategory(.valid_tokens);
    defer std.testing.allocator.free(fixtures);

    try std.testing.expect(fixtures.len > 0);
}
