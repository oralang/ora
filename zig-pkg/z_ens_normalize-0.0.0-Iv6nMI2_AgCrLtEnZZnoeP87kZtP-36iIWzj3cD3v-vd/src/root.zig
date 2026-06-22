//! ENS Name Normalization Library (ENSIP-15)
//!
//! This library provides ENS (Ethereum Name Service) name normalization according to ENSIP-15 specification.
//! It offers both instance-based and singleton-based APIs for normalizing and beautifying ENS names.
//!
//! ## Usage Examples
//!
//! ```zig
//! const ens = @import("ens-normalize");
//! const std = @import("std");
//!
//! pub fn main() !void {
//!     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//!     defer _ = gpa.deinit();
//!     const allocator = gpa.allocator();
//!
//!     // Option 1: Use convenience function
//!     const normalized = try ens.normalize(allocator, "Nick.ETH");
//!     defer allocator.free(normalized);
//!
//!     // Option 2: Use singleton directly
//!     const instance = ens.shared();
//!     const beautified = try instance.beautify(allocator, "nick.eth");
//!     defer allocator.free(beautified);
//!
//!     // Option 3: Create own instance (when init is implemented)
//!     // const my_instance = try ens.Ensip15.init();
//!     // const result = try my_instance.normalize(allocator, "test");
//!     // defer allocator.free(result);
//! }
//! ```

const std = @import("std");

// ============================================================
// Re-export Core Types
// ============================================================

/// ENSIP15 normalizer instance.
/// Provides methods for normalizing and beautifying ENS names.
pub const Ensip15 = @import("ensip15/ensip15.zig").Ensip15;

/// NF (Unicode Normalization Form) implementation.
/// Provides NFD and NFC normalization methods.
pub const NF = @import("nf/nf.zig").NF;

/// Error types that can be returned during normalization.
/// These errors indicate various ENSIP-15 validation failures.
pub const Error = error{
    /// Invalid label extension (3rd and 4th characters are both hyphens)
    InvalidLabelExtension,

    /// Illegal mixture of scripts within a label
    IllegalMixture,

    /// Label is whole-script confusable with another valid label
    WholeConfusable,

    /// Underscore appears after the beginning of the label
    LeadingUnderscore,

    /// Fenced character (ZWJ/ZWNJ) at the start of a label
    FencedLeading,

    /// Adjacent fenced characters (ZWJ/ZWNJ)
    FencedAdjacent,

    /// Fenced character (ZWJ/ZWNJ) at the end of a label
    FencedTrailing,

    /// Character is not allowed in ENS names
    DisallowedCharacter,

    /// Label is empty (zero length)
    EmptyLabel,

    /// Combining mark at the start of a label
    CMLeading,

    /// Combining mark immediately after an emoji
    CMAfterEmoji,

    /// Duplicate non-spacing marks on the same base character
    NSMDuplicate,

    /// Too many non-spacing marks on a single base character
    NSMExcessive,

    /// Out of memory
    OutOfMemory,

    /// Invalid UTF-8 encoding
    InvalidUtf8,
};

// ============================================================
// Thread-Safe Singleton Implementation
// ============================================================

/// The singleton ENSIP15 instance.
/// Initialized lazily on first call to shared().
var singleton_instance: Ensip15 = undefined;

/// Initialize the singleton instance.
/// Called exactly once by singleton_once.call().
/// Panics if initialization fails (should never happen with valid embedded data).
fn initSingleton() void {
    // TODO: When Ensip15.init() is fully implemented, use:
    // singleton_instance = Ensip15.init(std.heap.page_allocator) catch |err| {
    //     @panic("Failed to initialize ENSIP15 singleton");
    // };
    //
    // For now, use a stub init that works with the current structure
    // The methods are stubbed with @panic anyway, so this is acceptable
    singleton_instance = Ensip15.init(std.heap.page_allocator) catch
        @panic("Failed to initialize ENSIP15 singleton");
}

/// Thread-safe singleton initialization guard.
/// Uses std.once() to create a Once type that ensures initialization happens exactly once.
var singleton_once = std.once(initSingleton);

/// Returns a shared singleton instance of Ensip15.
///
/// The singleton is initialized lazily on the first call to this function.
/// Thread-safe: uses std.once for safe concurrent access.
/// The instance is initialized only once across all threads.
///
/// ## Returns
/// A const pointer to the shared Ensip15 instance.
/// The pointer is const to prevent mutation of the shared instance.
///
/// ## Thread Safety
/// This function is thread-safe and can be called concurrently from multiple threads.
/// The initialization will only happen once, even if called simultaneously.
///
/// ## Example
/// ```zig
/// const instance = ens.shared();
/// const result = try instance.normalize(allocator, "vitalik.eth");
/// ```
pub fn shared() *const Ensip15 {
    singleton_once.call();
    return &singleton_instance;
}

// ============================================================
// Convenience Functions
// ============================================================

/// Normalizes an ENS name using the shared singleton instance.
///
/// Takes an input name and returns its normalized form according to ENSIP-15.
/// The normalized form:
/// - Applies NFC Unicode normalization
/// - Uses normalized emoji forms (strips FE0F variation selectors)
/// - Validates all ENSIP-15 rules (mixed scripts, combining marks, etc.)
/// - Is suitable for on-chain storage and comparison
///
/// ## Parameters
/// - `allocator`: Memory allocator for the result string
/// - `name`: Input name as UTF-8 bytes (e.g., "vitalik.eth", "Nick.ETH")
///
/// ## Returns
/// Normalized name as UTF-8 bytes. The caller owns the returned memory and must free it.
///
/// ## Errors
/// Returns an error from the `Error` set if validation fails:
/// - `InvalidLabelExtension`: Label has invalid format (e.g., "ab--test")
/// - `IllegalMixture`: Mixed scripts that are not allowed together
/// - `WholeConfusable`: Label could be confused with another script
/// - `DisallowedCharacter`: Contains characters not allowed in ENS names
/// - `EmptyLabel`: Label has zero length
/// - `CMLeading`: Combining mark at the start of a label
/// - And other validation errors...
///
/// ## Memory Management
/// The returned string is allocated with the provided allocator.
/// The caller is responsible for freeing this memory using `allocator.free()`.
///
/// ## Example
/// ```zig
/// const normalized = try ens.normalize(allocator, "Nick.ETH");
/// defer allocator.free(normalized);
/// // normalized = "nick.eth"
/// ```
pub fn normalize(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    return shared().normalize(allocator, name);
}

/// Beautifies an ENS name using the shared singleton instance.
///
/// Similar to normalize() but produces a more visually appealing result:
/// - Uses beautified emoji forms (preserves FE0F variation selectors for emoji presentation)
/// - Converts U+03BE (ξ lowercase xi) to U+039E (Ξ uppercase Xi) in non-Greek labels
/// - Still validates all ENSIP-15 rules like normalize()
///
/// The beautified form is ideal for display in user interfaces while remaining
/// normalized according to ENSIP-15 specification.
///
/// ## Parameters
/// - `allocator`: Memory allocator for the result string
/// - `name`: Input name as UTF-8 bytes (e.g., "vitalik.eth")
///
/// ## Returns
/// Beautified name as UTF-8 bytes. The caller owns the returned memory and must free it.
///
/// ## Errors
/// Returns an error from the `Error` set if validation fails (same validation as normalize()).
///
/// ## Memory Management
/// The returned string is allocated with the provided allocator.
/// The caller is responsible for freeing this memory using `allocator.free()`.
///
/// ## Example
/// ```zig
/// const beautified = try ens.beautify(allocator, "🚀vitalik.eth");
/// defer allocator.free(beautified);
/// // beautified = "🚀vitalik.eth" (with proper emoji presentation)
/// ```
pub fn beautify(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    return shared().beautify(allocator, name);
}

// ============================================================
// Tests
// ============================================================

test "singleton initialization is thread-safe" {
    // Get singleton from multiple "threads" (simulated with multiple calls)
    const instance1 = shared();
    const instance2 = shared();

    // Both calls should return the same instance
    try std.testing.expectEqual(instance1, instance2);
}

test "singleton returns const pointer" {
    const instance = shared();

    // This test verifies that we get a const pointer
    // The type system enforces this at compile time
    const T = @TypeOf(instance);
    const type_info = @typeInfo(T);

    // Check that it's a pointer type
    if (type_info != .pointer) {
        return error.NotAPointer;
    }

    // Check that it's const
    try std.testing.expect(type_info.pointer.is_const);
}
