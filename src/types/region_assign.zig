//! Pure region implicit-coercion rule.
//!
//! This is the (intended) single source of truth for the `regionAssignable`
//! relation — a pure function of the `Region` enum with no `sema` dependencies,
//! so it can be consumed both by the type checker and by the Lean spec emitter
//! (`src/formal/emit_compiler_snapshot.zig`).
//!
//! `src/sema/region.zig:regionAssignable` delegates here, so the type checker
//! and formal snapshot emitter cannot drift on this relation.

const semantic = @import("semantic.zig");

pub const Region = semantic.Region;

/// May a value located in region `from` implicitly coerce to region `to`?
/// Mirrors `docs/formal-specs/region-implicit-coercions.md`.
pub fn regionAssignable(from: Region, to: Region) bool {
    if (from == to) return true;
    return switch (from) {
        .none => true,
        .memory => switch (to) {
            .none, .storage, .transient => true,
            .memory, .calldata => false,
        },
        .storage => switch (to) {
            .none, .memory => true,
            .storage, .transient, .calldata => false,
        },
        .transient => switch (to) {
            .none, .memory => true,
            .storage, .transient, .calldata => false,
        },
        .calldata => switch (to) {
            .none, .memory => true,
            .storage, .transient, .calldata => false,
        },
    };
}
