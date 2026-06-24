//! Pure region implicit-coercion rule.
//!
//! This is the (intended) single source of truth for the `regionAssignable`
//! relation — a pure function of the `Region` enum with no `sema` dependencies,
//! so it can be consumed both by the type checker and by the Lean spec emitter
//! (`tools/emit_formal_snapshot.zig`).
//!
//! DEDUP FOLLOW-UP: `src/sema/region.zig:regionAssignable` currently holds an
//! identical copy (it lives in a module with heavy deps that the emitter cannot
//! import). The intended end state is for `sema/region.zig` to re-export this
//! function so the two cannot drift. Until then they MUST be kept byte-identical
//! in behavior; the Lean gate checks the emitted table against the spec, which
//! independently mirrors the rule.

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
