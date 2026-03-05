// ============================================================================
// Memory Region Definitions
// ============================================================================
//
// Shared region model for located types and region checks.
//
// ============================================================================

/// Memory regions matching Ora specification
pub const MemoryRegion = enum {
    Stack, // let/var (default)
    Memory, // memory let/memory var
    Storage, // storage let/storage var
    TStore, // tstore let/tstore var
    Calldata, // call data (read-only)
};
