/-
Ora type system — regions.

This file defines the region layer used by located types.

  ρ ::= stack | memory | storage | transient | calldata

A located type will later have the form  σ ::= τ @ ρ.

Region describes WHERE a value currently lives. It is separate from PROVENANCE
(where the value originally came from).

## Bridge to the compiler (review finding 2)

The same five regions have three names across the codebase:

  Lean (here)   compiler `semantic.zig:Region`   spec / `region.zig:MemoryRegion`
  -----------   ------------------------------    -------------------------------
  stack         .none                             Stack
  memory        .memory                           Memory
  storage       .storage                          Storage
  transient     .transient                        TStore
  calldata      .calldata                         Calldata

So Lean `stack` IS the compiler's `.none` — the regionless / universally-coercible
lattice element. SEPARATE fact: the default region for an un-annotated `let x = …`
is `.memory`, not `.none`/stack (`declarationRegion(.none) = .memory`,
`src/sema/type_check.zig:99`). That is a surface-declaration elaboration default,
NOT the bottom of the region lattice.
-/

namespace Ora.Types

/-- Runtime/storage location of a value. `stack` ≡ compiler `.none`. -/
inductive Region where
  | stack
  | memory
  | storage
  | transient
  | calldata
  deriving Repr, DecidableEq

/-- Origin of a value. A SEPARATE axis from `Region` (`semantic.zig:Provenance`). -/
inductive Provenance where
  | local
  | calldata
  | storage
  | external
  deriving Repr, DecidableEq

/-! ## Read / write capability

    Runtime capabilities (used later for effect recording): every region is
    readable; every region except read-only `calldata` is writable. -/

/-- Regions that can be read from (all of them). -/
inductive CanReadFrom : Region → Prop where
  | stack     : CanReadFrom .stack
  | memory    : CanReadFrom .memory
  | storage   : CanReadFrom .storage
  | transient : CanReadFrom .transient
  | calldata  : CanReadFrom .calldata

/-- Regions that can be written to. `calldata` is read-only and excluded. -/
inductive CanWriteTo : Region → Prop where
  | stack     : CanWriteTo .stack
  | memory    : CanWriteTo .memory
  | storage   : CanWriteTo .storage
  | transient : CanWriteTo .transient

theorem calldata_readable : CanReadFrom .calldata := .calldata
theorem storage_writable  : CanWriteTo .storage  := .storage
theorem memory_writable   : CanWriteTo .memory   := .memory

theorem calldata_not_writable : ¬ CanWriteTo .calldata := by
  intro h; cases h

/-! ## Region coercion / implicit transitions

    THE move/coercion relation, a faithful arm-for-arm mirror of the compiler's
    `regionAssignable` (`src/sema/region.zig:31`) and the normative
    `docs/formal-specs/region-implicit-coercions.md`. `stack` plays `.none`.

    WARNING (review finding 1): the previous `CanMove := readable src ∧ writable
    dst` was UNSOUND — it admitted `calldata → storage`, `storage → transient`,
    etc., which the compiler REJECTS. "Readable + writable" is a strict
    over-approximation; the real relation is the table below. -/

/--
`a.assignableTo b` — may a value located in region `a` implicitly coerce to
region `b`? Mirrors `regionAssignable(from, to)` arm-for-arm.
-/
def Region.assignableTo : Region → Region → Bool
  -- `stack` (= `.none`): a regionless value coerces to ANY region.
  | .stack,     _          => true
  -- memory → stack / storage / transient (and itself); NOT calldata.
  | .memory,    .memory    => true
  | .memory,    .stack     => true
  | .memory,    .storage   => true
  | .memory,    .transient => true
  | .memory,    .calldata  => false
  -- storage → stack / memory (and itself); NOT transient, NOT calldata.
  | .storage,   .storage   => true
  | .storage,   .stack     => true
  | .storage,   .memory    => true
  | .storage,   .transient => false
  | .storage,   .calldata  => false
  -- transient → stack / memory (and itself); NOT storage, NOT calldata.
  | .transient, .transient => true
  | .transient, .stack     => true
  | .transient, .memory    => true
  | .transient, .storage   => false
  | .transient, .calldata  => false
  -- calldata → stack / memory (and itself, read-only); NOT storage, NOT transient.
  | .calldata,  .calldata  => true
  | .calldata,  .stack     => true
  | .calldata,  .memory    => true
  | .calldata,  .storage   => false
  | .calldata,  .transient => false

/-! ### Allowed transitions (match the `regionAssignable` tests) -/

theorem none_to_anything (b : Region) : Region.assignableTo .stack b = true := rfl
theorem memory_to_storage   : Region.assignableTo .memory .storage    = true := rfl
theorem memory_to_transient : Region.assignableTo .memory .transient  = true := rfl
theorem storage_to_memory   : Region.assignableTo .storage .memory    = true := rfl
theorem transient_to_memory : Region.assignableTo .transient .memory  = true := rfl
theorem calldata_to_memory  : Region.assignableTo .calldata .memory   = true := rfl

/-! ### Disallowed transitions

    These were the UNSOUND consequences of the old `CanMove`; they are now
    provably FALSE, matching the compiler. -/

theorem calldata_not_to_storage   : Region.assignableTo .calldata .storage    = false := rfl
theorem calldata_not_to_transient : Region.assignableTo .calldata .transient  = false := rfl
theorem storage_not_to_transient  : Region.assignableTo .storage .transient   = false := rfl
theorem transient_not_to_storage  : Region.assignableTo .transient .storage   = false := rfl
theorem storage_not_to_calldata   : Region.assignableTo .storage .calldata    = false := rfl
theorem transient_not_to_calldata : Region.assignableTo .transient .calldata  = false := rfl
theorem memory_not_to_calldata    : Region.assignableTo .memory .calldata     = false := rfl

/--
No MEMORY / STORAGE / TRANSIENT value implicitly coerces into `calldata`: the
only sources assignable to `calldata` are `stack` (a regionless `.none` value
inhabiting a calldata slot — not a runtime write) and `calldata` itself.
Mirrors "Disallowed: any write to Calldata" in `region-implicit-coercions.md`.
-/
theorem no_implicit_write_to_calldata (a : Region) :
    Region.assignableTo a .calldata = true ↔ a = .stack ∨ a = .calldata := by
  cases a <;> decide

/--
Storage and transient never directly transfer to each other (the spec requires
explicit syntax). Mirrors "Disallowed: Storage <-> TStore direct transfer".
-/
theorem no_storage_transient_direct :
    Region.assignableTo .storage .transient = false ∧
    Region.assignableTo .transient .storage = false := ⟨rfl, rfl⟩

/-! ## Region predicates -/

/-- Storage is persistent across transactions. -/
def Region.isPersistent : Region → Bool
  | .storage => true
  | _        => false

/-- Calldata is read-only. -/
def Region.isReadOnly : Region → Bool
  | .calldata => true
  | _         => false

/-- Stack and memory are local to a call. -/
def Region.isCallLocal : Region → Bool
  | .stack
  | .memory => true
  | _       => false

/-- Transient storage is transaction-local. -/
def Region.isTransactionLocal : Region → Bool
  | .transient => true
  | _          => false

theorem storage_is_persistent : Region.isPersistent .storage = true := rfl
theorem memory_not_persistent : Region.isPersistent .memory = false := rfl
theorem transient_not_persistent : Region.isPersistent .transient = false := rfl
theorem calldata_is_readonly : Region.isReadOnly .calldata = true := rfl
theorem storage_not_readonly : Region.isReadOnly .storage = false := rfl
theorem memory_is_call_local : Region.isCallLocal .memory = true := rfl
theorem stack_is_call_local : Region.isCallLocal .stack = true := rfl
theorem transient_is_transaction_local : Region.isTransactionLocal .transient = true := rfl

end Ora.Types
