/-
Ora type system — primitive type universe.

This file defines the primitive Ora types that exist at the static type layer.

Important:
This file does NOT define runtime values.

For example:

  u256 exists as a type here.

But this file does not yet define:

  UInt256 = Fin (2^256)

That belongs later in the value semantics layer.

Source of truth — the integer widths and `BuiltinTypeId` here are MACHINE-CHECKED
against the compiler by `formal/Ora/Sync.lean` (run `scripts/check-formal-sync.sh`):
the compiler emits its facts to `Ora/Generated/CompilerSnapshot.lean` and the
spec proves equality by `decide`. The remaining correspondence (type SHAPES) is
still matched by hand.
  * `src/types/semantic.zig`  — `TypeKind` / `Type` (the full type universe)
  * `src/types/builtin.zig`   — `BuiltinTypeId` (the spellable primitives)

This file models the SURFACE primitive universe — what a user can actually write.

Intentionally NOT modeled here (each verified against the compiler):
  * `never`   — NOT spellable: no lexer/parser keyword, absent from
                `BuiltinTypeId`. It is a compiler-internal bottom type: the type
                of compile-error / diverging expressions and the ⊥ of the type
                lattice (`src/sema/type_check.zig:7071`,
                `src/sema/type_descriptors.zig:215`). If ⊥ is needed for the
                metatheory it reappears in a later *internal* type layer — never
                as a surface Ora type.
  * `unknown` — compiler-internal fail-closed error sentinel (same bucket).
  * `comptime_integer` — a comptime-only unbounded literal type that lowers to a
                sized integer before runtime; belongs in a separate comptime
                layer, not in the runtime primitive universe.
-/

namespace Ora.Types

/--
Unsigned integer widths supported by Ora.

These are the only unsigned integer widths admitted by the compiler.
Invalid widths such as u24 are unrepresentable.
-/
inductive UIntWidth where
  | w8
  | w16
  | w32
  | w64
  | w128
  -- Deliberate unsigned 20-byte width for EVM address-sized arithmetic.
  -- `address` remains distinct, and the signed width universe has no peer.
  | w160
  | w256
  deriving Repr, DecidableEq

/--
Signed integer widths supported by Ora.

Ora deliberately has no i160.
-/
inductive SIntWidth where
  | w8
  | w16
  | w32
  | w64
  | w128
  | w256
  deriving Repr, DecidableEq

/--
Integer type descriptor.

This says which integer type exists.
It does not define integer values or arithmetic.
-/
inductive IntTy where
  | uint : UIntWidth → IntTy
  | sint : SIntWidth → IntTy
  deriving Repr, DecidableEq

/--
A valid fixed-bytes length.

Ora supports bytes1 through bytes32.
Invalid lengths such as bytes0 and bytes33 are unrepresentable.
-/
structure FixedBytesLen where
  n : Nat
  hLower : 1 ≤ n
  hUpper : n ≤ 32
  deriving Repr

/--
Smart constructor: builds a `FixedBytesLen` iff `1 ≤ n ≤ 32`, else `none`.

Useful when importing a length from compiler output (a `bytesN` decode or a
certificate), where the bound arrives as data rather than a proof.
-/
def FixedBytesLen.mk? (n : Nat) : Option FixedBytesLen :=
  if h1 : 1 ≤ n then
    if h2 : n ≤ 32 then some ⟨n, h1, h2⟩ else none
  else none

/--
Primitive Ora types.

This is still the static type layer, not runtime value semantics.

These mirror the spellable primitives in `src/types/builtin.zig`
(`BuiltinTypeId`): the integer family, `bool`, `address`, the `bytesN` family,
`string`, `bytes`, and `void`. Compiler-internal `TypeKind`s (`never`,
`unknown`) and the comptime-only `comptime_integer` are intentionally excluded
(see the file header).
-/
inductive PrimTy where
  | int : IntTy → PrimTy
  | bool
  | address
  | fixedBytes : FixedBytesLen → PrimTy
  | string
  | bytes
  | void
  deriving Repr

/-! ## Common primitive aliases -/

def u8 : PrimTy := .int (.uint .w8)
def u16 : PrimTy := .int (.uint .w16)
def u32 : PrimTy := .int (.uint .w32)
def u64 : PrimTy := .int (.uint .w64)
def u128 : PrimTy := .int (.uint .w128)
def u160 : PrimTy := .int (.uint .w160)
def u256 : PrimTy := .int (.uint .w256)

def i8 : PrimTy := .int (.sint .w8)
def i16 : PrimTy := .int (.sint .w16)
def i32 : PrimTy := .int (.sint .w32)
def i64 : PrimTy := .int (.sint .w64)
def i128 : PrimTy := .int (.sint .w128)
def i256 : PrimTy := .int (.sint .w256)

/-! ## Basic lemmas -/

/--
There is no signed 160-bit integer width in Ora.

This theorem is almost trivial because `SIntWidth` has no `w160` constructor.
That is the point: invalid integer widths are unrepresentable.
-/
theorem no_i160_constructor :
  ∀ w : SIntWidth,
    w = .w8 ∨
    w = .w16 ∨
    w = .w32 ∨
    w = .w64 ∨
    w = .w128 ∨
    w = .w256 := by
  intro w
  cases w <;> simp

/--
`address` is not the same primitive type as `u160`.

They may share a 160-bit runtime representation later, but they are distinct
at the type layer.
-/
theorem address_ne_u160 :
  PrimTy.address ≠ PrimTy.int (.uint .w160) := by
  intro h
  cases h

end Ora.Types
