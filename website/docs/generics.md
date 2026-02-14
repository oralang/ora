---
sidebar_position: 4.5
---

# Generics (Comptime Type Parameters)

Ora supports **Zig-style comptime generics**: type parameters are `comptime` and lead to **monomorphization** (one concrete implementation per instantiation). There is no runtime polymorphism or vtables.

## Design principles

- **Explicit type arguments** — You always pass types at the call site (e.g. `max(u256, a, b)`). Inference from value types is not supported.
- **Compile-time only** — Generics are resolved and specialized during compilation. No generic code is emitted; only concrete, mangled instances appear in the output.
- **Same naming as Zig** — `comptime T: type` for type parameters; no angle-bracket syntax for type arguments in Ora (we use function-call style: `Pair(u256)`).

---

## Generic functions

### Syntax

Declare type parameters as the first parameter(s), using `comptime Name: type`:

```ora
fn max(comptime T: type, a: T, b: T) -> T {
    if (a > b) { return a; } else { return b; }
}
```

- `comptime T: type` means “T is a type known at compile time.”
- Other parameters and the return type can use `T` (or other type parameter names).

### Call site

Pass the type explicitly, then the value arguments:

```ora
let x: u256 = max(u256, 10, 20);   // T = u256
let y: u8 = max(u8, 1, 2);         // T = u8
```

The compiler monomorphizes each use into a dedicated function (e.g. `max__u256`, `max__u8`) and rewrites the call to the mangled name. Comptime-only calls (all arguments constant) may be folded away entirely.

### Where generic functions are allowed

- **Inside a contract** — Generic functions may be defined and used inside contracts. The generic template is not lowered to MLIR; only its monomorphized instances are.

---

## Generic structs

### Syntax

Type parameters appear in parentheses after the struct name:

```ora
struct Pair(comptime T: type) {
    first: T;
    second: T;
}
```

- Only `comptime Name: type` parameters are supported (no value parameters in the struct header).

### Type reference (annotations)

Use function-call style for the concrete type:

```ora
fn make_pair(a: u256, b: u256) -> Pair(u256) {
    return Pair(u256) { first: a, second: b };
}
```

- `Pair(u256)` is the type (e.g. in return type or variable declarations).
- `Pair(u256) { first: a, second: b }` is the struct instantiation expression.

### Struct instantiation expression

To construct a value of a generic struct, use the same form: type arguments in parentheses, then a braced list of field initializers:

```ora
let p: Pair(u256) = Pair(u256) { first: 1, second: 2 };
```

Multiple type parameters (when supported) would look like: `StructName(T, U) { ... }`.

### Monomorphization

Each distinct instantiation (e.g. `Pair(u256)`, `Pair(u8)`) becomes a separate struct in the output (e.g. `Pair__u256`, `Pair__u8`). The generic `Pair(comptime T: type)` declaration is not emitted; only the concrete instances are.

---

## Generic contracts (syntax only)

Contract-level type parameters are **parsed** but **not yet monomorphized**:

```ora
contract Token(comptime T: type) {
    // ...
}
```

- Such contracts are **skipped** during MLIR lowering (no code is generated for the generic contract template).
- Full contract monomorphization (creating `Token__u256`, etc.) is planned for a future release.

---

## Name mangling

Concrete instances are given unique names so they do not collide:

- **Functions:** `baseName__type1__type2` (e.g. `max__u256`, `max__u8`).
- **Structs:** same pattern (e.g. `Pair__u256`, `Pair__u8`).

Mangling is internal; user code always refers to the generic name and type arguments.

---

## Allowed type arguments

Type arguments must be **concrete types** that the compiler can resolve at compile time. Supported today:

- Primitive integer types: `u8`, `u16`, `u32`, `u64`, `u128`, `u256`, `i8`, … `i256`
- `bool`, `address`, `string`, `bytes`
- Other struct types (including other monomorphized generics) when used as type arguments

Type arguments are passed as **identifier expressions** (e.g. `u256`, `Pair(u8)` when that becomes supported as an expression). Comptime expressions that evaluate to a type are not yet supported.

---

## Summary table

| Feature              | Syntax / convention |
|----------------------|---------------------|
| Function type param  | `comptime T: type` as first param(s) |
| Struct type params   | `struct Name(comptime T: type) { ... }` |
| Contract type params | `contract Name(comptime T: type) { ... }` (parsed only) |
| Passing type to fn   | `fn_name(u256, a, b)` |
| Using generic type   | `Pair(u256)` in annotations |
| Instantiating struct | `Pair(u256) { first: a, second: b }` |
| Instance names       | `name__type1__type2` (internal) |

This document defines the **generic style** for Ora. For comptime evaluation and folding, see [Comptime](./research/comptime.md). For future trait-based constraints on type parameters, see the type system design doc.
