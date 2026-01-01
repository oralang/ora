# Ora ABI v0.1 — Compiler-facing appendix

This appendix defines rules needed to implement stable IDs, validation, and emission.

---

## A1. Canonical JSON rules (for hashing)

To compute `typeId` (and optionally callable IDs):

1. Serialize JSON with:
   - UTF-8
   - object keys sorted lexicographically
   - no insignificant whitespace
2. Exclude the `typeId` field from the hashed payload.
3. Prefer numbers as strings when precision matters (e.g. `u256` constants) to avoid JSON number limits.

### Why “string numbers” for big ints?
JSON number handling differs across parsers; representing big values as strings avoids precision loss.

---

## A2. Type graph validation

A conforming compiler/emitter MUST validate:

- All referenced `typeId` exist.
- No recursive value types (cycle detection across `fields`, `components`, `elementType`, `base`).
- `enum.repr.typeId` resolves to an integer primitive.
- `refinement.base` resolves to a concrete encodable base type.
- Field names in `struct.fields[]` are unique.
- Tuple component ordering is stable (source order).

---

## A3. Callable validation

- `id` MUST be unique across all callables.
- For `function` and `error`:
  - `signature` MUST be canonical under the selected wire profile.
  - `wire.<profile>.selector` MUST match the profile’s selector derivation rules.
- For `event`:
  - `inputs[].indexed` must be boolean; absence implies `false`.

---

## A3.1 Effects and mutability projection (evm-default)

Ora effects are the authoritative source. When emitting Solidity-compatible JSON ABI:

- `reads` only → `stateMutability: "view"`
- no reads/writes/calls/value → `stateMutability: "pure"`
- `value` present → `stateMutability: "payable"`
- otherwise → `stateMutability: "nonpayable"`

If a compiler cannot prove purity, it should default to `nonpayable`.

---

## A4. Callable ID scheme (recommended)

`id = "c:" + canonical_signature`

Where `canonical_signature` is:
- `name(type1,type2,...)` for `function` and `error`
- `Name(type1,type2,...)` for `event`

> Alternative: content-hash callable nodes like types. Either is fine; v0.1 only requires stability + uniqueness.

---

## A5. Canonical type spelling (profile-dependent)

Each wire profile defines how to spell types in signatures and how to map `types[]` to wire types.

Implementation recommendation:
- Keep a single internal “canonical type string” per profile.
- Enforce it both in:
  - `callables[].signature`
  - `types[].wire[profile]`

---

## A6. Refinement predicate language (v0.1)

The `predicate` object MUST be a small AST with:
- `op`: one of `== != < <= > >= && || ! + - * / %` (you can restrict in v0.1)
- terminals:
  - `{ "var": "x" }`
  - `{ "const": "<string-int>" }`
- optional: `{ "constBool": true/false }`

Tooling requirement:
- A consumer MAY ignore `predicate` for execution, but SHOULD use it for validation/UI hints.

---

## A7. Emission pipeline (recommended)

1. Build internal type table.
2. Emit all `types` first (deduplicate; assign `typeId`).
3. Emit `callables` referencing typeIds.
4. Add `wireProfiles`.
5. (Optional) emit `ora.abi.schema.extras.json` for heavy annotations.

---

## A8. Versioning rules

- `schemaVersion` is semver-like but stringly typed for now.
- v0.1 consumers:
  - MUST reject unknown major versions.
  - SHOULD ignore unknown fields (forward compatibility).

---

## A9. JSON Schema (optional, recommended)

Provide a `ora-abi-0.1.schema.json` for:
- editor tooling
- CI validation
- third-party consumers

The compiler SHOULD be able to emit it or bundle it with releases.

---
