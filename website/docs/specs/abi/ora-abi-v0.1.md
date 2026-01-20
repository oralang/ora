# Ora ABI v0.1

**Ora ABI** is a professional, tooling-first interface layer for smart contracts, with strong support for UI/UX, code generation, and stable type identities.

Ora ABI is split into two layers:

1. **Manifest (authoritative):** a self-describing interface + type graph + metadata.
2. **Wire profiles (optional):** concrete byte encoding rules used for calls/returns/errors/events.

> v0.1 focuses on the **manifest** and the **profile declaration mechanism**, so we can implement Ora tooling cleanly and evolve wire formats without breaking the ABI model.

---

## 1. Goals

- **Self-describing:** consumers can understand the interface from the manifest alone.
- **Type graph:** named types are defined once and referenced everywhere.
- **Tooling-first:** supports codegen, explorers, and UI generation (forms, constraints, docs).
- **Stable identity:** types and callables have stable IDs suitable for caching and upgrades.
- **Size-constrained:** manifest stays compact; heavy data is optional and externalizable.

---

## 2. Non-goals (v0.1)

- Define a single mandatory wire encoding for all deployments.
- Standardize a full UI framework schema (we provide portable hints, not a renderer spec).
- Support recursive value types (types that contain themselves) in the manifest model.

> Note: storage constructs (like mappings) are *describable* in the manifest (layout, access patterns, UI intents), but they are not treated as “first-class value types” for argument/return encoding in v0.1.

---

## 3. Files and serialization

### 3.1 Canonical file
- **Canonical:** `ora.abi.schema.json`

### 3.2 Allowed alternates
- `ora.abi.schema.toml`, `ora.abi.schema.yaml`, etc.
- Alternates must be **losslessly convertible** into the canonical JSON model.

### Rationale
- **Choose JSON (A) over TOML-only (B):**
  - **A:** ubiquitous parsers, browser-native tooling, straightforward canonicalization rules.
  - **B rejected:** TOML is pleasant to hand-edit, but ecosystem-wide canonical hashing and schema validation is harder.

---

## 4. Manifest structure

### 4.1 Top-level object

```json
{
  "schemaVersion": "ora-abi-0.1",
  "contract": {
    "name": "Token",
    "namespace": "com.example",
    "build": { "compiler": "ora", "version": "0.1.0", "commit": "abc123" }
  },

  "wireProfiles": [
    { "id": "evm-default", "kind": "evm", "encoding": "abi-v2" }
  ],

  "types": { },
  "callables": [ ]
}
```

### Required fields
- `schemaVersion`, `contract`, `types`, `callables`

### Optional fields
- `wireProfiles`

---

## 5. Type graph

### 5.1 Type identity

Every type node has:
- `typeId` — stable ID (recommended: content-hash based, `t:<hash>`)
- `kind` — discriminator
- `name` — optional human-friendly name
- `wire` — profile-specific mapping hints
- `meta` — constraints, docs, UI hints

**Recommended `typeId` scheme**
- `typeId = "t:" + blake3(canonical_json(typeNodeWithoutTypeId))`

#### Rationale
- **Content-hash IDs (A) vs sequential IDs (B):**
  - **A:** stable across rebuilds, great for caching, deduplication, minimal diffs.
  - **B rejected:** reordering declarations changes IDs → breaks caches and makes diffs noisy.

### 5.2 Supported kinds (v0.1)

- `primitive` (bool, int/uint, address, bytes, string)
- `array` (fixed length)
- `slice` (dynamic length)
- `tuple`
- `struct`
- `enum`
- `alias`
- `refinement`

> v0.1 rule: **no recursive value types** (direct or indirect self containment).

### 5.3 Type node examples

#### Primitive

```json
{
  "typeId": "t:u256",
  "kind": "primitive",
  "name": "u256",
  "wire": { "evm-default": { "type": "uint256" } }
}
```

#### Struct

```json
{
  "typeId": "t:User",
  "kind": "struct",
  "name": "User",
  "fields": [
    { "name": "owner", "typeId": "t:address" },
    { "name": "balance", "typeId": "t:Balance" }
  ],
  "wire": { "evm-default": { "as": "tuple" } },
  "meta": {
    "doc": "User record used by clients",
    "ui": { "label": "User" }
  }
}
```

#### Refinement (constraints as metadata)

```json
{
  "typeId": "t:Balance",
  "kind": "refinement",
  "base": "t:u256",
  "predicate": {
    "op": "<=",
    "lhs": { "var": "x" },
    "rhs": { "const": "1000000" }
  },
  "meta": {
    "ui": { "label": "Balance", "min": 0, "max": 1000000, "unit": "wei" }
  },
  "wire": { "evm-default": { "type": "uint256" } }
}
```

#### Rationale
- **Predicate AST (A) vs free-form string constraints (B):**
  - **A:** machine-checkable, safe, renderable across UIs consistently.
  - **B rejected:** hard to validate, ambiguous parsing, fragments tooling.

#### Enum

```json
{
  "typeId": "t:Status",
  "kind": "enum",
  "name": "Status",
  "repr": { "typeId": "t:u8" },
  "variants": [
    { "name": "Inactive", "value": 0 },
    { "name": "Active", "value": 1 }
  ],
  "meta": { "ui": { "widget": "select" } },
  "wire": { "evm-default": { "type": "uint8" } }
}
```

#### Rationale
- **Explicit `repr` (A) vs implicit repr (B):**
  - **A:** stable meaning across compiler versions and upgrades.
  - **B rejected:** repr can drift with variant changes or compiler updates.

---

## 6. Callables

All public-facing interactions live in `callables[]`.

### 6.1 Callable common shape

```json
{
  "id": "c:transfer(address,uint256)",
  "kind": "function",
  "name": "transfer",
  "signature": "transfer(address,uint256)",
  "wire": { "evm-default": { "selector": "0xa9059cbb" } },

  "inputs": [
    { "name": "to", "typeId": "t:address" },
    { "name": "amount", "typeId": "t:Balance" }
  ],
  "outputs": [
    { "name": "ok", "typeId": "t:bool" }
  ],

  "meta": {
    "doc": "Send tokens",
    "ui": {
      "group": "Transfers",
      "dangerLevel": "normal",
      "forms": {
        "amount": { "widget": "number" }
      }
    },
    "effects": [
      { "kind": "writes", "path": "balances[to]" },
      { "kind": "emits", "eventId": "c:Transfer(address,address,uint256)" }
    ]
  }
}
```

### 6.2 Errors

```json
{
  "id": "c:InsufficientBalance(uint256,uint256)",
  "kind": "error",
  "name": "InsufficientBalance",
  "signature": "InsufficientBalance(uint256,uint256)",
  "wire": { "evm-default": { "selector": "0x..." } },

  "inputs": [
    { "name": "required", "typeId": "t:u256" },
    { "name": "available", "typeId": "t:u256" }
  ],

  "meta": {
    "ui": { "messageTemplate": "Need {required}, have {available}." }
  }
}
```

### 6.3 Events (supported by the manifest)

```json
{
  "id": "c:Transfer(address,address,uint256)",
  "kind": "event",
  "name": "Transfer",
  "signature": "Transfer(address,address,uint256)",
  "inputs": [
    { "name": "from", "typeId": "t:address", "indexed": true },
    { "name": "to", "typeId": "t:address", "indexed": true },
    { "name": "amount", "typeId": "t:u256", "indexed": false }
  ],
  "meta": { "ui": { "group": "Transfers" } }
}
```

---

## 6.4 Effects and mutability (Ora-native)

Ora does not use Solidity's `view`, `pure`, or `payable` keywords. Instead, Ora ABI describes observable behavior using **effects**.

Effects live under `callables[].meta.effects` and are derived from compiler analysis:

- `reads`: function reads storage
- `writes`: function writes storage
- `emits`: function emits event(s)
- `calls`: function makes an external call
- `value`: function may receive or forward value (explicitly annotated or inferred)

Example:

```json
"meta": {
  "effects": [
    { "kind": "reads", "path": "balances[from]" },
    { "kind": "writes", "path": "balances[to]" },
    { "kind": "emits", "eventId": "c:Transfer(address,address,uint256)" }
  ]
}
```

### Compatibility note
When emitting Solidity-compatible JSON ABI, Ora effects can be **projected** into legacy fields:
- `reads` only → `stateMutability: "view"`
- no reads/writes/calls/value → `stateMutability: "pure"`
- `value` present → `stateMutability: "payable"`
- otherwise → `stateMutability: "nonpayable"`

This mapping is an **output adapter**, not a core concept in the Ora ABI manifest.

---

## 7. UI/UX metadata

`meta.ui` is intentionally small but expressive:

- `label`, `group`
- `widget` hints: `number`, `text`, `select`, `address`, `bytes`, `json`, ...
- numeric hints: `min`, `max`, `step`, `decimals`, `unit`
- `messageTemplate` for errors
- `dangerLevel`: `info | normal | warning | dangerous`
- per-input overrides: `ui.forms.<inputName>`

### Rationale
- **Portable hints (A) vs “full UI spec” (B):**
  - **A:** works across frontends; doesn’t lock Ora ABI into a UI framework.
  - **B rejected:** becomes a UI framework spec and bloats v0.1.

---

## 8. Wire profiles

A manifest may define multiple wire profiles. Each profile can specify:
- selector derivation rules (if relevant)
- type mapping
- arg/return/error/event encoding

v0.1 standardizes only the *declaration mechanism*:

```json
"wireProfiles": [
  { "id": "evm-default", "kind": "evm", "encoding": "abi-v2" }
]
```

### Rationale
- **Profiles (A) vs single forever-format (B):**
  - **A:** allows compatibility and future evolution without changing the manifest model.
- **B rejected:** binds the ABI model to one encoding and blocks evolution.

---

## 8.1 Solidity ABI mapping (evm-default)

The `evm-default` profile uses Solidity ABI v2 rules for signature spelling and wire encoding.

### Canonical signature spelling

- `function` and `error`: `name(type1,type2,...)`
- `event`: `Name(type1,type2,...)`
- tuple types are spelled as `(t1,t2,...)`
- fixed arrays: `T[N]`, dynamic arrays: `T[]`

### Type mapping (examples)

- `primitive`
  - `u256` → `uint256`
  - `i256` → `int256`
  - `bool` → `bool`
  - `address` → `address`
  - `bytes` → `bytes`
  - `string` → `string`
- `array` → `T[N]`
- `slice` → `T[]`
- `struct`/`tuple` → `(T1,T2,...)` (field order)
- `enum` → `uintN` (via `repr`)
- `refinement` → base type spelling

Selectors are computed as `keccak256(signature)` and are included in `wire.evm-default.selector`.

---

## 9. Size constraints

Soft limits (recommended):
- `meta.doc`: ≤ 2 KB per item
- avoid embedding full ASTs, proofs, or large blobs

If you need more:
- ship `ora.abi.schema.json` (core)
- ship `ora.abi.schema.extras.json` (docs, spans, rich annotations)

---

## 10. Minimal complete example (full)

```json
{
  "schemaVersion": "ora-abi-0.1",
  "contract": {
    "name": "Token",
    "namespace": "com.example",
    "build": { "compiler": "ora", "version": "0.1.0", "commit": "abc123" }
  },
  "wireProfiles": [
    { "id": "evm-default", "kind": "evm", "encoding": "abi-v2" }
  ],
  "types": {
    "t:address": { "typeId": "t:address", "kind": "primitive", "name": "address", "wire": { "evm-default": { "type": "address" } } },
    "t:bool":    { "typeId": "t:bool",    "kind": "primitive", "name": "bool",    "wire": { "evm-default": { "type": "bool" } } },
    "t:u8":      { "typeId": "t:u8",      "kind": "primitive", "name": "u8",      "wire": { "evm-default": { "type": "uint8" } } },
    "t:u256":    { "typeId": "t:u256",    "kind": "primitive", "name": "u256",    "wire": { "evm-default": { "type": "uint256" } } },

    "t:Balance": {
      "typeId": "t:Balance",
      "kind": "refinement",
      "base": "t:u256",
      "predicate": { "op": "<=", "lhs": { "var": "x" }, "rhs": { "const": "1000000" } },
      "meta": { "ui": { "label": "Balance", "min": 0, "max": 1000000, "unit": "wei" } },
      "wire": { "evm-default": { "type": "uint256" } }
    },

    "t:User": {
      "typeId": "t:User",
      "kind": "struct",
      "name": "User",
      "fields": [
        { "name": "owner", "typeId": "t:address" },
        { "name": "balance", "typeId": "t:Balance" }
      ],
      "wire": { "evm-default": { "as": "tuple" } }
    },

    "t:Status": {
      "typeId": "t:Status",
      "kind": "enum",
      "name": "Status",
      "repr": { "typeId": "t:u8" },
      "variants": [
        { "name": "Inactive", "value": 0 },
        { "name": "Active", "value": 1 }
      ],
      "meta": { "ui": { "widget": "select" } },
      "wire": { "evm-default": { "type": "uint8" } }
    }
  },
  "callables": [
    {
      "id": "c:transfer(address,uint256)",
      "kind": "function",
      "name": "transfer",
      "signature": "transfer(address,uint256)",
      "wire": { "evm-default": { "selector": "0xa9059cbb" } },
      "inputs": [
        { "name": "to", "typeId": "t:address" },
        { "name": "amount", "typeId": "t:Balance" }
      ],
      "outputs": [
        { "name": "ok", "typeId": "t:bool" }
      ],
      "meta": {
        "ui": { "group": "Transfers", "dangerLevel": "normal" },
        "effects": [
          { "kind": "writes", "path": "balances[to]" }
        ]
      }
    },
    {
      "id": "c:InsufficientBalance(uint256,uint256)",
      "kind": "error",
      "name": "InsufficientBalance",
      "signature": "InsufficientBalance(uint256,uint256)",
      "wire": { "evm-default": { "selector": "0x..." } },
      "inputs": [
        { "name": "required", "typeId": "t:u256" },
        { "name": "available", "typeId": "t:u256" }
      ],
      "meta": {
        "ui": { "messageTemplate": "Need {required}, have {available}." }
      }
    }
  ]
}
```

---
