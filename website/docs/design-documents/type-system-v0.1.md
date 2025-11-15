# Ora Type System – Working Design Document (v0.1)

This document outlines the **initial design direction** of the Ora type system, defines core language concepts, incorporates the Solidity comparison table, and clarifies the philosophical lineage (Zig-first with selective Rust-inspired safety).  
It is a **working, evolving design document**, not a final specification.

Ora is designed for deterministic, explicit, auditable smart contract development on the EVM.  
It inherits:

- **From Zig**: explicit memory semantics, comptime execution, no hidden control flow  
- **From Rust**: affine (move-only) values for resource safety  
- **From formal methods**: refinement predicates for correctness invariants  
- **From Solidity**: ABI awareness and EVM transparency  

---

# **1. Type System Goals**

Ora's type system aims for **explicitness, predictability, and safety** while preserving EVM transparency.

### **1. Enforce memory-region correctness**  
Every variable/reference includes an explicit region:
- **storage**
- **memory**
- **calldata**
- **transient**

### **2. Prevent unintended aliasing of storage**
No implicit creation of two mutable references pointing to the same storage slot.

### **3. Support affine (move-only) values**
Explicit move semantics for sensitive resources.

### **4. Support optional refinement predicates**
Compile-time (and sometimes runtime) invariants.

### **5. Support generics via comptime parameters**
Zig-style, monomorphized, explicit.

### **6. Avoid subtyping/inheritance**
Flat, predictable type structure.

### **7. Maintain strict EVM transparency**
Every construct maps clearly to EVM behavior.

---

# **2. Memory Regions**

Ora introduces explicit region-annotated pointers and values:

```
storage     - persistent contract state  
memory      - temporary, local  
calldata    - immutable caller-provided input  
transient   - transaction-scoped scratch space  
```

### Example
```ora
storage var u256 balance;
memory var u256 temp;
transient var u256 counter;
```

### Open Question
Should `memory` be explicit or implicit for locals?

---

# **3. Ownership and Affinity**

Affine types prevent duplication of sensitive resources.

### Affine values:
- permission tokens  
- session handles  
- proof objects  
- commit tickets  

### Non-affine values:
- integers, bools  
- addresses  
- byte arrays  
- structs of non-affine components  

Ora's affine system is far simpler than Rust's and tailored for smart contract correctness.

---

# **4. Refinement Types**

Refinements specify invariants:

```ora
amount: { x: u256 | x <= self.balance }
```

### Semantics:
- Erased at **comptime** when verified
- Lowered into **bytecode** for external entrypoints
- Used for provable invariants (arithmetic, comparisons, logical predicates)

Dependent typing is intentionally excluded.

---

# **5. Traits (Static Interfaces)**

Traits describe behavior at **comptime**.  
They combine ideas from:
- Zig's comptime-based polymorphism  
- Rust's explicit interface-style constraints  

But they are not:
- Solidity interfaces (runtime ABI definition)  
- Rust traits (dynamic trait objects)  

### Example
```ora
trait ERC20 {
    fn totalSupply() -> u256;
    fn balanceOf(owner: address) -> u256;
    fn transfer(to: address, amount: u256) -> bool;
}
```

---

# **6. Traits and Storage**

Traits define behavior only.  
Implementations bind behavior to storage:

```ora
impl ERC20 for Token {
    fn balanceOf(owner: address) -> u256 {
        return self.balances[owner];
    }
}
```

Traits:
- cannot define storage  
- cannot impose layout  
- cannot define region annotations  
- do not exist at runtime  

---

# **7. External Trait Proxies**

```ora
let token = external<ERC20>(addr);
token.transfer(alice, 100);
```

This is purely syntactic sugar:
- compiler generates ABI stubs  
- no runtime conformance guarantee  

A Zig-like compile-time mechanism with Solidity-like ABI output.

---

# **8. Type Grammar**

```
T ::= u8 | u16 | u32 | u64 | u128 | u256
    | bool | address | bytes | bytesN
    | struct { (fᵢ : Tᵢ)* }
    | enum { (vᵢ)* }
    | array[T; n]
    | slice[T]
    | pointer[T @ R]
    | affine T
    | { x : T | φ(x) }
    | T(comptime params)
```

Regions:
```
R ::= storage | memory | calldata | transient
```

---

# **9. Trait Grammar**

```
trait T {
    fn f₁(...) -> T₁;
    ...
}
```

```
impl T for U {
    fn f₁(...) -> T₁ { ... }
}
```

```
external<T>(address) -> TProxy
```

---

# **10. Solidity Comparison Table**

Ora is Zig-first, Rust-influenced, EVM-native.  
Solidity is pragmatic, permissive, and ABI-first.

Below is the updated comparison:

| Aspect | **Ora (v0.3 Design)** | **Solidity (v0.8)** | Key Differences |
|--------|------------------------|----------------------|-----------------|
| **Philosophy** | Zig-like explicitness, comptime reasoning, safety features (affinity, refinements). No inheritance. | Pragmatic, permissive, easy for developers. Heavy reliance on runtime checks. | Ora is stricter and safer; Solidity is simpler and more flexible. |
| **Type System** | Primitive types + affine types + refinements + comptime generics. | Primitive types; no refinements or generics; no affine system. | Ora introduces formal verification concepts. |
| **Memory/Data Locations** | Explicit region annotations everywhere. No implicit aliasing. | storage/memory/calldata but not always explicit. Aliasing allowed. | Ora is stricter and safer. |
| **Aliasing Rules** | No implicit mutable aliasing in storage. | Storage references alias freely; risky. | Major difference: Ora prevents subtle bugs. |
| **Ownership Model** | Affine, move-only types (optional). | No ownership semantics. | Ora enables capability safety. |
| **Refinement Types** | Yes, with comptime verification + optional runtime guards. | No refinements; relies on `require()`. | Ora allows compile-time safety. |
| **Traits / Interfaces** | Static, compile-time traits (Zig-like). No inheritance. | Interfaces define ABI. Supports inheritance. | Ora traits are compile-time only. |
| **Generics** | Zig-style comptime generics. | No generics. | Ora enables abstraction without runtime cost. |
| **Inheritance** | None. | Multiple inheritance. | Large philosophical difference. |
| **Mappings & Collections** | Defined via generics (TBD). | Built-in mappings and arrays. | Ora chooses explicit constructions. |
| **EVM Transparency** | Extremely high; no hidden behavior. | High but with some implicit behaviors (aliasing, data location defaults). | Ora is predictable and explicit by design. |

---

# **11. Open Questions**

### Traits & Associated Types  
Should Ora support associated types (Zig-style type parameters)?

### Refinements in Traits  
Can trait methods include refinement constraints?

### Region-Sensitive Signatures  
Should traits define region-returning functions?

### Default Region  
Should `memory` be implicit for local variables?

### Affine Trait Methods  
Should traits allow affine parameters?

---

# **12. Summary**

Ora establishes:
- explicit memory regions  
- affine semantics  
- refinement predicates  
- Zig-style comptime generics  
- compile-time traits  
- ABI-safe external proxies  
- no inheritance or subtyping  
- strong EVM alignment  
- predictable audit-friendly semantics  

This document serves as the foundation for future RFCs and the formal type system specification.

