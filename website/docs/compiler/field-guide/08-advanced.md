# Advanced Topics (Legality, Passes, Z3/SMT)

This chapter is for when you’re adding real language features and need end-to-end coherence.

> If you’re new: skim this once, then come back when you hit conversion/verification issues.

---

## 8.1 Legality (what “supported” really means)

In Ora, “legal” in practice means:
1) the Ora MLIR is structurally valid and passes dialect verification  
2) the program survives passes/canonicalization without breaking invariants  
3) the IR can be converted to SIR (the backend-supported subset)

A useful mental model:
- **Ora → SIR conversion is a hard boundary.**
- If a feature reaches SIR, it is in a subset the backend knows how to handle.

## 8.2 Passes and canonicalization

Passes exist to:
- normalize IR
- reduce redundant operations
- simplify patterns before conversion
- catch illegal structures early

When debugging pass-related issues:
- inspect IR **before** and **after** the pass pipeline
- keep changes minimal: adjust lowering first if it produces obviously illegal shapes

## 8.3 SMT verification (Z3) as a product feature

Verification is not “academic add-on”.
It’s a practical guarantee layer:
- requires/ensures/invariants become constraints
- counterexamples become actionable bug reports

When Z3 fails:
- treat the model as a gift (it shows real inputs that violate constraints)
- ensure new ops are encoded or intentionally excluded
