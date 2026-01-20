# Debugging Playbook (symptom → stage → fix)

This chapter is the practical “when it breaks, do this” guide.

The core approach is simple:

1) Identify the earliest wrong artifact  
2) Run that phase alone  
3) Fix that phase (not the downstream ones)

---

## 6.1 The 60-second triage

When a user reports “the compiler is broken”, you do this:

1) Ask for the smallest `.ora` program that reproduces the issue  
2) Run:
   - `ora lex`
   - `ora parse`
   - `ora emit-mlir`
3) Note where it first becomes wrong:
   - tokens wrong → lexer
   - AST wrong → parser
   - typed fields wrong / type errors → type resolver / semantics
   - MLIR wrong → lowering
   - verifier complains → MLIR verifier / legality / passes
   - conversion fails → Ora → Sensei-IR (SIR) patterns
   - Z3 fails → encoder/constraints

---

## 6.2 Symptom map

### Symptom: tokens look wrong
**Run:** `ora lex file.ora`  
**Likely area:** lexer scanners, keyword tables  
**Fix style:** adjust scanner logic, add a lexer test.

### Symptom: parse fails or AST shape is wrong
**Run:** `ora parse file.ora`  
**Likely area:** parser rule ordering, error recovery  
**Fix style:** adjust parse functions, add regression test.

### Symptom: parser ok, but type resolution fails
**Signs:**
- “unknown type” where it shouldn’t be
- missing symbol binding
- type mismatch that looks nonsensical

**Likely area:** semantics phase 1 (symbols) or type resolver (typing rules)  
**Fix style:** add a minimal test and inspect the typed AST.

### Symptom: parsing ok, but MLIR emission fails
**Likely causes:**
- a node form has no lowering rule
- TypeInfo is missing and the lowerer refuses to guess

**Likely area:** `src/mlir/lower.zig` and the specific lowering module  
**Fix style:** implement one mapping rule + add a golden MLIR test.

### Symptom: MLIR emits, but verifier complains
**Likely area:** dialect verifier rules  
**Fix style:** decide whether:
- the lowering is producing an illegal op shape, or
- the verifier is missing/too strict for the intended feature

### Symptom: Ora → Sensei-IR (SIR) conversion fails
**Likely area:** missing conversion pattern(s)  
**Fix style:** add a conversion pattern and a test that fails before and passes after.

### Symptom: Z3 verification fails or becomes incomplete
**Likely causes:**
- missing encoding for an op
- incorrect encoding for a constraint
- new op breaks assumptions

**Likely area:** `src/z3/encoder.zig`  
**Fix style:** add encoding + add a solver test.

---

## 6.3 A debugging mindset that scales

When you fix a bug:
- always add a minimal repro
- always add a test
- write the fix so it is “obviously correct” at the artifact level

That’s how the compiler stays stable while it grows.
