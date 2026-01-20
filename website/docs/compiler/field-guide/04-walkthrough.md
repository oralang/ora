# One Story End-to-End (Walkthrough)

This chapter is the “learn by following one value” chapter.

We will keep using `Counter.ora` from Chapter 01 and watch it become:
Tokens → AST → Typed AST → MLIR.

> **Lane A:** Your goal is to understand what the artifacts represent.  
> **Lane B:** Your goal is to understand what each stage *assumes* from the previous stage.

## 4.1 Tokens: the source becomes a stream

The lexer’s job is to turn raw text into a stream of tokens with locations.

What matters for everyone:
- keywords and punctuation are recognized correctly
- locations/spans are correct
- the lexer can recover and keep producing tokens after minor issues

If the token stream is wrong, everything downstream becomes chaos.

## 4.2 AST: structure without meaning (yet)

The parser turns tokens into structure.
The AST answers questions like:
- “is this a function declaration or a variable declaration?”
- “what is the body of this function?”
- “what is the syntactic shape of the expression?”

At this point:
- types are often not fully known
- name binding is not fully resolved
- but spans are stable and structure is correct

## 4.3 Semantics Phase 1: names become addressable

The compiler builds symbol tables:
- “what does `value` refer to?”
- “is `Counter` declared twice?”
- “what is in scope here?”

This stage is often invisible to new contributors, but it’s the backbone for typing.

## 4.4 Typed AST: the compiler commits to types

Type resolution answers:
- what is the type of `value`? (u256 in storage)
- what is the type of `delta`? (u256 parameter)
- what is the type of `value + delta`? (u256)
- is `requires(delta > 0)` valid? (a boolean constraint)

Many important rules live here, including rules around error unions and `try`.

## 4.5 MLIR: the program becomes explicit operations

Lowering translates typed AST nodes into Ora MLIR operations.

Conceptually, `value = value + delta` becomes:
- load storage `value`
- add `delta`
- store back to `value`

When you read MLIR, you’re seeing the compiler’s “working language”.

## 4.6 Mini-exercise: intentionally break it

Change:

```ora
requires(delta > 0)
```

to something invalid (e.g., compare to an address, or use a wrong type).

Now re-run the pipeline:
- where does it fail?
- what diagnostic do you get?
- does the span point to the right source?

This exercise is how you become effective on the compiler quickly.


---

## Next: an intentionally failing walkthrough

If you want to learn the debugging workflow by doing, continue with:
- **[A Walkthrough That Intentionally Fails](04b-failing-walkthrough.md)**
