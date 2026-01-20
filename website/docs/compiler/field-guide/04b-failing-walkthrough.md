# A Walkthrough That Intentionally Fails (Debugging by Phase)

This chapter is a guided “failure tour”. You will take a small program, break it on purpose in **three different ways**, and learn how to locate the broken phase using the **artifact ladder**.

> **Lane A:** You’ll learn how to report bugs with minimal repros and phase-local evidence.  
> **Lane B:** You’ll learn where fixes usually live.

## The ritual (we’ll repeat it)

For each failure, you will run:

```bash
./zig-out/bin/ora lex  <file.ora>
./zig-out/bin/ora parse <file.ora>
./zig-out/bin/ora emit-mlir <file.ora>
```

And answer one question:

> **Where is the first artifact that looks wrong?**

That answer tells you *which phase to fix*.

---

## Failure 1 — Parser failure (structure breaks)

### Step 1: Start from a working file

Create `Fail01_Parse.ora`:

```ora
contract Counter {
    storage var value: u256;

    pub fn inc(delta: u256) -> u256
        requires(delta > 0)
    {
        value = value + delta;
        return value;
    }
}
```

Verify it works:

```bash
./zig-out/bin/ora parse Fail01_Parse.ora
```

### Step 2: Break the structure

Delete one closing brace `}` at the end.

Now re-run:

```bash
./zig-out/bin/ora lex   Fail01_Parse.ora
./zig-out/bin/ora parse Fail01_Parse.ora
```

### What you should observe
- **Tokens** still look fine (lexer does not care about nesting).
- **Parser** fails or reports an “expected `}`” style error.
- MLIR won’t be produced because the AST isn’t structurally valid.

### Conclusion (phase-local)
✅ **Parser/grammar/recovery issue**.

### Where fixes usually live
- parser modules (`src/parser/*`)
- recovery logic (how the parser reports and continues)

### A good contributor PR from this
- Add a regression test ensuring the diagnostic points at the right span.
- Improve the error message to mention what token was expected.

---

## Failure 2 — Type resolution failure (structure is fine, meaning is wrong)

Here we create a program that parses cleanly, but violates a typing rule. Most common onboarding confusion: “It parses, why doesn’t it compile?”

### Step 1: Create a file that uses an error union

Create `Fail02_Types.ora`:

```ora
contract SafeMath {
    pub fn safeDiv(a: u256, b: u256) -> !u256
        requires(b > 0)
    {
        // Pretend this can return an error (exact mechanism not important here)
        return a / b;
    }

    pub fn use(a: u256, b: u256) -> u256
        requires(b > 0)
    {
        // ❌ Intentionally wrong: using an error union value as if it were a u256
        let x: u256 = safeDiv(a, b);
        return x;
    }
}
```

### Step 2: Run the phases

```bash
./zig-out/bin/ora lex       Fail02_Types.ora
./zig-out/bin/ora parse     Fail02_Types.ora
./zig-out/bin/ora emit-mlir Fail02_Types.ora
```

### What you should observe
- **Lexer**: fine.
- **Parser**: fine (AST shape exists).
- **Type resolution**: fails (or MLIR emission fails with a typing diagnostic).

### Why it fails (in plain English)
`safeDiv` returns `!u256` (an error union), but you’re trying to treat it as `u256`.
In Ora, error unions must be **unwrapped** (typically with `try`) before you can use the success value.

### Conclusion (phase-local)
✅ **Typing rule issue (type resolver), not a parser issue.**

### Where fixes usually live
- type resolver rules (`src/ast/type_resolver/*`)
- semantic validation that checks “error union usage”

### A good contributor PR from this
- Add a minimal regression test that expects the specific diagnostic.
- Improve the diagnostic message to say: “This expression is an error union; use `try` to unwrap it.”

> Tip: When reporting this bug, include:
> - the tiny repro file
> - the output of `ora parse`
> - the exact typing diagnostic (with span)

---

## Failure 3 — Lowering/IR mismatch (types are fine, but IR coverage is missing)

This failure mode happens when:
- the language supports a construct in syntax + typing,
- but lowering doesn’t yet know how to emit MLIR for it.

You’ll practice identifying this class of bug even if you don’t fix it yet.

### Step 1: Create a “new-ish” construct

Create `Fail03_Lowering.ora`:

```ora
contract C {
    pub fn f(x: u256) -> u256 {
        // Use a construct that is likely newer or less commonly used in the pipeline.
        // Replace this with a real feature you are adding or testing.
        // Example placeholder: a match/switch-like expression or a special builtin.
        return x;
    }
}
```

Now, **change the body** to include the specific feature you’re working on (or the one that just landed in syntax).

### Step 2: Run the phases

```bash
./zig-out/bin/ora lex       Fail03_Lowering.ora
./zig-out/bin/ora parse     Fail03_Lowering.ora
./zig-out/bin/ora emit-mlir Fail03_Lowering.ora
```

### What you should observe (when lowering is missing)
- Lexer: fine
- Parser: fine
- Type resolution: fine
- **MLIR emission fails** with a message like:
  - “unhandled AST node in lowering”
  - “lowering not implemented for X”
  - or it emits an obviously incomplete/empty region for the construct

### Conclusion (phase-local)
✅ **Lowering coverage issue**.

### Where fixes usually live
- `src/mlir/lower.zig`
- specific lowerer modules that map AST nodes to ops
- `src/mlir/types.zig` if types aren’t mapped

### A good contributor PR from this
- Add a “lowering must fail loudly” test until the feature is implemented.
- Add a lowering pattern and a golden MLIR test for the emitted op shape.

---

## The meta-skill you just learned

When someone says:

> “The compiler doesn’t work”

You can respond with:

1) “Show me the smallest repro.”  
2) “Show me where the first artifact looks wrong: tokens, AST, typed AST, or MLIR.”  
3) “Now we know which part of the compiler to touch.”

That is the difference between *being blocked* and *shipping compiler work*.
