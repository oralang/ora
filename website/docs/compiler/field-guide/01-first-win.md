# Your First Win (20–30 minutes)

This chapter is about momentum: you’ll compile one tiny program, inspect artifacts, and learn the debugging rhythm.

> **Lane A:** Focus on “what output looks like” and “what to do when it’s wrong.”  
> **Lane B:** Also note the file pointers; you’ll use them later.

## 1. Build the compiler

From the repo root:

```bash
zig build
```

If you already have a build, run it once so you trust your local toolchain.

## 2. Create a tiny program

Create a file `Counter.ora`:

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

Why this example?
- it touches **storage** (load/store)
- it touches **typing** (u256 arithmetic)
- it touches **specification** (`requires`)
- it is small enough to understand in one screen

## 3. Emit tokens (Lexer artifact)

```bash
./zig-out/bin/ora lex Counter.ora
```

### What “good” looks like
You should see:
- keywords (`contract`, `storage`, `pub`, `fn`, `requires`, `return`)
- identifiers (`Counter`, `value`, `inc`, `delta`)
- punctuation (`{ } ( ) : ; + =`)
- integer literal (`0`)
- spans / location data (exact formatting depends on implementation)

### If it’s wrong
If tokenization looks off (e.g. keywords not recognized, weird punctuation grouping), you’re in **lexer land**.

### Where the code lives
- `src/lexer.zig`
- `src/lexer/scanners/`

## 4. Emit AST (Parser artifact)

```bash
./zig-out/bin/ora parse Counter.ora
```

### What “good” looks like
A tree that roughly corresponds to:

- ContractDecl(name=Counter)
  - StorageVarDecl(name=value, type=u256)
  - FunctionDecl(name=inc)
    - Param(delta: u256)
    - ReturnType(u256)
    - Requires(delta > 0)
    - Body
      - Assign(value = value + delta)
      - Return(value)

### If it’s wrong
If parsing fails or the structure is missing pieces, the bug is **parser/grammar**, not typing.

### Where the code lives
- `src/parser.zig` and modules under `src/parser/`
- AST allocation/structures under `src/ast/`

## 5. Emit MLIR (Lowering artifact)

```bash
./zig-out/bin/ora emit-mlir Counter.ora
```

### What “good” looks like
Conceptually:
- a global or storage declaration for `value`
- a function `inc`
- a storage load of `value`
- an add operation
- a storage store of the updated value
- a return

The exact op names vary by current dialect definition, but you should clearly see:
**storage access + arithmetic + return**

### If it’s wrong
If parsing succeeded but MLIR emission fails:
- either **type resolution** failed (typed AST wasn’t produced)
- or **lowering** is missing a mapping for some node

### Where the code lives
- `src/ast/type_resolver/` (typing)
- `src/mlir/` and `src/mlir/lower.zig` (lowering)

## 6. The rhythm: debug by phase

When something breaks, always ask:

1) What is the earliest artifact that looks wrong?  
2) Which stage produced it?  
3) Fix that stage, not the downstream stages.

You’ll use this rhythm constantly. Chapter 06 turns it into a playbook.
