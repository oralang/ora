---
slug: ora-debugger
title: "A Real Debugger for Smart Contracts"
authors: [axe]
tags: [debugger, tooling, evm, compiler]
---

No smart contract language has a real source-level debugger. You get transaction tracers. You get console.log. You get Tenderly replays with bytecode-level stepping if you're lucky. But you don't get what every other serious language has had for decades: set a breakpoint on your source line, step through statements, inspect your variables, see what the compiler did.

Ora has one now.

<!-- truncate -->

## What "source-level" actually means here

This is not a bytecode trace viewer with source annotations bolted on.

`ora debug` compiles your contract in a special mode that preserves the full mapping from Ora source lines through SIR (our intermediate representation) down to individual EVM opcodes. Then it deploys the contract in a local EVM, encodes your calldata, and drops you into an interactive TUI at the first statement of the function you asked to debug.

```bash
ora debug contracts/vault.ora \
  --init-signature 'init(u256,u256)' \
  --init-arg 1000 \
  --init-arg 250 \
  --signature 'deposit(u256)' \
  --arg 200
```

You step through Ora statements with `s`, not through PUSH/MLOAD/ADD sequences. When you press `n` to step over a helper call, the debugger runs the entire called function and stops at the next statement in the caller. When you ask "what is `total`?", it reads the actual storage slot that backs that variable and gives you the value.

## The hard part: keeping source maps honest through optimization

A single Ora statement like `let net = amount - fee` becomes roughly 30-50 EVM opcodes after lowering. Every local access is a PUSH+MLOAD or PUSH+MSTORE. A simple `sir.add` with two inputs and one output is 11 EVM bytes. A function call is 24+ bytes just for the call frame, plus 10 bytes per argument transferred through memory.

That expansion is fine — the debugger just stops at statement boundaries and skips the internal opcodes. The real problem is optimization. The Ora pipeline runs constant deduplication, copy propagation, canonicalization, DCE, and inlining. Each pass can destroy, merge, or reorder the ops that carry source locations. If the debugger's source map says "PC 0x2f corresponds to line 26" and the optimizer moved that code somewhere else, you get a lying debugger. That's worse than no debugger.

So `ora debug` compiles in a deliberately conservative mode:

- **Constant deduplication** off — duplicate constants keep their original source locations
- **Copy propagation** off in Sensei — op indices stay aligned between SIR and bytecode
- **Inlining** off — function boundaries preserved for step-over/step-out
- **Canonicalization CSE** off — constants aren't rehomed to unknown locations

The bytecode is larger than a release build. That's the right trade. You debug with `-O0 -g`, you ship with `-O2`. Every serious compiler works this way.

## Two views of the same code

The TUI shows your Ora source on the left and the lowered SIR text on the right, synchronized. As you step through Ora statements, the SIR pane scrolls to the corresponding lowered region. The header tells you exactly what happened:

```
lowered region | ora 26 => sir 42..47 | idx 12..17 | pc 98..134
```

Ora line 26 became SIR lines 42 through 47, which are op indices 12-17, which occupy bytecode PCs 98 through 134. You can see what the compiler did. You can see where your gas is going. You can see whether a `requires` guard survived as a runtime check or got folded away.

This matters for auditors. "Trust the compiler" is not an acceptable answer when the compiler controls what ends up on chain. The SIR view makes the lowering inspectable without reading raw opcodes.

## Bindings, not just bytes

The debugger knows about Ora names, not just EVM stack slots.

At every stop point, the Bindings pane shows what's in scope: function parameters, local variables, storage fields, comptime-folded constants. Each binding carries its runtime classification:

- **Storage fields** — the debugger reads the actual storage slot and shows the current value
- **Memory fields** — read from the reserved debug memory band
- **Comptime-folded constants** — shown with `[folded]` and their compile-time value
- **SSA locals** — visible, but may be opaque if they have no stable runtime location

This is honest. The debugger doesn't invent values for variables that were optimized away. If a local got eliminated by the compiler, it says so. If a constant was folded at compile time, it shows the folded value and marks it as such.

You can also mutate state live:

```
:set total = 1337
:set gas = 750000
:set slot 0x00 = 7
```

Only rooted bindings (storage, memory, transient storage) are writable. The debugger doesn't pretend SSA values are stable slots you can edit.

## Sessions: save your position, share it, come back

One of the things I wanted was the ability to save a debugging session and come back to it. Or hand it to someone else — "here, load this session and step forward twice, you'll see the bug."

```
:write-session artifacts/vault/session.json
:load-session artifacts/vault/session.json
```

Sessions are replayable JSON. They don't dump VM memory. They record the sequence of step commands that reached the current position, plus breakpoints, checkpoints, and UI state. Loading a session replays the steps to reconstruct the exact execution state.

This means sessions are small, deterministic, and human-readable. You can commit them to a repo. You can share the `artifacts/` directory and the recipient gets the bytecode, source map, debug info, SIR text, and the session file. They load it and they're looking at the same execution state you were.

## Comptime and the debugger

Ora evaluates constant expressions at compile time. A line like `comptime const FEE: u256 = 2 + 3 + 5` produces zero bytecode — the compiler folds it to `10` and inlines the literal wherever `FEE` is used.

In the debugger, folded lines have no statement boundary. The debugger skips them during stepping — there's nothing to stop on. But the Bindings pane still shows `FEE = 10 [folded]` so you can see what the compiler computed.

We're planning a comptime trace viewer that shows the evaluation tree — which expressions fed into which results — but that's a later phase. For now, you see the folded value but not the derivation.

## What it looks like

```
+--[ Ora Source ]----------+--[ SIR Text ]---+--[ Bindings ]------+
|  10    let fee = ...   . |  sir.mul ...    |  fee = 50          |
|  11>   let net = ...   . |  sir.sub ...    |  net = 950         |
|  12    total += net    . |  sir.sload ...  |  total = 1000 [s]  |
+--------------------------+-----------------+--------------------+
|                          |                 |--[ Machine ]-------+
|                          |                 |  PC: 0x2f  GAS: .. |
+--[ State: Stack  Memory  Storage  TStore  Calldata ]------------+
|  [0] 0x3e8                                                      |
|  [1] 0x32                                                       |
+--[ :step => stopped at line 11 ]--------------------------------+
```

Five panes. Ora source with the current line highlighted. SIR text synchronized to the current statement. Bindings with live values. Machine state (PC, opcode, gas, call depth). Tabbed EVM state (stack, memory, storage, transient storage, calldata). Vim-style command console at the bottom.

`s` steps in. `n` steps over. `o` steps out. `c` continues to breakpoint. `p` steps backward (replay-based). `x` steps one raw opcode when you need it. `j`/`k` scroll the source, `J`/`K` scroll the SIR, `=` re-syncs them.

## Why this matters

The Ora thesis is: comptime over runtime, formal verification in the workflow, explicit semantics. The debugger is how we make "explicit semantics" real for developers, not just a compiler design principle.

You can see your source code. You can see what the compiler lowered it to. You can see the EVM state underneath. You can see which values were folded at compile time and which are runtime. You can see whether a `requires` guard produced a runtime check or was proven away.

That's what "no hidden behavior" means when you're actually sitting at a terminal trying to understand why your contract does what it does.

---

*The debugger ships on the `debugger` branch. Full docs at [Interactive Debugger](/docs/debugger). Example contracts in `ora-example/debugger/`. Feedback welcome — this is v1 and we're iterating.*
