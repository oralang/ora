# Debugging a failing transaction

End-to-end walkthrough of the Ora EVM debugger on a contract whose
calldata triggers a custom-error revert. By the end you'll have used
decoded reverts, watchpoints, conditional breakpoints, and `:eval` —
the four features that make this debugger Ora-native rather than a
generic step-debugger.

This tutorial assumes you already finished
[`getting-started.md`](getting-started.md).

## The contract

`vault.ora`:

```ora
contract Vault {
    storage var owner: address = 0;
    storage var balances: map[address, u256] = empty;
    storage var total_supply: u256 = 0;

    error InsufficientBalance(have: u256, need: u256);
    error NotOwner(caller: address);

    pub fn deposit(amount: u256) {
        balances[msg.sender] = balances[msg.sender] + amount;
        total_supply = total_supply + amount;
    }

    pub fn withdraw(amount: u256) -> u256 {
        let have = balances[msg.sender];
        if (have < amount) {
            return error InsufficientBalance(have, amount);
        }
        balances[msg.sender] = have - amount;
        total_supply = total_supply - amount;
        return amount;
    }

    pub fn rescue(to: address, amount: u256) {
        if (msg.sender != owner) {
            return error NotOwner(msg.sender);
        }
        balances[to] = balances[to] + amount;
    }
}
```

The bug: a wallet calls `withdraw(100)` when their balance is only
`50`. Production logs would show a generic
`Transaction reverted: 0x...` — the debugger lets you see the *real*
error.

## Step 1 — launch with calldata that triggers the revert

```sh
ora debug vault.ora --signature 'withdraw(uint256)' --args 100
```

Once the storage map for `msg.sender` resolves to `0` (it's the empty
default), `have < amount` fires and the contract returns
`InsufficientBalance(have: 0, need: 100)`.

Press `c` to run to completion. The status line shows:

```
continue => reverted: InsufficientBalance(have=0, need=100)
```

That's the entire C1b feature — no console hex, no manual ABI
decoding. The debugger reads the loaded `abi.json` and the bytes the
contract left in `frame.output` and pretty-prints them.

## Step 2 — fix the setup, watch the slot

Quit (`q`) and re-launch with calldata that runs `deposit` first.
Imagine you're investigating *why* the balance never increased. Set a
watchpoint on the storage location:

```
:watch balances[msg.sender]
```

(The simpler form `:watch total_supply` watches the bare slot.)

Press `c` to continue. The debugger halts at the SSTORE that updates
the slot:

```
continue => watchpoint #1 hit at line 11
```

Switch to the storage tab (`3`) and inspect — the slot now holds `100`,
the SSTORE just landed. This is C1's storage-watchpoint feature: it
samples slot values after every step and halts on the *step that
produced the change*, not the next statement boundary.

## Step 3 — inspect the predicate that fired the revert

Now we want to see, for every iteration of the test harness, whether
`have < amount` is the actual culprit. Set a *conditional* breakpoint:

```
:break 16 when have < amount
```

Press `c`. The debugger only stops at line 16 when the predicate is
true:

```
continue => InsufficientBalance stmt 17 at line 16, region 3
```

If the predicate evaluator can't resolve a binding (e.g. you typed a
name that's not visible at this stop), the breakpoint *fails open* —
the debugger halts so you can fix the predicate, instead of silently
ignoring it.

## Step 4 — `:eval` for one-off questions

Without setting a breakpoint, you can ask the debugger to compute
arbitrary expressions over visible bindings:

```
:eval (have + 1) * 2
:eval have == 0 && amount > 0
:eval owner != 0
```

The evaluator is side-effect-free — it never writes. Use `:set` for
that.

## Step 5 — combine watchpoints with hit-count breakpoints

Suppose `balances` is updated by ten different SSTOREs in the trace
and you only care about the *fourth* one:

```
:break 11 hit 4
```

Press `c` four times' worth of run-time and the debugger only halts
on hit #4. `info break` shows the running tally:

```
breakpoints: 11 hit 4/4
```

## Step 6 — share the repro

Once you've reduced the failure to a clear sequence, save it:

```
:write-session vault-revert-2026-04-25.session
```

Anyone with the same Ora compiler can `:load-session` that file and
walk through the exact same trace. The session captures the seed,
the breakpoints (including their `when` predicates and `hit` targets),
and the step history.

## What you used

- **C1**: storage watchpoints (`:watch`, `:unwatch`, `:info watch`).
- **C1b**: ABI-decoded reverts (status line) and ABI-decoded events
  (`:print logs`).
- **C2**: side-effect-free expression evaluator (`:eval`).
- **C3**: conditional + hit-count breakpoints (`:break <line> when ...`,
  `:break <line> hit <n>`).
- **A4**: the entire trace is byte-deterministic — block context is
  pinned, so reruns produce the same opcode-by-opcode trace.

## Where to go next

- [`COMMANDS.md`](COMMANDS.md) — the full command reference
  (auto-validated against the dispatcher).
- [`KEYBINDINGS.md`](KEYBINDINGS.md) — keys.
- `:help` (in-app) — same content as `COMMANDS.md`, generated from
  the same table the dispatcher uses.
