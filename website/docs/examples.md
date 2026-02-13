---
sidebar_position: 4
---

# Examples

Feature-focused Ora snippets that mirror the current compiler surface.

> For the full set, see `ora-example/` in the repo and run the validation script.

## Contract + Storage

```ora
contract Counter {
    storage var value: u256;

    pub fn inc(delta: u256) {
        value = value + delta;
    }

    pub fn get() -> u256 {
        return value;
    }
}
```

## Regions (storage ↔ stack)

```ora
storage var s: u32;

fn f() -> void {
    var x: u32 = s;  // storage -> stack read
    s = x;           // stack -> storage write
}
```

## Error Unions

```ora
error InsufficientBalance;
error InvalidAddress;

fn transfer(to: address, amount: u256) -> !u256 | InsufficientBalance | InvalidAddress {
    if (amount == 0) return error.InvalidAddress;
    if (balance < amount) return error.InsufficientBalance;
    balance -= amount;
    return balance;
}
```

## Refinement Types (guards)

```ora
fn withdraw(amount: MinValue<u256, 1>) -> bool {
    // amount is guaranteed > 0 or guarded at runtime
    return true;
}
```

## Switch (expression)

```ora
fn classify(x: u32) -> u32 {
    var out: u32 = switch (x) {
        0 => 0,
        1...9 => 1,
        else => 2,
    };
    return out;
}
```

## Structs and Maps

```ora
struct User {
    balance: u256;
    active: bool;
}

storage var users: map<address, User>;

fn credit(who: address, amount: u256) {
    var user: User = users[who];
    user.balance = user.balance + amount;
    users[who] = user;
}
```

## Enums

```ora
enum Status : u8 { Pending, Active, Closed }

fn is_active(s: Status) -> bool {
    switch (s) {
        Status.Active => return true,
        else => return false,
    }
}
```

## Logs

```ora
log Transfer(sender: address, recipient: address, amount: u256);

fn emit_transfer(from: address, to: address, amount: u256) {
    log Transfer(from, to, amount);
}
```

## Specifications (requires/ensures)

```ora
pub fn transfer(to: address, amount: u256) -> bool
    requires amount > 0
    ensures  amount > 0
{
    // implementation
    return true;
}
```

## Assert vs Assume

```ora
fn check(x: u256) {
    assume(x >= 0);   // verification-only
    assert(x >= 0);   // runtime-visible
}
```

## While + Invariant

```ora
fn sum(n: u256) -> u256 {
    var i: u256 = 0;
    var acc: u256 = 0;

    while (i < n)
        invariant(i <= n)
    {
        acc = acc + i;
        i = i + 1;
    }

    return acc;
}
```

## Validate examples locally

```bash
# Validate all examples
./scripts/validate-examples.sh

# Parse a specific example
./zig-out/bin/ora parse ora-example/smoke.ora
```

