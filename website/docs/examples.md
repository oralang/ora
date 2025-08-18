---
sidebar_position: 4
---

# Examples

Explore real Ora snippets and repository fixtures that reflect the current implementation.

## Simple Storage

```ora
contract SimpleStorage {
    storage var value: u256;

    pub fn set(new_value: u256) {
        value = new_value;
    }

    pub fn get() -> u256 {
        return value;
    }
}
```

## Region Transitions (validated)

```ora
storage var s: u32;

fn f() -> void {
    let x: u32 = s;  // storage -> stack read
    s = x;           // stack -> storage write
}
```

## Error Unions (partial)

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

## Switch (basic expression)

```ora
fn classify(x: u32) -> u32 {
    switch (true) {
        x == 0 => 0,
        x < 10 => 1,
        else   => 2,
    }
}
```

## Switch (advanced)

### 1) Statement form with expression arms

```ora
fn tally(kind: u32) -> void {
    var counter: u32 = 0;
    switch (kind) {
        0 => counter += 1;,
        1 => counter += 2;,
        else => counter = 0;,
    }
}
```

- Note: statement arms using expressions must end with `;`. Commas between arms are optional.

### 2) Statement form with block bodies

```ora
fn update(kind: u32) -> void {
    var counter: u32 = 0;
    switch (kind) {
        0 => {
            counter += 1;
        },
        1 => {
            counter += 2;
        },
        else => {
            counter = 0;
        },
    }
}
```

### 3) Range patterns (inclusive)

```ora
fn grade(score: u32) -> u8 {
    var g: u8 = 0;
    switch (score) {
        0...59   => g = 0;,
        60...69  => g = 1;,
        70...79  => g = 2;,
        80...89  => g = 3;,
        90...100 => g = 4;,
        else     => g = 5;,
    }
    return g;
}
```

### 4) Enum variant patterns (qualified)

```ora
enum Status : u8 { Pending, Active, Suspended, Closed }

fn describe(s: Status) -> u32 {
    switch (s) {
        Status.Pending   => 0,
        Status.Active    => 1,
        Status.Suspended => 2,
        else             => 3,
    }
}
```

## Where to find more

- Semantics fixtures:
  - `tests/fixtures/semantics/valid/`
  - `tests/fixtures/semantics/invalid/`
- Parser fixtures: `tests/fixtures/parser/`
- Reference snippets: `ora-example/` (some files are experimental and may not compile under the current parser)

To inspect examples quickly:

```bash
./zig-out/bin/ora lex path/to/example.ora
./zig-out/bin/ora parse path/to/example.ora
./zig-out/bin/ora -o build ast path/to/example.ora
``` 