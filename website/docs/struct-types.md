---
sidebar_position: 3.5
title: Struct Types
description: Custom data structures with field access, storage packing, and region-aware layout.
---

# Struct Types

Structs group related fields into a named type. The compiler handles storage slot packing and region-aware layout.

## Declaration

```ora
struct Point {
    x: u32;
    y: u32;
}

struct User {
    name: string;
    age: u8;
    balance: u256;
    location: Point;
}
```

## Storage Structs

Storage structs are packed into 32-byte EVM storage slots. Small fields that fit within a single slot are packed together automatically.

```ora
contract TokenContract {
    storage var user_data: User;
    storage var origin: Point;

    pub fn init() {
        user_data = User {
            name: "Alice",
            age: 25,
            balance: 1000000,
            location: Point { x: 100, y: 200 }
        };
    }
}
```

### Packing Strategy

Fields are grouped by size to minimize slot usage:

```ora
struct OptimizedData {
    // Small fields packed together (1 storage slot)
    active: bool;     // 1 byte
    level: u8;        // 1 byte
    flags: u16;       // 2 bytes
    counter: u32;     // 4 bytes

    // Large fields get dedicated slots
    balance: u256;    // 32 bytes (1 slot)
    metadata: string; // Dynamic size
}
```

## Nested Structs

Structs can contain other struct types:

```ora
struct Address {
    street: string;
    city: string;
    postal_code: u32;
}

struct Employee {
    id: u256;
    name: string;
    home_address: Address;
    work_address: Address;
}

contract Company {
    storage var employees: map<u256, Employee>;

    pub fn add_employee(id: u256, name: string) {
        employees[id] = Employee {
            id: id,
            name: name,
            home_address: Address {
                street: "",
                city: "",
                postal_code: 0
            },
            work_address: Address {
                street: "Main St",
                city: "Tech City",
                postal_code: 12345
            }
        };
    }
}
```

## Field Access

Dot notation for reading and writing fields:

```ora
pub fn update_user_balance(new_balance: u256) {
    user_data.balance = new_balance;

    // Nested field access
    user_data.location.x = user_data.location.x + 10;

    // Conditional logic with struct fields
    if (user_data.age >= 18) {
        user_data.balance = user_data.balance * 2;
    }
}

pub fn get_user_location() -> (u32, u32) {
    return (user_data.location.x, user_data.location.y);
}
```

## Structs with Verification

Struct fields can appear in specification clauses:

```ora
struct BankAccount {
    balance: u256;
    frozen: bool;
    owner: address;
}

contract Bank {
    storage var account: BankAccount;

    pub fn withdraw(amount: u256)
        requires !account.frozen && account.balance >= amount
        ensures account.balance == old(account.balance) - amount
    {
        account.balance = account.balance - amount;
    }

    pub fn freeze_account()
        requires account.owner == std.msg.sender()
        ensures account.frozen == true
    {
        account.frozen = true;
    }
}
```

## Relationship to Bitfields

- **Structs** pack at byte granularity across one or more storage slots.
- **Bitfields** pack at bit granularity within a single word.

A struct field can itself be a bitfield for maximum density:

```ora
packed struct Position {
    owner:    address,         // 20 bytes
    flags:    PositionFlags,   // 1 byte (bitfield over u8)
    token_id: u32,             // 4 bytes
}
```

See [Bitfield Types](./bitfield-types) for the bitfield system.
