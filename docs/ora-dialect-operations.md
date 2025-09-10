# Ora MLIR Dialect Operations Specification

This document provides a comprehensive mapping from the Ora language grammar to the MLIR operations that need to be implemented in the Ora dialect.

## 📋 **Current Status**

### ✅ **Currently Implemented (4 operations)**
- `ora.contract` - Contract declarations
- `ora.global` - Storage variable declarations  
- `ora.sload` - Load from storage
- `ora.sstore` - Store to storage

### 🚧 **Operations Used But Not Defined (23 operations)**
These are currently created as unregistered operations in the lowering code:
- `ora.string.constant`, `ora.address.constant`, `ora.hex.constant`, `ora.binary.constant`
- `ora.power`, `ora.sequence`, `ora.struct_field_store`, `ora.indexed_store`
- `ora.comptime`, `ora.old`, `ora.quantified`, `ora.yield`
- `ora.try`, `ora.move`, `ora.call`, `ora.field_access`, `ora.index_access`
- `ora.cast`, `ora.requires`, `ora.ensures`, `ora.invariant`
- `ora.log`, `ora.error`, `ora.switch`, `ora.range`

## 🎯 **Complete Grammar-to-MLIR Mapping**

Based on the Ora grammar (`GRAMMAR.bnf`), here are all the operations we need to implement:

---

## **1. Contract & Module Operations**

### `ora.contract` ✅ **(Implemented)**
```mlir
ora.contract @ContractName {
  // Contract body with storage, functions, etc.
}
```
**Grammar**: `contract_declaration ::= "contract" identifier "{" contract_member* "}"`

### `ora.import` 🚧 **(Needed)**
```mlir
ora.import "module_path"
ora.import_binding %result = ora.import "module_path"
```
**Grammar**: `import_declaration`

---

## **2. Storage & Memory Operations**

### `ora.global` ✅ **(Implemented)**
```mlir
ora.global "variable_name" : type = initial_value
```

### `ora.sload` ✅ **(Implemented)**
```mlir
%value = ora.sload "variable_name" : () -> type
```

### `ora.sstore` ✅ **(Implemented)**
```mlir
ora.sstore %value, "variable_name" : (type) -> ()
```

### `ora.memory.alloc` 🚧 **(Needed)**
```mlir
%ptr = ora.memory.alloc : () -> !ora.ptr<type>
```
**Grammar**: `memory_region ::= "storage" | "memory" | "tstore"`

### `ora.memory.load` 🚧 **(Needed)**
```mlir
%value = ora.memory.load %ptr : (!ora.ptr<type>) -> type
```

### `ora.memory.store` 🚧 **(Needed)**
```mlir
ora.memory.store %value, %ptr : (type, !ora.ptr<type>) -> ()
```

### `ora.tstore.load` 🚧 **(Needed)**
```mlir
%value = ora.tstore.load "key" : () -> type
```

### `ora.tstore.store` 🚧 **(Needed)**
```mlir
ora.tstore.store %value, "key" : (type) -> ()
```

---

## **3. Variable & Declaration Operations**

### `ora.var.decl` 🚧 **(Needed)**
```mlir
%var = ora.var.decl : () -> !ora.var<type>
ora.var.init %var, %value : (!ora.var<type>, type) -> ()
```
**Grammar**: `variable_declaration`

### `ora.const.decl` 🚧 **(Needed)**
```mlir
%const = ora.const.decl %value : (type) -> !ora.const<type>
```

### `ora.let.decl` 🚧 **(Needed)**
```mlir
%let = ora.let.decl %value : (type) -> !ora.immutable<type>
```

---

## **4. Function Operations**

### `ora.func` 🚧 **(Needed)**
```mlir
ora.func @function_name(%args...) -> (return_types...) 
  attributes {visibility, inline, etc.} {
  // Function body
}
```
**Grammar**: `function_declaration`

### `ora.call` 🚧 **(Partially Used)**
```mlir
%results = ora.call @function_name(%args...) : (arg_types...) -> (return_types...)
```

### `ora.return` 🚧 **(Needed)**
```mlir
ora.return %values... : (types...)
```
**Grammar**: `return_statement`

---

## **5. Control Flow Operations**

### `ora.if` 🚧 **(Needed)**
```mlir
%result = ora.if %condition : (i1) -> type {
  // then block
  ora.yield %then_value : (type) -> ()
} else {
  // else block  
  ora.yield %else_value : (type) -> ()
}
```
**Grammar**: `if_statement`

### `ora.while` 🚧 **(Needed)**
```mlir
ora.while %condition : (i1) {
  // loop body
  ora.yield %new_condition : (i1) -> ()
}
```
**Grammar**: `while_statement`

### `ora.for` 🚧 **(Needed)**
```mlir
ora.for %iterable : (iterable_type) -> (key_type, value_type) {
^bb0(%key: key_type, %value: value_type):
  // loop body
}
```
**Grammar**: `for_statement`

### `ora.switch` 🚧 **(Partially Used)**
```mlir
%result = ora.switch %value : (type) -> result_type {
  case %pattern1 -> {
    ora.yield %result1 : (result_type) -> ()
  }
  case %pattern2 -> {
    ora.yield %result2 : (result_type) -> ()
  }
  default -> {
    ora.yield %default_result : (result_type) -> ()
  }
}
```
**Grammar**: `switch_statement`, `switch_expression`

### `ora.break` 🚧 **(Needed)**
```mlir
ora.break label %optional_value : (type) -> ()
```
**Grammar**: `break_statement`

### `ora.continue` 🚧 **(Needed)**
```mlir
ora.continue label
```
**Grammar**: `continue_statement`

---

## **6. Expression Operations**

### **Arithmetic Operations**
```mlir
%result = ora.add %lhs, %rhs : (type, type) -> type
%result = ora.sub %lhs, %rhs : (type, type) -> type
%result = ora.mul %lhs, %rhs : (type, type) -> type
%result = ora.div %lhs, %rhs : (type, type) -> type
%result = ora.mod %lhs, %rhs : (type, type) -> type
%result = ora.power %base, %exp : (type, type) -> type  // 🚧 Partially Used
```

### **Comparison Operations**
```mlir
%result = ora.eq %lhs, %rhs : (type, type) -> i1
%result = ora.ne %lhs, %rhs : (type, type) -> i1
%result = ora.lt %lhs, %rhs : (type, type) -> i1
%result = ora.le %lhs, %rhs : (type, type) -> i1
%result = ora.gt %lhs, %rhs : (type, type) -> i1
%result = ora.ge %lhs, %rhs : (type, type) -> i1
```

### **Logical Operations**
```mlir
%result = ora.and %lhs, %rhs : (i1, i1) -> i1
%result = ora.or %lhs, %rhs : (i1, i1) -> i1
%result = ora.not %operand : (i1) -> i1
```

### **Bitwise Operations**
```mlir
%result = ora.bitand %lhs, %rhs : (type, type) -> type
%result = ora.bitor %lhs, %rhs : (type, type) -> type
%result = ora.bitxor %lhs, %rhs : (type, type) -> type
%result = ora.bitnot %operand : (type) -> type
%result = ora.shl %lhs, %rhs : (type, type) -> type
%result = ora.shr %lhs, %rhs : (type, type) -> type
```

---

## **7. Data Structure Operations**

### **Struct Operations**
```mlir
%struct = ora.struct.create : () -> !ora.struct<{field1: type1, field2: type2}>
%value = ora.struct.get %struct["field_name"] : (!ora.struct<...>) -> type
ora.struct.set %struct["field_name"], %value : (!ora.struct<...>, type) -> ()
```
**Grammar**: `struct_declaration`, `anonymous_struct_literal`

### **Array Operations**
```mlir
%array = ora.array.create %size : (index) -> !ora.array<type>
%value = ora.array.get %array[%index] : (!ora.array<type>, index) -> type
ora.array.set %array[%index], %value : (!ora.array<type>, index, type) -> ()
```
**Grammar**: `array_type`, `array_literal`

### **Slice Operations**
```mlir
%slice = ora.slice.create : () -> !ora.slice<type>
%value = ora.slice.get %slice[%index] : (!ora.slice<type>, index) -> type
ora.slice.append %slice, %value : (!ora.slice<type>, type) -> ()
```
**Grammar**: `slice_type`

### **Map Operations**
```mlir
%map = ora.map.create : () -> !ora.map<key_type, value_type>
%value = ora.map.get %map[%key] : (!ora.map<key_type, value_type>, key_type) -> value_type
ora.map.set %map[%key], %value : (!ora.map<key_type, value_type>, key_type, value_type) -> ()
```
**Grammar**: `map_type`

### **DoubleMap Operations**
```mlir
%dmap = ora.doublemap.create : () -> !ora.doublemap<key1_type, key2_type, value_type>
%value = ora.doublemap.get %dmap[%key1, %key2] : (!ora.doublemap<...>, key1_type, key2_type) -> value_type
ora.doublemap.set %dmap[%key1, %key2], %value : (!ora.doublemap<...>, key1_type, key2_type, value_type) -> ()
```
**Grammar**: `doublemap_type`

---

## **8. Type Operations**

### `ora.cast` 🚧 **(Partially Used)**
```mlir
%result = ora.cast %value : (source_type) -> target_type
```
**Grammar**: `cast_expression`

### **Type Definitions**
```mlir
ora.type.alias @TypeName = type
ora.type.struct @StructName {field1: type1, field2: type2}
ora.type.enum @EnumName : base_type {variant1 = value1, variant2 = value2}
```

---

## **9. Error Handling Operations**

### `ora.error.declare` 🚧 **(Needed)**
```mlir
ora.error.declare @ErrorName(%params...) : (param_types...)
```
**Grammar**: `error_declaration`

### `ora.try` 🚧 **(Partially Used)**
```mlir
%result = ora.try {
  %value = // operation that might fail
  ora.yield %value : (type) -> ()
} catch {
^bb0(%error: !ora.error):
  %default = // handle error
  ora.yield %default : (type) -> ()
}
```
**Grammar**: `try_statement`

### `ora.error` 🚧 **(Partially Used)**
```mlir
%error = ora.error @ErrorName(%args...) : (arg_types...) -> !ora.error
```
**Grammar**: `error_expression`

---

## **10. Formal Verification Operations**

### `ora.requires` 🚧 **(Partially Used)**
```mlir
ora.requires %condition : (i1) -> ()
```
**Grammar**: `requires_clause`

### `ora.ensures` 🚧 **(Partially Used)**
```mlir
ora.ensures %condition : (i1) -> ()
```
**Grammar**: `ensures_clause`

### `ora.invariant` 🚧 **(Partially Used)**
```mlir
ora.invariant %condition : (i1) -> ()
```

### `ora.old` 🚧 **(Partially Used)**
```mlir
%old_value = ora.old %expression : (type) -> type
```
**Grammar**: `old_expression`

### `ora.quantified` 🚧 **(Partially Used)**
```mlir
%result = ora.quantified forall %var : type in %domain where %condition => %expression
%result = ora.quantified exists %var : type in %domain where %condition => %expression
```
**Grammar**: `quantified_expression`

---

## **11. Event & Logging Operations**

### `ora.log.declare` 🚧 **(Needed)**
```mlir
ora.log.declare @EventName(%params...) : (param_types...)
```
**Grammar**: `log_declaration`

### `ora.log.emit` 🚧 **(Partially Used)**
```mlir
ora.log.emit @EventName(%args...) : (arg_types...) -> ()
```
**Grammar**: `log_statement`

---

## **12. Financial Operations**

### `ora.move` 🚧 **(Partially Used)**
```mlir
ora.move %amount from %source to %destination : (amount_type, balance_type, balance_type) -> ()
```
**Grammar**: `move_statement`

### `ora.lock` 🚧 **(Needed)**
```mlir
ora.lock %resource : (resource_type) -> ()
```
**Grammar**: `lock_statement`

### `ora.unlock` 🚧 **(Needed)**
```mlir
ora.unlock %resource : (resource_type) -> ()
```
**Grammar**: `unlock_statement`

---

## **13. Constant & Literal Operations**

### **Literal Constants** 🚧 **(Some Partially Used)**
```mlir
%int = ora.constant.int 42 : i256
%bool = ora.constant.bool true : i1
%string = ora.constant.string "hello" : !ora.string
%address = ora.constant.address 0x1234... : !ora.address
%hex = ora.constant.hex 0x1a2b : !ora.hex
%binary = ora.constant.binary 0b1010 : !ora.binary
%char = ora.constant.char 'a' : !ora.char
```

---

## **14. Utility Operations**

### `ora.comptime` 🚧 **(Partially Used)**
```mlir
%result = ora.comptime {
  // compile-time computation
  ora.yield %computed_value : (type) -> ()
}
```
**Grammar**: `comptime_expression`

### `ora.yield` 🚧 **(Partially Used)**
```mlir
ora.yield %values... : (types...) -> ()
```

---

## 📊 **Summary Statistics**

- **Total Operations Needed**: ~80-90 operations
- **Currently Implemented**: 4 operations (5%)
- **Partially Used (Unregistered)**: 23 operations (26%)
- **Still Needed**: ~55-65 operations (69%)

## 🎯 **Recommended Implementation Priority**

### **Phase 1: Core Language (High Priority)**
1. Control flow: `ora.if`, `ora.while`, `ora.for`, `ora.switch`
2. Functions: `ora.func`, `ora.call`, `ora.return`
3. Variables: `ora.var.decl`, `ora.const.decl`, `ora.let.decl`
4. Basic expressions: arithmetic, comparison, logical operations

### **Phase 2: Data Structures (Medium Priority)**
1. Structs: `ora.struct.*` operations
2. Arrays: `ora.array.*` operations  
3. Maps: `ora.map.*`, `ora.doublemap.*` operations
4. Type system: `ora.cast`, type definitions

### **Phase 3: Advanced Features (Lower Priority)**
1. Error handling: `ora.try`, `ora.error.*` operations
2. Events: `ora.log.*` operations
3. Financial: `ora.move`, `ora.lock`, `ora.unlock`
4. Formal verification: `ora.requires`, `ora.ensures`, etc.

This document provides a complete roadmap for implementing the full Ora MLIR dialect based on the language grammar.
