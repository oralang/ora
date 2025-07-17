# Language Grammar

This document provides the complete formal grammar specification for the Ora smart contract language.

## Grammar Notation

Ora uses both BNF (Backus-Naur Form) and EBNF (Extended Backus-Naur Form) notation for precise syntax specification.

### BNF Grammar

```bnf
# Ora Language Grammar (BNF)
# Version: 0.1.0
# Description: Formal grammar specification for the Ora smart contract language

# ==========================================
# TOP-LEVEL PROGRAM STRUCTURE
# ==========================================

program ::= top_level_declaration*

top_level_declaration ::=
    | contract_declaration
    | function_declaration
    | variable_declaration
    | struct_declaration
    | enum_declaration
    | log_declaration
    | import_declaration

# ==========================================
# IMPORT DECLARATIONS
# ==========================================

import_declaration ::= "@" "import" "(" string_literal ")"

# ==========================================
# CONTRACT DECLARATIONS
# ==========================================

contract_declaration ::= "contract" identifier "{" contract_member* "}"

contract_member ::=
    | variable_declaration
    | function_declaration
    | log_declaration
    | struct_declaration
    | enum_declaration

# ==========================================
# FUNCTION DECLARATIONS
# ==========================================

function_declaration ::= visibility? "fn" identifier "(" parameter_list? ")" return_type? requires_clause* ensures_clause* block

visibility ::= "pub"

parameter_list ::= parameter ("," parameter)*

parameter ::= identifier ":" type

return_type ::= "->" type

requires_clause ::= "requires" "(" expression ")" ";"?

ensures_clause ::= "ensures" "(" expression ")" ";"?

# ==========================================
# VARIABLE DECLARATIONS
# ==========================================

variable_declaration ::= memory_region? variable_kind identifier ":" type ("=" expression)? ";"

memory_region ::= "storage" | "memory" | "tstore"

variable_kind ::= "var" | "let" | "const" | "immutable"

# ==========================================
# STRUCT DECLARATIONS
# ==========================================

struct_declaration ::= "struct" identifier "{" struct_member* "}"

struct_member ::= identifier ":" type ";"

# ==========================================
# ENUM DECLARATIONS
# ==========================================

enum_declaration ::= "enum" identifier "{" enum_member_list "}"

enum_member_list ::= enum_member ("," enum_member)*

enum_member ::= identifier ("=" expression)?

# ==========================================
# LOG DECLARATIONS (EVENTS)
# ==========================================

log_declaration ::= "log" identifier "(" parameter_list? ")" ";"

# ==========================================
# TYPE SYSTEM
# ==========================================

type ::=
    | primitive_type
    | map_type
    | doublemap_type
    | array_type
    | optional_type
    | error_union_type
    | identifier

primitive_type ::=
    | "u8" | "u16" | "u32" | "u64" | "u128" | "u256"
    | "i8" | "i16" | "i32" | "i64" | "i128" | "i256"
    | "bool"
    | "address"
    | "string"
    | "bytes"

map_type ::= "map" "[" type "," type "]"

doublemap_type ::= "doublemap" "[" type "," type "," type "]"

array_type ::= "[" type (";" expression)? "]"

# Array types in Ora:
# [T; N] - Fixed-size array (N elements of type T)
# [T]    - Dynamic-size array (variable number of elements)

optional_type ::= "?" type

error_union_type ::= type "|" type

# ==========================================
# STATEMENTS
# ==========================================

statement ::=
    | variable_declaration
    | assignment_statement
    | compound_assignment_statement
    | transfer_statement
    | expression_statement
    | if_statement
    | while_statement
    | return_statement
    | break_statement
    | continue_statement
    | log_statement
    | lock_statement
    | unlock_statement
    | try_statement
    | block

assignment_statement ::= expression "=" expression ";"

compound_assignment_statement ::= expression compound_operator expression ";"

compound_operator ::= "+=" | "-=" | "*="

transfer_statement ::= identifier "from" expression "->" expression ":" expression ";"

expression_statement ::= expression ";"

if_statement ::= "if" "(" expression ")" statement ("else" statement)?

while_statement ::= "while" "(" expression ")" statement

return_statement ::= "return" expression? ";"

break_statement ::= "break" ";"

continue_statement ::= "continue" ";"

log_statement ::= "log" identifier "(" expression_list? ")" ";"

lock_statement ::= "@" "lock" "(" expression ")" ";"

unlock_statement ::= "@" "unlock" "(" expression ")" ";"

try_statement ::= "try" expression ("catch" identifier block)?

block ::= "{" statement* "}"

# ==========================================
# EXPRESSIONS
# ==========================================

expression ::= assignment_expression

assignment_expression ::= logical_or_expression assignment_operator*

assignment_operator ::= "=" | "+=" | "-=" | "*="

logical_or_expression ::= logical_and_expression ("|" logical_and_expression)*

logical_and_expression ::= equality_expression ("&" equality_expression)*

equality_expression ::= relational_expression (("==" | "!=") relational_expression)*

relational_expression ::= additive_expression (("<" | "<=" | ">" | ">=") additive_expression)*

additive_expression ::= multiplicative_expression (("+" | "-") multiplicative_expression)*

multiplicative_expression ::= unary_expression (("*" | "/" | "%") unary_expression)*

unary_expression ::= ("!" | "-" | "+")* postfix_expression

postfix_expression ::= primary_expression postfix_operator*

postfix_operator ::=
    | "." identifier
    | "[" expression "]"
    | "[" expression "," expression "]"
    | "(" expression_list? ")"

primary_expression ::=
    | literal
    | identifier
    | "(" expression ")"
    | old_expression
    | comptime_expression
    | cast_expression
    | error_expression

old_expression ::= "old" "(" expression ")"

comptime_expression ::= "comptime" expression

cast_expression ::= expression "as" type

error_expression ::= identifier "!" identifier

expression_list ::= expression ("," expression)*

# ==========================================
# LITERALS
# ==========================================

literal ::=
    | integer_literal
    | string_literal
    | boolean_literal
    | address_literal
    | hex_literal

integer_literal ::= [0-9]+

string_literal ::= "\"" [^"]* "\""

boolean_literal ::= "true" | "false"

address_literal ::= "0x" [0-9a-fA-F]{40}

hex_literal ::= "0x" [0-9a-fA-F]+

# ==========================================
# IDENTIFIERS AND KEYWORDS
# ==========================================

identifier ::= [a-zA-Z_][a-zA-Z0-9_]*

# Reserved keywords (cannot be used as identifiers)
keyword ::=
    | "contract" | "pub" | "fn" | "let" | "var" | "const" | "immutable"
    | "storage" | "memory" | "tstore" | "init" | "log"
    | "if" | "else" | "while" | "break" | "continue" | "return"
    | "requires" | "ensures" | "invariant" | "old" | "comptime"
    | "as" | "import" | "struct" | "enum" | "true" | "false"
    | "error" | "try" | "catch" | "from"
    | "u8" | "u16" | "u32" | "u64" | "u128" | "u256"
    | "i8" | "i16" | "i32" | "i64" | "i128" | "i256"
    | "bool" | "address" | "string" | "bytes"
    | "map" | "doublemap"

# ==========================================
# OPERATORS AND PUNCTUATION
# ==========================================

operator ::=
    | "+" | "-" | "*" | "/" | "%"
    | "=" | "==" | "!=" | "<" | "<=" | ">" | ">="
    | "!" | "&" | "|" | "^" | "<<" | ">>"
    | "+=" | "-=" | "*="
    | "->"

punctuation ::=
    | "(" | ")" | "{" | "}" | "[" | "]"
    | "," | ";" | ":" | "." | "@"
```

### EBNF Grammar

```ebnf
# Ora Language Grammar (EBNF)
# Version: 0.1.0
# Description: Extended BNF grammar specification for the Ora smart contract language

# ==========================================
# PROGRAM STRUCTURE
# ==========================================

Program = { TopLevelDeclaration } ;

TopLevelDeclaration = 
    ContractDeclaration
  | FunctionDeclaration
  | VariableDeclaration
  | StructDeclaration
  | EnumDeclaration
  | LogDeclaration
  | ImportDeclaration ;

# ==========================================
# DECLARATIONS
# ==========================================

ImportDeclaration = "@" "import" "(" StringLiteral ")" ;

ContractDeclaration = "contract" Identifier "{" { ContractMember } "}" ;

ContractMember = 
    VariableDeclaration
  | FunctionDeclaration
  | LogDeclaration
  | StructDeclaration
  | EnumDeclaration ;

FunctionDeclaration = 
    [ "pub" ] "fn" Identifier "(" [ ParameterList ] ")" [ ReturnType ]
    { RequiresClause } { EnsuresClause } Block ;

ParameterList = Parameter { "," Parameter } ;

Parameter = Identifier ":" Type ;

ReturnType = "->" Type ;

RequiresClause = "requires" "(" Expression ")" [ ";" ] ;

EnsuresClause = "ensures" "(" Expression ")" [ ";" ] ;

VariableDeclaration = 
    [ MemoryRegion ] VariableKind Identifier ":" Type [ "=" Expression ] ";" ;

MemoryRegion = "storage" | "memory" | "tstore" ;

VariableKind = "var" | "let" | "const" | "immutable" ;

# ==========================================
# TYPE SYSTEM
# ==========================================

Type = 
    PrimitiveType
  | MapType
  | DoublemapType
  | ArrayType
  | OptionalType
  | ErrorUnionType
  | Identifier ;

PrimitiveType = 
    "u8" | "u16" | "u32" | "u64" | "u128" | "u256"
  | "i8" | "i16" | "i32" | "i64" | "i128" | "i256"
  | "bool" | "address" | "string" | "bytes" ;

MapType = "map" "[" Type "," Type "]" ;

DoublemapType = "doublemap" "[" Type "," Type "," Type "]" ;

ArrayType = "[" Type [ ";" Expression ] "]" ;

OptionalType = "?" Type ;

ErrorUnionType = Type "|" Type ;

# ==========================================
# STATEMENTS
# ==========================================

Statement = 
    VariableDeclaration
  | AssignmentStatement
  | CompoundAssignmentStatement
  | TransferStatement
  | ExpressionStatement
  | IfStatement
  | WhileStatement
  | ReturnStatement
  | BreakStatement
  | ContinueStatement
  | LogStatement
  | LockStatement
  | UnlockStatement
  | TryStatement
  | Block ;

TransferStatement = Identifier "from" Expression "->" Expression ":" Expression ";" ;

LockStatement = "@" "lock" "(" Expression ")" ";" ;

UnlockStatement = "@" "unlock" "(" Expression ")" ";" ;

TryStatement = "try" Expression [ "catch" Identifier Block ] ;

Block = "{" { Statement } "}" ;
```

## Key Grammar Features

### Memory Regions
- `storage` - Persistent contract state
- `memory` - Transaction-scoped temporary storage
- `tstore` - Transient storage (EIP-1153)

### Variable Kinds
- `var` - Mutable variable
- `let` - Immutable variable
- `const` - Compile-time constant
- `immutable` - Set once during deployment

### Special Syntax
- `@lock(expr)` - Lock a storage location
- `balances from sender -> recipient : amount` - Transfer statement
- `old(expr)` - Previous value in postconditions
- `comptime expr` - Compile-time evaluation

### Error Handling
- `!T` - Error union type
- `try expr` - Try expression
- `catch` - Error handling

### Formal Verification
- `requires(condition)` - Precondition
- `ensures(condition)` - Postcondition
- `invariant(condition)` - Loop invariant

## Operator Precedence

From highest to lowest precedence:

1. **Postfix operators** (`.`, `[]`, `()`)
2. **Unary operators** (`!`, `-`, `+`)
3. **Multiplicative** (`*`, `/`, `%`)
4. **Additive** (`+`, `-`)
5. **Relational** (`<`, `<=`, `>`, `>=`)
6. **Equality** (`==`, `!=`)
7. **Logical AND** (`&`)
8. **Logical OR** (`|`)
9. **Assignment** (`=`, `+=`, `-=`, `*=`)

## Example Usage

```ora
contract SimpleToken {
    storage var totalSupply: u256;
    storage var balances: map[address, u256];
    
    log Transfer(from: address, to: address, amount: u256);
    
    pub fn transfer(to: address, amount: u256) -> bool
        requires(balances[std.transaction.sender] >= amount)
        requires(to != std.constants.ZERO_ADDRESS)
        ensures(balances[std.transaction.sender] + balances[to] == 
                old(balances[std.transaction.sender]) + old(balances[to]))
    {
        @lock(balances[to]);
        balances from std.transaction.sender -> to : amount;
        log Transfer(std.transaction.sender, to, amount);
        return true;
    }
}
```

This grammar specification ensures precise syntax definition for the Ora language, supporting formal verification, memory safety, and smart contract-specific features. 