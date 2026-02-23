(comment) @comment
(string_literal) @string
(bytes_literal) @string.special
(number_literal) @number
(bool_literal) @boolean
(field_index) @number

;; Types and declarations
(primitive_type) @type.builtin
(contract_declaration name: (identifier) @type)
(struct_declaration name: (identifier) @type)
(enum_declaration name: (identifier) @type)
(bitfield_declaration name: (identifier) @type)
(error_declaration name: (identifier) @type)
(type (plain_type (identifier) @type))
(generic_type name: (identifier) @type)
(struct_literal name: (identifier) @type)

;; Functions and calls
(function_declaration name: (identifier) @function)
(log_declaration name: (identifier) @function)
((postfix_expression
   (primary_expression (identifier) @function.call)
   (postfix_operator "("))
 (#set! "priority" 85))

;; Distinct style for functions with comptime parameters
((function_declaration
   name: (identifier) @function.comptime
   (parameter_list
     (parameter (parameter_modifier))))
 (#set! "priority" 125))

;; Region-aware variables
((variable_declaration
   (memory_region) @keyword.storage
   name: (identifier) @variable.storage)
 (#eq? @keyword.storage "storage")
 (#set! "priority" 120))

((variable_declaration
   (memory_region) @keyword.memory
   name: (identifier) @variable.memory)
 (#eq? @keyword.memory "memory")
 (#set! "priority" 120))

((variable_declaration
   (memory_region) @keyword.tstore
   name: (identifier) @variable.tstore)
 (#eq? @keyword.tstore "tstore")
 (#set! "priority" 120))

((variable_declaration
   (memory_region) @keyword.calldata
   name: (identifier) @variable.calldata)
 (#eq? @keyword.calldata "calldata")
 (#set! "priority" 120))

;; Ghost declarations
(ghost_declaration "ghost" @keyword.ghost)
((ghost_declaration
   (variable_declaration name: (identifier) @variable.ghost))
 (#set! "priority" 120))
((ghost_declaration
   (function_declaration name: (identifier) @function.ghost))
 (#set! "priority" 120))

;; Formal verification constructs
(requires_clause "requires" @keyword.fv)
(ensures_clause "ensures" @keyword.fv)
(invariant_clause "invariant" @keyword.fv)
(decreases_clause "decreases" @keyword.fv)
(contract_invariant "invariant" @keyword.fv)
(requires_statement "requires" @keyword.fv)
(ensures_statement "ensures" @keyword.fv)
(quantified_expression ["forall" "exists" "where"] @keyword.fv)

;; Comptime constructs
((comptime_expression "comptime" @keyword.comptime)
 (#set! "priority" 120))
((comptime_statement "comptime" @keyword.comptime)
 (#set! "priority" 120))
((parameter_modifier) @keyword.comptime
 (#set! "priority" 120))
((parameter
   modifier: (parameter_modifier)
   name: (identifier) @variable.parameter.comptime)
 (#set! "priority" 120))

;; Builtins: std namespace
((postfix_expression
   (primary_expression (identifier) @module.builtin)
   (postfix_operator "." (identifier)))
 (#eq? @module.builtin "std")
 (#set! "priority" 130))

((postfix_expression
   (primary_expression (identifier) @_std_root)
   (postfix_operator "." (identifier) @module.builtin))
 (#eq? @_std_root "std")
 (#set! "priority" 130))

((postfix_expression
   (primary_expression (identifier) @_std_root)
   (postfix_operator "." (identifier) @_std_ns)
   (postfix_operator "." (identifier) @constant.builtin))
 (#eq? @_std_root "std")
 (#eq? @_std_ns "constants")
 (#set! "priority" 130))

((postfix_expression
   (primary_expression (identifier) @_std_root)
   (postfix_operator "." (identifier) @_std_ns)
   (postfix_operator "." (identifier) @function.builtin))
 (#eq? @_std_root "std")
 (#match? @_std_ns "^(msg|transaction)$")
 (#set! "priority" 130))

((postfix_expression
   (primary_expression (identifier) @_std_root)
   (postfix_operator "." (identifier) @_std_ns)
   (postfix_operator "." (identifier) @variable.builtin))
 (#eq? @_std_root "std")
 (#eq? @_std_ns "block")
 (#set! "priority" 130))

;; Builtins: intrinsic calls and error payloads
((intrinsic_expression name: (identifier) @function.builtin)
 (#set! "priority" 130))

((intrinsic_expression name: (identifier) @keyword.directive)
 (#match? @keyword.directive "^(lock|unlock)$")
 (#set! "priority" 135))

(error_expression "error" @module.builtin)
(error_expression name: (identifier) @constant.builtin)

;; Symbols and locals
(parameter name: (identifier) @variable.parameter)
(tuple_pattern (identifier) @variable)
(tuple_variable_declaration pattern: (tuple_pattern (identifier) @variable))
(variable_declaration name: (identifier) @variable)
(struct_field name: (identifier) @property)
(struct_literal_field name: (identifier) @property)
(log_parameter name: (identifier) @property)
(enum_member name: (identifier) @constant)
(labeled_block label: (identifier) @label)
(break_statement (identifier) @label)
(continue_statement (identifier) @label)

;; Memory-region keywords in non-declaration contexts
((memory_region) @keyword.storage (#eq? @keyword.storage "storage"))
((memory_region) @keyword.memory (#eq? @keyword.memory "memory"))
((memory_region) @keyword.tstore (#eq? @keyword.tstore "tstore"))
((memory_region) @keyword.calldata (#eq? @keyword.calldata "calldata"))

;; Keywords
[
  "contract"
  "fn"
  "pub"
  "struct"
  "bitfield"
  "enum"
  "log"
  "error"
  "ghost"
  "if"
  "else"
  "while"
  "for"
  "switch"
  "return"
  "break"
  "continue"
  "try"
  "catch"
  "assert"
  "const"
  "comptime"
  "let"
  "var"
  "immutable"
  "invariant"
  "decreases"
  "requires"
  "ensures"
  "forall"
  "exists"
  "where"
  "assume"
  "havoc"
  "old"
  "calldata"
  "indexed"
  "map"
  "slice"
] @keyword

(lock_statement "lock" @keyword.directive)
(unlock_statement "unlock" @keyword.directive)
(import_declaration "import" @include)

;; Operators
[
  "="
  "+="
  "-="
  "*="
  "/="
  "%="
  "||"
  "&&"
  "|"
  "^"
  "&"
  "=="
  "!="
  "<"
  "<="
  ">"
  ">="
  "<<"
  ">>"
  "<<%"
  ">>%"
  "+"
  "-"
  "*"
  "/"
  "%"
  "+%"
  "-%"
  "*%"
  "**"
  "**%"
  "->"
  "=>"
  ".."
  "..."
] @operator

;; Punctuation
[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

[
  ","
  ";"
  ":"
  "."
  "@"
] @punctuation.delimiter
