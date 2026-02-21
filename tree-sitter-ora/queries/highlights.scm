(comment) @comment
(string_literal) @string
(number_literal) @number
(bool_literal) @boolean

(primitive_type) @type.builtin

(contract_declaration name: (identifier) @type)
(struct_declaration name: (identifier) @type)
(enum_declaration name: (identifier) @type)
(bitfield_declaration name: (identifier) @type)
(error_declaration name: (identifier) @type)

(function_declaration name: (identifier) @function)
(log_declaration name: (identifier) @function)

(parameter name: (identifier) @variable.parameter)
(variable_declaration name: (identifier) @variable)
(struct_field name: (identifier) @property)
(struct_literal_field name: (identifier) @property)
(log_parameter name: (identifier) @property)

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
  "invariant"
  "requires"
  "ensures"
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
  "forall"
  "exists"
  "where"
  "const"
  "let"
  "var"
  "immutable"
  "storage"
  "memory"
  "tstore"
  "indexed"
] @keyword

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
  "+"
  "-"
  "*"
  "/"
  "%"
  "->"
  "=>"
  "..."
] @operator

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
