/**
 * @file A tree-sitter grammar for Ora
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

const PREC = {
  ASSIGN: 1,
  LOGICAL_OR: 2,
  LOGICAL_AND: 3,
  BIT_OR: 4,
  BIT_XOR: 5,
  BIT_AND: 6,
  EQUALITY: 7,
  RELATIONAL: 8,
  SHIFT: 9,
  ADDITIVE: 10,
  MULTIPLICATIVE: 11,
  UNARY: 12,
  POSTFIX: 13,
}

module.exports = grammar({
  name: "ora",

  word: ($) => $.identifier,

  extras: ($) => [
    /\s/,
    $.comment,
  ],

  conflicts: ($) => [
    [$.import_declaration, $.variable_kind],
    [$.lvalue, $.primary_expression],
    [$.assignment_statement, $.assignment_operator],
    [$.compound_operator, $.assignment_operator],
    [$.switch_statement, $.switch_expression],
  ],

  rules: {
    source_file: ($) => repeat($._top_level_declaration),

    _top_level_declaration: ($) => choice(
      $.contract_declaration,
      $.function_declaration,
      $.variable_declaration,
      $.struct_declaration,
      $.bitfield_declaration,
      $.enum_declaration,
      $.log_declaration,
      $.import_declaration,
      $.error_declaration,
      $.ghost_declaration,
      $.contract_invariant,
    ),

    import_declaration: ($) => choice(
      seq("@", "import", "(", field("path", $.string_literal), ")", ";"),
      seq(
        "const",
        field("name", $.identifier),
        "=",
        "@",
        "import",
        "(",
        field("path", $.string_literal),
        ")",
        ";",
      ),
    ),

    contract_declaration: ($) => seq(
      "contract",
      field("name", $.identifier),
      "{",
      repeat($.contract_member),
      "}",
    ),

    contract_member: ($) => choice(
      $.variable_declaration,
      $.function_declaration,
      $.log_declaration,
      $.struct_declaration,
      $.bitfield_declaration,
      $.enum_declaration,
      $.error_declaration,
      $.ghost_declaration,
      $.contract_invariant,
    ),

    function_declaration: ($) => seq(
      optional("pub"),
      "fn",
      field("name", choice($.identifier, "init")),
      "(",
      optional($.parameter_list),
      ")",
      optional($.return_type),
      repeat($.requires_clause),
      repeat($.ensures_clause),
      field("body", $.block),
    ),

    parameter_list: ($) => seq(
      $.parameter,
      repeat(seq(",", $.parameter)),
      optional(","),
    ),

    parameter: ($) => seq(
      field("name", $.identifier),
      ":",
      field("type", $.type),
    ),

    return_type: ($) => seq("->", field("type", $.type)),
    requires_clause: ($) => seq("requires", "(", field("condition", $._expression), ")"),
    ensures_clause: ($) => seq("ensures", "(", field("condition", $._expression), ")"),

    variable_declaration: ($) => seq(
      optional($.memory_region),
      field("kind", $.variable_kind),
      field("name", $.identifier),
      optional(seq(":", field("type", $.type))),
      optional(seq("=", field("value", $._expression))),
      ";",
    ),

    memory_region: (_) => choice("storage", "memory", "tstore"),
    variable_kind: (_) => choice("var", "let", "const", "immutable"),

    struct_declaration: ($) => seq(
      "struct",
      field("name", $.identifier),
      "{",
      repeat($.struct_field),
      "}",
    ),

    struct_field: ($) => seq(
      field("name", $.identifier),
      ":",
      field("type", $.type),
      ";",
    ),

    bitfield_declaration: ($) => seq(
      "bitfield",
      field("name", $.identifier),
      ":",
      field("type", $.type),
      "{",
      repeat($.bitfield_field),
      "}",
    ),

    bitfield_field: ($) => seq(
      field("name", $.identifier),
      ":",
      field("type", $.type),
      optional($.bitfield_layout),
      ";",
    ),

    bitfield_layout: ($) => choice(
      seq("@at", "(", $.number_literal, ",", $.number_literal, ")"),
      seq("@bits", "(", $.number_literal, "..", $.number_literal, ")"),
      seq("(", $.number_literal, ")"),
    ),

    enum_declaration: ($) => seq(
      "enum",
      field("name", $.identifier),
      optional(seq(":", field("base_type", $.type))),
      "{",
      optional($.enum_member_list),
      "}",
    ),

    enum_member_list: ($) => seq(
      $.enum_member,
      repeat(seq(",", $.enum_member)),
      optional(","),
    ),

    enum_member: ($) => seq(
      field("name", $.identifier),
      optional(seq("=", field("value", $._expression))),
    ),

    log_declaration: ($) => seq(
      "log",
      field("name", $.identifier),
      "(",
      optional($.log_parameter_list),
      ")",
      ";",
    ),

    log_parameter_list: ($) => seq(
      $.log_parameter,
      repeat(seq(",", $.log_parameter)),
      optional(","),
    ),

    log_parameter: ($) => seq(
      optional("indexed"),
      field("name", $.identifier),
      ":",
      field("type", $.type),
    ),

    error_declaration: ($) => seq(
      "error",
      field("name", $.identifier),
      optional(seq("(", optional($.parameter_list), ")")),
      ";",
    ),

    ghost_declaration: ($) => seq(
      "ghost",
      choice(
        $.variable_declaration,
        $.function_declaration,
        $.block,
      ),
    ),

    contract_invariant: ($) => seq(
      "invariant",
      field("name", $.identifier),
      "(",
      field("condition", $._expression),
      ")",
      ";",
    ),

    type: ($) => choice(
      $.error_union_type,
      $.plain_type,
    ),

    error_union_type: ($) => prec.right(seq(
      "!",
      field("member", $.plain_type),
      repeat(seq("|", field("member", $.plain_type))),
    )),

    plain_type: ($) => choice(
      $.primitive_type,
      $.map_type,
      $.array_type,
      $.slice_type,
      $.anonymous_struct_type,
      $.generic_type,
      $.identifier,
    ),

    primitive_type: (_) => choice(
      "u8", "u16", "u32", "u64", "u128", "u256",
      "i8", "i16", "i32", "i64", "i128", "i256",
      "bool", "address", "string", "bytes", "void",
    ),

    type_argument: ($) => choice(
      $.type,
      $.number_literal,
    ),

    generic_type: ($) => seq(
      field("name", $.identifier),
      "<",
      commaSep1($.type_argument),
      optional(","),
      ">",
    ),

    map_type: ($) => seq(
      "map",
      "<",
      field("key", $.type),
      ",",
      field("value", $.type),
      ">",
    ),

    array_type: ($) => seq(
      "[",
      field("element", $.type),
      ";",
      field("size", $._expression),
      "]",
    ),

    slice_type: ($) => seq(
      "slice",
      "[",
      field("element", $.type),
      "]",
    ),

    anonymous_struct_type: ($) => seq(
      "struct",
      "{",
      optional($.anonymous_struct_field_list),
      "}",
    ),

    anonymous_struct_field_list: ($) => seq(
      $.anonymous_struct_field,
      repeat(seq(",", $.anonymous_struct_field)),
      optional(","),
    ),

    anonymous_struct_field: ($) => seq(
      field("name", $.identifier),
      ":",
      field("type", $.type),
    ),

    _statement: ($) => choice(
      $.variable_declaration,
      $.assignment_statement,
      $.compound_assignment_statement,
      $.destructuring_assignment,
      $.expression_statement,
      $.if_statement,
      $.while_statement,
      $.for_statement,
      $.switch_statement,
      $.return_statement,
      $.break_statement,
      $.continue_statement,
      $.log_statement,
      $.lock_statement,
      $.unlock_statement,
      $.try_statement,
      $.assert_statement,
      $.labeled_block,
      $.block,
    ),

    assignment_statement: ($) => seq(
      field("left", $.lvalue),
      "=",
      field("right", $._expression),
      ";",
    ),

    compound_assignment_statement: ($) => seq(
      field("left", $.lvalue),
      field("operator", $.compound_operator),
      field("right", $._expression),
      ";",
    ),

    compound_operator: (_) => choice("+=", "-=", "*=", "/=", "%="),

    destructuring_assignment: ($) => seq(
      "let",
      $.destructuring_pattern,
      "=",
      field("value", $._expression),
      ";",
    ),

    destructuring_pattern: ($) => seq(
      ".",
      "{",
      optional($.destructuring_field_list),
      "}",
    ),

    destructuring_field_list: ($) => seq(
      $.destructuring_field,
      repeat(seq(",", $.destructuring_field)),
      optional(","),
    ),

    destructuring_field: ($) => choice(
      $.identifier,
      seq(field("from", $.identifier), ":", field("to", $.identifier)),
    ),

    expression_statement: ($) => seq($._expression, ";"),

    if_statement: ($) => prec.right(seq(
      "if",
      "(",
      field("condition", $._expression),
      ")",
      field("consequence", $._statement),
      optional(seq("else", field("alternative", $._statement))),
    )),

    while_statement: ($) => seq(
      "while",
      "(",
      field("condition", $._expression),
      ")",
      repeat($.invariant_clause),
      field("body", $._statement),
    ),

    for_statement: ($) => seq(
      "for",
      "(",
      field("iterable", $._expression),
      ")",
      "|",
      field("pattern", $.for_pattern),
      "|",
      repeat($.invariant_clause),
      field("body", $._statement),
    ),

    for_pattern: ($) => choice(
      seq($.identifier, optional(seq(",", $.identifier))),
      $.destructuring_pattern,
    ),

    invariant_clause: ($) => seq("invariant", "(", field("condition", $._expression), ")"),

    return_statement: ($) => seq("return", optional($._expression), ";"),

    break_statement: ($) => seq(
      "break",
      optional(seq(":", $.identifier, optional($._expression))),
      ";",
    ),

    continue_statement: ($) => seq(
      "continue",
      optional(seq(":", $.identifier)),
      ";",
    ),

    log_statement: ($) => seq(
      "log",
      field("name", $.identifier),
      "(",
      optional($.expression_list),
      ")",
      ";",
    ),

    lock_statement: ($) => seq("@", "lock", "(", field("target", $._expression), ")", ";"),
    unlock_statement: ($) => seq("@", "unlock", "(", field("target", $._expression), ")", ";"),

    assert_statement: ($) => seq(
      "assert",
      "(",
      field("condition", $._expression),
      optional(seq(",", field("message", $.string_literal))),
      ")",
      ";",
    ),

    try_statement: ($) => seq(
      "try",
      field("body", $.block),
      optional(seq(
        "catch",
        optional(seq("(", field("error", $.identifier), ")")),
        field("catch_body", $.block),
      )),
    ),

    block: ($) => seq("{", repeat($._statement), "}"),

    labeled_block: ($) => seq(field("label", $.identifier), ":", field("body", $.block)),

    switch_statement: ($) => seq(
      "switch",
      "(",
      field("value", $._expression),
      ")",
      "{",
      repeat($.switch_arm),
      "}",
    ),

    switch_expression: ($) => seq(
      "switch",
      "(",
      field("value", $._expression),
      ")",
      "{",
      repeat($.switch_expression_arm),
      "}",
    ),

    switch_arm: ($) => seq($.switch_pattern, "=>", $.switch_body, optional(",")),
    switch_expression_arm: ($) => seq($.switch_pattern, "=>", $._expression, optional(",")),

    switch_pattern: ($) => choice(
      $.range_pattern,
      "else",
      $._expression,
    ),

    range_pattern: ($) => prec.right(seq(
      field("start", $._expression),
      "...",
      field("end", $._expression),
    )),

    switch_body: ($) => choice(
      seq($._expression, ";"),
      $.block,
    ),

    _expression: ($) => choice(
      $.assignment_expression,
      $.binary_expression,
      $.unary_expression,
      $.postfix_expression,
      $.primary_expression,
    ),

    assignment_expression: ($) => prec.right(PREC.ASSIGN, seq(
      field("left", $.lvalue),
      field("operator", $.assignment_operator),
      field("right", $._expression),
    )),

    assignment_operator: (_) => choice("=", "+=", "-=", "*=", "/=", "%="),

    binary_expression: ($) => choice(
      prec.left(PREC.LOGICAL_OR, seq($._expression, "||", $._expression)),
      prec.left(PREC.LOGICAL_AND, seq($._expression, "&&", $._expression)),
      prec.left(PREC.BIT_OR, seq($._expression, "|", $._expression)),
      prec.left(PREC.BIT_XOR, seq($._expression, "^", $._expression)),
      prec.left(PREC.BIT_AND, seq($._expression, "&", $._expression)),
      prec.left(PREC.EQUALITY, seq($._expression, choice("==", "!="), $._expression)),
      prec.left(PREC.RELATIONAL, seq($._expression, choice("<", "<=", ">", ">="), $._expression)),
      prec.left(PREC.SHIFT, seq($._expression, choice("<<", ">>", "<<%", ">>%"), $._expression)),
      prec.left(PREC.ADDITIVE, seq($._expression, choice("+", "-", "+%", "-%"), $._expression)),
      prec.left(PREC.MULTIPLICATIVE, seq($._expression, choice("*", "/", "%", "*%"), $._expression)),
    ),

    unary_expression: ($) => prec(PREC.UNARY, seq(choice("!", "-", "+"), $._expression)),

    postfix_expression: ($) => prec.left(PREC.POSTFIX, seq(
      $.primary_expression,
      repeat1($.postfix_operator),
    )),

    postfix_operator: ($) => choice(
      seq(".", $.identifier),
      seq("[", $._expression, "]"),
      seq("(", optional($.expression_list), ")"),
    ),

    lvalue: ($) => seq(
      choice(
        $.identifier,
        seq("(", $.lvalue, ")"),
      ),
      repeat(choice(
        seq(".", $.identifier),
        seq("[", $._expression, "]"),
      )),
    ),

    primary_expression: ($) => choice(
      $.literal,
      $.identifier,
      $.parenthesized_expression,
      $.try_expression,
      $.old_expression,
      $.comptime_expression,
      $.cast_expression,
      $.error_expression,
      $.quantified_expression,
      $.anonymous_struct_literal,
      $.struct_literal,
      $.switch_expression,
      $.array_literal,
    ),

    literal: ($) => choice(
      $.bool_literal,
      $.number_literal,
      $.string_literal,
    ),

    parenthesized_expression: ($) => seq("(", $._expression, ")"),
    try_expression: ($) => seq("try", $._expression),
    old_expression: ($) => seq("old", "(", $._expression, ")"),
    comptime_expression: ($) => seq("comptime", $.block),
    cast_expression: ($) => seq("@", "cast", "(", $.type, ",", $._expression, ")"),

    error_expression: ($) => seq(
      "error",
      ".",
      field("name", $.identifier),
    ),

    quantified_expression: ($) => seq(
      choice("forall", "exists"),
      field("name", $.identifier),
      ":",
      field("type", $.type),
      optional(seq("where", field("guard", $._expression))),
      "=>",
      field("body", $._expression),
    ),

    struct_literal: ($) => seq(
      field("name", $.identifier),
      field("fields", $.struct_literal_fields),
    ),

    anonymous_struct_literal: ($) => seq(
      ".",
      field("fields", $.struct_literal_fields),
    ),

    struct_literal_fields: ($) => seq("{", optional($.struct_literal_field_list), "}"),

    struct_literal_field_list: ($) => seq(
      $.struct_literal_field,
      repeat(seq(",", $.struct_literal_field)),
      optional(","),
    ),

    struct_literal_field: ($) => seq(
      field("name", $.identifier),
      ":",
      field("value", $._expression),
    ),

    array_literal: ($) => seq("[", optional($.expression_list), "]"),

    expression_list: ($) => seq(
      $._expression,
      repeat(seq(",", $._expression)),
      optional(","),
    ),

    comment: (_) => token(choice(
      seq("//", /[^\n]*/),
      seq("/*", /[^*]*\*+([^/*][^*]*\*+)*/, "/"),
    )),

    bool_literal: (_) => choice("true", "false"),

    number_literal: (_) => choice(
      /0x[0-9A-Fa-f][0-9A-Fa-f_]*/,
      /0b[01][01_]*/,
      /[0-9][0-9_]*/,
    ),

    string_literal: (_) => /"(?:[^"\\]|\\.)*"/,

    identifier: (_) => /[A-Za-z_][A-Za-z0-9_]*/,
  },
})

function commaSep1(rule) {
  return seq(rule, repeat(seq(",", rule)))
}
