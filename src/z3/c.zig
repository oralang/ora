//===----------------------------------------------------------------------===//
//
// Z3 C API Bindings
//
//===----------------------------------------------------------------------===//
//
// This file provides Zig bindings to the Z3 SMT solver C API.
// It uses `@cImport` to directly import Z3 headers, enabling formal
// verification capabilities in the Ora compiler.
//
// Z3 is used for:
// - Arithmetic overflow/underflow detection
// - Array bounds checking
// - Storage consistency verification
// - User-defined invariants and contracts
//
//===----------------------------------------------------------------------===//

/// Z3 C API bindings
///
/// This uses Zig's `@cImport` to directly include Z3 headers.
/// Z3 must be installed on the system or built from vendor/z3.
///
/// Installation:
/// - macOS: `brew install z3`
/// - Linux: `sudo apt install z3` (Ubuntu/Debian)
/// - Windows: Download from https://github.com/Z3Prover/z3/releases
///
/// Build will automatically detect Z3 and link against it.
pub const c = @cImport({
    @cInclude("z3.h");
});

// Re-export commonly used Z3 types for convenience
pub const Z3_context = c.Z3_context;
pub const Z3_solver = c.Z3_solver;
pub const Z3_ast = c.Z3_ast;
pub const Z3_sort = c.Z3_sort;
pub const Z3_symbol = c.Z3_symbol;
pub const Z3_func_decl = c.Z3_func_decl;
pub const Z3_model = c.Z3_model;
pub const Z3_lbool = c.Z3_lbool;
pub const Z3_config = c.Z3_config;

// Re-export Z3_lbool enum values
pub const Z3_L_FALSE = c.Z3_L_FALSE;
pub const Z3_L_UNDEF = c.Z3_L_UNDEF;
pub const Z3_L_TRUE = c.Z3_L_TRUE;

// Re-export commonly used Z3 functions
pub const Z3_mk_config = c.Z3_mk_config;
pub const Z3_del_config = c.Z3_del_config;
pub const Z3_mk_context = c.Z3_mk_context;
pub const Z3_del_context = c.Z3_del_context;
pub const Z3_mk_solver = c.Z3_mk_solver;
pub const Z3_solver_inc_ref = c.Z3_solver_inc_ref;
pub const Z3_solver_dec_ref = c.Z3_solver_dec_ref;
pub const Z3_solver_assert = c.Z3_solver_assert;
pub const Z3_solver_check = c.Z3_solver_check;
pub const Z3_solver_get_model = c.Z3_solver_get_model;
pub const Z3_solver_reset = c.Z3_solver_reset;
pub const Z3_solver_push = c.Z3_solver_push;
pub const Z3_solver_pop = c.Z3_solver_pop;
pub const Z3_solver_to_string = c.Z3_solver_to_string;
pub const Z3_solver_from_string = c.Z3_solver_from_string;
pub const Z3_solver_set_params = c.Z3_solver_set_params;
pub const Z3_ast_to_string = c.Z3_ast_to_string;
pub const Z3_get_version = c.Z3_get_version;
pub const Z3_mk_params = c.Z3_mk_params;
pub const Z3_params_inc_ref = c.Z3_params_inc_ref;
pub const Z3_params_dec_ref = c.Z3_params_dec_ref;
pub const Z3_params_set_uint = c.Z3_params_set_uint;

// Bit-vector operations (for EVM u256)
pub const Z3_mk_bv_sort = c.Z3_mk_bv_sort;
pub const Z3_mk_bv_add = c.Z3_mk_bvadd;
pub const Z3_mk_bv_sub = c.Z3_mk_bvsub;
pub const Z3_mk_bv_mul = c.Z3_mk_bvmul;
pub const Z3_mk_bv_udiv = c.Z3_mk_bvudiv;
pub const Z3_mk_bv_urem = c.Z3_mk_bvurem;
pub const Z3_mk_bvult = c.Z3_mk_bvult;
pub const Z3_mk_bvugt = c.Z3_mk_bvugt;
pub const Z3_mk_bvule = c.Z3_mk_bvule;
pub const Z3_mk_bvuge = c.Z3_mk_bvuge;
pub const Z3_mk_bvslt = c.Z3_mk_bvslt;
pub const Z3_mk_bvsle = c.Z3_mk_bvsle;
pub const Z3_mk_bvsgt = c.Z3_mk_bvsgt;
pub const Z3_mk_bvsge = c.Z3_mk_bvsge;

pub const Z3_mk_tuple_sort = c.Z3_mk_tuple_sort;
pub const Z3_mk_bvand = c.Z3_mk_bvand;
pub const Z3_mk_bvor = c.Z3_mk_bvor;
pub const Z3_mk_bvxor = c.Z3_mk_bvxor;
pub const Z3_mk_bvshl = c.Z3_mk_bvshl;
pub const Z3_mk_bvashr = c.Z3_mk_bvashr;
pub const Z3_mk_bvlshr = c.Z3_mk_bvlshr;

// Constants and variables
pub const Z3_mk_const = c.Z3_mk_const;
pub const Z3_mk_int_symbol = c.Z3_mk_int_symbol;
pub const Z3_mk_string_symbol = c.Z3_mk_string_symbol;
pub const Z3_mk_unsigned_int64 = c.Z3_mk_unsigned_int64;
pub const Z3_mk_numeral = c.Z3_mk_numeral;
pub const Z3_mk_func_decl = c.Z3_mk_func_decl;
pub const Z3_mk_app = c.Z3_mk_app;

// Boolean operations
pub const Z3_mk_and = c.Z3_mk_and;
pub const Z3_mk_or = c.Z3_mk_or;
pub const Z3_mk_not = c.Z3_mk_not;
pub const Z3_mk_xor = c.Z3_mk_xor;
pub const Z3_mk_implies = c.Z3_mk_implies;
pub const Z3_mk_ite = c.Z3_mk_ite;
pub const Z3_mk_true = c.Z3_mk_true;
pub const Z3_mk_false = c.Z3_mk_false;
pub const Z3_mk_eq = c.Z3_mk_eq;
pub const Z3_mk_extract = c.Z3_mk_extract;
pub const Z3_mk_zero_ext = c.Z3_mk_zero_ext;
pub const Z3_mk_sign_ext = c.Z3_mk_sign_ext;
pub const Z3_mk_bool_sort = c.Z3_mk_bool_sort;
pub const Z3_mk_string_sort = c.Z3_mk_string_sort;
pub const Z3_is_string_sort = c.Z3_is_string_sort;
pub const Z3_mk_string = c.Z3_mk_string;
pub const Z3_mk_str_lt = c.Z3_mk_str_lt;
pub const Z3_mk_str_le = c.Z3_mk_str_le;
pub const Z3_mk_seq_concat = c.Z3_mk_seq_concat;
pub const Z3_mk_seq_length = c.Z3_mk_seq_length;

// Sort queries
pub const Z3_get_sort = c.Z3_get_sort;
pub const Z3_get_sort_kind = c.Z3_get_sort_kind;
pub const Z3_get_bv_sort_size = c.Z3_get_bv_sort_size;

// Sort kind constants
pub const Z3_BOOL_SORT = c.Z3_BOOL_SORT;
pub const Z3_BV_SORT = c.Z3_BV_SORT;
pub const Z3_ARRAY_SORT = c.Z3_ARRAY_SORT;

// Model evaluation
pub const Z3_model_inc_ref = c.Z3_model_inc_ref;
pub const Z3_model_dec_ref = c.Z3_model_dec_ref;
pub const Z3_model_eval = c.Z3_model_eval;
pub const Z3_model_to_string = c.Z3_model_to_string;
pub const Z3_get_numeral_string = c.Z3_get_numeral_string;

// Array operations (for storage maps)
pub const Z3_mk_array_sort = c.Z3_mk_array_sort;
pub const Z3_mk_select = c.Z3_mk_select;
pub const Z3_mk_store = c.Z3_mk_store;
