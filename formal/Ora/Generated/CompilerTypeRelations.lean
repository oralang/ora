/-
GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
`import` to this file. It contains only `def … := <literal>` relation rows
emitted from the compiler. The TRUSTED checks live in
`Ora/TypeRelationsSync.lean`.

Regenerate with `scripts/check-formal-type-relations-sync.sh`. Source:
src/formal/emit_type_relations_snapshot.zig,
src/sema/type_descriptors.zig, and src/sema/region.zig.
-/

namespace Ora.Generated

def compilerTypeEqlRows : List (String × String × String × Bool) :=
  [("primitive_u256_self", "u256", "u256", true),
   ("primitive_u8_vs_u256", "u8", "u256", false),
   ("fixed_bytes_same", "bytes4", "bytes4", true),
   ("fixed_bytes_different", "bytes4", "bytes32", false),
   ("tuple_same", "(u8,bool)", "(u8,bool)", true),
   ("tuple_element_width_diff", "(u8,bool)", "(u16,bool)", false),
   ("array_same", "[3]u8", "[3]u8", true),
   ("slice_element_diff", "slice[u8]", "slice[u16]", false),
   ("map_same", "map[address,u8]", "map[address,u8]", true),
   ("nominal_struct_same", "Point", "Point", true),
   ("nominal_struct_different", "Point", "OtherPoint", false),
   ("refinement_args_differ", "MinValue<u256,1>", "MinValue<u256,100>", false),
   ("function_same_name", "foo(u8)->bool", "foo(u8)->bool", true),
   ("function_different_name", "foo(u8)->bool", "bar(u8)->bool", false),
   ("resource_domain_same", "resource TokenUnit<u8>", "resource TokenUnit<u8>", true),
   ("resource_domain_carrier_diff", "resource TokenUnit<u8>", "resource TokenUnit<u16>", false)]

def compilerTypesAssignableRows : List (String × String × String × Bool) :=
  [("u8_to_u256", "u256", "u8", true),
   ("u256_to_u8_rejected", "u8", "u256", false),
   ("signedness_mismatch", "u8", "i8", false),
   ("tuple_widening", "(u16,bool)", "(u8,bool)", true),
   ("tuple_narrowing_rejected", "(u8,bool)", "(u16,bool)", false),
   ("array_widening", "[3]u16", "[3]u8", true),
   ("slice_widening", "slice[u16]", "slice[u8]", true),
   ("map_value_widening", "map[address,u16]", "map[address,u8]", true),
   ("nominal_struct_same", "Point", "Point", true),
   ("nominal_struct_different", "Point", "OtherPoint", false),
   ("function_same_name_identity", "foo(u8)->bool", "foo(u8)->bool", true),
   ("function_different_name_rejected", "foo(u8)->bool", "bar(u8)->bool", false),
   ("resource_domain_same_identity", "resource TokenUnit<u8>", "resource TokenUnit<u8>", true),
   ("resource_domain_widening_rejected", "resource TokenUnit<u16>", "resource TokenUnit<u8>", false),
   ("resource_place_same_identity", "Resource<TokenUnit<u8>>", "Resource<TokenUnit<u8>>", true),
   ("resource_place_widening_rejected", "Resource<TokenUnit<u16>>", "Resource<TokenUnit<u8>>", false),
   ("refinement_minvalue_widen", "MinValue<u256,1>", "MinValue<u256,100>", true),
   ("refinement_minvalue_narrow_rejected", "MinValue<u256,100>", "MinValue<u256,1>", false),
   ("error_union_subset", "u256!{Point,OtherPoint}", "u256!{Point}", true),
   ("error_union_superset_rejected", "u256!{Point}", "u256!{Point,OtherPoint}", false)]

def compilerLocatedTypeEqlRows : List (String × String × String × Bool) :=
  [("same_type_region_provenance", "u256@storage/local", "u256@storage/local", true),
   ("different_region", "u256@storage/local", "u256@memory/local", false),
   ("different_type", "u8@storage/local", "u256@storage/local", false),
   ("different_provenance", "u256@storage/local", "u256@storage/storage", false)]

def compilerLocatedAssignableRows : List (String × String × String × Bool) :=
  [("calldata_u8_to_memory_u256", "u8@calldata/local", "u256@memory/local", true),
   ("memory_u256_to_calldata_u256_rejected", "u256@memory/local", "u256@calldata/local", false),
   ("storage_bool_to_memory_bool", "bool@storage/local", "bool@memory/local", true),
   ("storage_bool_to_memory_address_rejected", "bool@storage/local", "address@memory/local", false),
   ("storage_u8_to_storage_u256", "u8@storage/local", "u256@storage/local", true),
   ("provenance_ignored_by_assignability", "u256@storage/storage", "u256@storage/local", true)]

end Ora.Generated
