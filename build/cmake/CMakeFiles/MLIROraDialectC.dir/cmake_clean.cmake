file(REMOVE_RECURSE
  "OraDialect.cpp.inc"
  "OraDialect.h.inc"
  "OraOps.cpp.inc"
  "OraOps.h.inc"
  "lib/libMLIROraDialectC.a"
  "lib/libMLIROraDialectC.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIROraDialectC.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
