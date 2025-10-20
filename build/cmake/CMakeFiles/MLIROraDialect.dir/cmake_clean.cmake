file(REMOVE_RECURSE
  "OraDialect.cpp.inc"
  "OraDialect.h.inc"
  "OraOps.cpp.inc"
  "OraOps.h.inc"
  "lib/libMLIROraDialect.a"
  "lib/libMLIROraDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIROraDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
