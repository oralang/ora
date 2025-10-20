file(REMOVE_RECURSE
  "OraDialect.cpp.inc"
  "OraDialect.h.inc"
  "OraOps.cpp.inc"
  "OraOps.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/mlir-doc.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
