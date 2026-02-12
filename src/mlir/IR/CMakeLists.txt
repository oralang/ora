# CMake configuration for SIR MLIR dialect
cmake_minimum_required(VERSION 3.20)
project(SIRDialect)

# -------------------------------
# MLIR / LLVM configuration
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIR_DIR: ${MLIR_DIR}")
message(STATUS "Using LLVM_DIR: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# -------------------------------
# Include dirs
# -------------------------------
include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}
)

# -------------------------------
# DIALECT: SIRDialect.td
# -------------------------------
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/td/SIRDialect.td)

mlir_tablegen(SIRDialect.h.inc 
  -gen-dialect-decls 
  -dialect=sir 
  -I${PROJECT_SOURCE_DIR}/td
)

mlir_tablegen(SIRDialect.cpp.inc 
  -gen-dialect-defs 
  -dialect=sir 
  -I${PROJECT_SOURCE_DIR}/td
)

add_public_tablegen_target(MLIRSIRDialectIncGen)

# -------------------------------
# TYPES: SIRTypes.td
# -------------------------------
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/td/SIRTypes.td)

mlir_tablegen(SIRTypes.h.inc 
  -gen-typedef-decls 
  -typedefs-dialect=sir
  -I${PROJECT_SOURCE_DIR}/td
)

mlir_tablegen(SIRTypes.cpp.inc 
  -gen-typedef-defs 
  -typedefs-dialect=sir
  -I${PROJECT_SOURCE_DIR}/td
)

add_public_tablegen_target(MLIRSIRTypesIncGen)

# -------------------------------
# OPS: SIROps.td
# -------------------------------
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/td/SIROps.td)

mlir_tablegen(SIROps.h.inc 
  -gen-op-decls 
  -I${PROJECT_SOURCE_DIR}/td
)

mlir_tablegen(SIROps.cpp.inc 
  -gen-op-defs 
  -I${PROJECT_SOURCE_DIR}/td
)

add_public_tablegen_target(MLIRSIROpsIncGen)

# -------------------------------
# SIR dialect library
# -------------------------------
add_mlir_dialect_library(MLIRSIRDialect
  lib/SIRDialect.cpp
  lib/SIRTypes.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

  DEPENDS
    MLIRSIRDialectIncGen
    MLIRSIRTypesIncGen
    MLIRSIROpsIncGen

  LINK_LIBS
    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces
    MLIRSymbolInterfaces
)

# Ensure generated TableGen includes exist before compiling sources that
# transitively include SIRDialect.h -> *.inc.
set_source_files_properties(
  lib/SIRDialect.cpp
  lib/SIRTypes.cpp
  PROPERTIES
    OBJECT_DEPENDS
      "${PROJECT_BINARY_DIR}/SIRDialect.h.inc;${PROJECT_BINARY_DIR}/SIRDialect.cpp.inc;${PROJECT_BINARY_DIR}/SIRTypes.h.inc;${PROJECT_BINARY_DIR}/SIRTypes.cpp.inc;${PROJECT_BINARY_DIR}/SIROps.h.inc;${PROJECT_BINARY_DIR}/SIROps.cpp.inc"
)

# -------------------------------
# Install targets
# -------------------------------
install(TARGETS MLIRSIRDialect
  EXPORT SIRDialectTargets
  LIBRARY DESTINATION lib
)

install(FILES
  include/SIRDialect.h
  include/SIRTypes.h
  include/SIROps.h
  ${PROJECT_BINARY_DIR}/SIRDialect.h.inc
  ${PROJECT_BINARY_DIR}/SIRTypes.h.inc
  ${PROJECT_BINARY_DIR}/SIROps.h.inc
  DESTINATION include/SIR
)
