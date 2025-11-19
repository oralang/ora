# CMake configuration for EthIR MLIR dialect
cmake_minimum_required(VERSION 3.20)
project(EthIRDialect)

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
# DIALECT: EthIRDialect.td
# -------------------------------
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/td/EthIRDialect.td)

mlir_tablegen(EthIRDialect.h.inc 
  -gen-dialect-decls 
  -dialect=ethir 
  -I${PROJECT_SOURCE_DIR}/td
)

mlir_tablegen(EthIRDialect.cpp.inc 
  -gen-dialect-defs 
  -dialect=ethir 
  -I${PROJECT_SOURCE_DIR}/td
)

add_public_tablegen_target(MLIREthIRDialectIncGen)

# -------------------------------
# TYPES: EthIRTypes.td
# -------------------------------
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/td/EthIRTypes.td)

mlir_tablegen(EthIRTypes.h.inc 
  -gen-typedef-decls 
  -typedefs-dialect=ethir
  -I${PROJECT_SOURCE_DIR}/td
)

mlir_tablegen(EthIRTypes.cpp.inc 
  -gen-typedef-defs 
  -typedefs-dialect=ethir
  -I${PROJECT_SOURCE_DIR}/td
)

add_public_tablegen_target(MLIREthIRTypesIncGen)

# -------------------------------
# OPS: EthIROps.td
# -------------------------------
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/td/EthIROps.td)

mlir_tablegen(EthIROps.h.inc 
  -gen-op-decls 
  -I${PROJECT_SOURCE_DIR}/td
)

mlir_tablegen(EthIROps.cpp.inc 
  -gen-op-defs 
  -I${PROJECT_SOURCE_DIR}/td
)

add_public_tablegen_target(MLIREthIROpsIncGen)

# -------------------------------
# EthIR dialect library
# -------------------------------
add_mlir_dialect_library(MLIREthIRDialect
  lib/EthIRDialect.cpp
  lib/EthIRTypes.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

  DEPENDS
    MLIREthIRDialectIncGen
    MLIREthIRTypesIncGen
    MLIREthIROpsIncGen

  LINK_LIBS
    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces
    MLIRSymbolInterfaces
)

# -------------------------------
# Install targets
# -------------------------------
install(TARGETS MLIREthIRDialect
  EXPORT EthIRDialectTargets
  LIBRARY DESTINATION lib
)

install(FILES
  include/EthIRDialect.h
  include/EthIRTypes.h
  include/EthIROps.h
  ${PROJECT_BINARY_DIR}/EthIRDialect.h.inc
  ${PROJECT_BINARY_DIR}/EthIRTypes.h.inc
  ${PROJECT_BINARY_DIR}/EthIROps.h.inc
  DESTINATION include/EthIR
)