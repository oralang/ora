# CMake configuration for Ora MLIR dialect
cmake_minimum_required(VERSION 3.20)
project(OraDialect)

# Force C++17 standard for MLIR compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include MLIR headers
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})

# Define the Ora dialect library
# Main dialect library disabled due to TableGen compatibility issues with LLVM 22.0.0git
# add_mlir_dialect_library(MLIROraDialect
#   OraDialect.cpp
#   
#   PARTIAL_SOURCES_INTENDED
#   
#   ADDITIONAL_HEADER_DIRS
#   ${PROJECT_SOURCE_DIR}/src/mlir/generated
#   
#   DEPENDS
#   MLIROraOpsIncGen
#   
#   LINK_LIBS
#   MLIRIR
#   MLIRSupport
#   MLIRSideEffectInterfaces
#   MLIRSymbolInterfaces
# )

# Generate TableGen files
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/../OraDialect.td)
mlir_tablegen(OraDialect.h.inc -gen-dialect-decls -dialect=ora)
mlir_tablegen(OraDialect.cpp.inc -gen-dialect-defs -dialect=ora)
add_public_tablegen_target(MLIROraDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/../OraOpsMinimal.td)
mlir_tablegen(OraOps.h.inc -gen-op-decls -I${PROJECT_SOURCE_DIR}/..)
mlir_tablegen(OraOps.cpp.inc -gen-op-defs -I${PROJECT_SOURCE_DIR}/..)
add_public_tablegen_target(MLIROraOpsIncGen)

# Dependencies removed since main dialect library is disabled
# add_dependencies(MLIROraDialect MLIROraDialectIncGen)
# add_dependencies(MLIROraDialect MLIROraOpsIncGen)

# Define the Ora dialect C API library
# Note: We include OraDialect.cpp because OraDialectC.cpp needs the full OraDialect implementation
add_mlir_dialect_library(MLIROraDialectC
  OraDialectC.cpp
  OraDialect.cpp
  
  PARTIAL_SOURCES_INTENDED
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  
  DEPENDS
  MLIROraDialectIncGen
  MLIROraOpsIncGen
  
  LINK_LIBS
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRSymbolInterfaces
)

# Install the libraries
install(TARGETS MLIROraDialectC
  DESTINATION lib
  EXPORT OraDialectTargets)

install(FILES
  OraDialect.h
  OraDialectC.h
  ${PROJECT_BINARY_DIR}/OraDialect.h.inc
  ${PROJECT_BINARY_DIR}/OraOps.h.inc
  DESTINATION include/Ora)
