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
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/lowering/OraToSIR)

# Include SIR dialect headers (from installed location)
# SIR is installed to ${CMAKE_INSTALL_PREFIX}/include/SIR
# We need to find CMAKE_INSTALL_PREFIX from MLIR_DIR
get_filename_component(MLIR_ROOT_DIR ${MLIR_DIR} DIRECTORY)
get_filename_component(MLIR_ROOT_DIR ${MLIR_ROOT_DIR} DIRECTORY)
# Check both possible locations
set(SIR_INCLUDE_DIR1 ${MLIR_ROOT_DIR}/include/SIR)
set(SIR_INCLUDE_DIR2 ${CMAKE_CURRENT_SOURCE_DIR}/../IR/include)
if(EXISTS ${SIR_INCLUDE_DIR1})
  include_directories(${SIR_INCLUDE_DIR1})
else()
  include_directories(${SIR_INCLUDE_DIR2})
endif()

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
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/td/OraDialect.td)
mlir_tablegen(OraDialect.h.inc -gen-dialect-decls -dialect=ora -I${PROJECT_SOURCE_DIR}/td)
mlir_tablegen(OraDialect.cpp.inc -gen-dialect-defs -dialect=ora -I${PROJECT_SOURCE_DIR}/td)
add_public_tablegen_target(MLIROraDialectIncGen)

# Generate type definitions - need to set LLVM_TARGET_DEFINITIONS to OraTypes.td
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/td/OraTypes.td)
mlir_tablegen(OraTypes.h.inc -gen-typedef-decls -typedefs-dialect=ora -I${PROJECT_SOURCE_DIR}/td)
mlir_tablegen(OraTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ora -I${PROJECT_SOURCE_DIR}/td)
add_public_tablegen_target(MLIROraTypesIncGen)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/td/OraOps.td)
mlir_tablegen(OraOps.h.inc -gen-op-decls -I${PROJECT_SOURCE_DIR}/td)
mlir_tablegen(OraOps.cpp.inc -gen-op-defs -I${PROJECT_SOURCE_DIR}/td)
add_public_tablegen_target(MLIROraOpsIncGen)

# Dependencies removed since main dialect library is disabled
# add_dependencies(MLIROraDialect MLIROraDialectIncGen)
# add_dependencies(MLIROraDialect MLIROraOpsIncGen)

# Define the Ora dialect C API library
# Note: We include OraDialect.cpp because OraDialectC.cpp needs the full OraDialect implementation
add_mlir_dialect_library(MLIROraDialectC
  lib/OraCAPI.cpp
  lib/OraDialect.cpp
  lowering/OraToSIR/OraToSIR.cpp
  lowering/OraToSIR/OraToSIRTypeConverter.cpp
  lowering/OraToSIR/patterns/Naming.cpp
  lowering/OraToSIR/patterns/Arithmetic.cpp
  lowering/OraToSIR/patterns/Struct.cpp
  lowering/OraToSIR/patterns/Storage.cpp
  lowering/OraToSIR/patterns/MemRef.cpp
  lowering/OraToSIR/patterns/ControlFlow.cpp
  lowering/OraToSIR/patterns/EVM.cpp
  
  PARTIAL_SOURCES_INTENDED
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/lowering/OraToSIR
  
  DEPENDS
  MLIROraDialectIncGen
  MLIROraTypesIncGen
  MLIROraOpsIncGen
  
  LINK_LIBS
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRSymbolInterfaces
  MLIRPass
  MLIRDialectConversion
  MLIRSIRDialect
)

# Install the libraries
install(TARGETS MLIROraDialectC
  DESTINATION lib
  EXPORT OraDialectTargets)

install(FILES
  include/OraDialect.h
  include/OraDialectC.h
  include/OraGasConstants.h
  lowering/OraToSIR/OraToSIR.h
  lowering/OraToSIR/OraToSIRTypeConverter.h
  ${PROJECT_BINARY_DIR}/OraDialect.h.inc
  ${PROJECT_BINARY_DIR}/OraTypes.h.inc
  ${PROJECT_BINARY_DIR}/OraOps.h.inc
  DESTINATION include/Ora)
