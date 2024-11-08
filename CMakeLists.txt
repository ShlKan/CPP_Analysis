cmake_minimum_required(VERSION 3.24.4)
project(CPPAnalysis)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(CLANG_TABLEGEN_EXE true)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to compile")
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)


message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddClang)
include(AddMLIR)




add_subdirectory(IRAnalysis)


