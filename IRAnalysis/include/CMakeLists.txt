set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --src-root
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --includedir
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)

include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(CIR)
add_subdirectory(SysIR)