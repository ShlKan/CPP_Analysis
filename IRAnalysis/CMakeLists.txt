include_directories(${MLIR_INCLUDE_DIRS})

# Report use of deprecated APIs as errors.
# TODO: Consider adding `/we4996` for MSVC when upstream MLIR resolves
# https://github.com/llvm/llvm-project/issues/65255.
if(NOT MSVC)
  add_compile_options("-Werror=deprecated-declarations")
endif()

include_directories(include)
add_subdirectory(include)
include_directories(${CMAKE_BINARY_DIR}/IRAnalysis/include)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
add_subdirectory(Dialect)
add_subdirectory(CodeGen)
add_subdirectory(FrontendAction)
add_subdirectory(Lowering)
add_subdirectory(Interfaces)
add_subdirectory(CPPFrontend)
add_subdirectory(Driver)
add_subdirectory(test)
