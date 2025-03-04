# Find LLVM package
find_package(LLVM REQUIRED CONFIG)

# Include LLVM CMake utilities
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# Add the test suite
add_lit_testsuite(IRAnalysisTests ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${LLVM_LIT}
    ARGS ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_custom_target(check
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/lit-run.py "${CMAKE_CURRENT_BINARY_DIR}" -v
    DEPENDS cpp_analysis FileCheck)