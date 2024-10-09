add_executable(cpp_analysis main.cpp ../CPPFrontend/CPPFrontend.cpp)

target_link_libraries(cpp_analysis
  PRIVATE
  CPPFrontend
  clangAST
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  clangCIRFrontendAction
  )