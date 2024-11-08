
add_clang_library(MLIRSYSIR
  SysTypes.cpp

  DEPENDS
  MLIRSysOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDLTIDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLoopLikeInterface
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  clangAST
  )