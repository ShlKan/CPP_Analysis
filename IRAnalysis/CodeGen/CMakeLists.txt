set(
  LLVM_LINK_COMPONENTS
  Core
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIR
  CIRAsm.cpp
  CIRGenAtomic.cpp
  CIRGenBuilder.cpp
  CIRGenBuiltin.cpp
  CIRGenBuiltinAArch64.cpp
  CIRGenBuiltinX86.cpp
  CIRGenCXX.cpp
  CIRGenCXXABI.cpp
  CIRGenCall.cpp
  CIRGenClass.cpp
  CIRGenCleanup.cpp
  CIRGenCoroutine.cpp
  CIRGenDecl.cpp
  CIRGenDeclCXX.cpp
  CIRGenException.cpp
  CIRGenExpr.cpp
  CIRGenExprComplex.cpp
  CIRGenExprConst.cpp
  CIRGenExprAgg.cpp
  CIRGenExprCXX.cpp
  CIRGenExprScalar.cpp
  CIRGenFunction.cpp
  CIRGenItaniumCXXABI.cpp
  CIRGenModule.cpp
  CIRGenOpenCLRuntime.cpp
  CIRGenOpenCL.cpp
  CIRGenOpenMPRuntime.cpp
  CIRGenStmt.cpp
  CIRGenStmtOpenMP.cpp
  CIRGenTBAA.cpp
  CIRGenTypes.cpp
  CIRGenVTables.cpp
  CIRGenerator.cpp
  SysIRGenerator.cpp
  SysGenModule.cpp
  SysGenProcess.cpp
  SysGenDecl.cpp
  SysMatcher.cpp
  SysGenExpr.cpp
  SysGenStmt.cpp
  CIRPasses.cpp
  CIRRecordLayoutBuilder.cpp
  ConstantInitBuilder.cpp
  TargetInfo.cpp

  DEPENDS
  MLIRCIR
  MLIRSYSIR
  MLIRCIROpsIncGen
  MLIRSysOpsIncGen
  MLIRSysAttrGen
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen
  ${dialect_libs}

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  clangTooling
  ${dialect_libs}
  MLIRCIR
  MLIRSYSIR
  MLIRCIRTransforms
  MLIRCIRInterfaces
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRDLTIDialect
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRParser
  MLIRPass
  MLIRSCFToControlFlow
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTargetLLVMIRImport
  MLIRTargetLLVMIRExport
  MLIRTransforms
)
