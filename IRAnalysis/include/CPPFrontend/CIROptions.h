
#ifndef LLVM_CLANG_CIR_CIROPTIONS_H
#define LLVM_CLANG_CIR_CIROPTIONS_H

#include <clang/Frontend/CompilerInstance.h>

#include <string>

namespace cir {
class CIROptions {
public:
  /// Use Sys IR pipeline to emit code
  unsigned SysIRPipeline : 1;

  /// Use Clang IR pipeline to emit code
  unsigned UseClangIRPipeline : 1;

  /// Lower directly from ClangIR to LLVM
  unsigned ClangIRDirectLowering : 1;

  /// Disable Clang IR specific (CIR) passes
  unsigned ClangIRDisablePasses : 1;

  /// Disable Clang IR (CIR) verifier
  unsigned ClangIRDisableCIRVerifier : 1;

  /// Disable ClangIR emission for CXX default (compiler generated methods).
  unsigned ClangIRDisableEmitCXXDefault : 1;

  /// Enable diagnostic verification for CIR
  unsigned ClangIRVerifyDiags : 1;

  // Enable Clang IR based lifetime check
  unsigned ClangIRLifetimeCheck : 1;

  // Enable Clang IR idiom recognizer
  unsigned ClangIRIdiomRecognizer : 1;

  // Enable Clang IR library optimizations
  unsigned ClangIRLibOpt : 1;

  // Enable Clang IR call conv lowering pass.
  unsigned ClangIREnableCallConvLowering : 1;

  // Enable Clang IR mem2reg pass on the flat CIR.
  unsigned ClangIREnableMem2Reg : 1;

  unsigned ClangIRSkipFunctionsFromSystemHeaders : 1;

  unsigned ClangIRBuildDeferredThreshold : 1;

  std::string ClangIRLifetimeCheckOpts;
  std::string ClangIRIdiomRecognizerOpts;
  std::string ClangIRLibOptOpts;
};

bool ExecuteCompilerInvocation(clang::CompilerInstance *compilerInstance,
                               std::unique_ptr<cir::CIROptions> &cirOpts);

} // namespace cir

#endif