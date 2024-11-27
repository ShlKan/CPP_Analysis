
#include "CPPFrontend/CIROptions.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include <clang/Basic/LLVM.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>

#include <cstring>
#include <iostream>
#include <memory>

int main(int argc, char **argv) {

  // Firstly, filter args specific to CPP_Analysis

  std::unique_ptr<cir::CIROptions> cirOpts =
      std::make_unique<cir::CIROptions>();
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-cir"))
      cirOpts->UseClangIRPipeline = 1;
    if (!strcmp(argv[i], "-sysir")) {
      cirOpts->SysIRPipeline = 1;
    }
  }

  llvm::SmallVector<const char *, 256> Args(argv, argv + argc);
  auto Argv = llvm::ArrayRef(Args).slice(1);

  std::unique_ptr<clang::CompilerInstance> Clang(new clang::CompilerInstance());
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(
      new clang::DiagnosticIDs());

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts =
      new clang::DiagnosticOptions();
  clang::TextDiagnosticBuffer *DiagsBuffer = new clang::TextDiagnosticBuffer;
  clang::DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  // Setup round-trip remarks for the DiagnosticsEngine used in CreateFromArgs.
  if (find(Argv, llvm::StringRef("-Rround-trip-cc1-args")) != Argv.end())
    Diags.setSeverity(clang::diag::remark_cc1_round_trip_generated,
                      clang::diag::Severity::Remark, {});

  bool Success = clang::CompilerInvocation::CreateFromArgs(
      Clang->getInvocation(), Argv, Diags);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics();
  if (!Clang->hasDiagnostics())
    return 1;

  cir::ExecuteCompilerInvocation(Clang.get(), cirOpts);

  return 0;
}