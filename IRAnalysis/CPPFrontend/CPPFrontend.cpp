
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <CIRFrontendAction/CIRGenAction.h>
#include <CPPFrontend/CIROptions.h>

#include <iostream>
#include <memory>

namespace cir {

bool ExecuteCompilerInvocation(clang::CompilerInstance *Clang,
                               std::unique_ptr<cir::CIROptions> &cirOpts) {

  if (cirOpts->UseClangIRPipeline) {
    auto cirAction = std::make_unique<cir::EmitCIRAction>();
    if (!cirAction)
      return false;
    cirAction->setCIROption(std::move(cirOpts));
    bool Success = Clang->ExecuteAction(*cirAction);
    return Success;
  } else {
    llvm_unreachable("Only support emit CIR Action");
  }

  return true;
}
} // namespace cir