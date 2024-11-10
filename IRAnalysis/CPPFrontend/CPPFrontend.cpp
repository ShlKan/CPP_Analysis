
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <CIRFrontendAction/CIRGenAction.h>
#include <CPPFrontend/CIROptions.h>
#include <SysIRFrontendAction/SysGenAction.h>

#include <iostream>
#include <memory>

namespace cir {

bool ExecuteCompilerInvocation(clang::CompilerInstance *Clang,
                               std::unique_ptr<cir::CIROptions> &cirOpts) {

  if (cirOpts->SysIRPipeline) {
    auto sysAction = std::make_unique<sys::EmitSysGenAction>();
    if (!sysAction)
      return false;
    bool Success = Clang->ExecuteAction(*sysAction);
    return Success;
  }
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