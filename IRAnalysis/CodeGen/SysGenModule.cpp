//===--- SysIR Module generation ---------===//

#include "SysGenModule.h"
#include "cir/Dialect/IR/CIROps.h.inc"
#include "mlir/IR/BuiltinOps.h"

namespace sys {

SysGenModule::SysGenModule(mlir::MLIRContext &context,
                           clang::ASTContext &astCtx,
                           const cir::CIROptions &CIROptions,
                           clang::DiagnosticsEngine &diags)
    : builder(&context), cirOptions(CIROptions), diags(diags), astCtx(astCtx) {}

void SysGenModule::buildSysModule(clang::CXXRecordDecl *moduleDecl) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  theModule.setName(moduleDecl->getDeclName().getAsString());
  theModule.dump();
}

SysGenModule::~SysGenModule() {}

} // namespace sys