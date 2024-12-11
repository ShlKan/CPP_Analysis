//===--- SysIR Module generation ---------===//

#include "SysGenModule.h"
#include "cir/Dialect/IR/CIROps.h.inc"
#include "mlir/IR/BuiltinOps.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace sys {

SysGenModule::SysGenModule(mlir::MLIRContext &context,
                           clang::ASTContext &astCtx,
                           const cir::CIROptions &CIROptions,
                           clang::DiagnosticsEngine &diags)
    : builder(&context), cirOptions(CIROptions), diags(diags), astCtx(astCtx) {}

void SysGenModule::buildSysModule(clang::CXXRecordDecl *moduleDecl) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  theModule.setName(moduleDecl->getDeclName().getAsString());
  for (auto method : moduleDecl->methods()) {
    if (llvm::isa<clang::CXXConstructorDecl>(method)) {
      auto stmt = method->getBody();
      if (!stmt)
        continue;
      for (auto childStmt : stmt->children()) {
        auto memCall = llvm::cast<clang::CXXMemberCallExpr>(childStmt);
        llvm::outs() << memCall->getCallee()->imp << "\n";
        memCall->getCallee()->getType();
        // memCall->dumpColor();
      }
    }
  }
  // theModule.dump();
}

SysGenModule::~SysGenModule() {}

} // namespace sys