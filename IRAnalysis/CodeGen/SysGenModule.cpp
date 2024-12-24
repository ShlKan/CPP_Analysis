//===--- SysIR Module generation ---------===//

#include "SysGenModule.h"
#include "cir/Dialect/IR/CIROps.h.inc"
#include "mlir/IR/BuiltinOps.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
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
      for (const auto &childStmt : stmt->children()) {
        auto memCall = llvm::cast<clang::CXXMemberCallExpr>(childStmt);
        if (!memCall)
          continue;
        // When this method in the constructor is a declare process statement,
        // We register this process in the module.
        if (memCall->getMethodDecl()->getNameAsString() !=
            "declare_thread_process")
          continue;
        // declare process statement's first non-object (second) argument is the
        // process name.
        auto procVar = llvm::cast<clang::ImplicitCastExpr>(memCall->getArg(1));
        auto procName = llvm::cast<clang::StringLiteral>(procVar->getSubExpr());
        processNames.push_back(procName);
      }
    }
  }
}

SysGenModule::~SysGenModule() {}

} // namespace sys