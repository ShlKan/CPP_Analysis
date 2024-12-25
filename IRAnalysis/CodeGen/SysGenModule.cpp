

#include "SysGenModule.h"
#include "CIR/Dialect/IR/CIROps.h.inc"
#include "SysGenProcess.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <utility>


namespace sys {

SysGenModule::SysGenModule(mlir::MLIRContext &context,
                           clang::ASTContext &astCtx,
                           const cir::CIROptions &CIROptions,
                           clang::DiagnosticsEngine &diags)
    : builder(&context), cirOptions(CIROptions), diags(diags), astCtx(astCtx) {}

void SysGenModule::collectProcess(clang::CXXRecordDecl *moduleDecl) {
  for (const auto &method : moduleDecl->methods()) {
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

void SysGenModule::buildSysModule(clang::CXXRecordDecl *moduleDecl) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  theModule.setName(moduleDecl->getDeclName().getAsString());

  SysGenProcess genProcess(*this, builder);
  builder.setInsertionPointToEnd(theModule.getBody());

  collectProcess(moduleDecl);
  llvm::SmallVector<mlir::sys::ProcDefOP, 4> processOPs;
  for (const auto &method : moduleDecl->methods()) {
    if (std::find_if(processNames.begin(), processNames.end(),
                     [&method](clang::StringLiteral *procName) {
                       return procName->getString() ==
                              method->getDeclName().getAsString();
                     }) != processNames.end()) {
      auto proc = genProcess.buildProcess(method);
      processOPs.push_back(proc);
    }
  }
  for (const auto &procOP : processOPs)
    genProcess.buildProcessRegister(procOP);
  theModule->dump();
}

SysGenModule::~SysGenModule() {}

} // namespace sys