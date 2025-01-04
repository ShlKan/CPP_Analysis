

#include "SysGenModule.h"
#include "CIR/Dialect/IR/CIRDialect.h"
#include "CIR/Dialect/IR/CIROps.h.inc"
#include "CIR/Dialect/IR/CIRTypes.h"
#include "CIRGenBuilder.h"
#include "SysGenProcess.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <string>
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

  for (const auto &field : moduleDecl->fields()) {
    auto ty = astCtx.getCanonicalType(field->getType());
    switch (field->getType().getTypePtr()->getTypeClass()) {
    case clang::Type::Builtin: {
      switch (llvm::dyn_cast<clang::BuiltinType>(ty)->getKind()) {
      case clang::BuiltinType::Int: {
        mlir::cir::IntType i32Ty =
            mlir::cir::IntType::get(builder.getContext(), 32, true);
        builder.create<mlir::cir::ConstantOp>(
            getLoc(field->getLocation()), mlir::cir::IntAttr::get(i32Ty, 2));
        break;
      }
      default:
        llvm_unreachable("Unsupport builtin type.");
      }
      break;
    }
    default:
      llvm_unreachable("Unsupport type.");
      break;
    }
  }

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