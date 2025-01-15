

#include "SysGenModule.h"
#include "CIR/Dialect/IR/CIRDialect.h"
#include "CIR/Dialect/IR/CIROps.h.inc"
#include "CIR/Dialect/IR/CIRTypes.h"
#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "SysGenProcess.h"
#include "SysIR/Dialect/IR/SysAttrs.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace sys {

SysGenModule::SysGenModule(mlir::MLIRContext &context,
                           clang::ASTContext &astCtx,
                           const clang::CodeGenOptions &codeGenOpts,
                           const cir::CIROptions &CIROptions,
                           clang::DiagnosticsEngine &diags)
    : cir::CIRGenModule(context, astCtx, codeGenOpts, CIROptions, diags) {}

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

mlir::sys::ConstantOp SysGenModule::getConstInt(mlir::Location loc,
                                                mlir::Type ty, uint64_t val) {
  return builder.create<mlir::sys::ConstantOp>(
      loc, ty, mlir::sys::IntAttr::get(ty, val));
}

void SysGenModule::buildSysModule(clang::CXXRecordDecl *moduleDecl) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  theModule.setName(moduleDecl->getDeclName().getAsString());

  SysGenProcess genProcess(*this, builder);
  builder.setInsertionPointToEnd(theModule.getBody());

  for (const auto &field : moduleDecl->fields()) {
    switch (field->getType().getTypePtr()->getTypeClass()) {
    case clang::Type::Builtin: {
      auto val = field->getInClassInitializer()->EvaluateKnownConstInt(
          moduleDecl->getASTContext());
      auto ty = astCtx.getCanonicalType(field->getType());
      switch (llvm::dyn_cast<clang::BuiltinType>(ty)->getKind()) {
      case clang::BuiltinType::Int: {
        builder.getConstInt(getLoc(field->getLocation()), val);
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

mlir::sys::SIntType SysGenModule::getSSignedIntType(uint32_t size) {
  if (sSignedIntTyMap.count(size))
    return sSignedIntTyMap[size];
  auto ty = mlir::sys::SIntType::get(builder.getContext(), size, true);
  sSignedIntTyMap[size] = ty;
  return ty;
}

mlir::sys::SIntType SysGenModule::getSUSignedIntType(uint32_t size) {
  if (sUnsigendIntTyMap.count(size))
    return sUnsigendIntTyMap[size];
  auto ty = mlir::sys::SIntType::get(builder.getContext(), size, false);
  sUnsigendIntTyMap[size] = ty;
  return ty;
}

SysGenModule::~SysGenModule() {}

} // namespace sys