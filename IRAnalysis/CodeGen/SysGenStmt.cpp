#include "SysGenModule.h"
#include "SysGenProcess.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace sys {

void SysGenProcess::buildStmt(mlir::Region &parent, clang::Stmt *stmt) {
  switch (stmt->getStmtClass()) {
  case Stmt::StmtClass::CompoundStmtClass:
    buildCompoundStmt(parent, llvm::cast<CompoundStmt>(stmt));
    break;
  case Stmt::StmtClass::DeclStmtClass:
    buildDeclStmt(llvm::cast<DeclStmt>(stmt));
    break;
  default:
    llvm_unreachable("Unsupported Stmt.");
  }
}

void SysGenProcess::buildCompoundStmt(mlir::Region &parent,
                                      clang::CompoundStmt *stmt) {

  auto savedPoint = builder.saveInsertionPoint();
  auto block = builder.createBlock(&parent);
  builder.setInsertionPointToStart(block);
  for (const auto subStmt : stmt->body()) {
    buildStmt(parent, subStmt);
  }
  builder.restoreInsertionPoint(savedPoint);
}

void SysGenProcess::buildDeclStmt(clang::DeclStmt *stmt) {
  for (const auto &decl : stmt->decls()) {
    if (auto varDecl = llvm::cast<clang::VarDecl>(decl))
      buildVarDecl(varDecl);
  }
}

void SysGenProcess::buildVarDecl(clang::VarDecl *varDecl) {
  auto varName = varDecl->getDeclName().getAsString();
  auto init = varDecl->getInit();
  auto type = varDecl->getType();
  if (varDecl->getType()->isBuiltinType()) {
    auto builtinTy = llvm::cast<clang::BuiltinType>(type);
    switch (builtinTy->getKind()) {
    case clang::BuiltinType::Int: {
      auto expr = SGM.buildExpr(init, SGM.getModule());
      mlir::SymbolTable::setSymbolName(expr.getDefiningOp(), varName);
      break;
    }
    default:
      break;
    }
  }
}

} // namespace sys
