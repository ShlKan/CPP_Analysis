#include "CIR/Dialect/IR/CIRDialect.h"
#include "CIR/Dialect/IR/CIRTypes.h"
#include "SysGenModule.h"
#include "SysGenProcess.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DXILABI.h"
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
  case Stmt::StmtClass::IfStmtClass:
    buildIfstmt(llvm::cast<IfStmt>(stmt));
    break;
  default:
    llvm_unreachable("Unsupported Stmt.");
  }
}

mlir::Block *SysGenProcess::buildCompoundStmt(mlir::Region &parent,
                                              clang::CompoundStmt *stmt) {
  llvm::ScopedHashTableScope<const clang::Decl *, mlir::Value> varScope(
      symbolTable);
  auto savedPoint = builder.saveInsertionPoint();
  auto block = builder.createBlock(&parent);
  builder.setInsertionPointToStart(block);
  for (const auto subStmt : stmt->body()) {
    buildStmt(parent, subStmt);
  }
  builder.restoreInsertionPoint(savedPoint);
  return block;
}

void SysGenProcess::buildDeclStmt(clang::DeclStmt *stmt) {
  for (const auto &decl : stmt->decls()) {
    if (auto varDecl = llvm::cast<clang::VarDecl>(decl))
      buildVarDecl(varDecl);
  }
}

void SysGenProcess::buildVarDecl(clang::VarDecl *varDecl) {
  auto varName = varDecl->getDeclName();
  auto init = varDecl->getInit();
  auto type = varDecl->getType();
  if (varDecl->getType()->isBuiltinType()) {
    auto builtinTy = llvm::cast<clang::BuiltinType>(type);
    switch (builtinTy->getKind()) {
    case clang::BuiltinType::Int: {
      auto expr = SGM.buildExpr(init, SGM.getModule(), symbolTable);
      symbolTable.insert(varDecl, expr);
      break;
    }
    default:
      break;
    }
  } else {
    auto expr = SGM.buildExpr(init, SGM.getModule(), symbolTable);
    symbolTable.insert(varDecl, expr);
  }
}

void SysGenProcess::buildIfstmt(clang::IfStmt *ifStmt) {
  // ifStmt->getCond()->dumpColor();
  auto cond = SGM.buildExpr(ifStmt->getCond(), SGM.getModule(), symbolTable);
  auto ifLoc = SGM.getLoc(ifStmt->getIfLoc());
  auto brCondLoc = builder.saveInsertionPoint();
  mlir::Block *trueBlock, *falseBlock;
  if (ifStmt->getThen()->getStmtClass() !=
      clang::Stmt::StmtClass::CompoundStmtClass) {
    llvm::ScopedHashTableScope<const clang::Decl *, mlir::Value> varScope(
        symbolTable);
    auto currentBlk = builder.getBlock();
    trueBlock = builder.createBlock(builder.getBlock()->getParent());
    builder.setInsertionPointToStart(trueBlock);
    auto parent = builder.getBlock()->getParent();
    buildStmt(*parent, ifStmt->getThen());
    builder.setInsertionPointToEnd(currentBlk);
  } else {
    auto parent = builder.getBlock()->getParent();
    trueBlock = buildCompoundStmt(
        *parent, llvm::dyn_cast<clang::CompoundStmt>(ifStmt->getThen()));
  }

  if (!ifStmt->getElse() || ifStmt->getElse()->getStmtClass() !=
                                clang::Stmt::StmtClass::CompoundStmtClass) {
    falseBlock = builder.createBlock(builder.getBlock()->getParent());
    if (ifStmt->getElse()) {
      llvm::ScopedHashTableScope<const clang::Decl *, mlir::Value> varScope(
          symbolTable);
      auto currentBlk = builder.getBlock();
      builder.setInsertionPointToStart(falseBlock);
      auto parent = builder.getBlock()->getParent();
      buildStmt(*parent, ifStmt->getElse());
      builder.setInsertionPointToEnd(currentBlk);
    }
  } else {
    auto parent = builder.getBlock()->getParent();
    falseBlock = buildCompoundStmt(
        *parent, llvm::dyn_cast<clang::CompoundStmt>(ifStmt->getElse()));
  }

  builder.restoreInsertionPoint(brCondLoc);
  builder.create<mlir::cir::BrCondOp>(ifLoc, cond, mlir::ValueRange{},
                                      mlir::ValueRange{}, trueBlock,
                                      falseBlock);
}

} // namespace sys
