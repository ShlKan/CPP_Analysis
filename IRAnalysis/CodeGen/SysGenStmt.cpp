#include "Address.h"
#include "CIR/Dialect/IR/CIRDialect.h"
#include "CIR/Dialect/IR/CIRTypes.h"
#include "CIRGenFunction.h"
#include "SysGenModule.h"
#include "SysGenProcess.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
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
    buildIfStmt(llvm::cast<IfStmt>(stmt));
    break;
  case Stmt::StmtClass::ForStmtClass:
    buildLoopStmt(llvm::cast<ForStmt>(stmt));
    break;
  case Stmt::StmtClass::BinaryOperatorClass:
    SGM.buildExpr(llvm::dyn_cast_or_null<clang::BinaryOperator>(stmt),
                  SGM.getModule(), symbolTable);
    break;
  case Stmt::StmtClass::CXXMemberCallExprClass:
    SGM.buildExpr(llvm::dyn_cast_or_null<clang::CXXMemberCallExpr>(stmt),
                  SGM.getModule(), symbolTable);
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
    mlir::Type varTy = nullptr;
    if (auto sysIntTy = SGM.getSysMatcher()->matchBitVecTy(type, "sc_bv");
        sysIntTy.has_value()) {
      varTy =
          mlir::sys::SBitVecType::get(builder.getContext(), sysIntTy.value());
    } else if (auto sysIntTy =
                   SGM.getSysMatcher()->matchBitVecTy(type, "sc_lv");
               sysIntTy.has_value()) {
      varTy =
          mlir::sys::SBitVecLType::get(builder.getContext(), sysIntTy.value());
    }
    auto expr = SGM.buildExpr(init, SGM.getModule(), symbolTable, varTy);
    symbolTable.insert(varDecl, expr);
  }
}

void SysGenProcess::buildIfStmt(clang::IfStmt *ifStmt) {
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
  builder.setInsertionPointAfter(falseBlock->getParentOp());
}

llvm::SmallVector<clang::Decl *> getRefDecls(clang::Expr *expr) {
  llvm::SmallVector<clang::Decl *> decls;
  if (auto binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    // TODO: Here only basic operation is supported.
    if (ImplicitCastExpr *implicitExpr =
            llvm::dyn_cast_or_null<ImplicitCastExpr>(binOp->getLHS())) {
      if (auto declRef =
              llvm::dyn_cast<clang::DeclRefExpr>(implicitExpr->getSubExpr())) {
        decls.push_back(declRef->getDecl());
      }
    }
    if (auto declRef = llvm::dyn_cast<clang::DeclRefExpr>(binOp->getRHS())) {
      decls.push_back(declRef->getDecl());
    }
  } else {
    llvm_unreachable("getRefDecls: Unsupported Expr.");
  }
  return decls;
}

void SysGenProcess::buildLoopStmt(clang::ForStmt *loopStmt) {
  auto parent = builder.getBlock()->getParent();
  buildStmt(*parent, loopStmt->getInit());
  auto condDeclRefs = getRefDecls(loopStmt->getCond());
  llvm::SmallVector<mlir::Value> condDeclVals;
  for (const auto &decl : condDeclRefs) {
    condDeclVals.push_back(symbolTable.lookup(decl));
  }
  mlir::ValueRange initOperands(condDeclVals);
  builder.create<mlir::sys::LoopOp>(
      SGM.getLoc(loopStmt->getForLoc()),
      mlir::cir::IntType::get(builder.getContext(), 32, true), initOperands,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        auto condVal =
            SGM.buildExpr(loopStmt->getCond(), SGM.getModule(), symbolTable);
        for (auto &declVal : condDeclVals) {
          declVal.replaceAllUsesWith(args.front());
          args.drop_front();
        }
        b.create<mlir::sys::ConditionOp>(
            SGM.getLoc(loopStmt->getCond()->getExprLoc()), condVal, args);
      },
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        auto parent = builder.getBlock()->getParent();
        if (auto compoundBlk = llvm::dyn_cast_or_null<clang::CompoundStmt>(
                loopStmt->getBody())) {
          for (const auto &stmt : compoundBlk->body()) {
            buildStmt(*parent, stmt);
          }
        } else
          buildStmt(*parent, loopStmt->getBody());
        buildStmt(*parent, loopStmt->getInc());
        llvm::SmallVector<mlir::Value> condDeclVals;
        for (const auto &decl : condDeclRefs) {
          condDeclVals.push_back(symbolTable.lookup(decl));
        }
        b.create<mlir::sys::YieldOp>(
            SGM.getLoc(loopStmt->getBody()->getEndLoc()), condDeclVals);
      });
}

} // namespace sys
