
#include "CIR/Dialect/IR/CIRDialect.h"
#include "CIR/Dialect/IR/CIROpsEnums.h"
#include "CIR/Dialect/IR/CIRTypes.h"
#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "SysGenModule.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Basic/OperatorKinds.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

#include "CIRGenFunction.h"

namespace cir {
class CIRGenFunction;
}
namespace sys {

mlir::Value SysGenModule::buildExpr(
    clang::Expr *expr, mlir::Operation *context,
    llvm::ScopedHashTable<const clang::Decl *, mlir::Value> &symTable) {

  // TODO This function needs a total rewrite.
  if (auto binExpr = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    return buildBinOp(binExpr, context, symTable);
  }

  if (auto sizeOpt = sysMatcher->matchSysInt(expr->getType());
      sizeOpt.has_value()) {
    if (auto initVal = sysMatcher->matchFieldInitAPInt(*expr);
        initVal.has_value())
      return getConstSysInt(getLoc(expr->getExprLoc()),
                            getSSignedIntType(sizeOpt.value()),
                            initVal.value());
  }

  if (auto intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr);
      intLit && sysMatcher->matchBuiltinInt(expr->getType()).has_value()) {
    return builder.getConstInt(getLoc(expr->getExprLoc()), intLit->getValue());
  }

  if (auto cxxOpCallExpr = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    if (cxxOpCallExpr->isComparisonOp(
            clang::OverloadedOperatorKind::OO_Greater)) {
      auto lhsOp = buildExpr(cxxOpCallExpr->getArg(0), context, symTable);
      auto rhsOp = buildExpr(cxxOpCallExpr->getArg(1), context, symTable);
      auto sCmpOpKind = mlir::sys::CmpOpKind::gt;
      auto boolTy = mlir::cir::BoolType::get(builder.getContext());
      return builder.create<mlir::sys::CmpOp>(getLoc(expr->getExprLoc()),
                                              boolTy, sCmpOpKind, lhsOp, rhsOp);
    }
  }

  if (auto implicitExpr = llvm::dyn_cast_or_null<clang::ImplicitCastExpr>(expr))
    expr = implicitExpr->getSubExpr();

  if (auto implicitExpr = (llvm::dyn_cast<clang::CXXConstructExpr>(expr))) {
    for (auto &child : implicitExpr->children()) {
      return buildExpr(llvm::dyn_cast<clang::Expr>(child), context, symTable);
    }
  }
  if (auto declRefExpr = sysMatcher->matchdeclRef(expr);
      declRefExpr.has_value()) {
    auto varName = declRefExpr.value()->getDecl()->getDeclName();

    if (symTable.count(declRefExpr.value()->getDecl()))
      return symTable.lookup(declRefExpr.value()->getDecl());
    if (auto op =
            mlir::SymbolTable::lookupSymbolIn(context, varName.getAsString())) {
      return op->getResults().front();
    }
    llvm_unreachable("The variable is not found");
  }

  if (auto memExpr = sysMatcher->matchMemExpr(expr); memExpr.has_value()) {
    auto memExpr1 = llvm::dyn_cast<clang::ImplicitCastExpr>(
                        *memExpr.value()->children().begin())
                        ->getSubExpr();
    auto varName = llvm::dyn_cast<clang::MemberExpr>(memExpr1)
                       ->getMemberDecl()
                       ->getDeclName();
    return mlir::SymbolTable::lookupSymbolIn(context, varName.getAsString())
        ->getResults()
        .front();
  }
}

mlir::Value SysGenModule::buildBinOp(
    clang::BinaryOperator *binExpr, mlir::Operation *context,
    llvm::ScopedHashTable<const clang::Decl *, mlir::Value> &symTable) {

  if (binExpr->getOpcode() == clang::BinaryOperatorKind::BO_Assign) {
    auto rhsOp = buildExpr(binExpr->getRHS(), context, symTable);
    auto declRef =
        llvm::dyn_cast_or_null<clang::DeclRefExpr>(binExpr->getLHS());
    if (symTable.count(declRef->getDecl())) {
      symTable.insert(declRef->getDecl(), rhsOp);
    } else
      mlir::SymbolTable::setSymbolName(
          rhsOp.getDefiningOp(),
          declRef->getDecl()->getDeclName().getAsString());
    return rhsOp;
  }
  auto lhsOp = buildExpr(binExpr->getLHS(), context, symTable);
  auto rhsOp = buildExpr(binExpr->getRHS(), context, symTable);
  bool bothSysOperands = llvm::isa<mlir::sys::SIntType>(lhsOp.getType()) ||
                         llvm::isa<mlir::sys::SIntType>(rhsOp.getType());

  if (bothSysOperands) {
    mlir::sys::SBinOpKind sBinOpKind;
    mlir::sys::CmpOpKind sCmpOpKind;
    auto loc = getLoc((binExpr->getExprLoc()));
    switch (binExpr->getOpcode()) {
    case clang::BinaryOperatorKind::BO_Mul: {
      sBinOpKind = mlir::sys::SBinOpKind::SMul;
      break;
    }
    case clang::BinaryOperatorKind::BO_Div: {
      sBinOpKind = mlir::sys::SBinOpKind::SDiv;
      break;
    }
    case clang::BinaryOperatorKind::BO_Sub: {
      sBinOpKind = mlir::sys::SBinOpKind::SSub;
      break;
    }
    case clang::BinaryOperatorKind::BO_Add: {
      sBinOpKind = mlir::sys::SBinOpKind::SAdd;
      break;
    }
    default:
      llvm_unreachable("Unsupported Binary Operator.");
    }
    return builder.create<mlir::sys::BinOp>(loc, sBinOpKind, lhsOp, rhsOp);
  }

  bool bothCIROperands = llvm::isa<mlir::cir::IntType>(lhsOp.getType()) ||
                         llvm::isa<mlir::cir::IntType>(rhsOp.getType());
  if (bothCIROperands) {

    mlir::cir::BinOpKind cBinOpKind;
    mlir::cir::CmpOpKind cCmpOpKind;
    bool isBinOp = false;
    switch (binExpr->getOpcode()) {
    case clang::BinaryOperatorKind::BO_Add:
      isBinOp = true;
      cBinOpKind = mlir::cir::BinOpKind::Add;
      break;
    case clang::BinaryOperatorKind::BO_GE:
      isBinOp = false;
      cCmpOpKind = mlir::cir::CmpOpKind::ge;
      break;
    case clang::BinaryOperatorKind::BO_GT:
      isBinOp = false;
      cCmpOpKind = mlir::cir::CmpOpKind::gt;
      break;
    case clang::BinaryOperatorKind::BO_LE:
      isBinOp = false;
      cCmpOpKind = mlir::cir::CmpOpKind::le;
      break;
    case clang::BinaryOperatorKind::BO_LT:
      isBinOp = false;
      cCmpOpKind = mlir::cir::CmpOpKind::lt;
      break;
    default:
      llvm_unreachable("Unsupported Binary Operator.");
    }
    if (isBinOp) {
      return builder.createBinop(lhsOp, cBinOpKind, rhsOp);
    } else {
      auto loc = getLoc((binExpr->getExprLoc()));
      auto boolTy = mlir::cir::BoolType::get(builder.getContext());
      return builder.create<mlir::cir::CmpOp>(loc, boolTy, cCmpOpKind, lhsOp,
                                              rhsOp);
    }
  }
}

} // namespace sys
