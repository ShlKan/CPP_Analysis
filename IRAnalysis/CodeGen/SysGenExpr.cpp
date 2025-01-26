
#include "SysGenExpr.h"
#include "CIRGenModule.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

namespace sys {

mlir::Value SysGenExpr::buildExpr(clang::Expr *expr, mlir::Operation *context) {
  // TODO This function needs a total rewrite.
  if (auto binExpr = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    return buildBinOp(binExpr, context);
  }
  if (!llvm::isa<clang::ImplicitCastExpr>(expr))
    llvm_unreachable("Unsupport expressions");

  expr = llvm::dyn_cast<clang::ImplicitCastExpr>(expr)->getSubExpr();

  if (auto implicitExpr = (llvm::dyn_cast<clang::CXXConstructExpr>(expr))) {
    for (auto &child : implicitExpr->children()) {
      return buildExpr(llvm::dyn_cast<clang::Expr>(child), context);
    }
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

mlir::Value SysGenExpr::buildBinOp(clang::BinaryOperator *binExpr,
                                   mlir::Operation *context) {
  auto lhsOp = buildExpr(binExpr->getLHS(), context);
  auto rhsOp = buildExpr(binExpr->getRHS(), context);
  bool containSysOperands = llvm::isa<mlir::sys::SIntType>(lhsOp.getType()) ||
                            llvm::isa<mlir::sys::SIntType>(rhsOp.getType());

  if (!containSysOperands) {
    llvm_unreachable("CIROp is not supported.");
  }

  mlir::sys::SBinOpKind sBinOpKind;
  switch (binExpr->getOpcode()) {
  case clang::BinaryOperatorKind::BO_Mul:
    sBinOpKind = mlir::sys::SBinOpKind::SMul;
  case clang::BinaryOperatorKind::BO_Div:
    sBinOpKind = mlir::sys::SBinOpKind::SDiv;
  case clang::BinaryOperatorKind::BO_Sub:
    sBinOpKind = mlir::sys::SBinOpKind::SSub;
  case clang::BinaryOperatorKind::BO_Add: {
    sBinOpKind = mlir::sys::SBinOpKind::SAdd;
    auto loc = theModule->getLoc((binExpr->getExprLoc()));
    return builder.create<mlir::sys::BinOp>(loc, sBinOpKind, lhsOp, rhsOp);
  }
  default:
    llvm_unreachable("Unsupported Binary Operator.");
  }
}

} // namespace sys
