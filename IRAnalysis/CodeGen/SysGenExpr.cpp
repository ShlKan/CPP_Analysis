
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
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/OperatorKinds.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <optional>
#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "CIRGenFunction.h"
#include "mlir/IR/Value.h"

namespace cir {
class CIRGenFunction;
}
namespace sys {

class ExprBuilder : public StmtVisitor<ExprBuilder, mlir::Value> {
private:
  cir::CIRGenBuilderTy &builder;
  SysGenModule &SGM;
  mlir::Operation *context;
  llvm::ScopedHashTable<const clang::Decl *, mlir::Value> &symTable;

public:
  ExprBuilder(cir::CIRGenBuilderTy &builder, SysGenModule &SGM,
              mlir::Operation *context,
              llvm::ScopedHashTable<const clang::Decl *, mlir::Value> &symTable)
      : builder(builder), SGM(SGM), context(context), symTable(symTable) {}
  mlir::Value VisitIntegerLiteral(IntegerLiteral *intLit) {
    return builder.getConstInt(SGM.getLoc(intLit->getExprLoc()),
                               intLit->getValue());
  }

  mlir::Value VisitBinAssign(BinaryOperator *binExpr) {
    auto rhsOp = Visit(binExpr->getRHS());
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

  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *implicitExpr) {
    if (auto cxxConstructExpr = llvm::dyn_cast_or_null<CXXConstructExpr>(
            implicitExpr->getSubExpr())) {
      if (auto sizeOpt =
              SGM.getSysMatcher()->matchSysInt2(cxxConstructExpr->getType());
          sizeOpt.has_value()) {
        if (auto initVal = llvm::dyn_cast_or_null<IntegerLiteral>(
                cxxConstructExpr->getArg(0));
            initVal) {
          auto val = initVal->getValue();
          return SGM.getConstSysInt(SGM.getLoc(implicitExpr->getExprLoc()),
                                    SGM.getSSignedIntType(sizeOpt.value()),
                                    val);
        } else {
          // The initializer can also be an non-integer literal expression.
          return Visit(cxxConstructExpr->getArg(0));
        }
      }

      if (auto sizeOpt =
              SGM.getSysMatcher()->matchBitVecTy(cxxConstructExpr->getType());
          sizeOpt.has_value()) {
        auto implicitCastExpr = llvm::dyn_cast_or_null<clang::ImplicitCastExpr>(
            cxxConstructExpr->getArg(0));
        if (implicitCastExpr &&
            llvm::isa<StringLiteral>(implicitCastExpr->getSubExpr())) {
          auto strLit = llvm::dyn_cast_or_null<StringLiteral>(
              implicitCastExpr->getSubExpr());
          auto str = strLit->getString();
          return SGM.getConstSysBV(SGM.getLoc(implicitExpr->getExprLoc()),
                                   SGM.getBitVecType(sizeOpt.value()), str);
        } else {
          // The initializer can also be an non-integer literal expression.
          return Visit(cxxConstructExpr->getArg(0));
        }
      }
    }
    return Visit(implicitExpr->getSubExpr());
  }

  std::optional<mlir::sys::CmpOpKind>
  getBinOpKind(CXXOperatorCallExpr *cxxOpCallExpr) {
    if (cxxOpCallExpr->isComparisonOp(
            clang::OverloadedOperatorKind::OO_Greater)) {
      return mlir::sys::CmpOpKind::gt;
    } else if (cxxOpCallExpr->isComparisonOp(
                   clang::OverloadedOperatorKind::OO_Less)) {
      return mlir::sys::CmpOpKind::lt;
    } else if (cxxOpCallExpr->isComparisonOp(
                   clang::OverloadedOperatorKind::OO_GreaterEqual)) {
      return mlir::sys::CmpOpKind::ge;
    } else if (cxxOpCallExpr->isComparisonOp(
                   clang::OverloadedOperatorKind::OO_LessEqual)) {
      return mlir::sys::CmpOpKind::le;
    } else if (cxxOpCallExpr->isComparisonOp(
                   clang::OverloadedOperatorKind::OO_Equal)) {
      return mlir::sys::CmpOpKind::eq;
    } else {
      return {};
    }
  }

  mlir::Value VisitCXXOperatorCallExpr(CXXOperatorCallExpr *cxxOpCallExpr) {
    auto lhsOp = Visit(cxxOpCallExpr->getArg(0));
    auto rhsOp = Visit(cxxOpCallExpr->getArg(1));

    if (auto sCmpOpKind = getBinOpKind(cxxOpCallExpr); sCmpOpKind.has_value()) {
      auto boolTy = mlir::cir::BoolType::get(builder.getContext());
      return builder.create<mlir::sys::CmpOp>(
          SGM.getLoc(cxxOpCallExpr->getExprLoc()), boolTy, sCmpOpKind.value(),
          lhsOp, rhsOp);
    }
    llvm_unreachable("Unsupported CXXOperatorCallExpr.");
  }

  mlir::Value VisitDeclRefExpr(DeclRefExpr *declRefExpr) {
    if (symTable.count(declRefExpr->getDecl()))
      return symTable.lookup(declRefExpr->getDecl());
    auto varName = declRefExpr->getDecl()->getDeclName();
    if (auto op =
            mlir::SymbolTable::lookupSymbolIn(context, varName.getAsString())) {
      return op->getResults().front();
    }
    llvm_unreachable("The variable is not found");
  }

  mlir::Value VisitCXXMemberCallExpr(CXXMemberCallExpr *memCallExpr) {
    // It checks if the member call is of the form s.uint64_t(...), where
    // s is a sc_base_int integer.
    if (auto callee = memCallExpr->getImplicitObjectArgument();
        callee && SGM.getSysMatcher()->matchSCIntBase(callee->getType())) {
      if (memCallExpr->getMethodDecl()->getDeclName().getAsString() ==
          "operator long long") {
        return Visit(memCallExpr->getImplicitObjectArgument());
      }
    }
    llvm_unreachable("Unsupported CXXMemberCallExpr.");
  }

  mlir::Value VisitMemberExpr(MemberExpr *memExpr) {
    auto varName = memExpr->getMemberDecl()->getDeclName();
    return mlir::SymbolTable::lookupSymbolIn(context, varName.getAsString())
        ->getResults()
        .front();
  }

  mlir::Value VisitBinaryOperator(BinaryOperator *binExpr) {
    auto lhsOp = Visit(binExpr->getLHS());
    auto rhsOp = Visit(binExpr->getRHS());
    bool bothSysOperands = llvm::isa<mlir::sys::SIntType>(lhsOp.getType()) ||
                           llvm::isa<mlir::sys::SIntType>(rhsOp.getType());

    if (bothSysOperands) {
      mlir::sys::SBinOpKind sBinOpKind;
      mlir::sys::CmpOpKind sCmpOpKind;
      auto loc = SGM.getLoc((binExpr->getExprLoc()));
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
        cCmpOpKind = mlir::cir ::CmpOpKind::ge;
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
      };
      if (isBinOp) {
        return builder.createBinop(lhsOp, cBinOpKind, rhsOp);
      } else {
        auto loc = SGM.getLoc((binExpr->getExprLoc()));
        auto boolTy = mlir::cir::BoolType::get(builder.getContext());
        return builder.create<mlir::cir::CmpOp>(loc, boolTy, cCmpOpKind, lhsOp,
                                                rhsOp);
      }
    }
    llvm_unreachable("Unsupported Binary Operator.");
  }
};

mlir::Value SysGenModule::buildExpr(
    clang::Expr *expr, mlir::Operation *context,
    llvm::ScopedHashTable<const clang::Decl *, mlir::Value> &symTable) {
  ExprBuilder exprBuilder(builder, *this, context, symTable);
  return exprBuilder.Visit(expr);
}

} // namespace sys
