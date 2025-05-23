
#include "CIR/Dialect/IR/CIRDialect.h"
#include "CIR/Dialect/IR/CIROpsEnums.h"
#include "CIR/Dialect/IR/CIRTypes.h"
#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "SysGenModule.h"
#include "SysIR/Dialect/IR/SysAttrs.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "CIRGenFunction.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace cir {
class CIRGenFunction;
}
namespace sys {

std::unordered_map<std::string, mlir::sys::SUnaryOpKind> unaryOpMap = {
    {"and_reduce", mlir::sys::SUnaryOpKind::AndRed},
    {"or_reduce", mlir::sys::SUnaryOpKind::OrRed},
    {"xor_reduce", mlir::sys::SUnaryOpKind::XorRed},
    {"nand_reduce", mlir::sys::SUnaryOpKind::NandRed},
    {"nor_reduce", mlir::sys::SUnaryOpKind::NorRed},
    {"xnor_reduce", mlir::sys::SUnaryOpKind::XnorRed}};

auto bvOps = std::unordered_map<std::string, mlir::sys::SBinOpKind>{
    {"&", mlir::sys::SBinOpKind::SAnd},
    {"|", mlir::sys::SBinOpKind::SOr},
    {"^", mlir::sys::SBinOpKind::SXor},
    {">>", mlir::sys::SBinOpKind::SShiftR},
    {"<<", mlir::sys::SBinOpKind::SShiftL},
};

auto timeUnitTBL = std::unordered_map<std::string, mlir::sys::STimeKind>{
    {"SC_FS", mlir::sys::STimeKind::SC_FS},
    {"SC_MS", mlir::sys::STimeKind::SC_MS},
    {"SC_NS", mlir::sys::STimeKind::SC_NS},
    {"SC_PS", mlir::sys::STimeKind::SC_PS},
    {"SC_SEC", mlir::sys::STimeKind::SC_SEC},
    {"SC_US", mlir::sys::STimeKind::SC_US}};

class ExprBuilder : public StmtVisitor<ExprBuilder, mlir::Value> {
private:
  cir::CIRGenBuilderTy &builder;
  SysGenModule &SGM;
  mlir::Operation *context;
  llvm::ScopedHashTable<const clang::Decl *, mlir::Value> &symTable;
  mlir::Type &type;

public:
  ExprBuilder(cir::CIRGenBuilderTy &builder, SysGenModule &SGM,
              mlir::Operation *context,
              llvm::ScopedHashTable<const clang::Decl *, mlir::Value> &symTable,
              mlir::Type &type)
      : builder(builder), SGM(SGM), context(context), symTable(symTable),
        type(type) {}

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

      if (auto sizeOpt = SGM.getSysMatcher()->matchBitVecTy(
              cxxConstructExpr->getType(), "sc_bv");
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

      if (auto sizeOpt = SGM.getSysMatcher()->matchBitVecTy(
              cxxConstructExpr->getType(), "sc_lv");
          sizeOpt.has_value()) {
        auto implicitCastExpr = llvm::dyn_cast_or_null<clang::ImplicitCastExpr>(
            cxxConstructExpr->getArg(0));
        if (implicitCastExpr &&
            llvm::isa<StringLiteral>(implicitCastExpr->getSubExpr())) {
          auto strLit = llvm::dyn_cast_or_null<StringLiteral>(
              implicitCastExpr->getSubExpr());
          auto str = strLit->getString();
          return SGM.getConstSysBVL(SGM.getLoc(implicitExpr->getExprLoc()),
                                    SGM.getBitVecLType(sizeOpt.value()), str);
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

  mlir::Value VisitCXXConstructExpr(CXXConstructExpr *cxxConstructExpr) {
    if (SGM.getSysMatcher()->matchSCLogicTy(cxxConstructExpr->getType())) {
      auto declRefExpr = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
          cxxConstructExpr->getArg(0));
      auto val = declRefExpr->getDecl()->getNameAsString();
      llvm::StringRef valStr("");
      if (val == "SC_LOGIC_Z") {
        valStr = "z";
      } else if (val == "SC_LOGIC_X") {
        valStr = "x";
      } else if (val == "SC_LOGIC_0") {
        valStr = "0";
      } else if (val == "SC_LOGIC_1") {
        valStr = "1";
      } else {
        llvm_unreachable("Unsupported SC Logic Type.");
      }
      return SGM.getConstSysBVL(SGM.getLoc(cxxConstructExpr->getExprLoc()),
                                SGM.getBitVecLType(1), valStr);
    } else if (SGM.getSysMatcher()->matchSCTimeTy(
                   cxxConstructExpr->getType())) {
      if (cxxConstructExpr->getNumArgs() == 2) {
        auto timeUnitPtr = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
            cxxConstructExpr->getArg(1));
        auto timeUnitStr = timeUnitPtr->getDecl()->getDeclName().getAsString();
        auto timeUnit = timeUnitTBL.find(timeUnitStr)->second;
        auto timeVal = llvm::dyn_cast_or_null<ImplicitCastExpr>(
            cxxConstructExpr->getArg(0));
        auto timeValInt =
            llvm::dyn_cast_or_null<IntegerLiteral>(timeVal->getSubExpr());
        auto timeValIntVal = timeValInt->getValue();
        auto timeType = mlir::sys::STimeType::get(builder.getContext());
        auto time = mlir::sys::TimeAttr::get(builder.getContext(), timeType,
                                             timeUnit, timeValIntVal);
        return builder.create<mlir::sys::ConstantOp>(
            SGM.getLoc(cxxConstructExpr->getExprLoc()), timeType, time);
      }
    }
    llvm_unreachable("Unsupported CXXConstructExpr.");
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

  mlir::Value VisitExprWithCleanups(ExprWithCleanups *exprWithClean) {
    for (const auto &[bvOpName, bvOpKind] : bvOps) {
      if (auto result =
              SGM.getSysMatcher()->matchBitVecOp(*exprWithClean, bvOpName);
          result.has_value()) {
        if (result->size() != 2) {
          llvm_unreachable("ExprWithCleanups: Two operands are expected.");
        }
        auto lhs = Visit(const_cast<clang::Expr *>(result.value()[0]));
        auto rhs = Visit(const_cast<clang::Expr *>(result.value()[1]));
        return builder.create<mlir::sys::BinOp>(
            SGM.getLoc(exprWithClean->getExprLoc()), type, bvOpKind, lhs, rhs);
      }
    }
    if (auto highLow = SGM.getSysMatcher()->matchRangeCall(*exprWithClean);
        highLow.has_value()) {
      auto declRef = highLow.value().first;
      auto caller = Visit(const_cast<clang::DeclRefExpr *>(declRef));
      auto intType = mlir::sys::SIntType::get(builder.getContext(), 32, false);
      auto high =
          mlir::sys::IntAttr::get(intType, highLow.value().second.first);
      auto low =
          mlir::sys::IntAttr::get(intType, highLow.value().second.second);
      return builder.create<mlir::sys::RangeOp>(
          SGM.getLoc(exprWithClean->getExprLoc()), type, caller, high, low);
    }

    llvm_unreachable("Unsupported ExprWithCleanups.");
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

    if (unaryOpMap.find(
            memCallExpr->getMethodDecl()->getDeclName().getAsString()) !=
        unaryOpMap.end()) {
      auto unaryOpKind =
          unaryOpMap
              .find(memCallExpr->getMethodDecl()->getDeclName().getAsString())
              ->second;
      if (auto implicitArg = llvm::dyn_cast_or_null<clang::ImplicitCastExpr>(
              memCallExpr->getImplicitObjectArgument())) {
        if (auto declRef = llvm::dyn_cast_or_null<DeclRefExpr>(
                implicitArg->getSubExpr())) {
          auto tyStr = declRef->getType().getAsString();
          mlir::Type reduceTy;
          if (tyStr.find("sc_bv") != std::string::npos) {
            reduceTy = mlir::sys::SBitVecType::get(builder.getContext(), 1);
          } else {
            reduceTy = mlir::sys::SBitVecLType::get(builder.getContext(), 1);
          }
          if (symTable.count(declRef->getDecl())) {
            return builder.create<mlir::sys::UnaryOp>(
                SGM.getLoc(memCallExpr->getExprLoc()), reduceTy, unaryOpKind,
                symTable.lookup(declRef->getDecl()));
          } else {
            auto input =
                mlir::SymbolTable::lookupSymbolIn(
                    context, declRef->getDecl()->getDeclName().getAsString())
                    ->getResults()
                    .front();
            return builder.create<mlir::sys::UnaryOp>(
                SGM.getLoc(memCallExpr->getExprLoc()), reduceTy, unaryOpKind,
                input);
          }
        }
      }
    }
    if (memCallExpr->getMethodDecl()->getDeclName().getAsString() == "wait") {
      if (auto eventPtr = memCallExpr->getArg(0)) {
        auto eventVal = Visit(eventPtr);
        builder.create<mlir::sys::WaitOp>(SGM.getLoc(memCallExpr->getExprLoc()),
                                          eventVal,
                                          mlir::sys::SEventCombKind::And);
        // TODO: It should a void value.
        return eventVal;
      }
    }

    if (memCallExpr->getMethodDecl()->getDeclName().getAsString() == "notify") {
      // TODO: Currently, only one argument is supported, i.e., e.notify(),
      // without time arguments in notify.
      auto eventVal = Visit(memCallExpr->getImplicitObjectArgument());
      eventVal.dump();
      builder.create<mlir::sys::NotifyOp>(SGM.getLoc(memCallExpr->getExprLoc()),
                                          eventVal);
      // TODO: It should a void value.
      return eventVal;
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
    llvm::ScopedHashTable<const clang::Decl *, mlir::Value> &symTable,
    mlir::Type type) {
  ExprBuilder exprBuilder(builder, *this, context, symTable, type);
  return exprBuilder.Visit(expr);
}

} // namespace sys
