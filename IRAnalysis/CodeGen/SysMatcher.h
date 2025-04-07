/*
 * Created Date: Th Jan 2025
 * Author: Shuanglong Kan
 * -----
 * Last Modified: Fri Jan 17 2025
 * Modified By: Shuanglong Kan
 * -----
 * Copyright (c) 2025 Shuanglong Kan
 * ---------------------------------------------------------
 */

#ifndef MLIR_SYS_MATCHER_H
#define MLIR_SYS_MATCHER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include <optional>

using namespace clang::ast_matchers;

namespace sys {
/*
 * This class contains the interface to match systemc elements in the clang AST.
 */
class SysMatcher {
public:
  SysMatcher(clang::ASTContext &astCtx) : astCtx(astCtx) {};

  /*
   * Match a type against a C++ builtin integer. If matched then
   * return the bitwidth of the type, otherwise return nullopt.
   */
  std::optional<clang::BuiltinType::Kind>
  matchBuiltinInt(const clang::QualType &type);

  /*
   * Match a type against a systemc integer type. If matched then
   * return the bitwidth of the type, otherwise return nullopt.
   */
  std::optional<uint32_t> matchSysInt(const clang::QualType &type);

  /*
   * Match a type against a systemc integer type. If matched then
   * return the bitwidth of the type, otherwise return nullopt.
   */
  std::optional<uint32_t> matchSysInt2(const clang::QualType &type);

  /*
   * Match a type against a systemc bit vector type. If matched then
   * return the bitwidth of the type, otherwise return nullopt.
   */
  std::optional<uint32_t> matchBitVecTy(const clang::QualType &type,
                                        const std::string &s);

  /*
   * Match a type against a systemc bit vector type. If matched then
   * return the bitwidth of the type, otherwise return nullopt.
   */
  std::optional<std::vector<const clang::DeclRefExpr *>>
  matchBitVecOp(const clang::Stmt &stmt, const std::string &s);

  /*
   * Match a type against a systemc sc_logic type. If matched then
   * return the bitwidth of the type, otherwise return nullopt.
   */
  bool matchSCLogicTy(const clang::QualType &type);

  /*
   * Match a FieldDecl's initial value.
   */
  std::optional<llvm::APInt> matchFieldInitAPInt(const clang::Expr &expr);

  /*
   * Match a statement expression.
   */
  std::optional<const clang::DeclRefExpr *>
  matchdeclRef(const clang::Stmt *stmt);

  /*
   * Match `sc_int_base` type.
   */
  bool matchSCIntBase(const clang::QualType &type);

private:
  clang::ASTContext &astCtx;
  /* The pattern of SystemC integer Type. */
  const TypeMatcher sysSigIntPattern = elaboratedType(
      hasQualifier(specifiesNamespace(hasName("sc_dt"))),
      namesType(templateSpecializationType(
          hasDeclaration(namedDecl(hasName("sc_int"))),
          hasAnyTemplateArgument(isExpr(constantExpr().bind("size"))))));

  /* The pattern of builtin integer Type. */
  const TypeMatcher cppBuiltinPattern = builtinType().bind("builtin");

  /* The pattern of builtin integer Type. */
  const TypeMatcher sysIntTypePattern =
      elaboratedType(namesType(templateSpecializationType(
          hasDeclaration(namedDecl(hasName("sc_int"))),
          hasAnyTemplateArgument(isExpr(constantExpr().bind("size"))))));

  /* The pattern of builtin bv Type. */
  TypeMatcher sysBitVecPattern(std::string s) {
    return elaboratedType(namesType(templateSpecializationType(
        hasDeclaration(namedDecl(hasName(s))),
        hasAnyTemplateArgument(isExpr(constantExpr().bind("size"))))));
  };

  /* The pattern of builtin sc_logic Type. */
  TypeMatcher sysSCLogicPattern =
      elaboratedType(namesType(hasDeclaration(namedDecl(hasName("sc_logic")))));

  /* The pattern of declaration  */
  const StatementMatcher fieldInitPattern = implicitCastExpr(hasDescendant(
      cxxConstructExpr(hasDescendant(integerLiteral().bind("init")))));

  /* ImplicitCastExpr  */
  const StatementMatcher implicitCasterPattern =
      findAll(memberExpr(findAll(memberExpr().bind("memExpr"))));

  /* declRefExpr  */
  const StatementMatcher declRefPattern =
      findAll(declRefExpr().bind("declRefExpr"));

  /* sc_base type */
  const TypeMatcher scBasePattern =
      recordType(hasDeclaration(namedDecl(hasName("sc_int_base"))));

  /* bit vector binary operator */
  const StatementMatcher bitVecBopPattern(std::string opName) {
    return findAll(cxxOperatorCallExpr(
        hasOverloadedOperatorName(opName),
        hasLHS(implicitCastExpr(hasDescendant(declRefExpr().bind("lhs")))),
        hasRHS(implicitCastExpr(hasDescendant(declRefExpr().bind("rhs"))))));
  }
};
} // namespace sys

#endif