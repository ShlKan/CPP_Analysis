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
   * Match a FieldDecl's initial value.
   */
  std::optional<llvm::APInt> matchFieldInitAPInt(const clang::Expr &expr);

  /*
   * Match a statement expression.
   */
  std::optional<const clang::MemberExpr *>
  matchMemExpr(const clang::Stmt *stmt);

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

  /* The pattern of declaration  */
  const StatementMatcher fieldInitPattern = implicitCastExpr(hasDescendant(
      cxxConstructExpr(hasDescendant(integerLiteral().bind("init")))));

  /* ImplicitCastExpr  */
  const StatementMatcher implicitCasterPattern =
      findAll(memberExpr(findAll(memberExpr().bind("memExpr"))));
};
} // namespace sys

#endif