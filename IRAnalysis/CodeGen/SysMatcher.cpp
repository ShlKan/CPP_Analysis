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

#include "SysMatcher.h"
#include "clang/AST/APValue.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include <optional>

namespace sys {

std::optional<uint32_t> SysMatcher::matchSysInt(const clang::QualType &type) {
  /* This is the pattern that matches the signed integer in SystemC. */
  auto matchResult = match(sysSigIntPattern, type, astCtx);
  if (matchResult.empty())
    return std::nullopt;

  auto size = matchResult.front().getNodeAs<clang::ConstantExpr>("size");
  return size->getResultAsAPSInt().getZExtValue();
}

std::optional<clang::BuiltinType::Kind>
SysMatcher::matchBuiltinInt(const clang::QualType &type) {
  auto matchResult = match(cppBuiltinPattern, type, astCtx);
  if (matchResult.empty())
    return std::nullopt;
  auto builtinTy = matchResult.front().getNodeAs<clang::BuiltinType>("builtin");
  return builtinTy->getKind();
}

std::optional<uint32_t> SysMatcher::matchSysInt2(const clang::QualType &type) {
  auto matchResult = match(sysIntTypePattern, type, astCtx);
  if (matchResult.empty())
    return std::nullopt;

  auto size = matchResult.front().getNodeAs<clang::ConstantExpr>("size");
  return size->getResultAsAPSInt().getZExtValue();
}

std::optional<llvm::APInt>
SysMatcher::matchFieldInitAPInt(const clang::Expr &expr) {
  auto matchResult = match(fieldInitPattern, expr, astCtx);
  if (matchResult.empty())
    return std::nullopt;
  return matchResult.front()
      .getNodeAs<clang::IntegerLiteral>("init")
      ->getValue();
}

std::optional<const clang::DeclRefExpr *>
SysMatcher::matchdeclRef(const clang::Stmt *stmt) {
  auto matchResult = match(declRefPattern, *stmt, astCtx);
  if (matchResult.empty())
    return std::nullopt;
  return matchResult.front().getNodeAs<clang::DeclRefExpr>("declRefExpr");
}

bool SysMatcher::matchSCIntBase(const clang::QualType &type) {
  auto matchResult = match(scBasePattern, type, astCtx);
  return !matchResult.empty();
}

} // namespace sys