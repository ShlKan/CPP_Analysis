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

std::optional<uint32_t> SysMatcher::matchSysInt(const clang::QualType &type,
                                                clang::ASTContext &ctx) {
  /* This is the pattern that matches the signed integer in SystemC. */
  auto matchResult = match(sysSigIntPattern, type, ctx);
  if (matchResult.empty())
    return std::nullopt;

  auto size = matchResult.front().getNodeAs<clang::ConstantExpr>("size");
  return size->getResultAsAPSInt().getZExtValue();
}

std::optional<clang::BuiltinType::Kind>
SysMatcher::matchBuiltinInt(const clang::QualType &type,
                            clang::ASTContext &ctx) {
  auto matchResult = match(cppBuiltinPattern, type, ctx);
  if (matchResult.empty())
    return std::nullopt;
  auto builtinTy = matchResult.front().getNodeAs<clang::BuiltinType>("builtin");
  return builtinTy->getKind();
}

llvm::APInt SysMatcher::matchFieldInitAPInt(const clang::FieldDecl &fieldDecl,
                                            clang::ASTContext &ctx) {
  auto matchResult = match(fieldInitPattern, fieldDecl, ctx);
  if (matchResult.empty())
    return llvm::APInt::getZero(32);
  return matchResult.front()
      .getNodeAs<clang::IntegerLiteral>("init")
      ->getValue();
}

} // namespace sys