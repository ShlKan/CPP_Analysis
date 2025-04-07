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

std::optional<uint32_t> SysMatcher::matchBitVecTy(const clang::QualType &type,
                                                  const std::string &s) {
  auto matchResult = match(sysBitVecPattern(s), type, astCtx);
  if (matchResult.empty())
    return std::nullopt;

  auto size = matchResult.front().getNodeAs<clang::ConstantExpr>("size");
  return size->getResultAsAPSInt().getZExtValue();
}

bool SysMatcher::matchSCLogicTy(const clang::QualType &type) {
  auto matchResult = match(sysSCLogicPattern, type, astCtx);
  return !matchResult.empty();
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

std::optional<std::vector<const clang::DeclRefExpr *>>
SysMatcher::matchBitVecOp(const clang::Stmt &stmt, const std::string &s) {
  auto matchResult = match(bitVecBopPattern(s), stmt, astCtx);
  if (matchResult.empty())
    return std::nullopt;
  std::vector<const clang::DeclRefExpr *> declRefs;
  declRefs.push_back(matchResult.front().getNodeAs<clang::DeclRefExpr>("lhs"));
  declRefs.push_back(matchResult.front().getNodeAs<clang::DeclRefExpr>("rhs"));
  return declRefs;
}

} // namespace sys