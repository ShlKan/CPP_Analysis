//===- SysDialect.cpp - MLIR Sys ops implementation -----------------------===//

#include "SysIR/Dialect/IR/SysDialect.h"
#include "CIR/Dialect/IR/CIRTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

using namespace ::mlir;
using namespace ::mlir::sys;

#include "SysIR/Dialect/IR/SysOpsDialect.cpp.inc"
#include "SysIR/Dialect/IR/SysOpsEnums.cpp.inc"

void SysDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "SysIR/Dialect/IR/SysOps.cpp.inc"
      >();
}

//===---     ProcDefOp      ---===//

void ProcDefOP::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, StringRef name,
                      SProcessType type) {
  mlir::ArrayAttr attr;
  build(odsBuilder, odsState, type, name, type, attr);
}

//===---     ProcRegisterOp     ---====//

Attribute SysDialect::parseAttribute(mlir::DialectAsmParser &,
                                     mlir::Type) const {
  // TODO
  llvm::llvm_unreachable_internal("parseAttribute: Have not yet implemented");
}

//===---     BinOp     ---====//
LogicalResult BinOp::verify() {
  // TODO,verification should be done here.
  return mlir::success();
}

//===---     ConditionOp     ---====//
void ConditionOp::getSuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  auto loopOp = getParentOp();
  FoldAdaptor adaptor(operands, *this);

  auto boolAttr = llvm::dyn_cast_or_null<BoolAttr>(adaptor.getCondition());
  if (!boolAttr || boolAttr.getValue())
    regions.emplace_back(&loopOp.getAfter(), loopOp.getAfter().getArguments());
  if (!boolAttr || !boolAttr.getValue())
    regions.emplace_back(loopOp->getResults());
}

MutableOperandRange
ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) {
  assert((point.isParent()) || point == getParentOp().getAfter());
  return getArgsMutable();
}

//===---     UnaryOp     ---====//
LogicalResult UnaryOp::verify() {
  // TODO,verification should be done here.
  return mlir::success();
}

//===---     LoopOp     ---====//
llvm::SmallVector<Region *> LoopOp::getLoopRegions() {
  return {&getBefore(), &getAfter()};
}

Block::BlockArgListType LoopOp::getRegionIterArgs() {
  return getBeforeArguments();
}

Block::BlockArgListType LoopOp::getBeforeArguments() {
  return getBeforeBody()->getArguments();
}

void LoopOp::getSuccessorRegions(RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // The parent op always branches to the condition region.
  if (point.isParent()) {
    regions.emplace_back(&getBefore(), getBefore().getArguments());
    return;
  }

  assert(llvm::is_contained({&getAfter(), &getBefore()}, point) &&
         "there are only two regions in a WhileOp");
  // The body region always branches back to the condition region.
  if (point == getAfter()) {
    regions.emplace_back(&getBefore(), getBefore().getArguments());
    return;
  }

  regions.emplace_back(getResults());
  regions.emplace_back(&getAfter(), getAfter().getArguments());
}

std::optional<llvm::MutableArrayRef<OpOperand>>
LoopOp::getYieldedValuesMutable() {
  return getYieldOp().getResultsMutable();
}

sys::YieldOp LoopOp::getYieldOp() {
  return cast<sys::YieldOp>(getAfterBody()->getTerminator());
}

OperandRange LoopOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  assert(point == getBefore() &&
         "LoopOp is expected to branch only to the first region");
  return getInits();
}

void LoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  // TODO: Add canonicalization patterns.
  return;
}

LogicalResult LoopOp::verify() {
  // TODO,verification should be done
  return success();
}

void LoopOp::build(
    mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
    TypeRange resultTypes, ValueRange operands,
    function_ref<void(OpBuilder &, Location, ValueRange)> beforeBuilder,
    function_ref<void(OpBuilder &, Location, ValueRange)> afterBuilder) {
  odsState.addOperands(operands);
  odsState.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(odsBuilder);
  // Build before region.
  llvm::SmallVector<Location, 4> beforeArgLocs;
  beforeArgLocs.reserve(operands.size());
  for (const auto &operand : operands)
    beforeArgLocs.push_back(operand.getLoc());

  Region *beforeRegion = odsState.addRegion();
  Block *beforeBlock = odsBuilder.createBlock(
      beforeRegion, {}, operands.getTypes(), beforeArgLocs);

  if (beforeBuilder)
    beforeBuilder(odsBuilder, odsState.location, beforeBlock->getArguments());

  // Build after region.
  llvm::SmallVector<Location, 4> afterArgLocs(resultTypes.size(),
                                              odsState.location);
  Region *afterRegion = odsState.addRegion();
  Block *afterBlock =
      odsBuilder.createBlock(afterRegion, {}, resultTypes, afterArgLocs);
  if (afterBuilder)
    afterBuilder(odsBuilder, odsState.location, afterBlock->getArguments());
}

#define GET_OP_CLASSES
#include "SysIR/Dialect/IR/SysOps.cpp.inc"
