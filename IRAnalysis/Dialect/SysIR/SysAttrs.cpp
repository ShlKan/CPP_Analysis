//===--------------- SysAttrs -------------------===//

#include "SysIR/Dialect/IR/SysAttrs.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::sys;
using namespace ::mlir;

namespace llvm {
llvm::hash_code hash_value(const BitVector &bv) {
  std::vector<int> bv1;
  for (int i = 0; i < bv.size(); i++) {
    bv1.push_back(bv[i]);
  }
  return hash_combine_range(bv1.cbegin(), bv1.cend());
}
} // namespace llvm

#define GET_ATTRDEF_CLASSES
#include "SysIR/Dialect/IR/SysOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// IntAttr definitions
//===----------------------------------------------------------------------===//

Attribute IntAttr::parse(AsmParser &parser, Type odsType) {
  mlir::APInt APValue;

  if (!mlir::isa<mlir::sys::SIntType>(odsType))
    return {};
  auto type = mlir::cast<SIntType>(odsType);

  // Consume the '<' symbol.
  if (parser.parseLess())
    return {};

  // Fetch arbitrary precision integer value.
  if (type.isSigned()) {
    int64_t value;
    if (parser.parseInteger(value))
      parser.emitError(parser.getCurrentLocation(), "expected integer value");
    APValue = mlir::APInt(type.getWidth(), value, type.isSigned());
    if (APValue.getSExtValue() != value)
      parser.emitError(parser.getCurrentLocation(),
                       "integer value too large for the given type");
  } else {
    uint64_t value;
    if (parser.parseInteger(value))
      parser.emitError(parser.getCurrentLocation(), "expected integer value");
    APValue = mlir::APInt(type.getWidth(), value, type.isSigned());
    if (APValue.getZExtValue() != value)
      parser.emitError(parser.getCurrentLocation(),
                       "integer value too large for the given type");
  }

  // Consume the '>' symbol.
  if (parser.parseGreater())
    return {};

  return IntAttr::get(type, APValue);
}

void IntAttr::print(AsmPrinter &printer) const {
  auto type = mlir::cast<SIntType>(getType());
  printer << '<';
  if (type.isSigned())
    printer << getSInt();
  else
    printer << getUInt();
  printer << '>';
}

void BitVecAttr::print(AsmPrinter &printer) const {
  auto type = mlir::cast<SBitVecType>(getType());
  if (getBV().size() == 0)
    printer << "[]";
  else {
    printer << '[';
    for (int i = 0; i < getBV().size(); i++) {
      printer << getBV()[i] << ";";
    }
    printer << ']';
  }
}

void SysDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  // TODO
  if (failed(generatedAttributePrinter(attr, os)))
    llvm_unreachable("unexpected CIR type kind");
}

void SysDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "SysIR/Dialect/IR/SysOpsAttributes.cpp.inc"
      >();
}