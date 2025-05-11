//===--------------- SysAttrs -------------------===//

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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cstdint>

#include "SysIR/Dialect/IR/SysAttrs.h"

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
llvm::hash_code hash_value(const SmallVector<uint8_t> &bv) {
  return hash_combine_range(bv.begin(), bv.end());
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
  if (getBV().empty())
    printer << "[]";
  else {
    printer << '[';
    for (int i = 0; i < getBV().size() - 1; i++) {
      printer << getBV()[i] << ";";
    }
    printer << getBV()[getBV().size() - 1];
    printer << ']';
  }
}

void BitVecLAttr::print(AsmPrinter &printer) const {
  auto type = mlir::cast<SBitVecLType>(getType());
  if (getValue().empty())
    printer << "[]";
  else {
    printer << '[';
    for (int i = 0; i < getValue().size(); i++) {
      if (getValue()[i] == 0)
        printer << '0';
      else if (getValue()[i] == 1)
        printer << '1';
      else if (getValue()[i] == 2)
        printer << 'z';
      else if (getValue()[i] == 3)
        printer << 'x';
      else
        llvm_unreachable("unexpected value in BitVecLAttr");
      if (i != getValue().size() - 1)
        printer << ";";
    }
    printer << ']';
  }
}

void TimeAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << ": " << getValue().getZExtValue() << " ";
  switch (getKind()) {
  case STimeKind::SC_FS:
    odsPrinter << "fs";
    break;
  case STimeKind::SC_PS:
    odsPrinter << "ps";
    break;
  case STimeKind::SC_NS:
    odsPrinter << "ns";
    break;
  case STimeKind::SC_US:
    odsPrinter << "us";
    break;
  case STimeKind::SC_MS:
    odsPrinter << "ms";
    break;
  case STimeKind::SC_SEC:
    odsPrinter << "sec";
    break;
  default:
    llvm_unreachable("unexpected STimeKind");
  }
}

void EventAttr::print(AsmPrinter &printer) const {
  printer << "[";
  printer << getValue();
  printer << "]";
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