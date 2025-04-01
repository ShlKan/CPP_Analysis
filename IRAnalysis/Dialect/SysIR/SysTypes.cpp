//===----------------------------------------------------------------------===//
//
// This file defines the types in the SysIR dialect.
//
//===----------------------------------------------------------------------===//
#include "SysIR/Dialect/IR/SysTypes.h"
#include "SysIR/Dialect/IR/SysDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------====//
//  Custom parser and printer.
//===----------------------------------------------------------------------====//

static mlir::ParseResult parseProcArgs(mlir::AsmParser &p,
                                       llvm::SmallVector<mlir::Type> &args);

static void printProcArgs(mlir::AsmPrinter &p, llvm::ArrayRef<mlir::Type> args);

//===----------------------------------------------------------------------====//
//  ODS Sys dialect type implementation.
//===----------------------------------------------------------------------====//

#define GET_TYPEDEF_CLASSES
#include "SysIR/Dialect/IR/SysOpsTypes.cpp.inc"

using namespace ::mlir::sys;

mlir::Type SIntType::parse(mlir::AsmParser &parser) {
  auto *context = parser.getBuilder().getContext();
  auto loc = parser.getCurrentLocation();
  bool isSigned;
  unsigned width;

  if (parser.parseLess())
    return {};

  // Fetch integer sign.
  llvm::StringRef sign;
  if (parser.parseKeyword(&sign))
    return {};
  if (sign == "s")
    isSigned = true;
  else if (sign == "u")
    isSigned = false;
  else {
    parser.emitError(loc, "expected 's' or 'u'");
    return {};
  }

  if (parser.parseComma())
    return {};

  // Fetch integer size.
  if (parser.parseInteger(width))
    return {};
  if (width < 1 || width > 64) {
    parser.emitError(loc, "expected integer width to be from 1 up to 64");
    return {};
  }

  if (parser.parseGreater())
    return {};

  return SIntType::get(context, width, isSigned);
}

void SIntType::print(::mlir::AsmPrinter &odsPrinter) const {
  auto sign = isSigned() ? 's' : 'u';
  odsPrinter << "<" << sign << ", " << this->getWidth() << ">";
}

mlir::LogicalResult
SIntType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                 unsigned width, bool isSigned) {

  // TODO

  return mlir::success();
}

mlir::ParseResult parseProcArgs(mlir::AsmParser &p,
                                llvm::SmallVector<mlir::Type> &args) {
  mlir::Type type;
  if (p.parseType(type))
    return mlir::failure();
  args.push_back(type);
  while (succeeded(p.parseOptionalComma())) {
    if (p.parseType(type))
      return mlir::failure();
    args.push_back(type);
  }
  return mlir::success();
}

void printProcArgs(mlir::AsmPrinter &p, llvm::ArrayRef<mlir::Type> args) {
  llvm::interleaveComma(args, p, [&p](mlir::Type type) { p.printType(type); });
}

//===--------------------------------------------------------------===//
//                            SysDialect
//===--------------------------------------------------------------===//

void SysDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SysIR/Dialect/IR/SysOpsTypes.cpp.inc"
      >();
}

//===--------------------------------------------------------------===//
//                            SIntType
//===--------------------------------------------------------------===//
uint64_t SIntType::getABIAlignment(const mlir::DataLayout &dataLayout,
                                   mlir::DataLayoutEntryListRef params) const {
  return (uint64_t)(getWidth() / 8);
}

llvm::TypeSize
SIntType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                            mlir::DataLayoutEntryListRef params) const {
  return llvm::TypeSize::getFixed(getWidth());
}

uint64_t
SIntType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const {
  return (uint64_t)(getWidth() / 8);
}

mlir::Type SysDialect::parseType(DialectAsmParser &printer) const {
  // TODO
  llvm::llvm_unreachable_internal("parseType: Have not yet implemented");
}

void SysDialect::printType(Type type, mlir::DialectAsmPrinter &printer) const {
  if (generatedTypePrinter(type, printer).succeeded())
    return;
  llvm::TypeSwitch<Type>(type)
      .Case<SIntType>([&](SIntType type) { type.print(printer); })
      .Case<SProcessType>([&](SProcessType type) { type.print(printer); })
      .Default([&](Type) {
        llvm::report_fatal_error("printer is missing a handler for this type");
      });
}

//==----------------------------------------------------------------===//
//                            S_BitVectorType
//==----------------------------------------------------------------===//

mlir::LogicalResult
SBitVecType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    unsigned width) {
  if (width < 1) {
    emitError() << "expected bit vector width to be greater than 0";
    return mlir::failure();
  }
  return mlir::success();
}

uint64_t
SBitVecType::getABIAlignment(const mlir::DataLayout &dataLayout,
                             mlir::DataLayoutEntryListRef params) const {
  return (uint64_t)(getWidth() / 8);
}

llvm::TypeSize
SBitVecType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                               mlir::DataLayoutEntryListRef params) const {
  return llvm::TypeSize::getFixed(getWidth());
}

uint64_t SBitVecType::getPreferredAlignment(
    const ::mlir::DataLayout &dataLayout,
    ::mlir::DataLayoutEntryListRef params) const {
  return (uint64_t)(getWidth() / 8);
}

void SBitVecType::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << this->getWidth() << ">";
}
