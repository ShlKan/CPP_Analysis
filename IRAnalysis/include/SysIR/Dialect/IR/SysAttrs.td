//===----------------------------------------------------------------------===//
// Sys Attrs
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYS_DIALECT_SYSIR_ATTRS
#define MLIR_SYS_DIALECT_SYSIR_ATTRS

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "SysIR/Dialect/IR/SysDialect.td"

class Sys_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Sys_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

def IntAttr : Sys_Attr<"Int", "int", [TypedAttrInterface]> {
  let summary = "An Attribute containing a integer value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type in SysIR.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type, "int64_t":$value), [{
      SIntType intType = mlir::cast<SIntType>(type);
      mlir::APInt apValue(intType.getWidth(), value, intType.isSigned());
      return $_get(intType.getContext(), intType, apValue);
    }]>,
  ];
  let extraClassDeclaration = [{
    int64_t getSInt() const { return getValue().getSExtValue(); }
    uint64_t getUInt() const { return getValue().getZExtValue(); }
    bool isNullValue() const { return getValue() == 0; }
    uint64_t getBitWidth() const { return mlir::cast<SIntType>(getType()).getWidth(); }
  }];
  //let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def BitVecAttr : Sys_Attr<"BitVec", "bv", [TypedAttrInterface]> {
  let summary = "An Attribute containing a bv value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type in SysIR.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "BitVector":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const llvm::StringRef &":$value), [{
      BitVector bv;
      for (const auto &c : value) {
        if (c == '1') {
          bv.push_back(true);
        } else if (c == '0') {
          bv.push_back(false);
        }
      }
      return $_get(type.getContext(), type, bv);
    }]>
  ];
  let extraClassDeclaration = [{
    BitVector getBV() const { return getValue(); }
  }];
  //let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def BitVecLAttr : Sys_Attr<"BitVecL", "bvl", [TypedAttrInterface]> {
  let summary = "An Attribute containing a bv value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type in SysIR.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "SmallVector<uint8_t>":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const llvm::StringRef &":$value), [{
      SmallVector<uint8_t> bv;
      for (const auto &c : value) {
        if (c == '1')
          bv.push_back(1);
        else if (c == '0')
          bv.push_back(0);
        else if (c == 'Z' or c == 'z')
          bv.push_back(2);
        else if (c == 'X' or c == 'x')
          bv.push_back(3);
      }
      return $_get(type.getContext(), type, bv);
    }]>
  ];
  let extraClassDeclaration = [{
    SmallVector<uint8_t> getBV() const { return getValue(); }
  }];
  //let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}



#endif
