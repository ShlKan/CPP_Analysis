# This replicates part of the add_mlir_dialect cmake function from MLIR that
# cannot be used here. This happens because it expects to be run inside MLIR
# directory which is not the case for CIR (and also FIR, both have similar
# workarounds).

# Equivalent to add_mlir_dialect(CIROps cir)
set(LLVM_TARGET_DEFINITIONS CIROps.td)
mlir_tablegen(CIROps.h.inc -gen-op-decls)
mlir_tablegen(CIROps.cpp.inc -gen-op-defs)
mlir_tablegen(CIROpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(CIROpsTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(CIROpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CIROpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRCIROpsIncGen)
add_dependencies(mlir-headers MLIRCIROpsIncGen)

# Equivalent to add_mlir_doc
add_clang_mlir_doc(CIROps CIROps Dialects/ -gen-op-doc)
add_clang_mlir_doc(CIRAttrs CIRAttrs Dialects/ -gen-attrdef-doc)
add_clang_mlir_doc(CIRTypes CIRTypes Dialects/ -gen-typedef-doc)

# Generate extra headers for custom enum and attrs.
mlir_tablegen(CIROpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(CIROpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(CIROpsStructs.h.inc -gen-attrdef-decls)
mlir_tablegen(CIROpsStructs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(CIROpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CIROpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRCIREnumsGen)

clang_tablegen(CIRBuiltinsLowering.inc -gen-cir-builtins-lowering
    SOURCE CIROps.td
    TARGET CIRBuiltinsLowering)
