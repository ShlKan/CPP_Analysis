//===----------------------------------------------------------------------===//
//
// Created by Shuanglong Kan, 25.10.2024
// This file declares the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef _SYSTEM_SMDIALECT_H
#define _SYSTEM_SMDIALECT_H

include "mlir/IR/OpBase.td"

def SM_Dialect : Dialect {
    let name = "sm";

    let summary = "A high-level dialect for supporting system models' "
                  "code generation and optimization";

    let cppNamespace = "::mlir::sm";

    let extraClassDeclaration = [{

        //Types manipulation methods.
        void registerTypes();
        Type parseType(DialectAsmParser &parser) const override;
        void printType(Type type, DialectAsmPrinter &printer) const override;

        //Attributes manipulation methods.
        void registerAttributes();
        Attribute parseAttribute(DialectAsmParser &parser,
                                    Type type) const override;

        void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

    }];

} 

#endif //_SYSTEMC_SMDIALECT_H









