//===----------------------------------------------------------------------===//
//
// Created by Shuanglong Kan, 25.10.2024
// This file declares the Sys dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef _SYSMTEM_SYSOPS_H
#define _SYSMTEM_SYSOPS_H

include "SysIR/Dialect/IR/SysTypes.td"
include "SySIR/Dialect/IR/SysDialect.td"

class SysOP<string mnemoic, list<Trait> traits=[]> :
    Op<Sys_Dialect, mnemoic, traits>;

def ProcDefOP : SysOP<"ProcDef"> {
    let summary = "Define a process.";
    
    let description = [{
        This OP corresponds to a definition of a process, 
        which may run in parallel with other processes in the system.
    }];

    let arguments = (ins 
        SymbolNameAttr:$proc_name,
        TypeAttrOf<SysProcType>:$proc_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs
    );

    let results = (outs SysProcType:$result);

    let regions = (region AnyRegion:$body);
    let builders = [OpBuilder<(ins 
        "StringRef":$name, 
        "SProcessType":$type,
        //CArg wraps a type and its default value
        CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def ProcRegisterOP : SysOP<"ProcRegister"> {
    let summary = "Register a process into the system";
    let description = [{
        With the definition of processes, we also need to register these processes 
        into the system. When the system starts, all registered processes will start to 
        run in parallel.
    }];

    let arguments = (ins SysProcType:$procs, Variadic<Sys_AnyType>:$args);
    let hasCustomAssemblyFormat = 1;
}


#endif
