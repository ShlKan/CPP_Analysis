// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

SC_MODULE(Process){// CHECK: "builtin.module"() <{sym_name = "Process"}> ({
                   SC_CTOR(Process){SC_THREAD(emptyProcess);
// CHECK: %0 = "sys.ProcDef"() <{proc_name = "emptyProcess", proc_type =
// !sys<s_proc Proc ()>}> ({
}
void emptyProcess() { sc_dt::sc_int<8> i = 0; }
}
// CHECK: %1 = "sys.const"() <{value = #sys.int<0> : !sys.s_int<s, 8>}> : () ->
// !sys.s_int<s, 8>
//  CHECK: }) : () -> !sys<s_proc Proc ()>
;
// CHECK: "sys.ProcRegister"(%0) : (!sys<s_proc Proc ()>) -> ()
// CHECK: }) : () -> ()