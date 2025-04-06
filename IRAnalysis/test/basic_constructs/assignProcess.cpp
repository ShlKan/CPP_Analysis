// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

// CHECK: !u32i = !cir.int<u, 32>
// CHECK: "builtin.module"() <{sym_name = "Process"}> ({
SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(process);
}
void process() {
  sc_dt::sc_int<33> x = 32;
  sc_dt::sc_int<33> y = x + 1;
  // CHECK: %0 = "sys.ProcDef"() <{proc_name = "process", proc_type =
  // CHECK: !sys<s_proc Proc ()>}> ({
  // CHECK: %1 = "sys.const"() <{value = #sys.int<32> : !sys.s_int<s, 33>}> :
  // CHECK: () -> !sys.s_int<s, 33>
  // CHECK: %2 = "cir.const"() <{value = #cir.int<1> : !u32i}> : () -> !u32i
  // CHECK: %3 = "sys.sbinop"(%1, %2) <{kind = 4 : i32}> : (!sys.s_int<s, 33>,
  // CHECK: !u32i) -> !sys.s_int<s, 33>
}
}
// CHECK:}) : () -> !sys<s_proc Proc ()>
;
// CHECK: "sys.ProcRegister"(%0) : (!sys<s_proc Proc ()>) -> ()
// CHECK: }) : () -> ()