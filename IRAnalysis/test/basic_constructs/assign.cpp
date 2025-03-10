// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

SC_MODULE(Process) {
  // CHECK: !u32i = !cir.int<u, 32>
  // CHECK: "builtin.module"() <{sym_name = "Process"}> ({
  int x = 123;
  // CHECK: %0 = "cir.const"() <{value = #cir.int<123> : !u32i}> {sym_name =
  // "x"} : () -> !u32i
  sc_dt::sc_int<33> y = 1;
  // CHECK: %1 = "sys.const"() <{value = #sys.int<1> : !sys.s_int<s, 33>}>
  // {sym_name = "y"} : () -> !sys.s_int<s, 33>
  sc_dt::sc_int<33> z = y + 1;
  // CHECK: %2 = "cir.const"() <{value = #cir.int<1> : !u32i}> : () -> !u32i
  // CHECK: %3 = "sys.sbinop"(%1, %2) <{kind = 4 : i32}> {sym_name = "z"} :
  // (!sys.s_int<s, 33>, !u32i) -> !sys.s_int<s, 33>
};
// CHECK: }