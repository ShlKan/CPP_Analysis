// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

using namespace sc_dt;

SC_MODULE(Process) {
  int x = 123;
  sc_int<33> y = 1;
  sc_int<33> z = y + 1;
  // CHECK: %2 = "cir.const"() <{value = #cir.int<1> : !u32i}> : () -> !u32i
  // CHECK: %3 = "sys.sbinop"(%1, %2) <{kind = 4 : i32}> {sym_name = "z"} :
  // CHECK: (!sys.s_int<s, 33>, !u32i) -> !sys.s_int<s, 33>
};