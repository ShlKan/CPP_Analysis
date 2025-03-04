// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

SC_MODULE(Process) {
  // CHECK: module @Process {
  int x = 0;
  // CHECK: %0 = cir.const {sym_name = "x"} #cir.int<0> : !u32i
  sc_dt::sc_int<33> y = 32;
  // CHECK: %1 = sys.const {sym_name = "y"} #sys.int<32> : !sys.s_int<s, 33>
};
// CHECK: }