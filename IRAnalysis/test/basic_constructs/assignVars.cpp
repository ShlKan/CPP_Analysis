// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

SC_MODULE(Process) {
  // CHECK: module @Process {
  sc_dt::sc_int<33> x = 7;
  // CHECK: %0 = sys.const {sym_name = "x"} #sys.int<7> : !sys.s_int<s, 33>
  sc_dt::sc_int<33> y = 8;
  // CHECK: %1 = sys.const {sym_name = "y"} #sys.int<8> : !sys.s_int<s, 33>
  sc_dt::sc_int<33> z = x + y;
  // CHECK: %2 = sys.sbinop(sadd, %0, %1) : !sys.s_int<s, 33> {sym_name = "z"}
};
// CHECK: }