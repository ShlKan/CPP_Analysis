// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

using namespace sc_dt;

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(emptyProcess);
}
void emptyProcess() {
  sc_bv<8> b1 = "01111011";
  sc_bv<7> b2 = "1000001";
  sc_bv<6> b3 = b2 & b1;
  // CHECK: %3 = "sys.sbinop"(%2, %1) <{kind = 6 : i32}> : (!sys.s_bv<7>,
  // CHECK: !sys.s_bv<8>) -> !sys.s_bv<6>
  sc_bv<6> b4 = b2 | b1;
  // CHECK: %4 = "sys.sbinop"(%2, %1) <{kind = 8 : i32}> : (!sys.s_bv<7>,
  // CHECK: !sys.s_bv<8>) -> !sys.s_bv<6>
  sc_bv<6> b5 = b2 ^ b1;
  // CHECK: %5 = "sys.sbinop"(%2, %1) <{kind = 7 : i32}> : (!sys.s_bv<7>,
  // CHECK: !sys.s_bv<8>) -> !sys.s_bv<6>

  sc_lv<8> c1 = "01111011";
  sc_lv<7> c2 = "10X0Z01";
  sc_lv<6> c3 = c2 & c1;
  // CHECK: %8 = "sys.sbinop"(%7, %6) <{kind = 6 : i32}> : (!sys.s_bvl<7>,
  // CHECK: !sys.s_bvl<8>) -> !sys.s_bvl<6>
  sc_lv<5> c4 = c2 | c1;
  // CHECK: %9 = "sys.sbinop"(%7, %6) <{kind = 8 : i32}> : (!sys.s_bvl<7>,
  // CHECK: !sys.s_bvl<8>) -> !sys.s_bvl<5>
  sc_lv<5> c5 = c2 ^ c1;
  // CHECK: %10 = "sys.sbinop"(%7, %6) <{kind = 7 : i32}> : (!sys.s_bvl<7>,
  // CHECK: !sys.s_bvl<8>) -> !sys.s_bvl<5>
}
}
;