// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

using namespace sc_dt;
using namespace sc_core;

SC_MODULE(Process) {
  SC_CTOR(Process) { SC_THREAD(emptyProcess); }
  sc_event e1;
  void emptyProcess() {}
};

// CHECK: %0 = "sys.const"() <{value = #sys<event[e1]> : !sys<s_event event>}> :
// CHECK: () -> !sys<s_event event>