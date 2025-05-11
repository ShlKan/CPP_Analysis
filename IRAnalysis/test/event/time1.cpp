// RUN: %cpp_analysis %args %s | FileCheck %s

#include <sys/wait.h>
#include <systemc>

using namespace sc_dt;
using namespace sc_core;

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(P1);
}

void P1() {
  sc_time t1(2, SC_MS);
  wait(t1);
}
}
;

// CHECK: %1 = "sys.const"() <{value = #sys<time: 2 ms> : !sys<s_time time>}> :
// CHECK: () -> !sys<s_time time>
// <{kind = 1 : i32}> : (!sys<s_time time>) -> ()