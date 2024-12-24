#include <systemc>

SC_MODULE(Process){

    SC_CTOR(Process){SC_THREAD(emptyProcess);
SC_THREAD(emptyProcess1);
}

void emptyProcess(void) {
  // Do nothing.
}

void emptyProcess1(void) {
  // Do nothing.
}
}
;
