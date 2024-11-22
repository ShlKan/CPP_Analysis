#include <systemc>

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(emptyProcess);
}

void emptyProcess(void) {
  // Do nothing.
}
}
;
