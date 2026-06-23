#include "gcu/trace.h"

#include <c10/util/Exception.h>

#include <iostream>
#include <thread>
#include <unordered_map>

thread_local std::unordered_map<std::string, TorchTracepoint>
    compile_tracepoints;

void set_user_trace_start(const int &category, const std::string &name,
                          const std::string &payload) {
  auto it = compile_tracepoints.find(name);
  if (it == compile_tracepoints.end()) {
    compile_tracepoints[name] =
        TorchTracepoint(TorchTrace::GetTorchTrace().domain());
  }
  const char *payload_ptr = payload.empty() ? nullptr : payload.c_str();
  compile_tracepoints[name].enter(category, name.c_str(), payload_ptr);
}

void set_user_trace_end(const int &category, const std::string &name) {
  auto it = compile_tracepoints.find(name);
  if (it == compile_tracepoints.end()) {
    TORCH_CHECK(false, "Not defined trace point name ", name);
  }
  compile_tracepoints[name].exit();
  compile_tracepoints.erase(it);
}
