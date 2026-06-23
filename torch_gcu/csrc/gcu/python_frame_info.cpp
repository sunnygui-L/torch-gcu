/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "gcu/python_frame_info.h"

#include <atomic>
#include <string>

namespace torch_gcu {

typedef std::string (*PythonCallstackFn)();

static std::atomic<PythonCallstackFn> python_callstack_fn = nullptr;

void SetPythonCallstackFn(PythonCallstackFn f) { python_callstack_fn.store(f); }

std::string GetPythonFrame() {
  if (python_callstack_fn.load() == nullptr) {
    return "";
  }
  return python_callstack_fn.load()();
}
}  // namespace torch_gcu
