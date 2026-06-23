#include "gcu/cpp_frame_info.h"

namespace torch_gcu {

static std::atomic<CppCallstackFn> cpp_callstack_fn = nullptr;

void SetCppCallstackFn(CppCallstackFn f) { cpp_callstack_fn.store(f); }

std::string GetCppFrame(c10::GatheredContext* context) {
  if (cpp_callstack_fn.load() == nullptr) {
    return "";
  }
  return cpp_callstack_fn.load()(context);
}
}  // namespace torch_gcu