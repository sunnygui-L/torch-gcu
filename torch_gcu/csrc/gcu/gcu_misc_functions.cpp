#include "gcu/gcu_misc_functions.h"

#include <stdlib.h>

namespace torch_gcu {

const char* get_gcu_check_suffix() noexcept {
  static char* device_blocking_flag = getenv("TOPS_LAUNCH_BLOCKING");
  static bool blocking_enabled =
      (device_blocking_flag && atoi(device_blocking_flag));
  if (blocking_enabled) {
    return "";
  } else {
    return "\nGCU kernel errors might be asynchronously reported at some"
           " other API call, so the stacktrace below might be incorrect."
           "\nFor debugging consider passing TOPS_LAUNCH_BLOCKING=1.";
  }
}
std::mutex* getFreeMutex() {
  static std::mutex gcu_free_mutex;
  return &gcu_free_mutex;
}

}  // namespace torch_gcu