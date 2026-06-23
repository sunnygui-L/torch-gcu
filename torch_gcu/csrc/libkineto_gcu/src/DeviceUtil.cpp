/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */
#include "DeviceUtil.h"

#include <tops_runtime.h>

#include <mutex>

namespace libkineto_gcu {

bool GcuAvailable = false;
bool isGcuAvailable() {
  static std::once_flag once;
  std::call_once(once, [] {
    // determine GCU availability on the system
    topsError_t error;
    int deviceCount;
    error = topsGetDeviceCount(&deviceCount);
    GcuAvailable = (error == topsSuccess && deviceCount > 0);
  });
  return GcuAvailable;
}

}  // namespace libkineto_gcu
