#pragma once
/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include <tops/tops_runtime_api.h>

namespace torch_gcu {

at::Device getDeviceFromPtr(void* ptr) {
  topsPointerAttribute_t attr{};

  C10_GCU_CHECK(topsPointerGetAttributes(&attr, ptr));

  // TODO: TORCH_CHECK(attr.type != topsMemoryTypeUnregistered)
  TORCH_CHECK(attr.memoryType == topsMemoryTypeDevice ||
                  attr.memoryType == topsMemoryTypeHost,
              "The specified pointer resides on host memory and is not "
              "registered with any GCU device.");

  return {c10::DeviceType::PrivateUse1,
          static_cast<at::DeviceIndex>(attr.device)};
}

}  // namespace torch_gcu