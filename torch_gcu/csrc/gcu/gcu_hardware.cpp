/*
 * Copyright 2024 Enflame. All Rights Reserved.
 */

#include "gcu/gcu_hardware.h"

#include <ATen/native/ReduceOpsUtils.h>
#include <c10/core/ScalarType.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <torch/script.h>

#include "gcu/gcu_context.h"

namespace torch_gcu {

BackendType toBackendType(int major) {
  switch (major) {
    case 3:
      return BackendType::kS60;
    case 4:
      return BackendType::kL600;
    default:
      TORCH_INTERNAL_ASSERT(false, "Cannot convert major ", major,
                            " to BackendType.");
      return BackendType::KNone;
  }
}

HardwareType& HardwareType::GetInstance() {
  static HardwareType ref_hardware;
  return ref_hardware;
}

HardwareType::HardwareType() {
  auto device_properties = getCurrentDeviceProperties();
  hardware_ = toBackendType(device_properties->major);
}

TORCH_GCU_API bool GetEnvBool(const char* name, bool defval) {
  const char* env = std::getenv(name);
  bool val = false;
  if (env == nullptr) {
    val = defval;
  } else if (std::strcmp(env, "true") == 0) {
    val = true;
  } else if (std::strcmp(env, "false") == 0) {
    val = false;
  } else {
    val = std::atoi(env) != 0;
  }
  return val;
}

TORCH_GCU_API void* gcu_device_ptr(const at::Tensor& self) {
  if (self.defined()) {
    if (self.device().is_privateuseone()) {
      void* host_ptr = gcu_data_ptr(self);
      void* device_ptr = nullptr;
      topsError_t error = topsSuccess;
      (void)topsPointerGetAttribute(
          &device_ptr, TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER, host_ptr);
      return device_ptr;
    } else {
      return self.data_ptr();
    }
  } else {
    return nullptr;
  }
}

}  // namespace torch_gcu
