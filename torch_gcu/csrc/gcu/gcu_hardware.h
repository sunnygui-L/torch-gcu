/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/Tensor.h>

#include "gcu/gcu_macros.h"

namespace torch_gcu {

TORCH_GCU_API bool GetEnvBool(const char* name, bool defval);

enum class BackendType {
  KNone = -1,
  kS60 = 0,
  kL600 = 1,
  NumHardwares,
};

class TORCH_GCU_API HardwareType {
 public:
  static HardwareType& GetInstance();
  BackendType getHardware() const { return hardware_; }

 private:
  HardwareType();
  HardwareType(const HardwareType&) = delete;
  HardwareType(HardwareType&&) = delete;
  HardwareType& operator=(const HardwareType&) = delete;
  HardwareType& operator=(HardwareType&&) = delete;

 private:
  BackendType hardware_ = BackendType::KNone;
};

TORCH_GCU_API inline bool is_narrow_type(const c10::ScalarType& dtype) {
  auto hardware = HardwareType::GetInstance().getHardware();
  switch (hardware) {
    case BackendType::kS60:
    case BackendType::kL600:
      if (GetEnvBool("TORCH_GCU_ENABLE_INT64_AND_UINT64", false)) {
        // When enabled, treat int64/uint64 as supported (i.e. not narrowed),
        // keep double/complex<double> narrowed.
        return dtype == c10::ScalarType::Double ||
               dtype == c10::ScalarType::ComplexDouble;
      } else {
        return dtype == c10::ScalarType::Double ||
               dtype == c10::ScalarType::Long ||
               dtype == c10::ScalarType::UInt64 ||
               dtype == c10::ScalarType::ComplexDouble;
      }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported hardware type.");
      return false;
  }
}

TORCH_GCU_API inline bool is_narrow_type_tmp(const c10::ScalarType& dtype) {
  auto hardware = HardwareType::GetInstance().getHardware();
  switch (hardware) {
    case BackendType::kS60:
      return dtype == c10::ScalarType::Double ||
             dtype == c10::ScalarType::Long ||
             dtype == c10::ScalarType::UInt64 ||
             dtype == c10::ScalarType::ComplexDouble;
    case BackendType::kL600:
      return dtype == c10::ScalarType::Double ||
             dtype == c10::ScalarType::ComplexDouble;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported hardware type.");
      return false;
  }
}

// gcu not support long && double dtype, narrow
TORCH_GCU_API inline c10::ScalarType get_narrow_type(
    const c10::ScalarType& dtype) {
  if (dtype == c10::ScalarType::Double) {
    return c10::ScalarType::Float;
  } else if (dtype == c10::ScalarType::Long) {
    return c10::ScalarType::Int;
  } else if (dtype == c10::ScalarType::UInt64) {
    return c10::ScalarType::UInt32;
  } else if (dtype == c10::ScalarType::ComplexDouble) {
    return c10::ScalarType::ComplexFloat;
  } else {
    return dtype;
  }
}

// gcu3.0 not support long && double dtype, narrow
TORCH_GCU_API inline c10::ScalarType get_gcu_scalar_type(
    const c10::ScalarType& dtype) {
  if (is_narrow_type(dtype)) {
    return get_narrow_type(dtype);
  } else {
    return dtype;
  }
}

// gcu3.0 not support long && double dtype, narrow
TORCH_GCU_API inline c10::ScalarType get_gcu_scalar_type_tmp(
    const c10::ScalarType& dtype) {
  if (is_narrow_type_tmp(dtype)) {
    return get_narrow_type(dtype);
  } else {
    return dtype;
  }
}

TORCH_GCU_API inline void* gcu_data_ptr(const at::Tensor& self) {
  if (self.defined()) {
    if (self.device().is_privateuseone()) {
      auto* self_ = self.unsafeGetTensorImpl();
      if (self_->is_empty()) {
        return nullptr;
      }
      size_t itemsize = self_->itemsize();
      if (is_narrow_type(self.scalar_type())) {
        itemsize = itemsize >> 1;
      }
      return static_cast<char*>(self_->storage().mutable_data()) +
             itemsize * self_->storage_offset();
    } else {
      return self.data_ptr();
    }
  } else {
    return nullptr;
  }
}

TORCH_GCU_API inline void* gcu_data_ptr_tmp(const at::Tensor& self) {
  if (self.defined()) {
    if (self.device().is_privateuseone()) {
      auto* self_ = self.unsafeGetTensorImpl();
      if (self_->is_empty()) {
        return nullptr;
      }
      size_t itemsize = self_->itemsize();
      if (is_narrow_type_tmp(self.scalar_type())) {
        itemsize = itemsize >> 1;
      }
      return static_cast<char*>(self_->storage().mutable_data()) +
             itemsize * self_->storage_offset();
    } else {
      return self.data_ptr();
    }
  } else {
    return nullptr;
  }
}

TORCH_GCU_API void* gcu_device_ptr(const at::Tensor& self);

}  // namespace torch_gcu
