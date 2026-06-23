
/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/util/Registry.h>

#include "aten/aot_ops/gcu_resize.h"
#include "gcu/gcu_macros.h"
namespace at {
struct Generator;
}  // namespace at

namespace torch_gcu {

struct TORCH_GCU_API GCUHooksArgs : public at::PrivateUse1HooksArgs {};

struct TORCH_GCU_API GCUHooks : public at::PrivateUse1HooksInterface {
  GCUHooks(GCUHooksArgs) {}

  GCUHooks() = default;

  ~GCUHooks() = default;

  bool isBuilt() const override { return true; }

  bool isAvailable() const override { return hasGCU(); }

  void init() const override;

  at::Generator getNewGenerator(
      c10::DeviceIndex device_index = -1) const override;

  const at::Generator& getDefaultGenerator(
      at::DeviceIndex device_index = -1) const override;

  at::Device getDeviceFromPtr(void* data) const override;

  bool isPinnedPtr(const void* data) const override;

  bool hasPrimaryContext(at::DeviceIndex device_index) const override;

  at::DeviceIndex current_device() const;

  at::Allocator* getPinnedMemoryAllocator() const override;

  at::Allocator* getGCUDeviceAllocator() const;

  std::string showConfig() const;

  bool hasGCU() const;

  int getNumGCUs() const;

  void deviceSynchronize(at::DeviceIndex device_index) const;

  void resizePrivateUse1Bytes(const c10::Storage& storage,
                              size_t newsize) const override {
    // TODO(gcu): Cope with 64-bit (narrow) input storage
    torch_gcu::aotops::resize_bytes(storage.unsafeGetStorageImpl(), newsize,
                                    false);
  }
};

namespace detail {

TORCH_GCU_API const GCUHooks& getGCUHooks();

TORCH_GCU_API void RegisterGCUHooks();

}  // namespace detail

}  // namespace torch_gcu
