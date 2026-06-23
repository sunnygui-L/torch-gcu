/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <tops/tops_runtime_api.h>

#include "gcu/gcu_stream.h"

namespace torch_gcu {

struct TORCH_GCU_API GCUGuardImpl final
    : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  GCUGuardImpl() {}

  explicit GCUGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
  }

  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }

  c10::Device exchangeDevice(c10::Device d) const override;

  c10::Device getDevice() const override;

  c10::optional<c10::Device> uncheckedGetDevice() const noexcept;

  void setDevice(c10::Device d) const override;

  void uncheckedSetDevice(c10::Device d) const noexcept override;

  c10::Stream getStream(c10::Device d) const noexcept override {
    return getCurrentGCUStream(d.index()).unwrap();
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    return getDefaultGCUStream(d.index());
  }

  c10::Stream getNewStream(c10::Device d, int priority = 0) const override;

  c10::Stream getStreamFromGlobalPool(
      c10::Device d, bool isHighPriority = false) const override {
    return getStreamFromPool(isHighPriority, d.index());
  }

  // NB: These do NOT set the current device
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    GCUStream cs(s);
    auto old_stream = getCurrentGCUStream(s.device().index());
    setCurrentGCUStream(cs);
    return old_stream.unwrap();
  }

  c10::DeviceIndex deviceCount() const noexcept override;

  // Event-related functions
  void createEvent(topsEvent_t* tops_event, const c10::EventFlag flag) const;

  void destroyEvent(
      void* event, const c10::DeviceIndex device_index) const noexcept override;

  void record(void** event, const c10::Stream& stream,
              const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override;

  void block(void* event, const c10::Stream& stream) const override;

  // May be called from any device
  bool queryEvent(void* event) const override;

  // Stream-related functions
  bool queryStream(const c10::Stream& stream) const override {
    GCUStream gcu_stream{stream};
    return gcu_stream.query();
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    GCUStream gcu_stream{stream};
    gcu_stream.synchronize();
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr,
                             const c10::Stream& stream) const override;

  void synchronizeEvent(void* event) const override;

  void synchronizeDevice(const c10::DeviceIndex device_index) const override;

  double elapsedTime(void* event1, void* event2,
                     const c10::DeviceIndex device_index) const override;
};

}  // namespace torch_gcu
