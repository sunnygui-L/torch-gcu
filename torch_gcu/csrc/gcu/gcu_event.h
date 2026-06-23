/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <c10/util/Exception.h>
#include <tops/tops_runtime_api.h>

#include "gcu/gcu_macros.h"
#include "gcu/gcu_stream.h"

namespace torch_gcu {
/*
 * GCUEvents are movable not copyable wrappers around GCU's events.
 *
 * GCUEvents are constructed lazily when first recorded unless it is
 * reconstructed from a topsIpcEventHandle_t. The event has a device, and this
 * device is acquired from the first recording stream. However, if reconstructed
 * from a handle, the device should be explicitly specified; or if ipc_handle()
 * is called before the event is ever recorded, it will use the current device.
 * Later streams that record the event must match this device.
 */
struct TORCH_GCU_API GCUEvent {
  // Constructors
  // Default value for `flags` is specified below - it's topsEventDisableTiming
  GCUEvent() noexcept = default;
  GCUEvent(unsigned int flags) noexcept : flags_{flags} {}

  GCUEvent(c10::DeviceIndex device_index, const topsIpcEventHandle_t* handle);

  // Note: event destruction done on creating device to avoid creating a
  // GCU context on other devices.
  ~GCUEvent();

  GCUEvent(const GCUEvent&) = delete;
  GCUEvent& operator=(const GCUEvent&) = delete;

  GCUEvent(GCUEvent&& other) noexcept { moveHelper(std::move(other)); }
  GCUEvent& operator=(GCUEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator topsEvent_t() const { return event(); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const GCUEvent& left, const GCUEvent& right) {
    return left.event_ < right.event_;
  }

  c10::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(at::kPrivateUse1, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const { return is_created_; }
  c10::DeviceIndex device_index() const { return device_index_; }
  topsEvent_t event() const { return event_; }

  // Note: topsEventQuery can be safely called from any device
  bool query() const;

  void record() { record(getCurrentGCUStream()); }

  void recordOnce(const GCUStream& stream) {
    if (!was_recorded_) record(stream);
  }

  // Note: topsEventRecord must be called on the same device as the event.
  void record(const GCUStream& stream);

  // Note: topsStreamWaitEvent must be called on the same device as the stream.
  // The event has no actual GCU resources associated with it.
  void block(const GCUStream& stream);

  // Note: topsEventElapsedTime can be safely called from any device
  float elapsed_time(const GCUEvent& other) const;

  // Note: topsEventSynchronize can be safely called from any device
  void synchronize() const;

  // Note: topsIpcGetEventHandle must be called on the same device as the event
  void ipc_handle(topsIpcEventHandle_t* handle);

 private:
  unsigned int flags_ = topsEventDisableTiming;
  bool is_created_ = false;
  bool was_recorded_ = false;
  c10::DeviceIndex device_index_ = -1;
  topsEvent_t event_{};

  void createEvent(c10::DeviceIndex device_index);

  void moveHelper(GCUEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

}  // namespace torch_gcu
