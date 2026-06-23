/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "gcu/gcu_guard_impl.h"

#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_functions.h"

namespace torch_gcu {

constexpr c10::DeviceType GCUGuardImpl::static_type;

c10::Device GCUGuardImpl::exchangeDevice(c10::Device d) const {
  TORCH_INTERNAL_ASSERT(d.is_privateuseone());
  int old_device_index = ExchangeDevice(d.index());
  return c10::Device(c10::DeviceType::PrivateUse1, old_device_index);
}

c10::Device GCUGuardImpl::getDevice() const {
  c10::DeviceIndex device;
  C10_GCU_CHECK(GetDevice(&device));
  return c10::Device(c10::DeviceType::PrivateUse1, device);
}

c10::optional<c10::Device> GCUGuardImpl::uncheckedGetDevice() const noexcept {
  c10::DeviceIndex device;
  const auto err = GetDevice(&device);
  C10_GCU_CHECK_WARN(err);
  if (err != topsSuccess) {
    return c10::nullopt;
  }
  return c10::Device(c10::DeviceType::PrivateUse1, device);
}

void GCUGuardImpl::setDevice(c10::Device d) const {
  TORCH_INTERNAL_ASSERT(d.is_privateuseone());
  C10_GCU_CHECK(SetDevice(d.index()));
}

void GCUGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept {
  C10_GCU_CHECK_WARN(MaybeSetDevice(d.index()));
}

c10::DeviceIndex GCUGuardImpl::deviceCount() const noexcept {
  return device_count();
}

c10::Stream GCUGuardImpl::getNewStream(c10::Device d, int priority) const {
  return getStreamFromPool(priority, d.index());
}

// Event-related functions
void GCUGuardImpl::createEvent(topsEvent_t* tops_event,
                               const c10::EventFlag flag) const {
  // Maps PyTorch's Event::Flag to GCU flag
  auto gcu_flag = topsEventDefault;
  switch (flag) {
    case c10::EventFlag::PYTORCH_DEFAULT:
      gcu_flag = topsEventDisableTiming;
      break;
    case c10::EventFlag::BACKEND_DEFAULT:
      gcu_flag = topsEventDefault;
      break;
    default:
      TORCH_CHECK(false, "GCU event received unknown flag");
  }

  C10_GCU_CHECK(topsEventCreateWithFlags(tops_event, gcu_flag));
  // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
  // if (C10_UNLIKELY(interp)) {
  //   (*interp)->trace_gcu_event_creation(
  //       reinterpret_cast<uintptr_t>(tops_event));
  // }
}

void GCUGuardImpl::destroyEvent(
    void* event, const c10::DeviceIndex device_index) const noexcept {
  if (!event) return;
  auto tops_event = static_cast<topsEvent_t>(event);
  c10::DeviceIndex orig_device;
  C10_GCU_CHECK_WARN(GetDevice(&orig_device));
  C10_GCU_CHECK_WARN(SetDevice(device_index));
  // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
  // if (C10_UNLIKELY(interp)) {
  //   (*interp)->trace_gcu_event_deletion(
  //       reinterpret_cast<uintptr_t>(tops_event));
  // }
  C10_GCU_CHECK_WARN(topsEventDestroy(tops_event));
  C10_GCU_CHECK_WARN(SetDevice(orig_device));
}

void GCUGuardImpl::recordDataPtrOnStream(const c10::DataPtr& data_ptr,
                                         const c10::Stream& stream) const {
  GCUStream gcu_stream{stream};
  GCUCachingAllocator::recordStream(data_ptr, gcu_stream);
}

void GCUGuardImpl::record(void** event, const c10::Stream& stream,
                          const c10::DeviceIndex device_index,
                          const c10::EventFlag flag) const {
  TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
              "Event device index ", device_index,
              " does not match recording stream's device index ",
              stream.device_index(), ".");

  topsEvent_t gcu_event = static_cast<topsEvent_t>(*event);
  GCUStream gcu_stream{stream};

  // Moves to stream's device to record
  const auto orig_device = getDevice();
  setDevice(stream.device());

  // Creates the event (lazily)
  if (!gcu_event) createEvent(&gcu_event, flag);
  C10_GCU_CHECK(topsEventRecord(gcu_event, gcu_stream));
  // Makes the void* point to the (possibly just allocated) GCU event
  *event = gcu_event;
  // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
  // if (C10_UNLIKELY(interp)) {
  //   (*interp)->trace_gcu_event_record(
  //       reinterpret_cast<uintptr_t>(gcu_event),
  //       reinterpret_cast<uintptr_t>(gcu_stream.stream()));
  // }

  // Resets device
  setDevice(orig_device);
}

void GCUGuardImpl::block(void* event, const c10::Stream& stream) const {
  if (!event) return;
  topsEvent_t gcu_event = static_cast<topsEvent_t>(event);
  GCUStream gcu_stream{stream};
  const auto orig_device = getDevice();
  setDevice(stream.device());
  C10_GCU_CHECK(topsStreamWaitEvent(gcu_stream, gcu_event,
                                    /*flags (must be zero)=*/0));
  // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
  // if (C10_UNLIKELY(interp)) {
  //   (*interp)->trace_gcu_event_wait(
  //       reinterpret_cast<uintptr_t>(gcu_event),
  //       reinterpret_cast<uintptr_t>(gcu_stream.stream()));
  // }
  setDevice(orig_device);
}

bool GCUGuardImpl::queryEvent(void* event) const {
  if (!event) return true;
  topsEvent_t gcu_event = static_cast<topsEvent_t>(event);
  const topsError_t err = topsEventQuery(gcu_event);
  if (err != topsErrorNotReady) {
    C10_GCU_CHECK(err);
  } else {
    // ignore and clear the error if not ready
    (void)topsGetLastError();
  }
  return (err == topsSuccess);
}

void GCUGuardImpl::synchronizeEvent(void* event) const {
  if (!event) return;
  topsEvent_t gcu_event = static_cast<topsEvent_t>(event);
  C10_GCU_CHECK(topsEventSynchronize(gcu_event));
}

void GCUGuardImpl::synchronizeDevice(
    const c10::DeviceIndex device_index) const {
  TORCH_CHECK(device_index >= 0, "device_index must be non-negative");
  const auto orig_device = getDevice();
  const auto target_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_index);
  if (orig_device != target_device) {
    setDevice(target_device);
  }
  C10_GCU_CHECK(topsDeviceSynchronize());
  if (orig_device != target_device) {
    setDevice(orig_device);
  }
}

double GCUGuardImpl::elapsedTime(void* event1, void* event2,
                                 const c10::DeviceIndex device_index) const {
  TORCH_CHECK(event1 && event2,
              "Both events must be recorded before calculating elapsed time.");
  TORCH_CHECK(device_index >= 0, "device_index must be non-negative");
  const auto orig_device = getDevice();
  const auto target_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_index);
  if (orig_device != target_device) {
    setDevice(target_device);
  }
  auto gcu_event1 = static_cast<topsEvent_t>(event1);
  auto gcu_event2 = static_cast<topsEvent_t>(event2);
  float time_ms = 0.0f;
  C10_GCU_CHECK(topsEventElapsedTime(&time_ms, gcu_event1, gcu_event2));
  if (orig_device != target_device) {
    setDevice(orig_device);
  }
  return static_cast<double>(time_ms);
}

}  // namespace torch_gcu

namespace c10 {

C10_REGISTER_GUARD_IMPL(PrivateUse1, torch_gcu::GCUGuardImpl);

}  // namespace c10