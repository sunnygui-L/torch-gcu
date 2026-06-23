/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <tops/tops_runtime_api.h>

#include "gcu/gcu_exception.h"
#include "gcu/gcu_macros.h"
#include "gcu/runtime_wrapper.h"

namespace torch_gcu {

TORCH_GCU_API at::DeviceIndex device_count() noexcept;

TORCH_GCU_API at::DeviceIndex device_count_ensure_non_zero();

TORCH_GCU_API at::DeviceIndex current_device();

TORCH_GCU_API void set_device(at::DeviceIndex device);

TORCH_GCU_API void device_synchronize();

TORCH_GCU_API void warn_or_error_on_sync();

// Raw GCU device management functions
TORCH_GCU_API topsError_t GetDeviceCount(int* dev_count);
inline bool check_device(at::ArrayRef<at::Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  at::Device cur_device =
      at::Device(at::DeviceType::PrivateUse1, current_device());
  for (const at::Tensor& t : ts) {
    if (t.device() != cur_device) return false;
  }
  return true;
}

TORCH_GCU_API topsError_t GetDevice(at::DeviceIndex* device);

TORCH_GCU_API topsError_t SetDevice(at::DeviceIndex device);

TORCH_GCU_API topsError_t MaybeSetDevice(at::DeviceIndex device);

TORCH_GCU_API at::DeviceIndex ExchangeDevice(at::DeviceIndex device);

TORCH_GCU_API at::DeviceIndex MaybeExchangeDevice(at::DeviceIndex device);

TORCH_GCU_API void SetTargetDevice();

enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

// this is a holder for c10 global state (similar to at GlobalContext)
// currently it's used to store gcu synchronization warning state,
// but can be expanded to hold other related global state, e.g. to
// record stream usage
class WarningState {
 public:
  void set_sync_debug_mode(SyncDebugMode l) { sync_debug_mode = l; }

  SyncDebugMode get_sync_debug_mode() { return sync_debug_mode; }

 private:
  SyncDebugMode sync_debug_mode = SyncDebugMode::L_DISABLED;
};

__inline__ WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}

void __inline__ memcpy_and_sync(void* dst, const void* src, int64_t nbytes,
                                topsMemcpyKind kind, topsStream_t stream) {
  if (C10_UNLIKELY(warning_state().get_sync_debug_mode() !=
                   SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
  // if (C10_UNLIKELY(interp)) {
  //   (*interp)->trace_gcu_stream_synchronization(
  //       reinterpret_cast<uintptr_t>(stream));
  // }
  C10_GCU_CHECK(topsMemcpyAsync(dst, src, nbytes, kind, stream));
  StreamSynchronize(stream);
}

void __inline__ stream_synchronize(topsStream_t stream) {
  if (C10_UNLIKELY(warning_state().get_sync_debug_mode() !=
                   SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
  // if (C10_UNLIKELY(interp)) {
  //   (*interp)->trace_gcu_stream_synchronization(
  //       reinterpret_cast<uintptr_t>(stream));
  // }
  StreamSynchronize(stream);
}

TORCH_GCU_API bool hasPrimaryContext(at::DeviceIndex device_index);

c10::optional<at::DeviceIndex> getDeviceIndexWithPrimaryContext();

TORCH_GCU_API bool is_available();

TORCH_GCU_API void manual_seed(uint64_t seed);

TORCH_GCU_API void manual_seed_all(uint64_t seed);

}  // namespace torch_gcu
