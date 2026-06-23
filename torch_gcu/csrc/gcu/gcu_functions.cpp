#include "gcu/gcu_functions.h"

#include <c10/util/irange.h>
#include <tops/tops_ext.h>

#include "gcu/gcu_exception.h"
#include "gcu/gcu_generator_impl.h"

namespace torch_gcu {

namespace {

// returns -1 on failure
int32_t driver_version() {
  int driver_version = -1;
  C10_GCU_IGNORE_ERROR(topsDriverGetVersion(&driver_version));
  return driver_version;
}

int device_count_impl(bool fail_if_no_driver) {
  int count = 0;
  auto err = C10_GCU_ERROR_HANDLED(GetDeviceCount(&count));
  if (err == topsSuccess) {
    return count;
  }
  // Clear out the error state, so we don't spuriously trigger someone else.
  // (This shouldn't really matter, since we won't be running very much GCU
  // code in this regime.)
  topsError_t last_err C10_UNUSED = topsGetLastError();
  switch (err) {
    case topsErrorNoDevice:
      // Zero devices is ok here
      count = 0;
      break;
    case topsErrorInsufficientDriver: {
      auto version = driver_version();
      if (version <= 0) {
        if (!fail_if_no_driver) {
          // No GCU driver means no devices
          count = 0;
          break;
        }
        TORCH_CHECK(
            false,
            "Found no ENFLAME driver on your system. Please check that you "
            "have an ENFLAME GCU and installed a driver");
      } else {
        TORCH_CHECK(
            false,
            "The ENFLAME driver on your system is too old (found version ",
            version,
            "). Please update your GCU driver by downloading and installing "
            "a new version Alternatively, go to: "
            "https://pytorch.org to install a PyTorch version that has been "
            "compiled with your version of the GCU driver.");
      }
    } break;
    case topsErrorInitializationError:
      TORCH_CHECK(false,
                  "GCU driver initialization failed, you might not "
                  "have a gcu.");
      break;
    case topsErrorUnknown:
      TORCH_CHECK(false,
                  "GCU unknown error - this may be due to an "
                  "incorrectly set up environment, e.g. changing env "
                  "variable TOPS_VISIBLE_DEVICES after program start. "
                  "Setting the available devices to be zero.");
      break;
    default:
      TORCH_CHECK(false,
                  "Unexpected error from topsGetDeviceCount(). Did you run "
                  "some gcu functions before calling NumGcuDevices() "
                  "that might have already set an error? Error ",
                  err, ": ", topsGetErrorString(err));
  }
  return count;
}

}  // namespace

at::DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl(/*fail_if_no_driver=*/false);
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<at::DeviceIndex>::max(),
          "Too many GCU devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("GCU initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<at::DeviceIndex>(count);
}

at::DeviceIndex device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl(/*fail_if_no_driver=*/true);
  // Zero gcus doesn't produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No ENFLAME GCUs are available");
  return static_cast<at::DeviceIndex>(count);
}

at::DeviceIndex current_device() {
  c10::DeviceIndex cur_device = 0;
  C10_GCU_CHECK(GetDevice(&cur_device));
  return static_cast<at::DeviceIndex>(cur_device);
}

void set_device(at::DeviceIndex device) {
  C10_GCU_CHECK(SetDevice(static_cast<int>(device)));
}

void device_synchronize() { C10_GCU_CHECK(topsDeviceSynchronize()); }

// this function has to be called from callers performing gcu synchronizing
// operations, to raise proper error or warning
void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing GCU operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_WARN("called a synchronizing GCU operation");
  }
}

c10::optional<at::DeviceIndex> getDeviceIndexWithPrimaryContext() {
  // check current device first
  auto current_device_index = current_device();
  if (current_device_index >= 0) {
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (const auto device_index : c10::irange(device_count())) {
    if (device_index == current_device_index) continue;
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return c10::nullopt;
}

namespace _internal {

bool dummyHasPrimaryContext(C10_UNUSED at::DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < device_count(),
      "hasPrimaryContext expects a valid device index, but got device_index=",
      std::to_string(device_index));
  return true;
  // TORCH_CHECK(false, "Should never been called");
}

bool (*hasPrimaryContext)(at::DeviceIndex) = dummyHasPrimaryContext;

// Private api to be called from GCUHooks.cpp
void setHasPrimaryContext(bool (*func)(at::DeviceIndex)) {
  hasPrimaryContext = func ? func : dummyHasPrimaryContext;
}

}  // namespace _internal

bool hasPrimaryContext(at::DeviceIndex device_index) {
  return _internal::hasPrimaryContext(device_index);
}

topsError_t GetDeviceCount(int* dev_count) {
  return topsGetDeviceCount(dev_count);
}

thread_local at::DeviceIndex targetDeviceIndex = -1;

topsError_t GetDevice(at::DeviceIndex* device) {
  if (targetDeviceIndex >= 0) {
    *device = targetDeviceIndex;
    return topsSuccess;
  }
  int tmp_device = -1;
  auto err = topsGetDevice(&tmp_device);
  if (err == topsSuccess) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<at::DeviceIndex>::max(),
        "topsGetDevice returns invalid device ", tmp_device);
    *device = static_cast<at::DeviceIndex>(tmp_device);
  }
  return err;
}

topsError_t SetDevice(at::DeviceIndex device) {
  TORCH_CHECK(device >= 0, "device id must be positive!", device);
  targetDeviceIndex = -1;
  int cur_device = -1;
  C10_GCU_CHECK(topsGetDevice(&cur_device));
  if (device == cur_device) {
    return topsSuccess;
  }
  return topsSetDevice(device);
}

topsError_t MaybeSetDevice(at::DeviceIndex device) {
  if (hasPrimaryContext(device)) {
    return SetDevice(device);
  }
  targetDeviceIndex = device;
  return topsSuccess;
}

at::DeviceIndex ExchangeDevice(at::DeviceIndex to_device) {
  auto cur_device = targetDeviceIndex;
  targetDeviceIndex = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    C10_GCU_CHECK(topsGetDevice(&tmp_device));
    cur_device = static_cast<at::DeviceIndex>(tmp_device);
    if (to_device == cur_device) {
      return cur_device;
    }
  }
  C10_GCU_CHECK(topsSetDevice(to_device));
  return cur_device;
}

at::DeviceIndex MaybeExchangeDevice(at::DeviceIndex to_device) {
  int tmp_cur_device = -1;
  C10_GCU_CHECK(topsGetDevice(&tmp_cur_device));
  TORCH_INTERNAL_ASSERT(
      tmp_cur_device >= 0 &&
          tmp_cur_device <= std::numeric_limits<at::DeviceIndex>::max(),
      "topsGetDevice returns invalid device ", tmp_cur_device);
  auto cur_device = static_cast<at::DeviceIndex>(tmp_cur_device);
  if (to_device == tmp_cur_device) {
    return cur_device;
  }
  if (hasPrimaryContext(to_device)) {
    C10_GCU_CHECK(topsSetDevice(to_device));
  } else {
    targetDeviceIndex = to_device;
  }
  return cur_device;
}

void SetTargetDevice() {
  if (targetDeviceIndex >= 0) {
    C10_GCU_CHECK(SetDevice(targetDeviceIndex));
  }
}

bool is_available() { return device_count() > 0; }

void manual_seed(uint64_t seed) {
  if (is_available()) {
    auto index = current_device();
    auto gen = getDefaultGCUGenerator(index);
    {
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

void manual_seed_all(uint64_t seed) {
  auto num_gcu = device_count();
  for (const auto i : c10::irange(num_gcu)) {
    auto gen = getDefaultGCUGenerator(i);
    {
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

}  // namespace torch_gcu
