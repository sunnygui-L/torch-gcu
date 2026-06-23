#include "gcu/gcu_hooks.h"

#include <ATen/DeviceGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/util/CallOnce.h>

#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_caching_host_allocator.h"
#include "gcu/gcu_context.h"
#include "gcu/gcu_device.h"
#include "gcu/gcu_functions.h"
#include "gcu/gcu_generator_impl.h"
#include "gcu/gcu_peer_to_peer_access.h"

namespace torch_gcu {

void GCUHooks::init() const {
  C10_LOG_API_USAGE_ONCE("aten.init.gcu");
  const auto num_devices = device_count_ensure_non_zero();
  GCUCachingAllocator::init(num_devices);
  detail::init_p2p_access_cache(num_devices);
}

at::Generator GCUHooks::getNewGenerator(c10::DeviceIndex device_index) const {
  return at::make_generator<torch_gcu::GCUGeneratorImpl>(device_index);
}

const at::Generator& GCUHooks::getDefaultGenerator(
    at::DeviceIndex device_index) const {
  at::globalContext().lazyInitDevice(at::kPrivateUse1);
  return getDefaultGCUGenerator(device_index);
}

at::Device GCUHooks::getDeviceFromPtr(void* data) const {
  at::globalContext().lazyInitDevice(at::kPrivateUse1);
  return torch_gcu::getDeviceFromPtr(data);
}

bool GCUHooks::isPinnedPtr(const void* data) const {
  if (!is_available()) {
    return false;
  }

  // at::OptionalDeviceGuard device_guard;
  at::OptionalDeviceGuard device_guard(at::Device(at::DeviceType::PrivateUse1));
  auto primary_ctx_device_index = getDeviceIndexWithPrimaryContext();
  if (primary_ctx_device_index.has_value()) {
    device_guard.reset_device(
        at::Device(at::DeviceType::PrivateUse1, *primary_ctx_device_index));
  }
  topsPointerAttribute_t attr;
  topsError_t err = topsPointerGetAttributes(&attr, data);
  if (err == topsErrorInvalidValue) {
    (void)topsGetLastError();  // clear GCU error
    return false;
  }
  C10_GCU_CHECK(err);
  return attr.memoryType == topsMemoryTypeHost;
}

bool GCUHooks::hasPrimaryContext(at::DeviceIndex device_index) const {
  return torch_gcu::hasPrimaryContext(device_index);
};

at::DeviceIndex GCUHooks::current_device() const {
  c10::DeviceIndex device;
  topsError_t err = GetDevice(&device);
  if (err == topsSuccess) {
    return device;
  }
  return -1;
}

at::Allocator* GCUHooks::getPinnedMemoryAllocator() const {
  return getCachingHostAllocator();
}

at::Allocator* GCUHooks::getGCUDeviceAllocator() const {
  return getGCUDeviceAllocator();
}

std::string GCUHooks::showConfig() const {
  std::ostringstream oss;

  int runtimeVersion;
  C10_GCU_CHECK(topsRuntimeGetVersion(&runtimeVersion));

  auto printGcuStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };

  oss << "  - GCU Runtime ";
  printGcuStyleVersion(runtimeVersion);
  oss << "\n";

  // maybe add topsflame version

  return oss.str();
}

bool GCUHooks::hasGCU() const { return is_available(); }

int GCUHooks::getNumGCUs() const { return device_count(); }

void GCUHooks::deviceSynchronize(at::DeviceIndex device_index) const {
  at::DeviceGuard device_guard(
      at::Device(at::DeviceType::PrivateUse1, device_index));
  device_synchronize();
}

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, GCUHooks, GCUHooksArgs);
#define REGISTER_PRIVATEUSE1_HOOKS(clsname) \
  C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, GCUHooks, GCUHooksArgs)

namespace detail {

static GCUHooks* gcu_hooks = nullptr;

const GCUHooks& getGCUHooks() {
  static c10::once_flag once;
  c10::call_once(once, [] {
    gcu_hooks =
        PrivateUse1HooksRegistry()->Create("PrivateUse1Hooks", {}).release();
    if (!gcu_hooks) {
      gcu_hooks = new GCUHooks();
    }
  });
  return *gcu_hooks;
}

void RegisterGCUHooks() {
  getGCUHooks();
  at::RegisterPrivateUse1HooksInterface(gcu_hooks);
}

}  // namespace detail

}  // namespace torch_gcu
