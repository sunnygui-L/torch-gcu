#include "gcu/gcu_context.h"

#include <c10/core/Allocator.h>

#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_functions.h"

namespace torch_gcu {

namespace {

c10::DeviceIndex num_gcus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<topsDeviceProp_t> device_properties;

void initGCUContextVectors() {
  num_gcus = torch_gcu::device_count();
  device_flags.resize(num_gcus);
  device_properties.resize(num_gcus);
}

void initDeviceProperty(c10::DeviceIndex device_index) {
  topsDeviceProp_t device_prop;
  C10_GCU_CHECK(topsGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

}  // anonymous namespace

topsDeviceProp_t* getCurrentDeviceProperties() {
  auto device = torch_gcu::current_device();
  return getDeviceProperties(device);
}

topsDeviceProp_t* getDeviceProperties(c10::DeviceIndex device) {
  c10::call_once(init_flag, initGCUContextVectors);
  if (device == -1) device = torch_gcu::current_device();
  AT_ASSERT(device >= 0 && device < num_gcus, "device=", device,
            ", num_gcus=", num_gcus);
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

bool canDeviceAccessPeer(c10::DeviceIndex device,
                         c10::DeviceIndex peer_device) {
  c10::call_once(init_flag, initGCUContextVectors);
  if (device == -1) device = torch_gcu::current_device();
  AT_ASSERT(device >= 0 && device < num_gcus, "device=", device,
            ", num_gcus=", num_gcus);
  AT_ASSERT(peer_device >= 0 && peer_device < num_gcus,
            "peer_device=", peer_device, ", num_gcus=", num_gcus);
  int can_access = 0;
  C10_GCU_CHECK(topsDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

c10::Allocator* getGCUDeviceAllocator() { return GCUCachingAllocator::get(); }

void registerGcuAllocator() {
  c10::SetAllocator(c10::DeviceType::PrivateUse1, getGCUDeviceAllocator());
}

}  // namespace torch_gcu
