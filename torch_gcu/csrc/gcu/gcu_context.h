/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/Context.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Logging.h>

#include "gcu/gcu_exception.h"
#include "gcu/gcu_hooks.h"
#include "gcu/gcu_macros.h"

namespace torch_gcu {

topsDeviceProp_t* getCurrentDeviceProperties();

TORCH_GCU_API topsDeviceProp_t* getDeviceProperties(c10::DeviceIndex device);

TORCH_GCU_API bool canDeviceAccessPeer(c10::DeviceIndex device,
                                       c10::DeviceIndex peer_device);

c10::Allocator* getGCUDeviceAllocator();

TORCH_GCU_API void registerGcuAllocator();

}  // namespace torch_gcu
