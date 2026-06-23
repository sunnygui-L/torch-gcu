/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <array>
#include <set>
#include <string>

#include "gcu/gcu_macros.h"

namespace libkineto_gcu {

// Note : All activity types are not enabled by default. Please add them
// at correct position in the enum
enum class TORCH_GCU_API ActivityType {
  // Activity types enabled by default
  CPU_OP = 0,  // cpu side ops
  USER_ANNOTATION,
  GCU_USER_ANNOTATION,
  GCU_MEMCPY,
  GCU_MEMSET,
  CONCURRENT_KERNEL,  // on-device kernels
  EXTERNAL_CORRELATION,
  GCU_RUNTIME,        // host side gcu runtime events
  GCU_DRIVER,         // host side gcu driver events
  CPU_INSTANT_EVENT,  // host side point-like events
  PYTHON_FUNCTION,
  OVERHEAD,      // CUPTI induced overhead events sampled from its overhead API.
  MTIA_RUNTIME,  // host side MTIA runtime events
  MTIA_CCP_EVENTS,  // MTIA ondevice CCP events
  GCU_SYNC,         // synchronization events between runtime and kernels

  // Optional Activity types
  GLOW_RUNTIME,        // host side glow runtime events
  GCU_PROFILER_RANGE,  // CUPTI Profiler range for performance metrics
  HPU_OP,              // HPU host side runtime event
  XPU_RUNTIME,         // host side xpu runtime events
  COLLECTIVE_COMM,     // collective communication
  MTIA_WORKLOADD,      // MTIA workloadd events

  // PRIVATEUSE1 Activity types are used for custom backends.
  // The corresponding device type is `DeviceType::PrivateUse1` in PyTorch.
  PRIVATEUSE1_RUNTIME,  // host side privateUse1 runtime events
  PRIVATEUSE1_DRIVER,   // host side privateUse1 driver events

  ENUM_COUNT,  // This is to add buffer and not used for any profiling logic.
               // Add your new type before it.
  OPTIONAL_ACTIVITY_TYPE_START = GLOW_RUNTIME,
};

TORCH_GCU_API const char* toString(ActivityType t);
TORCH_GCU_API ActivityType toActivityType(const std::string& str);

// Return an array of all activity types except COUNT
TORCH_GCU_API constexpr int activityTypeCount = (int)ActivityType::ENUM_COUNT;
TORCH_GCU_API constexpr int defaultActivityTypeCount =
    (int)ActivityType::OPTIONAL_ACTIVITY_TYPE_START;
TORCH_GCU_API const std::array<ActivityType, activityTypeCount> activityTypes();
TORCH_GCU_API const std::array<ActivityType, defaultActivityTypeCount>
defaultActivityTypes();

}  // namespace libkineto_gcu
