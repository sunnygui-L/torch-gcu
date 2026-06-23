/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <fstream>
#include <map>
#include <ostream>
#include <thread>
#include <unordered_map>

#include "GenericTraceActivity.h"
#include "IActivityProfiler.h"
#include "TraceSpan.h"
#include "gcu/gcu_macros.h"
#include "gcu/thread_util.h"

namespace libkineto_gcu {
struct ActivityBuffers;
}

namespace libkineto_gcu {

using namespace libkineto_gcu;

// Used by sortIndex to put GCU tracks at the bottom
// of the trace timelines. The largest valid CPU PID is 4,194,304,
// so 5000000 is enough to guarantee that GCU tracks are sorted after CPU.
TORCH_GCU_API constexpr int64_t kExceedMaxPid = 5000000;

class TORCH_GCU_API ActivityLogger {
 public:
  virtual ~ActivityLogger() = default;

  struct OverheadInfo {
    explicit OverheadInfo(const std::string& name) : name(name) {}
    const std::string name;
  };

  virtual void handleDeviceInfo(const DeviceInfo& info, uint64_t time) = 0;

  virtual void handleResourceInfo(const ResourceInfo& info, int64_t time) = 0;

  virtual void handleOverheadInfo(const OverheadInfo& info, int64_t time) = 0;

  virtual void handleTraceSpan(const TraceSpan& span) = 0;

  virtual void handleActivity(
      const libkineto_gcu::ITraceActivity& activity) = 0;
  virtual void handleGenericActivity(
      const libkineto_gcu::GenericTraceActivity& activity) = 0;

  virtual void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) = 0;

  void handleTraceStart() {
    handleTraceStart(std::unordered_map<std::string, std::string>());
  }

  virtual void finalizeTrace(
      const Config& config, std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata) = 0;

 protected:
  ActivityLogger() = default;
};

}  // namespace libkineto_gcu
