#pragma once

#include <ATen/record_function.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/profiler/api.h>

#include <utility>
namespace torch_gcu {
namespace profiler {
namespace impl {

// ----------------------------------------------------------------------------
// -- Profiler Config ---------------------------------------------------------
// ----------------------------------------------------------------------------
using ActivityType = torch::profiler::impl::ActivityType;

inline std::string actToString(ActivityType t) {
  const std::string ActivityTypeNames[] = {"CPU", "XPU", "CUDA", "MTIA",
                                           "PrivateUse1"};
  return ActivityTypeNames[static_cast<int>(t)];
}

using ProfilerState = torch::profiler::impl::ProfilerState;

enum class C10_API_ENUM ActiveProfilerType {
  NONE = 0,
  LEGACY,
  KINETO,
  NVTX,
  ITT,
  PRIVATEUSE1
};

using ExperimentalConfig = torch::profiler::impl::ExperimentalConfig;
using ProfilerConfig = torch::profiler::impl::ProfilerConfig;

// ----------------------------------------------------------------------------
// -- Profiler base class -----------------------------------------------------
// ----------------------------------------------------------------------------
struct TORCH_API ProfilerStateBase : public c10::MemoryReportingInfoBase {
  explicit ProfilerStateBase(ProfilerConfig config);
  ~ProfilerStateBase() override;

  static ProfilerStateBase* get(bool global);
  static ProfilerStateBase* get() {
    auto* out = get(/*global=*/true);
    return out ? out : get(/*global=*/false);
  }

  static void push(std::shared_ptr<ProfilerStateBase>&& state);

  static std::shared_ptr<ProfilerStateBase> pop(bool global);
  static std::shared_ptr<ProfilerStateBase> pop() {
    auto out = pop(/*global=*/true);
    return out ? std::move(out) : pop(/*global=*/false);
  }

  const ProfilerConfig& config() const { return config_; }

  void setCallbackHandle(at::CallbackHandle handle);
  void removeCallback();

  bool memoryProfilingEnabled() const override {
    return config_.profile_memory;
  }

  virtual ActiveProfilerType profilerType() = 0;

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::mutex state_mutex_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  ProfilerConfig config_ = ProfilerConfig(ProfilerState::Disabled);
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  at::CallbackHandle handle_ = 0;
};

// Note: The following are only for the active *thread local* profiler.
TORCH_API bool profilerEnabled();
TORCH_API ActiveProfilerType profilerType();
TORCH_API ProfilerConfig getProfilerConfig();

}  // namespace impl
}  // namespace profiler
}  // namespace torch_gcu
