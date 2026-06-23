#pragma once

#include <memory>
#include <string>

// Skip Kineto dependency on mobile unless explicitly asked for.
// When is it explicitly asked for?
//   KinetoEdgeCPUProfiler uses KinetoProfiler for cpu
//   event profiling. This has a dependency on cpu only libkineto

#include <libkineto_gcu/include/ActivityType.h>
#include <torch/csrc/Export.h>
#include <torch_gcu/csrc/profiler/api.h>

// Forward declarations so we don't have to include `libkineto.h` in a header.
namespace libkineto_gcu {
class GenericTraceActivity;
struct CpuTraceBuffer;
class ActivityTraceInterface;
}  // namespace libkineto_gcu

namespace torch_gcu {
namespace profiler {

constexpr bool kKinetoAvailable{true};

namespace impl::kineto {

// ----------------------------------------------------------------------------
// -- Interface (Does not require Kineto) -------------------------------------
// ----------------------------------------------------------------------------
struct DeviceAndResource {
  int32_t device;
  int32_t resource;
};
const DeviceAndResource kineto_ids();

using trace_t = libkineto_gcu::CpuTraceBuffer;
using interface_trace_t = libkineto_gcu::ActivityTraceInterface;
using activity_t = libkineto_gcu::GenericTraceActivity;

void addMetadata(activity_t* activity, const std::string& key,
                 const std::string& value);

// Wraps: libkineto_gcu::CpuTraceBuffer
struct TraceWrapper {
  TraceWrapper(const int64_t start_time, const std::string& name);
  TraceWrapper(TraceWrapper&&) = default;
  TraceWrapper(const TraceWrapper&) = delete;
  ~TraceWrapper();

  // The caller is expected to hold a mutex when calling `addCPUActivity`.
  activity_t* addCPUActivity(const std::string& name,
                             const libkineto_gcu::ActivityType type,
                             const DeviceAndResource device_and_resource,
                             const uint64_t correlation_id,
                             const int64_t start_time, const int64_t end_time);

  void transferCpuTrace(int64_t end_time);

  explicit operator bool() const;

  std::unique_ptr<trace_t>& get() { return cpu_trace_; }

 private:
  std::unique_ptr<trace_t> cpu_trace_;
};

// Wraps libkineto_gcu::ActivityTraceInterface
struct ActivityTraceWrapper {
  explicit ActivityTraceWrapper(std::unique_ptr<interface_trace_t>&& trace);
  ActivityTraceWrapper() = default;
  ActivityTraceWrapper(ActivityTraceWrapper&&) = default;
  ActivityTraceWrapper(const ActivityTraceWrapper&) = delete;
  explicit operator bool() const;
  void save(const std::string& path);

  const std::unique_ptr<interface_trace_t>& get() { return trace_; }

 private:
  std::unique_ptr<interface_trace_t> trace_;
  bool saved_ = false;  // Kineto's save is destructive
};

using ActivitySet = std::set<torch_gcu::autograd::profiler::ActivityType>;
void prepareTrace(const bool cpuOnly, const ActivitySet& activities,
                  const torch_gcu::profiler::impl::ExperimentalConfig& config);

void toggleCollectionDynamic(const bool enable);
void startTrace();
ActivityTraceWrapper stopTrace();
void pushCorrelationId(uint64_t correlation_id);
void pushUserCorrelationId(uint64_t correlation_id);
void popCorrelationId();
void popUserCorrelationId();
void recordThreadInfo();
bool collectivesProfilerExists();

void logInvariantViolation(const std::string& assertion,
                           const std::string& error,
                           const std::string& profile_id,
                           const std::string& group_profile_id);

}  // namespace impl::kineto

}  // namespace profiler

namespace autograd::profiler {
c10::DeviceType deviceTypeFromActivity(
    libkineto_gcu::ActivityType activity_type);

TORCH_API void addMetadataJson(const std::string& key,
                               const std::string& value);

TORCH_API void profilerStep();

}  // namespace autograd::profiler

}  // namespace torch_gcu
