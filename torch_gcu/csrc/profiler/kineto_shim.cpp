#include <c10/util/Exception.h>
#include <torch_gcu/csrc/profiler/collection.h>
#include <torch_gcu/csrc/profiler/kineto_shim.h>

#include "libkineto_gcu.h"

namespace torch_gcu {

namespace profiler::impl::kineto {

// Here lies pain and `#ifdef USE_KINETO`

namespace {
const std::set<libkineto_gcu::ActivityType> kCpuTypes{
    libkineto_gcu::ActivityType::CPU_OP,
    libkineto_gcu::ActivityType::CPU_INSTANT_EVENT,
    libkineto_gcu::ActivityType::USER_ANNOTATION,
    libkineto_gcu::ActivityType::EXTERNAL_CORRELATION,
    libkineto_gcu::ActivityType::XPU_RUNTIME,
    libkineto_gcu::ActivityType::GCU_RUNTIME,
    libkineto_gcu::ActivityType::GCU_DRIVER,
    libkineto_gcu::ActivityType::PYTHON_FUNCTION,
    libkineto_gcu::ActivityType::PRIVATEUSE1_RUNTIME,
    libkineto_gcu::ActivityType::PRIVATEUSE1_DRIVER,
};

const std::set<libkineto_gcu::ActivityType> kGcuTypes = {
    libkineto_gcu::ActivityType::GCU_MEMCPY,
    libkineto_gcu::ActivityType::GCU_MEMSET,
    libkineto_gcu::ActivityType::GCU_USER_ANNOTATION,
    libkineto_gcu::ActivityType::CONCURRENT_KERNEL,
    // GCU_RUNTIME appears in both kCpuTypes and kGcuTypes.
    libkineto_gcu::ActivityType::GCU_RUNTIME,
    libkineto_gcu::ActivityType::GCU_DRIVER,
};
const std::set<libkineto_gcu::ActivityType> kXpuTypes = {
    libkineto_gcu::ActivityType::GCU_MEMCPY,
    libkineto_gcu::ActivityType::GCU_MEMSET,
    libkineto_gcu::ActivityType::CONCURRENT_KERNEL,
    // XPU_RUNTIME appears in both kCpuTypes and kXpuTypes.
    libkineto_gcu::ActivityType::XPU_RUNTIME,
};
const std::set<libkineto_gcu::ActivityType> kMtiaTypes = {
    libkineto_gcu::ActivityType::MTIA_CCP_EVENTS,
    libkineto_gcu::ActivityType::MTIA_RUNTIME,
    libkineto_gcu::ActivityType::MTIA_WORKLOADD,
};
const std::set<libkineto_gcu::ActivityType> kPrivateUse1Types = {
    libkineto_gcu::ActivityType::GCU_MEMCPY,
    libkineto_gcu::ActivityType::GCU_MEMSET,
    libkineto_gcu::ActivityType::GCU_USER_ANNOTATION,
    libkineto_gcu::ActivityType::CONCURRENT_KERNEL,
    // PRIVATEUSE1_RUNTIME appears in both kCpuTypes and kPrivateUse1Types.
    libkineto_gcu::ActivityType::PRIVATEUSE1_RUNTIME,
    libkineto_gcu::ActivityType::PRIVATEUSE1_DRIVER,
};
}  // namespace

static_assert(std::is_pod_v<DeviceAndResource>,
              "Kineto specific details should be in `kineto_ids`.");

const DeviceAndResource kineto_ids() {
  return {/*device=*/torch_gcu::util::processId(),
          /*resource=*/torch_gcu::util::systemThreadId()};
}

void addMetadata(activity_t* activity, const std::string& key,
                 const std::string& value) {
  activity->addMetadata(key, value);
}

TraceWrapper::TraceWrapper(const int64_t start_time, const std::string& name)
    : cpu_trace_(std::make_unique<libkineto_gcu::CpuTraceBuffer>()) {
  cpu_trace_->span.startTime = start_time;
  cpu_trace_->gcuOpCount = -1;
  cpu_trace_->span.name = name;
}

TraceWrapper::~TraceWrapper() = default;

activity_t* TraceWrapper::addCPUActivity(
    const std::string& name, const libkineto_gcu::ActivityType type,
    const DeviceAndResource device_and_resource, const uint64_t correlation_id,
    const int64_t start_time, const int64_t end_time) {
  TORCH_CHECK((bool)(*this), "Cannot add event to non-existent trace.");
  cpu_trace_->emplace_activity(cpu_trace_->span, type, name);
  auto& act =
      libkineto_gcu::CpuTraceBuffer::toRef(cpu_trace_->activities.back());
  act.device = device_and_resource.device;
  act.resource = device_and_resource.resource;
  act.id = static_cast<int32_t>(correlation_id);
  act.startTime = start_time;
  if (type != libkineto_gcu::ActivityType::CPU_INSTANT_EVENT) {
    act.endTime = end_time;
  }
  return cpu_trace_->activities.back().get();
}

void TraceWrapper::transferCpuTrace(int64_t end_time) {
  cpu_trace_->span.endTime = end_time;
  libkineto_gcu::api().activityProfiler().transferCpuTrace(
      std::move(cpu_trace_));
}

TraceWrapper::operator bool() const { return cpu_trace_ != nullptr; }

ActivityTraceWrapper::ActivityTraceWrapper(
    std::unique_ptr<interface_trace_t>&& trace)
    : trace_(std::move(trace)) {}

ActivityTraceWrapper::operator bool() const { return trace_ != nullptr; }

void ActivityTraceWrapper::save(const std::string& path) {
  TORCH_CHECK(!saved_, "Trace is already saved.");
  TORCH_CHECK(trace_ != nullptr, "Missing trace.")
  trace_->save(path);
  saved_ = true;
}

namespace {
// Handles processing of Experimental Config options for Kineto
class ExperimentalConfigWrapper {
 public:
  explicit ExperimentalConfigWrapper(
      const torch_gcu::profiler::impl::ExperimentalConfig& config)
      : config_(config) {}

  bool assertValid() { return !config_.profiler_metrics.empty(); }

  void prepareTraceWithExperimentalOptions(bool add_cpu_activity) {
    std::set<libkineto_gcu::ActivityType> k_activities{
        libkineto_gcu::ActivityType::GCU_PROFILER_RANGE};

    // Only add CPU activities if we are measuring per kernel ranges
    if (add_cpu_activity && config_.profiler_measure_per_kernel) {
      k_activities.insert(kCpuTypes.begin(), kCpuTypes.end());
    }

    const size_t num_metrics = config_.profiler_metrics.size();
    std::stringstream configss;

    LOG(INFO) << "CUPTI profiler metrics size = " << num_metrics;

    configss << "ACTIVITIES_WARMUP_PERIOD_SECS=0\n"
             << "CUPTI_PROFILER_METRICS=";

    for (size_t i = 0; i < num_metrics; i++) {
      configss << config_.profiler_metrics[i];
      if (num_metrics > 1 && i < (num_metrics - 1)) {
        configss << ",";
      }
    }
    configss << "\nCUPTI_PROFILER_ENABLE_PER_KERNEL="
             << (config_.profiler_measure_per_kernel ? "true" : "false")
             << "\n";
    LOG(INFO) << "Generated config = " << configss.str();

    libkineto_gcu::api().activityProfiler().prepareTrace(k_activities,
                                                         configss.str());
  }

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const torch_gcu::profiler::impl::ExperimentalConfig& config_;
};
}  // namespace

bool collectivesProfilerExists() {
#ifdef KINETO_HAS_NCCL_PROFILER
  return true;
#else
  return false;
#endif
}

void prepareTrace(const bool cpuOnly, const ActivitySet& activities,
                  const torch_gcu::profiler::impl::ExperimentalConfig& config) {
  if (!libkineto_gcu::api().isProfilerRegistered()) {
    libkineto_gcu_init(/*cpuOnly=*/cpuOnly, /*logOnError=*/true);
    libkineto_gcu::api().suppressLogMessages();
  }

  if (!libkineto_gcu::api().isProfilerInitialized()) {
    libkineto_gcu::api().initProfilerIfRegistered();
  }

  std::set<libkineto_gcu::ActivityType> k_activities;
  bool has_cpu_activity =
      activities.count(torch_gcu::autograd::profiler::ActivityType::CPU);

  if (has_cpu_activity) {
    k_activities.insert(kCpuTypes.begin(), kCpuTypes.end());
  }
  if (activities.count(torch_gcu::autograd::profiler::ActivityType::XPU)) {
    k_activities.insert(kXpuTypes.begin(), kXpuTypes.end());
  }
  if (activities.count(torch_gcu::autograd::profiler::ActivityType::MTIA)) {
    k_activities.insert(kMtiaTypes.begin(), kMtiaTypes.end());
  }
  if (activities.count(
          torch_gcu::autograd::profiler::ActivityType::PrivateUse1)) {
    k_activities.insert(kGcuTypes.begin(), kGcuTypes.end());
    if (config.enable_cuda_sync_events || get_gcu_sync_enabled()) {
      LOG(INFO) << "Enabling GCU Sync Events";
      k_activities.insert(libkineto_gcu::ActivityType::GCU_SYNC);
    }
  }
  if (collectivesProfilerExists()) {
    k_activities.insert(libkineto_gcu::ActivityType::COLLECTIVE_COMM);
  }
  // if (activities.count(
  //         torch_gcu::autograd::profiler::ActivityType::PrivateUse1)) {
  //   k_activities.insert(kPrivateUse1Types.begin(), kPrivateUse1Types.end());
  // }

  ExperimentalConfigWrapper configWrap(config);

  // Experimental Configuration options are present
  if (config && configWrap.assertValid()) {
    configWrap.prepareTraceWithExperimentalOptions(has_cpu_activity);
    return;
  }

  libkineto_gcu::api().activityProfiler().prepareTrace(k_activities);
}

void toggleCollectionDynamic(const bool enable) {
  // TODO: We may want to consider adding another input arg for this function
  // if we want to support turning off certain devices and keeping others on.
  // For now, we can keep it simple at have it turn off all tracing of "GCU"
  // devices
  libkineto_gcu::api().activityProfiler().toggleCollectionDynamic(enable);
}

void startTrace() { libkineto_gcu::api().activityProfiler().startTrace(); }

ActivityTraceWrapper stopTrace() {
  return ActivityTraceWrapper{
      libkineto_gcu::api().activityProfiler().stopTrace()};
}

void pushCorrelationId(uint64_t correlation_id) {
  libkineto_gcu::api().activityProfiler().pushCorrelationId(correlation_id);
}

void pushUserCorrelationId(uint64_t correlation_id) {
  libkineto_gcu::api().activityProfiler().pushUserCorrelationId(correlation_id);
}

void popCorrelationId() {
  libkineto_gcu::api().activityProfiler().popCorrelationId();
}

void popUserCorrelationId() {
  libkineto_gcu::api().activityProfiler().popUserCorrelationId();
}

void recordThreadInfo() {
  libkineto_gcu::api().activityProfiler().recordThreadInfo();
}

void logInvariantViolation(const std::string& assertion,
                           const std::string& error,
                           const std::string& profile_id,
                           const std::string& group_profile_id) {
  if (libkineto_gcu::api().isProfilerInitialized()) {
    libkineto_gcu::api().activityProfiler().logInvariantViolation(
        profile_id, assertion, error, group_profile_id);
  }
}

}  // namespace profiler::impl::kineto

namespace autograd::profiler {
c10::DeviceType deviceTypeFromActivity(
    libkineto_gcu::ActivityType activity_type) {
  // fallthrough
  switch (activity_type) {
    case libkineto_gcu::ActivityType::GCU_MEMCPY:
    case libkineto_gcu::ActivityType::GCU_MEMSET:
    case libkineto_gcu::ActivityType::CONCURRENT_KERNEL:
    case libkineto_gcu::ActivityType::GCU_SYNC:
    case libkineto_gcu::ActivityType::GCU_USER_ANNOTATION:
    case libkineto_gcu::ActivityType::GCU_PROFILER_RANGE: {
      // PrivateUse1 kineto backend reuse above ActivityTypes,
      // If PrivateUse1 backend enabled, this should return
      // c10::DeviceType::PrivateUse1.
      c10::DeviceType device_type = []() {
        if (c10::get_privateuse1_backend() != "privateuseone") {
          return c10::DeviceType::PrivateUse1;
        }
        return c10::DeviceType::PrivateUse1;
      }();
      return device_type;
    }
    // TODO: T151322015
    case libkineto_gcu::ActivityType::MTIA_CCP_EVENTS:
    case libkineto_gcu::ActivityType::MTIA_WORKLOADD: {
      // PrivateUse1 kineto backend reuse above ActivityTypes,
      // If PrivateUse1 backend enabled, this should return
      // c10::DeviceType::PrivateUse1.
      c10::DeviceType device_type = []() {
        if (c10::get_privateuse1_backend() != "privateuseone") {
          return c10::DeviceType::PrivateUse1;
        }
        return c10::DeviceType::MTIA;
      }();
      return device_type;
    }
    case libkineto_gcu::ActivityType::CPU_OP:
    case libkineto_gcu::ActivityType::USER_ANNOTATION:
    case libkineto_gcu::ActivityType::EXTERNAL_CORRELATION:
    case libkineto_gcu::ActivityType::GCU_RUNTIME:
    case libkineto_gcu::ActivityType::XPU_RUNTIME:
    case libkineto_gcu::ActivityType::CPU_INSTANT_EVENT:
    case libkineto_gcu::ActivityType::GLOW_RUNTIME:
    case libkineto_gcu::ActivityType::MTIA_RUNTIME:
    case libkineto_gcu::ActivityType::PYTHON_FUNCTION:
    case libkineto_gcu::ActivityType::GCU_DRIVER:
    case libkineto_gcu::ActivityType::PRIVATEUSE1_RUNTIME:
    case libkineto_gcu::ActivityType::PRIVATEUSE1_DRIVER:
      return c10::DeviceType::CPU;
    default: {
      TORCH_WARN("Unknown activity type (", (uint8_t)activity_type,
                 "), assuming CPU device");
      return c10::DeviceType::CPU;
    }
  }
}

void addMetadataJson(const std::string& key, const std::string& value) {
  if (libkineto_gcu::api().isProfilerInitialized()) {
    libkineto_gcu::api().activityProfiler().addMetadata(key, value);
  } else {
    LOG(WARNING) << "Profiler is not initialized: skipping profiling metadata";
  }
}

void profilerStep() {
  libkineto_gcu::api().initProfilerIfRegistered();

  if (libkineto_gcu::api().isProfilerInitialized()) {
    libkineto_gcu::api().activityProfiler().step();
  } else {
    VLOG(1) << "Profiler is not initialized: skipping step() invocation";
  }
}

}  // namespace autograd::profiler

}  // namespace torch_gcu
