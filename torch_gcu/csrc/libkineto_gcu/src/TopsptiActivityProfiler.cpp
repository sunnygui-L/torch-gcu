/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "TopsptiActivityProfiler.h"

#include <fmt/format.h>
#include <time.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "ActivityBuffers.h"
#include "ApproximateClock.h"
#include "Config.h"
#include "DeviceUtil.h"
#include "Logger.h"
#include "TopsptiActivity.cpp"
#include "TopsptiActivity.h"
#include "TopsptiActivityApi.h"
#include "gcu/thread_util.h"
#include "output_base.h"
#include "time_since_epoch.h"
#include "topspti.h"

using namespace std::chrono;
using std::string;

struct CtxEventPair {
  uint32_t ctx = 0;
  uint32_t eventId = 0;

  bool operator==(const CtxEventPair& other) const {
    return (this->ctx == other.ctx) && (this->eventId == other.eventId);
  }
};

template <>
struct std::hash<CtxEventPair> {
  std::size_t operator()(const CtxEventPair& c) const {
    return libkineto_gcu::detail::hash_combine(
        std::hash<uint32_t>()(c.ctx), std::hash<uint32_t>()(c.eventId));
  }
};

struct WaitEventInfo {
  // GCU stream that the GCU event was recorded on
  uint32_t stream;
  // Correlation ID of the topsEventRecord event
  uint32_t correlationId;
};

namespace {

// Map (ctx, eventId) -> (stream, corr Id) that recorded the GCU event
std::unordered_map<CtxEventPair, WaitEventInfo>& waitEventMap() {
  static std::unordered_map<CtxEventPair, WaitEventInfo> waitEventMap_;
  return waitEventMap_;
}

// Map ctx -> deviceId
std::unordered_map<uint32_t, uint32_t>& ctxToDeviceId() {
  static std::unordered_map<uint32_t, uint32_t> ctxToDeviceId_;
  return ctxToDeviceId_;
}

}  // namespace

namespace libkineto_gcu {

// Sets the timestamp converter. If nothing is set then the converter just
// returns the input. For this reason, until we add profiler impl of passing in
// TSC converter we just need to guard the callback itself
std::function<time_t(approx_time_t)>& get_time_converter() {
  static std::function<time_t(approx_time_t)> _time_converter =
      [](approx_time_t t) { return t; };
  return _time_converter;
}

bool& use_topspti_tsc() {
  static bool use_topspti_tsc = false;
  return use_topspti_tsc;
}

ConfigDerivedState::ConfigDerivedState(const Config& config) {
  profileActivityTypes_ = config.selectedActivityTypes();
  profileStartTime_ = config.requestTimestamp();
  profileDuration_ = config.activitiesDuration();
  profileWarmupDuration_ = config.activitiesWarmupDuration();
  profilingByIter_ = config.hasProfileStartIteration();
  if (profilingByIter_) {
    profileStartIter_ = config.profileStartIteration();
    profileEndIter_ = profileStartIter_ + config.activitiesRunIterations();
  } else {
    profileEndIter_ = (std::numeric_limits<decltype(profileEndIter_)>::max)();
    profileEndTime_ = profileStartTime_ + config.activitiesDuration();
  }
}

bool ConfigDerivedState::canStart(
    const std::chrono::time_point<std::chrono::system_clock>& now) const {
  if (profilingByIter_) {
    return true;
  }
  if (profileStartTime_ < now) {
    LOG(ERROR) << "Not starting tracing - start timestamp is in the past. Time "
                  "difference (ms): "
               << duration_cast<milliseconds>(now - profileStartTime_).count();
    return false;
  } else if ((profileStartTime_ - now) < profileWarmupDuration_) {
    LOG(ERROR) << "Not starting tracing - insufficient time for warmup. Time "
                  "to warmup (ms): "
               << duration_cast<milliseconds>(profileStartTime_ - now).count();
    return false;
  }
  return true;
}

bool ConfigDerivedState::isWarmupDone(const time_point<system_clock>& now,
                                      int64_t currentIter) const {
  bool isTimestampBased = !profilingByIter_ && currentIter < 0;
  if (isTimestampBased) {
    // qualify that this check is not being called from application step() API
    // this avoids races between the step() API and periodically invoked
    // profiler run loop step() method
    return now >= profileStartTime_;
  }
  bool isIterationBased = profilingByIter_ && currentIter >= 0;
  if (isIterationBased) {
    return currentIter >= profileStartIter_;
  }
  return false;
}

bool ConfigDerivedState::isCollectionDone(const time_point<system_clock>& now,
                                          int64_t currentIter) const {
  bool isTimestampBased = !profilingByIter_ && currentIter < 0;
  if (isTimestampBased) {
    // qualify that this check is not being called from application step() API
    return now >= profileEndTime_;
  }
  bool isIterationBased = profilingByIter_ && currentIter >= 0;
  if (isIterationBased) {
    return currentIter >= profileEndIter_;
  }
  return false;
}

std::ostream& operator<<(std::ostream& oss,
                         const TopsptiActivityProfiler::ErrorCounts& ecs) {
  oss << "Out-of-range = " << ecs.out_of_range_events
      << ", Blocklisted runtime = " << ecs.blocklisted_runtime_events
      << ", Invalid ext correlations = "
      << ecs.invalid_external_correlation_events
      << ", CPU GCU out-of-order = " << ecs.gcu_and_cpu_op_out_of_order
      << ", Unexpected GCU events = " << ecs.unexpected_gcu_events
      << ", TOPSPTI stopped early? = " << ecs.topspti_stopped_early;

  return oss;
}

void TopsptiActivityProfiler::transferCpuTrace(
    std::unique_ptr<libkineto_gcu::CpuTraceBuffer> cpuTrace) {
  std::lock_guard<std::mutex> guard(mutex_);
  const string& trace_name = cpuTrace->span.name;
  if (currentRunloopState_ != RunloopState::CollectTrace &&
      currentRunloopState_ != RunloopState::ProcessTrace) {
    VLOG(0) << "Trace collection not in progress - discarding span "
            << trace_name;
    return;
  }

  cpuTrace->span.iteration = iterationCountMap_[trace_name]++;

  VLOG(0) << "Received iteration " << cpuTrace->span.iteration << " of span "
          << trace_name << " (" << cpuTrace->activities.size()
          << " activities / " << cpuTrace->gcuOpCount << " gcu activities)";
  traceBuffers_->cpu.push_back(std::move(cpuTrace));
}

TopsptiActivityProfiler::TopsptiActivityProfiler(TopsptiActivityApi& topspti,
                                                 bool cpuOnly)
    : topspti_(topspti),
      flushOverhead_{0, 0},
      setupOverhead_{0, 0},
      cpuOnly_{cpuOnly},
      currentRunloopState_{RunloopState::WaitForRequest} {
  if (isGcuAvailable()) {
    logGcuVersions();
  }
}

void TopsptiActivityProfiler::logGcuVersions() {
  // check Nvidia versions
  uint32_t topsptiVersion = 0;
  int topsRuntimeVersion = 0, topsDriverVersion = 0;
  TOPSPTI_CALL(topsptiGetVersion(&topsptiVersion));
  GCU_CALL(topsRuntimeGetVersion(&topsRuntimeVersion));
  GCU_CALL(topsDriverGetVersion(&topsDriverVersion));
  LOG(INFO) << "GCU versions. TOPSPTI: " << topsptiVersion
            << "; Runtime: " << topsRuntimeVersion
            << "; Driver: " << topsDriverVersion;

  LOGGER_OBSERVER_ADD_METADATA("topspti_version",
                               std::to_string(topsptiVersion));
  LOGGER_OBSERVER_ADD_METADATA("tops_runtime_version",
                               std::to_string(topsRuntimeVersion));
  LOGGER_OBSERVER_ADD_METADATA("tops_driver_version",
                               std::to_string(topsDriverVersion));
}

void TopsptiActivityProfiler::processTraceInternal(ActivityLogger& logger) {
  LOG(INFO) << "Processing " << traceBuffers_->cpu.size() << " CPU buffers";
  VLOG(0) << "Profile time range: " << captureWindowStartTime_ << " - "
          << captureWindowEndTime_;
  logger.handleTraceStart(metadata_);
  setCpuActivityPresent(false);
  setGcuActivityPresent(false);
  for (auto& cpu_trace : traceBuffers_->cpu) {
    string trace_name = cpu_trace->span.name;
    VLOG(0) << "Processing CPU buffer for " << trace_name << " ("
            << cpu_trace->span.iteration << ") - "
            << cpu_trace->activities.size() << " records";
    VLOG(0) << "Span time range: " << cpu_trace->span.startTime << " - "
            << cpu_trace->span.endTime;
    processCpuTrace(*cpu_trace, logger);
    LOGGER_OBSERVER_ADD_EVENT_COUNT(cpu_trace->activities.size());
  }

  if (!cpuOnly_) {
    VLOG(0) << "Retrieving GCU activity buffers";
    traceBuffers_->gcu = topspti_.activityBuffers();
    if (VLOG_IS_ON(1)) {
      addOverheadSample(flushOverhead_, topspti_.flushOverhead);
    }
    if (traceBuffers_->gcu) {
      const auto count_and_size = topspti_.processActivities(
          *traceBuffers_->gcu,
          std::bind(&TopsptiActivityProfiler::handleTopsptiActivity, this,
                    std::placeholders::_1, &logger));
      logDeferredEvents();
      LOG(INFO) << "Processed " << count_and_size.first << " GCU records ("
                << count_and_size.second << " bytes)";
      LOGGER_OBSERVER_ADD_EVENT_COUNT(count_and_size.first);

      // resourceOverheadCount_ is set while processing GCU activities
      if (resourceOverheadCount_ > 0) {
        LOG(INFO) << "Allocated " << resourceOverheadCount_
                  << " extra TOPSPTI buffers.";
      }
      LOGGER_OBSERVER_ADD_METADATA("ResourceOverhead",
                                   std::to_string(resourceOverheadCount_));
    }
    if (!GcuActivityPresent()) {
      LOG(WARNING) << "GCU trace is empty!";
    }
  }

  if (!traceNonEmpty()) {
    LOG(WARNING)
        << "No Valid Trace Events (CPU/GCU) found. Outputting empty trace.";
  }

  for (const auto& session : sessions_) {
    LOG(INFO) << "Processing child profiler trace";
    // cpuActivity() function here is used to get the linked cpuActivity for
    // session's activities. Passing captureWindowStartTime_ and
    // captureWindowEndTime_ in order to specify the range of activities that
    // need to be processed.
    session->processTrace(logger,
                          std::bind(&TopsptiActivityProfiler::cpuActivity, this,
                                    std::placeholders::_1),
                          captureWindowStartTime_, captureWindowEndTime_);
  }

  LOG(INFO) << "Record counts: " << ecs_;

  finalizeTrace(*config_, logger);
}

TopsptiActivityProfiler::CpuGcuSpanPair&
TopsptiActivityProfiler::recordTraceSpan(TraceSpan& span, int gcuOpCount) {
  TraceSpan gcu_span(gcuOpCount, span.iteration, span.name, "GCU: ");
  auto& iterations = traceSpans_[span.name];
  iterations.push_back({span, gcu_span});
  return iterations.back();
}

void TopsptiActivityProfiler::processCpuTrace(
    libkineto_gcu::CpuTraceBuffer& cpuTrace, ActivityLogger& logger) {
  if (cpuTrace.activities.size() == 0) {
    LOG(WARNING) << "CPU trace is empty!";
    return;
  }
  setCpuActivityPresent(true);

  CpuGcuSpanPair& span_pair =
      recordTraceSpan(cpuTrace.span, cpuTrace.gcuOpCount);
  TraceSpan& cpu_span = span_pair.first;
  for (auto const& act : cpuTrace.activities) {
    VLOG(2) << act->correlationId() << ": OP " << act->activityName;
    if (derivedConfig_->profileActivityTypes().count(act->type())) {
      static_assert(
          std::is_same<std::remove_reference<decltype(act)>::type,
                       const std::unique_ptr<GenericTraceActivity>>::value,
          "handleActivity is unsafe and relies on the caller to maintain not "
          "only lifetime but also address stability.");
      logger.handleActivity(*act);
    }
    clientActivityTraceMap_[act->correlationId()] = &span_pair;
    activityMap_[act->correlationId()] = act.get();

    recordThreadInfo(act->resourceId(), act->getThreadId(), act->deviceId());
  }
  logger.handleTraceSpan(cpu_span);
}

// inline void TopsptiActivityProfiler::handleCorrelationActivity(
//     const Topspti_ActivityExternalCorrelation* correlation) {
//   if (correlation->externalKind == TOPSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0)
//   {
//     cpuCorrelationMap_[correlation->correlationId] = correlation->externalId;
//   } else if (correlation->externalKind ==
//              TOPSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1) {
//     userCorrelationMap_[correlation->correlationId] =
//     correlation->externalId;
//   } else {
//     LOG(WARNING) << "Invalid Topspti_ActivityExternalCorrelation sent to "
//                     "handleTopsptiActivity";
//     ecs_.invalid_external_correlation_events++;
//   }
// }

static GenericTraceActivity createUserGcuSpan(
    const libkineto_gcu::ITraceActivity& cpuTraceActivity,
    const libkineto_gcu::ITraceActivity& gcuTraceActivity) {
  GenericTraceActivity res(*cpuTraceActivity.traceSpan(),
                           ActivityType::GCU_USER_ANNOTATION,
                           cpuTraceActivity.name());
  res.startTime = gcuTraceActivity.timestamp();
  res.device = gcuTraceActivity.deviceId();
  res.resource = gcuTraceActivity.resourceId();
  res.endTime = gcuTraceActivity.timestamp() + gcuTraceActivity.duration();
  res.id = cpuTraceActivity.correlationId();
  return res;
}

void TopsptiActivityProfiler::GcuUserEventMap::insertOrExtendEvent(
    const ITraceActivity& userActivity, const ITraceActivity& GcuActivity) {
  StreamKey key(GcuActivity.deviceId(), GcuActivity.resourceId());
  CorrelationSpanMap& correlationSpanMap = streamSpanMap_[key];
  auto it = correlationSpanMap.find(userActivity.correlationId());
  if (it == correlationSpanMap.end()) {
    auto it_success = correlationSpanMap.insert(
        {userActivity.correlationId(),
         createUserGcuSpan(userActivity, GcuActivity)});
    it = it_success.first;
  }
  GenericTraceActivity& span = it->second;
  if (GcuActivity.timestamp() < span.startTime || span.startTime == 0) {
    span.startTime = GcuActivity.timestamp();
  }
  int64_t gcu_activity_end = GcuActivity.timestamp() + GcuActivity.duration();
  if (gcu_activity_end > span.endTime) {
    span.endTime = gcu_activity_end;
  }
}

const TopsptiActivityProfiler::CpuGcuSpanPair&
TopsptiActivityProfiler::defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  static CpuGcuSpanPair span_pair(span, span);
  return span_pair;
}

void TopsptiActivityProfiler::GcuUserEventMap::logEvents(
    ActivityLogger* logger) {
  for (auto const& streamMapPair : streamSpanMap_) {
    for (auto const& correlationSpanPair : streamMapPair.second) {
      correlationSpanPair.second.log(*logger);
    }
  }
}

inline bool TopsptiActivityProfiler::outOfRange(const ITraceActivity& act) {
  bool out_of_range =
      act.timestamp() < captureWindowStartTime_ ||
      (act.timestamp() + act.duration()) > captureWindowEndTime_;
  if (out_of_range) {
    VLOG(2) << "TraceActivity outside of profiling window: " << act.name()
            << " (" << act.timestamp() << " < " << captureWindowStartTime_
            << " or " << (act.timestamp() + act.duration()) << " > "
            << captureWindowEndTime_;
    ecs_.out_of_range_events++;
  }
  return out_of_range;
}

inline static bool isBlockListedRuntimeCbid(Topspti_CallbackId cbid) {
  // Some GCU calls that are very frequent and also not very interesting.
  // Filter these out to reduce trace size.
  // if (cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsGetDevice ||
  //     cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsSetDevice ||
  //     cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsGetLastError ||
  //     // Support topsEventRecord and topsEventSynchronize, revisit if others
  //     are
  //     // needed
  //     cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsEventCreate ||
  //     cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsEventCreateWithFlags ||
  //     cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsEventDestroy) {
  //   return true;
  // }
  return false;
}

void TopsptiActivityProfiler::handleRuntimeActivity(
    const Topspti_ActivityAPI* activity, ActivityLogger* logger) {
  if (isBlockListedRuntimeCbid(activity->cbid)) {
    ecs_.blocklisted_runtime_events++;
    return;
  }
  VLOG(2) << activity->correlationId
          << ": TOPSPTI_ACTIVITY_KIND_RUNTIME, cbid=" << activity->cbid
          << " tid=" << activity->threadId;
  int32_t tid = activity->threadId;
  const auto& it = resourceInfo_.find({torch_gcu::util::processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  const ITraceActivity* linked =
      linkedActivity(activity->correlationId, cpuCorrelationMap_);
  const auto& runtime_activity =
      traceBuffers_->addActivityWrapper(RuntimeActivity(activity, linked, tid));
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
  setGcuActivityPresent(true);
}

void TopsptiActivityProfiler::handleDriverActivity(
    const Topspti_ActivityAPI* activity, ActivityLogger* logger) {
  // we only want to collect cuLaunchKernel events, for triton kernel launches
  if (!isKernelLaunchApi(*activity)) {
    // XXX should we count other driver events?
    return;
  }
  VLOG(2) << activity->correlationId
          << ": TOPSPTI_ACTIVITY_KIND_DRIVER, cbid=" << activity->cbid
          << " tid=" << activity->threadId;
  int32_t tid = activity->threadId;
  const auto& it = resourceInfo_.find({torch_gcu::util::processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  const ITraceActivity* linked =
      linkedActivity(activity->correlationId, cpuCorrelationMap_);
  const auto& runtime_activity =
      traceBuffers_->addActivityWrapper(DriverActivity(activity, linked, tid));
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
  setGcuActivityPresent(true);
}

// void TopsptiActivityProfiler::handleOverheadActivity(
//     const Topspti_ActivityOverhead* activity, ActivityLogger* logger) {
//   VLOG(2) << ": TOPSPTI_ACTIVITY_KIND_OVERHEAD"
//           << " overheadKind=" << activity->overheadKind;
//   const auto& overhead_activity =
//       traceBuffers_->addActivityWrapper(OverheadActivity(activity, nullptr));
//   // Monitor memory overhead
//   if (activity->overheadKind == TOPSPTI_ACTIVITY_OVERHEAD_TOPSPTI_RESOURCE) {
//     resourceOverheadCount_++;
//   }

//   if (outOfRange(overhead_activity)) {
//     return;
//   }
//   overhead_activity.log(*logger);
//   setGcuActivityPresent(true);
// }

std::optional<WaitEventInfo> getWaitEventInfo(uint32_t ctx, uint32_t eventId) {
  auto key = CtxEventPair{ctx, eventId};
  auto it = waitEventMap().find(key);
  if (it != waitEventMap().end()) {
    return it->second;
  }
  return std::nullopt;
}

// void TopsptiActivityProfiler::handleGcuEventActivity(
//     const Topspti_ActivityTopsEvent* activity) {
//   VLOG(2) << ": TOPSPTI_ACTIVITY_KIND_GCU_EVENT"
//           << " corrId=" << activity->correlationId
//           << " eventId=" << activity->eventId
//           << " streamId=" << activity->streamId
//           << " contextId=" << activity->contextId;

//   // Update the stream, corrID the topsEvent was last recorded on
//   auto key = CtxEventPair{activity->contextId, activity->eventId};
//   waitEventMap()[key] =
//       WaitEventInfo{activity->streamId, activity->correlationId};
// }

// void TopsptiActivityProfiler::handleGcuSyncActivity(
//     const Topspti_ActivitySynchronization* activity, ActivityLogger* logger)
//     {
//   VLOG(2) << ": TOPSPTI_ACTIVITY_KIND_SYNCHRONIZATION"
//           << " type=" << syncTypeString(activity->type)
//           << " corrId=" << activity->correlationId
//           << " streamId=" << activity->streamId
//           << " eventId=" << activity->topsEventId
//           << " contextId=" << activity->contextId;

//   if (!config_->activitiesGcuSyncWaitEvents() &&
//       isWaitEventSync(activity->type)) {
//     return;
//   }

//   auto device_id = contextIdtoDeviceId(activity->contextId);
//   int32_t src_stream = -1, src_corrid = -1;

//   if (isEventSync(activity->type)) {
//     auto maybe_wait_event_info =
//         getWaitEventInfo(activity->contextId, activity->gcuEventId);
//     if (maybe_wait_event_info) {
//       src_stream = maybe_wait_event_info->stream;
//       src_corrid = maybe_wait_event_info->correlationId;
//     }
//   }

//   // Marshal the logging to a functor so we can defer it if needed.
//   auto log_event = [=]() {
//     const ITraceActivity* linked =
//         linkedActivity(activity->correlationId, cpuCorrelationMap_);
//     const auto& gcu_sync_activity = traceBuffers_->addActivityWrapper(
//         GcuSyncActivity(activity, linked, src_stream, src_corrid));

//     if (outOfRange(gcu_sync_activity)) {
//       return;
//     }

//     if (int32_t(activity->streamId) != -1) {
//       recordStream(device_id, activity->streamId, "");
//     } else {
//       recordDevice(device_id);
//     }
//     VLOG(2) << "Logging sync event device = " << device_id
//             << " stream = " << activity->streamId
//             << " sync type = " << syncTypeString(activity->type);
//     gcu_sync_activity.log(*logger);
//     setGcuActivityPresent(true);
//   };

//   if (isWaitEventSync(activity->type)) {
//     // Defer logging wait event syncs till the end so we only
//     // log these events if a stream has some GCU kernels on it.
//     DeferredLogEntry entry{
//         .device = device_id,
//         .stream = activity->streamId,
//         .logMe = log_event,
//     };
//     logQueue_.push_back(entry);
//   } else {
//     log_event();
//   }
// }

void TopsptiActivityProfiler::logDeferredEvents() {
  // Stream Wait Events tend to be noisy, only pass these events if
  // there was some GCU kernel/memcopy/memset observed on it in the trace
  // window.
  for (const auto& entry : logQueue_) {
    if (seenDeviceStreams_.find({entry.device, entry.stream}) ==
        seenDeviceStreams_.end()) {
      VLOG(2) << "Skipping Event Sync as no kernels have run yet on stream = "
              << entry.stream;
    } else {
      entry.logMe();
    }
  }
}

inline void TopsptiActivityProfiler::updateGcuNetSpan(
    const ITraceActivity& gcuOp) {
  if (!gcuOp.linkedActivity()) {
    VLOG(0) << "Missing linked activity";
    return;
  }
  const auto& it =
      clientActivityTraceMap_.find(gcuOp.linkedActivity()->correlationId());
  if (it == clientActivityTraceMap_.end()) {
    // No correlation id mapping?
    return;
  }
  TraceSpan& gcu_span = it->second->second;
  if (gcuOp.timestamp() < gcu_span.startTime || gcu_span.startTime == 0) {
    gcu_span.startTime = gcuOp.timestamp();
  }
  if ((gcuOp.timestamp() + gcuOp.duration()) > gcu_span.endTime) {
    gcu_span.endTime = gcuOp.timestamp() + gcuOp.duration();
  }
}

// I've observed occasional broken timestamps attached to GCU events...
void TopsptiActivityProfiler::checkTimestampOrder(const ITraceActivity* act1) {
  // Correlated GCU runtime activity cannot
  // have timestamp greater than the GCU activity's
  const auto& it = correlatedGcuActivities_.find(act1->correlationId());
  if (it == correlatedGcuActivities_.end()) {
    correlatedGcuActivities_.insert({act1->correlationId(), act1});
    return;
  }

  // Activities may be appear in the buffers out of order.
  // If we have a runtime activity in the map, it should mean that we
  // have a GCU activity passed in, and vice versa.
  const ITraceActivity* act2 = it->second;
  if (act2->type() == ActivityType::GCU_RUNTIME) {
    // Buffer is out-of-order.
    // Swap so that runtime activity is first for the comparison below.
    std::swap(act1, act2);
  }
  if (act1->timestamp() > act2->timestamp()) {
    LOG_FIRST_N(WARNING, 10)
        << "GCU op timestamp (" << act2->timestamp()
        << ") < runtime timestamp (" << act1->timestamp() << ") by "
        << act1->timestamp() - act2->timestamp() << "us"
        << " Name: " << act2->name() << " Device: " << act2->deviceId()
        << " Stream: " << act2->resourceId();
    ecs_.gcu_and_cpu_op_out_of_order++;
  }
}

const ITraceActivity* TopsptiActivityProfiler::linkedActivity(
    int32_t correlationId,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlationId);
  if (it != correlationMap.end()) {
    const auto& it2 = activityMap_.find(it->second);
    if (it2 != activityMap_.end()) {
      return it2->second;
    }
  }
  return nullptr;
}

inline void TopsptiActivityProfiler::handleGcuActivity(
    const ITraceActivity& act, ActivityLogger* logger) {
  if (outOfRange(act)) {
    return;
  }
  checkTimestampOrder(&act);
  VLOG(2) << act.correlationId() << ": " << act.name();
  recordStream(act.deviceId(), act.resourceId(), "");
  seenDeviceStreams_.insert({act.deviceId(), act.resourceId()});

  act.log(*logger);
  setGcuActivityPresent(true);
  updateGcuNetSpan(act);
  if (derivedConfig_->profileActivityTypes().count(
          ActivityType::GCU_USER_ANNOTATION)) {
    const auto& it = userCorrelationMap_.find(act.correlationId());
    if (it != userCorrelationMap_.end()) {
      const auto& it2 = activityMap_.find(it->second);
      if (it2 != activityMap_.end()) {
        recordStream(act.deviceId(), act.resourceId(), "context");
        gcuUserEventMap_.insertOrExtendEvent(*it2->second, act);
      }
    }
  }
}

template <class T>
inline void TopsptiActivityProfiler::handleGcuActivity(const T* act,
                                                       ActivityLogger* logger) {
  const ITraceActivity* linked =
      linkedActivity(act->correlationId, cpuCorrelationMap_);
  const auto& gcu_activity =
      traceBuffers_->addActivityWrapper(GcuActivity<T>(act, linked));
  handleGcuActivity(gcu_activity, logger);
}

template <class T>
inline void updateCtxToDeviceId(const T* act) {
  if (ctxToDeviceId().count(act->contextId) == 0) {
    ctxToDeviceId()[act->contextId] = act->deviceId;
  }
}

uint32_t contextIdtoDeviceId(uint32_t contextId) {
  auto it = ctxToDeviceId().find(contextId);
  return it != ctxToDeviceId().end() ? it->second : 0;
}

void TopsptiActivityProfiler::handleTopsptiActivity(
    const Topspti_Activity* record, ActivityLogger* logger) {
  switch (record->kind) {
    // case TOPSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
    //   handleCorrelationActivity(
    //       reinterpret_cast<const
    //       Topspti_ActivityExternalCorrelation*>(record));
    // break;
    case TOPSPTI_ACTIVITY_KIND_RUNTIME:
      handleRuntimeActivity(
          reinterpret_cast<const Topspti_ActivityAPI*>(record), logger);
      break;
    case TOPSPTI_ACTIVITY_KIND_KERNEL:
      handleGcuActivity(reinterpret_cast<const Topspti_ActivityKernel*>(record),
                        logger);
      updateCtxToDeviceId(
          reinterpret_cast<const Topspti_ActivityKernel*>(record));
      break;
    // case TOPSPTI_ACTIVITY_KIND_SYNCHRONIZATION:
    //   handleGcuSyncActivity(
    //       reinterpret_cast<const Topspti_ActivitySynchronization*>(record),
    //       logger);
    //   break;
    // case TOPSPTI_ACTIVITY_KIND_GCU_EVENT:
    //   handleGcuEventActivity(
    //       reinterpret_cast<const Topspti_ActivityTopsEvent*>(record));
    //   break;
    case TOPSPTI_ACTIVITY_KIND_MEMCPY:
      handleGcuActivity(reinterpret_cast<const Topspti_ActivityMemcpy*>(record),
                        logger);
      break;
    // case TOPSPTI_ACTIVITY_KIND_MEMCPY2:
    //   handleGcuActivity(
    //       reinterpret_cast<const Topspti_ActivityMemcpy2*>(record), logger);
    //   break;
    case TOPSPTI_ACTIVITY_KIND_MEMSET:
      handleGcuActivity(reinterpret_cast<const Topspti_ActivityMemset*>(record),
                        logger);
      break;
    // case TOPSPTI_ACTIVITY_KIND_OVERHEAD:
    //   handleOverheadActivity(
    //       reinterpret_cast<const Topspti_ActivityOverhead*>(record), logger);
    //   break;
    case TOPSPTI_ACTIVITY_KIND_DRIVER:
      handleDriverActivity(reinterpret_cast<const Topspti_ActivityAPI*>(record),
                           logger);
      break;
    default:
      LOG(WARNING) << "Unexpected activity type: " << record->kind;
      ecs_.unexpected_gcu_events++;
      break;
  }
}

const ITraceActivity* TopsptiActivityProfiler::cpuActivity(
    int32_t correlationId) {
  const auto& it2 = activityMap_.find(correlationId);
  return (it2 != activityMap_.end()) ? it2->second : nullptr;
}

void TopsptiActivityProfiler::configureChildProfilers() {
  // If child profilers are enabled create profiler sessions
  int64_t start_time_ms =
      duration_cast<milliseconds>(
          derivedConfig_->profileStartTime().time_since_epoch())
          .count();
  for (auto& profiler : profilers_) {
    LOG(INFO) << "[Profiler = " << profiler->name() << "] "
              << "Evaluating whether to run child profiler.";
    auto session = profiler->configure(
        start_time_ms, derivedConfig_->profileDuration().count(),
        derivedConfig_->profileActivityTypes(), *config_);
    if (session) {
      LOG(INFO) << "[Profiler = " << profiler->name() << "] "
                << "Running child profiler " << profiler->name() << " for "
                << derivedConfig_->profileDuration().count() << " ms";
      sessions_.push_back(std::move(session));
    } else {
      LOG(INFO) << "[Profiler = " << profiler->name() << "] "
                << "Not running child profiler.";
    }
  }
}

void TopsptiActivityProfiler::configure(const Config& config,
                                        const time_point<system_clock>& now) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (isActive()) {
    LOG(WARNING) << "TopsptiActivityProfiler already busy, terminating";
    return;
  }

  config_ = config.clone();

  // Ensure we're starting in a clean state
  resetTraceData();

#if !USE_GOOGLE_LOG
  // Add a LoggerObserverCollector to collect all logs during the trace.
  loggerCollectorMetadata_ = std::make_unique<LoggerCollector>();
  Logger::addLoggerObserver(loggerCollectorMetadata_.get());
#endif  // !USE_GOOGLE_LOG

  derivedConfig_.reset();
  derivedConfig_ = std::make_unique<ConfigDerivedState>(*config_);

  // Check if now is a valid time to start.
  if (!derivedConfig_->canStart(now)) {
    return;
  }

  if (LOG_IS_ON(INFO)) {
    config_->printActivityProfilerConfig(LIBKINETO_DBG_STREAM);
  }
  if (!cpuOnly_ && !libkineto_gcu::api().client()) {
    if (derivedConfig_->isProfilingByIteration()) {
      LOG(INFO) << "GCU-only tracing for " << config_->activitiesRunIterations()
                << " iterations";
    } else {
      LOG(INFO) << "GCU-only tracing for "
                << config_->activitiesDuration().count() << "ms";
    }
  }

  // Set useful metadata into the logger.
  LOGGER_OBSERVER_SET_TRACE_DURATION_MS(config_->activitiesDuration().count());
  if (!config_->requestTraceID().empty()) {
    LOGGER_OBSERVER_SET_TRACE_ID(config_->requestTraceID());
  }
  if (!config_->requestGroupTraceID().empty()) {
    LOGGER_OBSERVER_SET_GROUP_TRACE_ID(config_->requestGroupTraceID());
  }

  if (!cpuOnly_) {
    // Enabling TOPSPTI activity tracing incurs a larger perf hit at first,
    // presumably because structures are allocated and initialized, callbacks
    // are activated etc. After a while the overhead decreases and stabilizes.
    // It's therefore useful to perform some warmup before starting recording.
    LOG(INFO) << "Enabling GCU tracing";
    topspti_.setMaxBufferSize(config_->activitiesMaxGcuBufferSize());
    time_point<system_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
    }

    // use_topspti_tsc() = config_->getTSCTimestampFlag();
    // if (use_topspti_tsc()) {
    //   TOPSPTI_CALL(topsptiActivityRegisterTimestampCallback(
    //       []() -> uint64_t { return getApproximateTime(); }));
    // }

    topspti_.enableTopsptiActivities(config_->selectedActivityTypes());
    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(setupOverhead_,
                        duration_cast<microseconds>(t2 - timestamp).count());
    }
  }

  if (profilers_.size() > 0) {
    configureChildProfilers();
  }

  if (libkineto_gcu::api().client()) {
    libkineto_gcu::api().client()->prepare(
        config_->isReportInputShapesEnabled(),
        config_->isProfileMemoryEnabled(), config_->isWithStackEnabled(),
        config_->isWithFlopsEnabled(), config_->isWithModulesEnabled());
  }

  if (derivedConfig_->isProfilingByIteration()) {
    LOG(INFO) << "Tracing starting on iteration = "
              << derivedConfig_->profileStartIteration();
    LOG(INFO) << "Tracing will end on iteration = "
              << derivedConfig_->profileEndIteration();
  } else {
    LOG(INFO) << "Tracing starting in "
              << duration_cast<seconds>(derivedConfig_->profileStartTime() -
                                        now)
                     .count()
              << "s";
    LOG(INFO) << "Tracing will end in "
              << duration_cast<seconds>(derivedConfig_->profileEndTime() - now)
                     .count()
              << "s";
  }

  traceBuffers_ = std::make_unique<ActivityBuffers>();
  captureWindowStartTime_ = captureWindowEndTime_ = 0;
  currentRunloopState_ = RunloopState::Warmup;
}

void TopsptiActivityProfiler::toggleCollectionDynamic(const bool enable) {
  if (enable) {
    topspti_.enableTopsptiActivities(derivedConfig_->profileActivityTypes());
  } else {
    topspti_.disableTopsptiActivities(derivedConfig_->profileActivityTypes());
  }
}

void TopsptiActivityProfiler::startTraceInternal(
    const time_point<system_clock>& now) {
  // TODO(torch_gcu): hack here
  captureWindowStartTime_ = libkineto_gcu::timeSinceEpoch(now);
  VLOG(0) << "Warmup -> CollectTrace";
  for (auto& session : sessions_) {
    LOG(INFO) << "Starting child profiler session";
    session->start();
  }
  currentRunloopState_ = RunloopState::CollectTrace;
}

void TopsptiActivityProfiler::stopTraceInternal(
    const time_point<system_clock>& now) {
  // TODO(torch_gcu): hack here
  captureWindowEndTime_ = libkineto_gcu::timeSinceEpoch(now);

  if (!cpuOnly_) {
    time_point<system_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
    }

    topspti_.disableTopsptiActivities(derivedConfig_->profileActivityTypes());

    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(setupOverhead_,
                        duration_cast<microseconds>(t2 - timestamp).count());
    }
  }

  if (currentRunloopState_ == RunloopState::CollectTrace) {
    VLOG(0) << "CollectTrace -> ProcessTrace";
  } else {
    LOG(WARNING) << "Called stopTrace with state == "
                 << static_cast<std::underlying_type<RunloopState>::type>(
                        currentRunloopState_.load());
  }
  for (auto& session : sessions_) {
    LOG(INFO) << "Stopping child profiler session";
    session->stop();
  }
  currentRunloopState_ = RunloopState::ProcessTrace;
}

void TopsptiActivityProfiler::resetInternal() {
  resetTraceData();
  currentRunloopState_ = RunloopState::WaitForRequest;
}

const time_point<system_clock> TopsptiActivityProfiler::performRunLoopStep(
    const time_point<system_clock>& now,
    const time_point<system_clock>& nextWakeupTime, int64_t currentIter) {
  auto new_wakeup_time = nextWakeupTime;
  bool warmup_done = false, collection_done = false;

  VLOG_IF(1, currentIter >= 0)
      << "Run loop on application step(), iteration = " << currentIter;

  switch (currentRunloopState_) {
    case RunloopState::WaitForRequest:
      VLOG(1) << "State: WaitForRequest";
      // Nothing to do
      break;

    case RunloopState::Warmup:
      VLOG(1) << "State: Warmup";
      warmup_done = derivedConfig_->isWarmupDone(now, currentIter);

      // Flushing can take a while so avoid doing it close to the start time
      if (!cpuOnly_ && currentIter < 0 &&
          (derivedConfig_->isProfilingByIteration() ||
           nextWakeupTime < derivedConfig_->profileStartTime())) {
        topspti_.clearActivities();
      }

      if (topspti_.stopCollection) {
        // Go to process trace to clear any outstanding buffers etc
        std::lock_guard<std::mutex> guard(mutex_);
        stopTraceInternal(now);
        resetInternal();
        VLOG(0) << "Warmup -> WaitForRequest";
        break;
      }

      if (warmup_done) {
        UST_LOGGER_MARK_COMPLETED(kWarmUpStage);
        if (!derivedConfig_->isProfilingByIteration() &&
            (now > derivedConfig_->profileStartTime() + milliseconds(10))) {
          LOG(INFO) << "Tracing started "
                    << duration_cast<milliseconds>(
                           now - derivedConfig_->profileStartTime())
                           .count()
                    << "ms late!";
        } else {
          LOG(INFO) << "Tracing started";
        }
        startTrace(now);
        if (libkineto_gcu::api().client()) {
          libkineto_gcu::api().client()->start();
        }
        if (nextWakeupTime > derivedConfig_->profileEndTime()) {
          new_wakeup_time = derivedConfig_->profileEndTime();
        }
      } else if (nextWakeupTime > derivedConfig_->profileStartTime()) {
        new_wakeup_time = derivedConfig_->profileStartTime();
      }

      break;

    case RunloopState::CollectTrace:
      VLOG(1) << "State: CollectTrace";
      collection_done = derivedConfig_->isCollectionDone(now, currentIter);

      if (collection_done

          || topspti_.stopCollection

      ) {
        // Update runloop state first to prevent further updates to shared state
        LOG(INFO) << "Tracing complete.";
        VLOG_IF(1, currentIter > 0)
            << "This state change was invoked by application's step() call";

        if (libkineto_gcu::api().client()) {
          libkineto_gcu::api().client()->stop();
        }

        ecs_.topspti_stopped_early = topspti_.stopCollection;

        std::lock_guard<std::mutex> guard(mutex_);
        stopTraceInternal(now);
        VLOG_IF(0, collection_done) << "Reached profile end time";
        UST_LOGGER_MARK_COMPLETED(kCollectionStage);
      } else if (derivedConfig_->isProfilingByIteration()) {
        // nothing to do here
      } else if (now < derivedConfig_->profileEndTime() &&
                 derivedConfig_->profileEndTime() < nextWakeupTime) {
        new_wakeup_time = derivedConfig_->profileEndTime();
      }

      break;

    case RunloopState::ProcessTrace:
      VLOG(1) << "State: ProcessTrace";
      // skip this state transition if it called from the step() api
      // of the profiler.
      // else it could lead to a race between the profiler thread and an
      // application thread calling step()
      if (currentIter >= 0) {
        return new_wakeup_time;
      }
      // FIXME: Probably want to allow interruption here
      // for quickly handling trace request via synchronous API
      std::lock_guard<std::mutex> guard(mutex_);
      processTraceInternal(*logger_);
      UST_LOGGER_MARK_COMPLETED(kPostProcessingStage);
      resetInternal();
      VLOG(0) << "ProcessTrace -> WaitForRequest";
      break;
  }

  return new_wakeup_time;
}

void TopsptiActivityProfiler::finalizeTrace(const Config& config,
                                            ActivityLogger& logger) {
  LOG(INFO) << "Traces Recorded:";
  {
    for (const auto& it : iterationCountMap_) {
      LOG(INFO) << it.first << ": " << it.second << " iterations";
    }
    iterationCountMap_.clear();
  }

  // Process names
  int32_t pid = torch_gcu::util::processId();
  string process_name = torch_gcu::util::processName(pid);
  if (!process_name.empty()) {
    logger.handleDeviceInfo({pid, pid, process_name, "CPU"},
                            captureWindowStartTime_);
    if (!cpuOnly_) {
      // Usually, GCU events use device id as pid (0-7).
      // In some cases, CPU sockets are numbered starting from 0.
      // In the worst case, 8 CPU sockets + 8 GCUs, so the max GCU ID is 15.
      constexpr int kMaxGcuID = 15;
      // sortIndex is gcu + kExceedMaxPid to put GCU tracks at the bottom
      // of the trace timelines.
      for (int gcu = 0; gcu <= kMaxGcuID; gcu++) {
        logger.handleDeviceInfo({gcu, gcu + kExceedMaxPid, process_name,
                                 fmt::format("GCU {}", gcu)},
                                captureWindowStartTime_);
      }
    }
  }

  // Thread & stream info
  for (auto pair : resourceInfo_) {
    const auto& resource = pair.second;
    logger.handleResourceInfo(resource, captureWindowStartTime_);
  }

  for (auto& session : sessions_) {
    auto device_info = session->getDeviceInfo();
    if (device_info != nullptr) {
      logger.handleDeviceInfo(*device_info, captureWindowStartTime_);
    }

    auto resource_infos = session->getResourceInfos();
    for (auto resource_info : resource_infos) {
      logger.handleResourceInfo(resource_info, captureWindowStartTime_);
    }
  }

  for (const auto& iterations : traceSpans_) {
    for (const auto& span_pair : iterations.second) {
      const TraceSpan& gcu_span = span_pair.second;
      if (gcu_span.opCount > 0) {
        logger.handleTraceSpan(gcu_span);
      }
    }
  }

  // Overhead info
  overheadInfo_.push_back(ActivityLogger::OverheadInfo("TOPSPTI Overhead"));
  for (const auto& info : overheadInfo_) {
    logger.handleOverheadInfo(info, captureWindowStartTime_);
  }

  gcuUserEventMap_.logEvents(&logger);

  for (auto& session : sessions_) {
    auto trace_buffer = session->getTraceBuffer();
    if (trace_buffer) {
      // Set child start time to profiling start time if not set
      if (trace_buffer->span.startTime == 0) {
        trace_buffer->span.startTime = captureWindowStartTime_;
      }
      traceBuffers_->cpu.push_back(std::move(trace_buffer));
    }
  }

  // Logger Metadata contains a map of LOGs collected in Kineto
  //   logger_level -> List of log lines
  // This will be added into the trace as metadata.
  std::unordered_map<std::string, std::vector<std::string>> loggerMD =
      getLoggerMetadata();
  logger.finalizeTrace(config, std::move(traceBuffers_), captureWindowEndTime_,
                       loggerMD);
}

std::unordered_map<std::string, std::vector<std::string>>
TopsptiActivityProfiler::getLoggerMetadata() {
  std::unordered_map<std::string, std::vector<std::string>> loggerMD;

#if !USE_GOOGLE_LOG
  // Save logs from LoggerCollector objects into Trace metadata.
  auto LoggerMDMap = loggerCollectorMetadata_->extractCollectorMetadata();
  for (auto& md : LoggerMDMap) {
    loggerMD[toString(md.first)] = md.second;
  }
#endif  // !USE_GOOGLE_LOG
  return loggerMD;
}

void TopsptiActivityProfiler::pushCorrelationId(uint64_t id) {
  TopsptiActivityApi::pushCorrelationID(
      id, TopsptiActivityApi::CorrelationFlowType::Default);

  for (auto& session : sessions_) {
    session->pushCorrelationId(id);
  }
}

void TopsptiActivityProfiler::popCorrelationId() {
  TopsptiActivityApi::popCorrelationID(
      TopsptiActivityApi::CorrelationFlowType::Default);

  for (auto& session : sessions_) {
    session->popCorrelationId();
  }
}

void TopsptiActivityProfiler::pushUserCorrelationId(uint64_t id) {
  TopsptiActivityApi::pushCorrelationID(
      id, TopsptiActivityApi::CorrelationFlowType::User);

  for (auto& session : sessions_) {
    session->pushUserCorrelationId(id);
  }
}

void TopsptiActivityProfiler::popUserCorrelationId() {
  TopsptiActivityApi::popCorrelationID(
      TopsptiActivityApi::CorrelationFlowType::User);

  for (auto& session : sessions_) {
    session->popUserCorrelationId();
  }
}

void TopsptiActivityProfiler::resetTraceData() {
  if (!cpuOnly_) {
    topspti_.clearActivities();
    topspti_.teardownContext();
  }

  activityMap_.clear();
  cpuCorrelationMap_.clear();
  correlatedGcuActivities_.clear();
  gcuUserEventMap_.clear();
  traceSpans_.clear();
  clientActivityTraceMap_.clear();
  seenDeviceStreams_.clear();
  logQueue_.clear();
  traceBuffers_ = nullptr;
  metadata_.clear();
  sessions_.clear();
  resourceOverheadCount_ = 0;
  ecs_ = ErrorCounts{};
#if !USE_GOOGLE_LOG
  Logger::removeLoggerObserver(loggerCollectorMetadata_.get());
#endif  // !USE_GOOGLE_LOG
}

}  // namespace libkineto_gcu
