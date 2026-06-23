/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "TopsptiActivity.h"

#include <fmt/format.h>

#include "Demangle.h"
#include "DeviceProperties.h"
#include "output_base.h"
#include "topspti.h"

namespace libkineto_gcu {

using namespace libkineto_gcu;

// forward declaration
uint32_t contextIdtoDeviceId(uint32_t contextId);

template <>
inline const std::string GcuActivity<Topspti_ActivityKernel>::name() const {
  return demangle(raw().name);
}

template <>
inline ActivityType GcuActivity<Topspti_ActivityKernel>::type() const {
  return ActivityType::CONCURRENT_KERNEL;
}

// inline bool isWaitEventSync(Topspti_ActivitySynchronizationType type) {
//   return (type == TOPSPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT);
// }

// inline bool isEventSync(Topspti_ActivitySynchronizationType type) {
//   return (type == TOPSPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE ||
//           type == TOPSPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT);
// }

// inline std::string eventSyncInfo(const Topspti_ActivitySynchronization& act,
//                                  int32_t srcStream, int32_t srcCorrId) {
//   return fmt::format(R"JSON(
//       "wait_on_stream": {},
//       "wait_on_gcu_event_record_corr_id": {},
//       "wait_on_gcu_event_id": {},)JSON",
//                      srcStream, srcCorrId, act.topsEventId);
// }

// inline const std::string GcuSyncActivity::name() const {
//   return syncTypeString(raw().type);
// }

// inline int64_t GcuSyncActivity::deviceId() const {
//   return contextIdtoDeviceId(raw().contextId);
// }

// inline int64_t GcuSyncActivity::resourceId() const {
//   // For Context and Device Sync events stream ID is invalid and
//   // set to TOPSPTI_SYNCHRONIZATION_INVALID_VALUE (-1)
//   // converting to an integer will automatically wrap the number to -1
//   // in the trace.
//   return static_cast<int32_t>(raw().streamId);
// }

// inline void GcuSyncActivity::log(ActivityLogger& logger) const {
//   logger.handleActivity(*this);
// }

// inline const std::string GcuSyncActivity::metadataJson() const {
//   const Topspti_ActivitySynchronization& sync = raw();
//   // clang-format off
//   return fmt::format(R"JSON(
//       "gcu_sync_kind": "{}",{}
//       "stream": {}, "correlation": {},
//       "device": {}, "context": {})JSON",
//       syncTypeString(sync.type),
//       isEventSync(raw().type) ? eventSyncInfo(raw(), srcStream_, srcCorrId_)
//       : "", static_cast<int32_t>(sync.streamId), sync.correlationId,
//       deviceId(),
//       sync.contextId);
//   // clang-format on
//   return "";
// }

template <class T>
inline void GcuActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

constexpr int64_t us(int64_t timestamp) {
  // It's important that this conversion is the same here and in the CPU trace.
  // No rounding!
  return timestamp / 1000;
}

template <>
inline const std::string GcuActivity<Topspti_ActivityKernel>::metadataJson()
    const {
  const Topspti_ActivityKernel& kernel = raw();
  float blocksPerSmVal = blocksPerSm(kernel);
  float warpsPerSmVal = warpsPerSm(kernel);

  // clang-format off
//   return fmt::format(R"JSON(
//       "queued": {}, "device": {}, "context": {},
//       "stream": {}, "correlation": {},
//       "registers per thread": {},
//       "shared memory": {},
//       "blocks per SM": {},
//       "warps per SM": {},
//       "grid": [{}, {}, {}],
//       "block": [{}, {}, {}],
//       "est. achieved occupancy %": {})JSON",
//       kernel.queued, kernel.deviceId, kernel.contextId,
//       kernel.streamId, kernel.correlationId,
//       kernel.registersPerThread,
//       kernel.staticSharedMemory + kernel.dynamicSharedMemory,
//       std::isinf(blocksPerSmVal) ? "\"inf\"" : std::to_string(blocksPerSmVal),
//       std::isinf(warpsPerSmVal) ? "\"inf\"" : std::to_string(warpsPerSmVal),
//       kernel.gridX, kernel.gridY, kernel.gridZ,
//       kernel.blockX, kernel.blockY, kernel.blockZ,
//       (int) (0.5 + kernelOccupancy(kernel) * 100.0));
  return fmt::format(R"JSON(
       "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "blocks per SM": {},
      "warps per SM": {},
      "grid": [{}, {}, {}],
      "block": [{}, {}, {}],
      "est. achieved occupancy %": {})JSON",
      kernel.deviceId, kernel.contextId,
      kernel.streamId, kernel.correlationId,
      std::isinf(blocksPerSmVal) ? "\"inf\"" : std::to_string(blocksPerSmVal),
      std::isinf(warpsPerSmVal) ? "\"inf\"" : std::to_string(warpsPerSmVal),
      kernel.gridX, kernel.gridY, kernel.gridZ,
      kernel.blockX, kernel.blockY, kernel.blockZ,
      (int) (0.5 + kernelOccupancy(kernel) * 100.0));
  // clang-format on
}

inline std::string memcpyName(uint8_t kind, uint8_t src, uint8_t dst) {
  return fmt::format("Memcpy {} ({} -> {})",
                     memcpyKindString((Topspti_ActivityMemcpyKind)kind),
                     memoryKindString((Topspti_ActivityMemoryKind)src),
                     memoryKindString((Topspti_ActivityMemoryKind)dst));
}

template <>
inline ActivityType GcuActivity<Topspti_ActivityMemcpy>::type() const {
  return ActivityType::GCU_MEMCPY;
}

template <>
inline const std::string GcuActivity<Topspti_ActivityMemcpy>::name() const {
  return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
}

inline std::string bandwidth(uint64_t bytes, uint64_t duration) {
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

template <>
inline const std::string GcuActivity<Topspti_ActivityMemcpy>::metadataJson()
    const {
  const Topspti_ActivityMemcpy& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memcpy.deviceId, memcpy.contextId,
      memcpy.streamId, memcpy.correlationId,
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
  // clang-format on
}

template <>
inline const std::string GcuActivity<Topspti_ActivityMemset>::name() const {
  const char* memory_kind =
      memoryKindString((Topspti_ActivityMemoryKind)raw().memoryKind);
  return fmt::format("Memset ({})", memory_kind);
}

template <>
inline ActivityType GcuActivity<Topspti_ActivityMemset>::type() const {
  return ActivityType::GCU_MEMSET;
}

template <>
inline const std::string GcuActivity<Topspti_ActivityMemset>::metadataJson()
    const {
  const Topspti_ActivityMemset& memset = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memset.deviceId, memset.contextId,
      memset.streamId, memset.correlationId,
      memset.bytes, bandwidth(memset.bytes, memset.end - memset.start));
  // clang-format on
}

inline void RuntimeActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline void DriverActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

// inline void OverheadActivity::log(ActivityLogger& logger) const {
//   logger.handleActivity(*this);
// }

// inline bool OverheadActivity::flowStart() const { return false; }

// inline const std::string OverheadActivity::metadataJson() const { return "";
// }

inline bool RuntimeActivity::flowStart() const {
  bool should_correlate =
      activity_.cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsLaunchKernel ||
      (activity_.cbid >= TOPSPTI_RUNTIME_TRACE_CBID_topsMemset &&
       activity_.cbid <=
           TOPSPTI_RUNTIME_TRACE_CBID_topsMemcpyFromSymbolAsync) ||
      activity_.cbid ==
          TOPSPTI_RUNTIME_TRACE_CBID_topsLaunchCooperativeKernel ||
      activity_.cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsGraphLaunch ||
      activity_.cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsStreamSynchronize ||
      activity_.cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsDeviceSynchronize ||
      activity_.cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsStreamWaitEvent;

  should_correlate |=
      (activity_.cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsLaunchKernelExC);

  return should_correlate;
}

inline const std::string RuntimeActivity::metadataJson() const {
  return fmt::format(R"JSON(
      "cbid": {}, "correlation": {})JSON",
                     activity_.cbid, activity_.correlationId);
}

inline bool isKernelLaunchApi(const Topspti_ActivityAPI& activity_) {
  //   return activity_.cbid == TOPSPTI_DRIVER_TRACE_CBID_topsLaunchKernel
  //          || activity_.cbid == TOPSPTI_DRIVER_TRACE_CBID_topsLaunchKernelEx;
  return activity_.cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsLaunchKernel ||
         activity_.cbid == TOPSPTI_RUNTIME_TRACE_CBID_topsLaunchKernelExC;
}

inline bool DriverActivity::flowStart() const {
  return isKernelLaunchApi(activity_);
}

inline const std::string DriverActivity::metadataJson() const {
  return fmt::format(R"JSON(
      "cbid": {}, "correlation": {})JSON",
                     activity_.cbid, activity_.correlationId);
}

inline const std::string DriverActivity::name() const {
  // currently only cuLaunchKernel/cuLaunchKernelEx is expected
  assert(isKernelLaunchApi(activity_));
  // not yet implementing full name matching
  //   if (activity_.cbid == TOPSPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
  //     return "cuLaunchKernel";
  // #if defined(GCU_VERSION) && GCU_VERSION >= 11060
  //   } else if (activity_.cbid == TOPSPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx)
  //   {
  //     return "cuLaunchKernelEx";
  // #endif
  //   } else {
  //     return "Unknown";  // should not reach here
  //   }
  return "Unknown";
}

template <class T>
inline const std::string GcuActivity<T>::metadataJson() const {
  return "";
}

}  // namespace libkineto_gcu
