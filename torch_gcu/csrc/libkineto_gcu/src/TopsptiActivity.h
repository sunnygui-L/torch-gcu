/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include "ApproximateClock.h"
#include "GenericTraceActivity.h"
#include "ITraceActivity.h"
#include "gcu/thread_util.h"
#include "topspti.h"
#include "topspti_strings.h"

namespace libkineto_gcu {
class ActivityLogger;
}

namespace libkineto_gcu {

using namespace libkineto_gcu;
struct TraceSpan;

// This function allows us to activate/deactivate TSC TOPSPTI callbacks
// via a killswitch
bool& use_topspti_tsc();

// These classes wrap the various TOPSPTI activity types
// into subclasses of ITraceActivity so that they can all be accessed
// using the ITraceActivity interface and logged via ActivityLogger.

// Abstract base class, templated on Topspti activity type
template <class T>
struct TopsptiActivity : public ITraceActivity {
  explicit TopsptiActivity(const T* activity, const ITraceActivity* linked)
      : activity_(*activity), linked_(linked) {}
  // If we are running on Windows or are on a GCU version < 11.6,
  // we use the default system clock so no conversion needed same for all
  // ifdefs below
  int64_t timestamp() const override {
    if (use_topspti_tsc()) {
      return get_time_converter()(activity_.start);
    } else {
      return activity_.start;
    }
  }

  int64_t duration() const override {
    if (use_topspti_tsc()) {
      return get_time_converter()(activity_.end) -
             get_time_converter()(activity_.start);
    } else {
      return activity_.end - activity_.start;
    }
  }
  // TODO(T107507796): Deprecate ITraceActivity
  int64_t correlationId() const override { return 0; }
  int32_t getThreadId() const override { return 0; }
  const ITraceActivity* linkedActivity() const override { return linked_; }
  int flowType() const override { return kLinkAsyncCpuGcu; }
  int flowId() const override { return correlationId(); }
  const T& raw() const { return activity_; }
  const TraceSpan* traceSpan() const override { return nullptr; }

 protected:
  const T& activity_;
  const ITraceActivity* linked_{nullptr};
};

// Topspti_ActivityAPI - GCU runtime activities
struct RuntimeActivity : public TopsptiActivity<Topspti_ActivityAPI> {
  explicit RuntimeActivity(const Topspti_ActivityAPI* activity,
                           const ITraceActivity* linked, int32_t threadId)
      : TopsptiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override { return activity_.correlationId; }
  int64_t deviceId() const override { return torch_gcu::util::processId(); }
  int64_t resourceId() const override { return threadId_; }
  ActivityType type() const override { return ActivityType::GCU_RUNTIME; }
  bool flowStart() const override;
  const std::string name() const override {
    return runtimeCbidName(activity_.cbid);
  }
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// Topspti_ActivityAPI - GCU driver activities
struct DriverActivity : public TopsptiActivity<Topspti_ActivityAPI> {
  explicit DriverActivity(const Topspti_ActivityAPI* activity,
                          const ITraceActivity* linked, int32_t threadId)
      : TopsptiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override { return activity_.correlationId; }
  int64_t deviceId() const override { return torch_gcu::util::processId(); }
  int64_t resourceId() const override { return threadId_; }
  ActivityType type() const override { return ActivityType::GCU_DRIVER; }
  bool flowStart() const override;
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// Topspti_ActivityAPI - GCU runtime activities
// struct OverheadActivity : public TopsptiActivity<Topspti_ActivityOverhead> {
//   explicit OverheadActivity(const Topspti_ActivityOverhead* activity,
//                             const ITraceActivity* linked, int32_t threadId =
//                             0)
//       : TopsptiActivity(activity, linked), threadId_(threadId) {}

//   int64_t timestamp() const override {
// #if defined(_WIN32) || GCU_VERSION < 11060
//     return activity_.start;
// #else
//     if (use_topspti_tsc()) {
//       return get_time_converter()(activity_.start);
//     } else {
//       return activity_.start;
//     }
// #endif
//   }

//   int64_t duration() const override {
// #if defined(_WIN32) || GCU_VERSION < 11060
//     return activity_.end - activity_.start;
// #else
//     if (use_topspti_tsc()) {
//       return get_time_converter()(activity_.end) -
//              get_time_converter()(activity_.start);
//     } else {
//       return activity_.end - activity_.start;
//     }
// #endif
//   }

//   // TODO: Update this with PID ordering
//   int64_t deviceId() const override { return -1; }
//   int64_t resourceId() const override { return threadId_; }
//   ActivityType type() const override { return ActivityType::OVERHEAD; }
//   bool flowStart() const override;
//   const std::string name() const override {
//     return overheadKindString(activity_.overheadKind);
//   }
//   void log(ActivityLogger& logger) const override;
//   const std::string metadataJson() const override;

//  private:
//   const int32_t threadId_;
// };

// Topspti_ActivitySynchronization - GCU synchronization events
// struct GcuSyncActivity : public
// TopsptiActivity<Topspti_ActivitySynchronization> {
//   explicit GcuSyncActivity(const Topspti_ActivitySynchronization* activity,
//                             const ITraceActivity* linked, int32_t srcStream,
//                             int32_t srcCorrId)
//       : TopsptiActivity(activity, linked),
//         srcStream_(srcStream),
//         srcCorrId_(srcCorrId) {}
//   int64_t correlationId() const override { return raw().correlationId; }
//   int64_t deviceId() const override;
//   int64_t resourceId() const override;
//   ActivityType type() const override { return ActivityType::GCU_SYNC; }
//   bool flowStart() const override { return false; }
//   const std::string name() const override;
//   void log(ActivityLogger& logger) const override;
//   const std::string metadataJson() const override;
//   const Topspti_ActivitySynchronization& raw() const {
//     return TopsptiActivity<Topspti_ActivitySynchronization>::raw();
//   }

//  private:
//   const int32_t srcStream_;
//   const int32_t srcCorrId_;
// };

// Base class for GCU activities.
// Can also be instantiated directly.
template <class T>
struct GcuActivity : public TopsptiActivity<T> {
  explicit GcuActivity(const T* activity, const ITraceActivity* linked)
      : TopsptiActivity<T>(activity, linked) {}
  int64_t correlationId() const override { return raw().correlationId; }
  int64_t deviceId() const override { return raw().deviceId; }
  int64_t resourceId() const override { return raw().streamId; }
  ActivityType type() const override;
  bool flowStart() const override { return false; }
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const T& raw() const { return TopsptiActivity<T>::raw(); }
};

}  // namespace libkineto_gcu
