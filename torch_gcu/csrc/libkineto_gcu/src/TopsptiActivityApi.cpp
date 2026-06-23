/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "TopsptiActivityApi.h"

#include <assert.h>
#include <unistd.h>

#include <chrono>
#include <mutex>
#include <thread>

#include "Config.h"
#include "DeviceUtil.h"
#include "Logger.h"
#include "gcu/thread_util.h"

using namespace std::chrono;

namespace libkineto_gcu {

// Set to 4MB to avoid constantly creating buffers (especially for networks
// that have many small memcpy such as sparseNN). TOPSPTI recommends between
// 1MB to 10MB.
// Given the kDefaultActivitiesMaxGcuBufferSize is around 128MB, in the worst
// case, there will be 32 buffers contending for the mutex.
constexpr size_t kBufSize(4 * 1024 * 1024);

inline bool topsptiTearDown_() {
  auto teardown_env = getenv("TEARDOWN_TOPSPTI");
  return teardown_env != nullptr && strcmp(teardown_env, "1") == 0;
}

inline bool topsptiLazyInit_() {
  return topsptiTearDown_() && getenv("DISABLE_TOPSPTI_LAZY_REINIT") == nullptr;
}

inline void reenableTopsptiCallbacks_(
    std::shared_ptr<TopsptiCallbackApi>& cbapi_) {
  // Re-enable callbacks from the past if they exist.
  LOG(INFO) << "Re-enable previous TOPSPTI callbacks - Starting";
  VLOG(1) << "  TOPSPTI subscriber before reinit:"
          << cbapi_->getTopsptiSubscriber();
  cbapi_->initCallbackApi();
  if (cbapi_->initSuccess()) {
    VLOG(1) << "  TOPSPTI subscriber after reinit:"
            << cbapi_->getTopsptiSubscriber();
    bool status = cbapi_->reenableCallbacks();
    if (!status) {
      LOG(WARNING) << "Re-enable previous TOPSPTI callbacks - Failed to "
                      "reenableCallbacks";
    } else {
      LOG(INFO) << "Re-enable previous TOPSPTI callbacks - Successful";
    }
  } else {
    LOG(WARNING)
        << "Re-enable previous TOPSPTI callbacks - Failed to initCallbackApi";
  }
}

TopsptiActivityApi& TopsptiActivityApi::singleton() {
  static TopsptiActivityApi instance;
  return instance;
}

TopsptiActivityApi::~TopsptiActivityApi() {
  // Force flush all activity buffers so that topspti's worker thread completes
  // processing and returns ownership via bufferCompleted callback before
  // allocatedGcuTraceBuffers_ is destroyed. Without this, the worker thread
  // may still be reading buffer contents (e.g. in convertToCpuTimestamp)
  // when the unique_ptrs in the map are deleted, causing heap-use-after-free.
  if (tracingEnabled_) {
    TOPSPTI_CALL(topsptiActivityFlushAll(TOPSPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  }
}

void TopsptiActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  //   VLOG(2) << "pushCorrelationID(" << id << ")";
  //   switch (type) {
  //     case Default:
  //       TOPSPTI_CALL(topsptiActivityPushExternalCorrelationId(
  //           TOPSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id));
  //       break;
  //     case User:
  //       TOPSPTI_CALL(topsptiActivityPushExternalCorrelationId(
  //           TOPSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, id));
  //   }
}

void TopsptiActivityApi::popCorrelationID(CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  //   switch (type) {
  //     case Default:
  //       TOPSPTI_CALL(topsptiActivityPopExternalCorrelationId(
  //           TOPSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, nullptr));
  //       break;
  //     case User:
  //       TOPSPTI_CALL(topsptiActivityPopExternalCorrelationId(
  //           TOPSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, nullptr));
  //   }
}

static bool nextActivityRecord(uint8_t* buffer, size_t valid_size,
                               Topspti_Activity*& record) {
  // TODO: why tops differ from cuda
  TopsptiResult status = TOPSPTI_CALL_NOWARN(
      topsptiActivityGetNextRecord(buffer, valid_size, &record));
  if (status != TOPSPTI_SUCCESS) {
    if (status != TOPSPTI_ERROR_MAX_LIMIT_REACHED) {
      TOPSPTI_CALL(status);
    }
    record = nullptr;
  }
  return record != nullptr;
}

void TopsptiActivityApi::setMaxBufferSize(int size) {
  maxGcuBufferCount_ = 1 + size / kBufSize;
}

void TopsptiActivityApi::setDeviceBufferSize(size_t size) {
  size_t valueSize = sizeof(size_t);
  //   TOPSPTI_CALL(topsptiActivitySetAttribute(
  //       TOPSPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &valueSize, &size));
}

void TopsptiActivityApi::setDeviceBufferPoolLimit(size_t limit) {
  size_t valueSize = sizeof(size_t);
  //   TOPSPTI_CALL(topsptiActivitySetAttribute(
  //       TOPSPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &valueSize, &limit));
}

void TopsptiActivityApi::forceLoadTopspti() {
  //   TOPSPTI_CALL(topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
}

void TopsptiActivityApi::preConfigureTOPSPTI() {
  if (!isGcuAvailable()) {
    return;
  }
}

void TopsptiActivityApi::bufferRequestedTrampoline(uint8_t** buffer,
                                                   size_t* size,
                                                   size_t* maxNumRecords) {
  singleton().bufferRequested(buffer, size, maxNumRecords);
}

void TopsptiActivityApi::bufferRequested(uint8_t** buffer, size_t* size,
                                         size_t* maxNumRecords) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (allocatedGcuTraceBuffers_.size() >= maxGcuBufferCount_) {
    stopCollection = true;
    LOG(WARNING) << "Exceeded max GCU buffer count ("
                 << allocatedGcuTraceBuffers_.size() << " > "
                 << maxGcuBufferCount_ << ") - terminating tracing";
  }

  auto buf = std::make_unique<TopsptiActivityBuffer>(kBufSize);
  *buffer = buf->data();
  *size = kBufSize;

  allocatedGcuTraceBuffers_[*buffer] = std::move(buf);

  *maxNumRecords = 0;
}

std::unique_ptr<TopsptiActivityBufferMap>
TopsptiActivityApi::activityBuffers() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGcuTraceBuffers_.empty()) {
      return nullptr;
    }
  }

  VLOG(1) << "Flushing GCU activity buffers";
  time_point<system_clock> t1;
  if (VLOG_IS_ON(1)) {
    t1 = system_clock::now();
  }
  // Can't hold mutex_ during this call, since bufferCompleted
  // will be called by libtopspti and mutex_ is acquired there.
  TOPSPTI_CALL(topsptiActivityFlushAll(TOPSPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  if (VLOG_IS_ON(1)) {
    flushOverhead =
        duration_cast<microseconds>(system_clock::now() - t1).count();
  }

  std::lock_guard<std::mutex> guard(mutex_);
  // Transfer ownership of buffers to caller. A new map is created on-demand.
  return std::move(readyGcuTraceBuffers_);
}

int TopsptiActivityApi::processActivitiesForBuffer(
    uint8_t* buf, size_t validSize,
    std::function<void(const Topspti_Activity*)> handler) {
  int count = 0;
  if (buf && validSize) {
    Topspti_Activity* record{nullptr};
    while ((nextActivityRecord(buf, validSize, record))) {
      handler(record);
      ++count;
    }
  }
  return count;
}

const std::pair<int, size_t> TopsptiActivityApi::processActivities(
    TopsptiActivityBufferMap& buffers,
    std::function<void(const Topspti_Activity*)> handler) {
  std::pair<int, size_t> res{0, 0};

  for (auto& pair : buffers) {
    // No lock needed - only accessed from this thread
    auto& buf = pair.second;
    res.first += processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.second += buf->size();
  }

  return res;
}

void TopsptiActivityApi::clearActivities() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGcuTraceBuffers_.empty()) {
      return;
    }
  }
  // Can't hold mutex_ during this call, since bufferCompleted
  // will be called by libtopspti and mutex_ is acquired there.

  TOPSPTI_CALL(topsptiActivityFlushAll(0));

  // FIXME: We might want to make sure we reuse
  // the same memory during warmup and tracing.
  // Also, try to use the amount of memory required
  // for active tracing during warmup.
  std::lock_guard<std::mutex> guard(mutex_);
  // Throw away ready buffers as a result of above flush
  readyGcuTraceBuffers_ = nullptr;
}

// NOTE(torch_gcu): differ from CUPTI api
// void TopsptiActivityApi::bufferCompletedTrampoline(
//     CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t /* unused */,
//     size_t validSize) {
//   singleton().bufferCompleted(ctx, streamId, buffer, 0, validSize);
// }

void TopsptiActivityApi::bufferCompletedTrampoline(uint8_t* buffer,
                                                   size_t /* unused */,
                                                   size_t validSize) {
  singleton().bufferCompleted(buffer, 0, validSize);
}

// NOTE(torch_gcu): differ from CUPTI api
// void TopsptiActivityApi::bufferCompleted(CUcontext ctx, uint32_t streamId,
//                                          uint8_t* buffer, size_t /* unused
//                                          */, size_t validSize)

void TopsptiActivityApi::bufferCompleted(uint8_t* buffer, size_t /* unused
                                                                  */
                                         ,
                                         size_t validSize) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = allocatedGcuTraceBuffers_.find(buffer);
  if (it == allocatedGcuTraceBuffers_.end()) {
    LOG(ERROR) << "bufferCompleted called with unknown buffer: "
               << (void*)buffer;
    return;
  }

  if (!readyGcuTraceBuffers_) {
    readyGcuTraceBuffers_ = std::make_unique<TopsptiActivityBufferMap>();
  }
  // Set valid size of buffer before moving to ready map
  it->second->setSize(validSize);
  (*readyGcuTraceBuffers_)[it->first] = std::move(it->second);
  allocatedGcuTraceBuffers_.erase(it);

  // TODO(torch_gcu): Handle dropped records
  // report any records dropped from the queue; to avoid unnecessary topspti
  // API calls, we make it report only in verbose mode (it doesn't happen
  // often in our testing anyways)
  // if (VLOG_IS_ON(1)) {
  //   size_t dropped = 0;
  //   TOPSPTI_CALL(topsptiActivityGetNumDroppedRecords(ctx, streamId,
  //   &dropped)); if (dropped != 0) {
  //     LOG(WARNING) << "Dropped " << dropped << " activity records";
  //   }
  // }
}

void TopsptiActivityApi::enableTopsptiActivities(
    const std::set<ActivityType>& selected_activities) {
  // Lazily support re-init of TOPSPTI Callbacks, if they were finalized before.
  auto cbapi_ = TopsptiCallbackApi::singleton();
  if (!tracingEnabled_ && !cbapi_->initSuccess() && topsptiLazyInit_()) {
    reenableTopsptiCallbacks_(cbapi_);
  }
  cbapi_.reset();

  TOPSPTI_CALL(topsptiActivityRegisterCallbacks(bufferRequestedTrampoline,
                                                bufferCompletedTrampoline));

  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GCU_MEMCPY) {
      TOPSPTI_CALL(topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_MEMCPY));
    }
    if (activity == ActivityType::GCU_MEMSET) {
      TOPSPTI_CALL(topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_MEMSET));
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      TOPSPTI_CALL(topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_KERNEL));
    }
    // if (activity == ActivityType::EXTERNAL_CORRELATION) {
    //   TOPSPTI_CALL(
    //       topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    //   externalCorrelationEnabled_ = true;
    // }
    // if (activity == ActivityType::GCU_SYNC) {
    //   TOPSPTI_CALL(
    //       topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_SYNCHRONIZATION));
    // }
    if (activity == ActivityType::GCU_RUNTIME) {
      TOPSPTI_CALL(topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_RUNTIME));
    }
    if (activity == ActivityType::GCU_DRIVER) {
      TOPSPTI_CALL(topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_DRIVER));
    }
    // if (activity == ActivityType::OVERHEAD) {
    //   TOPSPTI_CALL(topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_OVERHEAD));
    // }
  }

  tracingEnabled_ = 1;

  // Explicitly enabled, so reset this flag if set
  stopCollection = false;
}

void TopsptiActivityApi::disableTopsptiActivities(
    const std::set<ActivityType>& selected_activities) {
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GCU_MEMCPY) {
      TOPSPTI_CALL(topsptiActivityDisable(TOPSPTI_ACTIVITY_KIND_MEMCPY));
    }
    if (activity == ActivityType::GCU_MEMSET) {
      TOPSPTI_CALL(topsptiActivityDisable(TOPSPTI_ACTIVITY_KIND_MEMSET));
    }
    // if (activity == ActivityType::CONCURRENT_KERNEL) {
    //   TOPSPTI_CALL(
    //       topsptiActivityDisable(TOPSPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    // }
    // if (activity == ActivityType::EXTERNAL_CORRELATION) {
    //   TOPSPTI_CALL(
    //       topsptiActivityDisable(TOPSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    // }
    // if (activity == ActivityType::GCU_SYNC) {
    //   TOPSPTI_CALL(
    //       topsptiActivityDisable(TOPSPTI_ACTIVITY_KIND_SYNCHRONIZATION));
    // }
    if (activity == ActivityType::GCU_RUNTIME) {
      TOPSPTI_CALL(topsptiActivityDisable(TOPSPTI_ACTIVITY_KIND_RUNTIME));
    }
    if (activity == ActivityType::GCU_DRIVER) {
      TOPSPTI_CALL(topsptiActivityDisable(TOPSPTI_ACTIVITY_KIND_DRIVER));
    }
    // if (activity == ActivityType::OVERHEAD) {
    //   TOPSPTI_CALL(topsptiActivityDisable(TOPSPTI_ACTIVITY_KIND_OVERHEAD));
    // }
  }
  externalCorrelationEnabled_ = false;
}

void TopsptiActivityApi::teardownContext() {
  if (!tracingEnabled_) {
    return;
  }
  if (topsptiTearDown_()) {
    LOG(INFO) << "teardownTopspti starting";

    // PyTorch Profiler is synchronous, so teardown needs to be run async in
    // this thread.
    std::thread teardownThread([&] {
      torch_gcu::util::setThreadName("TorchTopsptiAct");
      auto cbapi_ = TopsptiCallbackApi::singleton();
      if (!cbapi_->initSuccess()) {
        cbapi_->initCallbackApi();
        if (!cbapi_->initSuccess()) {
          LOG(WARNING) << "TOPSPTI Callback failed to init, skipping teardown";
          return;
        }
      }
      // Subscribe callbacks to call topsptiFinalize in the exit callback of
      // these APIs
      bool status = cbapi_->enableCallbackDomain(TOPSPTI_CB_DOMAIN_RUNTIME_API);
      status =
          status && cbapi_->enableCallbackDomain(TOPSPTI_CB_DOMAIN_DRIVER_API);
      if (!status) {
        LOG(WARNING) << "TOPSPTI Callback failed to enable for domain, "
                        "skipping teardown";
        return;
      }

      // Force Flush before finalize
      TOPSPTI_CALL(topsptiActivityFlushAll(TOPSPTI_ACTIVITY_FLAG_FLUSH_FORCED));

      LOG(INFO) << "  TOPSPTI subscriber before finalize:"
                << cbapi_->getTopsptiSubscriber();
      teardownTopspti_ = 1;
      std::unique_lock<std::mutex> lck(finalizeMutex_);
      finalizeCond_.wait(lck, [&] { return teardownTopspti_ == 0; });
      lck.unlock();
      LOG(INFO) << "teardownTopspti complete";

      teardownTopspti_ = 0;
      tracingEnabled_ = 0;

      // Remove the callbacks used specifically for topsptiFinalize
      cbapi_->disableCallbackDomain(TOPSPTI_CB_DOMAIN_RUNTIME_API);
      cbapi_->disableCallbackDomain(TOPSPTI_CB_DOMAIN_DRIVER_API);

      // Re-init TOPSPTI Callbacks if Lazy Re-init is not enabled.
      if (!topsptiLazyInit_()) {
        reenableTopsptiCallbacks_(cbapi_);
      }
      cbapi_.reset();
    });
    teardownThread.detach();
  }
}

}  // namespace libkineto_gcu
