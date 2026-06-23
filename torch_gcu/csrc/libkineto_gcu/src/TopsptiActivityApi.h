/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>

#include "ActivityType.h"
#include "TopsptiActivityBuffer.h"
#include "TopsptiCallbackApi.h"
#include "topspti.h"

namespace libkineto_gcu {

using namespace libkineto_gcu;

class TopsptiActivityApi {
 public:
  enum CorrelationFlowType { Default, User };
  // Control Variables shared with TopsptiCallbackApi for teardown
  std::atomic<uint32_t> teardownTopspti_{0};
  std::mutex finalizeMutex_;
  std::condition_variable finalizeCond_;

  TopsptiActivityApi() = default;
  TopsptiActivityApi(const TopsptiActivityApi&) = delete;
  TopsptiActivityApi& operator=(const TopsptiActivityApi&) = delete;

  virtual ~TopsptiActivityApi();

  static TopsptiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableTopsptiActivities(
      const std::set<ActivityType>& selected_activities);
  void disableTopsptiActivities(
      const std::set<ActivityType>& selected_activities);
  void clearActivities();
  void teardownContext();

  virtual std::unique_ptr<TopsptiActivityBufferMap> activityBuffers();

  virtual const std::pair<int, size_t> processActivities(
      TopsptiActivityBufferMap&,
      std::function<void(const Topspti_Activity*)> handler);

  void setMaxBufferSize(int size);
  void setDeviceBufferSize(size_t size);
  void setDeviceBufferPoolLimit(size_t limit);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

  static void forceLoadTopspti();

  // TOPSPTI configuration that needs to be set before GCU context creation
  static void preConfigureTOPSPTI();

 private:
  int maxGcuBufferCount_{0};
  TopsptiActivityBufferMap allocatedGcuTraceBuffers_;
  std::unique_ptr<TopsptiActivityBufferMap> readyGcuTraceBuffers_;
  std::mutex mutex_;
  std::atomic<uint32_t> tracingEnabled_{0};
  bool externalCorrelationEnabled_{false};

  int processActivitiesForBuffer(
      uint8_t* buf, size_t validSize,
      std::function<void(const Topspti_Activity*)> handler);
  static void bufferRequestedTrampoline(uint8_t** buffer, size_t* size,
                                        size_t* maxNumRecords);

  // NOTE(torch_gcu): differ from cupti api
  //   static void bufferCompletedTrampoline(CUcontext ctx, uint32_t streamId,
  //                                         uint8_t* buffer, size_t /* unused
  //                                         */, size_t validSize);
  static void bufferCompletedTrampoline(uint8_t* buffer, size_t /* unused */,
                                        size_t validSize);

 protected:
  void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);

  // NOTE(torch_gcu): differ from cupti api
  //   void bufferCompleted(
  //         CUcontext ctx,
  //         uint32_t streamId,
  //         uint8_t* buffer,
  //         size_t /* unused */,
  //         size_t validSize);

  void bufferCompleted(uint8_t* buffer, size_t /* unused */, size_t validSize);
};

}  // namespace libkineto_gcu
