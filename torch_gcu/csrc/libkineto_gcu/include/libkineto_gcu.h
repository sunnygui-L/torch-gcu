/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

// Mediator for initialization and profiler control

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ActivityProfilerInterface.h"
#include "ActivityTraceInterface.h"
#include "ActivityType.h"
#include "ClientInterface.h"
#include "GenericTraceActivity.h"
#include "IActivityProfiler.h"
#include "ILoggerObserver.h"
#include "LoggingAPI.h"
#include "TraceSpan.h"
#include "gcu/gcu_macros.h"
#include "gcu/thread_util.h"

extern "C" {
void TORCH_GCU_API suppressLibkinetoLogMessages();
int TORCH_GCU_API InitializeInjection(void);
void TORCH_GCU_API libkineto_gcu_init(bool cpuOnly, bool logOnError);
bool TORCH_GCU_API hasTestEnvVar();
}

namespace libkineto_gcu {

class Config;
class ConfigLoader;

struct TORCH_GCU_API CpuTraceBuffer {
  template <class... Args>
  void emplace_activity(Args&&... args) {
    activities.emplace_back(
        std::make_unique<GenericTraceActivity>(std::forward<Args>(args)...));
  }

  static GenericTraceActivity& toRef(
      std::unique_ptr<GenericTraceActivity>& ref) {
    return *ref;
  }

  static const GenericTraceActivity& toRef(
      const std::unique_ptr<GenericTraceActivity>& ref) {
    return *ref;
  }

  TraceSpan span{0, 0, "none"};
  int gcuOpCount;
  std::deque<std::unique_ptr<GenericTraceActivity>> activities;
};

using ChildActivityProfilerFactory =
    std::function<std::unique_ptr<IActivityProfiler>()>;

class TORCH_GCU_API LibkinetoApi {
 public:
  explicit LibkinetoApi(ConfigLoader& configLoader)
      : configLoader_(configLoader) {}

  // Called by client that supports tracing API.
  // libkineto_gcu can still function without this.
  void registerClient(ClientInterface* client);

  // Called by libkineto_gcu on init
  void registerProfiler(std::unique_ptr<ActivityProfilerInterface> profiler) {
    activityProfiler_ = std::move(profiler);
    initClientIfRegistered();
  }

  ActivityProfilerInterface& activityProfiler() { return *activityProfiler_; }

  ClientInterface* client() { return client_; }

  void initProfilerIfRegistered() {
    static std::once_flag once;
    if (activityProfiler_) {
      std::call_once(once, [this] {
        if (!activityProfiler_->isInitialized()) {
          activityProfiler_->init();
          initChildActivityProfilers();
        }
      });
    }
  }

  bool isProfilerInitialized() const {
    return activityProfiler_ && activityProfiler_->isInitialized();
  }

  bool isProfilerRegistered() const { return activityProfiler_ != nullptr; }

  void suppressLogMessages() { suppressLibkinetoLogMessages(); }

  // Provides access to profiler configuration management
  ConfigLoader& configLoader() { return configLoader_; }

  void registerProfilerFactory(ChildActivityProfilerFactory factory) {
    if (isProfilerInitialized()) {
      activityProfiler_->addChildActivityProfiler(factory());
    } else {
      childProfilerFactories_.push_back(factory);
    }
  }

 private:
  void initChildActivityProfilers() {
    if (!isProfilerInitialized()) {
      return;
    }
    for (const auto& factory : childProfilerFactories_) {
      activityProfiler_->addChildActivityProfiler(factory());
    }
    childProfilerFactories_.clear();
  }

  // Client is initialized once both it and libkineto_gcu has registered
  void initClientIfRegistered();

  ConfigLoader& configLoader_;
  std::unique_ptr<ActivityProfilerInterface> activityProfiler_{};
  ClientInterface* client_{};
  int32_t clientRegisterThread_{0};

  std::vector<ChildActivityProfilerFactory> childProfilerFactories_;
};

// Singleton
TORCH_GCU_API LibkinetoApi& api();

}  // namespace libkineto_gcu
