/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include <memory>
#include <mutex>

#include "ActivityProfilerProxy.h"
#include "Config.h"
#include "ConfigLoader.h"
// #include "DaemonConfigLoader.h"
#include "DeviceUtil.h"
// #include "EventProfilerController.h"
#include "Logger.h"
#include "TopsptiActivityApi.h"
#include "TopsptiCallbackApi.h"
// #include "TopsptiRangeProfiler.h"
#include "libkineto_gcu.h"

namespace libkineto_gcu {

static bool initialized = false;

static void initProfilersCPU() {
  if (!initialized) {
    libkineto_gcu::api().initProfilerIfRegistered();
    initialized = true;
    VLOG(0) << "libkineto_gcu profilers activated";
  }
}

static std::mutex& initEventMutex() {
  static std::mutex initMutex_;
  return initMutex_;
}

bool enableEventProfiler() {
  if (getenv("KINETO_ENABLE_EVENT_PROFILER") != nullptr) {
    return true;
  } else {
    return false;
  }
}

static void initProfilers(Topspti_CallbackDomain /*domain*/,
                          Topspti_CallbackId /*cbid*/,
                          const Topspti_CallbackData* cbInfo) {
  VLOG(0) << "GCU Context created";
  initProfilersCPU();

  //   if (!enableEventProfiler()) {
  //     VLOG(0) << "Kineto EventProfiler disabled, skipping start";
  //     return;
  //   } else {
  //     std::lock_guard<std::mutex> lock(initEventMutex());
  //     Topspti_ResourceData* d = (Topspti_ResourceData*)cbInfo;
  //     CUcontext ctx = d->context;
  //     ConfigLoader& config_loader = libkineto_gcu::api().configLoader();
  //     config_loader.initBaseConfig();
  //     auto config = config_loader.getConfigCopy();
  //     if (config->eventProfilerEnabled()) {
  //       // This function needs to be called under lock.
  //       EventProfilerController::start(ctx, config_loader);
  //       LOG(INFO) << "Kineto EventProfiler started";
  //     }
  //   }
}

// Some models suffer from excessive instrumentation code gen
// on dynamic attach which can hang for more than 5+ seconds.
// If the workload was meant to be traced, preload the TOPSPTI
// to take the performance hit early on.
// https://docs.nvidia.com/topspti/r_main.html#r_overhead
static bool shouldPreloadTopsptiInstrumentation() { return false; }

static void stopProfiler(Topspti_CallbackDomain /*domain*/,
                         Topspti_CallbackId /*cbid*/,
                         const Topspti_CallbackData* cbInfo) {
  VLOG(0) << "GCU Context destroyed";
  //   std::lock_guard<std::mutex> lock(initEventMutex());
  //   Topspti_ResourceData* d = (Topspti_ResourceData*)cbInfo;
  //   CUcontext ctx = d->context;
  //   // This function needs to be called under lock.
  //   EventProfilerController::stopIfEnabled(ctx);
  //   LOG(INFO) << "Kineto EventProfiler stopped";
}

// static std::unique_ptr<TopsptiRangeProfilerInit> rangeProfilerInit;

}  // namespace libkineto_gcu

// Callback interface with TOPSPTI and library constructors
using namespace libkineto_gcu;
extern "C" {

// Return true if no TOPSPTI errors occurred during init
void libkineto_gcu_init(bool cpuOnly, bool logOnError) {
  // Start with initializing the log level
  const char* logLevelEnv = getenv("KINETO_LOG_LEVEL");
  if (logLevelEnv) {
    // atoi returns 0 on error, so that's what we want - default to VERBOSE
    static_assert(static_cast<int>(VERBOSE) == 0, "");
    SET_LOG_SEVERITY_LEVEL(atoi(logLevelEnv));
  }

  // Factory to connect to open source daemon if present
#if __linux__
//   if (getenv(kUseDaemonEnvVar) != nullptr) {
//     LOG(INFO) << "Registering daemon config loader, cpuOnly =  " << cpuOnly;
//     DaemonConfigLoader::registerFactory();
//   }
#endif

  if (!cpuOnly) {
    // libtopspti will be lazily loaded on this call.
    // If it is not available (e.g. GCU is not installed),
    // then this call will return an error and we just abort init.
    auto cbapi = TopsptiCallbackApi::singleton();
    cbapi->initCallbackApi();
    bool status = false;
    bool initRangeProfiler = true;

    if (cbapi->initSuccess()) {
      // const Topspti_CallbackDomain domain = TOPSPTI_CB_DOMAIN_RESOURCE;
      // status = cbapi->registerCallback(
      //     domain, TopsptiCallbackApi::RESOURCE_CONTEXT_CREATED,
      //     initProfilers);
      // if (status) {
      //   status = cbapi->enableCallback(
      //       domain, TopsptiCallbackApi::RESOURCE_CONTEXT_CREATED);
      // }

      // Register stopProfiler callback only for event profiler.
      // This callback is not required for activities tracing.
      // if (enableEventProfiler()) {
      //   if (status) {
      //     status = cbapi->registerCallback(
      //         domain, TopsptiCallbackApi::RESOURCE_CONTEXT_DESTROYED,
      //         stopProfiler);
      //   }
      //   if (status) {
      //     status = cbapi->enableCallback(
      //         domain, TopsptiCallbackApi::RESOURCE_CONTEXT_DESTROYED);
      //   }
      // }

      // TODO: TOPSPTI support TOPSPTI_CB_DOMAIN_RESOURCE
      status = true;
    }

    if (!cbapi->initSuccess() || !status) {
      initRangeProfiler = false;
      cpuOnly = true;
      if (logOnError) {
        TOPSPTI_CALL(cbapi->getTopsptiStatus());
        LOG(WARNING) << "TOPSPTI initialization failed - "
                     << "GCU profiler activities will be missing";
        LOG(INFO)
            << "If you see TOPSPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to "
            << "https://developer.nvidia.com/"
               "nvidia-development-tools-solutions-err-nvgpuctrperm-topspti";
      }
    }

    // initialize TOPSPTI Range Profiler API
    // if (initRangeProfiler) {
    //   rangeProfilerInit = std::make_unique<TopsptiRangeProfilerInit>();
    // }
  }

  if (shouldPreloadTopsptiInstrumentation()) {
    TopsptiActivityApi::forceLoadTopspti();
  }

  ConfigLoader& config_loader = libkineto_gcu::api().configLoader();
  libkineto_gcu::api().registerProfiler(
      std::make_unique<ActivityProfilerProxy>(cpuOnly, config_loader));

#if __linux__
  // When GCU is used the profiler initialization happens on the
  // creation of the first GCU stream (see initProfilers()).
  // This section bootstraps the profiler and its connection to a profiling
  // daemon in the CPU only case.
  if (cpuOnly && getenv(kUseDaemonEnvVar) != nullptr) {
    initProfilersCPU();
    libkineto_gcu::api().configLoader().initBaseConfig();
  }
#endif
}

// The tops driver calls this function if the GCU_INJECTION64_PATH environment
// variable is set
int InitializeInjection(void) {
  LOG(INFO) << "Injection mode: Initializing libkineto_gcu";
  libkineto_gcu_init(false /*cpuOnly*/, true /*logOnError*/);
  return 1;
}

bool hasTestEnvVar() {
  return getenv("GTEST_OUTPUT") != nullptr || getenv("FB_TEST") != nullptr ||
         getenv("PYTORCH_TEST") != nullptr || getenv("TEST_PILOT") != nullptr;
}

void suppressLibkinetoLogMessages() {
  // Only suppress messages if explicit override wasn't provided
  const char* logLevelEnv = getenv("KINETO_LOG_LEVEL");
  // For unit tests, don't suppress log verbosity.
  if (!hasTestEnvVar() && (!logLevelEnv || !*logLevelEnv)) {
    SET_LOG_SEVERITY_LEVEL(ERROR);
  }
}

}  // extern C
