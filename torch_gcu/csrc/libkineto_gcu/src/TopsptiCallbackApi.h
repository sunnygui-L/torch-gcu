/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <array>
#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>

#include "topspti.h"

namespace libkineto_gcu {

using namespace libkineto_gcu;

/* TopsptiCallbackApi : Provides an abstraction over TOPSPTI callback
 *  interface. This enables various callback functions to be registered
 *  with this class. The class registers a global callback handler that
 *  redirects to the respective callbacks.
 *
 *  Note: one design choice we made is to only support simple function pointers
 *  in order to speed up the implementation for fast path.
 */

using TopsptiCallbackFn = void (*)(Topspti_CallbackDomain domain,
                                   Topspti_CallbackId cbid,
                                   const Topspti_CallbackData* cbInfo);

class TopsptiCallbackApi {
 public:
  /* Global list of supported callback ids
   *  use the class namespace to avoid confusing with TOPSPTI enums*/
  enum TopsptiCallBackID {
    GCU_LAUNCH_KERNEL = 0,
    // can possibly support more callback ids per domain
    //
    __RUNTIME_CB_DOMAIN_START = GCU_LAUNCH_KERNEL,
    GCU_LAUNCH_KERNEL_EXC,  // Used in H100

    // Callbacks under Resource CB domain
    RESOURCE_CONTEXT_CREATED,
    RESOURCE_CONTEXT_DESTROYED,

    __RUNTIME_CB_DOMAIN_END = RESOURCE_CONTEXT_CREATED,
    __RESOURCE_CB_DOMAIN_START = RESOURCE_CONTEXT_CREATED,

    __RESOURCE_CB_DOMAIN_END = RESOURCE_CONTEXT_DESTROYED + 1,
  };

  TopsptiCallbackApi() = default;
  TopsptiCallbackApi(const TopsptiCallbackApi&) = delete;
  TopsptiCallbackApi& operator=(const TopsptiCallbackApi&) = delete;

  static std::shared_ptr<TopsptiCallbackApi> singleton();

  void initCallbackApi();

  bool initSuccess() const { return initSuccess_; }

  TopsptiResult getTopsptiStatus() const { return lastTopsptiStatus_; }

  Topspti_SubscriberHandle getTopsptiSubscriber() const { return subscriber_; }

  bool registerCallback(Topspti_CallbackDomain domain, TopsptiCallBackID cbid,
                        TopsptiCallbackFn cbfn);

  // returns false if callback was not found
  bool deleteCallback(Topspti_CallbackDomain domain, TopsptiCallBackID cbid,
                      TopsptiCallbackFn cbfn);

  // Topspti Callback may be enable for domain and cbid pairs, or domains alone.
  bool enableCallback(Topspti_CallbackDomain domain, Topspti_CallbackId cbid);
  bool disableCallback(Topspti_CallbackDomain domain, Topspti_CallbackId cbid);
  bool enableCallbackDomain(Topspti_CallbackDomain domain);
  bool disableCallbackDomain(Topspti_CallbackDomain domain);
  // Provide this API for when topsptiFinalize is executed, to allow the process
  // to re-enabled all previously running callback subscriptions.
  bool reenableCallbacks();

  // Please do not use this method. This has to be exposed as public
  // so it is accessible from the callback handler
  void __callback_switchboard(Topspti_CallbackDomain domain,
                              Topspti_CallbackId cbid,
                              const Topspti_CallbackData* cbInfo);

 private:
  friend class std::shared_ptr<TopsptiCallbackApi>;

  // For callback table design overview see the .cpp file
  using CallbackList = std::list<TopsptiCallbackFn>;

  // level 2 tables sizes are known at compile time
  constexpr static size_t RUNTIME_CB_DOMAIN_SIZE =
      (__RUNTIME_CB_DOMAIN_END - __RUNTIME_CB_DOMAIN_START);

  constexpr static size_t RESOURCE_CB_DOMAIN_SIZE =
      (__RESOURCE_CB_DOMAIN_END - __RESOURCE_CB_DOMAIN_START);

  // level 1 table is a struct
  struct CallbackTable {
    std::array<CallbackList, RUNTIME_CB_DOMAIN_SIZE> runtime;
    std::array<CallbackList, RESOURCE_CB_DOMAIN_SIZE> resource;

    CallbackList* lookup(Topspti_CallbackDomain domain, TopsptiCallBackID cbid);
  };

  CallbackTable callbacks_;
  bool initSuccess_ = false;
  // Record a list of enabled callbacks, so that after teardown, we can
  // re-enable the callbacks that were turned off to clean topspti context. As
  // an implementation detail, cbid == 0xffffffff means enable the domain.
  std::set<std::pair<Topspti_CallbackDomain, Topspti_CallbackId>>
      enabledCallbacks_;

  // Reader Writer lock types
  using ReaderWriterLock = std::shared_timed_mutex;
  using ReaderLockGuard = std::shared_lock<ReaderWriterLock>;
  using WriteLockGuard = std::unique_lock<ReaderWriterLock>;
  ReaderWriterLock callbackLock_;
  TopsptiResult lastTopsptiStatus_;
  Topspti_SubscriberHandle subscriber_{nullptr};
};

}  // namespace libkineto_gcu
