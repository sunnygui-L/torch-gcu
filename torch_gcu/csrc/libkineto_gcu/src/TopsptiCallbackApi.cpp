/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "TopsptiCallbackApi.h"

#include <assert.h>

#include <algorithm>
#include <chrono>
#include <mutex>

#include "DeviceUtil.h"
#include "Logger.h"
#include "TopsptiActivityApi.h"

namespace libkineto_gcu {

// limit on number of handles per callback type
constexpr size_t MAX_CB_FNS_PER_CB = 8;

// Use this value in enabledCallbacks_ set, when all cbids in a domain
// is enabled, not a specific cbid.
constexpr uint32_t MAX_TOPSPTI_CALLBACK_ID_ALL = 0xffffffff;

/* Callback Table :
 *  Overall goal of the design is to optimize the lookup of function
 *  pointers. The table is structured at two levels and the leaf
 *  elements in the table are std::list to enable fast access/inserts/deletes
 *
 *   <callback domain0> |
 *                     -> cb id 0 -> std::list of callbacks
 *                     ...
 *                     -> cb id n -> std::list of callbacks
 *   <callback domain1> |
 *                    ...
 *  CallbackTable is the finally table type above
 *  See type declrartions in header file.
 */

/* callback_switchboard : is the global callback handler we register
 *  with TOPSPTI. The goal is to make it as efficient as possible
 *  to re-direct to the registered callback(s).
 *
 *  Few things to care about :
 *   a) use if/then switches rather than map/hash structures
 *   b) avoid dynamic memory allocations
 *   c) be aware of locking overheads
 */

static void callback_switchboard(void* /* unused */,
                                 Topspti_CallbackDomain domain,
                                 Topspti_CallbackId cbid,
                                 const Topspti_CallbackData* cbInfo) {
  // below statement is likely going to call a mutex
  // on the singleton access
  TopsptiCallbackApi::singleton()->__callback_switchboard(domain, cbid, cbInfo);
}

void TopsptiCallbackApi::__callback_switchboard(
    Topspti_CallbackDomain domain, Topspti_CallbackId cbid,
    const Topspti_CallbackData* cbInfo) {
  LOG(INFO) << "Callback: domain = " << domain << ", cbid = " << cbid;
  CallbackList* cblist = nullptr;

  switch (domain) {
    // add the fastest path for kernel launch callbacks
    // as these are the most frequent ones
    case TOPSPTI_CB_DOMAIN_RUNTIME_API:
      switch (cbid) {
        case TOPSPTI_RUNTIME_TRACE_CBID_topsLaunchKernel:
          cblist = &callbacks_
                        .runtime[GCU_LAUNCH_KERNEL - __RUNTIME_CB_DOMAIN_START];
          break;
        case TOPSPTI_RUNTIME_TRACE_CBID_topsLaunchKernelExC:
          cblist =
              &callbacks_
                   .runtime[GCU_LAUNCH_KERNEL_EXC - __RUNTIME_CB_DOMAIN_START];
          break;
        default:
          break;
      }
      // This is required to teardown topspti after profiling to prevent QPS
      // slowdown.
      if (TopsptiActivityApi::singleton().teardownTopspti_) {
        if (cbInfo->callbackSite == TOPSPTI_API_EXIT) {
          LOG(INFO) << "  Calling topsptiFinalize in exit callsite";
          // Teardown TOPSPTI calling topsptiFinalize()
          TOPSPTI_CALL(topsptiUnsubscribe(subscriber_));
          TOPSPTI_CALL(topsptiFinalize());
          initSuccess_ = false;
          subscriber_ = nullptr;
          TopsptiActivityApi::singleton().teardownTopspti_ = 0;
          TopsptiActivityApi::singleton().finalizeCond_.notify_all();
          return;
        }
      }
      break;

      // case TOPSPTI_CB_DOMAIN_RESOURCE:
      //   switch (cbid) {
      //     case TOPSPTI_CBID_RESOURCE_CONTEXT_CREATED:
      //       cblist = &callbacks_.resource[RESOURCE_CONTEXT_CREATED -
      //                                     __RESOURCE_CB_DOMAIN_START];
      //       break;
      //     case TOPSPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING:
      //       cblist = &callbacks_.resource[RESOURCE_CONTEXT_DESTROYED -
      //                                     __RESOURCE_CB_DOMAIN_START];
      //       break;
      //     default:
      //       break;
      //   }
      //   break;

    default:
      return;
  }

  // ignore callbacks that are not handled
  if (cblist == nullptr) {
    return;
  }

  // make a copy of the callback list so we avoid holding lock
  // in common case this should be just one func pointer copy
  std::array<TopsptiCallbackFn, MAX_CB_FNS_PER_CB> callbacks;
  int num_cbs = 0;
  {
    ReaderLockGuard rl(callbackLock_);
    int i = 0;
    for (auto it = cblist->begin();
         it != cblist->end() && i < MAX_CB_FNS_PER_CB; it++, i++) {
      callbacks[i] = *it;
    }
    num_cbs = i;
  }

  for (int i = 0; i < num_cbs; i++) {
    auto fn = callbacks[i];
    fn(domain, cbid, cbInfo);
  }
}

std::shared_ptr<TopsptiCallbackApi> TopsptiCallbackApi::singleton() {
  static const std::shared_ptr<TopsptiCallbackApi> instance = [] {
    std::shared_ptr<TopsptiCallbackApi> inst =
        std::shared_ptr<TopsptiCallbackApi>(new TopsptiCallbackApi());
    return inst;
  }();
  return instance;
}

void TopsptiCallbackApi::initCallbackApi() {
  lastTopsptiStatus_ = TOPSPTI_ERROR_UNKNOWN;
  lastTopsptiStatus_ = TOPSPTI_CALL_NOWARN(topsptiSubscribe(
      &subscriber_, (Topspti_CallbackFunc)callback_switchboard, nullptr));

  // TODO: Remove temporarily to work around static initialization order issue
  // between this and GLOG.
  // if (lastTopsptiStatus_ != TOPSPTI_SUCCESS) {
  //   LOG(INFO) << "Failed topsptiSubscribe, status: " << lastTopsptiStatus_;
  // }

  initSuccess_ = (lastTopsptiStatus_ == TOPSPTI_SUCCESS);
}

TopsptiCallbackApi::CallbackList* TopsptiCallbackApi::CallbackTable::lookup(
    Topspti_CallbackDomain domain, TopsptiCallBackID cbid) {
  size_t idx;

  switch (domain) {
      // case TOPSPTI_CB_DOMAIN_RESOURCE:
      //   assert(cbid >= __RESOURCE_CB_DOMAIN_START);
      //   assert(cbid < __RESOURCE_CB_DOMAIN_END);
      //   idx = cbid - __RESOURCE_CB_DOMAIN_START;
      //   return &resource.at(idx);

    case TOPSPTI_CB_DOMAIN_RUNTIME_API:
      assert(cbid >= __RUNTIME_CB_DOMAIN_START);
      assert(cbid < __RUNTIME_CB_DOMAIN_END);
      idx = cbid - __RUNTIME_CB_DOMAIN_START;
      return &runtime.at(idx);

    default:
      LOG(WARNING) << " Unsupported callback domain : " << domain;
      return nullptr;
  }
}

bool TopsptiCallbackApi::registerCallback(Topspti_CallbackDomain domain,
                                          TopsptiCallBackID cbid,
                                          TopsptiCallbackFn cbfn) {
  CallbackList* cblist = callbacks_.lookup(domain, cbid);

  if (!cblist) {
    LOG(WARNING) << "Could not register callback -- domain = " << domain
                 << " callback id = " << cbid;
    return false;
  }

  // avoid duplicates
  auto it = std::find(cblist->begin(), cblist->end(), cbfn);
  if (it != cblist->end()) {
    LOG(WARNING) << "Adding duplicate callback -- domain = " << domain
                 << " callback id = " << cbid;
    return true;
  }

  if (cblist->size() == MAX_CB_FNS_PER_CB) {
    LOG(WARNING) << "Already registered max callback -- domain = " << domain
                 << " callback id = " << cbid;
  }

  WriteLockGuard wl(callbackLock_);
  cblist->push_back(cbfn);
  return true;
}

bool TopsptiCallbackApi::deleteCallback(Topspti_CallbackDomain domain,
                                        TopsptiCallBackID cbid,
                                        TopsptiCallbackFn cbfn) {
  CallbackList* cblist = callbacks_.lookup(domain, cbid);
  if (!cblist) {
    LOG(WARNING) << "Attempting to remove unsupported callback -- domain = "
                 << domain << " callback id = " << cbid;
    return false;
  }

  // Locks are not required here as
  //  https://en.cppreference.com/w/cpp/container/list/erase
  //  "References and iterators to the erased elements are invalidated.
  //   Other references and iterators are not affected."
  auto it = std::find(cblist->begin(), cblist->end(), cbfn);
  if (it == cblist->end()) {
    LOG(WARNING) << "Could not find callback to remove -- domain = " << domain
                 << " callback id = " << cbid;
    return false;
  }

  WriteLockGuard wl(callbackLock_);
  cblist->erase(it);
  return true;
}

bool TopsptiCallbackApi::enableCallback(Topspti_CallbackDomain domain,
                                        Topspti_CallbackId cbid) {
  if (initSuccess_) {
    lastTopsptiStatus_ = TOPSPTI_CALL_NOWARN(
        topsptiEnableCallback(1, subscriber_, domain, cbid));
    enabledCallbacks_.insert({domain, cbid});
    return (lastTopsptiStatus_ == TOPSPTI_SUCCESS);
  }

  return false;
}

bool TopsptiCallbackApi::disableCallback(Topspti_CallbackDomain domain,
                                         Topspti_CallbackId cbid) {
  enabledCallbacks_.erase({domain, cbid});
  if (initSuccess_) {
    lastTopsptiStatus_ = TOPSPTI_CALL_NOWARN(
        topsptiEnableCallback(0, subscriber_, domain, cbid));
    return (lastTopsptiStatus_ == TOPSPTI_SUCCESS);
  }

  return false;
}

bool TopsptiCallbackApi::enableCallbackDomain(Topspti_CallbackDomain domain) {
  if (initSuccess_) {
    lastTopsptiStatus_ =
        TOPSPTI_CALL_NOWARN(topsptiEnableDomain(1, subscriber_, domain));
    enabledCallbacks_.insert({domain, MAX_TOPSPTI_CALLBACK_ID_ALL});
    return (lastTopsptiStatus_ == TOPSPTI_SUCCESS);
  }

  return false;
}

bool TopsptiCallbackApi::disableCallbackDomain(Topspti_CallbackDomain domain) {
  enabledCallbacks_.erase({domain, MAX_TOPSPTI_CALLBACK_ID_ALL});
  if (initSuccess_) {
    lastTopsptiStatus_ =
        TOPSPTI_CALL_NOWARN(topsptiEnableDomain(0, subscriber_, domain));
    return (lastTopsptiStatus_ == TOPSPTI_SUCCESS);
  }

  return false;
}

bool TopsptiCallbackApi::reenableCallbacks() {
  if (initSuccess_) {
    for (auto& cbpair : enabledCallbacks_) {
      if ((uint32_t)cbpair.second == MAX_TOPSPTI_CALLBACK_ID_ALL) {
        lastTopsptiStatus_ = TOPSPTI_CALL_NOWARN(
            topsptiEnableDomain(1, subscriber_, cbpair.first));
      } else {
        lastTopsptiStatus_ = TOPSPTI_CALL_NOWARN(
            topsptiEnableCallback(1, subscriber_, cbpair.first, cbpair.second));
      }
    }
    return (lastTopsptiStatus_ == TOPSPTI_SUCCESS);
  }

  return false;
}

}  // namespace libkineto_gcu
