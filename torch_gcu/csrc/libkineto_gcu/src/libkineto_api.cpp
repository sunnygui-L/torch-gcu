/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "ConfigLoader.h"
#include "gcu/thread_util.h"
#include "libkineto_gcu.h"

namespace libkineto_gcu {

LibkinetoApi& api() {
  static LibkinetoApi instance(ConfigLoader::instance());
  return instance;
}

void LibkinetoApi::initClientIfRegistered() {
  if (client_) {
    if (clientRegisterThread_ != torch_gcu::util::threadId()) {
      fprintf(stderr,
              "ERROR: External init callback must run in same thread as "
              "registerClient "
              "(%d != %d)\n",
              torch_gcu::util::threadId(), (int)clientRegisterThread_);
    } else {
      client_->init();
    }
  }
}

void LibkinetoApi::registerClient(ClientInterface* client) {
  client_ = client;
  if (client && activityProfiler_) {
    // Can initialize straight away
    client->init();
  }
  // Assume here that the external init callback is *not* threadsafe
  // and only call it if it's the same thread that called registerClient
  clientRegisterThread_ = torch_gcu::util::threadId();
}

}  // namespace libkineto_gcu
