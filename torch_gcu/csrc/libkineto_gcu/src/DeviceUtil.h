/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <fmt/format.h>
#include <tops_runtime.h>

#include "topspti.h"

#define GCU_CALL(call)                                                     \
  [&]() -> topsError_t {                                                   \
    topsError_t _status_ = call;                                           \
    if (_status_ != topsSuccess) {                                         \
      const char* _errstr_ = topsGetErrorString(_status_);                 \
      LOG(WARNING) << fmt::format("function {} failed with error {} ({})", \
                                  #call, _errstr_, (int)_status_);         \
    }                                                                      \
    return _status_;                                                       \
  }()

#define TOPSPTI_CALL(call)                                                 \
  [&]() -> TopsptiResult {                                                 \
    TopsptiResult _status_ = call;                                         \
    if (_status_ != TOPSPTI_SUCCESS) {                                     \
      const char* _errstr_ = nullptr;                                      \
      topsptiGetResultString(_status_, &_errstr_);                         \
      LOG(WARNING) << fmt::format("function {} failed with error {} ({})", \
                                  #call, _errstr_, (int)_status_);         \
    }                                                                      \
    return _status_;                                                       \
  }()

#define TOPSPTI_CALL(call) call

#define TOPSPTI_CALL(call) call

#define TOPSPTI_CALL_NOWARN(call) call

namespace libkineto_gcu {

bool isGcuAvailable();

}  // namespace libkineto_gcu
