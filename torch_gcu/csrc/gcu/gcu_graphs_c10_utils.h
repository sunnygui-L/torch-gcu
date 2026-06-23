/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */
#pragma once

#include "gcu/gcu_exception.h"
#include "gcu/gcu_stream.h"
#include <utility>

// GCU Graphs utils used by c10 and aten.
// gcu/gcu_graphs_utils.h adds utils used by aten only.

namespace torch_gcu {

using CaptureId_t = unsigned long long;

// first is set if the instance is created by GCUGraph::capture_begin.
// second is set if the instance is created by torch_gcu::graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

// RAII guard for "topsStreamCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
struct C10_GCU_API GCUStreamCaptureModeGuard {
  GCUStreamCaptureModeGuard(topsStreamCaptureMode desired) {
    strictness_ = desired;
    C10_GCU_CHECK(topsThreadExchangeStreamCaptureMode(&strictness_));
  }
  ~GCUStreamCaptureModeGuard() {
    C10_GCU_CHECK_WARN(topsThreadExchangeStreamCaptureMode(&strictness_));
  }

 private:
  topsStreamCaptureMode strictness_;
};

// Protects against enum topsStreamCaptureStatus implementation changes.
// Some compilers seem not to like static_assert without the messages.
static_assert(int(topsStreamCaptureStatus::topsStreamCaptureStatusNone) == 0,
              "unexpected int(topsStreamCaptureStatusNone) value");
static_assert(int(topsStreamCaptureStatus::topsStreamCaptureStatusActive) == 1,
              "unexpected int(topsStreamCaptureStatusActive) value");
static_assert(
    int(topsStreamCaptureStatus::topsStreamCaptureStatusInvalidated) == 2,
    "unexpected int(topsStreamCaptureStatusInvalidated) value");

enum class CaptureStatus : int {
  None = int(topsStreamCaptureStatus::topsStreamCaptureStatusNone),
  Active = int(topsStreamCaptureStatus::topsStreamCaptureStatusActive),
  Invalidated = int(topsStreamCaptureStatus::topsStreamCaptureStatusInvalidated)
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::None:
      os << "topsStreamCaptureStatusNone";
      break;
    case CaptureStatus::Active:
      os << "topsStreamCaptureStatusActive";
      break;
    case CaptureStatus::Invalidated:
      os << "topsStreamCaptureStatusInvalidated";
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown GCU graph CaptureStatus",
                            int(status));
  }
  return os;
}

// Use this version where you're sure a GCU context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  topsStreamCaptureStatus is_capturing;
  C10_GCU_CHECK(
      topsStreamIsCapturing(torch_gcu::getCurrentGCUStream(), &is_capturing));
  return CaptureStatus(is_capturing);
}

}  // namespace torch_gcu
