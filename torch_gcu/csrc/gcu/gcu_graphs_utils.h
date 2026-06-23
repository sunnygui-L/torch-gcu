/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */
#pragma once
#include "gcu/gcu_functions.h"
#include "gcu/gcu_graphs_c10_utils.h"

// csrc/pytorchbridge/gcu_graphs_c10_utils.h has utils used by both c10 and
// aten. This file adds utils used by aten only.

namespace torch_gcu {

using CaptureId_t = torch_gcu::CaptureId_t;
using CaptureStatus = torch_gcu::CaptureStatus;

// Use this version where you don't want to create a GCU context if none exists.
inline CaptureStatus currentStreamCaptureStatus() {
  // don't create a context if we don't have to
  if (torch_gcu::hasPrimaryContext(torch_gcu::current_device())) {
    return torch_gcu::currentStreamCaptureStatusMayInitCtx();
  } else {
    return CaptureStatus::None;
  }
}

inline void assertNotCapturing(std::string attempt) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(
      status == CaptureStatus::None, attempt,
      " during GCU graph capture. If you need this call to be captured, "
      "please file an issue. "
      "Current topsStreamCaptureStatus: ",
      status);
}
}  // namespace torch_gcu
