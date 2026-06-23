/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <chrono>

#include "gcu/gcu_macros.h"

namespace libkineto_gcu {
template <class ClockT>
TORCH_GCU_API inline int64_t timeSinceEpoch(
    const std::chrono::time_point<ClockT>& t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             t.time_since_epoch())
      .count();
}

}  // namespace libkineto_gcu
