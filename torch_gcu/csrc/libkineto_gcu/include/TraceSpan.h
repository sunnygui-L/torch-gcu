/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "gcu/gcu_macros.h"

namespace libkineto_gcu {

struct TORCH_GCU_API TraceSpan {
  TraceSpan() = delete;
  TraceSpan(int64_t startTime, int64_t endTime, std::string name)
      : startTime(startTime), endTime(endTime), name(std::move(name)) {}
  TraceSpan(int opCount, int it, std::string name, std::string prefix)
      : opCount(opCount),
        iteration(it),
        name(std::move(name)),
        prefix(std::move(prefix)) {}

  // FIXME: change to duration?
  int64_t startTime{0};
  int64_t endTime{0};
  int opCount{0};
  int iteration{-1};
  // Name is used to identify timeline
  std::string name;
  // Prefix used to distinguish trace spans on the same timeline
  std::string prefix;
};

}  // namespace libkineto_gcu
