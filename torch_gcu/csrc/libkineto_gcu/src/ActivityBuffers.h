/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <list>
#include <memory>

#include "TopsptiActivityBuffer.h"
#include "libkineto_gcu.h"

namespace libkineto_gcu {

struct ActivityBuffers {
  std::list<std::unique_ptr<libkineto_gcu::CpuTraceBuffer>> cpu;
  std::unique_ptr<TopsptiActivityBufferMap> gcu;

  // Add a wrapper object to the underlying struct stored in the buffer
  template <class T>
  const ITraceActivity& addActivityWrapper(const T& act) {
    wrappers_.push_back(std::make_unique<T>(act));
    return *wrappers_.back().get();
  }

 private:
  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
};

}  // namespace libkineto_gcu
