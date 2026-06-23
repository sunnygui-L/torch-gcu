/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "gcu/gcu_macros.h"

namespace libkineto_gcu {

struct ITraceActivity;

class TORCH_GCU_API ActivityTraceInterface {
 public:
  virtual ~ActivityTraceInterface() {}
  virtual const std::vector<const ITraceActivity*>* activities() {
    return nullptr;
  }
  virtual void save(const std::string& path) {}
};

}  // namespace libkineto_gcu
