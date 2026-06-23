/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include "gcu/gcu_macros.h"
namespace libkineto_gcu {

class TORCH_GCU_API ClientInterface {
 public:
  virtual ~ClientInterface() {}
  virtual void init() = 0;
  virtual void prepare(bool, bool, bool, bool, bool) = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
};

}  // namespace libkineto_gcu
