/*
 * Copyright 2024-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/Tensor.h>

#include "gcu/gcu_macros.h"

namespace torch_gcu {

// Get DatePtr from GCU-Tensor
TORCH_GCU_API void* getGCUTensorDataPtr(const at::Tensor& input);

class TORCH_GCU_API GCUInitialization {
 public:
  GCUInitialization();
};

static GCUInitialization globalInit;

}  // namespace torch_gcu
