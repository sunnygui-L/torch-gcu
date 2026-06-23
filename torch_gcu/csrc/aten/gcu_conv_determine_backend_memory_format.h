/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/ConvUtils.h>
#include "gcu/gcu_macros.h"

namespace torch_gcu {

TORCH_GCU_API at::MemoryFormat _determine_backend_memory_format(
    const at::Tensor &input, const at::Tensor &weight,
    const at::native::ConvBackend backend =
        at::native::ConvBackend::Overrideable);

}  // namespace torch_gcu
