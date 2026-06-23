/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

at::Tensor& arange_out_shape_infer(const at::Scalar& start,
                                   const at::Scalar& end,
                                   const at::Scalar& step, at::Tensor& result);

}  // namespace aotops

}  // namespace torch_gcu
