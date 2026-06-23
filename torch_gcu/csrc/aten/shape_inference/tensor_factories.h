/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

at::Tensor& polar_out_shape_infer(const at::Tensor& abs,
                                  const at::Tensor& angle, at::Tensor& result);

at::Tensor& randperm_out_shape_infer(int64_t n,
                                     c10::optional<at::Generator> generator,
                                     at::Tensor& result);

}  // namespace aotops

}  // namespace torch_gcu