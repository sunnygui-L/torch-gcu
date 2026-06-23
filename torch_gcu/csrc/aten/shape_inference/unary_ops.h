/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

at::Tensor real_shape_infer(const at::Tensor& self);

at::Tensor imag_shape_infer(const at::Tensor& self);

}  // namespace aotops

}  // namespace torch_gcu
