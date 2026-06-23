/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

std::tuple<at::Tensor, at::Tensor> native_dropout_shape_infer(
    const at::Tensor &input, double p, c10::optional<bool> opt_train);

}  // namespace aotops

}  // namespace torch_gcu