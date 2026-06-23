/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_shape_infer(
    const at::Tensor &input, const c10::optional<at::Tensor> &weight_opt,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &running_mean_opt,
    const c10::optional<at::Tensor> &running_var_opt, bool training,
    double momentum, double eps);

}  // namespace aotops

}  // namespace torch_gcu