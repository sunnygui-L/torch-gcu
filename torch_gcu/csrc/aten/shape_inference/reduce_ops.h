/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

at::Tensor logsumexp_shape_infer(const at::Tensor& self, at::IntArrayRef dim,
                                 bool keepdim);

at::Tensor& logsumexp_out_shape_infer(const at::Tensor& self,
                                      at::IntArrayRef dim, bool keepdim,
                                      at::Tensor& out);

::std::tuple<at::Tensor, at::Tensor> std_mean_shape_infer(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction, bool keepdim);

::std::tuple<at::Tensor, at::Tensor> var_mean_shape_infer(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction, bool keepdim);

}  // namespace aotops

}  // namespace torch_gcu
