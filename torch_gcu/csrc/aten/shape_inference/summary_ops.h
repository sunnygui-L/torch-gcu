/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

at::Tensor& histc_out_shape_infer(const at::Tensor& self, int64_t nbins,
                                  const at::Scalar& min, const at::Scalar& max,
                                  at::Tensor& result);

at::Tensor histc_shape_infer(const at::Tensor& self, int64_t nbins,
                             const at::Scalar& min, const at::Scalar& max);

}  // namespace aotops

}  // namespace torch_gcu
