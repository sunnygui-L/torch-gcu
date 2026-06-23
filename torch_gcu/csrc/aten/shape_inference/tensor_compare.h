/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

at::Tensor &where_out_shape_infer(const at::Tensor &condition,
                                  const at::Tensor &self,
                                  const at::Tensor &other, at::Tensor &out);

at::Tensor where_shape_infer(const at::Tensor &condition,
                             const at::Tensor &self, const at::Tensor &other);

}  // namespace aotops

}  // namespace torch_gcu