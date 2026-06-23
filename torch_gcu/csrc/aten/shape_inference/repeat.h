/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

at::Tensor repeat_interleave_shape_infer(const at::Tensor &repeats,
                                         c10::optional<int64_t> output_size);

at::Tensor repeat_interleave_shape_infer(const at::Tensor &self,
                                         const at::Tensor &repeats,
                                         c10::optional<int64_t> dim,
                                         c10::optional<int64_t> output_size);

}  // namespace aotops

}  // namespace torch_gcu
