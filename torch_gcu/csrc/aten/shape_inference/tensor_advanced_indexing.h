/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

at::Tensor& masked_fill__shape_infer(at::Tensor& self, const at::Tensor& mask,
                                     const at::Tensor& value);

at::Tensor& _index_put_impl__shape_infer(
    at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value, bool accumulate, bool unsafe);

at::Tensor& index_fill__shape_infer(at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Scalar& source);

}  // namespace aotops

}  // namespace torch_gcu