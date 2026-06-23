/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/core/Tensor.h>

namespace torch_gcu {

namespace aotops {

void _foreach_mul__shape_infer(at::TensorList tensors,
                               const at::Tensor& scalar);

std::vector<at::Tensor> _foreach_mul_shape_infer(at::TensorList tensors,
                                                 const at::Tensor& scalar);

void _foreach_div__shape_infer(at::TensorList tensors,
                               const at::Tensor& scalar);

std::vector<at::Tensor> _foreach_div_shape_infer(at::TensorList tensors,
                                                 const at::Tensor& scalar);

}  // namespace aotops

}  // namespace torch_gcu
