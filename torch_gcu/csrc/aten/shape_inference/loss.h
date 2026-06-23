/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

namespace torch_gcu {

namespace aotops {

at::Tensor& mse_loss_backward_out_shape_infer(const at::Tensor& grad_output,
                                              const at::Tensor& self,
                                              const at::Tensor& target,
                                              int64_t reduction,
                                              at::Tensor& grad_input);

at::Tensor mse_loss_backward_shape_infer(const at::Tensor& grad_output,
                                         const at::Tensor& self,
                                         const at::Tensor& target,
                                         int64_t reduction);

}  // namespace aotops

}  // namespace torch_gcu
