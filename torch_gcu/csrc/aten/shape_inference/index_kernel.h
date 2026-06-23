/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */
#pragma once
#include <ATen/core/Tensor.h>

namespace torch_gcu {

namespace aotops {

at::Tensor masked_select_shape_infer(const at::Tensor& self,
                                     const at::Tensor& mask);

at::Tensor& masked_select_out_shape_infer(const at::Tensor& self,
                                          const at::Tensor& mask,
                                          at::Tensor& result);

at::Tensor& masked_scatter__shape_infer(at::Tensor& self,
                                        const at::Tensor& mask,
                                        const at::Tensor& source);
}  // namespace aotops

}  // namespace torch_gcu
