/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

at::Tensor floor_divide_shape_infer(const at::Tensor& self,
                                    const at::Tensor& other);

at::Tensor& floor_divide_out_shape_infer(const at::Tensor& self,
                                         const at::Tensor& other,
                                         at::Tensor& out);

at::Tensor& floor_divide__shape_infer(at::Tensor& self,
                                      const at::Tensor& other);

}  // namespace aotops

}  // namespace torch_gcu