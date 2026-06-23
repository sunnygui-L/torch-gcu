/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

namespace torch_gcu {

namespace aotops {

at::Tensor _grouped_mm_shape_infer(const at::Tensor& self,
                                   const at::Tensor& mat2,
                                   const ::std::optional<at::Tensor>& offs,
                                   const ::std::optional<at::Tensor>& bias,
                                   ::std::optional<at::ScalarType> out_dtype);

}  // namespace aotops

}  // namespace torch_gcu
