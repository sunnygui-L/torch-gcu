/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */
#include <ATen/ATen.h>

#include "gcu/gcu_macros.h"

namespace torch_gcu {

namespace aotfusion {

TORCH_GCU_API at::Tensor relu_convolution_fusion_shape_infer(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups);

TORCH_GCU_API at::Tensor relu_convolution_fusion_out(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, at::Tensor& out);

}  // namespace prims

}  // namespace torch_gcu