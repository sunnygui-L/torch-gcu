/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */
#include "aotfusion/conv_fusion.h"

#include "aten/GCUNativeFunctions.h"
#include "aten/shape_inference/aotops_shape_infer_func.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotfusion {

at::Tensor relu_convolution_fusion_shape_infer(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  auto relu_out = input;
  return aotops::convolution_shape_infer(relu_out, weight, bias, stride,
                                         padding, dilation, transposed,
                                         output_padding, groups);
}

at::Tensor relu_convolution_fusion_out(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, at::Tensor& out) {
  auto relu_out = GCUNativeFunctions::relu(input);
  return GCUNativeFunctions::convolution(relu_out, weight, bias, stride,
                                         padding, dilation, transposed,
                                         output_padding, groups);
}

}  // namespace aotfusion

}  // namespace torch_gcu