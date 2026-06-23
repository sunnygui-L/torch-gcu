/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include <ATen/core/Tensor.h>

#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

std::tuple<at::Tensor, at::Tensor> ps_roi_align_shape_infer(
    const at::Tensor& input, const at::Tensor& rois, double spatial_scale,
    int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio) {
  // Check if input tensors are CUDA tensors
  TORCH_CHECK(input.is_privateuseone(), "input must be a GCU tensor");
  TORCH_CHECK(rois.is_privateuseone(), "rois must be a GCU tensor");
  TORCH_CHECK(rois.size(1) == 5,
              "Tensor rois should have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ps_roi_align_shape_infer";
  at::checkAllSameType(c, {input_t, rois_t});

  auto num_rois = rois.size(0);
  auto channels = input.size(1);

  TORCH_CHECK(
      channels % (pooled_height * pooled_width) == 0,
      "input channels must be a multiple of pooling height * pooling width");
  int channels_out = channels / (pooled_height * pooled_width);

  auto output = at::zeros({num_rois, channels_out, pooled_height, pooled_width},
                          input.options());
  auto channel_mapping =
      at::zeros(output.sizes(), input.options().dtype(at::kInt));

  return std::make_tuple(output, channel_mapping);
}

std::tuple<at::Tensor, at::Tensor> ps_roi_pool_shape_infer(
    const at::Tensor& input, const at::Tensor& rois, double spatial_scale,
    int64_t pooled_height, int64_t pooled_width) {
  // Check if input tensors are CUDA tensors
  TORCH_CHECK(input.is_privateuseone(), "input must be a GCU tensor");
  TORCH_CHECK(rois.is_privateuseone(), "rois must be a GCU tensor");
  TORCH_CHECK(rois.size(1) == 5,
              "Tensor rois should have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ps_roi_pool_shape_infer";
  at::checkAllSameType(c, {input_t, rois_t});

  auto num_rois = rois.size(0);
  auto channels = input.size(1);

  TORCH_CHECK(
      channels % (pooled_height * pooled_width) == 0,
      "input channels must be a multiple of pooling height * pooling width");
  int channels_out = channels / (pooled_height * pooled_width);

  auto output = at::zeros({num_rois, channels_out, pooled_height, pooled_width},
                          input.options());
  auto channel_mapping =
      at::zeros(output.sizes(), input.options().dtype(at::kInt));

  return std::make_tuple(output, channel_mapping);
}

at::Tensor roi_align_shape_infer(const at::Tensor& input,
                                 const at::Tensor& rois, double spatial_scale,
                                 int64_t pooled_height, int64_t pooled_width,
                                 int64_t sampling_ratio, bool aligned) {
  TORCH_CHECK(input.is_privateuseone(), "input must be a GCU tensor");
  TORCH_CHECK(rois.is_privateuseone(), "rois must be a GCU tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};
  at::CheckedFrom c = "roi_align_shape_infer";
  at::checkAllSameType(c, {input_t, rois_t});

  auto num_rois = rois.size(0);
  auto channels = input.size(1);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());

  return output;
}

::std::tuple<at::Tensor, at::Tensor> roi_pool_shape_infer(
    const at::Tensor& input, const at::Tensor& rois, double spatial_scale,
    int64_t pooled_height, int64_t pooled_width) {
  TORCH_CHECK(input.is_privateuseone(), "input must be a GCU tensor");
  TORCH_CHECK(rois.is_privateuseone(), "rois must be a GCU tensor");
  TORCH_CHECK(rois.size(1) == 5,
              "Tensor rois should have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_pool_forward_kernel";
  at::checkAllSameType(c, {input_t, rois_t});

  auto num_rois = rois.size(0);
  auto channels = input.size(1);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  at::Tensor argmax =
      at::zeros({num_rois, channels, pooled_height, pooled_width},
                input.options().dtype(at::kInt));

  return std::make_tuple(output, argmax);
}

}  // namespace aotops

}  // namespace torch_gcu