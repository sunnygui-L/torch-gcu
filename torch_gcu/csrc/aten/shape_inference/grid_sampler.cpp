/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

at::Tensor grid_sampler_2d_shape_infer(const at::Tensor& input,
                                       const at::Tensor& grid,
                                       int64_t interpolation_mode,
                                       int64_t padding_mode,
                                       bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = aotops::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
  return output;
}

at::Tensor grid_sampler_3d_shape_infer(const at::Tensor& input,
                                       const at::Tensor& grid,
                                       int64_t interpolation_mode,
                                       int64_t padding_mode,
                                       bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = aotops::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
      input.options());
  return output;
}

}  // namespace aotops

}  // namespace torch_gcu
