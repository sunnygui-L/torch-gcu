/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/core/Tensor.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_resize.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {
namespace aotops {
::std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward_shape_infer(
    const at::Tensor &self) {
  auto shape = self.sizes().vec();
  auto output = empty(shape, self.options());
  auto buffer = empty(shape, self.options());
  return std::make_tuple(std::move(output), std::move(buffer));
}

::std::tuple<at::Tensor &, at::Tensor &> log_sigmoid_forward_out_shape_infer(
    const at::Tensor &self, at::Tensor &output, at::Tensor &buffer) {
  auto shape = self.sizes().vec();
  resize_output(output, shape);
  resize_output(buffer, shape);
  return std::forward_as_tuple(output, buffer);
}

}  // namespace aotops
}  // namespace torch_gcu
