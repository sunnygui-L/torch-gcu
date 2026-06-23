/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "aten/shape_inference/dropout.h"

#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

std::tuple<at::Tensor, at::Tensor> native_dropout_shape_infer(
    const at::Tensor &input, double p, c10::optional<bool> opt_train) {
  at::Tensor mask = at::empty_like(
      input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
  at::Tensor ret = at::empty_like(input);
  return std::tuple<at::Tensor, at::Tensor>(ret, mask);
}

at::Tensor native_dropout_backward_shape_infer(const at::Tensor &grad_output,
                                               const at::Tensor &mask,
                                               double scale) {
  at::Tensor ret =
      at::empty_like(grad_output, grad_output.suggest_memory_format());
  return ret;
}

}  // namespace aotops

}  // namespace torch_gcu