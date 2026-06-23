/*
 * Copyright 2024 Enflame. All Rights Reserved.
 */

#include <ATen/native/layer_norm.h>

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

::std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_layer_norm_backward_shape_infer(
    const at::Tensor& grad_out, const at::Tensor& input,
    at::IntArrayRef normalized_shape, const at::Tensor& mean,
    const at::Tensor& rstd, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, ::std::array<bool, 3> output_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight);
  const at::Tensor& weight_ = *weight_maybe_owned;
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor& bias_ = *bias_maybe_owned;

  auto M_N = at::native::_check_layer_norm_inputs(input, normalized_shape,
                                                  weight_, bias_);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight_.expect_contiguous();
  auto beta = bias_.expect_contiguous();

  at::Tensor dX, dgamma, dbeta;
  if (output_mask[0]) {
    dX = at::native::empty_like(
        *X, c10::nullopt /* dtype */, c10::nullopt /* layout */,
        c10::nullopt /* device */, c10::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (output_mask[1]) {
    dgamma = at::native::empty_like(
        *gamma, c10::nullopt /* dtype */, c10::nullopt /* layout */,
        c10::nullopt /* device */, c10::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (output_mask[2]) {
    dbeta = at::native::empty_like(
        *beta, c10::nullopt /* dtype */, c10::nullopt /* layout */,
        c10::nullopt /* device */, c10::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}

}  // namespace aotops

}  // namespace torch_gcu