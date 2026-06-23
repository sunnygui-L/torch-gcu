/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/native/cpu/mixed_data_type.h>

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

template <typename T>
void check_group_norm_inputs(const at::Tensor& input, const at::Tensor& weight,
                             const at::Tensor& bias, T C, int64_t num_groups) {
  TORCH_CHECK(num_groups > 0, "Expected num groups to be greater than 0, got ",
              num_groups);
  TORCH_CHECK(C % num_groups == 0,
              "Expected number of channels in input to be divisible by ",
              "num_groups, but got input of shape ", input.sizes(),
              " and "
              "num_groups=",
              num_groups);
  TORCH_CHECK(!weight.defined() ||
                  (weight.dim() == 1 && at::symint::numel<T>(weight) == C),
              "Expected weight to be a vector of size equal to the number of ",
              "channels in input, but got weight of shape ", weight.sizes(),
              " and input of shape ", input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && at::symint::numel<T>(bias) == C),
      "Expected bias to be a vector of size equal to the number of ",
      "channels in input, but got bias of shape ", weight.sizes(),
      " and input of shape ", input.sizes());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_shape_infer(
    const at::Tensor& X,
    const c10::optional<at::Tensor>& gamma_opt /* optional */,
    const c10::optional<at::Tensor>& beta_opt /* optional */, int64_t N,
    int64_t C, int64_t HxW, int64_t group, double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const at::Tensor& gamma = *gamma_maybe_owned;
  const at::Tensor& beta =
      c10::value_or_else(beta_opt, [] { return at::Tensor(); });

  // repeated check so expanded weights can call native_group_norm directly but
  // save mean and variance from forward
  check_group_norm_inputs(X, gamma, beta, C, group);
  auto memory_format = X.device().is_privateuseone()
                           ? X.suggest_memory_format()
                           : at::MemoryFormat::Contiguous;

  TORCH_CHECK(X.is_contiguous(memory_format));

  bool mixed_type = at::native::is_mixed_type(X, gamma, beta);

  at::Tensor Y = at::native::empty_like(
      X, c10::nullopt /* dtype */, c10::nullopt /* layout */,
      c10::nullopt /* device */, c10::nullopt /* pin_memory */, memory_format);
  const auto dtype = at::native::param_scalar_type(X, mixed_type);
  at::Tensor mean = at::empty({N, group}, X.options().dtype(dtype));
  at::Tensor rstd = at::empty({N, group}, X.options().dtype(dtype));

  return std::make_tuple(Y, mean, rstd);
}

}  // namespace aotops

}  // namespace torch_gcu
