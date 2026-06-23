/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */

#include <ATen/TensorIterator.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_resize.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

namespace {

void check_inputs_nll_loss2d(const at::Tensor& input, const at::Tensor& target,
                             const at::Tensor& weight) {
  TORCH_CHECK(target.dim() == 3,
              "only batches of spatial targets supported (3D tensors)"
              " but got targets of size: : ",
              target.sizes());
  TORCH_CHECK(input.dim() == 4,
              "only batches of spatial inputs supported (4D tensors), "
              "but got input of size: ",
              input.sizes());
  TORCH_CHECK(!weight.defined() || weight.numel() == input.size(1),
              "weight tensor should be defined either for all or no classes");

  TORCH_CHECK(input.size(0) == target.size(0) &&
                  input.size(2) == target.size(1) &&
                  input.size(3) == target.size(2),
              "input and target batch or spatial sizes don't match: target ",
              target.sizes(), ", input ", input.sizes());
}

void nll_loss2d_forward_out_gcu_template(
    at::Tensor& output, at::Tensor& total_weight, const at::Tensor& input,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction, int64_t ignore_index) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage in 'sum' or 'mean' reductions.
  if (reduction != at::Reduction::None) {
    at::globalContext().alertNotDeterministic(
        "nll_loss2d_forward_out_gcu_template");
  }

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;

  check_inputs_nll_loss2d(input, target, weight);
  total_weight.resize_({});

  if (reduction == at::Reduction::None) {
    int64_t batch_size = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t count = batch_size * H * W;

    aotops::resize_output(output, {batch_size, H, W});
    if (count == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with
      // 0 blocks.
      return;
    }
    return;
  }

  // produce scalar outputs for the reduction case
   aotops::resize_output(output, {});
}

}  // namespace

::std::tuple<at::Tensor, at::Tensor> nll_loss2d_forward_shape_infer(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt, int64_t reduction,
    int64_t ignore_index) {
  auto output = aotops::empty({0}, self.options());
  auto total_weight = aotops::empty({0}, self.options());
  nll_loss2d_forward_out_gcu_template(output, total_weight, self, target,
                                      weight_opt, reduction, ignore_index);
  return std::tuple<at::Tensor&, at::Tensor&>(output, total_weight);
}

::std::tuple<at::Tensor&, at::Tensor&> nll_loss2d_forward_out_shape_infer(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt, int64_t reduction,
    int64_t ignore_index, at::Tensor& output, at::Tensor& total_weight) {
  nll_loss2d_forward_out_gcu_template(output, total_weight, self, target,
                                      weight_opt, reduction, ignore_index);
  return std::tuple<at::Tensor&, at::Tensor&>(output, total_weight);
}

}  // namespace aotops

}  // namespace torch_gcu
