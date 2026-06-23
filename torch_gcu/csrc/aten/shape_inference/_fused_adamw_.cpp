/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {
namespace aotops {
void _fused_adamw__shape_infer(at::TensorList self, at::TensorList grads,
                               at::TensorList exp_avgs,
                               at::TensorList exp_avg_sqs,
                               at::TensorList max_exp_avg_sqs,
                               at::TensorList state_steps, double lr,
                               double beta1, double beta2, double weight_decay,
                               double eps, bool amsgrad, bool maximize,
                               const ::std::optional<at::Tensor>& grad_scale,
                               const ::std::optional<at::Tensor>& found_inf) {
  if (amsgrad) {
    TORCH_CHECK(at::native::check_fast_path_restrictions(
                    {self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
                "self, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must "
                "have same dtype, device, and layout");
  } else {
    TORCH_CHECK(at::native::check_fast_path_restrictions(
                    {self, grads, exp_avgs, exp_avg_sqs}),
                "self, grads, exp_avgs, and exp_avg_sqs must have same dtype, "
                "device, and layout");
  }

  return;
}

void _fused_adamw__shape_infer(at::TensorList self, at::TensorList grads,
                               at::TensorList exp_avgs,
                               at::TensorList exp_avg_sqs,
                               at::TensorList max_exp_avg_sqs,
                               at::TensorList state_steps, const at::Tensor& lr,
                               double beta1, double beta2, double weight_decay,
                               double eps, bool amsgrad, bool maximize,
                               const ::std::optional<at::Tensor>& grad_scale,
                               const ::std::optional<at::Tensor>& found_inf) {
  // Manually check devices since we specify no device check in
  // native_functions.yaml
  at::Device param_device = self[0].device();
  if (grad_scale.has_value()) {
    TORCH_CHECK(grad_scale->device() == param_device,
                "grad_scale must be on the same GCU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(found_inf->device() == param_device,
                "found_inf must be on the same GCU device as the params");
  }
  TORCH_CHECK(lr.device() == param_device,
              "lr must be on the same GCU device as the params");

  _fused_adamw__shape_infer(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                            state_steps, lr.item<double>(), beta1, beta2,
                            weight_decay, eps, amsgrad, maximize, grad_scale,
                            found_inf);

  return;
}

}  // namespace aotops
}  // namespace torch_gcu
