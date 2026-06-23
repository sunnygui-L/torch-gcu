/*
 * Copyright 2022-2025 Enflame. All Rights Reserved.
 */

#include <ATen/core/Tensor.h>

namespace torch_gcu {
namespace aotops {

void _fused_adamw__shape_infer(at::TensorList self, at::TensorList grads,
                               at::TensorList exp_avgs,
                               at::TensorList exp_avg_sqs,
                               at::TensorList max_exp_avg_sqs,
                               at::TensorList state_steps, const at::Tensor& lr,
                               double beta1, double beta2, double weight_decay,
                               double eps, bool amsgrad, bool maximize,
                               const ::std::optional<at::Tensor>& grad_scale,
                               const ::std::optional<at::Tensor>& found_inf);

}  // namespace aotops
}  // namespace torch_gcu
