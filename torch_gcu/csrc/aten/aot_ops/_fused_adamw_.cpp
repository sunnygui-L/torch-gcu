/*
 * Copyright 2022-2025 Enflame. All Rights Reserved.
 */

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/shape_infer_func.h"
#include "gcu/sys_util.h"

namespace torch_gcu {

namespace aotops {

void _fused_adamw_(at::TensorList self, at::TensorList grads,
                   at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                   at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
                   const at::Tensor &lr, double beta1, double beta2,
                   double weight_decay, double eps, bool amsgrad, bool maximize,
                   const ::std::optional<at::Tensor> &grad_scale,
                   const ::std::optional<at::Tensor> &found_inf) {
  _fused_adamw__shape_infer(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                            state_steps, lr, beta1, beta2, weight_decay, eps,
                            amsgrad, maximize, grad_scale, found_inf);

  if (lr.is_cpu()) {
    bridge_topsatenFusedAdamw_multi_out(
        std::make_index_sequence<std::tuple_size<::std::tuple<
            ::std::vector<at::Tensor> &, ::std::vector<at::Tensor> &,
            ::std::vector<at::Tensor> &, ::std::vector<at::Tensor> &,
            ::std::vector<at::Tensor> &>>::value>{},
        false,
        std::forward_as_tuple(self, grads, exp_avgs, exp_avg_sqs,
                              max_exp_avg_sqs),
        self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
        lr.item<double>(), beta1, beta2, weight_decay, eps, amsgrad, maximize,
        grad_scale, found_inf);
    return;
  }

  bridge_topsatenFusedAdamw_multi_out(
      std::make_index_sequence<std::tuple_size<
          ::std::tuple<::std::vector<at::Tensor> &, ::std::vector<at::Tensor> &,
                       ::std::vector<at::Tensor> &, ::std::vector<at::Tensor> &,
                       ::std::vector<at::Tensor> &>>::value>{},
      false,
      std::forward_as_tuple(self, grads, exp_avgs, exp_avg_sqs,
                            max_exp_avg_sqs),
      self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr,
      beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale,
      found_inf);
  return;
}

}  // namespace aotops

}  // namespace torch_gcu
