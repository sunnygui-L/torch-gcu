/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */

#include <ATen/native/LossMulti.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_resize.h"

#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

at::Tensor& multi_margin_loss_out_shape_infer(
    const at::Tensor& self, const at::Tensor& target, const at::Scalar& p_,
    const at::Scalar& margin, const c10::optional<at::Tensor>& weight,
    int64_t reduction, at::Tensor& out) {
  auto p = p_.toLong();
  int64_t nframe, dim;
  const auto ndims = self.dim();

  TORCH_CHECK(p == 1 || p == 2,
              "multi_margin_loss: Invalid p, expected 1 or 2 but got ", p);

  at::native::multi_margin_loss_shape_check(nframe, dim, ndims, self, target,
                                            weight);

  if (reduction == at::Reduction::None && target.dim() > 0) {
    aotops::resize_output(out, {nframe});
  } else {
    aotops::resize_output(out, {});
  }

  return out;
}

at::Tensor multi_margin_loss_shape_infer(
    const at::Tensor& self, const at::Tensor& target, const at::Scalar& p,
    const at::Scalar& margin, const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  auto out = aotops::empty({0}, self.options());
  multi_margin_loss_out_shape_infer(self, target, p, margin, weight, reduction,
                                    out);
  return out;
}

}  // namespace aotops

}  // namespace torch_gcu
