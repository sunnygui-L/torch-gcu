/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/core/ScalarType.h>

#include "ATen/ops/empty_like.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {
at::Tensor& cholesky_out_shape_infer(const at::Tensor& self, bool upper,
                                     at::Tensor& out) {
  TORCH_WARN_ONCE(
      "torch.cholesky is deprecated in favor of torch.linalg.cholesky and "
      "will "
      "be ",
      "removed in a future PyTorch release.\n", "L = torch.cholesky(A)\n",
      "should be replaced with\n", "L = torch.linalg.cholesky(A)\n",
      "and\n"
      "U = torch.cholesky(A, upper=True)\n",
      "should be replaced with\n",
      "U = torch.linalg.cholesky(A).mH\n"
      "This transform will produce equivalent results for all valid "
      "(symmetric "
      "positive definite) inputs.");
  if (self.numel() == 0) {
    auto ret = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    aotops::resize_output(out, ret.sizes());
    return out;
  }
  at::native::squareCheckInputs(self, "cholesky");
  auto raw_cholesky_output = at::native::cloneBatchedColumnMajor(self);
  aotops::resize_output(out, raw_cholesky_output.sizes());
  return out;
}

at::Tensor cholesky_shape_infer(const at::Tensor& self, bool upper) {
  at::Tensor result = at::empty_like(self);
  return cholesky_out_shape_infer(self, upper, result);
}

}  // namespace aotops

}  // namespace torch_gcu
