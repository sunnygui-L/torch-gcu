/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/shape_infer_func.h"

#include <ATen/DeviceGuard.h>
#include <ATen/core/op_registration/adaption.h>
#include "ATen/core/TensorBody.h"
#include "aten/aot_ops/gcu_ops.h"

namespace torch_gcu {
namespace aotops {
at::Tensor channel_shuffle_shape_infer(const at::Tensor& self, int64_t groups) {
  TORCH_CHECK(
      self.dim() > 2,
      "channel_shuffle expects input with > 2 dims, but got input with sizes ",
      self.sizes());
  int64_t c = self.size(1);
  TORCH_CHECK(groups > 0,
              "Number of groups to divide channels in must be positive.",
              " Value of groups:", groups);
  TORCH_CHECK((c % groups) == 0,
              "Number of channels must be divisible by groups. Got ", c,
              " channels and ", groups, " groups.");

  auto memory_format = self.suggest_memory_format();
  auto output = aotops::empty({0}, self.options());
  aotops::resize_(output, self.sizes(), memory_format);
  return output;
}
}  // namespace aotops
}  // namespace torch_gcu
