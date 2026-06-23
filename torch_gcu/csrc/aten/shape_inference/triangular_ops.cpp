/*
 * Copyright 2024-2025 Enflame. All Rights Reserved.
 */

#include <ATen/native/ReduceOpsUtils.h>
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

at::Tensor trace_shape_infer(const at::Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "expected a matrix");
  auto dtype = at::native::get_dtype_from_self(self, c10::nullopt, true);
  return aotops::empty({}, self.options().dtype(dtype));
}

}  // namespace aotops

}  // namespace torch_gcu