/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <c10/core/MemoryFormat.h>
#include <c10/util/Exception.h>
#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

static inline at::Tensor roll_common_shape_infer(const at::Tensor& self,
                                                 at::IntArrayRef shifts,
                                                 at::IntArrayRef dims) {
  TORCH_CHECK(!shifts.empty(), "`shifts` required");
  if (dims.empty() && shifts.size() == 1) {
    auto flattened = contiguous_shape_infer(self).view(self.numel());
    return roll_shape_infer(flattened, shifts[0], 0).view(self.sizes());
  }
  TORCH_CHECK(shifts.size() == dims.size(),
              "shifts and dimensions must align. shifts: ", shifts.size(),
              ", dims:", dims.size());
  AT_ASSERT(dims.size() > 1);
  auto tail_shifts = shifts.slice(1);
  auto tail_dims = dims.slice(1);
  auto first_dim_rolled = roll_shape_infer(self, shifts[0], dims[0]);
  return roll_shape_infer(first_dim_rolled, tail_shifts, tail_dims);
}

at::Tensor roll_shape_infer(const at::Tensor& self, at::IntArrayRef shifts,
                            at::IntArrayRef dims) {
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common_shape_infer(self, shifts, dims);
  }

  auto in_tensor = self;
  if (!self.is_contiguous()) {
    in_tensor = contiguous_shape_infer(self);
  }
  auto out_tensor = at::empty_like(in_tensor, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (out_tensor.numel() == 0) {
    return out_tensor;
  }

  return out_tensor;
}

}  // namespace aotops

}  // namespace torch_gcu