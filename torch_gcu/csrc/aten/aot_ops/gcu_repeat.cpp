/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/ops/repeat_interleave_native.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/repeat.h"

namespace torch_gcu {

namespace aotops {

namespace {

std::vector<int64_t> InferRepeatInterleaveIntOpShape(
    const std::vector<int64_t> self, int64_t repeats,
    c10::optional<int64_t> dim_opt, c10::optional<int64_t> output_size) {
  std::vector<int64_t> output = dim_opt ? self
                                        : std::vector<int64_t>({std::accumulate(
                                              self.begin(), self.end(), 0)});
  int64_t dim = c10::maybe_wrap_dim(dim_opt.value_or(0), self.size());
  TORCH_CHECK(repeats >= 0, "Repeats must be non-negative");

  // This argument doesn't really make sense for the scalar overload, but exists
  // for consistency with the tensor overload
  auto calculated_size = repeats * output[dim];
  if (output_size) {
    TORCH_CHECK(calculated_size == *output_size,
                "repeat_interleave: Invalid output_size, expected ",
                calculated_size, " but got ", *output_size);
  }
  output[dim] = calculated_size;
  return output;
}

}  // namespace

at::Tensor repeat_interleave(const at::Tensor &self, const at::Tensor &repeats,
                             c10::optional<int64_t> dim,
                             c10::optional<int64_t> output_size) {
  // Create GCU output tensor
  auto result = repeat_interleave_shape_infer(self, repeats, dim, output_size);

  bridge_topsatenRepeatInterleave_out1(result, self, repeats, dim, output_size);
  return result;
}

};  // namespace aotops

}  // namespace torch_gcu
