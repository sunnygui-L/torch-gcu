/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/core/Tensor.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

at::Tensor repeat_shape_infer(const at::Tensor &self, at::IntArrayRef repeats) {
  TORCH_CHECK(repeats.size() >= (size_t)self.dim(),
              "Number of dimensions of repeat dims can not be smaller than "
              "number of dimensions of tensor");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  at::DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(),
                     self.sizes().end());
  at::DimVector target_size(repeats.size());
  for (const auto idx : c10::irange(repeats.size())) {
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  at::Tensor result = aotops::empty(target_size, self.options());

  return result;
}

at::Tensor repeat_interleave_shape_infer(const at::Tensor &repeats,
                                         c10::optional<int64_t> output_size) {
  TORCH_CHECK(repeats.dim() == 1,
              "repeat_interleave only accept 1D vector as repeat");
  TORCH_CHECK(
      repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
      "repeats has to be Long or Int tensor");
  if (repeats.size(0) == 0) {
    return at::empty_like(repeats, c10::get_contiguous_memory_format());
  }

  int64_t total;
  if (output_size.has_value()) {
    total = output_size.value();
  } else {
    at::Tensor cumsum = cumsum_shape_infer(repeats, 0, c10::nullopt);
    aotops::cumsum_out(repeats, 0, c10::nullopt, cumsum);
    total = cumsum[-1].item<int64_t>();
    at::Tensor ge_out = ge_shape_infer(repeats, 0);
    at::Tensor all_out = all_shape_infer(ge_out);
    aotops::all_out(aotops::ge_out(repeats, 0, ge_out), all_out);
    TORCH_CHECK(all_out.item<uint8_t>(), "repeats can not be negative");
  }

  at::Tensor result = at::empty({total}, repeats.options());
  return result;
}

at::Tensor repeat_interleave_shape_infer(const at::Tensor &self,
                                         const at::Tensor &repeats,
                                         c10::optional<int64_t> dim,
                                         c10::optional<int64_t> output_size) {
  at::Tensor input = self;

  // Store conj and neg bits
  const auto conj = input.is_conj();
  if (conj) {
    input = input.conj();
  }
  const auto neg = input.is_neg();
  if (neg) {
    input = input._neg_view();
  }

  if (!dim) {
    input = input.flatten();
    dim = 0;
  }

  at::Tensor repeats_ = repeats;
  if (repeats.dim() == 0 || (repeats.dim() == 1 && repeats.sym_size(0) == 1)) {
    repeats_ =
        repeats.reshape({1}).expand_symint({input.sym_size(dim.value())});
  } else if (repeats.dim() == 1) {
    TORCH_CHECK(repeats.sym_size(0) == input.sym_size(dim.value()),
                "repeats must have the same size as input along dim, but got "
                "repeats.size(0) = ",
                repeats.sym_size(0), " and input.size(", dim.value(),
                ") = ", input.sym_size(dim.value()));
  } else {
    AT_ERROR("repeats must be 0-dim or 1-dim tensor");
  }

  auto ret = index_select_shape_infer(
      input, dim.value(), repeat_interleave_shape_infer(repeats_, output_size));
  // Restore conj and neg bits
  if (conj) {
    ret = ret.conj();
  }
  if (neg) {
    ret = ret._neg_view();
  }
  return ret;
}

}  // namespace aotops

}  // namespace torch_gcu
