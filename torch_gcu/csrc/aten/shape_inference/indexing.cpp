/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/MemoryOverlap.h>

#include "aten/shape_inference/shape_infer_func.h"
#include "gcu/gcu_functions.h"

namespace torch_gcu {

namespace aotops {

namespace {

#define MAX_TENSORINFO_DIMS 25

}  // namespace

at::Tensor& index_select_out_shape_infer(const at::Tensor& self, int64_t dim,
                                         const at::Tensor& index,
                                         at::Tensor& out) {
  static constexpr c10::string_view DIM_WARNING =
      "Tensor too large or too many (> 25) dimensions";
  TORCH_CHECK(check_device({out, self, index}),
              "Input, output and indices must be on the current device");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

  dim = at::maybe_wrap_dim(dim, self);
  TORCH_CHECK(self.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);
  if (self.is_quantized()) {
    // TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine,
    //             "Only per_tensor quantized quantized tensors are supported by
    //             " "index_select.")
    TORCH_CHECK(false, "gcu index_select_out not support quantized yet.");
  }

  auto numIndices = index.numel();
  int selfDims = self.dim() == 0 ? 1 : self.dim();

  TORCH_CHECK(index.dim() <= 1,
              "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(!(self.dim() == 0 && numIndices != 1),
              "index_select(): Index to scalar can have only 1 value, got ",
              numIndices, " value(s)");
  TORCH_CHECK(dim < selfDims, "Indexing dim is out of bounds");

  std::vector<int64_t> newSize = self.sizes().vec();

  if (self.dim() > 0) {
    newSize[dim] = numIndices;
  }

  if (self.is_quantized()) {
    out = at::empty_quantized(newSize, out);
  } else {
    aotops::resize_output(out, newSize);
  }
  return out;
}

at::Tensor index_select_shape_infer(const at::Tensor& self, int64_t dim,
                                    const at::Tensor& index) {
  at::Tensor out = empty({0}, self.options());
  return index_select_out_shape_infer(self, dim, index, out);
}

}  // namespace aotops

}  // namespace torch_gcu
