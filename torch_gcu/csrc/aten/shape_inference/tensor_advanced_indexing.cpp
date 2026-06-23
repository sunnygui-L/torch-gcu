/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/tensor_advanced_indexing.h"

#include <ATen/MemoryOverlap.h>

#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

at::Tensor& masked_fill__shape_infer(at::Tensor& self, const at::Tensor& mask,
                                     const at::Tensor& value) {
  return self;
}

at::Tensor& masked_fill__shape_infer(at::Tensor& self, const at::Tensor& mask,
                                     const at::Scalar& value) {
  return self;
}

at::Tensor& _index_put_impl__shape_infer(
    at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value, bool accumulate, bool unsafe) {
  return self;
}

at::Tensor& index_fill__shape_infer(at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Scalar& source) {
  TORCH_CHECK_INDEX(index.scalar_type() == at::ScalarType::Long,
                    "index_fill_(): Expected dtype int64 for index.");

  at::assert_no_overlap(self, index);
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_fill_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }

  if (!self.is_complex() && source.isComplex()) {
    TORCH_CHECK(false,
                "index_fill_(): Converting complex Scalar to non-complex type "
                "is not supported");
  }

  TORCH_CHECK(index.dim() <= 1, "Index has to be a vector/scalar");

  return self;
}

at::Tensor count_nonzero_shape_infer(const at::Tensor& self,
                                     at::IntArrayRef dims) {
  if (!dims.empty()) {
    auto reduce = self;
    if (self.scalar_type() != at::kBool) {
      reduce = aotops::empty(self.sizes(), self.options().dtype(at::kBool));
    }
    return sum_shape_infer(reduce, dims);
  }

  auto out = aotops::empty({}, self.options().dtype(at::kLong)).fill_(0);
  return out;
}

}  // namespace aotops

}  // namespace torch_gcu
