/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/sorting.h"

#include <ATen/MemoryOverlap.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SortingUtils.h>

#include "aten/aot_ops/gcu_aot_ops.h"
#include "aten/shape_inference/gcu_structured_shape_infer.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

#define MAX_TENSORINFO_DIMS 25

at::Tensor argsort_shape_infer(const at::Tensor &self, bool stable, int64_t dim,
                               bool descending) {
  return std::get<1>(sort_shape_infer(self, stable, dim, descending));
}

std::tuple<at::Tensor &, at::Tensor &> median_out_shape_infer(
    const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &values,
    at::Tensor &indices) {
  // See note [Writing Nondeterministic Operations]
  // If there are duplicate elements of a median value, the procedure for
  // choosing which of the duplicates to use for the indices output is
  // nondeterministic.
  at::globalContext().alertNotDeterministic("median GCU with indices output");
  at::NoNamesGuard guard;

  dim = at::maybe_wrap_dim(dim, self.dim());

  at::checkDeviceType("median", {values, indices}, self.device().type());
  at::checkScalarType("median", {indices, "indices", 1}, at::kLong);
  at::checkSameType("median", {values, "values", 0}, {self, "self", 2});

  TORCH_CHECK(self.dim() <= MAX_TENSORINFO_DIMS,
              "median() cannot operate on more than ", MAX_TENSORINFO_DIMS,
              " dimensions");

  std::vector<int64_t> out_shape = self.sizes().vec();
  at::native::zero_numel_check_dims(self, dim, "median()");
  if (self.dim() > 0) {
    assert(dim >= 0);
    assert(dim < static_cast<int64_t>(out_shape.size()));

    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  values.resize_(out_shape);
  indices.resize_(out_shape);

  guard.reset();
  at::namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  at::namedinference::propagate_names_for_reduction(indices, self, dim,
                                                    keepdim);

  return std::forward_as_tuple(values, indices);
}

at::Tensor nanmedian_shape_infer(const at::Tensor &self) {
  return aotops::empty({}, self.options());
}

std::tuple<at::Tensor &, at::Tensor &> nanmedian_out_shape_infer(
    const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &values,
    at::Tensor &indices) {
  // See note [Writing Nondeterministic Operations]
  // If there are duplicate elements of a median value, the procedure for
  // choosing which of the duplicates to use for the indices output is
  // nondeterministic.
  at::globalContext().alertNotDeterministic("median GCU with indices output");
  at::NoNamesGuard guard;

  dim = at::maybe_wrap_dim(dim, self.dim());

  at::checkDeviceType("median", {values, indices}, self.device().type());
  at::checkScalarType("median", {indices, "indices", 1}, at::kLong);
  at::checkSameType("median", {values, "values", 0}, {self, "self", 2});

  TORCH_CHECK(self.dim() <= MAX_TENSORINFO_DIMS,
              "median() cannot operate on more than ", MAX_TENSORINFO_DIMS,
              " dimensions");

  std::vector<int64_t> out_shape = self.sizes().vec();
  at::native::zero_numel_check_dims(self, dim, "median()");
  if (self.dim() > 0) {
    assert(dim >= 0);
    assert(dim < static_cast<int64_t>(out_shape.size()));

    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  values.resize_(out_shape);
  indices.resize_(out_shape);

  guard.reset();
  at::namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  at::namedinference::propagate_names_for_reduction(indices, self, dim,
                                                    keepdim);

  return std::forward_as_tuple(values, indices);
}

std::tuple<at::Tensor &, at::Tensor &> kthvalue_out_shape_infer(
    const at::Tensor &self, int64_t k, int64_t dim_, bool keepdim,
    at::Tensor &values, at::Tensor &indices) {
  // See note [Writing Nondeterministic Operations]
  // If there are duplicate elements of the kth value, the procedure for
  // choosing which of the duplicates to use for the indices output is
  // nondeterministic.
  at::globalContext().alertNotDeterministic("kthvalue GCU");
  auto result = [&]() {
    at::NoNamesGuard guard;
    // `kthvalue_out_impl_cuda` expects contiguous in input `self`.
    // return kthvalue_out_impl_cuda(values, indices, self.contiguous(), k, dim,
    // keepdim);
    int64_t dim = at::maybe_wrap_dim(dim_, self.dim());
    int64_t slicesize = self.dim() == 0 ? 1 : self.size(dim);
    at::native::zero_numel_check_dims(self, dim, "kthvalue()");

    TORCH_CHECK(k >= 1 && k <= slicesize,
                "kthvalue(): selected number k out of range for dimension ",
                dim);

    at::assert_no_overlap(self, values);

    at::native::_reduction_with_indices_allocate_or_resize_output(
        values, indices, self, dim, keepdim);
    if (self.dim() == 0 && self.numel() == 1) {
      values.copy_(self);
      indices.zero_();
      return std::forward_as_tuple(values, indices);
    }

    TORCH_CHECK(self.dim() <= MAX_TENSORINFO_DIMS,
                "cannot operate on more than ", MAX_TENSORINFO_DIMS,
                " dimensions");

    if (!keepdim) {
      values.squeeze_(dim);
      indices.squeeze_(dim);
    }

    return std::forward_as_tuple(values, indices);
  }();
  at::namedinference::propagate_names_for_reduction(values, self, dim_,
                                                    keepdim);
  at::namedinference::propagate_names_for_reduction(indices, self, dim_,
                                                    keepdim);
  return result;
}

}  // namespace aotops

}  // namespace torch_gcu
