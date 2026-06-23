/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */
#include "aten/shape_inference/index_kernel.h"

#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

static at::Tensor& masked_select_out_gcu_impl(at::Tensor& result,
                                              const at::Tensor& self,
                                              const at::Tensor& mask) {
  at::NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool,
              "masked_select: expected BoolTensor for mask");
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "masked_select(): self and result must have the same scalar type");

  auto mask_temp = (mask.dim() == 0)
                       ? c10::MaybeOwned<at::Tensor>::owned(mask.unsqueeze(0))
                       : c10::MaybeOwned<at::Tensor>::borrowed(mask);
  auto self_temp = (self.dim() == 0)
                       ? c10::MaybeOwned<at::Tensor>::owned(self.unsqueeze(0))
                       : c10::MaybeOwned<at::Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto mask_self_expanded = at::expand_outplace(*mask_temp, *self_temp);
  aotops::index_out_shape_infer(
      *std::get<1>(mask_self_expanded),
      c10::List<c10::optional<at::Tensor>>(
          {*std::move(std::get<0>(mask_self_expanded))}),
      result);

  return result;
}

at::Tensor masked_select_shape_infer(const at::Tensor& self,
                                     const at::Tensor& mask) {
  at::namedinference::compute_broadcast_outnames(self, mask);
  at::Tensor result = at::empty({0}, self.options());
  return masked_select_out_gcu_impl(result, self, mask);
}

at::Tensor& masked_select_out_shape_infer(const at::Tensor& self,
                                          const at::Tensor& mask,
                                          at::Tensor& result) {
  at::namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_gcu_impl(result, self, mask);
}

at::Tensor& masked_scatter__shape_infer(at::Tensor& self,
                                        const at::Tensor& mask,
                                        const at::Tensor& source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter_: expected self and source to have same dtypes but got ",
      self.scalar_type(), " and ", source.scalar_type());
  TORCH_CHECK(mask.dtype() == at::ScalarType::Bool,
              "masked_scatter_ only supports boolean masks, "
              "but got mask with dtype ",
              mask.dtype());
  return self;
}
}  // namespace aotops

}  // namespace torch_gcu
