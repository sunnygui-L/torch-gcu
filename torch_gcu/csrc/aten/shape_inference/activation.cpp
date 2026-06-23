/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/clamp_min_meta.h>

#include "ATen/core/op_registration/adaption.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

at::Tensor relu_shape_infer(const at::Tensor &self) {
  TORCH_CHECK(self.scalar_type() != at::kBool,
              "Boolean inputs not supported for relu");
  structured_clamp_min_gcu_functional op;
  op.meta(self, 0);
  return std::move(op.outputs_[0]);
}

at::Tensor &relu__shape_infer(at::Tensor &self) {
  TORCH_CHECK(self.scalar_type() != at::kBool,
              "Boolean inputs not supported for relu");
  structured_clamp_min_gcu_inplace op(self);
  op.meta(self, 0);
  return self;
}

at::Tensor _prelu_kernel_shape_infer(const at::Tensor &self,
                                     const at::Tensor &weight) {
  auto result = at::empty_like(self);
  return result;
}

at::Tensor hardtanh_shape_infer(const at::Tensor &self,
                                const at::Scalar &min_val,
                                const at::Scalar &max_val) {
  at::Tensor result = at::empty_like(self);
  return hardtanh_out_shape_infer(self, min_val, max_val, result);
}

at::Tensor &hardtanh_out_shape_infer(const at::Tensor &self,
                                     const at::Scalar &min_val,
                                     const at::Scalar &max_val,
                                     at::Tensor &out) {
  TORCH_CHECK(self.scalar_type() != at::kBool,
              "Bool inputs not supported for hardtanh");
  // preserve legacy behavior of boundaries not causing type promotion
  at::Scalar min_, max_;
  if (at::isIntegralType(self.scalar_type(), /*include_bool*/ false)) {
    int64_t minval = min_val.toLong();
    int64_t maxval = max_val.toLong();
    TORCH_CHECK(self.dtype() != at::kByte || (minval >= 0 && maxval >= 0),
                "cannot do hardtanh on an unsigned type with negative limits");
    min_ = minval;
    max_ = maxval;
  } else {
    min_ = min_val;
    max_ = max_val;
  }
  return clamp_out_shape_infer(self, min_, max_, out);
}

at::Tensor &hardswish_out_shape_infer(const at::Tensor &self, at::Tensor &out) {
  auto iter = at::TensorIterator::unary_op(out, self);
  return out;
}

at::Tensor hardswish_shape_infer(const at::Tensor &self) {
  at::Tensor result;
  auto iter = at::TensorIterator::unary_op(result, self);
  return iter.output();
}

at::Tensor &hardswish__shape_infer(at::Tensor &self) {
  auto iter = at::TensorIterator::unary_op(self, self);
  return self;
}

}  // namespace aotops

}  // namespace torch_gcu