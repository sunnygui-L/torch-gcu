/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/binary_ops.h"

#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

at::Tensor floor_divide_shape_infer(const at::Tensor& self,
                                    const at::Tensor& other) {
  at::Tensor result;
  auto iter = at::TensorIterator::binary_op(result, self, other);
  return iter.output();
}

at::Tensor& floor_divide_out_shape_infer(const at::Tensor& self,
                                         const at::Tensor& other,
                                         at::Tensor& out) {
  auto iter = at::TensorIterator::binary_op(out, self, other);
  if (!out.defined()) {
    out = iter.output();
  }
  return out;
}

at::Tensor& floor_divide__shape_infer(at::Tensor& self,
                                      const at::Tensor& other) {
  return floor_divide_out_shape_infer(self, other, self);
}

at::Tensor& logical_and_out_shape_infer(const at::Tensor& self,
                                        const at::Tensor& other,
                                        at::Tensor& out) {
  at::TensorIterator iter;
  iter.build_comparison_op(out, self, other);
  iter.cast_outputs();
  return out;
}

at::Tensor& logical_or_out_shape_infer(const at::Tensor& self,
                                       const at::Tensor& other,
                                       at::Tensor& out) {
  at::TensorIterator iter;
  iter.build_comparison_op(out, self, other);
  iter.cast_outputs();
  return out;
}

at::Tensor& logical_xor_out_shape_infer(const at::Tensor& self,
                                        const at::Tensor& other,
                                        at::Tensor& out) {
  at::TensorIterator iter;
  iter.build_comparison_op(out, self, other);
  iter.cast_outputs();
  return out;
}

at::Tensor& _add_relu_out_shape_infer(const at::Tensor& self,
                                      const at::Tensor& other,
                                      const at::Scalar& alpha,
                                      at::Tensor& out) {
  auto iter = at::TensorIterator::binary_op(out, self, other);

  if (self.dtype() != at::kInt && self.dtype() != at::kLong &&
      self.dtype() != at::kShort && self.dtype() != at::kChar &&
      self.dtype() != at::kFloat && self.dtype() != at::kDouble) {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported datatype for add_relu:", self.dtype().name());
  }
  out = iter.output();
  return out;
}

at::Tensor _add_relu_shape_infer(const at::Tensor& self,
                                 const at::Scalar& other,
                                 const at::Scalar& alpha) {
  at::Tensor out;
  return _add_relu_out_shape_infer(
      self, at::native::wrapped_scalar_tensor(other), alpha, out);
}

at::Tensor& _add_relu__shape_infer(at::Tensor& self, const at::Scalar& other,
                                   const at::Scalar& alpha) {
  return _add_relu_out_shape_infer(
      self, at::native::wrapped_scalar_tensor(other), alpha, self);
}

at::Tensor __rshift___shape_infer(const at::Tensor& self,
                                  const at::Tensor& other) {
  at::Tensor result;
  auto iter = at::TensorIterator::binary_op(result, self, other);
  return iter.output();
}

at::Tensor __rshift___shape_infer(const at::Tensor& self,
                                  const at::Scalar& other) {
  at::Tensor result;
  auto wrapper = at::native::wrapped_scalar_tensor(other);
  auto iter = at::TensorIterator::binary_op(result, self, wrapper);
  return iter.output();
}

at::Tensor __lshift___shape_infer(const at::Tensor& self,
                                  const at::Tensor& other) {
  at::Tensor result;
  auto iter = at::TensorIterator::binary_op(result, self, other);
  return iter.output();
}

at::Tensor __lshift___shape_infer(const at::Tensor& self,
                                  const at::Scalar& other) {
  at::Tensor result;
  auto wrapper = at::native::wrapped_scalar_tensor(other);
  auto iter = at::TensorIterator::binary_op(result, self, wrapper);
  return iter.output();
}

}  // namespace aotops

}  // namespace torch_gcu
