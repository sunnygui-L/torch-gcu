/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/tensor_factories.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

namespace {
static void complex_check_floating(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(
      (a.scalar_type() == at::kFloat || a.scalar_type() == at::kDouble ||
       a.scalar_type() == at::kHalf) &&
          (b.scalar_type() == at::kFloat || b.scalar_type() == at::kDouble ||
           b.scalar_type() == at::kHalf),
      "Expected both inputs to be Half, Float or Double tensors but got ",
      a.scalar_type(), " and ", b.scalar_type());
}

static void complex_check_dtype(const at::Tensor& result, const at::Tensor& a,
                                const at::Tensor& b) {
  complex_check_floating(a, b);
  TORCH_CHECK(a.scalar_type() == b.scalar_type(),
              "Expected object of scalar type ", a.scalar_type(),
              " but got scalar type ", b.scalar_type(), " for second argument");
  TORCH_CHECK(result.scalar_type() == c10::toComplexType(a.scalar_type()),
              "Expected object of scalar type ",
              c10::toComplexType(a.scalar_type()), " but got scalar type ",
              result.scalar_type(), " for argument 'out'");
}
}  // namespace

at::Tensor& fill__shape_infer(at::Tensor& self, const at::Scalar& value) {
  return self;
}

at::Tensor& fill__shape_infer(at::Tensor& self, const at::Tensor& value) {
  return self;
}

at::Tensor& polar_out_shape_infer(const at::Tensor& abs,
                                  const at::Tensor& angle, at::Tensor& result) {
  complex_check_floating(abs, angle);
  at::TensorIteratorConfig()
      .add_output(result)
      .add_input(abs)
      .add_input(angle)
      .check_all_same_dtype(false)
      .build();
  return result;
}

at::Tensor& randperm_out_shape_infer(int64_t n,
                                     c10::optional<at::Generator> /*generator*/,
                                     at::Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);

  at::native::check_supported_max_int_with_precision(n, result);

  aotops::resize_(result, {n}, c10::nullopt);

  return result;
}

at::Tensor& eye_out_shape_infer(int64_t n, at::Tensor& out) {
  return eye_out_shape_infer(n, n, out);
}

at::Tensor& eye_out_shape_infer(int64_t n, int64_t m, at::Tensor& out) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  aotops::resize_(out, {n, m}, c10::nullopt);
  return out;
}

}  // namespace aotops

}  // namespace torch_gcu