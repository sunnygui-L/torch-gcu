/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/unary_ops.h"

#include <ATen/TensorIterator.h>
#include <ATen/core/op_registration/adaption.h>
#include <c10/core/ScalarType.h>

#include "aten/aot_ops/gcu_ops.h"
#include "aten/shape_inference/gcu_structured_shape_infer.h"
#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

static inline at::Tensor& unary_op_impl_out_shape_infer(
    at::Tensor& result, const at::Tensor& self) {
  auto iter = at::TensorIterator::unary_op(result, self);
  return result;
}

static inline at::Tensor& unary_op_impl_float_out_shape_infer(
    at::Tensor& result, const at::Tensor& self) {
  auto iter = at::TensorIterator::unary_float_op(result, self);
  iter.cast_outputs();
  return result;
}

// An alternate version of unary_op_impl_out_shape_infer that follows the same
// pattern for non-complex inputs, but returns a floating point tensor for
// complex inputs by default. Note: This is done by running the operation as
// usual and then copying the operation's result to the expected result type.
static inline at::Tensor& unary_op_impl_with_complex_to_float_out_shape_infer(
    at::Tensor& result, const at::Tensor& self,
    bool promotes_integer_to_float) {
  if (self.is_complex() && !result.is_complex()) {
    // Checks if the corresponding float type can be cast to the desired dtype
    const auto float_type = c10::toRealValueType(self.scalar_type());
    TORCH_CHECK(c10::canCast(float_type, result.scalar_type()), "result type ",
                float_type, " can't be cast to the desired output type ",
                result.scalar_type());

    // Runs the function complex->complex, as TensorIterator expects
    at::Tensor complex_result = at::empty({0}, self.options());
    auto iter = at::TensorIterator::unary_op(complex_result, self);

    // Copies the complex result to the actual result and returns it
    aotops::resize_output(result, complex_result.sizes());
    return result;
  }

  if (promotes_integer_to_float) {
    return unary_op_impl_float_out_shape_infer(result, self);
  }

  return unary_op_impl_out_shape_infer(result, self);
}

at::Tensor& abs_out_shape_infer(const at::Tensor& self, at::Tensor& result) {
  return unary_op_impl_with_complex_to_float_out_shape_infer(
      result, self, /*promotes_integer_to_float=*/false);
}

at::Tensor real_shape_infer(const at::Tensor& self) {
  if (self.is_complex()) {
    at::Tensor real_tensor;
    if (self.is_conj()) {
      real_tensor = aotops::view_as_real(self._conj());
    } else {
      real_tensor = aotops::view_as_real(self);
    }
    return at::select(real_tensor, real_tensor.dim() - 1, 0);
  } else {
    return self;
  }
}

at::Tensor imag_shape_infer(const at::Tensor& self) {
  if (self.is_complex()) {
    at::Tensor real_tensor;
    if (self.is_conj()) {
      real_tensor = aotops::view_as_real(self._conj());
      // preemptively set the negative flag for the final imag tensor
      real_tensor = real_tensor._neg_view();
    } else {
      real_tensor = aotops::view_as_real(self);
    }
    return at::select(real_tensor, real_tensor.dim() - 1, 1);
  } else {
    TORCH_CHECK(false,
                "imag is not implemented for tensors with non-complex dtypes.");
  }
}

at::Tensor& nan_to_num_out_shape_infer(const at::Tensor& self,
                                       c10::optional<double> nan,
                                       c10::optional<double> pos_inf,
                                       c10::optional<double> neg_inf,
                                       at::Tensor& result) {
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "nan_to_num: dtype of out: ", result.scalar_type(),
              " should be same as input: ", self.scalar_type());

  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    aotops::resize_output(result, self.sizes());
  } else {
    at::TensorIterator::unary_op(result, self);
  }

  return result;
}

::std::tuple<at::Tensor&, at::Tensor&> frexp_out_shape_infer(
    const at::Tensor& self, at::Tensor& mantissa, at::Tensor& exponent) {
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device;  // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, mantissa,
                                            __FUNCTION__, "mantissa");
  c10::impl::check_and_update_common_device(common_device, exponent,
                                            __FUNCTION__, "exponent");
  c10::impl::check_and_update_common_device(common_device, self, __FUNCTION__,
                                            "self");
  const at::OptionalDeviceGuard device_guard(device_of(self));
  // torch.frexp is implemented for floating-point dtypes for now,
  // should add support for integral dtypes in the future.
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
              "torch.frexp() only supports floating-point dtypes");

  TORCH_CHECK(mantissa.dtype() == self.dtype(),
              "torch.frexp() expects mantissa to have dtype ", self.dtype(),
              " but got ", mantissa.dtype());
  TORCH_CHECK(exponent.dtype() == at::kInt,
              "torch.frexp() expects exponent to have int dtype "
              "but got ",
              exponent.dtype());

  auto iter = at::TensorIteratorConfig()
                  .add_output(mantissa)
                  .add_output(exponent)
                  .add_input(self)
                  .check_all_same_dtype(false)
                  .set_check_mem_overlap(true)
                  .build();
  return std::tuple<at::Tensor&, at::Tensor&>(mantissa, exponent);
}

at::Tensor& logical_not_out_shape_infer(const at::Tensor& self,
                                        at::Tensor& out) {
  at::TensorIterator iter = at::TensorIteratorConfig()
                                .check_all_same_dtype(false)
                                .add_output(out)
                                .add_input(self)
                                .build();
  iter.cast_outputs();
  return out;
}

at::Tensor& logit_out_shape_infer(const at::Tensor& self,
                                  c10::optional<double> eps,
                                  at::Tensor& result) {
  auto iter = at::TensorIterator::unary_float_op(result, self);
  iter.cast_outputs();
  return result;
}

at::Tensor logit_shape_infer(const at::Tensor& self,
                             c10::optional<double> eps) {
  at::Tensor result;
  auto iter = at::TensorIterator::unary_float_op(result, self);
  return iter.output();
}

at::Tensor& logit__shape_infer(at::Tensor& self, c10::optional<double> eps) {
  auto iter = at::TensorIterator::unary_float_op(self, self);
  iter.cast_outputs();
  return self;
}

at::Tensor& angle_out_shape_infer(const at::Tensor& self, at::Tensor& result) {
  if (self.is_complex() && !result.is_complex()) {
    // Checks if the corresponding float type can be cast to the desired dtype
    const auto float_type = c10::toRealValueType(self.scalar_type());
    TORCH_CHECK(canCast(float_type, result.scalar_type()), "result type ",
                float_type, " can't be cast to the desired output type ",
                result.scalar_type());

    // Runs the function complex->complex, as TensorIterator expects
    at::Tensor complex_result = aotops::empty({0}, self.options());
    auto iter = at::TensorIterator::unary_op(complex_result, self);

    aotops::resize_output(result, complex_result.sizes());
    return result;
  }

  auto iter = at::TensorIterator::unary_float_op(result, self);
  iter.cast_outputs();
  return result;
}

at::Tensor angle_shape_infer(const at::Tensor& self) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    at::Tensor result = aotops::empty({0}, self.options().dtype(float_type));
    return angle_out_shape_infer(self, result);
  }
  at::Tensor result;
  auto iter = at::TensorIterator::unary_float_op(result, self);
  return iter.output();
}

}  // namespace aotops

}  // namespace torch_gcu
