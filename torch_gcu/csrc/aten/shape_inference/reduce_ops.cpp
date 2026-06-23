/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */

#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/ReduceOpsUtils.h>

#include "aten/shape_inference/binary_ops.h"
#include "aten/shape_inference/shape_infer_func.h"
#include "aten/shape_inference/unary_ops.h"

namespace torch_gcu {

namespace aotops {

static at::TensorOptions options_to_value_type(at::TensorOptions opts) {
  auto scalar_type = at::typeMetaToScalarType(opts.dtype());
  return opts.dtype(c10::toRealValueType(scalar_type));
}

static at::Tensor& std_var_out_shape_infer(
    const char* fname, at::Tensor& result, const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction_opt, bool keepdim,
    bool take_sqrt) {
  TORCH_CHECK(self.layout() == at::Layout::Strided,
              "std and var only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) ||
                  at::isComplexType(self.scalar_type()),
              "std and var only support floating point and complex dtypes");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate variance of real and imaginary components
    // separately then add to get overall variance.
    at::ScalarType dtype =
        c10::toRealValueType(at::native::get_dtype_from_result(result, {}));
    at::Tensor real_in = real_shape_infer(self);
    at::Tensor real_out = at::empty({0}, self.options().dtype(dtype));
    std_var_out_shape_infer(fname, real_out, real_in, dim, correction_opt,
                            keepdim,
                            /*take_sqrt=*/false);

    at::Tensor imag_in = imag_shape_infer(self);
    at::Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    std_var_out_shape_infer(fname, imag_out, imag_in, dim, correction_opt,
                            keepdim,
                            /*take_sqrt=*/false);

    structured_add_Tensor_gcu_out add_out_op(result);
    add_out_op.meta(real_out, imag_out, 1.0);
    if (take_sqrt) {
      structured_sqrt_gcu_out sqrt_out_op(result);
      sqrt_out_op.meta(result);
    }
    return result;
  }

  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  at::ScalarType dtype = at::native::get_dtype_from_result(result, {});
  auto iter =
      at::native::make_reduction(fname, result, self, dim, keepdim, dtype);
  TORCH_CHECK(at::canCast(self.scalar_type(), result.scalar_type()),
              "result type ", self.scalar_type(),
              " can't be cast to the "
              "desired output type ",
              result.scalar_type());

  // if (iter.numel() == 0) {
  //   // Trivial reduction
  //   // result.fill_(std::numeric_limits<double>::quiet_NaN());
  //   return result;
  // // } else if (
  //   //   result.numel() == 1 && iter.device_type() == kCPU &&
  //   //   iter.common_dtype() != kBFloat16 && iter.common_dtype() != kHalf) {
  //   // // NOTE: CPU performance significantly regressed when attempting to
  //   port to
  //   // // ATen,
  //   // //   so all-reduce has a custom implementation.
  //   // //   See https://github.com/pytorch/pytorch/pull/43858.
  //   // result.fill_(std_var_all_cpu(self, correction, take_sqrt));
  // } else {
  //   std_var_stub(iter.device_type(), iter, correction, take_sqrt);
  // }
  return result;
}

static std::tuple<at::Tensor&, at::Tensor&> std_var_mean_out_shape_infer(
    const char* fname, at::Tensor& result1, at::Tensor& result2,
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction_opt, bool keepdim,
    bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(self.layout() == at::Layout::Strided, fname,
              " only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) ||
                  at::isComplexType(self.scalar_type()),
              fname, " only support floating point and complex dtypes");
  TORCH_CHECK(
      result1.scalar_type() == c10::toRealValueType(result2.scalar_type()),
      fname,
      " expected result1 to be real and match the precision of result2. Got ",
      result1.scalar_type(), " and ", result2.scalar_type(), ".");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate for real and imaginary components separately then
    // combine as: variance = var_real + var_imag mean = mean_real + j *
    // mean_imag
    at::ScalarType dtype =
        c10::toRealValueType(at::native::get_dtype_from_result(result1, {}));
    at::Tensor real_in = imag_shape_infer(self);
    at::Tensor real_out_var = aotops::empty({0}, self.options().dtype(dtype));
    at::Tensor real_out_mean = aotops::empty({0}, self.options().dtype(dtype));
    std_var_mean_out_shape_infer(fname, real_out_var, real_out_mean, real_in,
                                 dim, correction_opt, keepdim,
                                 /*take_sqrt=*/false);

    at::Tensor imag_in = imag_shape_infer(self);
    at::Tensor imag_out_var = aotops::empty({0}, self.options().dtype(dtype));
    at::Tensor imag_out_mean = aotops::empty({0}, self.options().dtype(dtype));
    std_var_mean_out_shape_infer(fname, imag_out_var, imag_out_mean, imag_in,
                                 dim, correction_opt, keepdim,
                                 /*take_sqrt=*/false);

    structured_add_Tensor_gcu_out add_out_op1(result1);
    add_out_op1.meta(real_out_var, imag_out_var, 1.0);
    if (take_sqrt) {
      structured_sqrt_gcu_out sqrt_out_op(result1);
      sqrt_out_op.meta(result1);
    }
    // TODO: enable complex shape infer when support this op
    // at::complex_out(result2, real_out_mean, imag_out_mean);
    TORCH_CHECK(false, "Complex output is not supported yet");
    return std::tuple<at::Tensor&, at::Tensor&>(result1, result2);
  }
  return std::tuple<at::Tensor&, at::Tensor&>(result1, result2);
}

static at::Tensor& logsumexp_out_impl_shape_infer(at::Tensor& result,
                                                  const at::Tensor& self,
                                                  at::IntArrayRef dims,
                                                  bool keepdim) {
  // can't take max of empty tensor
  if (self.numel() != 0) {
    auto maxes = amax_shape_infer(self, dims, true);
    auto maxes_squeezed = (keepdim ? maxes : at::squeeze(maxes, dims));
    auto temp = sub_shape_infer(self, maxes);
    sum_out_shape_infer(exp__shape_infer(temp), dims, keepdim, c10::nullopt,
                        result);
    add__shape_infer(log__shape_infer(result), maxes_squeezed);
  } else {
    sum_out_shape_infer(exp_shape_infer(self), dims, keepdim, c10::nullopt,
                        result);
    log__shape_infer(result);
  }
  return result;
}

at::Tensor max_shape_infer(const at::Tensor& self) {
  at::Tensor result = empty({}, self.options());
  return result;
}

at::Tensor min_shape_infer(const at::Tensor& self) {
  at::Tensor result = empty({}, self.options());
  return result;
}

at::Tensor prod_shape_infer(const at::Tensor& self,
                            c10::optional<at::ScalarType> opt_dtype) {
  auto dtype = at::native::get_dtype_from_self(self, opt_dtype, true);
  auto shape = at::meta::get_reduction_shape(self, {}, false);
  at::Tensor result = empty(shape, self.options().dtype(dtype));
  return result;
}

at::Tensor var_shape_infer(const at::Tensor& self, at::OptionalIntArrayRef dim,
                           const c10::optional<at::Scalar>& correction,
                           bool keepdim) {
  at::Tensor result = empty({0}, options_to_value_type(self.options()));
  return std_var_out_shape_infer("var", result, self, dim, correction, keepdim,
                                 false);
}

at::Tensor& var_out_shape_infer(const at::Tensor& self,
                                at::OptionalIntArrayRef dim,
                                const c10::optional<at::Scalar>& correction,
                                bool keepdim, at::Tensor& result) {
  return std_var_out_shape_infer("var", result, self, dim, correction, keepdim,
                                 false);
}

at::Tensor logsumexp_shape_infer(const at::Tensor& self, at::IntArrayRef dim,
                                 bool keepdim) {
  at::TensorOptions result_options;
  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    // even for integral inputs, result is floating dtype
    auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
    result_options = self.options().dtype(default_dtype);
  } else {
    result_options = self.options();
  }
  auto result = aotops::empty({0}, result_options);
  return logsumexp_out_shape_infer(self, dim, keepdim, result);
}

at::Tensor& logsumexp_out_shape_infer(const at::Tensor& self,
                                      at::IntArrayRef dim, bool keepdim,
                                      at::Tensor& out) {
  TORCH_CHECK(at::isFloatingType(out.scalar_type()),
              "logsumexp(): Expected floating point type for result tensor, "
              "but got: ",
              out.scalar_type());
  {
    at::NoNamesGuard guard;
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
      // for integral inputs, promote input to default floating type.
      auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
      logsumexp_out_impl_shape_infer(
          out, at::empty_like(self, self.options().dtype(default_dtype)), dim,
          keepdim);
    } else {
      logsumexp_out_impl_shape_infer(out, self, dim, keepdim);
    }
  }
  at::namedinference::propagate_names_for_reduction(out, self, dim, keepdim);
  return out;
}

at::Tensor std_shape_infer(const at::Tensor& self, at::OptionalIntArrayRef dim,
                           const c10::optional<at::Scalar>& correction,
                           bool keepdim) {
  auto result = aotops::empty({0}, options_to_value_type(self.options()));
  return std_var_out_shape_infer("std", result, self, dim, correction, keepdim,
                                 true);
}

at::Tensor& std_out_shape_infer(const at::Tensor& self,
                                at::OptionalIntArrayRef dim,
                                const c10::optional<at::Scalar>& correction,
                                bool keepdim, at::Tensor& out) {
  return std_var_out_shape_infer("std", out, self, dim, correction, keepdim,
                                 false);
}

::std::tuple<at::Tensor, at::Tensor> std_mean_shape_infer(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction, bool keepdim) {
  at::Tensor result1 =
      aotops::empty({0}, options_to_value_type(self.options()));
  at::Tensor result2 = aotops::empty({0}, self.options());
  return std_var_mean_out_shape_infer("std_mean", result1, result2, self, dim,
                                      correction, keepdim, true);
}

::std::tuple<at::Tensor, at::Tensor> var_mean_shape_infer(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction, bool keepdim) {
  at::Tensor result1 =
      aotops::empty({0}, options_to_value_type(self.options()));
  at::Tensor result2 = aotops::empty({0}, self.options());
  return std_var_mean_out_shape_infer("var_mean", result1, result2, self, dim,
                                      correction, keepdim, false);
}

at::Tensor& nansum_out_shape_infer(const at::Tensor& self,
                                   at::OptionalIntArrayRef dim, bool keepdim,
                                   c10::optional<at::ScalarType> opt_dtype,
                                   at::Tensor& result) {
  // For integral types, use existing sum as
  // integral types don't have `Nan`.
  if (c10::isIntegralType(self.scalar_type(), true)) {
    return sum_out_shape_infer(self, dim, keepdim, opt_dtype, result);
  }

  at::ScalarType dtype = at::native::get_dtype_from_result(result, opt_dtype);
  auto iter =
      at::native::make_reduction("nansum", result, self, dim, keepdim, dtype);
  iter.cast_outputs();
  return result;
}

at::Tensor nansum_shape_infer(const at::Tensor& self,
                              at::OptionalIntArrayRef dim, bool keepdim,
                              c10::optional<at::ScalarType> opt_dtype) {
  at::ScalarType dtype = at::native::get_dtype_from_self(self, opt_dtype, true);
  at::Tensor result =
      at::native::create_reduction_result(self, dim, keepdim, dtype);
  return nansum_out_shape_infer(self, dim, keepdim, dtype, result);
}


}  // namespace aotops

}  // namespace torch_gcu
