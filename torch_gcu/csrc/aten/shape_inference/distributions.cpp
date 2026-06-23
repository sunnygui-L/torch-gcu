/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace {

#define CHECK_EMPTY_AND_RETURN(tensor) \
  if (tensor.numel() == 0) {           \
    return tensor;                     \
  }

}  // namespace

namespace aotops {

at::Tensor& uniform__shape_infer(at::Tensor& self, double from, double to,
                                 c10::optional<at::Generator> generator) {
  CHECK_EMPTY_AND_RETURN(self);
  return self;
}

at::Tensor& exponential__shape_infer(at::Tensor& self, double lambda,
                                     c10::optional<at::Generator> gen) {
  TORCH_CHECK(lambda > 0.0,
              "exponential_ expects lambda > 0.0, but found lambda=", lambda);
  CHECK_EMPTY_AND_RETURN(self);
  return self;
}

at::Tensor multinomial_shape_infer(const at::Tensor& self, int64_t num_samples,
                                   bool replacement,
                                   c10::optional<at::Generator> generator) {
  at::Tensor result = empty({0}, self.options().dtype(at::kLong));
  multinomial_out_shape_infer(self, num_samples, replacement,
                              std::move(generator), result);
  return result;
}

at::Tensor& multinomial_out_shape_infer(const at::Tensor& self,
                                        int64_t n_sample, bool with_replacement,
                                        c10::optional<at::Generator> generator,
                                        at::Tensor& result) {
  TORCH_CHECK(result.device() == self.device(),
              "multinomial arguments must have the same device");
  TORCH_CHECK(self.dim() > 0 && self.dim() <= 2,
              "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "multinomial only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == at::ScalarType::Long,
      "multinomial expects Long tensor out, got: ", result.scalar_type());
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(with_replacement || (n_sample <= n_categories),
              "cannot sample n_sample > prob_dist.size(-1) samples without "
              "replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(n_categories <= (1 << (__FLT_MANT_DIG__)),
              "number of categories cannot exceed 2^24");

  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }
  return result;
}

at::Tensor& normal__shape_infer(at::Tensor& self, double mean, double std,
                                c10::optional<at::Generator> gen) {
  CHECK_EMPTY_AND_RETURN(self);
  return self;
}

at::Tensor binomial_shape_infer(const at::Tensor& count, const at::Tensor& prob,
                                c10::optional<at::Generator> generator) {
  at::Tensor ret = aotops::empty(count.sizes(), count.options());
  at::TensorIterator iter = at::TensorIteratorConfig()
                                .add_output(ret)
                                .add_input(count)
                                .add_input(prob)
                                .build();
  return ret;
}

at::Tensor poisson_shape_infer(const at::Tensor& self,
                               c10::optional<at::Generator> generator) {
  return aotops::empty(self.sizes(), self.options());
}

at::Tensor& bernoulli_out_shape_infer(const at::Tensor& self,
                                      c10::optional<at::Generator> generator,
                                      at::Tensor& out) {
  aotops::resize_output(out, self.sizes());
  return out;
}

at::Tensor& bernoulli__shape_infer(at::Tensor& self, const at::Tensor& p,
                                   c10::optional<at::Generator> generator) {
  return self;
}

at::Tensor& bernoulli__shape_infer(at::Tensor& self, double p,
                                   c10::optional<at::Generator> generator) {
  return self;
}

}  // namespace aotops

}  // namespace torch_gcu
