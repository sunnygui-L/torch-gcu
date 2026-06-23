/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/summary_ops.h"

namespace torch_gcu {

namespace aotops {

at::Tensor& histc_out_shape_infer(const at::Tensor& self, int64_t nbins,
                                  const at::Scalar& min, const at::Scalar& max,
                                  at::Tensor& result){
  auto out = histc_shape_infer(self, nbins, min, max);
  aotops::resize_output(result, out.sizes());
  return result;
}

at::Tensor histc_shape_infer(const at::Tensor& self, int64_t nbins,
                             const at::Scalar& min, const at::Scalar& max) {
  if (self.scalar_type() == at::ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }

  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  at::Tensor output =
      at::zeros({nbins}, self.scalar_type(), c10::nullopt /* layout */,
                at::DeviceType::PrivateUse1, c10::nullopt /* pin_memory */);
  return output;
}

}  // namespace aotops

}  // namespace torch_gcu
