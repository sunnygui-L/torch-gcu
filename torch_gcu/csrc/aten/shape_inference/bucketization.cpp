/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/DeviceGuard.h>
#include <ATen/core/op_registration/adaption.h>
#include "ATen/native/BucketizationUtils.h"
#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_resize.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {
namespace aotops {

at::Tensor bucketize_shape_infer(const at::Tensor& self,
                                 const at::Tensor& boundaries, bool out_int32,
                                 bool right) {
  at::ScalarType scalar_type =
      out_int32 ? at::ScalarType::Int : at::ScalarType::Long;
  c10::TensorOptions options = at::TensorOptions()
                                   .device(self.options().device())
                                   .dtype(scalar_type)
                                   .memory_format(at::MemoryFormat::Contiguous);
  at::Tensor result = aotops::empty({0}, options);
  aotops::resize_output(result, self.sizes());

  return result;
}

at::Tensor& bucketize_out_shape_infer(const at::Tensor& self,
                                      const at::Tensor& boundaries,
                                      bool out_int32, bool right,
                                      at::Tensor& out) {
  TORCH_CHECK(boundaries.dim() == 1,
              "boundaries tensor must be 1 dimension, but got dim(",
              boundaries.dim(), ")");
  aotops::resize_output(out, self.sizes());

  return out;
}

at::Tensor bucketize_shape_infer(const at::Scalar& self,
                                 const at::Tensor& boundaries, bool out_int32,
                                 bool right) {
  return bucketize_shape_infer(
      at::native::searchsorted_scalar_tensor(self, boundaries.device()),
      boundaries, out_int32, right);
}

}  // namespace aotops
}  // namespace torch_gcu
