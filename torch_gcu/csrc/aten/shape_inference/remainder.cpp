/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/core/Tensor.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

at::Tensor remainder_shape_infer(const at::Scalar &self,
                                 const at::Tensor &other) {
  return aotops::empty(other.sizes(), other.options());
}

}  // namespace aotops

}  // namespace torch_gcu
