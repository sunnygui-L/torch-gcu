/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

::std::tuple<at::Tensor, at::Tensor> _weight_norm_interface_shape_infer(
    const at::Tensor &v, const at::Tensor &g, int64_t dim) {
  auto w = at::empty_like(v, at::MemoryFormat::Contiguous);

  // norms is only needed to stash for backward.
  // g.scalar_type() may be at::ScalarType::Double, Float, Half, or BFloat16.
  // If Half or BFloat16, stash norms as float.
  at::ScalarType AccType = (g.scalar_type() == at::ScalarType::Half ||
                            g.scalar_type() == at::ScalarType::BFloat16)
                               ? at::ScalarType::Float
                               : g.scalar_type();
  // Will this create norms on the same device as g, regardless of what the
  // thread's default current device is?  I believe so, because Type::*
  // functions are DeviceGuard()ed.
  auto norms =
      at::empty_strided(g.sizes(), g.strides(), g.options().dtype(AccType));

  return std::tuple<at::Tensor, at::Tensor>{w, norms};
}

}  // namespace aotops

}  // namespace torch_gcu