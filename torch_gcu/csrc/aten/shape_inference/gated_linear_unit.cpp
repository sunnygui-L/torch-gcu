/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

at::Tensor glu_jvp_shape_infer(const at::Tensor& glu, const at::Tensor& x,
                               const at::Tensor& dx, int64_t dim) {
  dim = at::maybe_wrap_dim(dim, x.dim());
  const auto glu_size = glu.size(dim);
  const auto b = x.narrow(dim, glu_size, glu_size);
  const auto da = dx.narrow(dim, 0, glu_size);
  const auto db = dx.narrow(dim, glu_size, glu_size);
  auto dglu = aotops::empty(glu.sizes(), glu.options());
  auto iter = at::TensorIteratorConfig()
                  .add_output(dglu)
                  .add_input(glu)
                  .add_input(b)
                  .add_input(da)
                  .add_input(db)
                  .build();
  return dglu;
}

}  // namespace aotops

}  // namespace torch_gcu
