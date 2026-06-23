/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/sorting.h"

namespace torch_gcu {

namespace aotops {

std::tuple<at::Tensor &, at::Tensor &> sort_out(const at::Tensor &self,
                                                c10::optional<bool> stable,
                                                int64_t dim, bool descending,
                                                at::Tensor &values,
                                                at::Tensor &indices) {
  {
    // TODO
    // after topsaten fix, delete
    dim = at::maybe_wrap_dim(dim, self.dim());
    aotops::resize_output(values, self.sizes());
    aotops::resize_output(indices, self.sizes());
    auto xstable = stable.has_value() ? stable.value() : false;
    bridge_topsatenSort_out2(values, indices, self, xstable, dim, descending);
    return std::tie(values, indices);
  }
}

at::Tensor argsort(const at::Tensor &self, bool stable, int64_t dim,
                   bool descending) {
  dim = at::maybe_wrap_dim(dim, self.dim());
  auto out = argsort_shape_infer(self, stable, dim, descending);
  bridge_topsatenArgSort_out1(out, self, stable, dim, descending);
  return out;
}

}  // namespace aotops

}  // namespace torch_gcu
