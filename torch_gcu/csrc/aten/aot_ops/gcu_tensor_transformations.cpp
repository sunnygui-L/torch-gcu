#include "aten/aot_ops/gcu_ops.h"

#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/ops/empty_like.h>

#include "aten/aot_ops/topsaten_bridge.h"

namespace torch_gcu {

namespace aotops {

at::Tensor flip(const at::Tensor& self, at::IntArrayRef dims) {
  const int64_t total_dims = self.dim();
  // It wraps the dims and checks that there are no repeated dims
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);

  // Count dimensions in which we need to do work
  int n = 0;
  for (const auto i : c10::irange(total_dims)) {
    if (flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
      n++;
    }
  }

  at::Tensor out_tensor = at::empty_like(self);
  // Nothing to do, we return fast
  if (n == 0 || self.numel() <= 1) {
    out_tensor.copy_(self);
    return out_tensor;
  }

  bridge_topsatenFlip_out1(out_tensor, self, dims);
  return out_tensor;
}

}  // namespace aotops

}  // namespace torch_gcu