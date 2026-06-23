/*
 * Copyright 2022-2026 Enflame. All Rights Reserved.
 */

#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>

#include "aten/aot_ops/gcu_resize.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {
namespace aotops {

at::Tensor& adaptive_avg_pool2d_out_shape_infer(const at::Tensor& self,
                                                at::IntArrayRef output_size,
                                                at::Tensor& out) {
  at::TensorArg input_arg{self, "input", 1};
  at::TensorArg output_arg{out, "output", 2};
  checkAllSameGCU(__func__, {input_arg, output_arg});

  TORCH_CHECK(output_size.size() == 2,
              "adaptive_avg_pool2d: output_size must be 2");

  int64_t ndim = self.dim();
  TORCH_CHECK((ndim == 3 || ndim == 4),
              "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ",
              self.sizes());

  for (const auto i : {-2, -1}) {
    TORCH_CHECK(self.size(i) > 0,
                "adaptive_avg_pool2d(): Expected input to have non-zero size "
                "for non-batch dimensions, "
                "but input has sizes ",
                self.sizes(), " with dimension ", i + ndim,
                " being "
                "empty");
  }

  int64_t sizeD = self.size(-3);
  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  if (self.ndimension() == 4) {
    aotops::resize_output(out, {self.size(-4), sizeD, osizeH, osizeW});
  } else {
    aotops::resize_output(out, {sizeD, osizeH, osizeW});
  }

  return out;
}

}  // namespace aotops
}  // namespace torch_gcu