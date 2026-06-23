/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/aot_ops/gcu_aot_ops.h"

#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

at::Tensor& masked_fill_(at::Tensor& self, const at::Tensor& mask,
                         const at::Tensor& value) {
  masked_fill__shape_infer(self, mask, value);
  if (self.numel() == 0) return self;
  if (is_cpu_scalar(value)) {
    auto xvalue = scalarTensorToTopsatenScalar(value);
    bridge_topsatenMasked_fill_out1(self, self, mask, xvalue);
  } else {
    bridge_topsatenMasked_fill_out1(self, self, mask, value);
  }
  return self;
}

}  // namespace aotops

}  // namespace torch_gcu
