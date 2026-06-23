/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

at::Tensor masked_select(const at::Tensor& self, const at::Tensor& mask) {
  auto result = masked_select_shape_infer(self, mask);
  auto device_mask = mask.to(self.device());
  bridge_topsatenMaskedSelect_out1(result, self, device_mask);
  return result;
}

at::Tensor& masked_select_out(const at::Tensor& self, const at::Tensor& mask,
                              at::Tensor& out) {
  masked_select_out_shape_infer(self, mask, out);
  if (out.numel() == 0) return out;
  bridge_topsatenMaskedSelect_out1(out, self, mask);
  return out;
}

}  // namespace aotops

}  // namespace torch_gcu