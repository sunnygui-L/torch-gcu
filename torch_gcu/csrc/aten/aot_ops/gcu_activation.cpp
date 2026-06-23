/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/gcu_structured.h"
#include "aten/shape_inference/gcu_structured_shape_infer.h"
#include "gcu/sys_util.h"

namespace torch_gcu {

namespace aotops {

at::Tensor &gelu_out(const at::Tensor &self, c10::string_view approximate,
                     at::Tensor &out) {
  std::string erf_enable =
      util::GetEnvString("ENFLAME_PT_GELU_ERF_ENABLE", "false");
  std::transform(erf_enable.begin(), erf_enable.end(), erf_enable.begin(),
                 ::tolower);
  if (erf_enable == "false") {
    if (approximate == "none") {
      TORCH_WARN_ONCE(
          "GCU gelu default use tanh of approximate. You can use export "
          "ENFLAME_PT_GELU_ERF_ENABLE=true to enable erf.");
      approximate = c10::string_view("tanh");
    }
  }

  structured_gelu_gcu_out op(out);
  op.meta(self, approximate);
  bridge_topsatenGelu_out1(out, self, approximate);
  return out;
}

}  // namespace aotops

}  // namespace torch_gcu