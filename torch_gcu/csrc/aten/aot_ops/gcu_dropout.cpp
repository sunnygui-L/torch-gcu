/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/dropout.h"

namespace torch_gcu {

namespace aotops {

std::tuple<at::Tensor, at::Tensor> native_dropout(
    const at::Tensor &input, double p, c10::optional<bool> opt_train) {
  auto outputs = native_dropout_shape_infer(input, p, opt_train);
  auto out1 = std::get<0>(outputs);
  auto out2 = std::get<1>(outputs);

  bridge_topsatenNativeDropout_out2(out1, out2, input, p, opt_train,
                                    getDefaultTopsatenGenerator(input));

  return std::tie(out1, out2);
}

}  // namespace aotops

}  // namespace torch_gcu
