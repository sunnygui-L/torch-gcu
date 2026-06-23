/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include <ATen/ScalarOps.h>
#include <ATen/native/ForeachUtils.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

::std::vector<at::Tensor> _foreach_norm_shape_infer(
    at::TensorList self, const at::Scalar& ord,
    std::optional<at::ScalarType> opt_dtype) {
  double p;
  if (ord.isIntegral(false)) {
    p = ord.to<int64_t>();
  } else if (ord.isFloatingPoint()) {
    p = ord.to<double>();
  } else {
    TORCH_CHECK(false, "_foreach_norm expects ord to be integer or float");
  }

  const int ntensors = self.size();
  std::vector<at::Tensor> result;
  result.reserve(ntensors);
  for (const auto& i : c10::irange(ntensors)) {
    auto options = self[i].options().dtype(
        toRealValueType(opt_dtype.value_or(self[i].scalar_type())));
    result.emplace_back(empty({}, options));
  }
  return result;
}

}  // namespace aotops

}  // namespace torch_gcu
