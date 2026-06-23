/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
// #include <ATen/ScalarOps.h>
#include <ATen/native/ForeachUtils.h>

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

namespace {

inline void increment_version(at::TensorList tensors) {
  for (const auto& t : tensors) {
    t.unsafeGetTensorImpl()->bump_version();
  }
}

}  // namespace

#define FOREACH_TERNARY_OP_SHAPE_INFER(OP)                                \
  std::vector<at::Tensor> foreach_tensor_ternary_##OP##_slow_shape_infer( \
      at::TensorList tensors1, at::TensorList tensors2,                   \
      at::TensorList tensors3) {                                          \
    std::vector<at::Tensor> result;                                       \
    for (const auto i : c10::irange(tensors1.size())) {                   \
      result.emplace_back(                                                \
          OP##_shape_infer(tensors1[i], tensors2[i], tensors3[i]));       \
    }                                                                     \
    return result;                                                        \
  }

FOREACH_TERNARY_OP_SHAPE_INFER(lerp);

std::vector<at::Tensor> _foreach_lerp_shape_infer(at::TensorList tensors1,
                                                  at::TensorList tensors2,
                                                  at::TensorList tensors3) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  return vec_res;
}

void _foreach_lerp__shape_infer(at::TensorList tensors1,
                                at::TensorList tensors2,
                                at::TensorList tensors3) {
  increment_version(tensors1);
}

std::vector<at::Tensor> _foreach_lerp_shape_infer(at::TensorList tensors1,
                                                  at::TensorList tensors2,
                                                  const at::Scalar& weight) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  return vec_res;
}

void _foreach_lerp__shape_infer(at::TensorList tensors1,
                                at::TensorList tensors2,
                                const at::Scalar& weight) {
  increment_version(tensors1);
}

}  // namespace aotops

}  // namespace torch_gcu
