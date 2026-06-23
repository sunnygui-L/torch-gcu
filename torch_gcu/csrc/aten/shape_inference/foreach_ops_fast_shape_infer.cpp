/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include <ATen/native/ForeachUtils.h>

#include "aten/shape_inference/foreach_ops_fast_shape_infer.h"

namespace torch_gcu {

namespace aotops {

namespace {

inline void increment_version(at::TensorList tensors) {
  for (const auto& t : tensors) {
    t.unsafeGetTensorImpl()->bump_version();
  }
}

}  // namespace

std::vector<at::Tensor> foreach_op_fast_shape_infer(at::TensorList input) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  return vec_res;
}

void foreach_op__fast_shape_infer(at::TensorList input) {
  increment_version(input);
}

std::vector<at::Tensor> foreach_op_fast_shape_infer(at::TensorList tensors,
                                                    const at::Tensor& scalar) {
  TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1,
              "scalar tensor expected to be 0 dim but it has ", scalar.dim(),
              " dimensions and ", scalar.numel(), " elements.");
  TORCH_CHECK(tensors[0].device() == scalar.device(),
              "scalar tensor expected to be on ", tensors[0].device(),
              " but is on ", scalar.device());
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  return vec_res;
}

void foreach_op__fast_shape_infer(at::TensorList tensors,
                                  const at::Tensor& scalar) {
  TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1,
              "scalar tensor expected to be 0 dim but has ", scalar.dim(),
              " dimensions and ", scalar.numel(), " elements.");
  TORCH_CHECK(tensors[0].device() == scalar.device(),
              "scalar tensor is expected to be on ", tensors[0].device(),
              " but is on ", scalar.device());
  increment_version(tensors);
}

}  // namespace aotops

}  // namespace torch_gcu
