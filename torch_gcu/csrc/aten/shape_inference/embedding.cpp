/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "aten/shape_inference/shape_infer_func.h"
#include "aten/shape_inference/tensor_utils.h"
namespace torch_gcu {

namespace aotops {

at::Tensor embedding_dense_backward_shape_infer(const at::Tensor& grad_,
                                                const at::Tensor& indices_,
                                                int64_t num_weights,
                                                int64_t padding_idx,
                                                bool scale_grad_by_freq) {
  auto grad_arg = at::TensorArg(grad_, "grad", 1);
  auto indices_arg = at::TensorArg(indices_, "indices", 1);
  checkScalarTypes("embedding_backward", indices_arg, {at::kLong, at::kInt});
  checkSameGCU("embedding_backward", grad_arg, indices_arg);
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());
  return grad_weight;
}
}  // namespace aotops

}  // namespace torch_gcu
