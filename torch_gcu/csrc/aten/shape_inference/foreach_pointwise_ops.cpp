/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include <ATen/ScalarOps.h>
#include <ATen/native/ForeachUtils.h>

#include "aten/shape_inference/foreach_ops_fast_shape_infer.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

#define FOREACH_POINTWISE_OP_SCALAR_SHAPE_INFER(NAME)                         \
  std::vector<at::Tensor> _foreach_##NAME##_shape_infer(                      \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      const at::Scalar& scalar) {                                             \
    return foreach_op_fast_shape_infer(input);                                \
  }                                                                           \
                                                                              \
  void _foreach_##NAME##__shape_infer(                                        \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      const at::Scalar& scalar) {                                             \
    foreach_op__fast_shape_infer(input);                                      \
  }

#define FOREACH_POINTWISE_OP_SCALARLIST_SHAPE_INFER(NAME)                     \
  std::vector<at::Tensor> _foreach_##NAME##_shape_infer(                      \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      at::ArrayRef<at::Scalar> scalars) {                                     \
    return foreach_op_fast_shape_infer(input);                                \
  }                                                                           \
                                                                              \
  void _foreach_##NAME##__shape_infer(                                        \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      at::ArrayRef<at::Scalar> scalars) {                                     \
    foreach_op__fast_shape_infer(input);                                      \
  }

FOREACH_POINTWISE_OP_SCALAR_SHAPE_INFER(addcdiv)
FOREACH_POINTWISE_OP_SCALARLIST_SHAPE_INFER(addcdiv)

FOREACH_POINTWISE_OP_SCALAR_SHAPE_INFER(addcmul)
FOREACH_POINTWISE_OP_SCALARLIST_SHAPE_INFER(addcmul)

}  // namespace aotops

}  // namespace torch_gcu
