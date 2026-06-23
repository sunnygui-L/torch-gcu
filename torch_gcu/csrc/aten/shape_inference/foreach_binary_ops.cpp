/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include <ATen/ScalarOps.h>
#include <ATen/native/ForeachUtils.h>

#include "aten/shape_inference/foreach_ops_fast_shape_infer.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

#define FOREACH_BINARY_OP_LIST_ALPHA_SHAPE_INFER(NAME)           \
  void _foreach_##NAME##__shape_infer(at::TensorList tensors1,   \
                                      at::TensorList tensors2,   \
                                      const at::Scalar& alpha) { \
    foreach_op__fast_shape_infer(tensors1);                      \
  }                                                              \
                                                                 \
  std::vector<at::Tensor> _foreach_##NAME##_shape_infer(         \
      at::TensorList tensors1, at::TensorList tensors2,          \
      const at::Scalar& alpha) {                                 \
    return foreach_op_fast_shape_infer(tensors1);                \
  }

#define FOREACH_BINARY_OP_LIST_SHAPE_INFER(NAME, DIVISION_OP)    \
  void _foreach_##NAME##__shape_infer(at::TensorList tensors1,   \
                                      at::TensorList tensors2) { \
    foreach_op__fast_shape_infer(tensors1);                      \
  }                                                              \
                                                                 \
  std::vector<at::Tensor> _foreach_##NAME##_shape_infer(         \
      at::TensorList tensors1, at::TensorList tensors2) {        \
    return foreach_op_fast_shape_infer(tensors1);                \
  }

#define FOREACH_BINARY_OP_SCALAR_SHAPE_INFER(NAME, DIVISION_OP)   \
                                                                  \
  void _foreach_##NAME##__shape_infer(at::TensorList tensors,     \
                                      const at::Scalar& scalar) { \
    foreach_op__fast_shape_infer(tensors);                        \
  }                                                               \
                                                                  \
  std::vector<at::Tensor> _foreach_##NAME##_shape_infer(          \
      at::TensorList tensors, const at::Scalar& scalar) {         \
    return foreach_op_fast_shape_infer(tensors);                  \
  }

#define FOREACH_BINARY_OP_SCALARLIST_SHAPE_INFER(NAME, DIV_OP)            \
  void _foreach_##NAME##__shape_infer(at::TensorList tensors,             \
                                      at::ArrayRef<at::Scalar> scalars) { \
    foreach_op__fast_shape_infer(tensors);                                \
  }                                                                       \
                                                                          \
  std::vector<at::Tensor> _foreach_##NAME##_shape_infer(                  \
      at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {         \
    return foreach_op_fast_shape_infer(tensors);                          \
  }

#define FOREACH_BINARY_OP_SCALAR_TENSOR_SHAPE_INFER(NAME, DIVISION_OP) \
  void _foreach_##NAME##__shape_infer(at::TensorList tensors,          \
                                      const at::Tensor& scalar) {      \
    foreach_op_fast_shape_infer(tensors, scalar);                      \
  }                                                                    \
                                                                       \
  std::vector<at::Tensor> _foreach_##NAME##_shape_infer(               \
      at::TensorList tensors, const at::Tensor& scalar) {              \
    return foreach_op_fast_shape_infer(tensors, scalar);               \
  }

FOREACH_BINARY_OP_LIST_ALPHA_SHAPE_INFER(add);
FOREACH_BINARY_OP_SCALAR_SHAPE_INFER(add, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST_SHAPE_INFER(add, /*div_op*/ false);

FOREACH_BINARY_OP_LIST_ALPHA_SHAPE_INFER(sub);
FOREACH_BINARY_OP_SCALAR_SHAPE_INFER(sub, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST_SHAPE_INFER(sub, /*div_op*/ false);

FOREACH_BINARY_OP_LIST_SHAPE_INFER(div, true)
FOREACH_BINARY_OP_SCALAR_SHAPE_INFER(div, true)
FOREACH_BINARY_OP_SCALARLIST_SHAPE_INFER(div, true)
FOREACH_BINARY_OP_SCALAR_TENSOR_SHAPE_INFER(div, true)

FOREACH_BINARY_OP_LIST_SHAPE_INFER(mul, false)
FOREACH_BINARY_OP_SCALAR_SHAPE_INFER(mul, false)
FOREACH_BINARY_OP_SCALARLIST_SHAPE_INFER(mul, false)
FOREACH_BINARY_OP_SCALAR_TENSOR_SHAPE_INFER(mul, false)

FOREACH_BINARY_OP_LIST_SHAPE_INFER(clamp_max, true)
FOREACH_BINARY_OP_SCALAR_SHAPE_INFER(clamp_max, true)
FOREACH_BINARY_OP_SCALARLIST_SHAPE_INFER(clamp_max, true)

FOREACH_BINARY_OP_LIST_SHAPE_INFER(clamp_min, true)
FOREACH_BINARY_OP_SCALAR_SHAPE_INFER(clamp_min, true)
FOREACH_BINARY_OP_SCALARLIST_SHAPE_INFER(clamp_min, true)

FOREACH_BINARY_OP_LIST_SHAPE_INFER(minimum, true)
FOREACH_BINARY_OP_SCALAR_SHAPE_INFER(minimum, true)
FOREACH_BINARY_OP_SCALARLIST_SHAPE_INFER(minimum, true)

FOREACH_BINARY_OP_LIST_SHAPE_INFER(maximum, true)
FOREACH_BINARY_OP_SCALAR_SHAPE_INFER(maximum, true)
FOREACH_BINARY_OP_SCALARLIST_SHAPE_INFER(maximum, true)

}  // namespace aotops

}  // namespace torch_gcu
