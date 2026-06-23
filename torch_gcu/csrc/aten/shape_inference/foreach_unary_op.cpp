/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */
#include <ATen/ScalarOps.h>
#include <ATen/native/ForeachUtils.h>

#include "aten/shape_inference/foreach_ops_fast_shape_infer.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

#define FOREACH_UNARY_OP(op_name)                                  \
  std::vector<at::Tensor> _foreach_##op_name##_shape_infer(        \
      at::TensorList tensors) {                                    \
    return foreach_op_fast_shape_infer(tensors);                   \
  }                                                                \
  void _foreach_##op_name##__shape_infer(at::TensorList tensors) { \
    foreach_op__fast_shape_infer(tensors);                         \
  }

FOREACH_UNARY_OP(erf)
FOREACH_UNARY_OP(erfc)
FOREACH_UNARY_OP(atan)
FOREACH_UNARY_OP(ceil)
FOREACH_UNARY_OP(cos)
FOREACH_UNARY_OP(cosh)
FOREACH_UNARY_OP(sqrt)
FOREACH_UNARY_OP(expm1)
FOREACH_UNARY_OP(floor)
FOREACH_UNARY_OP(abs)
FOREACH_UNARY_OP(asin)
FOREACH_UNARY_OP(log2)
FOREACH_UNARY_OP(acos)
FOREACH_UNARY_OP(exp)

}  // namespace aotops

}  // namespace torch_gcu
