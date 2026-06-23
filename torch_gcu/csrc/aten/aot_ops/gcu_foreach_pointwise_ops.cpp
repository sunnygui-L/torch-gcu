#include <ATen/native/ForeachUtils.h>

#include "aten/aot_ops/gcu_aot_ops.h"

namespace torch_gcu {

namespace aotops {

#define FOREACH_POINTWISE_OP_TENSOR(NAME)                                     \
  std::vector<at::Tensor> _foreach_##NAME(                                    \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      const at::Tensor& scalars_) {                                           \
    auto scalars =                                                            \
        at::native::convert_tensor_to_scalar_list(scalars_, input.size());    \
    return aotops::_foreach_##NAME(input, tensors1, tensors2, scalars);       \
  }                                                                           \
                                                                              \
  void _foreach_##NAME##_(at::TensorList input, at::TensorList tensors1,      \
                          at::TensorList tensors2,                            \
                          const at::Tensor& scalars_) {                       \
    auto scalars =                                                            \
        at::native::convert_tensor_to_scalar_list(scalars_, input.size());    \
    return aotops::_foreach_##NAME##_(input, tensors1, tensors2, scalars);    \
  }

FOREACH_POINTWISE_OP_TENSOR(addcdiv)
FOREACH_POINTWISE_OP_TENSOR(addcmul)

bool _foreach_abs_check_slow_path(at::TensorList tensors) {
  at::native::check_foreach_api_restrictions(tensors);
  const bool has_complex = std::any_of(
      tensors.begin(), tensors.end(),
      [](const auto& t) { return at::isComplexType(t.scalar_type()); });
  return (!at::native::can_use_fast_route(tensors) || has_complex);
}

bool _foreach_abs__check_slow_path(at::TensorList tensors) {
  return _foreach_abs_check_slow_path(tensors);
}
}  // namespace aotops

}  // namespace torch_gcu
