#include <ATen/native/ForeachUtils.h>

#include "aten/aot_ops/gcu_aot_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/foreach_op_utils.h"
#include "aten/shape_inference/foreach_binary_ops.h"

namespace torch_gcu {

namespace aotops {

::std::vector<at::Tensor> _foreach_mul(at::TensorList self,
                                       const at::Tensor& other) {
  if (is_cpu_scalar(other)) {
    auto other_scalar = scalarTensorToScalar(other);
    return aotops::_foreach_mul(self, other_scalar);
  }
  auto use_slow = _foreach_mul_check_slow_path(self, other);
  if (use_slow) {
    return _foreach_mul_slow_path(self, other);
  }

  auto result = _foreach_mul_shape_infer(self, other);
  bridge_topsatenForeachMul_out1(result, self, other);
  return result;
}

void _foreach_mul_(at::TensorList self, const at::Tensor& other) {
  if (is_cpu_scalar(other)) {
    auto other_scalar = scalarTensorToScalar(other);
    return aotops::_foreach_mul_(self, other_scalar);
  }
  auto use_slow = _foreach_mul__check_slow_path(self, other);
  if (use_slow) {
    return _foreach_mul__slow_path(self, other);
  }

  _foreach_mul__shape_infer(self, other);
  bridge_topsatenForeachMul_out1(self, self, other);
  return;
}

::std::vector<at::Tensor> _foreach_div(at::TensorList self,
                                       const at::Tensor& other) {
  if (is_cpu_scalar(other)) {
    auto other_scalar = scalarTensorToScalar(other);
    return aotops::_foreach_div(self, other_scalar);
  }
  auto use_slow = _foreach_div_check_slow_path(self, other);
  if (use_slow) {
    return _foreach_div_slow_path(self, other);
  }

  auto result = _foreach_div_shape_infer(self, other);
  bridge_topsatenForeachDiv_out1(result, self, other);
  return result;
}

void _foreach_div_(at::TensorList self, const at::Tensor& other) {
  if (is_cpu_scalar(other)) {
    auto other_scalar = scalarTensorToScalar(other);
    return aotops::_foreach_div_(self, other_scalar);
  }
  auto use_slow = _foreach_div__check_slow_path(self, other);
  if (use_slow) {
    return _foreach_div__slow_path(self, other);
  }

  _foreach_div__shape_infer(self, other);
  bridge_topsatenForeachDiv_out1(self, self, other);
  return;
}

}  // namespace aotops

}  // namespace torch_gcu
