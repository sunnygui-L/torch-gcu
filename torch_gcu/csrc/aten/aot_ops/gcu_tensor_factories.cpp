#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/aotops_shape_infer_func.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"

namespace torch_gcu {
namespace aotops {

at::Tensor& zero_(at::Tensor& self) {
  aotops::fill_(self, 0);
  return self;
}

at::Scalar clamp_scalar(const at::ScalarType& target_scalar_type,
                        const at::Scalar& input, const at::Scalar& max,
                        const at::Scalar& min) {
  if (target_scalar_type == at::ScalarType::Double) {
    if (input.toDouble() > max.toDouble()) {
      TORCH_WARN_ONCE("GCU not support fill ", input, " to an ",
                      target_scalar_type, " tensor. Fill ", max, " instead.");
      return max;
    } else if (input.toDouble() < min.toDouble()) {
      return min;
      TORCH_WARN_ONCE("GCU not support fill ", input, " to an ",
                      target_scalar_type, " tensor. Fill ", min, " instead.");
    } else {
      return input;
    }
  }
  if (target_scalar_type == at::ScalarType::Long) {
    if (input.toLong() > max.toLong()) {
      TORCH_WARN_ONCE("GCU not support fill ", input, " to an ",
                      target_scalar_type, " tensor. Fill ", max, " instead.");
      return max;
    } else if (input.toLong() < min.toLong()) {
      TORCH_WARN_ONCE("GCU not support fill ", input, " to an ",
                      target_scalar_type, " tensor. Fill ", min, " instead.");
      return min;
    } else {
      return input;
    }
  }
  if (target_scalar_type == at::ScalarType::UInt64) {
    if (input.toUInt64() > max.toUInt64()) {
      // Check if input is UINT64_MAX (deterministic mode special value) to
      // avoid overflow when formatting input in TORCH_WARN_ONCE (which would
      // call input.toLong())
      if (input.toUInt64() >= std::numeric_limits<uint64_t>::max()) {
        // Skip TORCH_WARN_ONCE to avoid formatting input (which would trigger
        // overflow)
        return max;
      }
      TORCH_WARN_ONCE("GCU not support fill ", input, " to an ",
                      target_scalar_type, " tensor. Fill ", max, " instead.");
      return max;
    }
    if (input.toUInt64() < min.toUInt64()) {
      TORCH_WARN_ONCE("GCU not support fill ", input, " to an ",
                      target_scalar_type, " tensor. Fill ", min, " instead.");
      return min;
    }
    return input;
  }
  return input;
}

at::Tensor& fill_(at::Tensor& self, const at::Scalar& value) {
  auto iter = at::TensorIteratorConfig()
                  .set_check_mem_overlap(
                      false)  // Fill is idempotent, so overlap is okay
                  .check_all_same_dtype(false)
                  .add_output(self)
                  .resize_outputs(false)
                  .build();

  at::ScalarType dtype = iter.dtype();
  at::Scalar temp_value;
  if (is_narrow_type(dtype)) {
    switch (dtype) {
      case at::ScalarType::Long:
        temp_value = clamp_scalar(dtype, value, at::Scalar(INT_MAX),
                                  at::Scalar(INT_MIN));
        break;
      case at::ScalarType::Double:
        temp_value = clamp_scalar(
            dtype, value, at::Scalar(std::numeric_limits<float>::max()),
            at::Scalar(-std::numeric_limits<float>::max()));
        break;
      case at::ScalarType::UInt64:
        temp_value = clamp_scalar(dtype, value, at::Scalar(UINT32_MAX),
                                  at::Scalar(-UINT32_MAX - 1));
        break;
      default:
        temp_value = value;
    }
  } else {
    temp_value = value;
  }
  if (self.numel() == 0) return self;

  at::ScalarType narrow_dtype = get_gcu_scalar_type(dtype);
  topsatenScalar_t xvalue = scalarToTopsatenScalar(temp_value, narrow_dtype);

  bridge_topsatenFill__out1(self, xvalue);

  return self;
}

}  // namespace aotops

}  // namespace torch_gcu
