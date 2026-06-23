/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/tensor_compare.h"

#include "aten/shape_inference/gcu_structured_shape_infer.h"
#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace {

template <typename... Args>
at::Device out_device(Args &... inps) {
  for (const auto &i : {inps...}) {
    if (i.device() != at::kCPU) {
      return i.device();
    }
  }
  return at::kCPU;
}

}  // namespace

namespace aotops {

at::Tensor &where_self_out_shape_infer(const at::Tensor &condition,
                                       const at::Tensor &self,
                                       const at::Tensor &other,
                                       at::Tensor &out) {
  at::Tensor self_, other_, condition_;
  if (self.dtype() != other.dtype()) {
    auto result_type = at::native::result_type(self, other);
    self_ = at::empty_like(self, self.options().dtype(result_type));
    other_ = at::empty_like(other, other.options().dtype(result_type));
  } else {
    self_ = self;
    other_ = other;
  }

  auto device = out_device(condition, self_, other_);
  condition_ = condition;
  if (device != at::kCPU) {  // allow CPU scalars on non-cpu device
    if (condition.device() != device && condition.ndimension() == 0) {
      condition_ =
          at::empty_like(condition, condition.options().device(device));
    }
    if (self_.device() != device && self_.ndimension() == 0) {
      self_ = at::empty_like(self_, condition.options().device(device));
    }
    if (other_.device() != device && other_.ndimension() == 0) {
      other_ = at::empty_like(other_, condition.options().device(device));
    }
  }
  if (condition.scalar_type() == at::ScalarType::Byte) {
    TORCH_WARN_ONCE(
        "where received a uint8 condition tensor. This behavior is deprecated "
        "and will be removed in a future version of PyTorch. Use a boolean "
        "condition instead.");
  } else {
    TORCH_CHECK(condition.scalar_type() == at::ScalarType::Bool,
                "where expected condition to be a boolean tensor, but got a "
                "tensor with dtype ",
                condition.scalar_type());
  }
  condition_ =
      condition_.scalar_type() == at::ScalarType::Byte
          ? at::empty_like(condition_, condition.options().device(device))
          : condition_;
  // if there's still a device mismatch, let tensoriterator
  // error out with it
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(out)
                  .add_input(condition_)
                  .add_input(self_)
                  .add_input(other_)
                  .build();
  return out;
}

at::Tensor &where_out_shape_infer(const at::Tensor &condition,
                                  const at::Tensor &self,
                                  const at::Tensor &other, at::Tensor &out) {
  return where_self_out_shape_infer(condition, self, other, out);
}

at::Tensor where_shape_infer(const at::Tensor &condition,
                             const at::Tensor &self, const at::Tensor &other) {
  auto device = out_device(condition, self, other);
  auto result_type = at::native::result_type(self, other);
  at::Tensor ret = empty({0}, self.options().dtype(result_type).device(device));
  return where_self_out_shape_infer(condition, self, other, ret);
}

at::Tensor isnan_shape_infer(const at::Tensor &self) {
  structured_ne_Tensor_gcu_functional op;
  op.meta(self, self);
  return std::move(op.outputs_[0]);
}

}  // namespace aotops

}  // namespace torch_gcu
