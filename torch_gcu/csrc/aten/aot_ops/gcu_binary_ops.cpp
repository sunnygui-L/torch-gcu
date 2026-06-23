#include "aten/aot_ops/gcu_ops.h"

#include <ATen/ATen.h>
#include <ATen/ScalarOps.h>

#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/binary_ops.h"
#include "aten/shape_inference/gcu_structured_shape_infer.h"

namespace torch_gcu {

namespace aotops {

namespace {

#define BINARY_KERNEL_LAUNCH(topsatenop, out, lhs, rhs, alpha...) \
  if (is_cpu_scalar(lhs)) {                                       \
    auto xlhs = scalarTensorToTopsatenScalar(lhs);                \
    bridge_##topsatenop##_out1(out, xlhs, rhs, ##alpha);          \
  } else if (is_cpu_scalar(rhs)) {                                \
    auto xrhs = scalarTensorToTopsatenScalar(rhs);                \
    bridge_##topsatenop##_out1(out, lhs, xrhs, ##alpha);          \
  } else {                                                        \
    bridge_##topsatenop##_out1(out, lhs, rhs, ##alpha);           \
  }

#define BINARY_WITH_OUT_AND_SCALARTENSOR(topsatenop, metaop, out, lhs, rhs, \
                                         alpha...)                          \
  RESIZE_OUT(metaop, out, lhs, rhs, ##alpha)                                \
  BINARY_KERNEL_LAUNCH(topsatenop, out, lhs, rhs, ##alpha)                  \
  return out;

#define BINARY_COMPARISON_WITH_OUT(topsatenop, out, self, other)         \
  auto Iother = c10::IValue(other);                                      \
  auto rhs = Iother.isTensor()                                           \
                 ? Iother.toTensor()                                     \
                 : at::native::wrapped_scalar_tensor(Iother.toScalar()); \
  auto iter = at::TensorIterator();                                      \
  iter.build_borrowing_comparison_op(out, self, rhs);                    \
  out = iter.output(0);                                                  \
  BINARY_KERNEL_LAUNCH(topsatenop, out, iter.input(0), iter.input(1))    \
  return out;

}  // namespace

at::Tensor rsub(const at::Tensor &self, const at::Tensor &other,
                const at::Scalar &alpha) {
  return at::sub(other, self, alpha);
}

// TODO:
// delete after copysign_out has scalar interface
at::Tensor &copysign_out(const at::Tensor &self, const at::Tensor &other,
                         at::Tensor &out) {
  structured_copysign_Tensor_gcu_out op(out);
  op.meta(self, other);
  at::Tensor out_tmp = op.maybe_get_output(0);

  if (is_cpu_scalar(self)) {
    auto xlhs = self.to(out.device());
    bridge_topsatenCopysign_out1(out, xlhs, other);
  } else if (is_cpu_scalar(other)) {
    auto xrhs = other.to(out.device());
    bridge_topsatenCopysign_out1(out, self, xrhs);
  } else {
    bridge_topsatenCopysign_out1(out, self, other);
  }
  return out;
}

// TODO:
// delete after we have floor_divide topsaten op
at::Tensor floor_divide(const at::Tensor &self, const at::Tensor &other) {
  auto result = floor_divide_shape_infer(self, other);
  if (result.numel() == 0) return result;
  c10::optional<c10::string_view> rounding_mode = "floor";
  BINARY_KERNEL_LAUNCH(topsatenDiv, result, self, other, rounding_mode);
  return result;
}

// TODO:
// delete after we have floor_divide_out topsaten op
at::Tensor &floor_divide_out(const at::Tensor &self, const at::Tensor &other,
                             at::Tensor &out) {
  floor_divide_out_shape_infer(self, other, out);
  if (out.numel() == 0) return out;
  c10::optional<c10::string_view> rounding_mode = "floor";
  BINARY_KERNEL_LAUNCH(topsatenDiv, out, self, other, rounding_mode);
  return out;
}

// TODO:
// delete after we have floor_divide_ topsaten op
at::Tensor &floor_divide_(at::Tensor &self, const at::Tensor &other) {
  floor_divide__shape_infer(self, other);
  if (self.numel() == 0) return self;
  c10::optional<c10::string_view> rounding_mode = "floor";
  BINARY_KERNEL_LAUNCH(topsatenDiv, self, self, other, rounding_mode);
  return self;
}

}  // namespace aotops

}  // namespace torch_gcu
