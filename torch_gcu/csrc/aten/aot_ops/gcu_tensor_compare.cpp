/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/core/op_registration/adaption.h>

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/gcu_structured_shape_infer.h"
#include "aten/shape_inference/tensor_compare.h"

namespace torch_gcu {

namespace aotops {

namespace {
// NOTE: refer to pytorch/aten/src/ATen/native/ReduceOpsUtils.h
void zero_numel_check_dims(const at::Tensor &self, const int64_t dim,
                           const char *fn_name) {
  if (self.ndimension() == 0) {
    TORCH_CHECK_INDEX(dim == 0 || dim == -1, fn_name,
                      ": Expected reduction dim -1 or 0 for scalar but got ",
                      dim);
  } else {
    TORCH_CHECK_INDEX(self.size(dim) != 0, fn_name, ": Expected reduction dim ",
                      dim, " to have non-zero size.");
  }
}

std::vector<int64_t> get_zero_numel_tensor_size(const at::Tensor &self,
                                                const int64_t dim,
                                                const bool keepdim,
                                                const char *fn_name) {
  TORCH_INTERNAL_ASSERT(self.numel() == 0, fn_name,
                        ": Expected self.numel() == 0.");
  zero_numel_check_dims(self, dim, fn_name);
  std::vector<int64_t> sizes;
  if (keepdim) {
    sizes = self.sizes().vec();
    sizes[dim] = 1;
  } else {
    for (const auto d : c10::irange(self.dim())) {
      if (d != dim) {
        sizes.push_back(self.sizes()[d]);
      }
    }
  }
  return sizes;
}

inline bool _dimreduce_return_trivial_no_ident(at::Tensor &result,
                                               const at::Tensor &self,
                                               int64_t /*dim*/,
                                               bool /*keepdim*/,
                                               const char * /*fn_name*/) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }
  return false;
}

using IdxVec = std::vector<int64_t>;
inline IdxVec ensure_nonempty_vec(IdxVec vec) {
  if (vec.empty()) {
    vec.push_back(1);
  }
  return vec;
}

inline int64_t ensure_nonempty_dim(int64_t dim) {
  return std::max<int64_t>(dim, 1);
}

inline int64_t ensure_nonempty_size(const at::TensorBase &t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.size(dim);
}
}  // namespace

at::Tensor &clamp_out(const at::Tensor &self,
                      const c10::optional<at::Scalar> &min,
                      const c10::optional<at::Scalar> &max, at::Tensor &out) {
  structured_clamp_gcu_out op(out);
  op.meta(self,
          (min.has_value() ? at::OptionalScalarRef(&(min.value()))
                           : at::OptionalScalarRef()),
          (max.has_value() ? at::OptionalScalarRef(&(max.value()))
                           : at::OptionalScalarRef()));

  bridge_topsatenClamp_out1(
      op.maybe_get_output(0), self,
      (min.has_value() ? at::OptionalScalarRef(&(min.value()))
                       : at::OptionalScalarRef()),
      (max.has_value() ? at::OptionalScalarRef(&(max.value()))
                       : at::OptionalScalarRef()));
  return out;
}

at::Tensor &clamp_out(const at::Tensor &self,
                      const c10::optional<at::Tensor> &min,
                      const c10::optional<at::Tensor> &max, at::Tensor &out) {
  structured_clamp_Tensor_gcu_out op(out);
  op.meta(self,
          ((min.has_value() && (*min).defined()) ? at::OptionalTensorRef(*min)
                                                 : at::OptionalTensorRef()),
          ((max.has_value() && (*max).defined()) ? at::OptionalTensorRef(*max)
                                                 : at::OptionalTensorRef()));

  bridge_topsatenClamp_out1(
      op.maybe_get_output(0), self,
      ((min.has_value() && (*min).defined()) ? at::OptionalTensorRef(*min)
                                             : at::OptionalTensorRef()),
      ((max.has_value() && (*max).defined()) ? at::OptionalTensorRef(*max)
                                             : at::OptionalTensorRef()));
  return out;
}

at::Tensor &where_out(const at::Tensor &condition, const at::Tensor &self,
                      const at::Tensor &other, at::Tensor &out) {
  where_out_shape_infer(condition, self, other, out);
  auto xcondition =
      is_cpu_scalar(condition) ? condition.to(out.device()) : condition;
  if (is_cpu_scalar(self) && is_cpu_scalar(other)) {
    auto xself = scalarTensorToTopsatenScalar(self);
    auto xother = scalarTensorToTopsatenScalar(other);
    bridge_topsatenWhere_out1(out, xcondition, xself, xother);
  } else if (is_cpu_scalar(self)) {
    auto xself = scalarTensorToTopsatenScalar(self);
    bridge_topsatenWhere_out1(out, xcondition, xself, other);
  } else if (is_cpu_scalar(other)) {
    auto xother = scalarTensorToTopsatenScalar(other);
    bridge_topsatenWhere_out1(out, xcondition, self, xother);
  } else {
    bridge_topsatenWhere_out1(out, xcondition, self, other);
  }
  return out;
}

at::Tensor where(const at::Tensor &condition, const at::Tensor &self,
                 const at::Tensor &other) {
  auto out = where_shape_infer(condition, self, other);
  auto xcondition =
      is_cpu_scalar(condition) ? condition.to(out.device()) : condition;
  if (is_cpu_scalar(self) && is_cpu_scalar(other)) {
    auto xself = scalarTensorToTopsatenScalar(self);
    auto xother = scalarTensorToTopsatenScalar(other);
    bridge_topsatenWhere_out1(out, xcondition, xself, xother);
  } else if (is_cpu_scalar(self)) {
    auto xself = scalarTensorToTopsatenScalar(self);
    bridge_topsatenWhere_out1(out, xcondition, xself, other);
  } else if (is_cpu_scalar(other)) {
    auto xother = scalarTensorToTopsatenScalar(other);
    bridge_topsatenWhere_out1(out, xcondition, self, xother);
  } else {
    bridge_topsatenWhere_out1(out, xcondition, self, other);
  }
  return out;
}

::std::tuple<at::Tensor &, at::Tensor &> mode_out(const at::Tensor &self,
                                                  int64_t dim, bool keepdim,
                                                  at::Tensor &values,
                                                  at::Tensor &indices) {
  TORCH_CHECK(self.layout() == at::Layout::Strided,
              "mode only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == values.device(), "expected device '",
              self.device(), "' but got '", values.device(),
              "' for values output");
  TORCH_CHECK(self.device() == indices.device(), "expected device '",
              self.device(), "' but got '", indices.device(),
              "' for indices output");
  TORCH_CHECK(self.scalar_type() == values.scalar_type(),
              "expected scalar type '", self.scalar_type(), "' but got '",
              values.scalar_type(), "' for values output");
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Long,
              "expected scalar type '", at::ScalarType::Long, "' but got '",
              indices.scalar_type(), "' for indices output");
  dim = at::maybe_wrap_dim(dim, self.dim());
  if (self.numel() == 0) {
    auto sizes = get_zero_numel_tensor_size(self, dim, keepdim, "mode()");
    resize_output(values, sizes);
    resize_output(indices, sizes);
    return std::tie(values, indices);
  } else if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim,
                                                "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    auto result = [&]() {
      at::NoNamesGuard guard;
      auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
      int64_t ndim = ensure_nonempty_dim(self.dim());
      int64_t slice_size = ensure_nonempty_size(self, dim);
      int64_t slices = self.numel() / slice_size;

      // Resize output value, index Tensors to appropriate sizes (i.e. the same
      // as the input Tensor, except at dim=dimension, the size is 1)
      assert(0 <= dim && static_cast<size_t>(dim) < self_sizes.size());
      self_sizes[dim] = 1;

      if (!keepdim) {
        if (values.ndimension() >= dim) {
          values.unsqueeze_(dim);
        }
        if (indices.ndimension() >= dim) {
          indices.unsqueeze_(dim);
        }
      }

      resize_output(values, self_sizes);
      resize_output(indices, self_sizes);

      // If sliceSize is 1, copy input to values and set indices
      if (slice_size == 1) {
        values.copy_(self);
        indices.fill_(0);
        if (!keepdim) {
          values.squeeze_(dim);
          indices.squeeze_(dim);
        }
        return std::tuple<at::Tensor &, at::Tensor &>{values, indices};
      }

      if (!keepdim) {
        values.squeeze_(dim);
        indices.squeeze_(dim);
      }

      bridge_topsatenMode_out2(values, indices, self, dim, keepdim);
      return std::tuple<at::Tensor &, at::Tensor &>{values, indices};
    }();
    at::namedinference::propagate_names_for_reduction(std::get<0>(result), self,
                                                      dim, keepdim);
    at::namedinference::propagate_names_for_reduction(std::get<1>(result), self,
                                                      dim, keepdim);
    return result;
  }
}

::std::tuple<at::Tensor, at::Tensor> mode(const at::Tensor &self, int64_t dim,
                                          bool keepdim) {
  at::Tensor values = aotops::empty({0}, self.options());
  at::Tensor indices = aotops::empty({0}, self.options().dtype(at::kLong));
  return mode_out(self, dim, keepdim, values, indices);
}

at::Tensor &isin_out(const at::Scalar &element, const at::Tensor &test_elements,
                     bool assume_unique, bool invert, at::Tensor &out) {
  structured_isin_Scalar_Tensor_gcu_out op(out);
  op.meta(element, test_elements, assume_unique, invert);
  return isin_out(
      at::native::wrapped_scalar_tensor(element, test_elements.device()),
      test_elements, assume_unique, invert, out);
}
}  // namespace aotops

}  // namespace torch_gcu
