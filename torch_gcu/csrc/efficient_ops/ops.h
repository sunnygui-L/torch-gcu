#pragma once

#include <ATen/NamedTensorUtils.h>
#include <ATen/Tensor.h>
#include <ATen/TensorIterator.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "gcu/gcu_guard.h"

namespace torch_gcu {
namespace efficient {
at::Tensor create_out(at::IntArrayRef sizes, at::IntArrayRef strides,
                      const at::TensorOptions& options);
struct structured_device_int32_indices_index_Tensor
    : public at::TensorIteratorBase {
  template <bool SIZES = false, bool STRIDES = false>
  struct precompute_out {
    precompute_out<true, STRIDES> set_sizes(at::DimVector value) {
      static_assert(SIZES == false, "sizes already set");
      precompute_out<true, STRIDES> ret;
      ret.sizes = value;
      ret.strides = this->strides;
      return ret;
    }

    precompute_out<SIZES, true> set_strides(at::DimVector value) {
      static_assert(STRIDES == false, "strides already set");
      precompute_out<SIZES, true> ret;
      ret.sizes = this->sizes;
      ret.strides = value;
      return ret;
    }

    at::DimVector sizes;
    at::DimVector strides;
  };
  using meta_return_ty = precompute_out<true, true>;
  meta_return_ty meta(const at::Tensor& self, at::IOptTensorListRef indices);
};

struct structured_device_int32_indices_index_Tensor_gcu_functional final
    : public structured_device_int32_indices_index_Tensor {
  void set_output_strided(int64_t output_idx, at::IntArrayRef sizes,
                          at::IntArrayRef strides, at::TensorOptions options,
                          at::DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      at::namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    structured_device_int32_indices_index_Tensor::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(int64_t output_idx, at::IntArrayRef sizes,
                              at::IntArrayRef strides,
                              at::TensorOptions options,
                              at::DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      at::namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    structured_device_int32_indices_index_Tensor::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const at::Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx];
  }
  std::array<at::Tensor, 1> outputs_;
  torch_gcu::OptionalGCUGuard guard_;
};

TORCH_GCU_API at::Tensor device_int32_indices_index(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices);

}  // namespace efficient
}  // namespace torch_gcu
