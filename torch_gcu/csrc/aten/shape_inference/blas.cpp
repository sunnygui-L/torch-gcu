/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/blas.h"

#include <c10/core/ScalarType.h>
#include <c10/util/SmallVector.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

namespace {

// Shape inference version of check_valid_strides_and_return_transposed.
// Function name changed to indicate this is for shape inference.
bool check_valid_strides_and_return_transposed_shape_infer(
    const at::Tensor& mat) {
  at::IntArrayRef tensor_strides = mat.strides();
  at::IntArrayRef tensor_sizes = mat.sizes();
  int end_dim = mat.dim() - 1;
  int alignment = 16 / mat.element_size();
  TORCH_CHECK(uint64_t(mat.data_ptr()) % 16 == 0,
              "expected data_ptr to be aligned to 16 bytes\n");
  if ((tensor_strides[end_dim - 1] == 1) &&
      (tensor_strides[end_dim] >=
       std::max<int64_t>(1, tensor_sizes[end_dim - 1]))) {
    TORCH_CHECK(tensor_strides[end_dim] % alignment == 0,
                "strides should be multiple of 16 bytes");
    return true;
  } else if ((tensor_strides[end_dim] == 1) &&
             (tensor_strides[end_dim - 1] >=
              std::max<int64_t>(1, tensor_sizes[end_dim]))) {
    TORCH_CHECK(tensor_strides[end_dim - 1] % alignment == 0,
                "strides should be multiple of 16 bytes");
    return false;
  } else {
    TORCH_CHECK(false, "Invalid strides/sizes, got ", mat.strides(),
                " for strides and ", mat.sizes(), " for sizes");
  }
}

// Shape inference version of create_grouped_gemm_output_tensor.
// Function name changed to indicate this is for shape inference.
// Uses aotops::empty_strided instead of at::empty_strided for GCU shape
// inference.
at::Tensor create_grouped_gemm_output_tensor_shape_infer(
    const at::Tensor& mat_a, const at::Tensor& mat_b,
    const ::std::optional<at::Tensor>& offs,
    ::std::optional<at::ScalarType> out_dtype) {
  c10::SmallVector<int64_t, 3> out_size;
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  if (a_is_2d) {
    if (b_is_2d) {
      out_size = {offs->size(0), mat_a.size(0), mat_b.size(1)};
    } else {
      TORCH_CHECK(offs->size(0) == mat_b.size(0),
                  "matrix batch sizes have to match");
      out_size = {mat_a.size(0), mat_b.size(-1)};
    }
  } else {
    if (b_is_2d) {
      TORCH_CHECK(offs->size(0) == mat_a.size(0),
                  "matrix batch sizes have to match");
      out_size = {mat_a.size(1), mat_b.size(1)};
    } else {
      TORCH_CHECK(mat_a.size(0) == mat_b.size(0),
                  "batched dimension has to match");
      out_size = {mat_a.size(0), mat_a.size(1), mat_b.size(-1)};
    }
  }

  const auto out_dtype_ = out_dtype.value_or(at::kBFloat16);
  TORCH_CHECK(
      out_dtype_ == at::kBFloat16,
      "Only bf16 high precision output types are supported for grouped gemm");

  const auto last_dim = out_size.size() - 1;
  const auto alignment = 16 / c10::elementSize(out_dtype_);
  const int64_t size_padded =
      (out_size[last_dim] + alignment - 1) / alignment * alignment;
  std::vector<int64_t> out_stride;
  if (a_is_2d != b_is_2d) {
    out_stride = {size_padded, 1};
  } else {
    out_stride = {out_size[1] * size_padded, size_padded, 1};
  }
  // Use aotops::empty_strided instead of at::empty_strided for GCU shape
  // inference.
  auto out = at::empty_strided(out_size, out_stride,
                               mat_a.options().dtype(out_dtype_));

  return out;
}

}  // namespace

at::Tensor _grouped_mm_shape_infer(const at::Tensor& self,
                                   const at::Tensor& mat2,
                                   const ::std::optional<at::Tensor>& offs,
                                   const ::std::optional<at::Tensor>& bias,
                                   ::std::optional<at::ScalarType> out_dtype) {
  // Parameter names (self, mat2) match PyTorch schema, while original CUDA
  // implementation uses (mat_a, mat_b).
  TORCH_CHECK(self.dtype() == at::kBFloat16,
              "Expected mat_a to be BFloat16 matrix got ", self.scalar_type());
  TORCH_CHECK(mat2.dtype() == at::kBFloat16,
              "Expected mat_b to be BFloat16 matrix got ", mat2.scalar_type());
  TORCH_CHECK(self.dim() == 2 || self.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK(mat2.dim() == 2 || mat2.dim() == 3, "mat_b has to be 2 or 3d");
  const bool a_is_2d = self.dim() == 2;
  const bool b_is_2d = mat2.dim() == 2;

  // Use shape inference version of check_valid_strides_and_return_transposed.
  check_valid_strides_and_return_transposed_shape_infer(self);
  check_valid_strides_and_return_transposed_shape_infer(mat2);
  TORCH_CHECK(offs.has_value() == (a_is_2d || b_is_2d),
              "Have to provide offsets if there is a 2d matrix, or no offset "
              "if both matrices are 3d");

  if (offs.has_value()) {
    TORCH_CHECK(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32");
  }
  TORCH_CHECK(!bias.has_value(), "Bias not supported yet");

  // Use shape inference version of create_grouped_gemm_output_tensor.
  // Note: helper function uses mat_a/mat_b parameter names internally, but we
  // pass self/mat2 to match schema.
  at::Tensor out = create_grouped_gemm_output_tensor_shape_infer(
      self, mat2, offs, out_dtype);

  return out;
}

}  // namespace aotops

}  // namespace torch_gcu
