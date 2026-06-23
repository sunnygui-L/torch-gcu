/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include <ATen/native/SpectralOpsUtils.h>
#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_resize.h"
#include "aten/shape_inference/aotops_shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

namespace {
constexpr int64_t cufft_max_ndim = 3;
}  // namespace

at::Tensor _fft_c2r_shape_infer(const at::Tensor& self, at::IntArrayRef dim,
                                int64_t normalization, int64_t last_dim) {
  TORCH_CHECK(self.is_complex());
  auto in_sizes = self.sizes();
  at::DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim;
  auto output = aotops::empty(
      out_sizes,
      self.options().dtype(c10::toRealValueType(self.scalar_type())));
  return output;
}

at::Tensor& _fft_c2r_out_shape_infer(const at::Tensor& self,
                                     at::IntArrayRef dim, int64_t normalization,
                                     int64_t last_dim, at::Tensor& out) {
  TORCH_CHECK(self.is_complex());
  auto in_sizes = self.sizes();
  at::DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim;
  aotops::resize_output(out, out_sizes);
  return out;
}

at::Tensor& _exec_fft_shape_infer(at::Tensor& out, const at::Tensor& self,
                                  at::IntArrayRef out_sizes,
                                  at::IntArrayRef dim, bool forward) {
  // Shape inference extracted from _exec_fft (SpectralOps.cpp:166-273)
  // This function extracts the shape inference logic without actual computation

  const auto ndim = self.dim();
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;

  // Permute dimensions so batch dimensions come first, and in stride order
  // Extract from _exec_fft (SpectralOps.cpp:174-186)
  at::DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  c10::SmallVector<bool, 8> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }
  auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(),
                     [&](int64_t d) { return !is_transformed_dim[d]; });
  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end, [&](int64_t a, int64_t b) {
    return self_strides[a] > self_strides[b];
  });
  std::copy(dim.cbegin(), dim.cend(), batch_end);
  auto input = self.permute(dim_permute);

  // Collapse batch dimensions into a single dimension
  at::DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1;
  std::copy(input.sizes().cbegin() + batch_dims, input.sizes().cend(),
            batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  const auto batch_size = input.sizes()[0];
  at::DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;
  for (const auto i : c10::irange(signal_ndim)) {
    auto in_size = input.sizes()[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(in_size == signal_size[i + 1] ||
                          in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(out_size == signal_size[i + 1] ||
                          out_size == (signal_size[i + 1] / 2) + 1);
  }

  batched_sizes[0] = batch_size;
  at::DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());
  for (const auto i : c10::irange(dim.size())) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }
  out.resize_(batched_out_sizes, at::MemoryFormat::Contiguous);

  at::DimVector out_strides(ndim);
  int64_t batch_numel = 1;
  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];
    batch_numel *= out_sizes[dim_permute[i]];
  }
  for (const auto i : c10::irange(batch_dims, ndim)) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }
  out.as_strided_(out_sizes, out_strides, out.storage_offset());
  return out;
}

void _fft_fill_with_conjugate_symmetry_shape_infer(const at::Tensor& input,
                                                   at::IntArrayRef dim_) {
  // Shape inference extracted from _fft_fill_with_conjugate_symmetry_
  // (SpectralOps.cpp:1211-1300) This function extracts the shape inference
  // logic without actual computation Note: This is an in-place operation that
  // doesn't change shape, but we keep validation checks

  // Validation and shape-related checks from _fft_fill_with_conjugate_symmetry_
  const auto input_sizes = input.sizes();       // SpectralOps.cpp:1212
  const auto input_strides = input.strides();   // SpectralOps.cpp:1213
  TORCH_CHECK(!dim_.empty());                   // SpectralOps.cpp:1214
  at::DimVector dim(dim_.begin(), dim_.end());  // SpectralOps.cpp:1215
  at::maybe_wrap_dims(dim, input_strides.size(),
                      /*wrap_scalars=*/false);  // SpectralOps.cpp:1216

  // Early return check (SpectralOps.cpp:1218-1220)
  if (input.numel() == 0 || input_sizes[dim.back()] <= 2) {
    // No elements need writing, early return
    return;
  }

  // Small dimensions may be treated as batch dims (SpectralOps.cpp:1222-1227)
  dim.erase(std::remove_if(dim.begin(), dim.end(),
                           [&](int64_t d) { return (input_sizes[d] <= 2); }),
            dim.end());

  // TensorIterator setup (SpectralOps.cpp:1229-1236) - host operation, keep it
  auto iter = at::TensorIteratorConfig()
                  .add_output(input)
                  .add_input(input)
                  .resize_outputs(false)
                  .declare_static_shape(input_sizes, dim)
                  .build();

  const auto iter_strides = iter.strides(0);  // SpectralOps.cpp:1238
  const auto iter_sizes = iter.shape();       // SpectralOps.cpp:1239
  const auto ndim = static_cast<int64_t>(iter_strides.size() +
                                         dim.size());  // SpectralOps.cpp:1240
  at::DimVector in_strides(ndim),
      signal_half_sizes(ndim);  // SpectralOps.cpp:1241

  // Take coalesced batch dimensions from TensorIterator
  // (SpectralOps.cpp:1242-1244)
  std::copy(iter_strides.begin(), iter_strides.end(), in_strides.begin());
  std::copy(iter_sizes.begin(), iter_sizes.end(), signal_half_sizes.begin());

  // Take transformed dimensions directly from the input
  // (SpectralOps.cpp:1246-1252)
  const auto element_size = iter.element_size(0);  // SpectralOps.cpp:1247
  for (const auto i : c10::irange(dim.size())) {
    // Convert to byte strides to match TensorIterator
    in_strides[iter_strides.size() + i] =
        input_strides[dim[i]] * element_size;  // SpectralOps.cpp:1250
    signal_half_sizes[iter_strides.size() + i] =
        input_sizes[dim[i]];  // SpectralOps.cpp:1251
  }

  // For the last dimension, use negative strides to perform the mirroring
  // (SpectralOps.cpp:1254-1257)
  signal_half_sizes.back() =
      (input_sizes[dim.back()] - 1) / 2;  // SpectralOps.cpp:1255
  auto out_strides = in_strides;          // SpectralOps.cpp:1256
  out_strides.back() *= -1;               // SpectralOps.cpp:1257

  // Reorder dimensions by stride to maximize data locality
  // (SpectralOps.cpp:1264-1282)
  at::DimVector dim_permute(ndim);                       // SpectralOps.cpp:1265
  std::iota(dim_permute.begin(), dim_permute.end(), 0);  // SpectralOps.cpp:1266
  std::sort(dim_permute.begin(), dim_permute.end(),      // SpectralOps.cpp:1267
            [&](auto dim1, auto dim2) {
              return in_strides[dim1] <
                     in_strides[dim2];  // SpectralOps.cpp:1269
            });

  at::DimVector temp(ndim);                           // SpectralOps.cpp:1272
  auto apply_permutation = [&](at::DimVector& vec) {  // SpectralOps.cpp:1273
    // Do permuted index copy into a temporary, then copy back
    for (const auto i : c10::irange(ndim)) {
      temp[i] = vec[dim_permute[i]];  // SpectralOps.cpp:1276
    }
    vec = temp;  // SpectralOps.cpp:1278
  };
  apply_permutation(in_strides);         // SpectralOps.cpp:1280
  apply_permutation(out_strides);        // SpectralOps.cpp:1281
  apply_permutation(signal_half_sizes);  // SpectralOps.cpp:1282

  // Find dims.slice(dims.size() - 1) in the new permuted order.
  // These are the dimensions that need explicit Hermitian mirroring
  // (SpectralOps.cpp:1284-1294)
  at::DimVector mirror_dims;            // SpectralOps.cpp:1286
  mirror_dims.reserve(dim.size() - 1);  // SpectralOps.cpp:1287
  for (const auto i : c10::irange(ndim)) {
    if (dim_permute[i] >= static_cast<int64_t>(
                              iter_strides.size()) &&  // Not a batch dimension
        dim_permute[i] != ndim - 1) {  // Not the last dim, which is mirrored
                                       // separately with negative strides
      mirror_dims.push_back(i);        // SpectralOps.cpp:1291
    }
  }
  TORCH_INTERNAL_ASSERT(mirror_dims.size() ==
                        dim.size() - 1);  // SpectralOps.cpp:1294

  // fft_fill_with_conjugate_symmetry_stub() - removed, device kernel launch,
  // doesn't change shape SpectralOps.cpp:1297-1299
}

at::Tensor _fft_c2c_shape_infer(const at::Tensor& self, at::IntArrayRef dim,
                                int64_t normalization, bool forward) {
  // Shape inference extracted from _fft_c2c_cufft (SpectralOps.cpp:445-482)
  // Output shape is same as input (SpectralOps.cpp:451: auto out_sizes =
  // self.sizes())
  TORCH_CHECK(self.is_complex(), "Input must be a complex tensor");
  if (dim.empty()) {
    return self.clone();
  }

  auto out_sizes = self.sizes();
  auto output = at::empty(out_sizes, self.options());

  at::DimVector sorted_dims(dim.begin(), dim.end());
  auto working_tensor = self;
  while (true) {
    // Sort dimensions every time as _exec_fft re-strides the output
    auto strides = working_tensor.strides();
    std::sort(sorted_dims.begin(), sorted_dims.end(),
              [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });

    const auto max_dims =
        std::min(static_cast<size_t>(cufft_max_ndim), sorted_dims.size());
    auto first_dims = at::IntArrayRef(sorted_dims)
                          .slice(sorted_dims.size() - max_dims, max_dims);

    _exec_fft_shape_infer(output, working_tensor, out_sizes, first_dims,
                          forward);
    sorted_dims.resize(sorted_dims.size() - max_dims);

    if (sorted_dims.empty()) {
      break;
    }

    if (working_tensor.is_same(self)) {
      working_tensor = std::move(output);
      output = at::empty(out_sizes, self.options());
    } else {
      std::swap(output, working_tensor);
    }
  }

  return output;
}

at::Tensor& _fft_c2c_out_shape_infer(const at::Tensor& self,
                                     at::IntArrayRef dim, int64_t normalization,
                                     bool forward, at::Tensor& out) {
  auto result = _fft_c2c_shape_infer(
      self, dim, static_cast<int64_t>(at::native::fft_norm_mode::none),
      forward);

  // _fft_apply_normalization_out doesn't change shape, so out has the same
  // shape as result
  aotops::resize_output(out, result.sizes());
  return out;
}

// Helper function to check if optimized path should be used
// Extract from SpectralOps.cpp:306-315
namespace {
bool use_optimized_cufft_path(at::IntArrayRef dim) {
  // For performance reason, when dim starts with (0, 1), do not use the
  // optimized path.
  if (dim.size() > cufft_max_ndim ||
      (dim.size() >= 2 && dim[0] == 0 && dim[1] == 1)) {
    return false;
  } else {
    return true;
  }
}
}  // namespace

at::Tensor _fft_r2c_shape_infer(const at::Tensor& self, at::IntArrayRef dim,
                                int64_t normalization, bool onesided) {
  // Shape inference extracted from _fft_r2c_cufft (SpectralOps.cpp:318-385)
  // This function extracts the shape inference logic without actual computation

  // 1. Input validation (SpectralOps.cpp:319)
  TORCH_CHECK(self.is_floating_point());

  // 2. Calculate onesided sizes (SpectralOps.cpp:320-325)
  auto input_sizes = self.sizes();
  at::DimVector onesided_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  onesided_sizes[last_dim] = last_dim_halfsize;
  at::IntArrayRef out_sizes = onesided ? onesided_sizes : input_sizes;

  // 3. Create output tensor (SpectralOps.cpp:327-328)
  const auto out_options =
      self.options().dtype(c10::toComplexType(self.scalar_type()));
  auto output = aotops::empty(out_sizes, out_options);

  // 4. Alignment check (SpectralOps.cpp:330-339) - host operations, keep them
  // Note: movedim() and clone() are host operations, not device kernel launch
  // They don't change shape but may affect subsequent operations, so we keep
  // them
  const auto complex_size = 2 * self.element_size();
  const bool complex_aligned =
      (reinterpret_cast<std::uintptr_t>(self.const_data_ptr()) % complex_size ==
       0);
  auto working_tensor = self;
  if (!complex_aligned) {
    // movedim() and clone() are host operations, directly use them
    working_tensor = self.movedim(last_dim, -1)
                         .clone(at::MemoryFormat::Contiguous)
                         .movedim(-1, last_dim);
  }

  // 5. Execute FFT shape inference (SpectralOps.cpp:341-370)
  // Both optimized and non-optimized paths call _exec_fft, so we call
  // _exec_fft_shape_infer
  if (use_optimized_cufft_path(dim)) {
    // Optimized path (SpectralOps.cpp:341-342)
    _exec_fft_shape_infer(output, working_tensor, out_sizes, dim,
                          /*forward=*/true);
  } else {
    // Non-optimized path (SpectralOps.cpp:344-370)
    // First do the R2C transform on the last dimension
    {
      auto target_sizes = dim.size() == 1 ? out_sizes : onesided_sizes;
      _exec_fft_shape_infer(output, working_tensor, target_sizes, last_dim,
                            /*forward=*/true);
      if (dim.size() > 1) {
        // Create working tensor for multi-dimensional case
        // (SpectralOps.cpp:349)
        working_tensor = aotops::empty(out_sizes, out_options);
      }
    }

    // Then any remaining C2C transforms (SpectralOps.cpp:353-369)
    at::DimVector sorted_dims(dim.begin(), dim.end() - 1);
    while (!sorted_dims.empty()) {
      std::swap(output, working_tensor);

      // Resort dimensions every time as _exec_fft re-strides the output
      auto strides = working_tensor.strides();
      std::sort(sorted_dims.begin(), sorted_dims.end(),
                [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });

      const auto max_dims =
          std::min(static_cast<size_t>(cufft_max_ndim), sorted_dims.size());
      auto last_dims = at::IntArrayRef(sorted_dims)
                           .slice(sorted_dims.size() - max_dims, max_dims);

      // Intermediate results are always onesided
      _exec_fft_shape_infer(output, working_tensor, onesided_sizes, last_dims,
                            /*forward=*/true);
      sorted_dims.resize(sorted_dims.size() - max_dims);
    }
  }

  // 6. Normalization (SpectralOps.cpp:373-374)
  // slice() is host operation, keep it; _fft_apply_normalization is device
  // operation, remove it
  auto out_slice = output.slice(last_dim, 0, last_dim_halfsize);
  // _fft_apply_normalization(out_slice, normalization, input_sizes, dim); -
  // removed, device operation, doesn't change shape

  // 7. Handle non-onesided case (SpectralOps.cpp:376-383)
  if (!onesided) {
    if (output.sizes()[last_dim] != out_sizes[last_dim]) {
      // Resize working tensor (SpectralOps.cpp:378) - host operation, keep it
      at::Tensor working_tensor = aotops::empty(out_sizes, out_options);
      working_tensor.resize_(out_sizes, at::MemoryFormat::Contiguous);
      // slice() is host operation, keep it; copy_ is device operation, remove
      // it
      auto working_slice = working_tensor.slice(last_dim, 0, last_dim_halfsize);
      // working_slice.copy_(output); - removed, device operation
      output = std::move(working_tensor);
    }

    // Call shape inference for _fft_fill_with_conjugate_symmetry_
    // (SpectralOps.cpp:382)
    _fft_fill_with_conjugate_symmetry_shape_infer(output, dim);
  }

  return output;
}

at::Tensor& _fft_r2c_out_shape_infer(const at::Tensor& self,
                                     at::IntArrayRef dim, int64_t normalization,
                                     bool onesided, at::Tensor& out) {
  // Shape inference extracted from _fft_r2c_cufft_out (SpectralOps.cpp:387-402)
  // This function extracts the shape inference logic without actual computation

  // 1. Call non-out version shape inference (SpectralOps.cpp:389)
  auto result = _fft_r2c_shape_infer(
      self, dim, static_cast<int64_t>(at::native::fft_norm_mode::none),
      /*onesided=*/true);

  // 2. Handle onesided case (SpectralOps.cpp:390-391)
  if (onesided) {
    // _fft_apply_normalization_out doesn't change shape, so out has the same
    // shape as result
    aotops::resize_output(out, result.sizes());
    return out;
  }

  // 3. Handle non-onesided case (SpectralOps.cpp:394-401)
  // Resize output to input sizes (SpectralOps.cpp:394)
  aotops::resize_output(out, self.sizes());

  // Calculate slice parameters (SpectralOps.cpp:396-398)
  auto last_dim = dim.back();
  auto last_dim_halfsize = result.sizes()[last_dim];
  // slice() is host operation, keep it; _fft_apply_normalization_out is device
  // operation, remove it
  auto out_slice = out.slice(last_dim, 0, last_dim_halfsize);
  // _fft_apply_normalization_out(out_slice, result, normalization,
  // self.sizes(), dim); - removed, device operation, doesn't change shape

  // Call shape inference for _fft_fill_with_conjugate_symmetry_
  // (SpectralOps.cpp:400) Call shape inference for
  // _fft_fill_with_conjugate_symmetry_ (SpectralOps.cpp:400)
  _fft_fill_with_conjugate_symmetry_shape_infer(out, dim);

  return out;
}

}  // namespace aotops

}  // namespace torch_gcu
