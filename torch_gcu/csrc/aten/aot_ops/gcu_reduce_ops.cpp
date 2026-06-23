/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/native/ReduceOpsUtils.h>

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/gcu_structured_shape_infer.h"
#include "aten/shape_inference/reduce_ops.h"
#include "gcu/gcu_utils.h"
#include "topsaten/topsaten_ops.h"

namespace torch_gcu {

namespace aotops {

namespace {
inline void warn_invalid_degrees_of_freedom(const char* fname,
                                            const at::TensorIterator& iter,
                                            double correction) {
  int64_t reducing_over_num_elements =
      iter.num_output_elements() == 0
          ? 0
          : iter.numel() / iter.num_output_elements();
  if (reducing_over_num_elements - correction <= 0) {
    TORCH_WARN(
        fname,
        "(): degrees of freedom is <= 0. Correction should be strictly less "
        "than the reduction factor (input numel divided by output numel).");
  }
}
}  // namespace

// TODO
// after topsaten fix, delete
at::Tensor& cumprod_out(const at::Tensor& self, int64_t dim,
                        c10::optional<at::ScalarType> dtype, at::Tensor& out) {
  structured_cumprod_gcu_out op(out);
  op.meta(self, dim, dtype);
  auto xdtype = dtype.value_or(op.maybe_get_output(0).scalar_type());
  bridge_topsatenCumprod_out1(op.maybe_get_output(0), self, dim, xdtype);
  return out;
}

::std::tuple<at::Tensor, at::Tensor> std_mean(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction_opt, bool keepdim) {
  auto result = std_mean_shape_infer(self, dim, correction_opt, keepdim);
  auto result1 = std::get<0>(result);
  auto result2 = std::get<1>(result);
  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  at::ScalarType dtype = at::native::get_dtype_from_result(result1, {});
  auto iter = at::native::make_reduction("std_mean", result1, result2, self,
                                         dim, keepdim, dtype);
  warn_invalid_degrees_of_freedom("std_mean", iter, correction);

  if (iter.numel() == 0) {
    // Trivial reduction
    aotops::fill_(result1, std::numeric_limits<double>::quiet_NaN());
    aotops::fill_(result2, std::numeric_limits<double>::quiet_NaN());
  } else {
    bridge_topsatenStdMean_out2(result1, result2, self, dim, correction_opt,
                                keepdim);
  }
  return std::tuple<at::Tensor&, at::Tensor&>(result1, result2);
}

::std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction_opt, bool keepdim) {
  auto result = var_mean_shape_infer(self, dim, correction_opt, keepdim);
  auto result1 = std::get<0>(result);
  auto result2 = std::get<1>(result);
  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  at::ScalarType dtype = at::native::get_dtype_from_result(result1, {});
  auto iter = at::native::make_reduction("std_mean", result1, result2, self,
                                         dim, keepdim, dtype);
  warn_invalid_degrees_of_freedom("var_mean", iter, correction);

  if (iter.numel() == 0) {
    // Trivial reduction
    aotops::fill_(result1, std::numeric_limits<double>::quiet_NaN());
    aotops::fill_(result2, std::numeric_limits<double>::quiet_NaN());
  } else {
    bridge_topsatenVarMean_multi_out(
        std::make_index_sequence<
            std::tuple_size<::std::tuple<at::Tensor, at::Tensor>>::value>{},
        false, std::forward_as_tuple(result1, result2), self, dim,
        correction_opt, keepdim);
  }
  return std::tuple<at::Tensor&, at::Tensor&>(result1, result2);
}

void _cummax_helper(const at::Tensor& self, at::Tensor& values,
                    at::Tensor& indices, int64_t dim) {
  bridge_topsatenCummaxHelper_out2(values, indices, self, dim);
}

at::Tensor bincount(const at::Tensor& self,
                    const ::std::optional<at::Tensor>& weights,
                    int64_t minlength) {
  at::Tensor result;
  bincount_out(self, weights, minlength, result);
  return result;
}

at::Tensor& bincount_out(const at::Tensor& self,
                         const ::std::optional<at::Tensor>& weights,
                         int64_t minlength, at::Tensor& out) {
  // Step 1: Calculate output size (runtime calculation, similar to nonzero's
  // CountNonzero). Must determine output size before calling topsaten API
  // to avoid resize during operator execution.
  //
  // topsatenMax does not support I64 input. When
  // TORCH_GCU_ENABLE_INT64_AND_UINT64 is set, self may be I64. Cast to I32
  // explicitly before calling max() so that topsatenMax always receives an I32
  // tensor regardless of the env var.
  int64_t max_val = 0;

  if (self.numel() > 0) {
    // Cast self to Int32 for topsatenMax (which does not support I64 input).
    const at::Tensor self_i32 = self.scalar_type() == at::ScalarType::Long
                                    ? self.to(at::ScalarType::Int)
                                    : self;
    auto max_tensor = self_i32.max();
    max_val = static_cast<int64_t>(max_tensor.item<int32_t>());
  }

  // Calculate output size.
  // When input is empty, max_val stays 0 but the correct output size is
  // minlength (not max_val+1=1). Use numel check to distinguish the two cases.
  int64_t nbins =
      (self.numel() == 0) ? minlength : std::max(max_val + 1, minlength);

  // Step 2: Pre-resize output (fully aligned with nonzero pattern)
  // Key: Output size must be determined before calling topsaten API
  std::vector<int64_t> out_shape{nbins};
  if (!out.defined()) {
    // Determine output type
    at::ScalarType out_dtype = at::ScalarType::Long;
    if (weights.has_value() && weights->defined()) {
      out_dtype = weights->scalar_type();
    }
    // Allocate output via aotops::empty (bypasses PyTorch dispatcher), then
    // zero-fill via aotops::fill_ -> bridge_topsatenFill__out1.
    // topsatenBincount uses atomic-add internally so the buffer must be
    // pre-zeroed; also handles the empty-input early-return case where topsaten
    // writes nothing and we need all counts = 0.
    // Note: topsatenZero is NOT used here because it calls topsatenZeros
    // internally which does not support I64 dtype (TOPSATEN_STATUS_NOT_SUPPORT).
    out = aotops::empty(out_shape, self.options().dtype(out_dtype));
    aotops::fill_(out, 0);
  } else {
    // Validate output type
    at::ScalarType expected_dtype = at::ScalarType::Long;
    if (weights.has_value() && weights->defined()) {
      expected_dtype = weights->scalar_type();
    }
    TORCH_CHECK(out.scalar_type() == expected_dtype,
                "bincount: output dtype mismatch");
    // Dynamically resize, then zero-fill: topsatenBincount uses atomic-add
    // internally, so the output buffer must be zeroed before the call.
    aotops::resize_output(out, out_shape);
    out.zero_();
  }

  // Step 3: Call topsatenBincount.
  // Per topsaten source (op_aten_bincount.cc):
  //   - output dtype: must be I64 or FP32 (I32 causes BAD_PARAM)
  //   - weights dtype: must be FP32 when present
  //   - no weights: pass TOPSATEN_DATA_NONE via SetTensorDataType on a real
  //     topsatenTensor (null handle causes BAD_PARAM internally)
  //
  // Requires TORCH_GCU_ENABLE_INT64_AND_UINT64=1 in the environment so that
  // the Long tensor allocated for topsaten_out keeps I64 dtype on GCU (without
  // the env var, GCU narrows Long to I32 which topsaten rejects). The result is
  // then copied back to the user-visible `out` (I32) via out.copy_().
  if (out.numel() > 0) {
    const bool has_weights = weights.has_value() && weights->defined();

    // Always build a real topsatenTensor for weights (valid shape + memory).
    // For the no-weights case, allocate a same-shape Float placeholder via
    // aotops::empty (bypasses dispatcher) and override its dtype to
    // TOPSATEN_DATA_NONE after construction. No zeroing needed: topsaten does
    // not read weights data when dtype is TOPSATEN_DATA_NONE.
    at::Tensor weights_tensor =
        has_weights
            ? weights.value()
            : aotops::empty({self.numel()},
                            self.options().dtype(at::ScalarType::Float));

    // topsatenBincount requires output dtype I64 or FP32.
    // No-weights case: allocate a Long tensor on GCU via aotops::empty
    // (bypasses dispatcher). With TORCH_GCU_ENABLE_INT64_AND_UINT64=1 the
    // Long->I32 narrowing is suppressed and the tensor keeps I64 dtype, which
    // topsaten accepts. With-weights case: out is already FP32, pass directly.
    at::Tensor topsaten_out =
        has_weights
            ? out
            : aotops::empty(out_shape,
                            self.options().dtype(at::ScalarType::Long));

    auto x_out = createTopsatenTensor(topsaten_out);
    auto x_self = createTopsatenTensor(self);
    auto x_weights = createTopsatenTensor(weights_tensor);
    if (!has_weights) {
      x_weights.SetTensorDataType(TOPSATEN_DATA_NONE);
    }
    auto stream = getCurrentGCUStream();

    // topsatenBincount uses atomic-add internally so topsaten_out must be
    // pre-zeroed. Use aotops::fill_ -> bridge_topsatenFill__out1 which handles
    // the GCU Long->I32 narrowing correctly (topsatenZero/topsatenZeros do not
    // support I64 and would return TOPSATEN_STATUS_NOT_SUPPORT).
    if (!has_weights) {
      aotops::fill_(topsaten_out, 0);
    }

    auto op_info = [&self, &topsaten_out, &x_weights, &minlength, &stream,
                    has_weights]() {
      std::stringstream ss;
      ss << "topsatenBincount"
         << ": {\n"
         << "self: " << tensorToString(self) << "\n"
         << "out: " << tensorToString(topsaten_out) << "\n"
         << "weights dtype: "
         << (has_weights ? std::to_string(x_weights.GetTensorDataType())
                         : "TOPSATEN_DATA_NONE(-1)")
         << "\n"
         << "minlength: " << minlength << "\n"
         << "stream: " << stream << "\n"
         << "}\n";
      return ss.str();
    };
    PTDLOG(OP) << op_info();
    CHECK_TOPSATEN_CALL(
        topsaten::topsatenBincount(x_out, x_self, x_weights, minlength, stream),
        op_info);
    maybeGCUStreamSynchronize(stream);

    // Copy result from int64 intermediate buffer back to the user-visible
    // out tensor (which is Int32 due to GCU Long narrowing).
    if (!has_weights) {
      out.copy_(topsaten_out);
    }
  }

  return out;
}

}  // namespace aotops

}  // namespace torch_gcu
