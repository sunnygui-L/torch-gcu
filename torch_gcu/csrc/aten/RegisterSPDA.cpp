/*
 * Copyright 2024 Enflame. All Rights Reserved.
 */

#include <ATen/NestedTensorImpl.h>
#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <c10/util/Array.h>
#include <torch/library.h>

#include <iostream>

#include "aten/aot_ops/gcu_op_check.h"
#include "aten/torch_log_block.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

namespace {

constexpr int32_t num_backends = 4;
std::array<sdp::SDPBackend, num_backends> priority_order(
    sdp::sdp_params const& params) {
  constexpr std::array<sdp::SDPBackend, num_backends> default_order{
      sdp::SDPBackend::flash_attention, sdp::SDPBackend::efficient_attention,
      sdp::SDPBackend::math};
  return default_order;
}

bool check_all_tensors_on_device(sdp::sdp_params const& params, bool debug) {
  // Check that all tensors are on the GPU device
  // This should be handled by the stub dispatch, but when call
  // can_use_*_attention directly from python we need to ensure that the tensors
  // are on gcu
  if (params.query.device().type() != at::DeviceType::PrivateUse1) {
    if (debug) {
      TORCH_WARN("All tensors need to be on gcu device. Got query on device: ",
                 params.query.device(),
                 ", key on device: ", params.key.device(),
                 ", value on device: ", params.value.device());
    }
    return false;
  }
  return true;
}

bool check_head_dim_size_flash(sdp::sdp_params const& params, bool debug) {
  // All head_dim sizes must be equal and less than 256
  const auto max_size = c10::SymInt(256);
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last <= max_size))) {
    if (debug) {
      TORCH_WARN(
          "Flash attention requires q,k,v to have the same last dimension and "
          "to be less than or equal to 256.",
          " Got Query.size(-1): ", query_size_last,
          ", Key.size(-1): ", key_size_last,
          ", Value.size(-1): ", value_size_last, " instead.");
    }
    return false;
  }
  return true;
}

bool check_head_dim_size_flash_nested(sdp::sdp_params const& params,
                                      bool debug) {
  const auto max_size = c10::SymInt(256);
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last % 8 == 0) &&
        (query_size_last <= max_size))) {
    if (debug) {
      TORCH_WARN(
          "For NestedTensor inputs, Flash attention requires q,k,v to have the "
          "same last dimension and to be a multiple of 8 and less than or "
          "equal to 256.",
          " Got Query.size(-1): ", query_size_last,
          ", Key.size(-1): ", params.key.sym_size(-1),
          ", Value.size(-1): ", params.value.sym_size(-1), " instead.");
    }
    return false;
  }
  return true;
}

bool check_flash_causal_non_square_seqlens(sdp::sdp_params const& params,
                                           bool debug) {
  // FlashAttention 2 updated the default mask meaning for causal in this PR:
  // 9e5e8bc91e it is now aligned to lower_right which would be a BC break
  // for non-square masks. We will not support non-square masks for causal w/
  // FAV2
  if (params.is_causal && !params.query.is_nested() &&
      !params.key.is_nested() &&
      params.query.sym_size(-2) != params.key.sym_size(-2)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention does not support the is_causal flag when seqlen_q "
          "!= seqlen_k. ",
          "Got seqlen_q: ", params.query.sym_size(-2),
          " seqlen_k: ", params.key.sym_size(-2),
          ". If you would like to use causal attention with non-square masks, "
          "please see CausalAttnMask.");
    }
    return false;
  }
  return true;
}

bool check_dtypes_low_precision(sdp::sdp_params const& params, bool debug) {
  auto hardware = HardwareType::GetInstance().getHardware();
  if (hardware == BackendType::kS60) {
    // S60 supports float, half and bfloat16
    constexpr auto s60_dtypes =
        c10::array_of<at::ScalarType>(at::kFloat, at::kHalf, at::kBFloat16);
    return check_tensor_dtype(params, s60_dtypes, debug);
  } else {
    constexpr auto default_dtypes =
        c10::array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
    return check_tensor_dtype(params, default_dtypes, debug);
  }
}

bool can_use_flash_attention(sdp::sdp_params const& params, bool debug) {
  // Define gate functions that determine if a flash kernel can be ran
  // Replace with std::to_array when we migrate to c++20
  constexpr auto general_constraints =
      c10::array_of<bool (*)(sdp::sdp_params const&, bool)>(
          sdp::check_runtime_disabled_flash, check_all_tensors_on_device,
          sdp::check_tensor_shapes, sdp::check_for_attn_mask,
          check_head_dim_size_flash, check_flash_causal_non_square_seqlens,
          check_dtypes_low_precision);
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints =
        c10::array_of<bool (*)(sdp::sdp_params const&, bool)>(
            sdp::check_batch_size_nested, check_head_dim_size_flash_nested,
            sdp::check_for_seq_len_0_nested_tensor);
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints =
        c10::array_of<bool (*)(sdp::sdp_params const&, bool)>(
            sdp::check_batch_size_and_num_heads_dense<
                true /*supports_grouped_query_attention=*/>,
            sdp::check_nonzero_sequence_lengths_dense,
            sdp::check_last_dim_stride_equals_1_dense<
                true /*ignore_singleton_dim=*/>);
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  return true;
}

bool can_use_mem_efficient_attention(sdp::sdp_params const& params,
                                     bool debug) {
  //  Define gate functions that determine if a mem efficient kernel can be ran
  constexpr auto general_constraints =
      c10::array_of<bool (*)(sdp::sdp_params const&, bool)>(
          sdp::check_runtime_disabled_mem_efficient,
          check_all_tensors_on_device, sdp::check_tensor_shapes);
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints =
        c10::array_of<bool (*)(sdp::sdp_params const&, bool)>(
            sdp::check_requires_grad_and_nested, sdp::check_batch_size_nested,
            sdp::check_for_seq_len_0_nested_tensor);
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints =
        c10::array_of<bool (*)(sdp::sdp_params const&, bool)>(
            sdp::check_nonzero_sequence_lengths_dense,
            sdp::check_last_dim_stride_equals_1_dense<
                false /*ignore_singleton_dim=*/>,
            sdp::check_batch_size_and_num_heads_dense<
                false /*supports_grouped_query_attention=*/>);
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  return true;
}

}  // anonymous namespace

sdp::SDPBackend select_sdp_backend(sdp::sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP()) {
    return sdp::SDPBackend::error;
  }
  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case sdp::SDPBackend::flash_attention:
        if (torch_gcu::can_use_flash_attention(kernel_params, print_debug)) {
          return sdp::SDPBackend::flash_attention;
        }
        break;
      case sdp::SDPBackend::efficient_attention:
        if (torch_gcu::can_use_mem_efficient_attention(kernel_params,
                                                       print_debug)) {
          return sdp::SDPBackend::efficient_attention;
        }
        break;
      case sdp::SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return sdp::SDPBackend::math;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_flash_attention or use_mem_efficient did not satisfy the
  // constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("Memory efficient kernel not used because:");
  torch_gcu::can_use_mem_efficient_attention(kernel_params, print_debug);
  TORCH_WARN("Flash attention kernel not used because:");
  torch_gcu::can_use_flash_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
  return sdp::SDPBackend::error;
}

}  // namespace torch_gcu

namespace at {
// NB: TORCH_LIBRARY_IMPL must be in an anonymous namespace to avoid
// ambiguity with conflicting identifiers that may have been defined in
// at namespace already.

namespace {

int64_t _fused_sdp_choice_gcu(const Tensor& query_, const Tensor& key,
                              const Tensor& value,
                              const c10::optional<Tensor>& attn_mask_,
                              double dropout_p, bool is_causal,
                              c10::optional<double> scale, bool enable_gqa) {
  sdp::sdp_params kernel_params{query_,    key,       value,     attn_mask_,
                                dropout_p, is_causal, enable_gqa};
  auto backend = torch_gcu::select_sdp_backend(kernel_params);
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false, "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused "
        "kernels.");
  }
  return static_cast<int64_t>(backend);
}

inline void validate_sdpa_input(const Tensor& query_, const Tensor& key,
                                const Tensor& value,
                                const std::optional<Tensor>& attn_mask_,
                                double dropout_p, bool is_causal,
                                std::optional<double> scale) {
  TORCH_CHECK(query_.dtype() == key.dtype() && query_.dtype() == value.dtype(),
              "Expected query, key, and value to have the same dtype, but got "
              "query.dtype: ",
              query_.dtype(), " key.dtype: ", key.dtype(),
              " and value.dtype: ", value.dtype(), " instead.");
  TORCH_CHECK(
      query_.device() == key.device() && query_.device() == value.device(),
      "Expected query, key, and value to have the same device type, but got "
      "query.device: ",
      query_.device(), " key.device: ", key.device(),
      " and value.device: ", value.device(), " instead.");
  TORCH_CHECK(query_.dim() >= 2 && key.dim() >= 2 && value.dim() >= 2,
              "Expected query, key, and value to all be  at least 2 "
              "dimensional, but got query.dim: ",
              query_.dim(), " key.dim: ", key.dim(),
              " and value.dim: ", value.dim(), " instead.");
  if (attn_mask_.has_value()) {
    auto mask_dtype = attn_mask_->dtype();
    TORCH_CHECK(mask_dtype == at::kBool || mask_dtype == at::kFloat ||
                    mask_dtype == query_.dtype(),
                "Expected attn_mask dtype to be bool or float or to match "
                "query dtype, but got attn_mask.dtype: ",
                mask_dtype, " and  query.dtype: ", query_.dtype(), " instead.");
    TORCH_CHECK(!query_.is_nested() && !key.is_nested(),
                "Scaled_dot_product_attention: Nested tensors for query / key "
                "are not supported "
                "when an explicit attn_mask is set");
  }
  return;
}
// This function is used to produce an attn_mask
// in a standard format that can be consumed by both
// the math and memory efficient attn_mask implementation
//  Args:
//    attn_mask: attn_mask of shape (B, L, S) or (L, S) or (B, N_heads, L, S)
std::optional<Tensor> convert_boolean_attn_mask(
    const std::optional<Tensor>& attn_mask, caffe2::TypeMeta dtype) {
  // Pass through
  if (!attn_mask.has_value()) {
    return std::nullopt;
  }
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask->dtype() == at::kBool) {
    return at::where(
        attn_mask->logical_not(), -std::numeric_limits<double>::infinity(),
        at::scalar_tensor(
            0.0, at::TensorOptions().dtype(dtype).device(attn_mask->device())));
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

// Memory Efficient Attention requires a padded attn mask bias
// This function pads the attn_mask bias to be a multiple of 16
// Then slices the padded bias to the original size
// We apply this function to the top level SDPA so that
// if padding is done it will be tracked for backward automatically

template <int alignment>
bool aligned_tensor(const at::Tensor& tensor) {
  for (const auto i : c10::irange(tensor.dim() - 1)) {
    if (tensor.sym_stride(i) % alignment != 0) {
      return false;
    }
  }
  return tensor.sym_stride(-1) == 1;
}

template <int alignment>
at::Tensor pad_bias(const at::Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  auto pad_count = alignment - (last_dim_size % alignment);
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  return padded_bias.slice_symint(-1, 0, last_dim_size);
}

at::Tensor preprocess_mask(const at::Tensor& mask, const at::Tensor& query,
                           const at::Tensor& key, const at::Tensor& value) {
  constexpr int mem_eff_alignment = 8;
  at::Tensor result_mask = mask;
  if (!aligned_tensor<mem_eff_alignment>(mask)) {
    result_mask = pad_bias<mem_eff_alignment>(mask);
  }
  return result_mask.expand_symint({query.sym_size(0), query.sym_size(1),
                                    query.sym_size(2), key.sym_size(2)});
}
// FlashAttentionV2 requires that head dimension be a multiple of 8
// This was previously done within the kernel, however
// This causes the kernel to maybe alias query, key, value
// So instead we pad the head_dimensions to be a multiple of 8 in the composite
// region
template <int alignment_size, bool slice>
at::Tensor pad_last_dim(const at::Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  if (last_dim_size % alignment_size == 0) {
    return attn_bias;
  }
  auto pad_count = alignment_size - (last_dim_size % alignment_size);
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  if (slice) {
    return padded_bias.slice_symint(-1, 0, last_dim_size);
  }
  return padded_bias;
}

at::Tensor post_process_flash_output(at::Tensor out,
                                     c10::SymInt const& og_size) {
  if (!out.is_nested() && out.sym_size(-1) != og_size) {
    out = out.slice_symint(-1, 0, og_size);
  }
  return out;
}

bool should_compute_logsumexp(const Tensor& query, const Tensor& key,
                              const Tensor& value) {
  const bool any_inputs_require_grad =
      query.requires_grad() || key.requires_grad() || value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  return any_inputs_require_grad && gradmode_enabled;
}

Tensor scaled_dot_product_attention_gcu(const Tensor& query_, const Tensor& key,
                                        const Tensor& value,
                                        const std::optional<Tensor>& attn_mask_,
                                        double dropout_p, bool is_causal,
                                        std::optional<double> scale,
                                        bool enable_gqa) {
  validate_sdpa_input(query_, key, value, attn_mask_, dropout_p, is_causal,
                      scale);
  int64_t choice_int = static_cast<int64_t>(sdp::SDPBackend::math);

  choice_int = _fused_sdp_choice_gcu(query_, key, value, attn_mask_, dropout_p,
                                     is_causal, scale, enable_gqa);

  sdp::SDPBackend backend = static_cast<sdp::SDPBackend>(choice_int);
  switch (backend) {
    case sdp::SDPBackend::flash_attention: {
      c10::SymInt og_size = query_.sym_size(-1);
      Tensor query_padded = pad_last_dim<8, false>(query_);
      Tensor key_padded = pad_last_dim<8, false>(key);
      Tensor value_padded = pad_last_dim<8, false>(value);
      // We need to calculate the scale based off the OG head dim size
      auto og_scale = sdp::calculate_scale(query_, scale);
      auto out_lse_softmax = at::_scaled_dot_product_flash_attention(
          query_padded, key_padded, value_padded, dropout_p, is_causal,
          false /*return_debug_mask*/, og_scale.as_float_unchecked());
      return post_process_flash_output(std::get<0>(out_lse_softmax), og_size);
    }
    case sdp::SDPBackend::efficient_attention: {
      // efficient aten op will handle bool attn_mask
      bool compute_logsumexp = should_compute_logsumexp(query_, key, value);
      // We do not need to pad the attn_mask for the efficient attention
      // if pad the attn_mask we will get an uncontiguous mask and will need
      // do copy in the kernel.
      auto out_and_lse = at::_scaled_dot_product_efficient_attention(
          query_, key, value, attn_mask_, compute_logsumexp, dropout_p,
          is_causal, scale);
      return std::get<0>(out_and_lse);
    }
    case sdp::SDPBackend::math: {
      std::optional<Tensor> attn_mask =
          convert_boolean_attn_mask(attn_mask_, query_.dtype());
      if ((!GradMode::is_enabled() ||
           (!query_.requires_grad() && !key.requires_grad() &&
            !value.requires_grad())) &&
          query_.device().type() == DeviceType::MPS && dropout_p == 0.0 &&
          query_.is_contiguous() && key.is_contiguous() &&
          value.is_contiguous() && !query_.is_nested() && !key.is_nested() &&
          !value.is_nested()) {
        return std::get<0>(at::_scaled_dot_product_attention_math_for_mps(
            query_, key, value, attn_mask, dropout_p, is_causal,
            std::nullopt, /*dropout_mask*/
            scale));
      }
      return std::get<0>(at::_scaled_dot_product_attention_math(
          query_, key, value, attn_mask, dropout_p, is_causal,
          std::nullopt, /*dropout_mask*/
          scale, enable_gqa));
    }
    default:
      TORCH_CHECK(
          false,
          "No viable backend for scaled_dot_product_attention was found.");
      return Tensor();
  }
}

namespace {
at::Tensor wrapper_CompositeImplicitAutograd_gcu__scaled_dot_product_attention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const c10::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, c10::optional<double> scale, bool enable_gqa) {
  // No device check
  // DeviceGuard omitted
  if (query.device().type() == DeviceType::PrivateUse1) {
    auto& cfg = torch_gcu::OpDebugConfig::GetInstance();
    OP_CALLTRACE(cfg, scaled_dot_product_attention)
    PRINT_OP_NAME_WITH_OP_ALL(cfg, scaled_dot_product_attention)
    OP_COMMON_MACRO(query, key, value, attn_mask, dropout_p, is_causal, scale,
                    enable_gqa)
    static bool enable_fallback =
        cfg.enableFallback("scaled_dot_product_attention");
    static bool disable_fallback =
        cfg.disableFallback("scaled_dot_product_attention");
    bool fallback_scope = cfg.inFallbackScope();
    bool fallback_cpu =
        (enable_fallback || fallback_scope) && (!disable_fallback);
    if (fallback_cpu) {
      return at::native::call_fallback_fn<
          &torch_gcu::gcu_cpu_fallback,
          ATEN_OP(scaled_dot_product_attention)>::call(query, key, value,
                                                       attn_mask, dropout_p,
                                                       is_causal, scale,
                                                       enable_gqa);
    }
    static bool enable_op_check =
        cfg.enableOpCheck("scaled_dot_product_attention");
    static bool disable_op_check =
        cfg.disableOpCheck("scaled_dot_product_attention");
    bool op_check_scope = cfg.inOpCheckScope();
    bool op_check = (enable_op_check || op_check_scope) && (!disable_op_check);
    // NOTE: when dropout_p not equals, op check will be skipped because cpu and
    // gcu use different random algorithm and the result will be different
    bool no_dropout = (std::abs(dropout_p) < 1e-9);
    if (op_check && no_dropout) {
      OP_CHECK_INPUT_INFO_RECOED(query, key, value, attn_mask, dropout_p,
                                 is_causal, scale, enable_gqa)
      auto clone_input =
          torch_gcu::clone_args(query, key, value, attn_mask, dropout_p,
                                is_causal, scale, enable_gqa);
      auto clone_op_check_input =
          torch_gcu::clone_args(query, key, value, attn_mask, dropout_p,
                                is_causal, scale, enable_gqa);
      auto&& xdevice_out =
          at::native::call_fallback_fn<&torch_gcu::gcu_opcheck_run,
                                       ATEN_OP(scaled_dot_product_attention)>::
              call(std::get<0>(clone_op_check_input),
                   std::get<1>(clone_op_check_input),
                   std::get<2>(clone_op_check_input),
                   std::get<3>(clone_op_check_input),
                   std::get<4>(clone_op_check_input),
                   std::get<5>(clone_op_check_input),
                   std::get<6>(clone_op_check_input),
                   std::get<7>(clone_op_check_input));
      auto gcu_out = scaled_dot_product_attention_gcu(
          query, key, value, attn_mask, dropout_p, is_causal, scale,
          enable_gqa);
      auto result = torch_gcu::gcu_out_check(
          gcu_out, xdevice_out, std::string("scaled_dot_product_attention"));
      if (result.acc_pass && !cfg.enableTestMode()) {
        PTDLOG(OP) << result.check_info;
      } else {
        OP_CHECK_DEBUG_INFO(cfg, ss, gcu_out, xdevice_out, clone_input)
      }
      return gcu_out;
    } else {
      if (op_check) {
        PTDLOG(OP) << "dropout_p not equals to 0.0, skip op check";
      }
      return scaled_dot_product_attention_gcu(query, key, value, attn_mask,
                                              dropout_p, is_causal, scale,
                                              enable_gqa);
    }
  }
  return at::native::scaled_dot_product_attention(query, key, value, attn_mask,
                                                  dropout_p, is_causal, scale);
}
}  // anonymous namespace

TORCH_LIBRARY_IMPL(aten, CompositeImplicitAutograd, m) {
  SuppressTorchWarn suppress_torch_warn;
  m.impl(
      "scaled_dot_product_attention",
      TORCH_FN(
          wrapper_CompositeImplicitAutograd_gcu__scaled_dot_product_attention));
};

}  // namespace

}  // namespace at