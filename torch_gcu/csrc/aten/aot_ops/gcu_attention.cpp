#include <ATen/Context.h>
#include <ATen/OpMathType.h>
#include <topsaten/topsaten_ops.h>

#include <tuple>
#include <utility>

#include "ATen/native/transformers/sdp_utils_cpp.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/attention.h"
#include "c10/core/ScalarType.h"
#include "csrc/gcu/gcu_graphs_utils.h"
#include "csrc/gcu/logging.h"
#include "gcu/gcu_generator_impl.h"

namespace torch_gcu {

namespace aotops {

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
             c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention(const at::Tensor &query,
                                    const at::Tensor &key,
                                    const at::Tensor &value, double dropout_p,
                                    bool is_causal, bool return_debug_mask,
                                    c10::optional<double> scale) {
  // 1. shape infer
  at::Tensor attention, log_sumexp, cumulative_sequence_length_q,
      cumulative_sequence_length_k, rng_state, _unused, debug_attn_mask;
  int64_t max_seqlen_batch_q, max_seqlen_batch_k;
  std::tie(attention, log_sumexp, cumulative_sequence_length_q,
           cumulative_sequence_length_k, max_seqlen_batch_q, max_seqlen_batch_k,
           rng_state, _unused, debug_attn_mask) =
      _scaled_dot_product_flash_attention_shape_infer(
          query, key, value, dropout_p, is_causal, return_debug_mask, scale);

  PhiloxGcuState state;
  if (dropout_p > 0.0) {
    // 2. get offset
    auto gen =
        getDefaultGCUGenerator(query.get_device()).get<GCUGeneratorImpl>();
    uint64_t offset = 0;
    auto tops_query = topsaten_variable(query).value;
    auto tops_key = topsaten_variable(key).value;
    auto tops_value = topsaten_variable(value).value;
    CHECK_TOPSATEN_CALL(
        topsaten::topsatenScaledDotProductFlashAttentionGetOffset(
            offset, tops_query, tops_key, tops_value));
    state = gen->philox_gcu_state(offset);
  }

  // 3. call topsaten op
  bool is_capturing =
      torch_gcu::currentStreamCaptureStatus() != torch_gcu::CaptureStatus::None;
  bool is_dropout = dropout_p > 0.0;

  auto rng_state_tops = createTopsatenTensor_tmp(rng_state);
  auto _unused_tops = createTopsatenTensor_tmp(_unused);
  bridge_topsatenScaledDotProductFlashAttention_multi_out(
      std::make_index_sequence<std::tuple_size<::std::tuple<
          at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
          c10::SymInt, at::Tensor, at::Tensor, at::Tensor>>::value>{},
      false,
      std::forward_as_tuple(attention, log_sumexp, cumulative_sequence_length_q,
                            cumulative_sequence_length_k, max_seqlen_batch_q,
                            max_seqlen_batch_k, rng_state_tops, _unused_tops,
                            debug_attn_mask),
      query, key, value, dropout_p, is_causal, return_debug_mask, scale, state);
  if (!is_capturing && is_dropout) {
    rng_state = at::empty({2}, at::dtype(c10::kUInt64).device(at::kCPU));
    rng_state[0] = static_cast<uint64_t>(state.seed_.val);
    rng_state[1] = static_cast<uint64_t>(state.offset_.val);
    _unused = at::empty({}, at::dtype(at::kUInt64).device(at::kCPU));
  }
  return std::make_tuple(attention, log_sumexp, cumulative_sequence_length_q,
                         cumulative_sequence_length_k, max_seqlen_batch_q,
                         max_seqlen_batch_k, rng_state, _unused,
                         debug_attn_mask);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_efficient_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const ::std::optional<at::Tensor> &attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, ::std::optional<double> scale) {
  // 1. shape infer
  at::Tensor attention, log_sumexp, philox_seed, philox_offset;
  std::tie(attention, log_sumexp, philox_seed, philox_offset) =
      _scaled_dot_product_efficient_attention_shape_infer(
          query, key, value, attn_bias, compute_log_sumexp, dropout_p,
          is_causal, scale);

  PhiloxGcuState state;
  if (dropout_p > 0.0) {
    // 2. get offset and update generator state
    auto gen =
        getDefaultGCUGenerator(query.get_device()).get<GCUGeneratorImpl>();
    uint64_t offset = 0;
    auto tops_query = topsaten_variable(query).value;
    auto tops_key = topsaten_variable(key).value;
    auto tops_value = topsaten_variable(value).value;
    CHECK_TOPSATEN_CALL(
        topsaten::topsatenScaledDotProductEfficientAttentionGetOffset(
            offset, tops_query, tops_key, tops_value));
    PhiloxGcuState state = gen->philox_gcu_state(offset);
  }

  // 3. call topsaten op
  bool is_capturing =
      torch_gcu::currentStreamCaptureStatus() != torch_gcu::CaptureStatus::None;
  bool is_dropout = dropout_p > 0.0;
  bridge_topsatenScaledDotProductEfficientAttention_multi_out(
      std::make_index_sequence<std::tuple_size<
          std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>::value>{},
      false,
      std::forward_as_tuple(attention, log_sumexp, philox_seed, philox_offset),
      query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal,
      scale, state);
  if (!is_capturing && is_dropout) {
    // NOTE: In non-capturing mode, torch_gcu is responsible for recording seed
    // and offset since we passed undefined tensor to topsaten, see
    // aot_ops::_efficient_attention_forward_shape_infer
    philox_seed =
        at::scalar_tensor(at::Scalar(static_cast<uint64_t>(state.seed_.val)),
                          at::dtype(at::kUInt64));
    philox_offset =
        at::scalar_tensor(at::Scalar(static_cast<uint64_t>(state.offset_.val)),
                          at::dtype(at::kUInt64));
  }
  return std::forward_as_tuple(attention, log_sumexp, philox_seed,
                               philox_offset);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_backward(
    const at::Tensor &grad_out, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &out, const at::Tensor &logsumexp,
    const at::Tensor &cum_seq_q, const at::Tensor &cum_seq_k, int64_t max_q,
    int64_t max_k, double dropout_p, bool is_causal,
    const at::Tensor &philox_seed, const at::Tensor &philox_offset,
    ::std::optional<double> scale) {
  if (!grad_out.defined()) {
    return std::make_tuple(at::Tensor{}, at::Tensor{}, at::Tensor{});
  }
  at::Tensor grad_query, grad_key, grad_value;
  int64_t max_seqlen_batch_q, max_seqlen_batch_k;
  std::tie(grad_query, grad_key, grad_value) =
      _scaled_dot_product_flash_attention_backward_shape_infer(
          grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k,
          max_q, max_k, dropout_p, is_causal, philox_seed, philox_offset,
          scale);
  EFCHECK(philox_seed.device() == philox_offset.device())
      << "philox_seed and philox_offset must be on the same device.";

  // Read deterministic setting
  bool deterministic{false};
  auto &ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "Flash Attention defaults to a non-deterministic algorithm. ",
          "To explicitly enable determinism call "
          "torch.use_deterministic_algorithms(True, warn_only=False).");
    } else {
      deterministic = true;
    }
  }

  // Sync global state to operator library
  TORCH_WARN(
      "[GCU_DETERMINISTIC] _scaled_dot_product_flash_attention_backward: "
      "Setting deterministic mode to ",
      deterministic);
  topsaten::topsatenSetDeterministicMode(deterministic);

  bool is_cpu_seed = philox_seed.device().is_cpu();
  if (is_cpu_seed) {
    topsatenScalar_t tops_seed, tops_offset;
    tops_seed = {TOPSATEN_DATA_U64, {}};
    memcpy(&tops_seed.ival, philox_seed.data_ptr<uint64_t>(), sizeof(int64_t));
    tops_offset = {TOPSATEN_DATA_U64, {}};
    memcpy(&tops_offset.ival, philox_seed.data_ptr<uint64_t>() + 1,
           sizeof(int64_t));
    bridge_topsatenScaledDotProductFlashAttentionBackward_multi_out(
        std::make_index_sequence<std::tuple_size<
            ::std::tuple<at::Tensor, at::Tensor, at::Tensor>>::value>{},
        false, std::forward_as_tuple(grad_query, grad_key, grad_value),
        grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k,
        max_q, max_k, dropout_p, is_causal, tops_seed, tops_offset, scale);
  } else {
    auto philox_seed_tops = createTopsatenTensor_tmp(philox_seed);
    auto philox_offset_tops = createTopsatenTensor_tmp(philox_offset);
    bridge_topsatenScaledDotProductFlashAttentionBackward_multi_out(
        std::make_index_sequence<std::tuple_size<
            ::std::tuple<at::Tensor, at::Tensor, at::Tensor>>::value>{},
        false, std::forward_as_tuple(grad_query, grad_key, grad_value),
        grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k,
        max_q, max_k, dropout_p, is_causal, philox_seed_tops,
        philox_offset_tops, scale);
  }
  return std::make_tuple(grad_query, grad_key, grad_value);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_efficient_attention_backward(
    const at::Tensor &grad_out_, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &attn_bias, const at::Tensor &out,
    const at::Tensor &logsumexp, const at::Tensor &philox_seed,
    const at::Tensor &philox_offset, double dropout_p,
    ::std::array<bool, 4> grad_input_mask, bool causal,
    ::std::optional<double> scale) {
  if (!grad_out_.defined()) {
    return std::make_tuple(at ::Tensor{}, at::Tensor{}, at::Tensor{},
                           at::Tensor{});
  }

  // Shape inference
  // grad_q: shape(batch, q_heads, q_seq_len, head_size)
  // grad_k: shape(batch, k_heads, k_seq_len, head_size)
  // grad_v: shape(batch, v_heads, v_seq_len, head_size)

  auto [grad_q, grad_k, grad_v, grad_bias] =
      _scaled_dot_product_efficient_attention_backward_shape_infer(
          grad_out_, query, key, value, attn_bias, out, logsumexp, philox_seed,
          philox_offset, dropout_p, grad_input_mask, causal, scale);

  // Call topsaten op
  EFCHECK(philox_seed.device() == philox_offset.device())
      << "philox_seed and philox_offset must be on the same device.";
  bool is_cpu_seed = philox_seed.device().is_cpu();

  // Get deterministic flag from global context
  // Reference: aten/src/ATen/native/transformers/cuda/attention_backward.cu
  bool deterministic{false};
  auto &ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "Scaled Dot Product Efficient Attention defaults to a "
          "non-deterministic algorithm. ",
          "To explicitly enable determinism call "
          "torch.use_deterministic_algorithms(True, warn_only=False).");
    } else {
      deterministic = true;
    }
  }

  // Sync global state to operator library
  TORCH_WARN(
      "[GCU_DETERMINISTIC] _scaled_dot_product_efficient_attention_backward: "
      "Setting deterministic mode to ",
      deterministic);
  topsaten::topsatenSetDeterministicMode(deterministic);
  // topsaten::topsatenSetDeterministicMode
  if (is_cpu_seed) {
    topsatenScalar_t tops_seed, tops_offset;
    tops_seed = {TOPSATEN_DATA_U64, {}};
    memcpy(&tops_seed.ival, philox_seed.data_ptr<uint64_t>(), sizeof(int64_t));
    tops_offset = {TOPSATEN_DATA_U64, {}};
    memcpy(&tops_offset.ival, philox_offset.data_ptr<uint64_t>(),
           sizeof(int64_t));
    bridge_topsatenScaledDotProductEfficientAttentionBackward_multi_out(
        std::make_index_sequence<std::tuple_size<std::tuple<
            at::Tensor, at::Tensor, at::Tensor, at::Tensor>>::value>{},
        false, std::forward_as_tuple(grad_q, grad_k, grad_v, grad_bias),
        grad_out_, query, key, value, attn_bias, out, logsumexp, tops_seed,
        tops_offset, dropout_p, grad_input_mask, causal, scale, deterministic);
  } else {
    bridge_topsatenScaledDotProductEfficientAttentionBackward_multi_out(
        std::make_index_sequence<std::tuple_size<std::tuple<
            at::Tensor, at::Tensor, at::Tensor, at::Tensor>>::value>{},
        false, std::forward_as_tuple(grad_q, grad_k, grad_v, grad_bias),
        grad_out_, query, key, value, attn_bias, out, logsumexp, philox_seed,
        philox_offset, dropout_p, grad_input_mask, causal, scale,
        deterministic);
  }

  return std::make_tuple(grad_q, grad_k, grad_v, grad_bias);
}

}  // namespace aotops

}  // namespace torch_gcu
