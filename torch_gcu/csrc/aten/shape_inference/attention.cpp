/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "aten/shape_inference/attention.h"

#include <ATen/native/transformers/sdp_utils_cpp.h>

#include <optional>
#include <tuple>

#include "ATen/native/transformers/sdp_utils_cpp.h"
#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/shape_inference/aotops_shape_infer_func.h"
#include "c10/core/DeviceType.h"
#include "csrc/gcu/gcu_graphs_utils.h"
#include "csrc/gcu/logging.h"
#include "gcu/gcu_graphs_utils.h"
#include "gcu/gcu_hardware.h"

namespace torch_gcu {

namespace aotops {

namespace {

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_privateuseone(), #x " must be on GCU")
#define CHECK_SHAPE(x, ...)                                \
  TORCH_CHECK(x.sizes() == at::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

inline void validate_sdpa_input(const at::Tensor &query_, const at::Tensor &key,
                                const at::Tensor &value,
                                const c10::optional<at::Tensor> &attn_mask_,
                                double dropout_p, bool is_causal,
                                c10::optional<double> scale) {
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
    TORCH_CHECK(mask_dtype == at::kBool || mask_dtype == query_.dtype(),
                "Expected attn_mask dtype to be bool or to match query dtype, "
                "but got attn_mask.dtype: ",
                mask_dtype, " and  query.dtype: ", query_.dtype(), " instead.");
    TORCH_CHECK(!query_.is_nested() && !key.is_nested(),
                "Scaled_dot_product_attention: Nested tensors for query / key "
                "are not supported "
                "when an explicit attn_mask is set");
  }
  return;
}

// return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p};
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
mha_fwd_shape_infer(const at::Tensor &q, const at::Tensor &k,
                    const at::Tensor &v, const float p_dropout, bool is_causal,
                    int window_size_left, int window_size_right,
                    const bool return_softmax) {
  auto q_dtype = q.dtype();

  TORCH_CHECK(
      q_dtype == at::kHalf || q_dtype == at::kBFloat16 || q_dtype == at::kFloat,
      "FlashAttention only support float32, fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size_og % 8 == 0,
              "head_size must be a multiple of 8, this is ensured by padding!");
  TORCH_CHECK(
      head_size_og <= 256,
      "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }

  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1) {
    is_causal = false;
  }
  if (is_causal) {
    window_size_right = 0;
  }

  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups,
  // nheads_kv, d) in this case H/t Daniel Haziza
  const int seqlenq_ngroups_swapped =
      seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
      window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0;

  at::Tensor temp_q = q;
  if (seqlenq_ngroups_swapped) {
    const int ngroups = num_heads / num_heads_k;
    temp_q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og})
                 .transpose(1, 2);
    seqlen_q = ngroups;
    num_heads = num_heads_k;
  }

  CHECK_SHAPE(temp_q, batch_size, seqlen_q, num_heads, head_size_og);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

  at::Tensor q_padded, k_padded, v_padded;
  q_padded = temp_q;
  k_padded = k;
  v_padded = v;

  at::Tensor out = at::empty_like(temp_q);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  GCUGuard device_guard{(char)q.get_device()};
  auto opts = q.options();

  auto softmax_lse =
      aotops::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

  at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax) {
    TORCH_CHECK(p_dropout > 0.0f,
                "return_softmax is only supported when p_dropout > 0.0");
    p = aotops::empty(
        {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
  }

  at::Tensor rng_state, _unused;
  auto hardware = HardwareType::GetInstance().getHardware();
  if (hardware == BackendType::kS60) {
    // S60 does not support 64 bit wise, use 2x32 instead
    rng_state = at::empty({4}, at::dtype(at::kUInt32).device(at::kPrivateUse1));
  } else if (hardware == BackendType::kL600) {
    rng_state =
        empty_tmp({2}, at::TensorOptions(at::kPrivateUse1).dtype(at::kUInt64));
  }
  _unused = at::empty({}, at::dtype(at::kUInt64).device(at::kPrivateUse1));

  if (seqlenq_ngroups_swapped) {
    out = out.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
    q_padded = q_padded.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
    softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
  }
  return {out,         q_padded,  k_padded, v_padded,
          softmax_lse, rng_state, _unused,  p};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_bwd_shape_infer(
    const at::Tensor
        &dout,              // batch_size x seqlen_q x num_heads, x head_size_og
    const at::Tensor &q,    // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &k,    // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor &v,    // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor &out,  // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &softmax_lse,  // b x h x seqlen_q
    std::optional<at::Tensor>
        &dq_,  // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>
        &dk_,  // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>
        &dv_,  // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>
        &alibi_slopes_,     // num_heads or batch_size x num_heads
    const float p_dropout,  // probability to drop
    const float softmax_scale, const bool is_causal, int window_size_left,
    int window_size_right, const bool deterministic,
    const at::Tensor philox_seed, const at::Tensor philox_offset) {
  if (is_causal) {
    window_size_right = 0;
  }

  bool is_dropout = p_dropout > 0.0;

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype,
              "query and dout must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(out);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(softmax_lse);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1,
              "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1,
              "dout tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads = sizes[2];
  const int head_size_og = dout.size(3);
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(
      head_size_og % 8 == 0,
      "head_size_og should be a multiple of 8, this is ensured by padding!");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention backward only supports head dimension at most 256");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
              "head_size must be head_size_og rounded to a multiple of 8");

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

  at::Tensor dq, dk, dv;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    CHECK_DEVICE(dq);
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
  } else {
    dq = at::empty_like(q);
  }
  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    CHECK_DEVICE(dk);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dk = at::empty_like(k);
  }
  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    CHECK_DEVICE(dv);
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dv = at::empty_like(v);
  }

  auto opts = q.options();
  auto softmax_d = at::empty({batch_size, num_heads, seqlen_q_rounded},
                             opts.dtype(at::kFloat));

  at::Tensor dk_expanded, dv_expanded;
  if (num_heads_k != num_heads) {  // MQA / GQA
    dk_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    dv_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
  } else {
    dk_expanded = dk;
    dv_expanded = dv;
  }

  // For MQA/GQA we need to sum dK and dV across the groups
  if (num_heads_k != num_heads) {
    aotops::sum_out_shape_infer(
        at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k,
                                  num_heads / num_heads_k, head_size}),
        {3}, false, std::nullopt, dk);
    aotops::sum_out_shape_infer(
        at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k,
                                  num_heads / num_heads_k, head_size}),
        {3}, false, std::nullopt, dv);
  }
  return {dq, dk, dv, softmax_d};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_flash_attention_forward_shape_infer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    int64_t max_seqlen_batch_q, int64_t max_seqlen_batch_k, double dropout_p,
    bool is_causal, bool return_debug_mask, c10::optional<double> scale) {
  auto [output, q_padded, k_padded, v_padded, logsumexp, philox_seed,
        philox_offset, debug_attn_mask] =
      mha_fwd_shape_infer(query, key, value, dropout_p, is_causal,
                          -1, /*window_size_left*/
                          -1, /*window_size_right*/
                          return_debug_mask);

  debug_attn_mask =
      return_debug_mask ? debug_attn_mask : at::empty({0}, query.options());

  return std::make_tuple(std::move(output), std::move(logsumexp),
                         std::move(philox_seed), std::move(philox_offset),
                         std::move(debug_attn_mask));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_flash_attention_backward_shape_infer(
    const at::Tensor &grad_out, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &out, const at::Tensor &logsumexp,
    const at::Tensor &cumulative_sequence_length_q,
    const at::Tensor &cumulative_sequence_length_k, int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k, double dropout_p, bool is_causal,
    const at::Tensor &philox_seed, const at::Tensor &philox_offset,
    std::optional<double> scale, std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right) {
  const auto softmax_scale =
      sdp::calculate_scale(query, scale).as_float_unchecked();
  //  CUDA code assumes that dout is contiguous
  auto contiguous_grad_out = grad_out.contiguous();
  auto contiguous_out = out.contiguous();

  const int non_null_window_left =
      window_size_left.has_value() ? window_size_left.value() : -1;
  const int non_null_window_right =
      window_size_right.has_value() ? window_size_right.value() : -1;

  std::optional<at::Tensor> dq{std::nullopt};
  std::optional<at::Tensor> dk{std::nullopt};
  std::optional<at::Tensor> dv{std::nullopt};

  //  The kernel computes regard we will drop for this functions return
  at::Tensor grad_softmax;

  // Currently unused args:
  std::optional<at::Tensor> alibi_slopes{std::nullopt};

  bool determinisitic{false};
  auto &ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "Flash Attention defaults to a non-deterministic algorithm. ",
          "To explicitly enable determinism call "
          "torch.use_deterministic_algorithms(True, warn_only=False).");
    } else {
      determinisitic = true;
    }
  }

  PTCHECK(!cumulative_sequence_length_q.defined())
      << "cumulative_sequence_length_q is not supported for backward pass";

  // Dense forward
  auto [dQuery, dKey, dValue, dSoftmax] = mha_bwd_shape_infer(
      contiguous_grad_out, query, key, value, contiguous_out, logsumexp, dq, dk,
      dv, alibi_slopes, dropout_p, softmax_scale, is_causal,
      non_null_window_left, non_null_window_right, determinisitic, philox_seed,
      philox_offset);
  return std::make_tuple(std::move(dQuery), std::move(dKey), std::move(dValue));
}
}  // namespace

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t,
             at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_shape_infer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double dropout_p, bool is_causal, bool return_debug_mask,
    c10::optional<double> scale) {
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t max_seqlen_batch_k = key.size(2);
  const int64_t max_seqlen_batch_v = value.size(2);

  at::Tensor q_t = query.transpose(1, 2);
  at::Tensor k_t = key.transpose(1, 2);
  at::Tensor v_t = value.transpose(1, 2);

  auto [output, logsumexp, philox_seed, philox_offset, debug_attn_mask] =
      _flash_attention_forward_shape_infer(q_t, k_t, v_t, max_seqlen_batch_q,
                                           max_seqlen_batch_k, dropout_p,
                                           is_causal, return_debug_mask, scale);
  // Reshape output to convert nnz to batch_size and seq_len
  at::Tensor attention = output.transpose(1, 2);

  return std::make_tuple(attention, logsumexp, at::Tensor(), at::Tensor(),
                         max_seqlen_batch_q, max_seqlen_batch_k, philox_seed,
                         philox_offset, debug_attn_mask);
}

#define CHECK_NOSPARSE_CONTIGUOUS_GCU(TENSOR)                              \
  TORCH_CHECK(TENSOR.is_privateuseone(), #TENSOR " must be a GCU tensor"); \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor");     \
  TORCH_CHECK(TENSOR.is_contiguous());

#define CHECK_NOSPARSE_LASTCONTIGUOUS_GCU(TENSOR)                          \
  TORCH_CHECK(TENSOR.is_privateuseone(), #TENSOR " must be a GCU tensor"); \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor");     \
  TORCH_CHECK(TENSOR.stride(-1) == 1,                                      \
              #TENSOR ": last dimension must be contiguous");

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt>
_efficient_attention_forward_shape_infer(
    const at::Tensor &query,                // [b, seqlen, num_heads, K]
    const at::Tensor &key,                  // [b, seqlen, num_heads, K]
    const at::Tensor &value,                // [b, seqlen, num_heads, Kv]
    const std::optional<at::Tensor> &bias,  // [b, num_heads, seqlen, seqlen]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor> &seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor> &seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const std::optional<int64_t> max_seqlen_q_,
    const std::optional<int64_t> max_seqlen_k_,
    double dropout_p,  // attention matrix dropout probability
    int64_t custom_mask_type, bool compute_logsumexp = false,
    std::optional<double> scale = ::std::nullopt,
    const std::optional<at::Tensor> &seqlen_k = {},
    const std::optional<int64_t> window_size = ::std::nullopt) {
  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(1) == value.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  int64_t max_seqlen_q = 0, max_seqlen_k = 0;
  TORCH_CHECK(seqstart_q.has_value() == seqstart_k.has_value());
  if (seqstart_q.has_value()) {
    TORCH_CHECK(seqstart_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_q->dim() == 1 && seqstart_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_GCU((*seqstart_q));
    CHECK_NOSPARSE_CONTIGUOUS_GCU((*seqstart_k));
    TORCH_CHECK(seqstart_q->size(0) == seqstart_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q_.has_value());
    max_seqlen_q = *max_seqlen_q_;
    max_seqlen_k =
        0;  // TODO: is this actually being set inside the kernel anywhere?
            // see https://github.com/pytorch/pytorch/issues/115590s
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  CHECK_NOSPARSE_LASTCONTIGUOUS_GCU(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_GCU(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_GCU(value);

  torch_gcu::GCUGuard device_guard(query.device());

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t num_heads = query.size(-2);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  // undefined tensor
  at::Tensor res;
  at::Tensor logsumexp;
  at::Tensor seed_t, offset_t;

  // const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

  // Note [Seed and Offset Device]
  // If we are currently in graph capture mode, we need to create the seed and
  // offset tensors on the device. This is necessary for GCU graph-safe random
  // number generation, which requires the seed and offset tensors to be single
  // element tensors on device. During graph capture, when the seed and offset
  // tensors are passed the pointers act as scratch space for storing the RNG
  // state for the backwards pass. When calling backwards, we either construct a
  // PhiloxState with the pointers or the actual values. For more information on
  // GCU graph-safe RNG states, see Note [GCU Graph-safe RNG states].
  // const bool in_capture_stream =
  //     torch_gcu::currentStreamCaptureStatus() !=
  //     torch_gcu::CaptureStatus::None;
  // auto device = in_capture_stream ? at::kPrivateUse1 : at::kCPU;
  // if (use_dropout) {
  //   if (in_capture_stream) {
  //     // The seed and offset will be populated by the kernel
  //     // NOTE: GCU do NOT support 64-bit, we use 2 * int32_t to mock int64_t
  //     // NOTE: at:KLong is hacked to be int32_t, see gcu_data_ptr()
  //     seed_t = at::empty({2}, at::dtype(at::kLong).device(device));
  //     offset_t = at::empty({2}, at::dtype(at::kLong).device(device));
  //   } else {
  //     // NOTE: In non-capture mode, torch_gcu record the seed_t and offset_t
  //     in
  //     // aotops::_scaled_dot_product_efficient_attention since we use
  //     // topsatenScaledDotProductEfficientAttentionGetOffset() to get_offset

  //     // auto [seed, offset] = philox::unpack(philox_state);
  //     // const auto options = at::dtype(at::kLong);
  //     // seed_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(seed)),
  //     // options); offset_t =
  //     // at::scalar_tensor(at::Scalar(static_cast<int64_t>(offset)),
  //     options);
  //   }
  // } else {
  //   // Not using dropout
  //   // NOTE: just put undefined seed_t and offset_t to topsaten
  //   // seed_t = at::empty({}, at::dtype(at::kLong).device(device));
  //   // offset_t = at::empty({}, at::dtype(at::kLong).device(device));
  // }

  // NOTE: at:KLong is hacked to be int32_t, see gcu_data_ptr()
  seed_t = at::empty({2}, at::dtype(at::kLong).device(at::kPrivateUse1));
  offset_t = at::empty({2}, at::dtype(at::kLong).device(at::kPrivateUse1));

  res = at::empty({B, M, num_heads, Kv}, query.options());

  logsumexp = at::empty({seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B,
                         num_heads, compute_logsumexp ? max_seqlen_q : 0},
                        query.options().dtype(at::ScalarType::Float));

  return std::make_tuple(
      std::move(res), std::move(logsumexp), std::move(seed_t),
      std::move(offset_t), max_seqlen_q,
      // TODO: why isn't this being set in the kernel?
      max_seqlen_k_.has_value() ? max_seqlen_k_.value() : max_seqlen_k);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_efficient_attention_shape_infer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const ::std::optional<at::Tensor> &attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, ::std::optional<double> scale) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("torch_gcu.sdpa.mem_efficient_attention");
  // Query -> Query(Batch x Q_seq_len x Num_heads x Dim_per_head)
  // Key   -> Key(Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x  Num_heads x Dim_per_head)
  at::Tensor q_t = query.transpose(1, 2);
  at::Tensor k_t = key.transpose(1, 2);
  at::Tensor v_t = value.transpose(1, 2);
  sdp::CustomMaskType custom_mask_type =
      is_causal ? sdp::CustomMaskType::CausalFromTopLeft
                : sdp::CustomMaskType::NoCustomMask;

  auto [attention, log_sumexp, seed, offset, max_seqlen_batch_q,
        max_seqlen_batch_kv] =
      torch_gcu::aotops::_efficient_attention_forward_shape_infer(
          q_t, k_t, v_t, attn_bias, std::nullopt, std::nullopt, std::nullopt,
          std::nullopt, dropout_p, static_cast<int64_t>(custom_mask_type),
          compute_log_sumexp, scale);

  attention = attention.transpose(1, 2);
  return std::make_tuple(std::move(attention), std::move(log_sumexp),
                         std::move(seed), std::move(offset));
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_backward_shape_infer(
    const at::Tensor &grad_out, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &out, const at::Tensor &logsumexp,
    const at::Tensor &cum_seq_q, const at::Tensor &cum_seq_k, int64_t max_q,
    int64_t max_k, double dropout_p, bool is_causal,
    const at::Tensor &philox_seed, const at::Tensor &philox_offset,
    ::std::optional<double> scale) {
  auto q_t = query.transpose(1, 2);
  auto k_t = key.transpose(1, 2);
  auto v_t = value.transpose(1, 2);

  auto grad_out_t = grad_out.transpose(1, 2);
  auto out_t = out.transpose(1, 2);

  auto [grad_q, grad_k, grad_v] = _flash_attention_backward_shape_infer(
      grad_out_t, q_t, k_t, v_t, out_t, logsumexp, cum_seq_q, cum_seq_k, max_q,
      max_k, dropout_p, is_causal, philox_seed, philox_offset, scale,
      std::nullopt, std::nullopt);

  grad_q = grad_q.transpose(1, 2);
  grad_k = grad_k.transpose(1, 2);
  grad_v = grad_v.transpose(1, 2);

  return std::make_tuple(std::move(grad_q), std::move(grad_k),
                         std::move(grad_v));
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_efficient_attention_backward_shape_infer(
    const at::Tensor &grad_out_, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value,
    const std::optional<at::Tensor> &kernel_bias,  // additive attention bias
    const at::Tensor &out,
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor> &cu_seqlens_q_dummy,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor> &cu_seqlens_k_dummy,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_q,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_k, const at::Tensor &logsumexp,
    double dropout_p,  // dropout probability
    const at::Tensor
        &philox_seed,  // seed using for generating random numbers for dropout
    const at::Tensor &philox_offset,  // offset into random number sequence
    int64_t custom_mask_type, const bool bias_requires_grad,
    const std::optional<double> scale = std::nullopt,
    std::optional<int64_t> num_splits_key = std::nullopt,
    const std::optional<int64_t> window_size = std::nullopt,
    const bool shared_storage_dqdkdv = false) {
  if (!grad_out_.defined()) {
    return std::make_tuple(at::Tensor{}, at::Tensor{}, at::Tensor{},
                           at::Tensor{});
  }

  // This path is used when we directly call _efficient_attention_forward
  // from python.
  // This is needed because SaveVariable automatically converts
  // std::optional to undefined tensor
  std::optional<at::Tensor> bias, cu_seqlens_q, cu_seqlens_k;
  bias = kernel_bias.has_value() && !kernel_bias->defined() ? std::nullopt
                                                            : kernel_bias;
  cu_seqlens_q =
      cu_seqlens_q_dummy.has_value() && !cu_seqlens_q_dummy->defined()
          ? std::nullopt
          : cu_seqlens_q_dummy;
  cu_seqlens_k =
      cu_seqlens_k_dummy.has_value() && !cu_seqlens_k_dummy->defined()
          ? std::nullopt
          : cu_seqlens_k_dummy;

  // ndim
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 4);

  // batch size
  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // seqlen
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));
  TORCH_CHECK(query.size(2) == grad_out_.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));
  TORCH_CHECK(value.size(3) == grad_out_.size(3));

  // handle potentially non-contiguous grad_out through a copy
  auto grad_out = grad_out_.contiguous();
  CHECK_NOSPARSE_CONTIGUOUS_GCU(grad_out);

  CHECK_NOSPARSE_LASTCONTIGUOUS_GCU(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_GCU(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_GCU(value);

  TORCH_CHECK(cu_seqlens_q.has_value() == cu_seqlens_k.has_value());
  TORCH_CHECK(!(cu_seqlens_q.has_value() && bias.has_value()),
              "cu seqlen + bias not supported");
  if (cu_seqlens_q.has_value()) {
    TORCH_CHECK(cu_seqlens_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_GCU((*cu_seqlens_q));
    CHECK_NOSPARSE_CONTIGUOUS_GCU((*cu_seqlens_k));
    TORCH_CHECK(cu_seqlens_q->size(0) == cu_seqlens_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q > 0, "max_seqlen_q required with `cu_seqlens_q`");
    TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k required with `cu_seqlens_k`");
    TORCH_CHECK(max_seqlen_k <= key.size(1),
                "Invalid max_seqlen_k:", max_seqlen_k);
    TORCH_CHECK(max_seqlen_q <= query.size(1),
                "Invalid max_seqlen_q:", max_seqlen_q);
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t nH = query.size(2);
  int64_t K = query.size(3);
  int64_t Kv = value.size(3);

  at::Tensor grad_q, grad_k, grad_v, grad_bias;
  if (shared_storage_dqdkdv) {
    // Create one big contiguous chunk
    // This is because q, k and v usually come from a single
    // output of a linear layer that is chunked.
    // Creating the gradients with the right layout saves us
    // a `torch.cat` call in the backward pass
    TORCH_CHECK(query.size(1) == key.size(1),
                "`shared_storage_dqdkdv` is only supported when Q/K/V "
                "have the same sequence length: got ",
                query.size(1), " query tokens and ", key.size(1),
                " key/value tokens");
    TORCH_CHECK(query.size(3) == key.size(3),
                "`shared_storage_dqdkdv` is only supported when Q/K/V "
                "have the same embed dim: got ",
                query.size(3), " for Q, and ", key.size(3), " for K");
    at::Tensor chunk = at::empty({B, M, 3, nH, K}, query.options());
    grad_q = chunk.select(2, 0);
    grad_k = chunk.select(2, 1);
    grad_v = chunk.select(2, 2);
  } else {
    grad_q = at::empty(query.sizes(), query.options());
    grad_k = at::empty(key.sizes(), key.options());
    grad_v = at::empty(value.sizes(), value.options());
  }

  if (bias_requires_grad) {
    // force alignment for the last dim
    // std::vector<int64_t> sz = bias->sizes().vec();
    // int64_t lastDim = sz[sz.size() - 1];
    // int64_t alignTo = 16;
    // sz[sz.size() - 1] = alignTo * ((lastDim + alignTo - 1) / alignTo);
    // grad_bias = at::empty(sz, bias->options())
    //                 .slice(/*dim=*/-1, /*start=*/0, /*end=*/lastDim);
    grad_bias = at::empty(bias->sizes(), bias->options());
  }

  return std::make_tuple(std::move(grad_q), std::move(grad_k),
                         std::move(grad_v), std::move(grad_bias));
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_efficient_attention_backward_shape_infer(
    const at::Tensor &grad_out_, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &attn_bias, const at::Tensor &out,
    const at::Tensor &logsumexp, const at::Tensor &philox_seed,
    const at::Tensor &philox_offset, double dropout_p,
    ::std::array<bool, 4> grad_input_mask, bool causal,
    ::std::optional<double> scale) {
  /**
   * @param grad_out: Input tensor gradient of out
   * shape(batch, o_heads, o_seq_len, head_size).
   * @param query: Input tensor query
   * shape(batch, q_heads, q_seq_len, head_size).
   * @param key: Input tensor key
   * shape(batch, k_heads, k_seq_len, head_size).
   * @param value: Input tensor value
   * shape(batch, v_heads, v_seq_len, head_size).
   * @param attn_bias: Input tensor value
   * shape(batch, q_heads, q_seq_len, k_seq_len).
   * @param out: Input tensor out
   * shape(batch, o_heads, o_seq_len, head_size).
   */
  if (!grad_out_.defined()) {
    return std::make_tuple(at::Tensor{}, at::Tensor{}, at::Tensor{},
                           at::Tensor{});
  }

  auto grad_out = grad_out_.transpose(
      1, 2);  // shape(batch, o_seq_len, o_heads, head_size).
  auto out_t =
      out.transpose(1, 2);  // shape(batch, o_seq_len, o_heads, head_size).
  auto q_t =
      query.transpose(1, 2);  // shape(batch, q_seq_len, q_heads, head_size).
  auto k_t =
      key.transpose(1, 2);  // shape(batch, k_seq_len, k_heads, head_size).
  auto v_t =
      value.transpose(1, 2);  // shape(batch, v_seq_len, v_heads, head_size).

  // This is needed because SaveVariable automatically converts
  // std::optional to undefined tensor
  std::optional<at::Tensor> kernel_bias;
  if (attn_bias.defined()) {
    kernel_bias = attn_bias;
  }

  // Will add with signauter changes for dropout and bias
  // We are only handling Dense inputs, but this should be passed
  // from forward to backward
  int64_t max_seqlen_q = q_t.size(1);
  int64_t max_seqlen_k = k_t.size(1);

  sdp::CustomMaskType custom_mask_type =
      causal ? sdp::CustomMaskType::CausalFromTopLeft
             : sdp::CustomMaskType::NoCustomMask;

  // grad_q: shape(batch, q_seq_len, q_heads, head_size)
  // grad_k: shape(batch, k_seq_len, k_heads, head_size)
  // grad_v: shape(batch, v_seq_len, v_heads, head_size)
  auto [grad_q, grad_k, grad_v, grad_bias] =
      _efficient_attention_backward_shape_infer(
          grad_out, q_t, k_t, v_t, kernel_bias, out_t, std::nullopt,
          std::nullopt, max_seqlen_q, max_seqlen_k, logsumexp, dropout_p,
          philox_seed, philox_offset, static_cast<int64_t>(custom_mask_type),
          grad_input_mask[3], scale,
          std::nullopt);  // num_split_keys

  // grad_q: shape(batch, q_heads, q_seq_len, head_size)
  // grad_k: shape(batch, k_heads, k_seq_len, head_size)
  // grad_v: shape(batch, v_heads, v_seq_len, head_size)

  // std::make_tuple(grad_q.transpose(1, 2), grad_k.transpose(1, 2),
  //                 grad_v.transpose(1, 2), grad_bias);

  grad_q = grad_q.transpose(1, 2);
  grad_k = grad_k.transpose(1, 2);
  grad_v = grad_v.transpose(1, 2);
  return std::make_tuple(std::move(grad_q), std::move(grad_k),
                         std::move(grad_v), std::move(grad_bias));
}

}  // namespace aotops

}  // namespace torch_gcu
