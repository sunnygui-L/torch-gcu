/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/core/Tensor.h>

#include "aten/shape_inference/gcu_structured.h"

namespace torch_gcu {

namespace aotops {

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t,
             at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_shape_infer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double dropout_p, bool is_causal, bool return_debug_mask,
    c10::optional<double> scale);

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_efficient_attention_shape_infer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const ::std::optional<at::Tensor> &attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, ::std::optional<double> scale);
::std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_backward_shape_infer(
    const at::Tensor &grad_out, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &out, const at::Tensor &logsumexp,
    const at::Tensor &cum_seq_q, const at::Tensor &cum_seq_k, int64_t max_q,
    int64_t max_k, double dropout_p, bool is_causal,
    const at::Tensor &philox_seed, const at::Tensor &philox_offset,
    ::std::optional<double> scale);

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_efficient_attention_backward_shape_infer(
    const at::Tensor &grad_out_, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &attn_bias, const at::Tensor &out,
    const at::Tensor &logsumexp, const at::Tensor &philox_seed,
    const at::Tensor &philox_offset, double dropout_p,
    ::std::array<bool, 4> grad_input_mask, bool causal,
    ::std::optional<double> scale);
}  // namespace aotops

}  // namespace torch_gcu
