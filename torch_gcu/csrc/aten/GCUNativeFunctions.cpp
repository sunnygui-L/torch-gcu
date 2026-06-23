
/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "aten/GCUNativeFunctions.h"

#include <ATen/native/ForeachUtils.h>
#include <ATen/record_function.h>
#include <tops/tops_ext.h>

#include "ATen/ops/_scaled_dot_product_flash_attention_for_cpu_ops.h"
#include "aten/aot_ops/gcu_aot_ops.h"
#include "aten/aot_ops/gcu_op_check.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/aten_cpu_fallback.h"
#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_hooks.h"
#include "gcu/gcu_macros.h"

// [Implementation Guidelines]
// - If you want to call a at::func which doesn't have a kernel registered
// according to gcu_native_functions.yaml,
//   you can call a boxed CPU fallback kernel instead.
//   E.g. don't call tensor.op() or at::op(tensor).
//   use at::native::call_fallback_fn<&gcu_cpu_fallback,
//         ATEN_OP2(op_name, overload_name)>::call(args...)
//   ATEN_OP accepts an operator name without an overload, and
//   ATEN_OP2 accepts an operator name along with its overload name.
//   The description of these acros can be found in
//   https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/templates/Operators.h
//   (You can find some examples below)
namespace torch_gcu {

// // NOTE:Override this because pytorch hardcode this function
// // pytorch/c10/core/TensorImpl.h Do not has PrivateUse1 in dense key_set
// bool GCUNativeFunctions::_has_compatible_shallow_copy_type(
//     const at::Tensor &self, const at::Tensor &from) {
//   c10::DispatchKeySet self_key = self.key_set();
//   c10::DispatchKeySet from_key = from.key_set();
//   auto is_dense = [](c10::DispatchKeySet ts) {
//     return ts.has(c10::DispatchKey::CPU) ||
//            ts.has(c10::DispatchKey::PrivateUse1);
//   };
//   return (self_key == from_key) || (is_dense(self_key) &&
//   is_dense(from_key));
// }

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
             c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
GCUNativeFunctions::_scaled_dot_product_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double dropout_p, bool is_causal, bool return_debug_mask,
    ::std::optional<double> scale) {
  auto &cfg = torch_gcu::OpDebugConfig::GetInstance();
  OP_CALLTRACE(cfg, _scaled_dot_product_flash_attention)
  PRINT_OP_NAME_CHECK(cfg)
  OP_COMMON_MACRO(query, key, value, dropout_p, is_causal, return_debug_mask,
                  scale);
  static bool enable_op_check = cfg.enableOpCheck(__func__);
  static bool disable_op_check = cfg.disableOpCheck(__func__);
  bool op_check_scope = cfg.inOpCheckScope();
  bool op_check = (enable_op_check || op_check_scope) && (!disable_op_check);
  // NOTE: when dropout_p not equals, op check will be skipped because cpu and
  // gcu use different random algorithm and the result will be different
  bool no_dropout = (std::abs(dropout_p) < 1e-9);
  if (op_check && no_dropout) {
    OP_CHECK_INPUT_INFO_RECOED(query, key, value, dropout_p, is_causal,
                               return_debug_mask, scale)
    auto clone_input = clone_args(query, key, value, dropout_p, is_causal,
                                  return_debug_mask, scale);
    auto clone_op_check_input = clone_args(query, key, value, dropout_p,
                                           is_causal, return_debug_mask, scale);
    auto &&xdevice_out = at::native::call_fallback_fn<
        &torch_gcu::gcu_opcheck_run,
        ATEN_OP(_scaled_dot_product_flash_attention_for_cpu)>::
        call(std::get<0>(clone_op_check_input),
             std::get<1>(clone_op_check_input),
             std::get<2>(clone_op_check_input),
             std::get<3>(clone_op_check_input),
             std::get<4>(clone_op_check_input), std::nullopt,
             std::get<6>(clone_op_check_input));
    auto &&gcu_out = aotops::_scaled_dot_product_flash_attention(
        query, key, value, dropout_p, is_causal, return_debug_mask, scale);
    auto gcu_check_out =
        std::make_tuple(std::get<0>(gcu_out), std::get<1>(gcu_out));
    auto result =
        gcu_out_check(gcu_check_out, xdevice_out, std::string(__func__));
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
    return aotops::_scaled_dot_product_flash_attention(
        query, key, value, dropout_p, is_causal, return_debug_mask, scale);
  }
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
GCUNativeFunctions::_scaled_dot_product_efficient_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const ::std::optional<at::Tensor> &attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, ::std::optional<double> scale) {
  auto &cfg = torch_gcu::OpDebugConfig::GetInstance();
  OP_CALLTRACE(cfg, _scaled_dot_product_efficient_attention);
  PRINT_OP_NAME_CHECK(cfg)
  OP_COMMON_MACRO(query, key, value, attn_bias, compute_log_sumexp, dropout_p,
                  is_causal, scale);
  static bool enable_op_check = cfg.enableOpCheck(__func__);
  static bool disable_op_check = cfg.disableOpCheck(__func__);
  bool op_check_scope = cfg.inOpCheckScope();
  bool op_check = (enable_op_check || op_check_scope) && (!disable_op_check);
  // NOTE: when dropout_p not equals 0, op check will be skipped because cpu
  // and gcu use different random algorithm and the result will be different
  bool no_dropout = (std::abs(dropout_p) < 1e-9);
  // NOTE: when attn_bias is defined, op check will be skipped.
  bool no_attn_bias = !(attn_bias.has_value() && attn_bias.value().defined());
  if (op_check && no_dropout && no_attn_bias) {
    OP_CHECK_INPUT_INFO_RECOED(query, key, value, attn_bias, compute_log_sumexp,
                               dropout_p, is_causal, scale);
    auto clone_input =
        clone_args(query, key, value, attn_bias, compute_log_sumexp, dropout_p,
                   is_causal, scale);
    auto clone_op_check_input =
        clone_args(query, key, value, attn_bias, compute_log_sumexp, dropout_p,
                   is_causal, scale);

    auto &&xdevice_out = at::native::call_fallback_fn<
        &torch_gcu::gcu_opcheck_run,
        ATEN_OP(_scaled_dot_product_flash_attention_for_cpu)>::
        call(std::get<0>(clone_op_check_input),  // query
             std::get<1>(clone_op_check_input),  // key
             std::get<2>(clone_op_check_input),  // value
             std::get<5>(clone_op_check_input),  // dropout_p
             std::get<6>(clone_op_check_input),  // is_causal
             {},                                 // attn_mask
             std::get<7>(clone_op_check_input)   // scale
        );
    auto &&gcu_out = aotops::_scaled_dot_product_efficient_attention(
        query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal,
        scale);

    // NOTE: Only check the first output (attention)
    // TODO: Should we print a warning message here?
    auto gcu_check_out = std::get<0>(gcu_out);
    auto result = gcu_out_check(gcu_check_out, std::get<0>(xdevice_out),
                                std::string(__func__));

    if (result.acc_pass && !cfg.enableTestMode()) {
      PTDLOG(OP) << result.check_info;
    } else {
      OP_CHECK_DEBUG_INFO(cfg, ss, gcu_out, xdevice_out, clone_input)
    }
    return gcu_out;
  } else {
    if (op_check) {
      if (!no_dropout) {
        PTDLOG(OP) << "dropout_p not equals to 0.0, skip op check";
      }
      if (!no_attn_bias) {
        PTDLOG(OP) << "attn_bias is defined, skip op check";
      }
    }
    return aotops::_scaled_dot_product_efficient_attention(
        query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal,
        scale);
  }
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
GCUNativeFunctions::_scaled_dot_product_efficient_attention_backward(
    const at::Tensor &grad_out_, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &attn_bias, const at::Tensor &out,
    const at::Tensor &logsumexp, const at::Tensor &philox_seed,
    const at::Tensor &philox_offset, double dropout_p,
    ::std::array<bool, 4> grad_input_mask, bool is_causal,
    ::std::optional<double> scale) {
  auto &cfg = torch_gcu::OpDebugConfig::GetInstance();
  OP_CALLTRACE(cfg, _scaled_dot_product_efficient_attention_backward);
  PRINT_OP_NAME_CHECK(cfg)
  OP_COMMON_MACRO(grad_out_, query, key, value, attn_bias, out, logsumexp,
                  philox_seed, philox_offset, dropout_p, grad_input_mask,
                  is_causal, scale);
  static bool enable_op_check = cfg.enableOpCheck(__func__);
  static bool disable_op_check = cfg.disableOpCheck(__func__);
  bool op_check_scope = cfg.inOpCheckScope();
  bool op_check = (enable_op_check || op_check_scope) && (!disable_op_check);
  if (op_check) {
    PTDLOG(OP) << "_scaled_dot_product_efficient_attention_backward do NOT "
                  "support op_check";
  }
  return aotops::_scaled_dot_product_efficient_attention_backward(
      grad_out_, query, key, value, attn_bias, out, logsumexp, philox_seed,
      philox_offset, dropout_p, grad_input_mask, is_causal, scale);
}

}  // namespace torch_gcu
