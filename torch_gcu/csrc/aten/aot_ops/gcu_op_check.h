/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include "aten/aot_ops/gcu_opcheck_utils.h"
#include "aten/aten_cpu_fallback.h"
#include "aten/op_debug_config.h"
#include "aten/op_statistics.h"
#include "aten/register_fallback_ops.h"
#include "gcu/logging.h"
#include "gcu/trace.h"

namespace torch_gcu {

namespace {

#define PRINT_OP_NAME(cfg, USE)                                       \
  if (cfg.getDumpOpNameScopeState())                                  \
    PTDLOG(OP) << "torch_gcu op name \"" << std::string(__FUNCTION__) \
               << "\" for " << #USE;

#define PRINT_OP_NAME_WITH_OP(cfg, op, USE) \
  if (cfg.getDumpOpNameScopeState())        \
    PTDLOG(OP) << "torch_gcu op name \"" << #op << "\" for " << #USE;

#define GCU_CPU_FALLBACK_WITH_LIMITED(cfg, namespace, op, limited_fallback, \
                                      ...)                                  \
  static bool enable_fallback = cfg.enableFallback(__FUNCTION__);           \
  static bool disable_fallback = cfg.disableFallback(__FUNCTION__);         \
  bool fallback_scope = cfg.inFallbackScope();                              \
  bool fallback_cpu =                                                       \
      (enable_fallback || fallback_scope) && (!disable_fallback);           \
  if (fallback_cpu || limited_fallback) {                                   \
    return at::native::call_fallback_fn<                                    \
        &gcu_cpu_fallback, namespace##_OP(op)>::call(__VA_ARGS__);          \
  }

#define GCU_CPU_FALLBACK(cfg, namespace, op, ...)                   \
  static bool enable_fallback = cfg.enableFallback(__FUNCTION__);   \
  static bool disable_fallback = cfg.disableFallback(__FUNCTION__); \
  bool fallback_scope = cfg.inFallbackScope();                      \
  bool fallback_cpu =                                               \
      (enable_fallback || fallback_scope) && (!disable_fallback);   \
  if (fallback_cpu) {                                               \
    return at::native::call_fallback_fn<                            \
        &gcu_cpu_fallback, namespace##_OP(op)>::call(__VA_ARGS__);  \
  }

#define OP_CHECK_INPUT_INFO_RECOED(...)                 \
  std::stringstream ss;                                 \
  auto error_idx = cfg.getErrorIdx();                   \
  ss << "op_name:" << std::string(__FUNCTION__) << "\n" \
     << "op_id: " << error_idx << "\ninput:\n";         \
  torch_gcu::print_args(ss, 0, __VA_ARGS__);

#define OP_CHECK_DEBUG_INFO(cfg, ss, gcu_out, xdevice_out, ...)        \
  ss << "output:\n";                                                   \
  torch_gcu::print_args(ss, 0, gcu_out);                               \
  if (cfg.enableOpCheckDumpData()) {                                   \
    auto file_path = cfg.getOpCheckDumpPath();                         \
    auto op_path = file_path + "/" + std::string(__FUNCTION__) + "_" + \
                   std::to_string(error_idx) + "_";                    \
    std::string input_path = op_path + "input";                        \
    torch_gcu::dump_args(input_path, 0, __VA_ARGS__);                  \
    std::string gcu_output_path = op_path + "gcu_output";              \
    torch_gcu::dump_args(gcu_output_path, 0, gcu_out);                 \
    std::string cpu_output_path = op_path + "cpu_output";              \
    torch_gcu::dump_args(cpu_output_path, 0, xdevice_out);             \
  }                                                                    \
  cfg.addErrorIdx();                                                   \
  if (cfg.enableOpCheckNoBreak()) {                                    \
    PTDLOG(OP) << result.check_info << ss.str();                       \
  } else {                                                             \
    PTCHECK(0) << result.check_info << ss.str();                       \
  }

#define OP_CALLTRACE(cfg, op)                                       \
  if (cfg.enableOpCalltrace()) {                                    \
    PTDLOG(OP) << "op name:" << #op << torch_gcu::GetPythonFrame(); \
  }

}  // namespace

void gcu_opcheck_run(const c10::OperatorHandle& op, torch::jit::Stack* stack);

struct OpCheckResult {
  bool acc_pass;
  std::string check_info;
};

OpCheckResult result_check_func(const at::Tensor& out1, const at::Tensor& out2,
                                const std::string& op_name);

OpCheckResult result_check_func(const int64_t v1, const int64_t v2,
                                const std::string& op_name);

OpCheckResult result_check_func(const c10::SymInt& v1, const c10::SymInt& v2,
                                const std::string& op_name);

inline OpCheckResult gcu_out_check(const at::Tensor& gcu_out,
                                   const at::Tensor& xdevice_out,
                                   const std::string& op_name) {
  return result_check_func(gcu_out, xdevice_out, op_name);
}

inline OpCheckResult gcu_out_check(const std::vector<at::Tensor>& gcu_outs,
                                   const std::vector<at::Tensor>& xdevice_outs,
                                   const std::string& op_name) {
  auto gcu_out_num = gcu_outs.size();
  auto xdevice_out_num = xdevice_outs.size();
  std::stringstream ss;
  if (gcu_out_num != xdevice_out_num) {
    ss << op_name << " check failed.\n"
       << "gcu out num must equal to xdevice out, "
       << "but gcu has " << gcu_out_num << " results, "
       << "and xdevice has " << xdevice_out_num << " results!\n";
    return {false, ss.str()};
  }
  for (unsigned int i = 0; i < xdevice_out_num; i++) {
    auto gcu_out = gcu_outs[i];
    auto xpu_out = xdevice_outs[i];
    auto result = result_check_func(gcu_out, xpu_out, op_name);
    if (!result.acc_pass) {
      return result;
    }
  }
  ss << op_name << " acc check pass.\n";
  return {true, ss.str()};
}

template <typename Tuple1, typename Tuple2, size_t... I>
inline OpCheckResult tuple_result_check(const Tuple1& t1, const Tuple2& t2,
                                        std::index_sequence<I...>,
                                        const std::string& op_name) {
  std::vector<OpCheckResult> results = {
      (result_check_func(std::get<I>(t1), std::get<I>(t2), op_name))...};
  for (auto result : results) {
    if (!result.acc_pass) {
      return result;
    }
  }
  return {true, op_name + " acc check pass.\n"};
}

template <typename... Args1, typename... Args2>
inline OpCheckResult gcu_out_check(const std::tuple<Args1...>& gcu_outs,
                                   const std::tuple<Args2...>& xdevice_outs,
                                   const std::string& op_name) {
  using tuple1_type = std::tuple<Args1...>;
  using tuple2_type = std::tuple<Args2...>;
  auto gcu_out_num = std::tuple_size<tuple1_type>::value;
  auto xdevice_out_num = std::tuple_size<tuple2_type>::value;
  if (gcu_out_num != xdevice_out_num) {
    std::stringstream ss;
    ss << op_name << " check failed.\n"
       << "gcu out num must equal to xdevice out, "
       << "but gcu has " << gcu_out_num << " results, "
       << "and xdevice has " << xdevice_out_num << " results!\n";
    return {false, ss.str()};
  }
  return tuple_result_check(
      gcu_outs, xdevice_outs,
      std::make_index_sequence<std::tuple_size_v<tuple2_type> >{}, op_name);
}

// pytorch profiler trace
#define OP_COMMON_MACRO(...)                                        \
  RECORD_FUNCTION(std::string("GCU::") + std::string(__FUNCTION__), \
                  c10::ArrayRef<const c10::IValue>{__VA_ARGS__});   \
  AOTOPS_TRACE_FUNC;

#define PRINT_OP_NAME_WITH_OP_ALL(cfg, op) \
  PRINT_OP_NAME_WITH_OP(cfg, op, fallback_cpu and op_check);

#define PRINT_OP_NAME_ALL(cfg) PRINT_OP_NAME(cfg, fallback_cpu and op_check);

#define PRINT_OP_NAME_FALLBACK(cfg) PRINT_OP_NAME(cfg, fallback_cpu);

#define PRINT_OP_NAME_CHECK(cfg) PRINT_OP_NAME(cfg, op_check);

// ======================== GCU Runner Macros

};  // namespace torch_gcu
