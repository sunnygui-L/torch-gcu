/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#pragma once

#include <unordered_set>

// WARN: Do not delete this header file, else the config log will not be printed
#include "gcu/gcu_macros.h"
#include "gcu/logging.h"

namespace torch_gcu {

class TORCH_GCU_API OpDebugConfig {
 public:
  static OpDebugConfig& GetInstance();

  struct op_selection {
    bool select_all = false;
    std::unordered_set<std::string> include;
    std::unordered_set<std::string> exclude;
    bool enable(const std::string& name) const {
      return (select_all || include.count(name));
    }
    bool disable(const std::string& name) const { return exclude.count(name); }
  };

  void init();

  bool enableSyncMode() const { return _op_sync_mode; }

  bool enableFallback(const std::string& op_name) const;

  bool disableFallback(const std::string& op_name) const;

  bool inFallbackScope() const { return _fallback_cpu_scope; }

  bool enableOpCheck(const std::string& op_name) const;

  bool disableOpCheck(const std::string& op_name) const;

  bool inOpCheckScope() const { return _op_check_scope; }

  bool enableDumpInput(const std::string& op_name) const;

  bool disableDumpInput(const std::string& op_name) const;

  bool enableDumpOutput(const std::string& op_name) const;

  bool disableDumpOutput(const std::string& op_name) const;

  bool enableOpCheckNoBreak() const;

  bool enableOpCheckDumpData() const;

  bool enableOpStatisticsDumpData() const;

  bool enableOpCalltrace() const;

  bool enableOpStatistics() const { return _op_statistics; }

  bool isDeregister(const std::string& op_name) const;

  void enterDumpOpNameScope() { _dump_op_name_scope = true; }

  void exitDimpOpNameScope() { _dump_op_name_scope = false; }

  void enterFallbackCpuScope() { _fallback_cpu_scope = true; }

  void exitFallbackCpuScope() { _fallback_cpu_scope = false; }

  void enterOpCheckScope() { _op_check_scope = true; }

  void exitOpCheckScope() { _op_check_scope = false; }

  bool getFallbackCpuScopeState() const { return _fallback_cpu_scope; }

  bool getDumpOpNameScopeState() const { return _dump_op_name_scope; }

  bool getOpCheckScopeState() const { return _op_check_scope; }

  std::string getDumpPath() const;

  std::string getOpCheckDumpPath() const;

  std::string getOpDumpPath() const;

  int64_t getErrorIdx() const;

  void addErrorIdx();

  double getFP32Rtol() const;

  double getFP32Atol() const;

  double getFP16Rtol() const;

  double getFP16Atol() const;

  double getBF16Rtol() const;

  double getBF16Atol() const;

  // some behavior is different in debug mode
  bool enableTestMode() const;

 private:
  void parserOperators(const std::string& str, const std::string& key,
                       op_selection& op_sets,
                       const std::string& delimiter = " ",
                       bool deprecated = false,
                       const std::string& new_key = "");
  void parserBoolFlags(const std::string& str, const std::string& key,
                       bool& flag, const std::string& delimiter = " ",
                       bool deprecated = false,
                       const std::string& new_key = "");
  void parserTolValue(const std::string& str, const std::string& key,
                      double& rtol, double& atol,
                      const std::string& delimiter = " ",
                      const std::string& sub_delimiter = ",");
  void parserStringValue(const std::string& str, const std::string& key,
                         std::string& value,
                         const std::string& delimiter = " ");
  std::string initOpCheckDumpPath(std::string pid_timestep);
  std::string initOpDumpPath(std::string pid_timestep);
  std::string dumpOperators(const op_selection& op_sets);
  bool anyCheckOp();
  bool anyOpDumpInput();
  bool anyOpDumpOutput();
  OpDebugConfig();
  OpDebugConfig(const OpDebugConfig&) = delete;
  OpDebugConfig(OpDebugConfig&&) = delete;
  OpDebugConfig& operator=(const OpDebugConfig&) = delete;
  OpDebugConfig& operator=(OpDebugConfig&&) = delete;

 private:
  static int64_t _opcheck_error_idx;
  bool _op_sync_mode = false;
  bool _dump_op_name_scope = false;
  bool _fallback_cpu_scope = false;
  bool _op_check_scope = false;
  bool _op_check_no_break = false;
  bool _op_check_dump = false;
  double _op_check_rtol_fp32 = 1e-6;
  double _op_check_atol_fp32 = 1e-6;
  double _op_check_rtol_fp16 = 1e-3;
  double _op_check_atol_fp16 = 1e-3;
  double _op_check_rtol_bf16 = 1e-3;
  double _op_check_atol_bf16 = 1e-3;
  std::string _dump_path = "./torch_gcu_dump_file";
  std::string _op_check_dump_path;
  std::string _op_dump_path;
  bool _op_statistics = false;
  bool _op_statistics_dump = false;
  bool _op_calltrace = false;
  op_selection _fallback_cpu;
  op_selection _op_check;
  op_selection _deregister;
  op_selection _op_input_dump;
  op_selection _op_output_dump;
  bool _test_mode = false;
};
}  // namespace torch_gcu
