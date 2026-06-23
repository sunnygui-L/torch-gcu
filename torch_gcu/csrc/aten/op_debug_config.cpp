/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "aten/op_debug_config.h"

#include <c10/util/Exception.h>
#include <unistd.h>

#include <string>

#include "gcu/logging.h"
#include "gcu/sys_util.h"

namespace torch_gcu {

int64_t OpDebugConfig::_opcheck_error_idx = 0;

OpDebugConfig::OpDebugConfig() { init(); }

OpDebugConfig& OpDebugConfig::GetInstance() {
  static OpDebugConfig cf;
  return cf;
}

void OpDebugConfig::init() {
  std::string kAotOpCFG = util::GetEnvString("ENFLAME_PT_AOT_CFG", "");
  std::string kOpDebugConfig =
      util::GetEnvString("ENFLAME_PT_OP_DEBUG_CONFIG", "");
  if (!kAotOpCFG.empty()) {
    TORCH_WARN(
        "Config ENFLAME_PT_AOT_CFG is deprecated, use "
        "ENFLAME_PT_OP_DEBUG_CONFIG instead.");
    if (kOpDebugConfig.empty()) {
      kOpDebugConfig = kAotOpCFG;
    }
  }

  // new
  const std::string op_sync_mode = "op_sync_mode=";
  const std::string op_check_dump = "op_check_dump=";
  const std::string dump_path = "dump_path=";
  const std::string fallback_cpu = "fallback_cpu=";
  const std::string op_check = "op_check=";
  const std::string op_input_dump = "op_input_dump=";
  const std::string op_output_dump = "op_output_dump=";
  const std::string op_check_no_break = "op_check_no_break=";
  const std::string op_check_tol_fp32 = "op_check_tol_fp32=";
  const std::string op_check_tol_fp16 = "op_check_tol_fp16=";
  const std::string op_check_tol_bf16 = "op_check_tol_bf16=";
  const std::string deregister = "deregister=";
  const std::string op_statistics = "op_statistics=";
  const std::string op_statistics_dump = "op_statistics_dump=";
  const std::string op_calltrace = "op_calltrace=";

  // test mode
  const std::string test_mode = "test_mode=";

  // key parser

  // new
  parserBoolFlags(kOpDebugConfig, op_sync_mode, _op_sync_mode);
  parserBoolFlags(kOpDebugConfig, op_check_dump, _op_check_dump);
  parserBoolFlags(kOpDebugConfig, op_check_no_break, _op_check_no_break);
  parserBoolFlags(kOpDebugConfig, op_statistics, _op_statistics);
  parserBoolFlags(kOpDebugConfig, op_statistics_dump, _op_statistics_dump);
  parserBoolFlags(kOpDebugConfig, test_mode, _test_mode);
  parserOperators(kOpDebugConfig, op_check, _op_check);
  parserOperators(kOpDebugConfig, fallback_cpu, _fallback_cpu);
  parserOperators(kOpDebugConfig, deregister, _deregister);
  parserOperators(kOpDebugConfig, op_input_dump, _op_input_dump);
  parserOperators(kOpDebugConfig, op_output_dump, _op_output_dump);
  parserTolValue(kOpDebugConfig, op_check_tol_fp32, _op_check_rtol_fp32,
                 _op_check_atol_fp32);
  parserTolValue(kOpDebugConfig, op_check_tol_fp16, _op_check_rtol_fp16,
                 _op_check_atol_fp16);
  parserTolValue(kOpDebugConfig, op_check_tol_bf16, _op_check_rtol_bf16,
                 _op_check_atol_bf16);
  parserStringValue(kOpDebugConfig, dump_path, _dump_path);
  parserBoolFlags(kOpDebugConfig, op_calltrace, _op_calltrace);
  // clang-format on

  std::string pid_timestep = "";
  if ((((_op_check_dump && anyCheckOp())) ||
       (anyOpDumpInput() || anyOpDumpOutput())) &&
      !_test_mode) {
    pid_timestep = "_" + std::to_string(getpid()) + "_" + util::GetTimeStamp();
  }

  if (_op_check_dump && anyCheckOp()) {
    _op_check_dump_path = initOpCheckDumpPath(pid_timestep);
  }

  if (anyOpDumpInput() || anyOpDumpOutput()) {
    _op_dump_path = initOpDumpPath(pid_timestep);
  }

  // clang-format off

  PTDLOG(OP) << "ENFLAME_PT_OP_DEBUG_CONFIG: {\n"
             << "  op_sync_mode: " << (_op_sync_mode ? "True" : "False") << "\n"
             << "  fallback_cpu: " << dumpOperators(_fallback_cpu).data() << "\n"
             << "  op_check: " << dumpOperators(_op_check).data() << "\n"
             << "  op_input_dump: " << dumpOperators(_op_input_dump) << "\n"
             << "  op_output_dump: " << dumpOperators( _op_output_dump) << "\n"
             << "  op_check_no_break: " << (_op_check_no_break ? "True" : "False") << "\n"
             << "  op_check_tol_fp32: " << _op_check_rtol_fp32 << "," << _op_check_atol_fp32 << "\n"
             << "  op_check_tol_fp16: " << _op_check_rtol_fp16 << "," << _op_check_atol_fp16 << "\n"
             << "  op_check_tol_bf16: " << _op_check_rtol_bf16 << "," << _op_check_atol_bf16 << "\n"
             << "  op_check_dump: " << (_op_check_dump ? "True" : "False") << "\n"
             << "  op_statistics: " << (_op_statistics ? "True" : "False") << "\n"
             << "  op_statistics_dump: "<< (_op_statistics_dump ? "True" : "False") << "\n"
             << "  dump_path: " << _dump_path << "\n"
             << "  deregister: " << dumpOperators(_deregister).data() << "\n"
             << "  op_calltrace: " << (_op_calltrace ? "True" : "False") << "\n"
             << "  test_mode: " << (_test_mode ? "True" : "False") << "\n"
             << "}\n";
  // clang-format on
}

bool OpDebugConfig::enableFallback(const std::string& op_name) const {
  return _fallback_cpu.enable(op_name);
}

bool OpDebugConfig::disableFallback(const std::string& op_name) const {
  return _fallback_cpu.disable(op_name);
}

bool OpDebugConfig::enableOpCheck(const std::string& op_name) const {
  return _op_check.enable(op_name);
}

bool OpDebugConfig::disableOpCheck(const std::string& op_name) const {
  return _op_check.disable(op_name);
}

bool OpDebugConfig::enableDumpInput(const std::string& op_name) const {
  return _op_input_dump.enable(op_name);
}

bool OpDebugConfig::disableDumpInput(const std::string& op_name) const {
  return _op_input_dump.disable(op_name);
}

bool OpDebugConfig::enableDumpOutput(const std::string& op_name) const {
  return _op_output_dump.enable(op_name);
}

bool OpDebugConfig::disableDumpOutput(const std::string& op_name) const {
  return _op_output_dump.disable(op_name);
}

bool OpDebugConfig::isDeregister(const std::string& op_name) const {
  return _deregister.enable(op_name) && (!_deregister.disable(op_name));
}

bool OpDebugConfig::enableOpCheckNoBreak() const { return _op_check_no_break; }

std::string OpDebugConfig::getDumpPath() const {
  util::CreatDirIfNotExist(_dump_path);
  return _dump_path;
}

std::string OpDebugConfig::getOpCheckDumpPath() const {
  util::CreatDirIfNotExist(_op_check_dump_path);
  return _op_check_dump_path;
}

std::string OpDebugConfig::getOpDumpPath() const {
  util::CreatDirIfNotExist(_op_dump_path);
  return _op_dump_path;
}

std::string OpDebugConfig::initOpCheckDumpPath(std::string pid_timestep) {
  return _dump_path + "/op_check_dump" + pid_timestep;
}

std::string OpDebugConfig::initOpDumpPath(std::string pid_timestep) {
  return _dump_path + "/op_dump" + pid_timestep;
}

bool OpDebugConfig::enableOpCheckDumpData() const { return _op_check_dump; }

bool OpDebugConfig::enableOpStatisticsDumpData() const {
  return _op_statistics_dump;
}

bool OpDebugConfig::enableOpCalltrace() const { return _op_calltrace; }

double OpDebugConfig::getFP32Rtol() const { return _op_check_rtol_fp32; }

double OpDebugConfig::getFP32Atol() const { return _op_check_atol_fp32; }

double OpDebugConfig::getFP16Rtol() const { return _op_check_rtol_fp16; }

double OpDebugConfig::getFP16Atol() const { return _op_check_atol_fp16; }

double OpDebugConfig::getBF16Rtol() const { return _op_check_rtol_bf16; }

double OpDebugConfig::getBF16Atol() const { return _op_check_atol_bf16; }

int64_t OpDebugConfig::getErrorIdx() const { return _opcheck_error_idx; }

void OpDebugConfig::addErrorIdx() { _opcheck_error_idx++; }

bool OpDebugConfig::enableTestMode() const { return _test_mode; }

void OpDebugConfig::parserOperators(
    const std::string& str, const std::string& key, op_selection& op_sets,
    const std::string& delimiter, bool deprecated, const std::string& new_key) {
  auto info = str;
  auto pos = info.find(key);

  auto insert = [&](const std::string& val) {
    if (val == "all" || val == "all_ops") {
      if (val == "all") {
        TORCH_WARN("Config all is deprecated, use all_ops instead.");
      }
      op_sets.select_all = true;
    } else if (val[0] == '-') {
      op_sets.exclude.insert(val.substr(1));
    } else {
      op_sets.include.insert(val);
    }
  };

  const std::string op_delimiter = ",";
  if (pos != info.npos) {
    if (deprecated) {
      TORCH_WARN("Config: " + key + " is deprecated, use " + new_key +
                 " instead.");
    }
    auto val = info.substr(pos + key.size());
    pos = val.find(delimiter);
    val = val.substr(0, pos);
    while (!val.empty()) {
      pos = val.find(op_delimiter);
      if (pos == op_delimiter.npos) {
        insert(val);
        break;
      } else {
        auto op = val.substr(0, pos);
        if (!op.empty()) {
          insert(op);
        }
        val = val.substr(pos + op_delimiter.size());
      }
    }
  }
}

void OpDebugConfig::parserBoolFlags(const std::string& str,
                                    const std::string& key, bool& flag,
                                    const std::string& delimiter,
                                    bool deprecated,
                                    const std::string& new_key) {
  auto info = str;
  auto pos = info.find(key);

  const std::string true_str = "true";
  const std::string false_str = "false";
  if (pos != info.npos) {
    if (deprecated) {
      TORCH_WARN("Config: " + key + " is deprecated, use " + new_key +
                 " instead.");
    }
    auto val = info.substr(pos + key.size());
    pos = val.find(delimiter);
    val = val.substr(0, pos);
    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
    if (true_str == val) {
      flag = true;
    } else if (false_str == val) {
      flag = false;
    }
  }
}

void OpDebugConfig::parserTolValue(const std::string& str,
                                   const std::string& key, double& rtol,
                                   double& atol, const std::string& delimiter,
                                   const std::string& sub_delimiter) {
  auto info = str;
  auto pos = info.find(key);

  if (pos != info.npos) {
    auto val = info.substr(pos + key.size());
    pos = val.find(delimiter);
    val = val.substr(0, pos);
    pos = val.find(sub_delimiter);
    auto rtol_str = val.substr(0, pos);
    auto atol_str = val.substr(pos + sub_delimiter.size());
    try {
      if (!rtol_str.empty()) {
        rtol = std::stold(rtol_str);
      }
      if (!atol_str.empty()) {
        atol = std::stold(atol_str);
      }
    } catch (...) {
      TORCH_WARN("Config: " + key + " value is ilegal, get \'" + val +
                 "\'. Use defaule value, rtol:" + std::to_string(rtol) +
                 " atol:" + std::to_string(atol));
    }
  }
}

void OpDebugConfig::parserStringValue(const std::string& str,
                                      const std::string& key,
                                      std::string& value,
                                      const std::string& delimiter) {
  auto info = str;
  auto pos = info.find(key);

  if (pos != info.npos) {
    auto val = info.substr(pos + key.size());
    pos = val.find(delimiter);
    val = val.substr(0, pos);
    try {
      value = val;
    } catch (...) {
      TORCH_WARN("Config: " + key + " value is ilegal, get \'" + val +
                 "\'. Use defaule value: " + value);
    }
  }
}

std::string OpDebugConfig::dumpOperators(const op_selection& op_sets) {
  std::stringstream ss;
  ss << "select_all = " << (op_sets.select_all ? "True" : "False") << ", ";
  ss << "include = {";
  for (auto& op : op_sets.include) {
    ss << op << ",";
  }
  ss << "}, ";
  ss << "exclude = {";
  for (auto& op : op_sets.exclude) {
    ss << op << ",";
  }
  ss << "}";
  return ss.str();
}

bool OpDebugConfig::anyCheckOp() {
  if (!_op_check.select_all && _op_check.include.empty()) {
    return false;
  } else {
    return true;
  }
}

bool OpDebugConfig::anyOpDumpInput() {
  if (!_op_input_dump.select_all && _op_input_dump.include.empty()) {
    return false;
  } else {
    return true;
  }
}

bool OpDebugConfig::anyOpDumpOutput() {
  if (!_op_output_dump.select_all && _op_output_dump.include.empty()) {
    return false;
  } else {
    return true;
  }
}

}  // namespace torch_gcu
