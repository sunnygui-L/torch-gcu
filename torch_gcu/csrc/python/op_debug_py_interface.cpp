/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "python/op_debug_py_interface.h"

#include "aten/op_debug_config.h"
#include "aten/op_statistics.h"

namespace torch_gcu {

namespace interface {

TORCH_GCU_API void enterDumpOpNameScope() {
  torch_gcu::OpDebugConfig::GetInstance().enterDumpOpNameScope();
}

TORCH_GCU_API void exitDimpOpNameScope() {
  torch_gcu::OpDebugConfig::GetInstance().exitDimpOpNameScope();
}

TORCH_GCU_API void enterFallbackCpuScope() {
  torch_gcu::OpDebugConfig::GetInstance().enterFallbackCpuScope();
}

TORCH_GCU_API void exitFallbackCpuScope() {
  torch_gcu::OpDebugConfig::GetInstance().exitFallbackCpuScope();
}

TORCH_GCU_API void enterOpCheckScope() {
  torch_gcu::OpDebugConfig::GetInstance().enterOpCheckScope();
}

TORCH_GCU_API void exitOpCheckScope() {
  torch_gcu::OpDebugConfig::GetInstance().exitOpCheckScope();
}

TORCH_GCU_API bool getFallbackCpuScopeState() {
  return torch_gcu::OpDebugConfig::GetInstance().getFallbackCpuScopeState();
}

TORCH_GCU_API bool getDumpOpNameScopeState() {
  return torch_gcu::OpDebugConfig::GetInstance().getDumpOpNameScopeState();
}

TORCH_GCU_API bool getOpCheckScopeState() {
  return torch_gcu::OpDebugConfig::GetInstance().getOpCheckScopeState();
}

TORCH_GCU_API std::string getOpStatisticsInfo() {
  return torch_gcu::OpStatistics::dumpToStr();
}

TORCH_GCU_API void dumpOpStatisticsToJson() { return torch_gcu::OpStatistics::dumpToJson(); }

TORCH_GCU_API void clearOpStatistics() { return torch_gcu::OpStatistics::clear(); }
}  // namespace interface

}  // namespace torch_gcu
