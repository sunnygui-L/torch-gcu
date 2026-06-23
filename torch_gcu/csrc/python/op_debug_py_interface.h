/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include <string>
#include "gcu/gcu_macros.h"

namespace torch_gcu {

namespace interface {

TORCH_GCU_API void enterDumpOpNameScope();

TORCH_GCU_API void exitDimpOpNameScope();

TORCH_GCU_API void enterFallbackCpuScope();

TORCH_GCU_API void exitFallbackCpuScope();

TORCH_GCU_API void enterOpCheckScope();

TORCH_GCU_API void exitOpCheckScope();

TORCH_GCU_API bool getFallbackCpuScopeState();

TORCH_GCU_API bool getDumpOpNameScopeState();

TORCH_GCU_API bool getOpCheckScopeState();

TORCH_GCU_API std::string getOpStatisticsInfo();

TORCH_GCU_API void dumpOpStatisticsToJson();

TORCH_GCU_API void clearOpStatistics();

}  // namespace interface

}  // namespace torch_gcu
