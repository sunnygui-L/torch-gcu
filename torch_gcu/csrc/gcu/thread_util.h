/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gcu/gcu_macros.h"
#include "gcu/namespace.h"

GCU_TORCH_GCU_UTIL_NS_BEGIN

TORCH_GCU_API int32_t systemThreadId();
TORCH_GCU_API int32_t threadId();
TORCH_GCU_API bool setThreadName(const std::string& name);
TORCH_GCU_API std::string getThreadName();

TORCH_GCU_API int32_t processId();
TORCH_GCU_API std::string processName(int32_t pid);

// Return a list of pids and process names for the current process
// and its parents.
TORCH_GCU_API std::vector<std::pair<int32_t, std::string>>
pidCommandPairsOfAncestors();

GCU_TORCH_GCU_UTIL_NS_END
