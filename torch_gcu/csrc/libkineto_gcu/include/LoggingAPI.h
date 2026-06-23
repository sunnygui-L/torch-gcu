/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include "gcu/gcu_macros.h"
namespace libkineto_gcu {
TORCH_GCU_API int getLogSeverityLevel();
TORCH_GCU_API void setLogSeverityLevel(int level);
}  // namespace libkineto_gcu
