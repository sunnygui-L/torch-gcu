// =============================================================================
//
// Copyright 2021-2023 Enflame. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#pragma once

#include "logging_api.h"
#include "python_frame_info.h"

// 定义module列表
#define EF_LOG_TEST_MOUDLE_LIST(DEF_LOG_MSG) \
  DEF_LOG_MSG(TORCH_GCU)                     \
  DEF_LOG_MSG(TORCH_GCU, TORCH_GCU)          \
  DEF_LOG_MSG(TORCH_GCU, COMPILE)            \
  DEF_LOG_MSG(TORCH_GCU, MEMORY)             \
  DEF_LOG_MSG(TORCH_GCU, OP)                 \
  DEF_LOG_MSG(TORCH_GCU, FALLBACK)

// 注册module列表
EFLOG_LOGMODULE_DEFINE(TORCH_GCU, EF_LOG_TEST_MOUDLE_LIST);

#define PTDLOG(module) EFDLOG(TORCH_GCU, TORCH_GCU, module)

#define PTCHECK(condition) \
  EFCHECK(condition) << torch_gcu::GetPythonFrame() << "\n"
