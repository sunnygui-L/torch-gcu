#pragma once
#include <string>

#include "c10/core/Allocator.h"
#include "gcu/gcu_macros.h"

namespace torch_gcu {
typedef std::string (*CppCallstackFn)(c10::GatheredContext* context);

TORCH_GCU_API void SetCppCallstackFn(CppCallstackFn f);

TORCH_GCU_API std::string GetCppFrame(c10::GatheredContext* context);
}  // namespace torch_gcu