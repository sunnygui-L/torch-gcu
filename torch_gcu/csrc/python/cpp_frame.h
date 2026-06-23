#pragma once
#include <c10/core/Allocator.h>

#include <string>

#include "gcu/gcu_macros.h"
namespace torch_gcu {
TORCH_GCU_API std::string GetCppFrameImpl(c10::GatheredContext* context);
}