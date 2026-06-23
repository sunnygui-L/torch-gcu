/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "python/env.h"

#include "gcu/logging.h"
#include "gcu/sys_util.h"

#define PRINT_ENV(name, val) PTDLOG(NODE) << "Set ENV:" << name << "=" << val;

namespace torch_gcu {
// Show Python frame detail in HLIR op_name
bool ShowFrameInOpname() {
  static const bool kShowFrameInOpname =
      util::GetEnvBool("PT_SHOW_FRAME_IN_OPNAME", false);
  return kShowFrameInOpname;
}

}  // namespace torch_gcu
