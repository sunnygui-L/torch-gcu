
/*
 * Copyright 2020-2023 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/record_function.h>
#include <tops/tops_runtime_api.h>

#include "gcu/gcu_exception.h"

namespace torch_gcu {
inline void StreamSynchronize(topsStream_t stream) {
  RECORD_USER_SCOPE("topsStreamSynchronize");
  C10_GCU_CHECK(topsStreamSynchronize(stream));
}
}  // namespace torch_gcu
