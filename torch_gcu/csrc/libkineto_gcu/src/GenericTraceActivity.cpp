/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "GenericTraceActivity.h"

#include "output_base.h"

namespace libkineto_gcu {
void GenericTraceActivity::log(ActivityLogger& logger) const {
  logger.handleGenericActivity(*this);
}
}  // namespace libkineto_gcu
