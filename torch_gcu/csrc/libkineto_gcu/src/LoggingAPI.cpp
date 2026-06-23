/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */
#include "LoggingAPI.h"

#include "Logger.h"

namespace libkineto_gcu {
int getLogSeverityLevel() { return Logger::severityLevel(); }

void setLogSeverityLevel(int level) { SET_LOG_SEVERITY_LEVEL(level); }
}  // namespace libkineto_gcu
