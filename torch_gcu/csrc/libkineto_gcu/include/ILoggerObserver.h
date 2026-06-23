/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <string>

#include "gcu/gcu_macros.h"

// Stages in libkineto_gcu used when pushing logs to UST Logger.
constexpr char kWarmUpStage[] = "Warm Up";
constexpr char kCollectionStage[] = "Collection";
constexpr char kPostProcessingStage[] = "Post Processing";

#if !USE_GOOGLE_LOG

#include <stdint.h>

#include <map>
#include <vector>

namespace libkineto_gcu {

enum TORCH_GCU_API LoggerOutputType {
  VERBOSE = 0,
  INFO = 1,
  WARNING = 2,
  STAGE = 3,
  ERROR = 4,
  ENUM_COUNT = 5
};

const char* toString(LoggerOutputType t);
LoggerOutputType toLoggerOutputType(const std::string& str);

constexpr int LoggerTypeCount = (int)LoggerOutputType::ENUM_COUNT;

class TORCH_GCU_API ILoggerObserver {
 public:
  virtual ~ILoggerObserver() = default;
  virtual void write(const std::string& message, LoggerOutputType ot) = 0;
  virtual const std::map<LoggerOutputType, std::vector<std::string>>
  extractCollectorMetadata() = 0;
  virtual void reset() = 0;
  virtual void addDevice(const int64_t device) = 0;
  virtual void setTraceDurationMS(const int64_t duration) = 0;
  virtual void addEventCount(const int64_t count) = 0;
  virtual void setTraceID(const std::string&) {}
  virtual void setGroupTraceID(const std::string&) {}
  virtual void addDestination(const std::string& dest) = 0;
  virtual void setTriggerOnDemand() {}
  virtual void addMetadata(const std::string& key,
                           const std::string& value) = 0;
};

}  // namespace libkineto_gcu

#endif  // !USE_GOOGLE_LOG
