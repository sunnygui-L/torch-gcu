/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#if !USE_GOOGLE_LOG

#include <atomic>
#include <map>
#include <set>
#include <vector>

#include "ILoggerObserver.h"

namespace libkineto_gcu {

using namespace libkineto_gcu;

class LoggerCollector : public ILoggerObserver {
 public:
  LoggerCollector() : buckets_() {}

  void write(const std::string& message, LoggerOutputType ot = ERROR) override {
    // Skip STAGE output type which is only used by USTLoggerCollector.
    // Skip VERBOSE output type which may bloat metadata section of traces.
    if (ot == STAGE || ot == VERBOSE) {
      return;
    }
    buckets_[ot].push_back(message);
  }

  const std::map<LoggerOutputType, std::vector<std::string>>
  extractCollectorMetadata() override {
    return buckets_;
  }

  void reset() override {
    trace_duration_ms = 0;
    event_count = 0;
    destinations.clear();
  }

  void addDevice(const int64_t device) override { devices.insert(device); }

  void setTraceDurationMS(const int64_t duration) override {
    trace_duration_ms = duration;
  }

  void addEventCount(const int64_t count) override { event_count += count; }

  void addDestination(const std::string& dest) override {
    if (!dest.empty()) {
      destinations.insert(dest);
    }
  }

  void addMetadata(const std::string& key, const std::string& value) override {}

 protected:
  std::map<LoggerOutputType, std::vector<std::string>> buckets_;

  // These are useful metadata to collect from TOPSPTIActivityProfiler for
  // internal tracking.
  std::set<int64_t> devices;
  int64_t trace_duration_ms{0};
  std::atomic<uint64_t> event_count{0};
  std::set<std::string> destinations;
};

}  // namespace libkineto_gcu

#endif  // !USE_GOOGLE_LOG
