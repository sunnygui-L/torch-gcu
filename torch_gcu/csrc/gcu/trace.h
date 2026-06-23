#pragma once

#include <string>

#include "gcu/gcu_macros.h"
#include "topstx/topstx.h"

enum TorchTraceCategory { USER, AOTOPS, COMPILE, OTHERS, DIST };

class TorchTrace {
 public:
  static TorchTrace &GetTorchTrace() {
    static TorchTrace inst;
    return inst;
  }

  static topstxDomainHandle_t &domain() { return GetTorchTrace().domain_; }

  static topstxDomainHandle_t &cpu_domain() {
    return GetTorchTrace().cpu_domain_;
  }

  static bool is_topstx_enabled() { return topstxDomainIsEnabled(domain()); }
  static bool is_topstx_cpu_domain_enabled() {
    return topstxDomainIsEnabled(cpu_domain());
  }

  TorchTrace() {
    domain_ = topstxDomainCreate("TORCH_GCU");
    //  Register categories, the order is consistent with TorchTraceCategory
    //  enum.
    topstxDomainNameCategory(domain_, TorchTraceCategory::USER, "USER");
    topstxDomainNameCategory(domain_, TorchTraceCategory::AOTOPS, "AOTOPS");
    topstxDomainNameCategory(domain_, TorchTraceCategory::COMPILE, "COMPILE");
    topstxDomainNameCategory(domain_, TorchTraceCategory::OTHERS, "OTHERS");
    topstxDomainNameCategory(domain_, TorchTraceCategory::DIST, "DIST");
    cpu_domain_ = topstxDomainCreate("__cpuop");
  }

  ~TorchTrace() {
    topstxDomainDestroy(domain_);
    topstxDomainDestroy(cpu_domain_);
  }

  topstxDomainHandle_t domain_;
  topstxDomainHandle_t cpu_domain_;
};

class TorchTracepoint {
 public:
  TorchTracepoint() {
    domain_ = nullptr;
    rngId_ = -1;
  }
  TorchTracepoint(topstxDomainHandle_t domain) {
    domain_ = domain;
    rngId_ = -1;
  }
  ~TorchTracepoint() { exit(); }

  void enter(int category, const char *message, const char *payload = nullptr) {
    if (rngId_ == -1) {
      topstxEventAttributes event = {};
      event.size = TOPSTX_EVENT_ATTRIB_STRUCT_SIZE;
      event.messageType = TOPSTX_MESSAGE_TYPE_STRING;
      event.message.str = message;
      event.category = category;
      if (payload) {
        event.payloadType = TOPSTX_PAYLOAD_TYPE_STRING;
        event.payload.stringValue = payload;
      }
      rngId_ = topstxDomainRangeStart(domain_, &event);
    }
  }
  void exit() {
    if (rngId_ != -1) {
      topstxDomainRangeEnd(domain_, rngId_);
      rngId_ = -1;
    }
  }

 protected:
  topstxDomainHandle_t domain_;
  topstxRangeId_t rngId_;
};

#define TORCH_VAR_CONCATENATE_DIRECT(s1, s2) s1##s2
#define TORCH_VAR_CONCATENATE(s1, s2) TORCH_VAR_CONCATENATE_DIRECT(s1, s2)
#define TORCH_VAR_CREATE(str) TORCH_VAR_CONCATENATE(str, __LINE__)

#define TORCH_TRACE_RANGE(category, name)                                    \
  TorchTracepoint TORCH_VAR_CREATE(tracepoint_##category##_##name)(          \
      TorchTrace::domain());                                                 \
  if (TorchTrace::is_topstx_enabled()) {                                     \
    TORCH_VAR_CREATE(tracepoint_##category##_##name).enter(category, #name); \
  }

#define TORCH_TRACE_FUNCTION(category)                                     \
  TorchTracepoint TORCH_VAR_CREATE(tracepoint_##category)(                 \
      TorchTrace::domain());                                               \
                                                                           \
  if (TorchTrace::is_topstx_enabled()) {                                   \
    TORCH_VAR_CREATE(tracepoint_##category).enter(category, __FUNCTION__); \
  }

#define TORCH_TRACE_START(category, name)                             \
  TorchTracepoint tracepoint_##category_##name(TorchTrace::domain()); \
  if (TorchTrace::is_topstx_enabled()) {                              \
    tracepoint_##category_##name.enter(category, #name);              \
  };

#define TORCH_TRACE_END(category, name)  \
  if (TorchTrace::is_topstx_enabled()) { \
    tracepoint_##category_##name.exit(); \
  }

#define AOTOPS_TRACE_FUNC TORCH_TRACE_FUNCTION(AOTOPS)
#define AOTOPS_TRACE_START(name) TORCH_TRACE_START(AOTOPS, name)
#define AOTOPS_TRACE_END(name) TORCH_TRACE_END(AOTOPS, name)

#define DIST_API_TRACE_FUNC() TORCH_TRACE_FUNCTION(DIST)
#define DIST_API_TRACE(name) TORCH_TRACE_RANGE(DIST, name)
#define DIST_TRACE_START(name) TORCH_TRACE_START(DIST, name)
#define DIST_TRACE_END(name) TORCH_TRACE_END(DIST, name)

TORCH_GCU_API void set_user_trace_start(const int &category,
                                        const std::string &name,
                                        const std::string &payload = "");

TORCH_GCU_API void set_user_trace_end(const int &category,
                                      const std::string &name);

#define COMPILE_PYTHON_TRACE_START(...)         \
  if (TorchTrace::is_topstx_enabled()) {        \
    set_user_trace_start(COMPILE, __VA_ARGS__); \
  }

#define COMPILE_PYTHON_TRACE_END(name)   \
  if (TorchTrace::is_topstx_enabled()) { \
    set_user_trace_end(COMPILE, name);   \
  }

#define OTHER_PYTHON_TRACE_START(...)          \
  if (TorchTrace::is_topstx_enabled()) {       \
    set_user_trace_start(OTHERS, __VA_ARGS__); \
  }

#define OTHER_PYTHON_TRACE_END(name)     \
  if (TorchTrace::is_topstx_enabled()) { \
    set_user_trace_end(OTHERS, name);    \
  }
