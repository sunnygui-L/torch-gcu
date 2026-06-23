/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <assert.h>
#include <stdlib.h>
#include <sys/types.h>

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "ITraceActivity.h"

namespace libkineto_gcu {

class TopsptiActivityBuffer {
 public:
  explicit TopsptiActivityBuffer(size_t size) : size_(size) {
    buf_.reserve(size);
  }
  TopsptiActivityBuffer() = delete;
  TopsptiActivityBuffer& operator=(const TopsptiActivityBuffer&) = delete;
  TopsptiActivityBuffer(TopsptiActivityBuffer&&) = default;
  TopsptiActivityBuffer& operator=(TopsptiActivityBuffer&&) = default;

  size_t size() const { return size_; }

  void setSize(size_t size) {
    assert(size <= buf_.capacity());
    size_ = size;
  }

  uint8_t* data() { return buf_.data(); }

 private:
  std::vector<uint8_t> buf_;
  size_t size_;

  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
};

using TopsptiActivityBufferMap =
    std::map<uint8_t*, std::unique_ptr<TopsptiActivityBuffer>>;

}  // namespace libkineto_gcu
