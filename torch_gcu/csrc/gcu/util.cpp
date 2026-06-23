/*
 * Copyright 2021-2023 Enflame. All Rights Reserved.
 */
#include "gcu/util.h"

GCU_TORCH_GCU_UTIL_NS_BEGIN

std::vector<std::string> StrSplit(std::string text, char delim) {
  size_t start;
  size_t end = 0;
  std::vector<std::string> tokens;
  while ((start = text.find_first_not_of(delim, end)) != std::string::npos) {
    end = text.find(delim, start);
    auto token = text.substr(start, end - start);
    tokens.emplace_back(token.begin(), token.end());
  }
  return tokens;
}

std::string StrJoin(std::vector<std::string> strs, std::string separator) {
  std::string result;
  if (strs.empty()) {
    return result;
  }
  unsigned int i = 0;
  for (; i < strs.size() - 1; i++) {
    result.append(strs[i]);
    result.append(separator);
  }
  result.append(strs[i]);
  return result;
}

void StrReplaceAll(std::string& str, const std::string& from,
                   const std::string& to) {
  if (from.empty()) return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();  // In case 'to' contains 'from', like replacing
                               // 'x' with 'yx'
  }
}

std::string StrReplaceAll(const std::string& str, const std::string& from,
                          const std::string& to) {
  std::string out = str;
  if (from.empty()) return out;
  size_t start_pos = 0;
  while ((start_pos = out.find(from, start_pos)) != std::string::npos) {
    out.replace(start_pos, from.length(), to);
    start_pos += to.length();  // In case 'to' contains 'from', like replacing
                               // 'x' with 'yx'
  }
  return out;
}

#define INT8_IMPL_VECTOSTR                        \
  do {                                            \
    if (vec.empty()) {                            \
      return "";                                  \
    }                                             \
    std::stringstream result;                     \
    for (size_t i = 0; i + 1 < vec.size(); ++i) { \
      result << std::to_string(vec[i]) << sep;    \
    }                                             \
    result << std::to_string(vec.back());         \
    return result.str();                          \
  } while (0)

template <>
std::string VecToStr(const std::vector<uint8_t>& vec, const char* sep) {
  INT8_IMPL_VECTOSTR;
}

template <>
std::string VecToStr(const std::vector<int8_t>& vec, const char* sep) {
  INT8_IMPL_VECTOSTR;
}
#undef INT8_IMPL_VECTOSTR

int64_t ElementsIn(const std::vector<int64_t>& shape) {
  return std::accumulate<decltype(shape.begin()), int64_t>(
      shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

bool HasZeroShape(const std::vector<int64_t>& shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      return true;
    }
  }
  return false;
}

GCU_TORCH_GCU_UTIL_NS_END
