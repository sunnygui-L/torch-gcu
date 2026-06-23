/*
 * Copyright 2021-2023 Enflame. All Rights Reserved.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>
#include <vector>

// #include "utils/hasher.h"
#include "gcu/gcu_macros.h"
#include "gcu/namespace.h"

#define FILE_AND_FUNC std::string(__FILE__) + "::" + std::string(__func__)

GCU_TORCH_GCU_UTIL_NS_BEGIN

template <typename T>
class Cleanup {
 public:
  using StatusType = T;

  explicit Cleanup(std::function<void(StatusType)> func)
      : func_(std::move(func)) {}
  Cleanup(Cleanup&& ref)
      : func_(std::move(ref.func_)), status_(std::move(ref.status_)) {}
  Cleanup(const Cleanup&) = delete;

  ~Cleanup() {
    if (func_ != nullptr) {
      func_(std::move(status_));
    }
  }

  Cleanup& operator=(const Cleanup&) = delete;

  Cleanup& operator=(Cleanup&& ref) {
    if (this != &ref) {
      func_ = std::move(ref.func_);
      status_ = std::move(ref.status_);
    }
    return *this;
  }

  void Release() { func_ = nullptr; }

  void SetStatus(StatusType status) { status_ = std::move(status); }

  const StatusType& GetStatus() const { return status_; }

 private:
  std::function<void(StatusType)> func_;
  StatusType status_;
};

using ExceptionCleanup = Cleanup<std::exception_ptr>;

struct MidPolicy {
  size_t operator()(size_t size) const { return size / 2; }
};

template <typename C>
std::vector<const typename C::value_type::element_type*> GetConstSharedPointers(
    const C& shared_pointers) {
  std::vector<const typename C::value_type::element_type*> pointers;
  pointers.reserve(shared_pointers.size());
  for (auto& shared_pointer : shared_pointers) {
    pointers.push_back(shared_pointer.get());
  }
  return pointers;
}

template <typename C>
std::vector<typename C::value_type::element_type*> GetSharedPointers(
    const C& shared_pointers) {
  std::vector<typename C::value_type::element_type*> pointers;
  pointers.reserve(shared_pointers.size());
  for (auto& shared_pointer : shared_pointers) {
    pointers.push_back(shared_pointer.get());
  }
  return pointers;
}

template <typename C, typename K, typename T, typename F>
void InsertCombined(C* map, const K& key, const T& value, const F& combiner) {
  auto it = map->find(key);
  if (it == map->end()) {
    map->emplace(key, value);
  } else {
    it->second = combiner(it->second, value);
  }
}

template <typename T>
std::vector<T> Iota(size_t size, T init = 0, T incr = 1) {
  std::vector<T> result(size);
  T value = init;
  for (size_t i = 0; i < size; ++i, value += incr) {
    result[i] = value;
  }
  return result;
}

template <typename T>
std::vector<T> Range(T start, T end, T step = 1) {
  std::vector<T> result;
  result.reserve(static_cast<size_t>((end - start) / step));
  if (start < end) {
    for (; start < end; start = start + step) {
      result.push_back(start);
    }
  } else {
    for (; start > end; start = start + step) {
      result.push_back(start);
    }
  }
  return result;
}

template <typename T, typename S>
std::vector<T> ToVector(const S& input) {
  return std::vector<T>(input.begin(), input.end());
}

template <typename T, typename S>
bool Equal(const T& v1, const S& v2) {
  return std::equal(v1.begin(), v1.end(), v2.begin());
}

template <typename T>
typename T::mapped_type FindOr(const T& cont, const typename T::key_type& key,
                               const typename T::mapped_type& defval) {
  auto it = cont.find(key);
  return it != cont.end() ? it->second : defval;
}

template <typename T, typename G>
const typename T::mapped_type& MapInsert(T* cont,
                                         const typename T::key_type& key,
                                         const G& gen) {
  auto it = cont->find(key);
  if (it == cont->end()) {
    it = cont->emplace(key, gen()).first;
  }
  return it->second;
}

template <typename T>
typename std::underlying_type<T>::type GetEnumValue(T value) {
  return static_cast<typename std::underlying_type<T>::type>(value);
}

template <typename T, typename S>
T Multiply(const S& input) {
  return std::accumulate(input.begin(), input.end(), T(1),
                         std::multiplies<T>());
}

template <typename T>
std::string VecToStr(const std::vector<T>& vec, const char* sep = " ") {
  if (vec.empty()) {
    return "";
  }
  std::stringstream result;
  std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(result, sep));
  result << std::to_string(vec.back());
  return result.str();
}

template <>
TORCH_GCU_API std::string VecToStr(const std::vector<uint8_t>& vec,
                                   const char* sep);

template <>
TORCH_GCU_API std::string VecToStr(const std::vector<int8_t>& vec,
                                   const char* sep);

TORCH_GCU_API std::vector<std::string> StrSplit(std::string text, char delim);

TORCH_GCU_API std::string StrJoin(std::vector<std::string> strs,
                                  std::string separator);

// `AlphaNum` acts as the main parameter type for `StrCat()` and `StrAppend()`,
class AlphaNum {
 public:
  // No bool ctor -- bools convert to an integral type.
  // A bool ctor would also convert incoming pointers (bletch).

  AlphaNum(int x) : digits_(std::to_string(x)) {}
  AlphaNum(unsigned int x) : digits_(std::to_string(x)) {}
  AlphaNum(long x) : digits_(std::to_string(x)) {}
  AlphaNum(unsigned long x) : digits_(std::to_string(x)) {}
  AlphaNum(long long x) : digits_(std::to_string(x)) {}
  AlphaNum(unsigned long long x) : digits_(std::to_string(x)) {}

  AlphaNum(float f) : digits_(std::to_string(f)) {}
  AlphaNum(double f) : digits_(std::to_string(f)) {}

  AlphaNum(const char* c_str) : digits_(c_str) {}

  template <typename Allocator>
  AlphaNum(
      const std::basic_string<char, std::char_traits<char>, Allocator>& str)
      : digits_(str) {}

  // Use string literals ":" instead of character literals ':'.
  AlphaNum(char c) = delete;

  AlphaNum(const AlphaNum&) = delete;
  AlphaNum& operator=(const AlphaNum&) = delete;

  // Normal enums are already handled by the integer formatters.
  // This overload matches only scoped enums.
  template <typename T,
            typename = typename std::enable_if<
                std::is_enum<T>{} && !std::is_convertible<T, int>{}>::type>
  AlphaNum(T e)
      : AlphaNum(static_cast<typename std::underlying_type<T>::type>(e)) {}

  std::string::size_type size() const { return digits_.size(); }
  const char* data() const { return digits_.data(); }
  std::string Piece() const { return digits_; }

  // vector<bool>::reference and const_reference require special help to
  // convert to `AlphaNum` because it requires two user defined conversions.
  template <
      typename T,
      typename std::enable_if<
          std::is_class<T>::value &&
          (std::is_same<T, std::vector<bool>::reference>::value ||
           std::is_same<T, std::vector<bool>::const_reference>::value)>::type* =
          nullptr>
  AlphaNum(T e) : AlphaNum(static_cast<bool>(e)) {}

 private:
  std::string digits_;
};

// Merges given strings or numbers, using no delimiter(s), returning the merged
// result as a string.
inline std::string StrCat() { return std::string(); }

template <typename... AV>
inline std::string StrCat(const AV&... args) {
  std::string result;
  for (const auto& arg : {static_cast<const AlphaNum&>(args).Piece()...}) {
    result.append(arg.begin(), arg.end());
  }
  return result;
}

// Appends a string or set of strings to an existing string, in a similar
// fashion to `StrCat()`.
inline void StrAppend(std::string*) {}

template <typename... AV>
inline void StrAppend(std::string* dest, const AV&... args) {
  for (const auto& arg : {static_cast<const AlphaNum&>(args).Piece()...}) {
    dest->append(arg.data(), arg.size());
  }
}

// Replace all 'from' using 'to' in given string
void StrReplaceAll(std::string& str, const std::string& from,
                   const std::string& to);

// Replace all 'from' using 'to' in given string, return a new string
std::string StrReplaceAll(const std::string& str, const std::string& from,
                          const std::string& to);

template <typename S>
std::vector<int64_t> I64List(const S& input) {
  return ToVector<int64_t>(input);
}

int64_t ElementsIn(const std::vector<int64_t>& shape);

bool HasZeroShape(const std::vector<int64_t>& shape);

GCU_TORCH_GCU_UTIL_NS_END
