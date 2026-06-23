/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/core/function_schema.h>
#include <json/json.h>

#include <iostream>
#include <mutex>
#include <unordered_map>

#include "gcu/gcu_macros.h"
namespace torch_gcu {

/**
 * @class TensorInfo
 * @brief save tensor info for meta tensor in OpStatistics
 * Including shape, stride and dtype
 * Use undefine to represent undefined tensor
 */
class TensorInfo {
 public:
  TensorInfo() = default;

  TensorInfo(const TensorInfo&) = default;

  TensorInfo(const at::Tensor& t);

  TensorInfo(const c10::optional<at::Tensor>& t);

  std::vector<int64_t> shape() const { return _shape; }

  std::vector<int64_t> strides() const { return _stride; }

  at::ScalarType dtype() const { return _dtype; }

  bool isDefined() const { return _define; }

  bool isContiguous() const { return _is_contiguous; }

 private:
  bool _define = false;
  at::ScalarType _dtype = at::ScalarType::Undefined;
  std::vector<int64_t> _shape;
  std::vector<int64_t> _stride;
  bool _is_contiguous = false;
};

/**
 * @class StorageInfo
 * @brief save storage info for storage in OpStatistics
 * Including nbytes and resizable
 */
class StorageInfo {
 public:
  StorageInfo() = default;

  StorageInfo(const StorageInfo&) = default;

  StorageInfo(const at::Storage& s);

  size_t nbytes() const { return _nbytes; }

  bool resizable() const { return _resizable; }

 private:
  size_t _nbytes;
  bool _resizable;
};

enum class MetaIValueType {
  kNone,
  kIValue,
  kTensor,
  kTensorList,
  kOptionalTensorList,
  KStorage
};

/**
 * @class MetaIValue
 * @brief Extend of at::IValue, because at::IValue do not deal with meta tensor
 * well Equal to at::IValue if not tensor, tensor list or optional tensor list
 */
class MetaIValue {
 public:
  MetaIValue(const at::IValue& ivalue);

  bool idDefined() const { return _tag != MetaIValueType::kNone; }

  bool isTensor() const { return _tag == MetaIValueType::kTensor; }

  bool isTensorList() const { return _tag == MetaIValueType::kTensorList; }

  bool isOptionalTensorList() const {
    return _tag == MetaIValueType::kOptionalTensorList;
  }

  bool isStorage() const { return _tag == MetaIValueType::KStorage; }

  bool isIValue() const { return _tag == MetaIValueType::kIValue; }

  MetaIValueType tagKind() const { return _tag; }

  at::IValue toIValue() const;

  TensorInfo toTensorInfo() const;

  std::vector<TensorInfo> toTensorInfoList() const;

  StorageInfo toStorageInfo() const;

 private:
  MetaIValueType _tag = MetaIValueType::kNone;
  at::IValue _v;
  std::vector<TensorInfo> _tvec;
  StorageInfo _s;
};

using Params = std::vector<MetaIValue>;

/**
 * @brief check two tensor info equal
 *      only check shape, stride and dtype
 *      only support meta tensor
 *
 * @param lhs given tensor info
 * @param rhs given tensor info
 * @return true if equal
 */
bool tensorInfoEqual(const TensorInfo& lhs, const TensorInfo& rhs);

/**
 * @brief check two storage info equal
 *      only check nbytes and resizable
 *
 * @param lhs given storage info
 * @param rhs given storage info
 * @return true if equal
 */
bool StorageInfoEqual(const StorageInfo& lhs, const StorageInfo& rhs);

/**
 * @brief convert MetaIValue update to Json::Value for debug
 * creat update MetaIValue to ["value"] target for given Json::Value
 *
 * @param arg which need to the identify param is scalar list
 * @param m_val given IValue
 * @param j_value Json::Value need to be update
 */
void updateJsonValue(const at::Argument& arg, const MetaIValue& m_val,
                     Json::Value& j_value);

/**
 * @brief convert params to string for debug
 * Call toString(MetaIValue) for each param
 *
 * @param params given params
 * @return string humans readable string
 */
std::string toString(const Params& params);

/**
 * @brief convert MetaIValue to string for debug
 * Call toString(TensorInfo) for TensorInfo and TensorInfoList
 * Call IValue's operator<< for other types
 *
 * @param val given IValue
 * @return string humans readable string
 */
std::string toString(const MetaIValue& val);

/**
 * @brief convert TensorInfo to string for debug
 *
 * @param t given tensor info
 * @return string humans readable string
 */
std::string toString(const TensorInfo& t);

/**
 * @brief compute hash value for TensorInfo
 *
 * @param t given meta tensor
 * @return size_t hash value
 */
size_t hashTensorInfo(const TensorInfo& t);

/**
 * @brief compute hash value for StorageInfo
 *
 * @param s given storage
 * @return size_t hash value
 */
size_t hashStorageInfo(const StorageInfo& s);

size_t hashIvalue(const at::IValue& ival);

struct FunctionSchemaHash {
  size_t operator()(const c10::FunctionSchema& fun) const noexcept {
    auto func_str = c10::toString(fun);
    return std::hash<std::string>{}(func_str);
  }
};

}  // namespace torch_gcu

namespace std {

/**
 * @brief specialization std::hash struct for Params, only deal with Tensor,
 * TensorList and optional TensorList, other types will use default c10::hash
 * Needed by std::unordered_map
 */
template <>
struct hash<torch_gcu::Params> : public c10::hash<torch_gcu::Params> {
  size_t operator()(const torch_gcu::Params& params) const {
    size_t seed = 0;
    size_t new_seed = 0;
    for (const auto& elem : params) {
      if (elem.isTensor()) {
        new_seed = torch_gcu::hashTensorInfo(elem.toTensorInfo());
      } else if (elem.isTensorList() || elem.isOptionalTensorList()) {
        auto tl = elem.toTensorInfoList();
        for (const auto& t : tl) {
          new_seed = at::hash_combine(new_seed, torch_gcu::hashTensorInfo(t));
        }
      } else if (elem.isStorage()) {
        new_seed = torch_gcu::hashStorageInfo(elem.toStorageInfo());
      } else {
        auto ival = elem.toIValue();
        new_seed = torch_gcu::hashIvalue(ival);
      }
      seed = at::hash_combine(seed, new_seed);
    }
    return seed;
  }
};

/**
 * @brief specialization std::equal_to struct for Params, only deal with Tensor,
 * TensorList and optional TensorList, other types will use default operator==
 * Needed by std::unordered_map
 */
template <>
struct equal_to<torch_gcu::Params> {
  bool operator()(const torch_gcu::Params& lhs,
                  const torch_gcu::Params& rhs) const {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      auto l = lhs[i];
      auto r = rhs[i];
      if (l.tagKind() != r.tagKind()) {
        return false;
      }
      if (l.isTensor()) {
        auto lt = l.toTensorInfo();
        auto rt = r.toTensorInfo();
        if (!torch_gcu::tensorInfoEqual(lt, rt)) {
          return false;
        }
      } else if (l.isTensorList() || l.isOptionalTensorList()) {
        auto lt = l.toTensorInfoList();
        auto rt = r.toTensorInfoList();
        if (lt.size() != rt.size()) {
          return false;
        }
        for (size_t j = 0; j < lt.size(); j++) {
          if (!torch_gcu::tensorInfoEqual(lt[j], rt[j])) {
            return false;
          }
        }
      } else if (l.isStorage()) {
        auto ls = l.toStorageInfo();
        auto rs = r.toStorageInfo();
        if (torch_gcu::StorageInfoEqual(ls, rs)) {
          return false;
        }
      } else {
        if (l.toIValue() != r.toIValue()) {
          return false;
        }
      }
    }
    return true;
  }
};
}  // namespace std

namespace torch_gcu {

/**
 * @class OpEntry
 * @brief Record op param and count
 *     use unordered_map to record param and count
 *     each param is a vector of IValue
 *     when record, first check if param exist, if not, insert it
 *     then increase count
 */
struct OpEntry {
  std::unordered_map<Params, int> _param_map;
  uint64_t _total_count = 0;
};

/**
 * @class OpStatistics
 * @brief Op statistics manager
 *    use singleton pattern to manage op statistics
 *    use unordered_map to record op and param
 *    key is op schema, value is OpEntry above
 *    when record, first check if op exist, if not, insert it
 *    then record param.
 *
 *    Structure:
 *    OpStatistics
 *    |_op_map<op_schema, OpEntry>
 *      |_op_schema1
 *        |_param1 -> count
 *        |_param2 -> count
 *        |...
 *      |_op_schema2
 *      |...
 */
class TORCH_GCU_API OpStatistics {
 public:
  /**
   * @brief insert op and param to OpStatistics
   *
   * @param op_schema aten op schema
   * @param param call param
   */
  static void record(const c10::FunctionSchema& op_schema, const Params& param);

  /**
   * @brief dump op statistics to json file
   *       file name: op_statistics_<timestamp>.json
   */
  static void dumpToJson();

  /**
   * @brief dump op statistics to string
   *
   * @return std::string op statistics string
   */
  TORCH_GCU_API static std::string dumpToStr();

  /**
   * @brief clear all OpStatistics info
   */
  static void clear();

 private:
  ~OpStatistics();

  OpStatistics() = default;

  OpStatistics(const OpStatistics&) = delete;

  OpStatistics(OpStatistics&&) = delete;

  OpStatistics& operator=(const OpStatistics&) = delete;

  OpStatistics& operator=(OpStatistics&&) = delete;

  static OpStatistics& getInstance();

  std::unordered_map<c10::FunctionSchema, OpEntry, FunctionSchemaHash> _op_map;
  std::mutex _mtx;
};

/**
 * @brief add input arguments to OpStatistics params
 *
 * @param arguments IValue vector of op inputs
 * @return params which recorded input arguments
 */
Params op_record_input(const std::vector<c10::IValue>& arguments);

/**
 * @brief add return arguments to OpStatistics params
 * and record a op call in OpStatistics,
 *
 * @param schema op schema
 * @param return_arguments IValue vector of op returns
 * @param params params which recorded input arguments
 */
void op_record_output(const c10::FunctionSchema& schema,
                      const std::vector<c10::IValue>& return_arguments,
                      Params params);

/**
 * @brief record a op call in OpStatistics without fallback
 *
 * @param schema op schema
 * @param input_arguments IValue vector of op inputs
 * @param return_arguments IValue vector of op returns
 */
void op_record(const c10::FunctionSchema& schema,
               const std::vector<c10::IValue>& input_arguments,
               const std::vector<c10::IValue>& return_arguments);

/**
 * @brief record a op call in OpStatistics, refer to cpu_fallback(), used by
 * call_fallback_fn, this func will change the input stack.
 * Must match the signature
 * void (*)(const c10::OperatorHandle&, c10::Stack*)
 * @param op op handle
 * @param stack op call stack
 */
void op_record_mutable(const c10::OperatorHandle& op, torch::jit::Stack* stack);

/**
 * @brief Check if a op argument is write
 *
 * @param arg argument to check
 * @return true if is write
 */
bool isWrite(const c10::Argument& arg);

/**
 * @brief Get op fullname from op schema
 *
 * @param op_schema op schema
 * @return std::string op fullname
 */
std::string getOpFullname(const c10::FunctionSchema& op_schema);

}  // namespace torch_gcu
