/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "aten/op_statistics.h"

#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "ATen/core/List.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/ops/_to_cpu.h"
#include "aten/aten_cpu_fallback.h"
#include "aten/op_debug_config.h"
#include "gcu/logging.h"
#include "gcu/sys_util.h"

namespace torch_gcu {

namespace {

std::string aten_op_name_to_topsaten_name(const std::string& aten_op_name) {
  auto pos = aten_op_name.find("::");
  auto name_space = aten_op_name.substr(0, pos);
  auto op_name = aten_op_name.substr(pos + 2);
  pos = op_name.find(".");
  auto root_name = op_name.substr(0, pos);
  bool is_private = (root_name.find("_") == 0);
  bool is_inplace = (root_name.find_last_of("_") == (root_name.size() - 1));
  if (root_name[root_name.size() - 1] != '_') {
    root_name += '_';
  }
  std::string topsaten_name;
  auto start = 0;
  pos = -1;
  while ((pos = root_name.find_first_of('_', pos + 1)) != std::string::npos) {
    auto sub_name = root_name.substr(start, pos - start);
    start = pos + 1;
    if (!sub_name.empty()) {
      std::transform(sub_name.begin(), (sub_name.begin() + 1), sub_name.begin(),
                     ::toupper);
      topsaten_name = topsaten_name + sub_name;
    }
  }

  topsaten_name = "tops" + name_space + topsaten_name;

  if (is_private) {
    topsaten_name = "_" + topsaten_name;
  }
  if (is_inplace) {
    topsaten_name = topsaten_name + "_";
  }
  return topsaten_name;
}

}  // namespace

// clang-format off

/**
 * @brief op those output shape is not determined by input shape
 * can not run on meta device
 */
const std::unordered_set<std::string> dynamic_op = {
    "aten::argwhere",
    "aten::bincount",
    "aten::bincount.out",
    "aten::_ctc_loss",
    "aten::_ctc_loss.out",
    "aten::_ctc_loss.Tensor",
    "aten::_ctc_loss.Tensor_out",
    "aten::index.Tensor",
    "aten::index.Tensor_out",
    "aten::linalg_lstsq",
    "aten::linalg_lstsq.out",
    "aten::masked_select.out",
    "aten::masked_select",
    "aten::nonzero.out",
    "aten::nonzero",
    "aten::one_hot",
    "aten::repeat_interleave.Tensor",
    "aten::repeat_interleave.Tensor_out",
    "aten::repeat_interleave.self_Tensor",
    "aten::unique_dim",
    "aten::unique_dim.out",
    "aten::unique_consecutive",
    "aten::unique_consecutive.out",
    "aten::unique_dim_consecutive",
    "aten::unique_dim_consecutive.out",
    "aten::_unique",
    "aten::_unique.out",
    "aten::_unique2",
    "aten::_unique2.out"
};

// clang-format on

TensorInfo::TensorInfo(const at::Tensor& t) {
  if (t.defined()) {
    _dtype = t.scalar_type();
    _shape = t.sizes().vec();
    _stride = t.strides().vec();
    _define = true;
    _is_contiguous = t.is_contiguous();
  }
}

TensorInfo::TensorInfo(const c10::optional<at::Tensor>& t) {
  if (t.has_value() && t.value().defined()) {
    _dtype = t->scalar_type();
    _shape = t->sizes().vec();
    _stride = t->strides().vec();
    _define = true;
    _is_contiguous = t->is_contiguous();
  }
}

StorageInfo::StorageInfo(const at::Storage& s) {
  _nbytes = s.nbytes();
  _resizable = s.resizable();
}

MetaIValue::MetaIValue(const at::IValue& ivalue) {
  if (ivalue.isTensor()) {
    _tvec.push_back(ivalue.toTensor());
    _tag = MetaIValueType::kTensor;
  } else if (ivalue.isTensorList()) {
    auto tvec = ivalue.toTensorVector();
    for (const auto& t : tvec) {
      _tvec.push_back(t);
    }
    _tag = MetaIValueType::kTensorList;
  } else if (ivalue.isOptionalTensorList()) {
    auto opt_tvec = ivalue.toOptionalTensorVector();
    for (const auto& opt_t : opt_tvec) {
      _tvec.push_back(opt_t);
    }
    _tag = MetaIValueType::kOptionalTensorList;
  } else if (ivalue.isStorage()) {
    _s = ivalue.toStorage();
    _tag = MetaIValueType::KStorage;
  } else {
    _v = ivalue.deepcopy();
    _tag = MetaIValueType::kIValue;
  }
}

at::IValue MetaIValue::toIValue() const {
  PTCHECK(_tag == MetaIValueType::kIValue);
  return _v;
}

TensorInfo MetaIValue::toTensorInfo() const {
  PTCHECK(_tag == MetaIValueType::kTensor);
  PTCHECK(_tvec.size() == 1) << "More than one tensor when tag is Tensor";
  return _tvec[0];
}

StorageInfo MetaIValue::toStorageInfo() const { return _s; }

std::vector<TensorInfo> MetaIValue::toTensorInfoList() const {
  PTCHECK(_tag == MetaIValueType::kTensorList ||
          _tag == MetaIValueType::kOptionalTensorList);
  // PTCHECK(!_tvec.empty()) << "No tensor when tag is TensorList";
  return _tvec;
}

Json::Value TensorToJson(const TensorInfo& t) {
  Json::Value tensor;
  auto json_vec = [](const std::vector<int64_t>& vec) -> Json::Value {
    Json::Value vec_json(Json::arrayValue);
    for (int64_t i : vec) {
      vec_json.append(i);
    }
    return vec_json;
  };
  if (t.isDefined()) {
    tensor["dtype"] = c10::toString(t.dtype());
    tensor["size"] = json_vec(t.shape());
    tensor["strides"] = json_vec(t.strides());
    tensor["is_contiguous"] = t.isContiguous();
  } else {
    tensor["dtype"] = "undef";
  }
  return tensor;
}

Json::Value StorageToJson(const StorageInfo& s) {
  Json::Value storage;
  storage["nbytes"] = s.nbytes();
  storage["resizable"] = s.resizable();
  return storage;
}

void NoListIValueToJson(const at::IValue& i_val, Json::Value& j_value) {
  if (i_val.isScalar()) {
    if (i_val.isDouble()) {
      j_value["value"] = i_val.toDouble();
    } else if (i_val.isInt()) {
      j_value["value"] = i_val.toInt();
    } else if (i_val.isBool()) {
      j_value["value"] = i_val.toBool();
    } else {
      std::stringstream ss;
      ss << i_val;
      j_value["value"] = ss.str();
    }
  } else {
    std::stringstream ss;
    ss << i_val;
    j_value["value"] = ss.str();
  }
}

Json::Value ListToJsonValue(const c10::List<at::IValue> list_val) {
  Json::Value elements(Json::arrayValue);
  auto vec_val = list_val.vec();
  for (auto val_element : vec_val) {
    if (val_element.isScalar()) {
      if (val_element.isDouble()) {
        elements.append(val_element.toDouble());
      } else if (val_element.isInt()) {
        elements.append(val_element.toInt());
      } else if (val_element.isBool()) {
        elements.append(val_element.toBool());
      } else {
        std::stringstream ss;
        ss << val_element;
        elements.append(ss.str());
      }
    } else {
      std::stringstream ss;
      ss << val_element;
      elements.append(ss.str());
    }
  }
  return elements;
}

Json::Value listScalarToJsonValue(const c10::List<at::IValue> list_val) {
  Json::Value elements(Json::arrayValue);
  auto vec_val = list_val.vec();
  for (auto val_element : vec_val) {
    PTCHECK(val_element.isScalar())
        << "Input of listScalarToString is not Scalar";
    Json::Value element;
    std::string type_str = "Scalar";
    type_str += "(" + val_element.type()->repr_str() + ")";
    element["dtype"] = type_str;
    if (val_element.isDouble()) {
      element["value"] = val_element.toDouble();
    } else if (val_element.isInt()) {
      element["value"] = val_element.toInt();
    } else if (val_element.isBool()) {
      element["value"] = val_element.toBool();
    } else {
      std::stringstream ss;
      ss << val_element;
      element["value"] = ss.str();
    }

    elements.append(element);
  }
  return elements;
}

Json::Value ListToJsonValue(const c10::List<at::IValue> list_val,
                            bool is_scalar_list) {
  if (is_scalar_list) {
    return listScalarToJsonValue(list_val);
  } else {
    return ListToJsonValue(list_val);
  }
}

void IValueToJsonValue(const at::Argument& arg, const MetaIValue& m_val,
                       Json::Value& j_value) {
  PTCHECK(m_val.isIValue()) << "Input of listScalarToString is not kIValue";

  auto i_val = m_val.toIValue();
  if (i_val.isList()) {
    bool is_list_scalar =
        arg.type()->kind() == c10::TypeKind::ListType &&
        arg.type()->containedTypes().size() == 1 &&
        arg.type()->containedTypes()[0]->kind() == c10::TypeKind::NumberType;
    j_value["value"] = ListToJsonValue(i_val.toList(), is_list_scalar);
  } else {
    NoListIValueToJson(i_val, j_value);
  }
}

void updateJsonValue(const at::Argument& arg, const MetaIValue& m_val,
                     Json::Value& j_value) {
  Json::Value param;
  if (m_val.isTensor()) {
    auto t = m_val.toTensorInfo();
    j_value["value"] = TensorToJson(t);
  } else if (m_val.isTensorList() || m_val.isOptionalTensorList()) {
    auto tvec = m_val.toTensorInfoList();
    Json::Value list_val(Json::arrayValue);
    for (const auto& t : tvec) {
      list_val.append(TensorToJson(t));
    }
    j_value["value"] = list_val;
  } else if (m_val.isStorage()) {
    auto s = m_val.toStorageInfo();
    j_value["value"] = StorageToJson(s);
  } else if (m_val.isIValue()) {
    IValueToJsonValue(arg, m_val, j_value);
  } else {
    j_value["value"] = "None";
  }
}

std::string toString(const Params& params) {
  std::stringstream ss;
  for (const auto& param : params) {
    ss << toString(param) << ",\n";
  }
  return ss.str();
}

std::string toString(const TensorInfo& t) {
  std::stringstream ss;
  auto print_vec = [](const std::vector<int64_t>& vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); i++) {
      ss << vec[i];
      if (i != vec.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    return ss.str();
  };
  if (t.isDefined()) {
    ss << "shape" << print_vec(t.shape()) << ", strides"
       << print_vec(t.strides()) << ", dtype[" << t.dtype() << "]";
  } else {
    ss << "Undefined Tensor";
  }
  return ss.str();
}

std::string toString(const StorageInfo& s) {
  std::stringstream ss;
  std::string resizable = s.resizable() ? "True" : "False";
  ss << "nbytes:" << s.nbytes() << ", resizable:" << resizable;
  return ss.str();
}

std::string toString(const MetaIValue& val) {
  std::stringstream ss;
  const int w = 50;
  if (val.isTensor()) {
    auto t = val.toTensorInfo();
    ss << toString(t);
  } else if (val.isTensorList() || val.isOptionalTensorList()) {
    auto tvec = val.toTensorInfoList();
    for (const auto& t : tvec) {
      ss << "\n" << std::setw(w) << std::left << "" << toString(t);
    }
  } else if (val.isStorage()) {
    auto s = val.toStorageInfo();
    ss << toString(s);
  } else if (val.isIValue()) {
    ss << val.toIValue();
  } else {
    ss << "None";
  }
  return ss.str();
}

std::string listScalarToString(const MetaIValue& val) {
  std::stringstream ss;
  const int w1 = 30;
  const int w2 = 20;
  PTCHECK(val.isIValue()) << "Input of listScalarToString is not kIValue";

  auto i_val = val.toIValue();
  PTCHECK(i_val.isList()) << "Input of listScalarToString is not List";

  auto list_i_val = i_val.toList().vec();
  for (auto val_element : list_i_val) {
    PTCHECK(val_element.isScalar())
        << "Input of listScalarToString is not Scalar";
    std::string type_str = "Scalar";
    type_str += "(" + val_element.type()->repr_str() + ")";
    ss << "\n"
       << std::setw(w1) << std::left << "" << std::setw(w2) << std::left
       << type_str << val_element;
  }
  return ss.str();
}

std::string toString(const at::Argument& arg, const MetaIValue& val) {
  bool is_list_scalar =
      arg.type()->kind() == c10::TypeKind::ListType &&
      arg.type()->containedTypes().size() == 1 &&
      arg.type()->containedTypes()[0]->kind() == c10::TypeKind::NumberType;
  if (is_list_scalar) {
    return listScalarToString(val);
  } else {
    return toString(val);
  }
}

bool tensorInfoEqual(const TensorInfo& lhs, const TensorInfo& rhs) {
  if (!lhs.isDefined() && !rhs.isDefined()) {
    return true;
  }
  return lhs.isDefined() == rhs.isDefined() && lhs.dtype() == rhs.dtype() &&
         lhs.shape() == rhs.shape() && lhs.strides() == rhs.strides();
}

bool StorageInfoEqual(const StorageInfo& lhs, const StorageInfo& rhs) {
  return lhs.nbytes() == rhs.nbytes() && lhs.resizable() == rhs.resizable();
}

size_t hashTensorInfo(const TensorInfo& t) {
  if (!t.isDefined()) {
    return 0;
  }
  size_t seed = 0;
  seed = at::hash_combine(seed, at::_hash_detail::simple_get_hash(t.shape()));
  seed = at::hash_combine(seed, at::_hash_detail::simple_get_hash(t.strides()));
  seed = at::hash_combine(seed, at::_hash_detail::simple_get_hash(t.dtype()));
  return seed;
}

size_t hashStorageInfo(const StorageInfo& s) {
  size_t seed = 0;
  seed = at::hash_combine(seed, at::_hash_detail::simple_get_hash(s.nbytes()));
  seed =
      at::hash_combine(seed, at::_hash_detail::simple_get_hash(s.resizable()));
  return seed;
}

size_t hashIvalue(const at::IValue& ival) {
  size_t seed = 0;
  if (ival.isList()) {
    auto l = ival.toList().vec();
    for (const auto& e : l) {
      auto tmp = hashIvalue(e);
      seed = at::hash_combine(seed, tmp);
    }
  } else {
    seed = at::_hash_detail::simple_get_hash(ival);
  }
  return seed;
}

bool isWrite(const c10::Argument& arg) {
  auto info = arg.alias_info();
  if (info && info->isWrite()) {
    return true;
  }
  return false;
}

OpStatistics& OpStatistics::getInstance() {
  static OpStatistics instance;
  return instance;
}

OpStatistics::~OpStatistics() {
  if (!_op_map.empty()) {
    PTDLOG(OP) << std::endl << dumpToStr();
    if (torch_gcu::OpDebugConfig::GetInstance().enableOpStatisticsDumpData()) {
      dumpToJson();
    }
  }
}

std::string getOpFullname(const c10::FunctionSchema& op_schema) {
  auto op_fullname = op_schema.name();
  if (!op_schema.overload_name().empty()) {
    op_fullname += "." + op_schema.overload_name();
  }
  return op_fullname;
}

void OpStatistics::record(const c10::FunctionSchema& op_schema,
                          const Params& param) {
  auto& instance = getInstance();
  auto fullname = getOpFullname(op_schema);
  std::lock_guard<std::mutex> lock(instance._mtx);
  // 1. Insert op_schema
  if (instance._op_map.find(op_schema) == instance._op_map.end()) {
    instance._op_map[op_schema] = OpEntry();
  }
  // 2. Insert param
  auto& op_entry = instance._op_map[op_schema];
  auto& param_map = op_entry._param_map;
  if (!param_map.count(param)) {
    param_map[param] = 1;
  } else {
    param_map[param]++;
  }
  op_entry._total_count++;
}

std::string getArgsTypeStr(const at::Argument& arg, const MetaIValue& val) {
  bool is_scalar = arg.type()->kind() == c10::TypeKind::NumberType;
  bool is_list_scalar =
      arg.type()->kind() == c10::TypeKind::ListType &&
      arg.type()->containedTypes().size() == 1 &&
      arg.type()->containedTypes()[0]->kind() == c10::TypeKind::NumberType;
  std::string type_str;
  if (is_scalar) {
    type_str = "Scalar";
    PTCHECK(val.isIValue()) << "Output is not IVlaue";
    type_str += "(" + val.toIValue().type()->repr_str() + ")";
  } else if (is_list_scalar) {
    type_str = "List[Scalar]";
  } else {
    type_str = arg.type()->repr_str();
  }
  return type_str;
}

void OpStatistics::dumpToJson() {
  auto& instance = getInstance();
  std::lock_guard<std::mutex> lock(instance._mtx);
  if (!instance._op_map.empty()) {
    std::string file = torch_gcu::OpDebugConfig::GetInstance().getDumpPath() +
                       "/" + "op_stat_" + std::to_string(getpid()) + "_" +
                       util::GetTimeStamp() + ".json";
    Json::Value root(Json::arrayValue);
    // Level 0: reorder op by alphabet
    std::vector<c10::FunctionSchema> op_list;
    for (auto& op_entry : instance._op_map) {
      op_list.push_back(op_entry.first);
    }
    std::sort(
        op_list.begin(), op_list.end(),
        [](const c10::FunctionSchema& lhs, const c10::FunctionSchema& rhs) {
          return getOpFullname(lhs) < getOpFullname(rhs);
        });
    PTCHECK(op_list.size() == instance._op_map.size())
        << "op_list size not equal to op_map size after reorder";

    // Level 1: op
    for (auto& op_schema : op_list) {
      auto& op_entry = instance._op_map[op_schema];
      auto op_fullname = getOpFullname(op_schema);
      auto& op_param_map = op_entry._param_map;
      auto& op_total_count = op_entry._total_count;
      auto topsaten_name = aten_op_name_to_topsaten_name(op_fullname);
      auto is_dynamic_op = bool(dynamic_op.count(op_fullname));
      auto args = op_schema.arguments();
      auto rets = op_schema.returns();
      Json::Value op;
      op["aten_op_name"] = op_fullname;
      op["topsaten_name"] = topsaten_name;
      op["dynamic_op"] = is_dynamic_op;
      op["schema"] = c10::toString(op_schema);
      op["use_count"] = op_total_count;
      op["params_count"] = op_param_map.size();

      // Level 2: op_param
      size_t param_idx = 0;
      Json::Value params_val(Json::arrayValue);
      for (auto& param_map : op_param_map) {
        param_idx++;
        Json::Value param_val;
        param_val["idx"] = param_idx;
        param_val["use_count"] = param_map.second;

        auto params = param_map.first;

        // 2.1 Print output first, as topsaten op calling convention
        // dynamic op do not have output
        size_t return_num = rets.size();
        if (is_dynamic_op) {
          return_num = 0;
        }

        Json::Value outputs(Json::arrayValue);
        for (size_t i = 0; i < return_num; i++) {
          Json::Value out_put;
          auto ret = rets[i];
          auto prefix =
              ret.name().empty() ? "out" + std::to_string(i) : ret.name();
          auto type = getArgsTypeStr(ret, params[i]);
          out_put["idx"] = i;
          out_put["name"] = prefix;
          out_put["args_type"] = type;
          updateJsonValue(ret, params[i], out_put);
          outputs.append(out_put);
        }
        param_val["outputs"] = outputs;
        // 2.2 Print input second
        // Use another index to record the params number
        size_t j = 0;
        Json::Value inputs(Json::arrayValue);
        for (size_t i = 0; i < args.size(); i++) {
          Json::Value input;
          auto arg = args[i];
          auto param = params[j + return_num];
          auto type = getArgsTypeStr(arg, param);
          input["idx"] = i;
          input["name"] = args[i].name();
          input["isWrite"] = isWrite(arg);
          input["args_type"] = type;
          updateJsonValue(arg, param, input);
          inputs.append(input);
          j++;
        }
        param_val["inputs"] = inputs;
        params_val.append(param_val);
      }
      op["params"] = params_val;
      root.append(op);
    }
    util::SaveJsonToFile(file, root);
  }
}

std::string OpStatistics::dumpToStr() {
  auto& instance = getInstance();
  std::lock_guard<std::mutex> lock(instance._mtx);
  std::stringstream ss;
  std::stringstream s_summary;
  if (instance._op_map.empty()) {
    ss << "Torch_gcu Op Statistics no data!" << std::endl;
    return ss.str();
  }
  s_summary << "Torch_gcu Op Statistics" << std::endl;
  s_summary << "#############################################Summary###########"
               "##################################"
            << std::endl;
  // Level 0: reorder op by alphabet
  std::vector<c10::FunctionSchema> op_list;
  for (auto& op_entry : instance._op_map) {
    op_list.push_back(op_entry.first);
  }
  std::sort(op_list.begin(), op_list.end(),
            [](const c10::FunctionSchema& lhs, const c10::FunctionSchema& rhs) {
              return getOpFullname(lhs) < getOpFullname(rhs);
            });
  PTCHECK(op_list.size() == instance._op_map.size())
      << "op_list size not equal to op_map size after reorder";
  // Level 1: op
  for (auto& op_schema : op_list) {
    ss << "#################################################" << std::endl;
    auto& op_entry = instance._op_map[op_schema];
    auto op_fullname = getOpFullname(op_schema);
    auto& op_param_map = op_entry._param_map;
    auto& op_total_count = op_entry._total_count;
    auto is_dynamic_op = dynamic_op.count(op_fullname);
    // is_dynamic_op = true;
    auto args = op_schema.arguments();
    auto rets = op_schema.returns();
    const int w1 = 30;
    const int w2 = 20;
    ss << std::setw(w1) << std::left << "Op name:" << op_fullname << std::endl;
    ss << std::setw(w1) << std::left
       << "Is dynamic op:" << (is_dynamic_op ? "true" : "false") << std::endl;
    ss << std::setw(w1) << std::left << "Schema:" << op_schema << std::endl;
    ss << std::setw(w1) << std::left << "Use count:" << op_total_count
       << std::endl;
    ss << std::setw(w1) << std::left << "Params count:" << op_param_map.size()
       << std::endl;
    ss << "-------------------------------------------------" << std::endl;
    s_summary << std::setw(w1 * 2) << std::left << "Op name: " + op_fullname
              << "Use count: " << op_total_count << std::endl;
    size_t param_idx = 0;
    // Level 2: op_param
    for (auto& param_map : op_param_map) {
      param_idx++;
      ss << "Param " << param_idx << ", use count " << param_map.second
         << std::endl;
      auto params = param_map.first;
      // 2.1 Print output first, as topsaten op calling convention
      // dynamic op do not have output
      size_t return_num = rets.size();
      if (is_dynamic_op) {
        return_num = 0;
      } else {
        for (size_t i = 0; i < return_num; i++) {
          auto ret = rets[i];
          auto prefix =
              ret.name().empty() ? "out" + std::to_string(i) : ret.name();
          // NOTE: print Scalar instead of number, and print actual type
          auto type = getArgsTypeStr(ret, params[i]);
          ss << std::setw(w1) << std::left << prefix << std::setw(w2)
             << std::left << type << toString(ret, params[i]) << std::endl;
        }
      }
      // 2.2 Print input second
      // NOTE: params number do not equal to args number, because isWrite args
      // do not need to be recorded
      // Use another index to record the params number
      size_t j = 0;
      for (size_t i = 0; i < args.size(); i++) {
        auto arg = args[i];
        auto param = params[j + return_num];
        auto type = getArgsTypeStr(arg, param);
        ss << std::setw(w1) << std::left << args[i].name() << std::setw(w2)
           << std::left << type << toString(arg, param) << std::endl;
        j++;
      }
      if (param_idx < op_param_map.size()) {
        ss << "- - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
      }
    }
  }
  s_summary << "###############################################################"
               "##################################"
            << std::endl;
  ss << "#################################################" << std::endl;
  return s_summary.str() + ss.str();
}

void OpStatistics::clear() {
  auto& instance = getInstance();
  std::lock_guard<std::mutex> lock(instance._mtx);
  instance._op_map.clear();
}

using OptionalTensorList = c10::ArrayRef<c10::optional<at::Tensor>>;

// TODO: remove when cpu_fallback support optional tensor list
// static std::vector<c10::optional<at::Tensor>> to_cpu(
//    const OptionalTensorList& tensors) {
//  std::vector<c10::optional<at::Tensor>> cpu_tensors(tensors.size());
//  std::vector<at::Tensor> valid_tensors;
//  std::vector<bool> to_translate(tensors.size());
//  for (const auto i : c10::irange(tensors.size())) {
//    const c10::optional<at::Tensor>& tensor = tensors[i];
//    if (tensor.has_value() && tensor.value().defined()) {
//      to_translate[i] = true;
//      valid_tensors.push_back(tensor.value());
//    } else {
//      cpu_tensors[i] = tensor;
//    }
//  }
//  auto cpu_valid_tensors = at::_to_cpu(valid_tensors);
//  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
//    if (to_translate[i]) {
//      cpu_tensors[i] = std::move(cpu_valid_tensors[defined_pos++]);
//    }
//  }
//  return cpu_tensors;
//}

at::IValue deepcopyIValue(const at::IValue& ival) {
  at::IValue copy;
  if (ival.isTensor()) {
    const at::Tensor& src_tensor = ival.toTensor();
    copy = src_tensor.defined() && !src_tensor.device().is_meta()
               ? at::IValue(src_tensor.clone())
               : at::IValue(at::Tensor());
  } else if (ival.isTensorList()) {
    const auto& src_list = ival.toTensorList().vec();
    std::vector<at::Tensor> dst_list;
    dst_list.reserve(src_list.size());
    for (const auto& src_tensor : src_list) {
      dst_list.push_back(src_tensor.defined() && !src_tensor.device().is_meta()
                             ? src_tensor.clone()
                             : at::Tensor());
    }
    copy = at::IValue(dst_list);
  } else if (ival.isOptionalTensorList()) {
    const auto& src_list = ival.toOptionalTensorList().vec();
    std::vector<c10::optional<at::Tensor>> dst_list;
    dst_list.reserve(src_list.size());
    for (const auto& src : src_list) {
      if (src.has_value()) {
        auto src_tensor = src.value();
        if (src_tensor.defined() && !src_tensor.device().is_meta()) {
          dst_list.push_back(src_tensor.clone());
        } else {
          dst_list.push_back(at::Tensor());
        }
      } else {
        dst_list.push_back(c10::nullopt);
      }
    }
    copy = at::IValue(dst_list);
  } else {
    copy = ival.deepcopy();
  }
  return copy;
}

Params op_record_input(const std::vector<c10::IValue>& arguments) {
  Params params;
  for (const auto& ivalue : arguments) {
    params.push_back(ivalue);
  }
  return params;
}

void op_record_output(const c10::FunctionSchema& schema,
                      const std::vector<c10::IValue>& return_arguments,
                      Params params) {
  auto op_fullname = getOpFullname(schema);
  bool is_dynamic_op = dynamic_op.count(op_fullname);

  if (!is_dynamic_op) {
    for (const auto idx : c10::irange(return_arguments.size())) {
      const auto ivalue = return_arguments[idx];
      // NOTE:Insert output info in head of stack, because output is head
      // of all params in topsaten op call
      params.insert(params.begin() + idx, ivalue);
    }
  }

  // Step 4: Record op info in OpStatistics
  return OpStatistics::record(schema, params);
}

void op_record(const c10::FunctionSchema& schema,
               const std::vector<c10::IValue>& input_arguments,
               const std::vector<c10::IValue>& return_arguments) {
  Params params = op_record_input(input_arguments);
  return op_record_output(schema, return_arguments, params);
}

void op_record_mutable(const c10::OperatorHandle& op,
                       torch::jit::Stack* stack) {
  auto schema = op.schema();

  const auto num_rets = schema.returns().size();
  const auto num_args = schema.arguments().size();

  // Step 1: Save arguments info to params
  auto arguments = torch::jit::last(stack, num_args);
  Params params = op_record_input(arguments.vec());

  // Step 2: Call the fallback function to get output info
  try {
    // gcu_cpu_fallback(op, stack);
    gcu_cpu_fallback(op, stack);
  } catch (const c10::Error& error) {
    // Step 3: Save string error for output info to params,
    // and record op info in OpStatistics
    op_record_output(schema, {at::IValue("cpu fallback fail NoValue.")},
                     params);
    throw error;
  }

  // Step 3: Save output info to params, only when op is not dynamic
  // and record op info in OpStatistics
  auto returns = torch::jit::last(stack, num_rets);
  op_record_output(schema, returns.vec(), params);
}
}  // namespace torch_gcu
