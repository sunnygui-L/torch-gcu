/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "aten/register_fallback_ops.h"

namespace torch_gcu {

// at::Tensor
template <>
std::vector<std::string> getFallbackType(std::string arg_name,
                                         const at::Tensor& tensor) {
  std::vector<std::string> types;
  auto dtype = std::string(c10::toString(tensor.scalar_type()));
  types.push_back(arg_name + "::" + dtype);
  return types;
}

// c10::optional<at::Tensor>
template <>
std::vector<std::string> getFallbackType(
    std::string arg_name, const c10::optional<at::Tensor>& tensor) {
  if (!tensor.has_value()) {
    return {};
  }
  return getFallbackType(arg_name, tensor.value());
}

// at::TensorList
template <>
std::vector<std::string> getFallbackType(std::string arg_name,
                                         const at::TensorList& tensors) {
  auto tensor_vec = tensors.vec();
  std::vector<std::string> types;
  types.reserve(tensor_vec.size());
  for (size_t i; i < tensor_vec.size(); ++i) {
    auto dtype = std::string(c10::toString(tensor_vec[i].scalar_type()));
    types[i] = arg_name + "::" + dtype;
  }
  return types;
}

// c10::List<::std::optional<Tensor>>
template <>
std::vector<std::string> getFallbackType(
    std::string arg_name, const c10::List<c10::optional<at::Tensor>>& tensors) {
  std::vector<std::string> types;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].has_value()) {
      auto dtype = std::string(c10::toString(tensors[i].value().scalar_type()));
      types.push_back(arg_name + "::" + dtype);
    }
  }
  return types;
}

// at::ITensorListRef
template <>
std::vector<std::string> getFallbackType(std::string arg_name,
                                         const at::ITensorListRef& tensors) {
  auto materialized = tensors.materialize();
  std::vector<std::string> types;
  types.reserve(materialized.size());
  for (size_t i = 0; i < materialized.size(); ++i) {
    const at::Tensor& tensor = materialized[i];
    auto dtype = std::string(c10::toString(tensor.scalar_type()));
    types[i] = arg_name + "::" + dtype;
  }
  return types;
}

// c10::string_view
template <>
std::vector<std::string> getFallbackType(std::string arg_name,
                                         const c10::string_view& var) {
  std::vector<std::string> types;
  std::stringstream ss;
  ss << std::basic_string_view<char>(var.data(), var.size());
  types.push_back(arg_name + "::" + ss.str());
  return types;
}

RegisterFallBackOps& RegisterFallBackOps::GetInstance() {
  static RegisterFallBackOps rf_ops;
  return rf_ops;
}

void RegisterFallBackOps::registerFallbackOps(
    const std::string& op_name, const std::vector<std::string>& fallbacktype) {
  if (fallbacktype.empty()) {
    backend_fallback_[op_name] = FallbackType::kNone;
  } else {
    if (fallbacktype.size() == 1 && fallbacktype[0] == "all") {
      backend_fallback_[op_name] = FallbackType::kAll;
      return;
    }
    backend_fallback_[op_name] = FallbackType::kLimited;
    for (auto& type : fallbacktype) {
      args_fallback_[op_name][type] = true;
    }
  }
}

FallbackType RegisterFallBackOps::getBackendFallbackType(
    const std::string& op_name) {
  return backend_fallback_[op_name];
}

bool RegisterFallBackOps::isLimitedFallback(const std::string& op_name,
                                            std::vector<std::string> types) {
  std::set<std::string> types_set(types.begin(), types.end());
  auto op_map = args_fallback_[op_name];
  for (auto& type : types_set) {
    auto it = op_map.find(type);
    if (it != op_map.end()) {
      return true;
    }
  }
  return false;
}

RegisterFallBackOps::RegisterFallBackOps() {
  initHardware();
  initRegisterMap();
}

void RegisterFallBackOps::initHardware() {
  auto& hardware = HardwareType::GetInstance();
  hardware_ = hardware.getHardware();
}

void RegisterFallBackOps::initRegisterMap() {
  switch (hardware_) {
    case BackendType::kS60:
      initS60Register();
      break;
    case BackendType::kL600:
      initL600Register();
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported hardware.");
      break;
  }
}

std::vector<std::string> concatArgs(
    std::vector<std::vector<std::string>> args) {
  std::vector<std::string> types;
  for (auto& arg : args) {
    types.insert(types.end(), arg.begin(), arg.end());
  }
  return types;
}

}  // namespace torch_gcu
