/*
 * Copyright 2025 Enflame. All Rights Reserved.
 */

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <map>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

enum class FallbackType {
  kNone = 0,
  kLimited = 1,
  kAll = 2,
};

template <typename T>
std::vector<std::string> getFallbackType(std::string arg_name, const T& var) {
  std::vector<std::string> types;
  types.push_back(arg_name + "::" + std::to_string(var));
  return types;
}

// at::Tensor
template <>
std::vector<std::string> getFallbackType(std::string arg_name,
                                         const at::Tensor& tensor);

// c10::optional<at::Tensor>
template <>
std::vector<std::string> getFallbackType(
    std::string arg_name, const c10::optional<at::Tensor>& tensor);

// at::TensorList
template <>
std::vector<std::string> getFallbackType(std::string arg_name,
                                         const at::TensorList& tensors);

// c10::List<::std::optional<Tensor>>
template <>
std::vector<std::string> getFallbackType(
    std::string arg_name, const c10::List<c10::optional<at::Tensor>>& tensors);

// at::ITensorListRef
template <>
std::vector<std::string> getFallbackType(std::string arg_name,
                                         const at::ITensorListRef& tensors);

template <>
std::vector<std::string> getFallbackType(std::string arg_name,
                                         const c10::string_view& var);

class RegisterFallBackOps {
 public:
  static RegisterFallBackOps& GetInstance();
  void registerFallbackOps(const std::string& op_name,
                           const std::vector<std::string>& fallbacktype);
  FallbackType getBackendFallbackType(const std::string& op_name);
  bool isLimitedFallback(const std::string& op_name,
                         std::vector<std::string> types);

 private:
  RegisterFallBackOps();
  RegisterFallBackOps(const RegisterFallBackOps&) = delete;
  RegisterFallBackOps(RegisterFallBackOps&&) = delete;
  RegisterFallBackOps& operator=(const RegisterFallBackOps&) = delete;
  RegisterFallBackOps& operator=(RegisterFallBackOps&&) = delete;
  void initHardware();
  void initRegisterMap();
  void initS60Register();
  void initL600Register();

 private:
  std::map<std::string, std::map<std::string, bool>> args_fallback_;
  std::map<std::string, FallbackType> backend_fallback_;
  BackendType hardware_ = BackendType::KNone;
};

std::vector<std::string> concatArgs(std::vector<std::vector<std::string>> args);

template <typename... Args>
bool isFallback(const std::string& op_name,
                const std::tuple<std::string, Args>&... args) {
  auto& fallback_ops = RegisterFallBackOps::GetInstance();
  auto fallback_type = fallback_ops.getBackendFallbackType(op_name);
  switch (fallback_type) {
    case FallbackType::kNone:
      return false;
    case FallbackType::kAll:
      return true;
    case FallbackType::kLimited:
      return fallback_ops.isLimitedFallback(
          op_name, concatArgs({getFallbackType(std::get<0>(args),
                                               std::get<1>(args))...}));
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported fallback type.");
      return false;
  }
}

}  // namespace torch_gcu