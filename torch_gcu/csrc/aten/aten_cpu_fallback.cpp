#include "aten/aten_cpu_fallback.h"

#include "aten/op_debug_config.h"
#include "aten/op_statistics.h"
#include "gcu/gcu_functions.h"
#include "gcu/logging.h"
#include "gcu/trace.h"
#include "topstx/topstx.hpp"

namespace torch_gcu {

namespace {
std::vector<at::Tensor> to_xdevice(const at::TensorList& tensors,
                                   const c10::Device& xdevice) {
  std::vector<at::Tensor> xdevice_tensors(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    const at::Tensor& tensor = tensors[i];
    if (tensor.defined()) {
      auto cpu_tensor = tensor.detach().cpu();
      if (xdevice.is_cpu()) {
        xdevice_tensors[i] = cpu_tensor;
      } else {
        xdevice_tensors[i] = cpu_tensor.to(xdevice);
      }
    } else {
      xdevice_tensors[i] = tensor;
    }
  }
  return xdevice_tensors;
}

std::vector<c10::optional<at::Tensor>> to_xdevice(
    const std::vector<c10::optional<at::Tensor>>& tensors,
    const c10::Device& xdevice) {
  std::vector<c10::optional<at::Tensor>> xdevice_tensors(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    const c10::optional<at::Tensor>& tensor_opt = tensors[i];
    if (tensor_opt.has_value()) {
      const at::Tensor& tensor = tensor_opt.value();
      if (tensor.defined()) {
        auto cpu_tensor = tensor.detach().cpu();
        if (xdevice.is_cpu()) {
          xdevice_tensors[i] = cpu_tensor;
        } else {
          xdevice_tensors[i] = cpu_tensor.to(xdevice);
        }
      } else {
        xdevice_tensors[i] = tensor;
      }
    } else {
      xdevice_tensors[i] = tensor_opt;
    }
  }
  return xdevice_tensors;
}

void copy_back_to_org_device(
    const c10::OperatorHandle& op, torch::jit::Stack* stack,
    const c10::Device& return_device,
    std::vector<c10::List<c10::optional<at::Tensor>>> opt_tensorlist_args,
    std::vector<int> opt_tensorlist_args_indices,
    std::vector<c10::IValue> opt_tensorlist_cpu_args) {
  auto& schema_args = op.schema().arguments();
  // update input back to org device
  for (const auto i : c10::irange(opt_tensorlist_args_indices.size())) {
    auto tensorlist_idx = opt_tensorlist_args_indices[i];
    const at::AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cpu_tensors =
          opt_tensorlist_cpu_args[i].toOptionalTensorList().vec();
      for (const auto idx : c10::irange(opt_tensorlist_args[i].size())) {
        if (opt_tensorlist_args[i].vec()[idx].has_value()) {
          at::_copy_from_and_resize(cpu_tensors[idx].value(),
                                    opt_tensorlist_args[i].vec()[idx].value());
        }
      }
    }
  }
  // update output back to org device
  for (size_t i = 0; i < stack->size(); ++i) {
    auto v = stack->at(i);
    if (v.isOptionalTensorList()) {
      auto opt_t_list = v.toOptionalTensorList().vec();
      for (size_t j = 0; j < opt_t_list.size(); ++j) {
        auto t_opt = opt_t_list.at(j);
        if (t_opt.has_value()) {
          auto t = t_opt.value();
          if (t.defined() && t.device() != return_device) {
            auto t_return_device = t.to(return_device);
            opt_t_list[j] = t_return_device;
          }
        }
      }
      auto opt_t_list_ivalue =
          at::IValue(c10::List<c10::optional<at::Tensor>>(opt_t_list));
      stack->at(i) = std::move(opt_t_list_ivalue);
    }
  }
}

void convert_tensor_to_f32(
    int64_t num_arguments, torch::jit::Stack* stack,
    std::vector<at::Tensor>* tensor_f32, std::vector<int>* tensor_args_indices,
    std::vector<at::Tensor>* tensorlist_f32,
    std::vector<std::tuple<int, int>>* tensorlist_args_indices,
    std::vector<at::Tensor>* optionalTensorlist_f32,
    std::vector<std::tuple<int, int>>* optionalTensorlist_args_indices) {
  const auto arguments_begin = stack->size() - num_arguments;
  for (int64_t i = 0; i < num_arguments; ++i) {
    auto v = stack->at(arguments_begin + i);
    if (v.isTensor()) {
      auto t = v.toTensor();
      if (t.defined() && t.scalar_type() == at::kHalf) {
        auto t_device = t.device();
        auto t_f32 = t.cpu().to(c10::ScalarType::Float).to(t_device);
        tensor_args_indices->push_back(i);
        tensor_f32->push_back(t_f32);
        stack->at(arguments_begin + i) = at::IValue(t_f32);
      }
    } else if (v.isTensorList()) {
      auto t_list = v.toTensorList().vec();
      for (size_t j = 0; j < t_list.size(); ++j) {
        auto t = t_list.at(j);
        if (t.defined() && t.scalar_type() == at::kHalf) {
          auto t_device = t.device();
          auto t_f32 = t.cpu().to(c10::ScalarType::Float).to(t_device);
          tensorlist_args_indices->push_back({i, j});
          tensorlist_f32->push_back(t_f32);
          t_list[j] = t_f32;
        }
      }
      auto t_list_ivalue = at::IValue(c10::List<at::Tensor>(t_list));
      stack->at(arguments_begin + i) = std::move(t_list_ivalue);
    } else if (v.isOptionalTensorList()) {
      auto tl_opt = v.toOptionalTensorList().vec();
      for (size_t j = 0; j < tl_opt.size(); ++j) {
        auto t_opt = tl_opt.at(j);
        if (t_opt.has_value()) {
          auto t = t_opt.value();
          if (t.defined() && t.scalar_type() == at::kHalf) {
            auto t_device = t.device();
            auto t_f32 = t.cpu().to(c10::ScalarType::Float).to(t_device);
            optionalTensorlist_args_indices->push_back({i, j});
            optionalTensorlist_f32->push_back(t_f32);
            tl_opt[j] = t_f32;
          }
        }
      }
    }
  }
}

void convert_to_meta(int64_t num_arguments, torch::jit::Stack* stack) {
  const auto arguments_begin = stack->size() - num_arguments;
  for (int64_t i = 0; i < num_arguments; ++i) {
    auto v = stack->at(arguments_begin + i);
    if (v.isTensor()) {
      auto t = v.toTensor();
      if (t.defined()) {
        auto meta_tensor = at::empty(t.sizes(), t.options().device(at::kMeta));
        stack->at(arguments_begin + i) = at::IValue(meta_tensor);
      }
    } else if (v.isTensorList()) {
      auto t_list = v.toTensorList().vec();
      for (size_t j = 0; j < t_list.size(); ++j) {
        auto t = t_list.at(j);
        if (t.defined()) {
          t_list[j] = at::empty(t.sizes(), t.options().device(at::kMeta));
        }
      }
      auto t_list_ivalue = at::IValue(c10::List<at::Tensor>(t_list));
      stack->at(arguments_begin + i) = std::move(t_list_ivalue);
    } else if (v.isOptionalTensorList()) {
      auto tl_opt = v.toOptionalTensorList().vec();
      for (size_t j = 0; j < tl_opt.size(); ++j) {
        auto t_opt = tl_opt.at(j);
        if (t_opt.has_value()) {
          auto t = t_opt.value();
          if (t.defined() && t.scalar_type() == at::kHalf) {
            tl_opt[j] = at::empty(t.sizes(), t.options().device(at::kMeta));
          }
        }
      }
      auto t_opt_ivalue =
          at::IValue(c10::List<c10::optional<at::Tensor>>(tl_opt));
      stack->at(arguments_begin + i) = std::move(t_opt_ivalue);
    }
  }
}

}  // namespace

void xdevice_cpu_fallback_with_convert(const c10::OperatorHandle& op,
                                       const torch::jit::Stack& stack_backup,
                                       torch::jit::Stack* stack,
                                       bool is_cpu_cpu_fallback) {
  auto& schema_args = op.schema().arguments();
  const auto num_arguments = op.schema().arguments().size();

  std::vector<at::Tensor> tensor_f32;
  std::vector<int> tensor_args_indices;

  std::vector<at::Tensor> tensorlist_f32;
  std::vector<std::tuple<int, int>> tensorlist_args_indices;

  std::vector<at::Tensor> optionalTensorlist_f32;
  std::vector<std::tuple<int, int>> optionalTensorlist_args_indices;

  // reload backup stack
  for (size_t i = 0; i < num_arguments; ++i) {
    stack->at(i) = stack_backup[i];
  }

  // use meta to get org output dtype
  torch::jit::Stack meta_stack = stack_backup;
  convert_to_meta(num_arguments, &meta_stack);
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::Meta), &meta_stack);

  convert_tensor_to_f32(num_arguments, stack, &tensor_f32, &tensor_args_indices,
                        &tensorlist_f32, &tensorlist_args_indices,
                        &optionalTensorlist_f32,
                        &optionalTensorlist_args_indices);

  if (is_cpu_cpu_fallback) {
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
  } else {
    at::native::cpu_fallback(op, stack);
  }

  // keep input tensor(a!) update data
  for (size_t i = 0; i < tensor_args_indices.size(); ++i) {
    auto tensor_idx = tensor_args_indices[i];
    auto t = stack_backup.at(tensor_idx).toTensor();

    const at::AliasInfo* alias_info = schema_args[tensor_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      auto t32_device = tensor_f32[i].device();
      if (is_cpu_cpu_fallback) {
        t.resize_(tensor_f32[i].sizes())
            .copy_(tensor_f32[i].to(c10::ScalarType::Half));
      } else {
        at::_copy_from_and_resize(
            tensor_f32[i].cpu().to(c10::ScalarType::Half).to(t32_device), t);
      }
    }
  }

  // keep input tensorList update data
  for (size_t i = 0; i < tensorlist_args_indices.size(); ++i) {
    auto tensorlist_idx = std::get<0>(tensorlist_args_indices[i]);
    const at::AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      auto t_list = stack_backup.at(tensorlist_idx).toTensorList().vec();
      auto t32_device = tensorlist_f32[i].device();
      if (is_cpu_cpu_fallback) {
        t_list[std::get<1>(tensorlist_args_indices[i])]
            .resize_(tensorlist_f32[i].sizes())
            .copy_(tensorlist_f32[i].to(c10::ScalarType::Half));
      } else {
        at::_copy_from_and_resize(
            tensorlist_f32[i].cpu().to(c10::ScalarType::Half).to(t32_device),
            t_list[std::get<1>(tensorlist_args_indices[i])]);
      }
    }
  }

  // keep input optionalTensorList update data
  for (size_t i = 0; i < optionalTensorlist_args_indices.size(); ++i) {
    auto opt_tensorlist_idx = std::get<0>(optionalTensorlist_args_indices[i]);
    const at::AliasInfo* alias_info =
        schema_args[opt_tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      auto opt_t_list =
          stack_backup.at(opt_tensorlist_idx).toOptionalTensorList().vec();
      auto t32_device = optionalTensorlist_f32[i].device();
      if (is_cpu_cpu_fallback) {
        opt_t_list[std::get<1>(optionalTensorlist_args_indices[i])]
            .value()
            .resize_(optionalTensorlist_f32[i].sizes())
            .copy_(optionalTensorlist_f32[i].to(c10::ScalarType::Half));
      } else {
        at::_copy_from_and_resize(
            optionalTensorlist_f32[i]
                .cpu()
                .to(c10::ScalarType::Half)
                .to(t32_device),
            opt_t_list[std::get<1>(optionalTensorlist_args_indices[i])]
                .value());
      }
    }
  }

  // keep output dtype same as meta output
  for (size_t i = 0; i < stack->size(); ++i) {
    auto v = stack->at(i);
    auto v_meta = meta_stack.at(i);
    if (v.isTensor()) {
      auto t = v.toTensor();
      auto t_meta = v_meta.toTensor();
      if (t.defined() && t.scalar_type() != t_meta.scalar_type()) {
        auto t_device = t.device();
        auto t_convert = t.cpu().to(t_meta.scalar_type()).to(t_device);
        stack->at(i) = at::IValue(t_convert);
      }
    } else if (v.isTensorList()) {
      auto t_list = v.toTensorList().vec();
      auto t_meta_list = v_meta.toTensorList().vec();
      for (size_t j = 0; j < t_list.size(); ++j) {
        auto t = t_list.at(j);
        auto t_meta = t_meta_list.at(j);
        if (t.defined() && t.scalar_type() != t_meta.scalar_type()) {
          auto t_device = t.device();
          auto t_convert = t.cpu().to(t_meta.scalar_type()).to(t_device);
          t_list[j] = t_convert;
        }
      }
      auto t_list_ivalue = at::IValue(c10::List<at::Tensor>(t_list));
      stack->at(i) = std::move(t_list_ivalue);
    } else if (v.isOptionalTensorList()) {
      auto opt_t_list = v.toOptionalTensorList().vec();
      auto opt_t_meta_list = v_meta.toOptionalTensorList().vec();
      for (size_t j = 0; j < opt_t_list.size(); ++j) {
        auto t_opt = opt_t_list.at(j);
        auto t_opt_meta = opt_t_meta_list.at(j);
        if (t_opt.has_value()) {
          auto t = t_opt.value();
          auto t_meta = t_opt_meta.value();
          if (t.defined() && t.scalar_type() != t_meta.scalar_type()) {
            auto t_device = t.device();
            auto t_convert = t.cpu().to(t_meta.scalar_type()).to(t_device);
            opt_t_list[j] = t_convert;
          }
        }
      }
      auto opt_t_list_ivalue =
          at::IValue(c10::List<c10::optional<at::Tensor>>(opt_t_list));
      stack->at(i) = std::move(opt_t_list_ivalue);
    }
  }
}

void gcu_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto fallback_op_name = c10::toString(op.operator_name());
  TorchTracepoint scoped_fallback_range(TorchTrace::GetTorchTrace().domain());
  if (TorchTrace::is_topstx_enabled()) {
    scoped_fallback_range.enter(TorchTraceCategory::AOTOPS, "cpu_fallback",
                                fallback_op_name.data());
  }

  TorchTracepoint scoped_range(TorchTrace::GetTorchTrace().cpu_domain());
  if (TorchTrace::is_topstx_cpu_domain_enabled()) {
    scoped_range.enter(TorchTraceCategory::USER, fallback_op_name.data());
  }

  PTDLOG(FALLBACK) << c10::toString(op.operator_name())
                   << " not supported in torch_gcu, and fallback to torch cpu.";

  const auto num_arguments = op.schema().arguments().size();

  // convert optional tensorList to cpu, since pytorch not do this.
  const c10::Device& xdevice = c10::Device(c10::DeviceType::CPU);
  std::vector<c10::List<c10::optional<at::Tensor>>> optional_tensorlist_args;
  std::vector<int> optional_tensorlist_args_indices;

  // save converted cpu tensor for optional TensorList
  std::vector<c10::IValue> optional_tensorlist_cpu_args;

  auto arguments = torch::jit::last(stack, num_arguments);
  for (const auto idx : c10::irange(arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isOptionalTensorList()) {
      optional_tensorlist_args.push_back(ivalue.toOptionalTensorList());
      optional_tensorlist_args_indices.push_back(idx);
      auto cpu_ivalue = c10::IValue(c10::List<c10::optional<at::Tensor>>(
          to_xdevice(ivalue.toOptionalTensorList().vec(), xdevice)));
      optional_tensorlist_cpu_args.push_back(cpu_ivalue);
      (*stack)[idx] = std::move(cpu_ivalue);
    }
  }

  torch::jit::Stack stack_backup;
  for (size_t i = 0; i < num_arguments; ++i) {
    stack_backup.push_back(stack->at(i));
  }

  at::Device return_device =
      at::Device(at::DeviceType::PrivateUse1, current_device());

  try {
    at::native::cpu_fallback(op, stack);
    copy_back_to_org_device(op, stack, return_device, optional_tensorlist_args,
                            optional_tensorlist_args_indices,
                            optional_tensorlist_cpu_args);
  } catch (const c10::Error& error) {
    // NOTE: Only deal with runtime error, other error should be throw directly.
    if (typeid(error) != typeid(c10::Error)) {
      throw;
    }
    TORCH_CHECK(
        std::string(error.what_without_backtrace())
                .find("not implemented for 'Half'") != std::string::npos,
        std::string(error.what_without_backtrace()));
    TORCH_WARN("cpu_fallback: ", fallback_op_name,
               " not support Half, convert Half tensor to Float.");

    xdevice_cpu_fallback_with_convert(op, stack_backup, stack);
    copy_back_to_org_device(op, stack, return_device, optional_tensorlist_args,
                            optional_tensorlist_args_indices,
                            optional_tensorlist_cpu_args);
  }
}

void gcu_cpu_fallback_and_statistics(const c10::OperatorHandle& op,
                                     torch::jit::Stack* stack) {
  static bool enable_statistics =
      torch_gcu::OpDebugConfig::GetInstance().enableOpStatistics();
  if (enable_statistics) {
    // NOTE: op_record use gcu_cpu_fallback to get op output info, so we can
    // reuse stack from op_record_mutable to avoid repetitive computation.
    try {
      return op_record_mutable(op, stack);
    } catch (const c10::Error& error) {
      TORCH_WARN("op_statistics cpu_fallback fail: ",
                 c10::toString(op.operator_name()),
                 " will not record out_put params. ",
                 std::string(error.what_without_backtrace()));
    }
  }
  gcu_cpu_fallback(op, stack);
}

bool _has_compatible_shallow_copy_type(const at::Tensor& self,
                                       const at::Tensor& from) {
  c10::DispatchKeySet self_key = self.key_set();
  c10::DispatchKeySet from_key = from.key_set();
  auto is_dense = [](c10::DispatchKeySet ts) {
    return ts.has(c10::DispatchKey::CPU) ||
           ts.has(c10::DispatchKey::PrivateUse1);
  };
  return (self_key == from_key) || (is_dense(self_key) && is_dense(from_key));
}

TORCH_LIBRARY_IMPL(aten, CatchAll, m) {
  m.impl("_has_compatible_shallow_copy_type",
         TORCH_FN(_has_compatible_shallow_copy_type));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &gcu_cpu_fallback_and_statistics>());
}

}  // namespace torch_gcu
