#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <ATen/EmptyTensor.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/Storage.h>

#include "aten/aot_ops/gcu_ops.h"

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
#include <ATen/ops/_pin_memory_ops.h>
#include <ATen/ops/is_pinned_ops.h>
#endif

#include "gcu/gcu_exception.h"
#include "gcu/gcu_hooks.h"

namespace torch_gcu {

namespace aotops {

bool is_pinned(const at::Tensor& self, c10::optional<at::Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() ||
                                   device->is_privateuseone());
  return detail::getGCUHooks().isPinnedPtr(self.storage().data());
}

at::Tensor _pin_memory(const at::Tensor& self,
                       c10::optional<at::Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() ||
                                   device->is_privateuseone());
  auto* allocator = detail::getGCUHooks().getPinnedMemoryAllocator();
  auto storage =
      c10::Storage(c10::Storage::use_byte_size_t(),
                   at::detail::computeStorageNbytes(
                       self.sizes(), self.strides(), self.dtype().itemsize()),
                   allocator,
                   /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options())
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

}  // namespace aotops

namespace cpu {

c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    return detail::getGCUHooks().getPinnedMemoryAllocator();
  }
  return c10::GetCPUAllocator();
}

at::TensorBase empty_cpu(c10::IntArrayRef size, at::ScalarType dtype,
                         bool pin_memory,
                         c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_generic(size, allocator, cpu_ks, dtype,
                                   memory_format_opt);
}

at::TensorBase empty_cpu(c10::IntArrayRef size,
                         c10::optional<at::ScalarType> dtype_opt,
                         c10::optional<at::Layout> layout_opt,
                         c10::optional<at::Device> device_opt,
                         c10::optional<bool> pin_memory_opt,
                         c10::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() ==
                                   at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) ==
                                   at::Layout::Strided);

  auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_cpu(size, dtype, pin_memory, memory_format_opt);
}

at::TensorBase empty_cpu(c10::IntArrayRef size,
                         const at::TensorOptions& options) {
  return empty_cpu(size, c10::optTypeMetaToScalarType(options.dtype_opt()),
                   options.layout_opt(), options.device_opt(),
                   options.pinned_memory_opt(), options.memory_format_opt());
}

at::TensorBase empty_strided_cpu(c10::IntArrayRef size, c10::IntArrayRef stride,
                                 at::ScalarType dtype, bool pin_memory) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_strided_generic(size, stride, allocator, cpu_ks,
                                           dtype);
}

at::TensorBase empty_strided_cpu(c10::IntArrayRef size, c10::IntArrayRef stride,
                                 c10::optional<at::ScalarType> dtype_opt,
                                 c10::optional<at::Layout> layout_opt,
                                 c10::optional<at::Device> device_opt,
                                 c10::optional<bool> pin_memory_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() ==
                                   at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) ==
                                   at::Layout::Strided);

  auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_strided_cpu(size, stride, dtype, pin_memory);
}

at::TensorBase empty_strided_cpu(c10::IntArrayRef size, c10::IntArrayRef stride,
                                 const at::TensorOptions& options) {
  return empty_strided_cpu(
      size, stride, c10::optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}

at::Tensor empty_memory_format(
    c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  at::Tensor result = empty_cpu(size, dtype_opt, layout_opt, device_opt,
                                pin_memory_opt, memory_format_opt);
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(result);
  }
  return result;
}

at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                         c10::optional<at::ScalarType> dtype_opt,
                         c10::optional<at::Layout> layout_opt,
                         c10::optional<at::Device> device_opt,
                         c10::optional<bool> pin_memory_opt) {
  at::Tensor result = empty_strided_cpu(size, stride, dtype_opt, layout_opt,
                                        device_opt, pin_memory_opt);
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(result);
  }
  return result;
}

bool is_pinned(const at::Tensor& self, c10::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  c10::DispatchKeySet _dk = c10::DispatchKeySet(
      c10::computeDispatchKey(c10::nullopt, self.layout(),
                              device.value_or(c10::DeviceType::PrivateUse1)));
  return at::_ops::is_pinned::redispatch(_dk, self, device);
}

at::Tensor _pin_memory(const at::Tensor& self,
                       c10::optional<at::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(),
              "' only dense CPU tensors can be pinned");
  c10::DispatchKeySet _dk = c10::DispatchKeySet(
      c10::computeDispatchKey(c10::nullopt, self.layout(),
                              device.value_or(c10::DeviceType::PrivateUse1)));
  return at::_ops::_pin_memory::redispatch(_dk, self, device);
}

IGNORE_OVERRIDE_OPERATOR_WARNING

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
  m.impl("empty_strided", TORCH_FN(empty_strided));
}

TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"), TORCH_FN(is_pinned));
  m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"), TORCH_FN(_pin_memory));
}

RESTORE_OVERRIDE_OPERATOR_WARNING

}  // namespace cpu

}  // namespace torch_gcu