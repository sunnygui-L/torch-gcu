#include "aten/aot_ops/gcu_empty_tensor.h"

#include <ATen/native/TensorFactories.h>

#include "aten/aot_ops/gcu_ops.h"
#include "gcu/gcu_context.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

namespace aotops {

namespace {

inline void raise_warning_for_complex_half(c10::ScalarType dtype) {
  if (dtype == c10::kComplexHalf) {
    TORCH_WARN_ONCE(
        "ComplexHalf support is experimental and many operators don't support "
        "it yet.");
  }
}

}  // namespace

at::Tensor empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
                 c10::optional<at::Layout> layout_opt,
                 c10::optional<at::Device> device_opt,
                 c10::optional<bool> pin_memory_opt,
                 c10::optional<at::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt,
              "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(at::layout_or_default(layout_opt) ==
                                   at::Layout::Strided);

  const auto scalar_type = at::dtype_or_default(dtype_opt);

  const auto device = at::device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_privateuseone(),
                        "Got unexpected device type: ", device);
  const c10::DeviceGuard device_guard(device);
  auto* allocator = getGCUDeviceAllocator();
  constexpr c10::DispatchKeySet gcu_dks(c10::DispatchKey::PrivateUse1);
  // gcu limits, can't use at::detail::empty_generic
  // auto tensor = at::detail::empty_generic(size, allocator, gcu_dks,
  // scalar_type, memory_format_opt);

  at::detail::check_size_nonnegative(size);
  raise_warning_for_complex_half(scalar_type);
  auto gcu_scalar_type = get_gcu_scalar_type(scalar_type);
  if (is_narrow_type(scalar_type)) {
    warn_type_narrow(scalar_type);
  }
  caffe2::TypeMeta dtype = at::scalarTypeToTypeMeta(gcu_scalar_type);
  auto size_bytes =
      at::detail::computeStorageNbytesContiguous(size, dtype.itemsize());
  auto data = allocator->allocate(size_bytes);
  if (gcu_scalar_type != scalar_type) size_bytes <<= 1;
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data),
      allocator, /*resizeable=*/true);

  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage_impl), gcu_dks, at::scalarTypeToTypeMeta(scalar_type));
  // Default TensorImpl has size [0]
  // NB: test for meta dispatch key to avoid guarding on zero-ness
  if (gcu_dks.has(c10::DispatchKey::Meta) || size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);
  }

  if (memory_format_opt.has_value()) {
    // Restriding a just-created empty contiguous tensor does nothing.
    if (*memory_format_opt != at::MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(tensor);
  }

  // clang-format off
  PTDLOG(OP) << "empty" << ": {\n"
             << tensorArgsToString({}, {tensor})
             << "}\n";
  // clang-format on

  return tensor;
}

at::Tensor empty_tmp(at::IntArrayRef size,
                     c10::optional<at::ScalarType> dtype_opt,
                     c10::optional<at::Layout> layout_opt,
                     c10::optional<at::Device> device_opt,
                     c10::optional<bool> pin_memory_opt,
                     c10::optional<at::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt,
              "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(at::layout_or_default(layout_opt) ==
                                   at::Layout::Strided);

  const auto scalar_type = at::dtype_or_default(dtype_opt);

  const auto device = at::device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_privateuseone(),
                        "Got unexpected device type: ", device);
  const c10::DeviceGuard device_guard(device);
  auto* allocator = getGCUDeviceAllocator();
  constexpr c10::DispatchKeySet gcu_dks(c10::DispatchKey::PrivateUse1);
  // gcu limits, can't use at::detail::empty_generic
  // auto tensor = at::detail::empty_generic(size, allocator, gcu_dks,
  // scalar_type, memory_format_opt);

  at::detail::check_size_nonnegative(size);
  raise_warning_for_complex_half(scalar_type);
  auto gcu_scalar_type = get_gcu_scalar_type_tmp(scalar_type);
  if (is_narrow_type_tmp(scalar_type)) {
    warn_type_narrow(scalar_type);
  }
  caffe2::TypeMeta dtype = at::scalarTypeToTypeMeta(gcu_scalar_type);
  auto size_bytes =
      at::detail::computeStorageNbytesContiguous(size, dtype.itemsize());
  auto data = allocator->allocate(size_bytes);
  if (gcu_scalar_type != scalar_type) size_bytes <<= 1;
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data),
      allocator, /*resizeable=*/true);

  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage_impl), gcu_dks, at::scalarTypeToTypeMeta(scalar_type));
  // Default TensorImpl has size [0]
  // NB: test for meta dispatch key to avoid guarding on zero-ness
  if (gcu_dks.has(c10::DispatchKey::Meta) || size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);
  }

  if (memory_format_opt.has_value()) {
    // Restriding a just-created empty contiguous tensor does nothing.
    if (*memory_format_opt != at::MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(tensor);
  }

  // clang-format off
  PTDLOG(OP) << "empty" << ": {\n"
             << tensorArgsToString({}, {tensor})
             << "}\n";
  // clang-format on

  return tensor;
}

at::Tensor empty(at::IntArrayRef size, at::TensorOptions options) {
  return empty(size, c10::optTypeMetaToScalarType(options.dtype_opt()),
               options.layout_opt(), options.device_opt(),
               options.pinned_memory_opt(), options.memory_format_opt());
}

at::Tensor empty_tmp(at::IntArrayRef size, at::TensorOptions options) {
  return empty_tmp(size, c10::optTypeMetaToScalarType(options.dtype_opt()),
                   options.layout_opt(), options.device_opt(),
                   options.pinned_memory_opt(), options.memory_format_opt());
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         c10::optional<at::ScalarType> dtype_opt,
                         c10::optional<at::Layout> layout_opt,
                         c10::optional<at::Device> device_opt,
                         c10::optional<bool> pin_memory_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt,
              "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(at::layout_or_default(layout_opt) ==
                                   at::Layout::Strided);

  const auto scalar_type = at::dtype_or_default(dtype_opt);

  const auto device = at::device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_privateuseone());
  const c10::DeviceGuard device_guard(device);
  auto* allocator = getGCUDeviceAllocator();
  constexpr c10::DispatchKeySet gcu_dks(c10::DispatchKey::PrivateUse1);
  // gcu limits, can't use at::detail::empty_strided_generic
  // auto tensor = at::detail::empty_strided_generic(size, stride, allocator,
  //                                                 gcu_dks, scalar_type);

  at::detail::check_size_nonnegative(size);
  raise_warning_for_complex_half(scalar_type);
  auto gcu_scalar_type = get_gcu_scalar_type(scalar_type);
  if (is_narrow_type(scalar_type)) {
    warn_type_narrow(scalar_type);
  }
  caffe2::TypeMeta dtype = at::scalarTypeToTypeMeta(gcu_scalar_type);
  auto size_bytes =
      at::detail::computeStorageNbytes(size, stride, dtype.itemsize());
  auto data = allocator->allocate(size_bytes);
  if (gcu_scalar_type != scalar_type) size_bytes <<= 1;
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data),
      allocator, /*resizeable=*/true);

  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage_impl), gcu_dks, at::scalarTypeToTypeMeta(scalar_type));
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(tensor);
  }

  // clang-format off
  PTDLOG(OP) << "empty_strided" << ": {\n"
             << tensorArgsToString({}, {tensor})
             << "}\n";
  // clang-format on

  return tensor;
}

at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                         const at::TensorOptions& options) {
  return empty_strided(
      size, stride, c10::optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}

}  // namespace aotops

}  // namespace torch_gcu
