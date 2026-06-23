#include "aten/aot_ops/gcu_resize.h"

#include "aten/aot_ops/gcu_ops.h"
#include "gcu/gcu_functions.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_stream.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

namespace aotops {

void resize_bytes(at::StorageImpl* storage, size_t size_bytes, bool is_narrow) {
  TORCH_CHECK(storage->resizable(),
              "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(allocator != nullptr,
              "Trying to resize storage without an allocator");

  auto device = current_device();
  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(
        at::DataPtr(nullptr, at::Device(at::DeviceType::PrivateUse1, device)));
    storage->set_nbytes(0);
    return;
  }

  size_t size_bytes_ = is_narrow ? size_bytes >> 1 : size_bytes;
  size_t old_storage_bytes = storage->nbytes();
  old_storage_bytes = is_narrow ? old_storage_bytes >> 1 : old_storage_bytes;
  at::DataPtr data = allocator->allocate(size_bytes_);
  if (storage->data_ptr()) {
    // // Enable p2p access when the memcpy is across devices
    // get_p2p_access(device, storage->device().index());

    C10_GCU_CHECK(topsMemcpyAsync(
        data.get(), storage->data(), std::min(old_storage_bytes, size_bytes_),
        topsMemcpyDeviceToDevice, getCurrentGCUStream()));
  }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size,
                          c10::optional<at::MemoryFormat> memory_format_opt) {
  // clang-format off
  PTDLOG(OP) << "resize_" << ": {\n"
             << tensorArgsToString({self}, {})
             << "size: " << size << "\n"
             << "memory_format_opt: " << memory_format_opt.has_value() << "\n"
             << "}\n";
  // clang-format on

  if (self.has_names()) {
    return at::native::resize_named_tensor_(self, size, memory_format_opt);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  int64_t old_storage_nbytes =
      self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  resize_impl_(self_, size, /*strides=*/c10::nullopt);
  if (memory_format_opt.has_value()) {
    auto memory_format = memory_format_opt.value();
    TORCH_CHECK(memory_format != at::MemoryFormat::Preserve,
                "Unsupported memory format", memory_format);
    self_->empty_tensor_restride(memory_format);
  }

  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms())) {
    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
  }

  return self;
}

const at::Tensor& _resize_output_(const at::Tensor& self, at::IntArrayRef size,
                                  at::Device device) {
  TORCH_CHECK(self.device() == device,
              "out Tensor doesn't have the correct device set");
  if (at::native::resize_output_check(self, size)) {
    aotops::resize_(self, size, c10::nullopt);
  }
  return self;
}

bool resize_output(const at::Tensor& output, at::IntArrayRef shape) {
  if (at::native::resize_output_check(output, shape)) {
    // avoid a redispatch for gcu, must layout contiguous.
    aotops::resize_(output, shape, c10::nullopt);
    return true;
  } else {
    return false;
  }
}

// from build/aten/src/ATen/RegisterMeta.cpp
void resize_out(const at::Tensor& out, c10::IntArrayRef sizes,
                c10::IntArrayRef strides, const c10::TensorOptions& options) {
  // clang-format off
  PTDLOG(OP) << "resize_out" << ": {\n"
             << tensorArgsToString({out}, {})
             << "sizes: " << sizes << "\n"
             << "strides: " << strides << "\n"
             << "options: " << options << "\n"
             << "}\n";
  // clang-format on

  TORCH_CHECK(options.dtype() == out.dtype(),
              "Expected out tensor to have dtype ", options.dtype(),
              ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(),
              "Expected out tensor to have device ", options.device(),
              ", but got ", out.device(), " instead");
  const bool resized = resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(
          *options.memory_format_opt());
    }
  }
}

}  // namespace aotops

}  // namespace torch_gcu
