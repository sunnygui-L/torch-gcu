#include <ATen/InferSize.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <topsaten/topsaten_ops.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/gcu_resize.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "gcu/gcu_context.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"
namespace torch_gcu {

namespace aotops {

namespace {

//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
at::Tensor alias_with_sizes_and_strides(const at::Tensor& self,
                                        const Vec& sizes, const Vec& strides) {
  // caller should make sure that sizes and strides are valid for self
  //(storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  at::Tensor self_;
  if (self.is_quantized()) {
    self_ = at::detail::make_tensor<at::QTensorImpl>(
        c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
        self.dtype(), at::get_qtensorimpl(self)->quantizer());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
    self_tmp_->set_sizes_and_strides(sizes, strides);
  } else {
    self_ = at::detail::make_tensor<at::TensorImpl>(
        c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
        self.dtype());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
    self_tmp_->set_sizes_and_strides(sizes, strides);
  }
  at::namedinference::propagate_names(self_, self);
  return self_;
}

}  // namespace

at::Tensor& set_(at::Tensor& result) {
  // clang-format off
  PTDLOG(OP) << "set_" << ": {\n"
             << tensorArgsToString({}, {result})
             << "}\n";
  // clang-format on

  caffe2::TypeMeta dtype = result.dtype();
  c10::Storage storage(c10::Storage::use_byte_size_t(), 0,
                       getGCUDeviceAllocator(), true);
  aotops::set_(result, std::move(storage), 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

at::Tensor& set_(at::Tensor& result, at::Storage source) {
  // clang-format off
  PTDLOG(OP) << "set_" << ": {\n"
             << tensorArgsToString({}, {result})
             << "source: " << storageToString(source)
             << "}\n";
  // clang-format on

  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / result.dtype().itemsize());
  return aotops::set_(result, std::move(source), 0, new_size, {});
}

at::Tensor& set_(at::Tensor& result, at::Storage storage,
                 int64_t storage_offset, at::IntArrayRef size,
                 at::IntArrayRef stride) {
  // clang-format off
  PTDLOG(OP) << "set_" << ": {\n"
             << tensorArgsToString({}, {result})
             << "storage: " << storageToString(storage)
             << "storage_offset: " << storage_offset << "\n"
             << "size: " << size << "\n"
             << "stride: " << stride << "\n"
             << "}\n";
  // clang-format on

  at::native::checkSetStorage(result, storage, storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt =
      stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : c10::nullopt;
  aotops::resize_impl_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

at::Tensor& set_(at::Tensor& result, const at::Tensor& source) {
  // clang-format off
  PTDLOG(OP) << "set_" << ": {\n"
             << tensorArgsToString({source}, {result})
             << "}\n";
  // clang-format on

  if (result.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    return aotops::set_(result, source.storage(), source.storage_offset(),
                        source.sizes(), source.strides());
  }
  return result;
}

at::Tensor as_strided(const at::Tensor& self, at::IntArrayRef size,
                      at::IntArrayRef stride,
                      c10::optional<int64_t> storage_offset_opt) {
  auto storage_offset = storage_offset_opt.value_or(self.storage_offset());

  // clang-format off
  PTDLOG(OP) << "as_strided" << ": {\n"
             << tensorArgsToString({self}, {})
             << "size: " << size << "\n"
             << "stride: " << stride << "\n"
             << "storage_offset: " << storage_offset << "\n"
             << "}\n";
  // clang-format on

  auto result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  at::native::setStrided(result, size, stride, storage_offset);
  return result;
}

at::Tensor _reshape_alias(const at::Tensor& self, at::IntArrayRef sizes,
                          at::IntArrayRef strides) {
  // clang-format off
  PTDLOG(OP) << "_reshape_alias" << ": {\n"
             << tensorArgsToString({self}, {})
             << "sizes: " << sizes << "\n"
             << "strides: " << strides << "\n"
             << "}\n";
  // clang-format on

  return alias_with_sizes_and_strides(self, sizes, strides);
}

at::Tensor unfold(const at::Tensor& self, int64_t d, int64_t size,
                  int64_t step) {
  // clang-format off
  PTDLOG(OP) << "unfold" << ": {\n"
             << tensorArgsToString({self}, {})
             << "d: " << d << "\n"
             << "size: " << size << "\n"
             << "step: " << step << "\n"
             << "}\n";
  // clang-format on

  // some special handling to deal with allow d == 0 when self.dim() == 0
  auto ndim = self.dim();
  d = at::maybe_wrap_dim(d, ndim, /*wrap_scalar=*/true);

  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  int64_t max_size = self.dim() == 0 ? 1 : sizes[d];
  TORCH_CHECK(size <= max_size, "maximum size for tensor at dimension ", d,
              " is ", max_size, " but size is ", size);
  TORCH_CHECK(step > 0, "step is ", step, " but must be > 0");
  sizes.push_back(size);
  strides.push_back(self.dim() == 0 ? 1 : strides[d]);
  // The if handles the self.dim() == 0 case
  if (d < ndim) {
    sizes[d] = (sizes[d] - size) / step + 1;
    strides[d] *= step;
  }
  return aotops::as_strided(self, sizes, strides, c10::nullopt);
}

at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
  // clang-format off
  PTDLOG(OP) << "view" << ": {\n"
             << tensorArgsToString({self}, {})
             << "size: " << size << "\n"
             << "}\n";
  // clang-format on

  at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one "
      "dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  return alias_with_sizes_and_strides(self, inferred_size, *stride);
}

}  // namespace aotops

}  // namespace torch_gcu
