#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/gcu_resize.h"
#include "aten/aot_ops/topsaten_bridge.h"

namespace torch_gcu {
namespace aotops {

std::tuple<at::Tensor, at::Tensor> _unique(const at::Tensor &self, bool sorted,
                                           bool return_inverse) {
  auto out = aotops::_unique2(self, sorted, return_inverse, false);
  return std::make_tuple(std::get<0>(out), std::get<1>(out));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _unique2(const at::Tensor &self,
                                                        bool sorted,
                                                        bool return_inverse,
                                                        bool return_counts) {
  int64_t numel = self.numel();
  auto output = at::empty({0}, self.options());
  auto inverse_indices = at::empty({0}, self.options().dtype(at::kLong));
  auto counts = at::empty({0}, self.options().dtype(at::kLong));

  if (numel == 0) {
    if (return_inverse) {
      aotops::resize_output(inverse_indices, self.sizes());
    }
    return std::make_tuple(output, inverse_indices, counts);
  }

  // resize to max (input size)
  aotops::resize_output(output, {
                                    numel,
                                });
  if (return_inverse) {
    aotops::resize_output(inverse_indices, self.sizes());
  }
  if (return_counts) {
    aotops::resize_output(counts, {
                                      numel,
                                  });
  }

  auto x_self = topsaten_variable(self).value;
  auto x_output = topsaten_variable(output).value;
  auto x_inverse_indices = topsaten_variable(inverse_indices).value;
  auto x_counts = topsaten_variable(counts).value;

  auto stream = getCurrentGCUStream();

  auto op_info = [&]() -> std::string {
    std::stringstream ss;
    // clang-format off
    ss << "topsatenUnique2" << " :\n" \
       << tensorArgsToString({self}, {output, inverse_indices, counts}) \
       << "sorted: " << sorted  << "\n" \
       << "return_inverse: " << return_inverse << "\n" \
       << "return_counts: " << return_counts << "\n" \
       << "stream: " << stream << "\n";
    // clang-format on
    return ss.str();
  };
  PTDLOG(OP) << op_info();

  CHECK_TOPSATEN_CALL(
      topsaten::topsatenUnique2(x_output, x_inverse_indices, x_counts, x_self,
                                sorted, return_inverse, return_counts, stream),
      op_info);

  // always synchronize to get output shape
  torch_gcu::stream_synchronize(stream);

  std::vector<int64_t> output_shape(
      x_output.GetTensorShape().data,
      x_output.GetTensorShape().data + x_output.GetTensorShape().len);

  // Reset tensor metadata to zero elements before resizing to actual output
  // shape.  This avoids the PyTorch deprecation warning:
  //   "An output with one or more elements was resized since it had shape [N],
  //    which does not match the required output shape [M]."
  // The warning is triggered by at::native::resize_output_check() when a
  // non-empty tensor is resized.  Resizing to 0 first is the recommended
  // workaround (see ATen/native/Resize.cpp).
  //
  // NOTE: We use resize_impl_ directly instead of Tensor::resize_() to avoid
  // going through the ATen dispatcher.  This is an internal implementation
  // detail of _unique2 and should not pollute op statistics tracking.
  int64_t zero_dim = 0;
  at::IntArrayRef zero_shape(&zero_dim, 1);
  aotops::resize_impl_(output.unsafeGetTensorImpl(), zero_shape,
                       /*stride=*/c10::nullopt, /*device_guard=*/false);
  aotops::resize_output(output, output_shape);
  if (return_counts) {
    aotops::resize_impl_(counts.unsafeGetTensorImpl(), zero_shape,
                         /*stride=*/c10::nullopt, /*device_guard=*/false);
    aotops::resize_output(counts, output_shape);
  }

  return std::make_tuple(output, inverse_indices, counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_consecutive(
    const at::Tensor &self, bool return_inverse, bool return_counts,
    std::optional<int64_t> dim) {
  int64_t numel = self.numel();
  auto output = at::empty({0}, self.options());
  auto inverse_indices = at::empty({0}, self.options().dtype(at::kLong));
  auto counts = at::empty({0}, self.options().dtype(at::kLong));

  if (numel == 0 && !dim.has_value()) {
    if (return_inverse) {
      aotops::resize_output(inverse_indices, self.sizes());
    }
    return std::make_tuple(output, inverse_indices, counts);
  }

  if (!dim.has_value()) {
    aotops::resize_output(output, {numel});
    if (return_inverse) {
      aotops::resize_output(inverse_indices, self.sizes());
    }
    if (return_counts) {
      aotops::resize_output(counts, {numel});
    }
  } else {
    aotops::resize_output(output, self.sizes());
    int64_t dim_size = self.size(dim.value());
    if (return_inverse) {
      aotops::resize_output(inverse_indices, {dim_size});
    }
    if (return_counts) {
      aotops::resize_output(counts, {dim_size});
    }
  }

  auto x_self = topsaten_variable(self).value;
  auto x_output = topsaten_variable(output).value;
  auto x_inverse_indices = topsaten_variable(inverse_indices).value;
  auto x_counts = topsaten_variable(counts).value;
  auto x_dim = topsaten_variable(dim).value;

  auto stream = getCurrentGCUStream();

  auto op_info = [&]() -> std::string {
    std::stringstream ss;
    // clang-format off
    ss << "topsatenUniqueConsecutive" << " :\n" \
       << tensorArgsToString({self}, {output, inverse_indices, counts}) \
       << "return_inverse: " << return_inverse << "\n" \
       << "return_counts: " << return_counts << "\n" \
       << "dim: " << (dim.has_value() ? std::to_string(dim.value()) : "nullopt") << "\n" \
       << "stream: " << stream << "\n";
    // clang-format on
    return ss.str();
  };
  PTDLOG(OP) << op_info();

  CHECK_TOPSATEN_CALL(topsaten::topsatenUniqueConsecutive(
                          x_output, x_inverse_indices, x_counts, x_self,
                          return_inverse, return_counts, x_dim, stream),
                      op_info);

  torch_gcu::stream_synchronize(stream);

  std::vector<int64_t> output_shape(
      x_output.GetTensorShape().data,
      x_output.GetTensorShape().data + x_output.GetTensorShape().len);

  int64_t zero_dim = 0;
  at::IntArrayRef zero_shape(&zero_dim, 1);
  aotops::resize_impl_(output.unsafeGetTensorImpl(), zero_shape,
                       /*stride=*/c10::nullopt, /*device_guard=*/false);
  aotops::resize_output(output, output_shape);

  if (return_counts) {
    std::vector<int64_t> counts_shape(
        x_counts.GetTensorShape().data,
        x_counts.GetTensorShape().data + x_counts.GetTensorShape().len);
    aotops::resize_impl_(counts.unsafeGetTensorImpl(), zero_shape,
                         /*stride=*/c10::nullopt, /*device_guard=*/false);
    aotops::resize_output(counts, counts_shape);
  }

  return std::make_tuple(output, inverse_indices, counts);
}

}  // namespace aotops

}  // namespace torch_gcu
