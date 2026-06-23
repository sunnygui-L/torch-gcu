/*
 * Copyright 2024 Enflame. All Rights Reserved.
 */

#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>

#include "aten/gcu_conv_determine_backend_memory_format.h"
#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

namespace {

template <typename T>
struct ConvParams {
  std::vector<int64_t> stride;
  std::vector<T> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<T> output_padding;
  int groups;

  bool is_padding_neg() const {
    bool is_non_neg = false;
    for (const auto& p : padding) {
      is_non_neg |= (p < 0);
    }
    return is_non_neg;
  }

  bool is_output_padding_neg() const {
    bool is_non_neg = false;
    for (const auto& p : output_padding) {
      is_non_neg |= (p < 0);
    }
    return is_non_neg;
  }

  bool is_stride_nonpos() const {
    bool is_nonpos = false;
    for (auto s : stride) {
      is_nonpos |= (s <= 0);
    }
    return is_nonpos;
  }
};

template <typename T>
void check_shape_forward(const at::Tensor& input,
                         const c10::ArrayRef<T>& weight_sizes,
                         const at::Tensor& bias, const ConvParams<T>& params) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight_sizes.size();
  int64_t groups = params.groups;
  const auto& padding = params.padding;
  const auto& dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(!params.is_output_padding_neg(),
              "negative output_padding is not supported");
  TORCH_CHECK(!params.is_stride_nonpos(),
              "non-positive stride is not supported");

  TORCH_CHECK(weight_dim == k, "Expected ", weight_dim,
              "-dimensional input for ", weight_dim, "-dimensional weight ",
              weight_sizes, ", but got ", k, "-dimensional input of size ",
              at::symint::sizes<T>(input), " instead");
  TORCH_CHECK(weight_sizes[0] >= groups, "Given groups=", groups,
              ", expected weight to be at least ", groups,
              " at dimension 0, but got weight of size ", weight_sizes,
              " instead");
  TORCH_CHECK(weight_sizes[0] % groups == 0, "Given groups=", groups,
              ", expected weight to be divisible by ", groups,
              " at dimension 0, but got weight of size [", weight_sizes,
              "] instead");

  if (!transposed) {
    std::vector<T> input_shape;
    std::vector<T> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(at::symint::size<T>(input, 1) == (weight_sizes[1] * groups),
                "Given groups=", groups, ", weight of size ", weight_sizes,
                ", expected input", at::symint::sizes<T>(input), " to have ",
                (weight_sizes[1] * groups), " channels, but got ",
                at::symint::size<T>(input, 1), " channels instead");

    TORCH_CHECK(
        !bias.defined() || (bias.ndimension() == 1 &&
                            at::symint::size<T>(bias, 0) == weight_sizes[0]),
        "Given weight of size ", weight_sizes,
        ", expected bias to be 1-dimensional with ", weight_sizes[0],
        " elements", ", but got bias of size ", at::symint::sizes<T>(bias),
        " instead");

    for (const auto i : c10::irange(2, k)) {
      input_shape.push_back(at::symint::size<T>(input, i) + 2 * padding[i - 2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(),
                "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::string separator = "";

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      AT_ERROR("Calculated padded input size per channel: (", input_ss.str(),
               "). "
               "Kernel size: (",
               kernel_ss.str(),
               "). Kernel size can't be greater than actual input size");
    }
  } else {  // transposed
    TORCH_CHECK(at::symint::size<T>(input, 1) == weight_sizes[0],
                "Given transposed=", transposed, ", weight of size ",
                weight_sizes, ", expected input", at::symint::sizes<T>(input),
                " to have ", weight_sizes[0], " channels, but got ",
                at::symint::size<T>(input, 1), " channels instead");
    TORCH_CHECK(!bias.defined() ||
                    (bias.ndimension() == 1 &&
                     at::symint::size<T>(bias, 0) == weight_sizes[1] * groups),
                "Given transposed=", transposed, ", weight of size ",
                weight_sizes, ", expected bias to be 1-dimensional with ",
                weight_sizes[1] * groups, " elements",
                ", but got bias of size ", at::symint::sizes<T>(bias),
                " instead");
  }
}

template <typename T>
void check_shape_backward(const at::Tensor& input,
                          const c10::ArrayRef<T>& weight_sizes,
                          const ConvParams<T>& params) {
  check_shape_forward<T>(input, weight_sizes, /*bias=*/at::Tensor(), params);
}

}  // namespace

at::Tensor convolution_shape_infer(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  int64_t dim = weight.ndimension() - 2;
  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
  TORCH_CHECK(groups > 0, "non-positive groups is not supported");

  ConvParams<int64_t> params;
  params.stride = at::native::expand_param_if_needed(stride, "stride", dim);
  params.padding = at::native::expand_param_if_needed(padding, "padding", dim);
  params.dilation =
      at::native::expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding =
      at::native::expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;

  c10::IntArrayRef weight_sizes = weight.sizes();
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor& bias_ = *bias_maybe_owned;
  check_shape_forward(input, weight_sizes, bias_, params);

  std::vector<int64_t> out_size;
  c10::IntArrayRef input_sizes = input.sizes();
  if (!transposed) {
    out_size =
        at::native::conv_output_size(input_sizes, weight_sizes, params.padding,
                                     params.stride, params.dilation);
  } else {
    out_size = at::native::conv_input_size(
        input_sizes, weight_sizes, params.padding, params.output_padding,
        params.stride, params.dilation, groups);
  }

  auto memory_format = _determine_backend_memory_format(input, weight);
  return empty(out_size, input.options().memory_format(memory_format));
}

at::Tensor _convolution_shape_infer(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, bool benchmark,
    bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  return convolution_shape_infer(input, weight, bias, stride, padding, dilation,
                                 transposed, output_padding, groups);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor>
convolution_backward_shape_infer(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::OptionalIntArrayRef bias_sizes,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    ::std::array<bool, 3> output_mask) {
  int64_t dim = weight.ndimension() - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  ConvParams<int64_t> params;
  params.stride = at::native::expand_param_if_needed(stride, "stride", dim);
  params.padding = at::native::expand_param_if_needed(padding, "padding", dim);
  params.dilation =
      at::native::expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding =
      at::native::expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;

  // Validate inputs.
  check_shape_backward(input, weight.sizes(), params);
  TORCH_CHECK(input.dim() == grad_output.dim(),
              "Expected input and grad_output to have the same number of "
              "dimensions, but got: ",
              input.dim(), " and ", grad_output.dim());

  // output_padding is only supported for transposed convolutions
  if (!params.transposed) {
    for (auto pad : params.output_padding) {
      TORCH_CHECK(pad == 0,
                  "output_padding is not supported for non-transposed "
                  "convolutions; got: ",
                  params.output_padding);
    }
  }

  auto memory_format = _determine_backend_memory_format(input, weight);
  at::Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
  if (output_mask[0]) {
    backend_grad_input = aotops::empty(
        input.sizes(), input.options().memory_format(memory_format));
  }
  if (output_mask[1]) {
    backend_grad_weight = aotops::empty(
        weight.sizes(), weight.options().memory_format(memory_format));
  }
  if (output_mask[2]) {
    backend_grad_bias = aotops::empty(
        *bias_sizes,
        weight.options().memory_format(at::MemoryFormat::Contiguous));
  }

  return std::make_tuple(backend_grad_input, backend_grad_weight,
                         backend_grad_bias);
}

}  // namespace aotops

}  // namespace torch_gcu
