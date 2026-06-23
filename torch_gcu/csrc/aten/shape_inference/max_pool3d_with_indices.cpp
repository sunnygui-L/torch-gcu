/*
 * Copyright 2022-2026 Enflame. All Rights Reserved.
 */

#include <ATen/core/Tensor.h>
#include <ATen/div_rtn.h>
#include <ATen/native/ForeachUtils.h>

#include <tuple>

#include "aten/aot_ops/gcu_resize.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {
namespace aotops {

template <typename T>
inline T pooling_output_shape_pad_lr(T inputSize, T kernelSize, T pad_l,
                                     T pad_r, T stride, T dilation,
                                     bool ceil_mode) {
  T outputSize =
      div_rtn<T>(inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
                     (ceil_mode ? stride - 1 : 0),
                 stride) +
      1;
  if (ceil_mode) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputSize - 1) * stride >= inputSize + pad_l) {
      --outputSize;
    }
  }
  return outputSize;
}

template <typename T>
inline T pooling_output_shape(T inputSize, T kernelSize, T pad, T stride,
                              T dilation, bool ceil_mode) {
  TORCH_CHECK(stride != 0, "stride should not be zero");
  TORCH_CHECK(pad >= 0, "pad must be non-negative, but got pad: ", pad);
  TORCH_CHECK(
      pad <= ((kernelSize - 1) * dilation + 1) / 2,
      "pad should be at most half of effective kernel size, but got pad=", pad,
      ", kernel_size=", kernelSize, " and dilation=", dilation)
  return pooling_output_shape_pad_lr(inputSize, kernelSize, pad, pad, stride,
                                     dilation, ceil_mode);
}

inline void pool3d_shape_check(const at::Tensor& input, int64_t nslices,
                               int64_t kT, int64_t kH, int64_t kW, int64_t dT,
                               int64_t dH, int64_t dW, int64_t pT, int64_t pH,
                               int64_t pW, int64_t dilationT, int64_t dilationH,
                               int64_t dilationW, int64_t itime,
                               int64_t iheight, int64_t iwidth, int64_t otime,
                               int64_t oheight, int64_t owidth,
                               const char* fn_name,
                               bool check_input_size = false) {
  const int64_t ndim = input.ndimension();

  TORCH_CHECK(kT > 0 && kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got ", "kT: ", kT,
              " kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dT > 0 && dW > 0 && dH > 0,
              "stride should be greater than zero, but got ", "dT: ", dT,
              " dH: ", dH, " dW: ", dW);
  TORCH_CHECK(dilationT > 0 && dilationW > 0 && dilationH > 0,
              "dilation should be greater than zero, but got ",
              "dilationT: ", dilationT, " dilationH: ", dilationH,
              " dilationW: ", dilationW);

  TORCH_CHECK(ndim == 4 || ndim == 5, fn_name,
              ": Expected 4D or 5D tensor for input, but got: ", input.sizes());

  for (const auto i : c10::irange(ndim)) {
    if (ndim == 5 && i == 0) {
      // size of batch-dim can be 0.
      continue;
    }
    TORCH_CHECK(
        input.size(i) > 0, fn_name,
        ": Expected input's non-batch dimensions to have positive length,"
        " but input has a shape of ",
        input.sizes(), " and non-batch dimension ", input.size(i),
        " has length zero!")
  }

  if (check_input_size) {  // AveragePool3d
    TORCH_CHECK(itime >= kT && iheight >= kH && iwidth >= kW, "input image ",
                "(T: ", itime, " H: ", iheight, " W: ", iwidth,
                ") smaller than ", "kernel size ", "(kT: ", kT, " kH: ", kH,
                " kW: ", kW, ")");
  }

  TORCH_CHECK(
      kT / 2 >= pT && kW / 2 >= pW && kH / 2 >= pH,
      "pad should be smaller than or equal to half of kernel size, but got "
      "kT: ",
      kT, " kW: ", kW, " kH: ", kH, " padT: ", pT, " padW: ", pW,
      " padH: ", pH);

  TORCH_CHECK(otime >= 1 && owidth >= 1 && oheight >= 1, "Given input size: (",
              nslices, "x", itime, "x", iheight, "x", iwidth, "). ",
              "Calculated output size: (", nslices, "x", otime, "x", oheight,
              "x", owidth, "). ", "Output size is too small");
}

std::tuple<at::Tensor&, at::Tensor&> max_pool3d_with_indices_out_shape_infer(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& out, at::Tensor& indices) {
  at::TensorArg output_arg{out, "out", 1};
  at::TensorArg indices_arg{indices, "indices", 2};
  at::TensorArg input_arg{self, "self", 3};

  checkAllSameGCU(__func__, {output_arg, indices_arg, input_arg});

  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
              "max_pool3d: kernel_size must either be a single int, or a tuple "
              "of three ints")
  const int64_t kT = kernel_size[0];
  const int64_t kH = kernel_size.size() == 1 ? kT : kernel_size[1];
  const int64_t kW = kernel_size.size() == 1 ? kT : kernel_size[2];

  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
              "max_pool3d: stride must either be omitted, a single int, or a "
              "tuple of three ints")
  const int64_t dT = stride.empty() ? kT : stride[0];
  const int64_t dH = stride.empty() ? kH : stride.size() == 1 ? dT : stride[1];
  const int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dT : stride[2];

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
              "max_pool3d: padding must either be a single int, or a tuple of "
              "three ints");
  const int64_t pT = padding[0];
  const int64_t pH = padding.size() == 1 ? pT : padding[1];
  const int64_t pW = padding.size() == 1 ? pT : padding[2];

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
              "max_pool3d: dilation must be either a single int, or a tuple of "
              "three ints");
  const int64_t dilationT = dilation[0];
  const int64_t dilationH = dilation.size() == 1 ? dilationT : dilation[1];
  const int64_t dilationW = dilation.size() == 1 ? dilationT : dilation[2];

  const int64_t nbatch = self.ndimension() == 5 ? self.size(-5) : 1;
  const int64_t nslices = self.size(-4);
  const int64_t itime = self.size(-3);
  const int64_t iheight = self.size(-2);
  const int64_t iwidth = self.size(-1);

  const int64_t otime =
      pooling_output_shape<int64_t>(itime, kT, pT, dT, dilationT, ceil_mode);
  const int64_t oheight =
      pooling_output_shape<int64_t>(iheight, kH, pH, dH, dilationH, ceil_mode);
  const int64_t owidth =
      pooling_output_shape<int64_t>(iwidth, kW, pW, dW, dilationW, ceil_mode);

  pool3d_shape_check(self, nslices, kT, kH, kW, dT, dH, dW, pT, pH, pW,
                     dilationT, dilationH, dilationW, itime, iheight, iwidth,
                     otime, oheight, owidth,
                     "max_pool3d_with_indices_out_gcu_template()");

  bool channels_last =
      self.ndimension() == 5 &&
      self.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
  if (self.ndimension() == 4) {
    at::Tensor input_channels_last_check = self.unsqueeze(0);
    // work around buggy behavior of suggest_memory_format here where
    // suggested format of unsqueezed tensor is contiguous while it is
    // really only contiguous in ChannelsLast3d
    channels_last = (!input_channels_last_check.is_contiguous()) &&
                    input_channels_last_check.is_contiguous(
                        at::MemoryFormat::ChannelsLast3d);
    if (!channels_last) {
      out.resize_({nslices, otime, oheight, owidth});
      indices.resize_({nslices, otime, oheight, owidth});
    } else {
      out.resize_({1, nslices, otime, oheight, owidth},
                  at::MemoryFormat::ChannelsLast3d);
      indices.resize_({1, nslices, otime, oheight, owidth},
                      at::MemoryFormat::ChannelsLast3d);
      out = out.squeeze(0);
      indices = indices.squeeze(0);
    }
  } else {
    if (!channels_last) {
      out.resize_({nbatch, nslices, otime, oheight, owidth});
      indices.resize_({nbatch, nslices, otime, oheight, owidth});
    } else {
      out.resize_({nbatch, nslices, otime, oheight, owidth},
                  at::MemoryFormat::ChannelsLast3d);
      indices.resize_({nbatch, nslices, otime, oheight, owidth},
                      at::MemoryFormat::ChannelsLast3d);
    }
  }

  return std::tuple<at::Tensor&, at::Tensor&>(out, indices);
}

std::tuple<at::Tensor, at::Tensor> max_pool3d_with_indices_shape_infer(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  at::Tensor output = at::empty({0}, self.options());
  at::Tensor indices =
      at::empty({0}, self.options().dtype(at::ScalarType::Int));
  max_pool3d_with_indices_out_shape_infer(self, kernel_size, stride, padding,
                                          dilation, ceil_mode, output, indices);

  return std::make_tuple(std::move(output), std::move(indices));
}

}  // namespace aotops
}  // namespace torch_gcu