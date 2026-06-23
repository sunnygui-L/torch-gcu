#include "python/gcu_fusion_op.h"

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "aotfusion/fusion_ops.h"

namespace torch_gcu {

void RegisterFusionOp(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("relu_convolution_fusion_shape_infer",
        [](const at::Tensor& input, const at::Tensor& weight,
           const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
           at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
           at::IntArrayRef output_padding, int64_t groups) {
          return aotfusion::relu_convolution_fusion_shape_infer(
              input, weight, bias, stride, padding, dilation, transposed,
              output_padding, groups);
        });
  m.def("relu_convolution_fusion_out",
        [](const at::Tensor& input, const at::Tensor& weight,
           const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
           at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
           at::IntArrayRef output_padding, int64_t groups, at::Tensor& out) {
          return aotfusion::relu_convolution_fusion_out(
              input, weight, bias, stride, padding, dilation, transposed,
              output_padding, groups, out);
        });
}

}  // namespace torch_gcu