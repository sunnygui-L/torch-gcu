#include "aten/aot_ops/gcu_ops.h"

#include <ATen/native/Padding.h>

#include "aten/aot_ops/gcu_resize.h"
#include "aten/aot_ops/topsaten_bridge.h"
namespace torch_gcu {

namespace aotops {

at::Tensor reflection_pad2d(const at::Tensor &self, at::IntArrayRef padding) {
  at::Tensor output = at::empty({0}, self.options());
  reflection_pad2d_out(self, padding, output);
  return output;
}

at::Tensor &reflection_pad2d_out(const at::Tensor &self,
                                 at::IntArrayRef padding, at::Tensor &out) {
  int dim_w = 2;
  int dim_h = 1;
  int dim_slices = 0;
  int64_t nbatch = 1;

  at::native::padding::check_valid_input<2>(self, padding);

  int ndim = self.dim();
  if (ndim == 4) {
    nbatch = self.size(0);
    dim_w++;
    dim_h++;
    dim_slices++;
  }

  /* sizes */
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t nplane = self.size(dim_slices);
  int64_t input_h = self.size(dim_h);
  int64_t input_w = self.size(dim_w);
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(pad_l < input_w && pad_r < input_w,
              "Argument #4: Padding size should be less than the corresponding "
              "input dimension, but got: padding (",
              pad_l, ", ", pad_r, ") at dimension ", dim_w, " of input ",
              self.sizes());

  TORCH_CHECK(pad_t < input_h && pad_b < input_h,
              "Argument #6: Padding size should be less than the corresponding "
              "input dimension, but got: padding (",
              pad_t, ", ", pad_b, ") at dimension ", dim_h, " of input ",
              self.sizes());

  TORCH_CHECK(output_w >= 1 || output_h >= 1, "input (H: ", input_h,
              ", W: ", input_w,
              ") is too small. Calculated "
              "output H: ",
              output_h, " W: ", output_w);

  /* resize output */
  if (ndim == 3) {
    aotops::resize_output(out, {nplane, output_h, output_w});
  } else {
    if (self.is_quantized()) {
      // quantized tensor can not be resized with argument `memory_format`
      aotops::resize_output(out, {nbatch, nplane, output_h, output_w});
    } else {
      aotops::resize_output(out, {nbatch, nplane, output_h, output_w});
    }
  }

  bridge_topsatenReflectionPad2d_out1(out, self, padding);
  return out;
}

}  // namespace aotops

}  // namespace torch_gcu