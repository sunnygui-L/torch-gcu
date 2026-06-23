#include <cstdint>
#include <tuple>
#include <vector>

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/gcu_resize.h"
#include "aten/aot_ops/topsaten_bridge.h"

namespace torch_gcu {
namespace aotops {

at::Tensor nms(const at::Tensor& dets, const at::Tensor& scores,
               double iou_threshold) {
  TORCH_CHECK(dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(),
              "D");
  TORCH_CHECK(dets.size(1) == 4,
              "boxes should have 4 elements in dimension 1, got ",
              dets.size(1));
  TORCH_CHECK(scores.dim() == 1, "scores should be a 1d tensor, got ",
              scores.dim(), "D");
  TORCH_CHECK(dets.size(0) == scores.size(0),
              "boxes and scores should have same number of elements in ",
              "dimension 0, got ", dets.size(0), " and ", scores.size(0));

  if (dets.numel() == 0) {
    return aotops::empty({0}, dets.options().dtype(at::kLong));
  }
  int64_t dets_num = dets.size(0);
  at::Tensor result =
      aotops::empty({dets_num}, dets.options().dtype(at::kLong));

  auto stream = getCurrentGCUStream();
  auto op_info = [&]() -> std::string {
    return get_op_info("topsatenNms", result, dets, scores, iou_threshold,
                       stream);
  };
  PTDLOG(OP) << op_info();

  auto x_output = topsaten_variable(result).value;
  auto x_dets = topsaten_variable(dets).value;
  auto x_scores = topsaten_variable(scores).value;
  auto x_iou_threshold = topsaten_variable(iou_threshold).value;
  topsatenStatus_t status = topsaten::topsatenNms(x_output, x_dets, x_scores,
                                                  x_iou_threshold, stream);
  CHECK_TOPSATEN_CALL(status, op_info);

  // always synchronize to get output shape
  torch_gcu::stream_synchronize(stream);

  std::vector<int64_t> output_shape(
      x_output.GetTensorShape().data,
      x_output.GetTensorShape().data + x_output.GetTensorShape().len);

  aotops::resize_output(result, output_shape);

  return result;
}

}  // namespace aotops

}  // namespace torch_gcu
