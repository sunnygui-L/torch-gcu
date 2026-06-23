#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/shape_inference/shape_infer_func.h"
#include "aten/shape_inference/tensor_utils.h"

namespace torch_gcu {

namespace aotops {

namespace {
void inline check_rnn_cell_forward_input(const at::Tensor &input,
                                         c10::SymInt input_size) {
  TORCH_CHECK(input.sym_size(1) == input_size,
              "input has inconsistent input_size: got ", input.sym_size(1),
              " expected ", input_size);
}

void inline check_rnn_cell_forward_hidden(const at::Tensor &input,
                                          const at::Tensor &hx,
                                          c10::SymInt hidden_size,
                                          c10::SymInt hidden_label) {
  TORCH_CHECK(input.sym_size(0) == hx.sym_size(0), "Input batch size ",
              input.sym_size(0), " doesn't match hidden", hidden_label,
              " batch size ", hx.sym_size(0));

  TORCH_CHECK(hx.sym_size(1) == hidden_size, "hidden", hidden_label,
              " has inconsistent hidden_size: got ", hx.sym_size(1),
              ", expected ", hidden_size);
}

// Factor will be 3 for GRU and 4 for LSTM
void checkSizes(at::CheckedFrom c, const at::TensorArg &input_gates,
                const at::TensorArg &hidden_gates,
                const at::TensorArg &input_bias,
                const at::TensorArg &hidden_bias, int64_t factor,
                const at::TensorArg &prev_hidden) {
  at::checkDim(c, input_gates, 2);
  at::checkSameSize(c, input_gates, hidden_gates);
  int64_t gates_size = input_gates->size(1);

  if (input_bias->defined()) {
    at::checkDim(c, input_bias, 1);
    at::checkNumel(c, input_bias, gates_size);
    at::checkSameSize(c, input_bias, hidden_bias);
  }

  at::checkDim(c, prev_hidden, 2);
  at::checkNumel(c, prev_hidden, input_gates->size(0) * gates_size / factor);

  checkAllSameGCU(
      c, {input_gates, hidden_gates, input_bias, hidden_bias, prev_hidden});
}

}  // namespace

::std::tuple<at::Tensor, at::Tensor> lstm_cell_shape_infer(
    const at::Tensor &input, at::TensorList hx, const at::Tensor &w_ih,
    const at::Tensor &w_hh, const c10::optional<at::Tensor> &b_ih,
    const c10::optional<at::Tensor> &b_hh) {
  TORCH_CHECK(hx.size() == 2, "lstm_cell expects two hidden states");
  check_rnn_cell_forward_input(input, w_ih.sym_size(1));
  auto hidden_size = w_hh.sym_size(1);
  check_rnn_cell_forward_hidden(input, hx[0], hidden_size, 0);
  check_rnn_cell_forward_hidden(input, hx[1], std::move(hidden_size), 0);

  auto cx = hx[1];
  auto hy = aotops::empty(cx.sizes(), cx.options());
  auto cy = aotops::empty(cx.sizes(), cx.options());
  return std::make_tuple(std::move(hy), std::move(cy));
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor>
_thnn_fused_lstm_cell_shape_infer(
    const at::Tensor &input_gates, const at::Tensor &hidden_gates,
    const at::Tensor &cx, const c10::optional<at::Tensor> &input_bias,
    const c10::optional<at::Tensor> &hidden_bias) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> input_bias_maybe_owned =
      at::borrow_from_optional_tensor(input_bias);
  const at::Tensor &input_bias_ = *input_bias_maybe_owned;
  const at::Tensor &hidden_bias_ =
      c10::value_or_else(hidden_bias, [] { return at::Tensor(); });

  checkSizes("_thnn_fused_lstm_cell_shape_infer",
             {input_gates, "input_gates", 1}, {hidden_gates, "hidden_gates", 2},
             {input_bias_, "input_bias", 3}, {hidden_bias_, "hidden_bias", 4},
             /*factor=*/4, {cx, "prev_hidden", 5});

  auto workspace = at::empty_like(input_gates, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto hy = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto cy = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  return std::make_tuple(std::move(hy), std::move(cy), std::move(workspace));
}

}  // namespace aotops

}  // namespace torch_gcu
