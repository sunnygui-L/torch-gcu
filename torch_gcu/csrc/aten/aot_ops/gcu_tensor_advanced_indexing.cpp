#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/gcu_structured_shape_infer.h"
#include "aten/shape_inference/tensor_advanced_indexing.h"
#include "topsaten/topsaten_ops.h"

namespace torch_gcu {

namespace aotops {

c10::List<c10::optional<at::Tensor>> make_device_indices(
    const c10::List<c10::optional<at::Tensor>>& org_indices,
    const at::TensorIterator& op) {
  std::vector<c10::optional<at::Tensor>> device_indices;
  auto indices_vec = org_indices.vec();
  int64_t index_num = 1;
  for (size_t i = 0; i < indices_vec.size(); ++i) {
    if (indices_vec[i].has_value()) {
      if (indices_vec[i].value().defined()) {
        at::Tensor indice = indices_vec[i].value();
        if (indices_vec[i].value().dtype() == at::kInt) {
          indice = indice.to(at::kLong);
        }
        if (indice.device() != op.input(index_num).device()) {
          indice = indice.to(op.input(index_num).device());
        }
        device_indices.push_back(indice);
        index_num++;
      } else {
        at::Tensor undef;
        device_indices.push_back(undef);
      }
    } else {
      device_indices.push_back(c10::nullopt);
    }
  }
  return c10::List<c10::optional<at::Tensor>>(device_indices);
}

at::Tensor index(const at::Tensor& self,
                 const c10::List<c10::optional<at::Tensor>>& indices) {
  structured_index_Tensor_gcu_functional op;
  op.meta(self, at::IOptTensorListRef(indices));
  auto device_indices = make_device_indices(indices, op);
  bridge_topsatenIndex_out1(op.maybe_get_output(0), self, device_indices);
  return op.outputs_[0];
}

at::Tensor& index_out(const at::Tensor& self,
                      const c10::List<c10::optional<at::Tensor>>& indices,
                      at::Tensor& out) {
  structured_index_Tensor_gcu_out op(out);
  op.meta(self, at::IOptTensorListRef(indices));
  if (out.numel() == 0) return out;
  auto device_indices = make_device_indices(indices, op);
  bridge_topsatenIndex_out1(op.maybe_get_output(0), self, device_indices);
  return out;
}

at::Tensor& _index_put_impl_(
    at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value, bool accumulate, bool unsafe) {
  c10::List<c10::optional<at::Tensor>> dev_indices;
  at::Device dev = self.device();
  for (const c10::optional<at::Tensor>& indice : indices) {
    if (indice.has_value() && indice.value().defined()) {
      dev_indices.push_back(c10::optional<at::Tensor>(indice.value().to(dev)));
    } else {
      dev_indices.push_back(indice);
    }
  }

  // Sync global deterministic mode to operator library
  auto& ctx = at::globalContext();
  bool deterministic_mode = ctx.deterministicAlgorithms();
  PTDLOG(OP)
      << "[GCU_DETERMINISTIC] _index_put_impl_: setting deterministic mode to "
      << deterministic_mode;
  topsaten::topsatenSetDeterministicMode(deterministic_mode);

  if (is_cpu_scalar(value)) {
    auto xvalue = scalarTensorToTopsatenScalar(value);
    bridge_topsatenIndexPutImpl_out1(self, dev_indices, xvalue, accumulate,
                                     unsafe);
  } else {
    bridge_topsatenIndexPutImpl_out1(self, dev_indices, value, accumulate,
                                     unsafe);
  }
  return self;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim,
                        const at::Tensor& index, const at::Tensor& src,
                        at::Tensor& out) {
  {
    // TODO
    // after topsaten fix, delete
    dim = at::maybe_wrap_dim(dim, self.dim());

    structured_scatter_src_gcu_out op(out);
    op.meta(self, dim, index, src);
    auto xreduce = getScatterOperatorEnum("update");

    // Sync global deterministic mode to operator library
    auto& ctx = at::globalContext();
    bool deterministic_mode = ctx.deterministicAlgorithms();
    PTDLOG(OP) << "[GCU_DETERMINISTIC] scatter.src_out: Setting deterministic "
                  "mode to "
               << deterministic_mode;
    topsaten::topsatenSetDeterministicMode(deterministic_mode);

    bridge_topsatenScatter_out1(op.maybe_get_output(0), self, index, src, dim,
                                xreduce);
    return out;
  }
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim,
                        const at::Tensor& index, const at::Scalar& value,
                        at::Tensor& out) {
  {
    // TODO
    // after topsaten fix, delete
    dim = at::maybe_wrap_dim(dim, self.dim());

    aotops::resize_out(out, self.sizes(), {}, self.options());
    if (out.numel() > 0) {
      // Sync global deterministic mode to operator library
      auto& ctx = at::globalContext();
      bool deterministic_mode = ctx.deterministicAlgorithms();
      PTDLOG(OP) << "[GCU_DETERMINISTIC] scatter.value_out: Setting "
                    "deterministic mode to "
                 << deterministic_mode;
      topsaten::topsatenSetDeterministicMode(deterministic_mode);

      auto xself = createTopsatenTensor(self);
      auto xindex = createTopsatenTensor(index);
      auto xvalue = scalarToTopsatenScalar(value);
      auto xout = createTopsatenTensor(out);
      auto reduce_type = getScatterOperatorEnum("update");
      auto stream = getCurrentGCUStream();

      auto value_tensor = at::native::wrapped_scalar_tensor(value);
      auto op_info = [&]() -> std::string {
        std::stringstream ss;
        // clang-format off
        ss << "topsatenScatter" << ": {\n"
           << tensorArgsToString({self, index, value_tensor}, {out})
           << "dim: " << dim << "\n"
           << "reduce_type: update" << "\n"
           << "stream: " << (topsStream_t)stream << "\n"
           << "}\n";
        // clang-format on
        return ss.str();
      };
      PTDLOG(OP) << op_info();
      auto status = topsaten::topsatenScatter(xout, xself, xindex, xvalue, dim,
                                              reduce_type, stream);

      CHECK_TOPSATEN_CALL(status, op_info);

      maybeGCUStreamSynchronize(stream);
    }

    return out;
  }
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim,
                        const at::Tensor& index, const at::Tensor& src,
                        c10::string_view reduce, at::Tensor& out) {
  TORCH_WARN_ONCE(
      "The reduce argument of torch.scatter with Tensor src is deprecated and "
      "will be removed ",
      "in a future PyTorch release. Use torch.scatter_reduce instead for more "
      "reduction options.");

  {
    // TODO
    // after topsaten fix, delete
    dim = at::maybe_wrap_dim(dim, self.dim());

    structured_scatter_reduce_gcu_out op(out);
    op.meta(self, dim, index, src, reduce);
    auto xreduce = getScatterOperatorEnum(reduce);

    // Sync global deterministic mode to operator library
    auto& ctx = at::globalContext();
    bool deterministic_mode = ctx.deterministicAlgorithms();
    PTDLOG(OP)
        << "[GCU_DETERMINISTIC] scatter.reduce_out: Setting deterministic mode "
           "to "
        << deterministic_mode;
    topsaten::topsatenSetDeterministicMode(deterministic_mode);

    bridge_topsatenScatter_out1(op.maybe_get_output(0), self, index, src, dim,
                                xreduce);
    return out;
  }
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim,
                        const at::Tensor& index, const at::Scalar& value,
                        c10::string_view reduce, at::Tensor& out) {
  {
    // TODO
    // after topsaten fix, delete
    dim = at::maybe_wrap_dim(dim, self.dim());

    aotops::resize_out(out, self.sizes(), {}, self.options());
    if (out.numel() > 0) {
      // Sync global deterministic mode to operator library
      auto& ctx = at::globalContext();
      bool deterministic_mode = ctx.deterministicAlgorithms();
      PTDLOG(OP) << "[GCU_DETERMINISTIC] scatter.value_reduce_out: Setting "
                    "deterministic "
                    "mode to "
                 << deterministic_mode;
      topsaten::topsatenSetDeterministicMode(deterministic_mode);

      auto xself = createTopsatenTensor(self);
      auto xindex = createTopsatenTensor(index);
      auto xvalue = scalarToTopsatenScalar(value);
      auto xout = createTopsatenTensor(out);
      auto reduce_type = getScatterOperatorEnum(reduce);
      auto stream = getCurrentGCUStream();

      auto value_tensor = at::native::wrapped_scalar_tensor(value);
      auto op_info = [&]() -> std::string {
        std::stringstream ss;
        // clang-format off
        ss << "topsatenScatter" << ": {\n"
           << tensorArgsToString({self, index, value_tensor}, {out})
           << "reduce: " << reduce << "\n"
           << "dim: " << dim << "\n"
           << "stream: " << (topsStream_t)stream << "\n"
           << "}\n";
        // clang-format on
        return ss.str();
      };
      PTDLOG(OP) << op_info();

      auto status = topsaten::topsatenScatter(xout, xself, xindex, xvalue, dim,
                                              reduce_type, stream);

      CHECK_TOPSATEN_CALL(status, op_info);

      maybeGCUStreamSynchronize(stream);
    }

    return out;
  }
}

at::Tensor& nonzero_out(const at::Tensor& self, at::Tensor& out) {
  // 1. get output shape
  int32_t non_zero_count{0};
  auto x_self = createTopsatenTensor(self);
  auto stream = getCurrentGCUStream();
  auto shape_infer_op_info = [&self, &stream]() {
    std::stringstream ss;
    // clang-format off
      ss << "topsatenCountNonzero" << ": {\n"      \
         << "self: " << tensorToString(self) \
         << "stream: " << stream << "\n" \
         << "}\n";
    // clang-format on
    return ss.str();
  };
  PTDLOG(OP) << shape_infer_op_info();
  CHECK_TOPSATEN_CALL(
      topsaten::topsatenCountNonzero(&non_zero_count, x_self, stream),
      shape_infer_op_info);
  torch_gcu::stream_synchronize(stream);

  // 2. resize output shape
  std::vector<int64_t> out_shape{non_zero_count, self.dim()};
  if (!out.defined()) {
    out = at::empty(out_shape, self.options().dtype(at::kLong));
  } else {
    aotops::resize_output(out, out_shape);
  }

  if (out.numel() > 0) {
    // 3. call topsaten nonzero
    auto x_out = createTopsatenTensor(out);
    auto op_info = [&self, &out, &stream]() {
      std::stringstream ss;
      // clang-format off
     ss << "topsatenNonzero" << ": {\n" \
        << "self: " << tensorToString(self) \
        << "out: " << tensorToString(out)  \
        << "stream: " << stream << "\n" \
        << "}\n";
      // clang-format on
      return ss.str();
    };
    PTDLOG(OP) << op_info();
    CHECK_TOPSATEN_CALL(topsaten::topsatenNonzero(x_out, x_self, stream),
                        op_info);
    maybeGCUStreamSynchronize(stream);
  }

  return out;
}

at::Tensor nonzero(const at::Tensor& self) {
  at::Tensor out;
  nonzero_out(self, out);
  return out;
}

}  // namespace aotops

}  // namespace torch_gcu
