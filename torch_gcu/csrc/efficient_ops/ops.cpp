#include "efficient_ops/ops.h"

#include <ATen/MemoryOverlap.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/IListRef_inl.h>

#include "aten/aot_ops/topsaten_bridge.h"
#include "topsaten/topsaten_ops.h"

namespace torch_gcu {
namespace efficient {

static at::Tensor restride_src(const at::Tensor& src, int64_t dims_before,
                               int64_t dims_indexed,
                               at::IntArrayRef replacement_shape) {
  auto shape = at::DimVector(src.sizes());
  auto strides = at::DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(shape.begin() + dims_before, replacement_shape.begin(),
               replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

static at::Tensor reshape_indexer(const at::Tensor& index, int64_t dims_before,
                                  int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = at::DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

struct AdvancedIndex {
  AdvancedIndex(const at::Tensor& src, at::TensorList indices);

  at::Tensor src;
  std::vector<at::Tensor> indices;
  at::DimVector indexed_sizes;
  at::DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};

AdvancedIndex::AdvancedIndex(const at::Tensor& src,
                             at::TensorList indices_list) {
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  at::IntArrayRef replacement_shape;
  for (const auto dim : c10::irange(indices_list.size())) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) !=
          indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) ==
          replacement_shape.end()) {
    TORCH_CHECK_INDEX(false,
                      "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }
}

at::Tensor create_out(at::IntArrayRef sizes, at::IntArrayRef strides,
                      const at::TensorOptions& options) {
  if (strides.empty()) {
    return torch_gcu::aotops::empty(sizes, options);
  } else {
    return torch_gcu::aotops::empty_strided(sizes, strides, options);
  }
}

[[noreturn]] static void invalid_mask(const at::Tensor& self, int64_t idx,
                                      const at::Tensor& mask, int64_t maskIdx) {
  TORCH_CHECK_INDEX(false, "The shape of the mask ", mask.sizes(), " at index ",
                    maskIdx, " does not match the shape of the indexed tensor ",
                    self.sizes(), " at index ", idx);
}

C10_UNUSED static std::vector<at::Tensor> expandTensors(
    const at::Tensor& self, at::IOptTensorListRef indices) {
  // If indices come in as ByteTensor or BoolTensor (masks), expand them into
  // the equivalent indexing by LongTensors
  std::vector<at::Tensor> result;
  for (const auto& index_opt : indices) {
    if (!index_opt.has_value()) {
      result.emplace_back();
    } else {
      const auto& index = *index_opt;
      if (index.scalar_type() == at::kByte ||
          index.scalar_type() == at::kBool) {
        if (index.scalar_type() == at::kByte) {
          TORCH_WARN(
              "indexing with dtype torch.uint8 is now deprecated,"
              " please use a dtype torch.bool instead.");
        }
        // The sizes of the ByteTensor mask or bool tensor must match the sizes
        // of the corresponding dimensions in self
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = static_cast<int64_t>(result.size() + j);
          if (index.size(j) != self.size(srcIdx)) {
            invalid_mask(self, srcIdx, index, j);
          }
        }
        // Replace with nonzeros
        auto nonzero = index.nonzero();
        for (const auto j : c10::irange(index.dim())) {
          result.emplace_back(nonzero.select(1, j));
        }
      } else {
        result.emplace_back(index);
      }
    }
  }
  return result;
}

C10_UNUSED static void checkIndexTensorTypes(at::IOptTensorListRef indices,
                                             bool allow_int = false) {
  for (const auto& tensor : indices) {
    if (tensor.has_value() && tensor->defined()) {
      auto scalarType = tensor->scalar_type();
      if (allow_int) {
        if (scalarType != at::kLong && scalarType != at::kByte &&
            scalarType != at::kBool && scalarType != at::kInt) {
          TORCH_CHECK_INDEX(false,
                            "tensors used as indices must be long, int, byte "
                            "or bool tensors");
        }
      } else {
        if (scalarType != at::kLong && scalarType != at::kByte &&
            scalarType != at::kBool) {
          TORCH_CHECK_INDEX(
              false,
              "tensors used as indices must be long, byte or bool tensors");
        }
      }
    }
  }
}

C10_UNUSED static bool hasContiguousSubspace(at::TensorList tl) {
  // true if all the non-null tensors are adjacent
  auto isDefined = [](const at::Tensor& tensor) { return tensor.defined(); };
  auto isNull = [](const at::Tensor& tensor) { return !tensor.defined(); };
  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
  auto it = std::find_if(start, stop.base(), isNull);
  return it == stop.base();
}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
// transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
// tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
C10_UNUSED static std::tuple<at::Tensor, std::vector<at::Tensor>>
transposeToFront(const at::Tensor& self, at::TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<at::Tensor> transposedIndices;
  dims.reserve(self.dim());
  for (const auto i : c10::irange(self.dim())) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

inline std::string shapes_as_str(at::TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}

AdvancedIndex make_info(at::Tensor self, at::IOptTensorListRef orig) {
  checkIndexTensorTypes(orig, /*allow_int*/ true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more
  // LongTensors
  auto indices = expandTensors(self, orig);
  // next broadcast all index tensors together
  try {
    indices = expand_outplace(indices);
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(
        false,
        "shape mismatch: indexing tensors could not be broadcast together"
        " with shapes ",
        shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  // Ensure indices are on the same device as self
  for (auto& indice : indices) {
    if (indice.defined() && indice.device() != self.device()) {
      TORCH_CHECK_INDEX(
          false,
          "indice error, indice should be in the same device as self "
          "but got self: ",
          self.device(), " indice: ", indice.device());
    }
  }
  for (auto& indice : indices) {
    if (indice.defined() && indice.dtype() != at::kInt) {
      indice = indice.to(at::kLong);
      TORCH_CHECK_INDEX(false, "indice error, should be int in gcu_index ",
                        "but got ", indice.dtype())
    }
  }

  return AdvancedIndex(self, indices);
}

static void check_indices_on_selfdevice(
    const at::Tensor& self, const at::MaterializedIOptTensorListRef& indices) {
  auto dev = self.device();
  bool indices_on_dev = std::all_of(
      indices.begin(), indices.end(), [=](const at::OptionalTensorRef& opt) {
        return opt.has_value() ? (opt->is_cpu() || opt->device() == dev) : true;
      });
  TORCH_CHECK(
      indices_on_dev,
      "indices should be either on the same device as the indexed tensor (",
      dev, ")");
}

static void build_index_op(at::TensorIteratorBase& iter,
                           const AdvancedIndex& info,
                           const at::Tensor& result) {
  // 'TensorIterator' needs to own the things coming from 'info', since
  // 'info' will be destroyed after the META function.
  at::TensorIteratorConfig config;
  // info.src is a restrided view of result
  config.set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .add_output(result)
      .add_owned_const_input(info.src);
  for (auto& index : info.indices) {
    config.add_owned_const_input(index);
  }
  if (!result.defined()) {
    config.declare_static_dtype_and_device(info.src.scalar_type(),
                                           info.src.device());
  }
  iter.build(config);
}

structured_device_int32_indices_index_Tensor::meta_return_ty
structured_device_int32_indices_index_Tensor::meta(
    const at::Tensor& self, at::IOptTensorListRef indices) {
  auto materialized = indices.materialize();

  TORCH_CHECK_INDEX(materialized.size() <= (size_t)self.dim(),
                    "too many indices for tensor of dimension ", self.dim(),
                    " (got ", materialized.size(), ")");

  // Only allow: `dev_tensor[{cpu,dev}_tensor]`.
  // See: https://github.com/pytorch/pytorch/pull/69607
  check_indices_on_selfdevice(self, materialized);

  const auto& result = maybe_get_output();

  if (result.defined()) {
    TORCH_CHECK(self.scalar_type() == result.scalar_type(), "index_out: self (",
                self.scalar_type(), ") and result (", result.scalar_type(),
                ") must have the same scalar type");
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, self);
    for (const at::OptionalTensorRef& index : materialized) {
      if (index.has_value()) {
        at::assert_no_overlap(result, *index);
      }
    }
  }

  auto info = make_info(self, std::move(indices));
  build_index_op(*this, info, result);
  return TORCH_PRECOMPUTE_STRUCT2(device_int32_indices_index, Tensor)()
      .set_sizes(std::move(info.indexed_sizes))
      .set_strides(std::move(info.indexed_strides));
}

at::Tensor device_int32_indices_index(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices) {
  structured_device_int32_indices_index_Tensor_gcu_functional op;
  op.meta(self, at::IOptTensorListRef(indices));
  bridge_topsatenIndex_out1(op.maybe_get_output(0), self, indices);
  return op.outputs_[0];
}

}  // namespace efficient
}  // namespace torch_gcu