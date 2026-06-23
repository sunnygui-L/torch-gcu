/*
 * Copyright 2024 Enflame. All Rights Reserved.
 */

#include "gcu/gcu_utils.h"

#include <ATen/native/ReduceOpsUtils.h>
#include <c10/core/ScalarType.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <torch/script.h>

#include <set>

#include "gcu/gcu_context.h"
#include "gcu/gcu_functions.h"
#include "gcu/gcu_generator_impl.h"
#include "gcu/gcu_hardware.h"
#include "gcu/logging.h"

namespace torch_gcu {

namespace {

int GetIgnoredTopsatenStatus() {
  static int val = []() {
    const char *env = std::getenv("TORCH_GCU_IGNORE_TOPSATEN_STATUS");
    return (env != nullptr) ? std::atoi(env) : -1;
  }();
  return val;
}

template <typename scalar_t>
void PrintDataPtr(const scalar_t *p, const int number, std::stringstream &ss) {
  int n_for_line = 10;
  auto line = number / n_for_line + 1;
  for (int i = 0; i < line; i++) {
    for (int j = 0; j < 10; j++) {
      auto idx = i * n_for_line + j;
      if (idx >= number) {
        break;
      }
      ss << std::left << std::setw(10) << p[idx] << ", ";
    }
    ss << "\n";
  }
}

}  // namespace

void CHECK_TOPSATEN_CALL(topsatenStatus_t status) {
  if (status != TOPSATEN_STATUS_SUCCESS) {
    if (static_cast<int>(status) == GetIgnoredTopsatenStatus()) {
      PTDLOG(TORCH_GCU)
          << "Call TOPSOP fail, but environment variable "
             "TORCH_GCU_IGNORE_TOPSATEN_STATUS set to ignore status of "
          << status << ".";
      return;
    }
    PTCHECK(0) << "Call TOPSOP fail, got error: " << status;
  }
}

void CHECK_TOPSATEN_CALL(topsatenStatus_t status,
                         std::function<std::string()> op_info) {
  if (status != TOPSATEN_STATUS_SUCCESS) {
    if (static_cast<int>(status) == GetIgnoredTopsatenStatus()) {
      PTDLOG(TORCH_GCU)
          << "Call TOPSOP fail, but environment variable "
             "TORCH_GCU_IGNORE_TOPSATEN_STATUS set to ignore status of "
          << status << ".";
      return;
    }
    PTCHECK(0) << "Call TOPSOP fail, got error: " << status << "!\n"
               << op_info();
  }
}

topsatenTensor createTopsatenTensor(const at::Tensor &tensor) {
  if (!tensor.defined()) return {};

  PTCHECK(is_gcu(tensor)) << "tensor's device must be gcu, but with "
                          << tensor.device();
  auto dims = tensor.sizes().vec();
  auto strides = tensor.strides().vec();
  auto rank = tensor.dim();
  if (rank == 0) {
    rank = 1;
    dims = {1};
    strides = {1};
  }
  auto xdims = topsatenSize_t{dims.data(), rank};
  auto xstrides = topsatenSize_t{strides.data(), rank};
  auto xdtype = scalarTypeToTopsatenDataType(tensor.scalar_type());
  PTDLOG(TORCH_GCU) << "topsopCreateTensor: {\n"
                    << tensorToString(tensor) << "}\n";
  auto xt = topsatenTensor(xdims, xstrides, xdtype, gcu_data_ptr(tensor));
  if (EF_DLOG_IS_ON(TORCH_GCU, TORCH_GCU, OP)) {
    auto t_dims = xt.GetTensorShape();
    auto t_strides = xt.GetTensorStrides();
    auto t_dtype = xt.GetTensorDataType();
    PTDLOG(OP) << "Create tensor handle sueeccd with:\n"
               << " dims: " << t_dims << "\n"
               << " strides: " << t_strides << "\n"
               << " dtype: " << t_dtype << "\n";
  }
  return xt;
}

topsatenTensor createTopsatenTensor_tmp(const at::Tensor &tensor) {
  if (!tensor.defined()) return {};

  PTCHECK(is_gcu(tensor)) << "tensor's device must be gcu, but with "
                          << tensor.device();
  auto dims = tensor.sizes().vec();
  auto strides = tensor.strides().vec();
  auto rank = tensor.dim();
  if (rank == 0) {
    rank = 1;
    dims = {1};
    strides = {1};
  }
  auto xdims = topsatenSize_t{dims.data(), rank};
  auto xstrides = topsatenSize_t{strides.data(), rank};
  auto xdtype = scalarTypeToTopsatenDataType_tmp(tensor.scalar_type());
  PTDLOG(TORCH_GCU) << "topsopCreateTensor: {\n"
                    << tensorToString(tensor) << "}\n";
  auto xt = topsatenTensor(xdims, xstrides, xdtype, gcu_data_ptr_tmp(tensor));
  if (EF_DLOG_IS_ON(TORCH_GCU, TORCH_GCU, OP)) {
    auto t_dims = xt.GetTensorShape();
    auto t_strides = xt.GetTensorStrides();
    auto t_dtype = xt.GetTensorDataType();
    PTDLOG(OP) << "Create tensor handle sueeccd with:\n"
               << " dims: " << t_dims << "\n"
               << " strides: " << t_strides << "\n"
               << " dtype: " << t_dtype << "\n";
  }
  return xt;
}

topsatenTensor optionalTensorToTopsatenTensor(
    const c10::optional<at::Tensor> &opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return createTopsatenTensor(*opt_tensor);
  } else {
    return {};
  }
}

topsatenTensor optionalTensorRefToTopsatenTensor(
    const at::OptionalTensorRef opt_tensor) {
  if (opt_tensor) {
    return createTopsatenTensor(opt_tensor.getTensorRef());
  } else {
    return {};
  }
}

#define FETCH_AND_RETUEN_SCALAR_CASE(type, scalartype) \
  case c10::ScalarType::scalartype:                    \
    return at::Scalar(c10::load<type>(tensor.data_ptr()));

at::Scalar scalarTensorToScalar(const at::Tensor &tensor) {
  if (!(tensor.is_cpu())) {
    TORCH_WARN("convert non cpu scalar tensor to at::Scalar, with device = ",
               tensor.device());
    return tensor.item();
  }
  switch (tensor.scalar_type()) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(FETCH_AND_RETUEN_SCALAR_CASE)
    default:
      PTCHECK(false) << "unsupported cast tensor to Scalar "
                     << tensor.scalar_type();
  }
  return at::Scalar(0);  // just to avoid compiler warning
}

topsatenScalar_t scalarTensorToTopsatenScalar(const at::Tensor &tensor) {
  if (!(tensor.is_cpu())) {
    TORCH_WARN(
        "convert non cpu scalar tensor to topsatenScalar_t, with device = ",
        tensor.device());
  }
  return scalarToTopsatenScalar(scalarTensorToScalar(tensor));
}

topsatenScalar_t scalarTensorToTopsatenScalar(
    const at::Tensor &tensor, const c10::ScalarType &scalar_type) {
  if (!(tensor.is_cpu())) {
    TORCH_WARN(
        "convert non cpu scalar tensor to topsatenScalar_t, with device = ",
        tensor.device());
  }
  return scalarToTopsatenScalar(scalarTensorToScalar(tensor), scalar_type);
}

topsatenDataType_t scalarTypeToTopsatenDataType(
    const c10::ScalarType &scalar_type) {
  auto hardware = HardwareType::GetInstance().getHardware();
  auto narrow_dtype = get_gcu_scalar_type(scalar_type);
  switch (narrow_dtype) {
    case c10::ScalarType::Bool:
      return TOPSATEN_DATA_PRED;
    case c10::ScalarType::Byte:
      return TOPSATEN_DATA_U8;
    case c10::ScalarType::UInt16:
      return TOPSATEN_DATA_U16;
    case c10::ScalarType::UInt32:
      return TOPSATEN_DATA_U32;
    case c10::ScalarType::UInt64:
      return TOPSATEN_DATA_U64;
    case c10::ScalarType::Char:
      return TOPSATEN_DATA_I8;
    case c10::ScalarType::Short:
      return TOPSATEN_DATA_I16;
    case c10::ScalarType::Int:
      return TOPSATEN_DATA_I32;
    case c10::ScalarType::Long:
      return TOPSATEN_DATA_I64;
    case c10::ScalarType::Float8_e5m2:
      return TOPSATEN_DATA_FP8E5M2;
    case c10::ScalarType::Float8_e4m3fn:
      return TOPSATEN_DATA_FP8E4M3;
    case c10::ScalarType::Half:
      return TOPSATEN_DATA_FP16;
    case c10::ScalarType::BFloat16:
      return TOPSATEN_DATA_BF16;
    case c10::ScalarType::Float:
      return TOPSATEN_DATA_FP32;
    case c10::ScalarType::ComplexHalf:
      return TOPSATEN_DATA_CFP16;
    case c10::ScalarType::ComplexFloat:
      return TOPSATEN_DATA_CFP32;
    default: {
      TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", narrow_dtype,
                            " to topsatenDataType_t.");
      return TOPSATEN_DATA_FP32;
    }
  }
}

topsatenDataType_t scalarTypeToTopsatenDataType_tmp(
    const c10::ScalarType &scalar_type) {
  auto hardware = HardwareType::GetInstance().getHardware();
  auto narrow_dtype = get_gcu_scalar_type_tmp(scalar_type);
  switch (narrow_dtype) {
    case c10::ScalarType::Bool:
      return TOPSATEN_DATA_PRED;
    case c10::ScalarType::Byte:
      return TOPSATEN_DATA_U8;
    case c10::ScalarType::UInt16:
      return TOPSATEN_DATA_U16;
    case c10::ScalarType::UInt32:
      return TOPSATEN_DATA_U32;
    case c10::ScalarType::UInt64:
      return TOPSATEN_DATA_U64;
    case c10::ScalarType::Char:
      return TOPSATEN_DATA_I8;
    case c10::ScalarType::Short:
      return TOPSATEN_DATA_I16;
    case c10::ScalarType::Int:
      return TOPSATEN_DATA_I32;
    case c10::ScalarType::Long:
      return TOPSATEN_DATA_I64;
    case c10::ScalarType::Float8_e5m2:
      return TOPSATEN_DATA_FP8E5M2;
    case c10::ScalarType::Float8_e4m3fn:
      return TOPSATEN_DATA_FP8E4M3;
    case c10::ScalarType::Half:
      return TOPSATEN_DATA_FP16;
    case c10::ScalarType::BFloat16:
      return TOPSATEN_DATA_BF16;
    case c10::ScalarType::Float:
      return TOPSATEN_DATA_FP32;
    case c10::ScalarType::ComplexHalf:
      return TOPSATEN_DATA_CFP16;
    case c10::ScalarType::ComplexFloat:
      return TOPSATEN_DATA_CFP32;
    default: {
      TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", narrow_dtype,
                            " to topsatenDataType_t.");
      return TOPSATEN_DATA_FP32;
    }
  }
}

topsatenDataType_t optionalScalarTypeToTopsatenDataType(
    const c10::optional<at::ScalarType> &opt_type) {
  if (opt_type.has_value()) {
    return scalarTypeToTopsatenDataType(opt_type.value());
  } else {
    return TOPSATEN_DATA_NONE;
  }
}

template <typename T>
void memcpyComplexScalar(const at::Scalar &value, topsatenScalar_t &dst) {
  T real_value = value.to<T>();
  char *real_value_ptr = (reinterpret_cast<char *>(&real_value));

  std::set<topsatenDataType_t> int_types = {
      TOPSATEN_DATA_I8,    TOPSATEN_DATA_U8,   TOPSATEN_DATA_I16,
      TOPSATEN_DATA_U16,   TOPSATEN_DATA_I32,  TOPSATEN_DATA_U32,
      TOPSATEN_DATA_I64,   TOPSATEN_DATA_U64,  TOPSATEN_DATA_PRED,
      TOPSATEN_DATA_I4,    TOPSATEN_DATA_CI8,  TOPSATEN_DATA_CU8,
      TOPSATEN_DATA_CI16,  TOPSATEN_DATA_CU16, TOPSATEN_DATA_CI32,
      TOPSATEN_DATA_CU32,  TOPSATEN_DATA_CI64, TOPSATEN_DATA_CU64,
      TOPSATEN_DATA_CPRED, TOPSATEN_DATA_CI4,
  };

  char *value_ptr;
  if (int_types.find(dst.dtype) != int_types.end()) {
    value_ptr = (reinterpret_cast<char *>(&dst.ival));
  } else {
    value_ptr = (reinterpret_cast<char *>(&dst.fval));
  }
  for (int i = 0; i < sizeof(T); ++i) {
    value_ptr[i] = real_value_ptr[i];
  }
}

topsatenScalar_t scalarToTopsatenScalar(const at::Scalar &value,
                                        const c10::ScalarType &scalar_type) {
  topsatenScalar_t xvalue;
  switch (scalar_type) {
    case c10::ScalarType::Bool:
      xvalue.dtype = TOPSATEN_DATA_PRED;
      xvalue.ival = value.to<bool>();
      return xvalue;
    case c10::ScalarType::Byte:
      xvalue.dtype = TOPSATEN_DATA_U8;
      xvalue.ival = value.to<uint8_t>();
      return xvalue;
    case c10::ScalarType::UInt16:
      xvalue.dtype = TOPSATEN_DATA_U16;
      xvalue.ival = value.to<uint16_t>();
      return xvalue;
    case c10::ScalarType::UInt32:
      xvalue.dtype = TOPSATEN_DATA_U32;
      xvalue.ival = value.to<uint32_t>();
      return xvalue;
    case c10::ScalarType::UInt64:
      xvalue.dtype = TOPSATEN_DATA_U64;
      xvalue.ival = value.to<uint64_t>();
      return xvalue;
    case c10::ScalarType::Char:
      xvalue.dtype = TOPSATEN_DATA_I8;
      xvalue.ival = value.to<int8_t>();
      return xvalue;
    case c10::ScalarType::Short:
      xvalue.dtype = TOPSATEN_DATA_I16;
      xvalue.ival = value.to<int16_t>();
      return xvalue;
    case c10::ScalarType::Int:
      xvalue.dtype = TOPSATEN_DATA_I32;
      xvalue.ival = value.to<int>();
      return xvalue;
    case c10::ScalarType::Long:
      xvalue.dtype = TOPSATEN_DATA_I64;
      xvalue.ival = value.to<int64_t>();
      return xvalue;
    case c10::ScalarType::Half:
      xvalue.dtype = TOPSATEN_DATA_FP16;
      xvalue.fval = value.to<at::Half>();
      return xvalue;
    case c10::ScalarType::BFloat16:
      xvalue.dtype = TOPSATEN_DATA_BF16;
      xvalue.fval = value.to<at::BFloat16>();
      return xvalue;
    case c10::ScalarType::Float:
      xvalue.dtype = TOPSATEN_DATA_FP32;
      xvalue.fval = value.to<float>();
      return xvalue;
    case c10::ScalarType::Double:
      xvalue.dtype = TOPSATEN_DATA_F64;
      xvalue.fval = value.to<double>();
      return xvalue;
    case c10::ScalarType::Float8_e4m3fn:
      xvalue.dtype = TOPSATEN_DATA_FP8E4M3;
      xvalue.fval = value.to<at::Float8_e4m3fn>();
      return xvalue;
    case c10::ScalarType::Float8_e5m2:
      xvalue.dtype = TOPSATEN_DATA_FP8E5M2;
      xvalue.fval = value.to<at::Float8_e5m2>();
      return xvalue;
    case c10::ScalarType::ComplexHalf:
      xvalue.dtype = TOPSATEN_DATA_CFP16;
      memcpyComplexScalar<c10::complex<c10::Half>>(value, xvalue);
      return xvalue;
    case c10::ScalarType::ComplexFloat:
      xvalue.dtype = TOPSATEN_DATA_CFP32;
      memcpyComplexScalar<c10::complex<float>>(value, xvalue);
      return xvalue;
    // // TODO:
    // // topsatenScalar_t not support
    // case c10::ScalarType::ComplexDouble:
    //   xvalue.dtype = TOPSATEN_DATA_CF64;
    //   xvalue.xxx_val = value.to<c10::complex<double>>();
    //   return xvalue;
    default: {
      PTCHECK(false) << "can't handle " << scalar_type;
    }
  }
  return xvalue;
}

topsatenScalar_t scalarToTopsatenScalar(const at::Scalar &value) {
  topsatenScalar_t xvalue;
  c10::ScalarType scalar_type = value.type();
  switch (scalar_type) {
    case c10::ScalarType::Bool:
      xvalue.dtype = TOPSATEN_DATA_PRED;
      xvalue.ival = value.to<bool>();
      return xvalue;
    case c10::ScalarType::UInt64:
      xvalue.dtype = TOPSATEN_DATA_U64;
      xvalue.ival = value.to<uint64_t>();
      return xvalue;
    case c10::ScalarType::Long:
      xvalue.dtype = TOPSATEN_DATA_I64;
      xvalue.ival = value.to<int64_t>();
      return xvalue;
    case c10::ScalarType::Double:
      xvalue.dtype = TOPSATEN_DATA_F64;
      xvalue.fval = value.to<double>();
      return xvalue;
    // TODO:
    // topsatenScalar_t not support
    // case c10::ScalarType::ComplexDouble:
    //   xvalue.dtype = TOPSATEN_DATA_CF64;
    //   xvalue.xxx_val = value.to<c10::complex<double>>();
    //   return xvalue;
    default: {
      PTCHECK(false) << "can't handle " << scalar_type;
    }
  }
  return xvalue;
}

topsatenScalar_t optionalScalarToTopsatenScalar(
    const c10::optional<at::Scalar> &opt_scalar) {
  if (opt_scalar.has_value()) {
    return scalarToTopsatenScalar(*opt_scalar);
  } else {
    return {TOPSATEN_DATA_NONE, {.ival = 0}};
  }
}

topsatenScalar_t optionalScalarRefToTopsatenScalar(
    const c10::OptionalRef<at::Scalar> opt_scalar) {
  if (opt_scalar) {
    return scalarToTopsatenScalar(opt_scalar.get());
  } else {
    return {TOPSATEN_DATA_NONE, {.ival = 0}};
  }
}

topsatenScalar_t optionalScalarToTopsatenScalar(
    const c10::optional<int64_t> &opt_int) {
  if (opt_int.has_value()) {
    return scalarToTopsatenScalar(at::Scalar(*opt_int));
  } else {
    return {TOPSATEN_DATA_NONE, {.ival = 0}};
  }
}

topsatenScalar_t optionalScalarToTopsatenScalar(
    const c10::optional<double> &opt_double) {
  if (opt_double.has_value()) {
    return scalarToTopsatenScalar(at::Scalar(*opt_double));
  } else {
    return {TOPSATEN_DATA_NONE, {.ival = 0}};
  }
}

topsatenScalar_t optionalScalarToTopsatenScalar(
    const c10::optional<bool> &opt_bool) {
  if (opt_bool.has_value()) {
    return scalarToTopsatenScalar(at::Scalar(*opt_bool));
  } else {
    return {TOPSATEN_DATA_NONE, {.ival = 0}};
  }
}

topsatenSize_t intArrayRefToTopsatenSize(const at::IntArrayRef &dims) {
  return {dims.data(), static_cast<int64_t>(dims.size())};
}

topsatenSize_t optionalIntArrayRefToTopsatenSize(
    const at::OptionalIntArrayRef &opt_dims, at::DimVector &dim_vec) {
  if (opt_dims.has_value()) {
    dim_vec = at::DimVector(opt_dims.value());
    return {dim_vec.data(), static_cast<int64_t>(dim_vec.size())};
  } else {
    return {};
  }
}

// TODO: match topsaten
topsatenGenerator_t getDefaultTopsatenGenerator(const at::Tensor &self) {
  PhiloxGcuState::Payload seed{0};
  PhiloxGcuState::Payload offset{0};
  uint32_t offset_intragraph{0};
  bool captured{false};
  auto gen = getDefaultGCUGenerator(self.get_device()).get<GCUGeneratorImpl>();
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    // std::tie(seed, offset) = gen->philox_engine_inputs(self.numel());
    PhiloxGcuState state = gen->philox_gcu_state(self.numel());
    seed = state.seed_;
    offset = state.offset_;
    offset_intragraph = state.offset_intragraph_;
    captured = state.captured_;
  }
  // TODO(torch_gcu): return {seed, offset, offset_intragraph};
  return {seed.val, offset.val};
}

// TODO: match topsaten
topsatenGenerator_t generatorToTopsatenGenerator(
    const at::Generator &generator) {
  return {generator.current_seed(), generator.get_offset()};
}

topsatenPhiloxState_t philoxStateToTopsatenPhilocState(
    const PhiloxGcuState &state) {
  topsatenPhiloxState_t tops_state;
  tops_state.captured = state.captured_;
  tops_state.offset_intragraph = state.offset_intragraph_;
  if (state.captured_) {
    tops_state.seed.ptr = state.seed_.ptr;
    tops_state.offset.ptr = state.offset_.ptr;
  } else {
    tops_state.seed.val = state.seed_.val;
    tops_state.offset.val = state.offset_.val;
  }
  return tops_state;
}

// TODO: match topsaten
topsatenGenerator_t optionalGeneratorToTopsatenGenerator(
    const at::Tensor &self, const c10::optional<at::Generator> &generator) {
  uint64_t seed(0), offset(0);
  auto gen = at::get_generator_or_default<GCUGeneratorImpl>(
      generator, getDefaultGCUGenerator(self.device().index()));
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    std::tie(seed, offset) = gen->philox_engine_inputs(self.numel());
  }
  return {seed, offset};
}

topsatenMemoryFormat_t memoryFormatToTopsatenMemoryFormat(
    const at::MemoryFormat &at_format) {
  switch (at_format) {
    case at::MemoryFormat::Contiguous:
      return topsatenMemoryFormat_t::TOPSATEN_MEMORY_CONTIGUOUS;
    case at::MemoryFormat::ChannelsLast:
      return topsatenMemoryFormat_t::TOPSATEN_MEMORY_NHWC;
    case at::MemoryFormat::ChannelsLast3d:
      return topsatenMemoryFormat_t::TOPSATEN_MEMORY_NDHWC;
    case at::MemoryFormat::Preserve:
      return topsatenMemoryFormat_t::TOPSATEN_MEMORY_PRESERVE;
    default:
      TORCH_CHECK(false, "Unsupported memory format: ", at_format);
  }
}

topsatenMemoryFormat_t optionalMemoryFormatToTopsatenMemoryFormat(
    const c10::optional<at::MemoryFormat> &opt_memory_format) {
  if (opt_memory_format.has_value()) {
    return memoryFormatToTopsatenMemoryFormat(*opt_memory_format);
  } else {
    return topsatenMemoryFormat_t::TOPSATEN_MEMORY_NONE;
  }
}

std::string storageToString(const at::Storage &storage) {
  std::stringstream ss;
  if (!(bool(storage))) {
    ss << "{undefined storage}\n";
  } else {
    ss << "{";
    ss << "device: " << storage.device();
    ss << ", nbytes: " << storage.nbytes();
    ss << ", base ptr: " << storage.data();
    ss << "}\n";
  }
  return ss.str();
}

std::string tensorToString(const at::Tensor &tensor) {
  std::stringstream ss;
  if (!tensor.defined()) {
    ss << "{\n";
    ss << " undefined tensor\n";
    ss << "}\n";
  } else if (is_cpu_scalar(tensor)) {
    ss << "{\n";
    ss << " scalar value: " << tensor.item() << "\n";
    ss << "}\n";
  } else {
    ss << "{\n";
    auto storage_offset = tensor.storage_offset();
    if (storage_offset == 0) {
      ss << " data ptr: " << tensor.data_ptr() << "\n";
    } else {
      void *base_ptr = nullptr;
      void *data_ptr = nullptr;
      auto *self_ = tensor.unsafeGetTensorImpl();
      if (!(self_->is_empty())) {
        base_ptr = self_->storage().mutable_data();
        data_ptr = gcu_data_ptr(tensor);
      }
      ss << " base ptr: " << base_ptr << "\n";
      ss << " data ptr: " << data_ptr << "\n";
    }
    ss << " device: " << tensor.device() << "\n";
    ss << " dtype: " << tensor.scalar_type();
    if (is_narrow_type(tensor.scalar_type())) {
      ss << "(actual " << get_gcu_scalar_type(tensor.scalar_type()) << ")";
    }
    ss << "\n";
    ss << " sizes: " << tensor.sizes() << "\n";
    ss << " strides: " << tensor.strides() << "\n";
    ss << " storage_offset: " << storage_offset << "\n";
    ss << " is_contiguous: " << tensor.is_contiguous() << "\n";
    ss << "}\n";
  }
  return ss.str();
}

std::string tensorToString(const c10::optional<at::Tensor> &tensor) {
  if (!tensor) {
    return "{\n undefined tensor\n}\n";
  }
  return tensorToString(*tensor);
}

std::string tensorVectorToString(const std::vector<at::Tensor> &tensors) {
  std::stringstream ss;
  ss << "{\n";
  if (tensors.size() == 0) {
    ss << " There is no tensor!\n";
  } else {
    for (const auto &tensor : tensors) {
      ss << tensorToString(tensor);
    }
  }
  ss << "}\n";
  return ss.str();
}

std::string tensorArgsToString(const std::vector<at::Tensor> &ins,
                               const std::vector<at::Tensor> &outs) {
  std::stringstream ss;
  for (size_t idx = 0; idx < ins.size(); idx++) {
    ss << "input " << idx << ": " << tensorToString(ins[idx]);
  }
  for (size_t idx = 0; idx < outs.size(); idx++) {
    ss << "output " << idx << ": " << tensorToString(outs[idx]);
  }
  return ss.str();
}

void SaveTensorToFile(const at::Tensor &tensor, const std::string &file) {
  PTCHECK(tensor.defined()) << "can't save undefined tensor";
  auto t = tensor.cpu();
  auto bytes = torch::jit::pickle_save(t);
  std::ofstream fout(file, std::ios::out | std::ios::binary);
  if (fout.is_open()) {
    fout.write(bytes.data(), bytes.size());
    fout.close();
    PTDLOG(TORCH_GCU) << "Success save tensor to file: " << file;
  }
}

void SaveStrToFile(const std::string &val, const std::string &file) {
  std::ofstream fout(file, std::ios::out);
  if (fout.is_open()) {
    fout << val << std::endl;
    fout.close();
    PTDLOG(TORCH_GCU) << "Success save to file: " << file;
  }
}

topsatenScatterComputationType_t getScatterOperatorEnum(
    const c10::string_view reduce) {
  if (reduce == "add") {
    return TOPSATEN_SCATTER_COMP_ADD;
  } else if (reduce == "update") {
    return TOPSATEN_SCATTER_COMP_UPDATE;
  } else if (reduce == "multiply") {
    return TOPSATEN_SCATTER_COMP_MUL;
  } else {
    PTCHECK(0) << "unsupported scatter reduce type " << reduce;
    return TOPSATEN_SCATTER_COMP_NUM;
  }
}

namespace {

const char *topsatenDataTypeToName(topsatenDataType_t dtype) {
  switch (dtype) {
    case TOPSATEN_DATA_I8:
      return "int8";
    case TOPSATEN_DATA_U8:
      return "uint8";
    case TOPSATEN_DATA_I16:
      return "int16";
    case TOPSATEN_DATA_U16:
      return "uint16";
    case TOPSATEN_DATA_FP16:
      return "float16";
    case TOPSATEN_DATA_BF16:
      return "bfloat16";
    case TOPSATEN_DATA_I32:
      return "int32";
    case TOPSATEN_DATA_U32:
      return "uint32";
    case TOPSATEN_DATA_FP32:
      return "float32";
    case TOPSATEN_DATA_I64:
      return "int64";
    case TOPSATEN_DATA_U64:
      return "uint64";
    case TOPSATEN_DATA_F64:
      return "float64";
    case TOPSATEN_DATA_PRED:
      return "bool";
    default:
      return "unknown";
  }
}

int64_t topsatenDataTypeSizeBytes(topsatenDataType_t dtype) {
  switch (dtype) {
    case TOPSATEN_DATA_I8:
    case TOPSATEN_DATA_U8:
    case TOPSATEN_DATA_PRED:
      return 1;
    case TOPSATEN_DATA_I16:
    case TOPSATEN_DATA_U16:
    case TOPSATEN_DATA_FP16:
    case TOPSATEN_DATA_BF16:
      return 2;
    case TOPSATEN_DATA_I32:
    case TOPSATEN_DATA_U32:
    case TOPSATEN_DATA_FP32:
      return 4;
    case TOPSATEN_DATA_I64:
    case TOPSATEN_DATA_U64:
    case TOPSATEN_DATA_F64:
      return 8;
    default:
      return 0;
  }
}

template <typename T>
void formatElements(std::stringstream &ss, const void *host_buf, int64_t count,
                    int64_t max_elems) {
  const T *data = static_cast<const T *>(host_buf);
  int64_t to_print = (max_elems > 0 && max_elems < count) ? max_elems : count;
  ss << "[";
  for (int64_t i = 0; i < to_print; ++i) {
    if (i > 0) ss << ", ";
    ss << data[i];
  }
  if (to_print < count) {
    ss << ", ... (" << (count - to_print) << " more)";
  }
  ss << "]";
}

}  // namespace

std::string topsatenTensorToString(const topsatenTensor &xt,
                                   const std::string &name, int64_t max_elems) {
  std::stringstream ss;
  if (!name.empty()) {
    ss << name << ": ";
  }

  auto shape = xt.GetTensorShape();
  auto strides = xt.GetTensorStrides();
  auto dtype = xt.GetTensorDataType();
  auto num_elems = xt.GetTensorElementNums();
  auto dev_ptr = xt.GetTensorData();

  ss << "topsatenTensor {\n";
  ss << "  dtype: " << topsatenDataTypeToName(dtype) << "\n";
  ss << "  shape: [";
  for (int64_t i = 0; i < shape.len; ++i) {
    if (i > 0) ss << ", ";
    ss << shape.data[i];
  }
  ss << "]\n";
  ss << "  strides: [";
  for (int64_t i = 0; i < strides.len; ++i) {
    if (i > 0) ss << ", ";
    ss << strides.data[i];
  }
  ss << "]\n";
  ss << "  num_elems: " << num_elems << "\n";
  ss << "  dev_ptr: " << dev_ptr << "\n";

  int64_t elem_size = topsatenDataTypeSizeBytes(dtype);
  if (dev_ptr == nullptr || num_elems == 0 || elem_size == 0) {
    ss << "  data: (empty or unsupported dtype)\n";
    ss << "}\n";
    return ss.str();
  }

  int64_t nbytes = num_elems * elem_size;
  std::vector<char> host_buf(nbytes);

  auto stream = getCurrentGCUStream();
  memcpy_and_sync(host_buf.data(), dev_ptr, nbytes, topsMemcpyDeviceToHost,
                  stream);

  ss << "  data: ";
  switch (dtype) {
    case TOPSATEN_DATA_FP32:
      formatElements<float>(ss, host_buf.data(), num_elems, max_elems);
      break;
    case TOPSATEN_DATA_F64:
      formatElements<double>(ss, host_buf.data(), num_elems, max_elems);
      break;
    case TOPSATEN_DATA_I32:
      formatElements<int32_t>(ss, host_buf.data(), num_elems, max_elems);
      break;
    case TOPSATEN_DATA_I64:
      formatElements<int64_t>(ss, host_buf.data(), num_elems, max_elems);
      break;
    case TOPSATEN_DATA_I16:
      formatElements<int16_t>(ss, host_buf.data(), num_elems, max_elems);
      break;
    case TOPSATEN_DATA_I8:
      formatElements<int8_t>(ss, host_buf.data(), num_elems, max_elems);
      break;
    case TOPSATEN_DATA_U8:
      formatElements<uint8_t>(ss, host_buf.data(), num_elems, max_elems);
      break;
    case TOPSATEN_DATA_U32:
      formatElements<uint32_t>(ss, host_buf.data(), num_elems, max_elems);
      break;
    case TOPSATEN_DATA_U64:
      formatElements<uint64_t>(ss, host_buf.data(), num_elems, max_elems);
      break;
    case TOPSATEN_DATA_PRED: {
      const uint8_t *bdata = reinterpret_cast<const uint8_t *>(host_buf.data());
      int64_t to_print =
          (max_elems > 0 && max_elems < num_elems) ? max_elems : num_elems;
      ss << "[";
      for (int64_t i = 0; i < to_print; ++i) {
        if (i > 0) ss << ", ";
        ss << (bdata[i] ? "true" : "false");
      }
      if (to_print < num_elems) {
        ss << ", ... (" << (num_elems - to_print) << " more)";
      }
      ss << "]";
      break;
    }
    default:
      ss << "(unsupported dtype for value printing)";
      break;
  }
  ss << "\n}\n";
  return ss.str();
}

void debugPrintTopsatenTensor(const topsatenTensor &xt, const std::string &name,
                              int64_t max_elems) {
  std::cerr << topsatenTensorToString(xt, name, max_elems) << std::flush;
}

}  // namespace torch_gcu
