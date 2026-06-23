#include "python/gcu_module.h"

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/CallOnce.h>
#include <c10/util/UniqueVoidPtr.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

#include <thread>
#include <unordered_set>

#include "gcu/gcu_allocator_config.h"
#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_common.h"
#include "gcu/gcu_context.h"
#include "gcu/gcu_functions.h"
#include "gcu/gcu_generator_impl.h"
#include "gcu/gcu_graph.h"
#include "gcu/gcu_graphs_utils.h"
#include "gcu/gcu_misc_functions.h"
#include "gcu/gcu_stream.h"
#include "python/gcu_py_stream.h"

#ifdef USE_KINETO_GCU
#include "profiler/autograd/python_autograd.h"
#include "profiler/python/combined_traceback.h"
#else
#include <torch/csrc/profiler/python/combined_traceback.h>
using CapturedTraceback = torch::CapturedTraceback;
#endif
#include "profiler/python/init.h"
#include "python/gcu_memory_snapshot.h"

using namespace torch;
using namespace torch_gcu;

static bool in_bad_fork = false;  // True for children forked after gcu init

#ifndef WIN32
// Called in the forked child if gcu has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_device_init(at::kPrivateUse1, true);
}
#endif

// Should be called before the first gcu call.
// Note: This is distinct from initExtension because a stub gcu implementation
// has some working functions (e.g. device_count) but cannot fully initialize.
static void poison_fork() {
#ifndef WIN32
  static c10::once_flag flag;
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

PyObject* THGPModule_setDevice_wrap(PyObject* /*self*/, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackLong(arg);

  torch::utils::device_lazy_init(at::kPrivateUse1);
  torch_gcu::set_device(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_exchangeDevice(PyObject* /*self*/, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = torch_gcu::ExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_maybeExchangeDevice(PyObject* /*self*/, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = torch_gcu::MaybeExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_getDevice_wrap(PyObject* /*self*/, PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int32_t>(torch_gcu::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_canDeviceAccessPeer_wrap(PyObject* /*self*/,
                                              PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    THPUtils_invalidArguments(args, nullptr, "can_device_peer_access", 1,
                              "(int device, int peer_device);");
    return nullptr;
  }
  TORCH_CHECK(THPUtils_checkLong(arg1),
              "invalid argument to canDeviceAccessPeer");
  TORCH_CHECK(THPUtils_checkLong(arg2),
              "invalid argument to canDeviceAccessPeer");
  auto device = THPUtils_unpackDeviceIndex(arg1);
  auto peer_device = THPUtils_unpackDeviceIndex(arg2);

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto can_access = torch_gcu::canDeviceAccessPeer(device, peer_device);
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_getDeviceCount_wrap(PyObject* /*self*/,
                                         PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  poison_fork();
  return THPUtils_packUInt64(torch_gcu::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_getArchFlags(PyObject* /*self*/, PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  poison_fork();
#ifdef GCU_ARCH_FLAGS
  static const char* flags = C10_STRINGIZE(GCU_ARCH_FLAGS);
  return THPUtils_packString(flags);
#else
  Py_RETURN_NONE;
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPModule_isInBadFork(PyObject* /*self*/,
                                        PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_getCurrentStream_wrap(PyObject* /* unused */,
                                           PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(device_index),
              "invalid argument to getCurrentStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  auto stream = torch_gcu::getCurrentGCUStream(c10_device_index);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(output_tuple, 0,
                  THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(output_tuple, 1,
                  THPUtils_packDeviceIndex(stream.device_index()));
  PyTuple_SetItem(
      output_tuple, 2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_getCurrentStream_raw(PyObject* /* unused */,
                                          PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(device_index),
              "invalid argument to getCurrentStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  return PyLong_FromVoidPtr(
      torch_gcu::getCurrentGCUStream(c10_device_index).stream());
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_getDefaultStream_wrap(PyObject* /* unused */,
                                           PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(device_index),
              "invalid argument to getDefaultStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  auto stream = torch_gcu::getDefaultGCUStream(c10_device_index);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(output_tuple, 0,
                  THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(output_tuple, 1,
                  THPUtils_packDeviceIndex(stream.device_index()));
  PyTuple_SetItem(
      output_tuple, 2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_setStream_wrap(PyObject* /*self*/, PyObject* args,
                                    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {"stream_id", "device_index", "device_type",
                                    nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|LLL",
                                   const_cast<char**>(kwlist), &stream_id,
                                   &device_index, &device_type)) {
  }

  auto stream = torch_gcu::GCUStream::unpack3(
      stream_id, device_index, static_cast<c10::DeviceType>(device_type));

  auto device = torch_gcu::current_device();
  if (device != stream.device_index()) {
    torch_gcu::set_device(stream.device_index());
  }
  torch_gcu::setCurrentGCUStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// PyObject* THGPModule_getCompiledVersion(PyObject* self, PyObject* noargs) {
//   return THPUtils_packInt64((int64_t)TOPS_VERSION);
// }

// PyObject* THGPModule_gcuHostAllocator(PyObject* _unused, PyObject* noargs) {
//   HANDLE_TH_ERRORS
//   c10::Allocator* allocator = torch_gcu::getCachingHostAllocator();
//   return PyLong_FromVoidPtr(allocator);
//   END_HANDLE_TH_ERRORS
// }

PyObject* THGPModule_gcuCachingAllocator_raw_alloc(PyObject* /*_unused*/,
                                                   PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    THPUtils_invalidArguments(args, nullptr, "caching_allocator_alloc", 1,
                              "(ssize_t size, intptr_t stream);");
    return nullptr;
  }
  auto size = PyLong_AsSsize_t(size_o);
  topsStream_t stream = static_cast<topsStream_t>(PyLong_AsVoidPtr(stream_o));
  void* mem =
      torch_gcu::GCUCachingAllocator::raw_alloc_with_stream(size, stream);
  return PyLong_FromVoidPtr(mem);
  END_HANDLE_TH_ERRORS
}

// Unpack a PyObject to at::Scalar, throw an exception if it fails
at::Scalar as_scalar(PyObject* arg) {
  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't
  // currently handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(arg)) {
    return THPVariable_Unpack(arg).item();
  }

  if (THPUtils_checkLong(arg)) {
    return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(arg)));
  }

  if (PyBool_Check(arg)) {
    return at::Scalar(THPUtils_unpackBool(arg));
  }

  if (PyComplex_Check(arg)) {
    return at::Scalar(THPUtils_unpackComplexDouble(arg));
  }
  return at::Scalar(THPUtils_unpackDouble(arg));
}

PyObject* THGPModule_gcuCachingAllocator_raw_delete(PyObject* /*_unused*/,
                                                    PyObject* obj) {
  HANDLE_TH_ERRORS
  void* mem_ptr = PyLong_AsVoidPtr(obj);
  torch_gcu::GCUCachingAllocator::raw_delete(mem_ptr);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_gcuCachingAllocator_set_allocator_settings(
    PyObject* /*_unused*/, PyObject* env) {
  HANDLE_TH_ERRORS
  torch_gcu::GCUCachingAllocator::setAllocatorSettings(
      THPUtils_unpackString(env));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_getAllocatorBackend(PyObject* /*_unused*/,
                                         PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(torch_gcu::GCUCachingAllocator::name());
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_topsMallocHostAccessible(PyObject* /*_unused*/,
                                              PyObject* args) {
  HANDLE_TH_ERRORS

  PyObject* shape_obj = nullptr;
  PyObject* dtype_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &shape_obj, &dtype_obj)) {
    THPUtils_setError("topsMallocHostAccessable() expected (shape, dtype)");
    return nullptr;
  }

  std::vector<int64_t> shape = py::cast<std::vector<int64_t>>(shape_obj);

  at::ScalarType dtype;
  if (!THPDtype_Check(dtype_obj)) {
    THPUtils_setError("dtype must be a torch.dtype");
    return nullptr;
  }
  dtype = reinterpret_cast<THPDtype*>(dtype_obj)->scalar_type;

  size_t element_size = at::elementSize(dtype);
  size_t num_elements = 1;
  for (auto dim : shape) {
    if (dim <= 0) {
      THPUtils_setError("Invalid dimension size: %ld", dim);
      return nullptr;
    }
    num_elements *= static_cast<size_t>(dim);
  }

  size_t total_bytes = num_elements * element_size;

  void* handle = nullptr;
  topsError_t status =
      topsExtMallocWithFlags(&handle, total_bytes, topsMallocHostAccessable);

  if (status != topsSuccess || handle == nullptr) {
    THPUtils_setError(
        "Failed to allocate %zu bytes of host-accessible memory (error: %d)",
        total_bytes, status);
    return nullptr;
  }

  auto deleter = [](void* ptr) {
    if (ptr) topsFree(ptr);
  };

  const auto device = at::DeviceType::PrivateUse1;
  auto options = at::TensorOptions().dtype(dtype).device(device);

  at::Tensor tensor = at::from_blob(
      handle, shape, std::function<void(void*)>(deleter), options);

  return THPVariable_Wrap(tensor);

  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_gcuSynchronize(PyObject* /*_unused*/,
                                    PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  torch_gcu::device_synchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// PyObject* THGPModule_gcuSleep(PyObject* _unused, PyObject* cycles) {
//   HANDLE_TH_ERRORS
//   TORCH_CHECK(THPUtils_checkLong(cycles),
//                   "torch_gcu._sleep(): expected 'int'");
//   torch_gcu::sleep(THPUtils_unpackLong(cycles));
//   Py_RETURN_NONE;
//   END_HANDLE_TH_ERRORS
// }

// We need to ensure that as long as a thread will NEVER loose the GIL as long
// as it holds the GCU mutex. Otherwise another thread might be scheduled and
// try to e.g. allocate a new tensor which will cause a deadlock. It's enough to
// have a single global, because it can be only set once (gcuMutex is not
// recursive) by the thread that owns the mutex (obviously there can be only one
// such thread).
static PyGILState_STATE gcuMutexGILState;

PyObject* THGPModule_gcuLockMutex(PyObject* /*module*/, PyObject* /*noargs*/) {
  auto mutex = torch_gcu::getFreeMutex();
  // This has to be a busy loop because we **absolutely need to** hold the GIL
  // or it's a recipe for a deadlock otherwise (if we let other Python threads
  // run while we have the gcuMutex, but not the GIL, they might try to e.g.
  // free a GCU tensor and acquire the gcuMutex without giving up the GIL,
  // because it happens deep within THC).
  while (true) {
    if (mutex->try_lock()) break;
    {
      pybind11::gil_scoped_release no_gil;
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  gcuMutexGILState = PyGILState_Ensure();
  Py_RETURN_NONE;
}

PyObject* THGPModule_gcuUnlockMutex(PyObject* /*module*/,
                                    PyObject* /*noargs*/) {
  auto mutex = torch_gcu::getFreeMutex();
  PyGILState_Release(gcuMutexGILState);
  mutex->unlock();
  Py_RETURN_NONE;
}

PyObject* THGPModule_hasPrimaryContext(PyObject* /*_unused*/, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg),
              "invalid argument to has_primary_context");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (torch_gcu::hasPrimaryContext(device_index)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_setMemoryFraction(PyObject* /*_unused*/, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* fraction_o = nullptr;
  PyObject* device_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
    THPUtils_invalidArguments(args, nullptr, "set_memory_fraction", 1,
                              "(double fraction, int device);");
    return nullptr;
  }
  double fraction = PyFloat_AsDouble(fraction_o);
  int64_t device = PyLong_AsLongLong(device_o);

  torch_gcu::GCUCachingAllocator::setMemoryFraction(fraction, device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THGPModule_emptyCache(PyObject* /*_unused*/, PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  torch_gcu::GCUCachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THGPModule_memoryStats(PyObject* /*_unused*/, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const auto device = THPUtils_unpackDeviceIndex(arg);

  using torch_gcu::GCUCachingAllocator::DeviceStats;
  using torch_gcu::GCUCachingAllocator::Stat;
  using torch_gcu::GCUCachingAllocator::StatArray;
  using torch_gcu::GCUCachingAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      torch_gcu::GCUCachingAllocator::getDeviceStats(device);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["requested_bytes"] = statArrayToDict(stats.requested_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_resetAccumulatedMemoryStats(PyObject* /*_unused*/,
                                                 PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg),
              "invalid argument to reset_accumulated_memory_stats");
  const auto device = THPUtils_unpackDeviceIndex(arg);
  torch_gcu::GCUCachingAllocator::resetAccumulatedStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THGPModule_resetPeakMemoryStats(PyObject* /*_unused*/,
                                          PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg),
              "invalid argument to reset_peak_memory_stats");
  const auto device = THPUtils_unpackDeviceIndex(arg);
  torch_gcu::GCUCachingAllocator::resetPeakStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

CapturedTraceback* getFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  if (CapturedTraceback* sc = dynamic_cast<CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

PyObject* THGPModule_memorySnapshot(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  using torch_gcu::GCUCachingAllocator::BlockInfo;
  using torch_gcu::GCUCachingAllocator::SegmentInfo;

  py::str device_s = "device";
  py::str address_s = "address";
  py::str total_size_s = "total_size";
  py::str allocated_size_s = "allocated_size";
  py::str active_size_s = "active_size";
  py::str requested_size_s = "requested_size";
  py::str stream_s = "stream";
  py::str segment_type_s = "segment_type";
  py::str segment_pool_id = "segment_pool_id";
  py::str large_s = "large";
  py::str small_s = "small";
  py::str size_s = "size";
  py::str state_s = "state";
  py::str active_allocated_s = "active_allocated";
  py::str active_pending_free_s = "active_pending_free";
  py::str inactive_s = "inactive";
  py::str addr_s = "addr";
  py::str cpp_frames_s = "cpp_frames";
  py::str blocks_s = "blocks";
  py::str is_expandable_s = "is_expandable";
  py::str frames_s = "frames";

  py::list empty_frames;
  std::vector<CapturedTraceback*> to_gather_frames;
  std::vector<py::dict> to_gather_dest;

  auto add_frame_key = [&](const py::dict& d,
                           const std::shared_ptr<c10::GatheredContext> ctx) {
    if (ctx) {
      auto sc = getFromContext(ctx);
      to_gather_frames.emplace_back(sc);
      to_gather_dest.emplace_back(d);
    } else {
      d[frames_s] = empty_frames;
    }
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict[device_s] = segmentInfo.device;
    segmentDict[address_s] = segmentInfo.address;
    segmentDict[total_size_s] = segmentInfo.total_size;
    segmentDict[allocated_size_s] = segmentInfo.allocated_size;
    segmentDict[active_size_s] = segmentInfo.active_size;
    segmentDict[requested_size_s] = segmentInfo.requested_size;
    // we want the python objects to pickle easily so use an int to
    // represent the stream rather than a torch_gcu.stream object
    segmentDict[stream_s] = int64_t(segmentInfo.stream);
    segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);
    // segmentDict[segment_pool_id] = segmentInfo.owner_private_pool_id;
    // segmentDict[is_expandable_s] = segmentInfo.is_expandable;
    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    auto address = segmentInfo.address;
    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict[address_s] = address;
      blockDict[size_s] = blockInfo.size;
      blockDict[requested_size_s] = blockInfo.requested_size;
      blockDict[state_s] =
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s));
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      blocks.append(blockDict);
      address += blockInfo.size;
    }
    segmentDict[blocks_s] = blocks;

    return segmentDict;
  };

  auto snapshot = torch_gcu::GCUCachingAllocator::snapshot();

  py::list segments;

  for (const auto& segmentInfo : snapshot.segments) {
    segments.append(segmentInfoToDict(segmentInfo));
  }

  py::list traces;
  py::str action_s = "action";
  py::str alloc_s = "alloc";
  py::str free_requested_s = "free_requested";
  py::str free_completed_s = "free_completed";
  py::str segment_alloc_s = "segment_alloc";
  py::str segment_free_s = "segment_free";
  py::str segment_map_s = "segment_map";
  py::str segment_unmap_s = "segment_unmap";

  py::str snapshot_s = "snapshot";
  py::str oom_s = "oom";
  py::str device_free_s = "device_free";

  using namespace torch_gcu::GCUCachingAllocator;

  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    py::list trace;
    for (const auto& te : traceInfo) {
      py::dict trace_entry;
      if (te.context_) {
        // without further compression frames can get really large on dump
        auto sc = getFromContext(te.context_);
        to_gather_frames.emplace_back(sc);
        to_gather_dest.emplace_back(trace_entry);
      }
      trace_entry[action_s] = action_to_str(te.action_);
      trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] =
          te.addr_;
      trace_entry[size_s] = te.size_;
      trace_entry[stream_s] = int64_t(te.stream_);
      trace.append(trace_entry);
    }
    traces.append(trace);
  }

  py::dict result;
  result["segments"] = segments;
  result["device_traces"] = traces;

  auto frames = py_symbolize(to_gather_frames);
  for (auto i : c10::irange(frames.size())) {
    to_gather_dest.at(i)[frames_s] = frames.at(i);
  }

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_attachOutOfMemoryObserver(PyObject* /*_unused*/,
                                               PyObject* observer) {
  HANDLE_TH_ERRORS
  Py_XINCREF(observer);
  auto obs = [observer](int64_t device, int64_t alloc, int64_t device_allocated,
                        int64_t device_free) {
    py::gil_scoped_acquire g;
    PyObject* result = PyObject_CallFunction(observer, "LLLL", device, alloc,
                                             device_allocated, device_free);
    if (!result) {
      throw py::error_already_set();
    }
    Py_XDECREF(result);
  };
  torch_gcu::GCUCachingAllocator::attachOutOfMemoryObserver(std::move(obs));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_gcuSetSyncDebugMode(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_WARN_ONCE(
      "Synchronization debug mode is a prototype feature and does not yet "
      "detect all "
      "synchronizing operations");
  TORCH_CHECK(THPUtils_checkLong(arg),
              "invalid argument to set_sync_debug_mode");
  int64_t debug_mode = THPUtils_unpackLong(arg);
  TORCH_CHECK(debug_mode >= 0 && debug_mode <= 2,
              "invalid value of debug_mode, expected one of 0,1,2");
  torch_gcu::SyncDebugMode l;
  switch (debug_mode) {
    case 0:
      l = torch_gcu::SyncDebugMode::L_DISABLED;
      break;
    case 1:
      l = torch_gcu::SyncDebugMode::L_WARN;
      break;
    case 2:
      l = torch_gcu::SyncDebugMode::L_ERROR;
      break;
    default:
      l = torch_gcu::SyncDebugMode::L_DISABLED;
      break;  // can't happen
  }
  torch_gcu::warning_state().set_sync_debug_mode(l);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THGPModule_gcuGetSyncDebugMode(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto debug_mode = torch_gcu::warning_state().get_sync_debug_mode();
  switch (debug_mode) {
    case torch_gcu::SyncDebugMode::L_DISABLED:
      return THPUtils_packInt32(0);
    case torch_gcu::SyncDebugMode::L_WARN:
      return THPUtils_packInt32(1);
    case torch_gcu::SyncDebugMode::L_ERROR:
      return THPUtils_packInt32(2);
    default:
      return THPUtils_packInt32(-1);  // can't happen
  }
  END_HANDLE_TH_ERRORS
}

////////////////////////////////////////////////////////////////////////////////
// Gcu module initialization
////////////////////////////////////////////////////////////////////////////////

static void registerGcuDeviceProperties(PyObject* module) {
  // Add _gcuDevicePropertires class to torch._C
  auto m = py::handle(module).cast<py::module>();
  py::class_<topsDeviceProp_t>(m, "_GcuDeviceProperties")
      .def_readonly("name", &topsDeviceProp_t::name)
      .def_readonly("major", &topsDeviceProp_t::major)
      .def_readonly("minor", &topsDeviceProp_t::minor)
      // .def_readonly("is_multi_gcu_board", &topsDeviceProp_t::isMultiGcuBoard)
      // .def_readonly("is_integrated", &topsDeviceProp_t::integrated)
      .def_readonly("multi_processor_count",
                    &topsDeviceProp_t::multiProcessorCount)
      .def_readonly("total_memory", &topsDeviceProp_t::totalGlobalMem)
      .def_readonly("max_threads_per_multi_processor",
                    &topsDeviceProp_t::maxThreadsPerMultiProcessor)
      .def_readonly("sip_count", &topsDeviceProp_t::sipCount)
      .def("__repr__", [](const topsDeviceProp_t& prop) {
        std::ostringstream stream;
        stream << "_GcuDeviceProperties(name='" << prop.name
               << "', major=" << prop.major << ", minor=" << prop.minor
               << ", total_memory=" << prop.totalGlobalMem / (1024 * 1024)
               << "MB, multi_processor_count=" << prop.multiProcessorCount
               << ", sip_count=" << prop.sipCount << ")";
        return stream.str();
      });

  m.def("_gcu_record_memory_history_legacy",
        static_cast<void (*)(bool, bool, int64_t, bool, bool)>(
            torch_gcu::_record_memory_history));

  m.def("_gcu_record_memory_history",
        static_cast<void (*)(c10::optional<std::string>,
                             c10::optional<std::string>, std::string, size_t)>(
            torch_gcu::_record_memory_history));

  m.def("_gcu_isHistoryEnabled",
        []() { return torch_gcu::GCUCachingAllocator::isHistoryEnabled(); });

  // m.def("_gcu_get_conv_benchmark_empty_cache",
  //       []() { return torch_gcu::_topsdnn_get_conv_benchmark_empty_cache();
  //       });

  // m.def("_topsdnn_set_conv_benchmark_empty_cache", [](bool enable) {
  //   return torch_gcu::_topsdnn_set_conv_benchmark_empty_cache(enable);
  // });
}

// We choose to ignore certain blocks that are currently allocated
// when we set the pool to its checkpoint. For those blocks, we need
// to swap out the deleter function of their corresponding blocks
// so that a deallocation is not triggered when they die.
void removeStorageDeleterFns(
    const std::vector<c10::StorageImpl*>& stale_live_storages,
    std::unordered_set<void*> definitely_stale_pointers) {
  for (c10::StorageImpl* stale_storage : stale_live_storages) {
    auto ptr = stale_storage->data_ptr().get();
    auto allocated_pointer = definitely_stale_pointers.find(ptr);
    TORCH_CHECK(allocated_pointer != definitely_stale_pointers.end());
    auto t = torch_gcu::GCUCachingAllocator::get();
    bool succeeded = stale_storage->mutable_data_ptr().compare_exchange_deleter(
        t->raw_deleter(), &c10::detail::deleteNothing);

    TORCH_CHECK(
        succeeded,
        "Unexpected deleter function on storage, could not swap function");
  }
}

void addStorageDeleterFns(
    std::vector<c10::StorageImpl*>& storages_to_add_deleters_to,
    torch_gcu::GCUCachingAllocator::CheckpointDelta& delta) {
  std::unordered_map<void*, c10::StorageImpl*> storages;
  for (auto& storage : storages_to_add_deleters_to) {
    storages[storage->data_ptr().get()] = storage;
  }

  for (auto& data_ptr : delta.dataptrs_allocd) {
    auto storage_pair = storages.find(data_ptr.get());
    if (storage_pair != storages.end()) {
      auto ctx = storage_pair->second->data_ptr().get_context();
      TORCH_CHECK(ctx == nullptr, " Not expecting deleter function");
      storage_pair->second->set_data_ptr_noswap(std::move(data_ptr));
    } else {
      data_ptr.release_context();
    }
  }
}

static void bindGetDeviceProperties(PyObject* module) {
  // Add method to torch.gcu
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_get_device_properties",
      [](int device) -> topsDeviceProp_t* {
        return torch_gcu::getDeviceProperties(device);
      },
      py::return_value_policy::reference);

  // Add _get_sip_count function to directly get SIP count (internal C++
  // binding)
  m.def(
      "_get_sip_count",
      [](int device) -> int {
        auto* props = torch_gcu::getDeviceProperties(device);
        return props->sipCount;
      },
      py::arg("device") = -1,
      "Get the SIP (Streaming Instruction Processor) count for the specified "
      "device.\n"
      "\n"
      "Args:\n"
      "    device (int, optional): Device index. Defaults to current device.\n"
      "\n"
      "Returns:\n"
      "    int: The number of SIPs on the device.");
}

// Callback for python part. Used for additional initialization of python
// classes
static PyObject* THGPModule_initExtension(PyObject* /*self*/,
                                          PyObject* /*noargs*/) {
#if C10_ASAN_ENABLED
  TORCH_WARN(
      "torch.gcu: your pytorch binary has address sanitizer (asan) built in, "
      "asan is currently not compatible with torch.gcu module, "
      "you might get unexpected behavior (eg. out of memory, crash, etc.), "
      "please rebuild pytorch without asan if you need to use this module");
#endif
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork);  // Handled at python level
  poison_fork();
  at::globalContext().lazyInitDevice(at::kPrivateUse1);

  auto m = THPObjectPtr(PyImport_ImportModule("torch.gcu"));
  if (!m) throw python_error();

  bool has_half = true;

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  set_module_attr("has_half", has_half ? Py_True : Py_False);

  auto num_gcus = torch_gcu::device_count();
  auto default_gcu_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gcus));
  for (const auto i : c10::irange(num_gcus)) {
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(
        torch_gcu::getDefaultGCUGenerator(i));
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_gcu_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_gcu_generators);
  bindGetDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// PyObject* THGPModule_getCurrentBlasHandle_wrap(PyObject* self,
//                                                PyObject* noargs) {
//   HANDLE_TH_ERRORS
//   // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
//   cublasHandle_t handle = torch_gcu::getCurrentGCUBlasHandle();
//   return PyLong_FromVoidPtr(handle);
//   END_HANDLE_TH_ERRORS
// }

// static PyObject* THGPModule_clearBlasWorkspaces_wrap(PyObject* self,
//                                                      PyObject* noargs) {
//   HANDLE_TH_ERRORS
//   torch_gcu::clearCublasWorkspaces();
//   Py_RETURN_NONE;
//   END_HANDLE_TH_ERRORS
// }

static PyObject* THGPModule_isCurrentStreamCapturing_wrap(PyObject* self,
                                                          PyObject* noargs) {
  HANDLE_TH_ERRORS
  // If there's no gcu context, torch_gcu::currentStreamCaptureStatus returns
  // CaptureStatus::None without initializing a context.
  if (torch_gcu::currentStreamCaptureStatus() ==
      torch_gcu::CaptureStatus::None) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
  END_HANDLE_TH_ERRORS
}

// PyObject* THGPModule_setBenchmarkLimitTopsDNN(PyObject* _unused,
//                                               PyObject* arg) {
//   TORCH_CHECK(THPUtils_checkLong(arg),
//                   "set_benchmark_limit_cudnn expects an int, "
//                   "but got %s",
//                   THPUtils_typename(arg));
//   auto benchmark_limit = static_cast<int>(THPUtils_unpackLong(arg));
//   at::globalContext().setBenchmarkLimitTopsDNN(benchmark_limit);
//   Py_RETURN_NONE;
// }

// PyObject* THGPModule_benchmarkLimitTopsDNN(PyObject* _unused,
//                                            PyObject* noargs) {
//   return
//   THPUtils_packInt32(at::globalContext().benchmarkLimitTopsDNN());
// }

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)

static struct PyMethodDef _THGPModule_methods[] = {
    {"_gcu_init", THGPModule_initExtension, METH_NOARGS, nullptr},
    {"_gcu_setDevice", THGPModule_setDevice_wrap, METH_O, nullptr},
    {"_gcu_exchangeDevice", THGPModule_exchangeDevice, METH_O, nullptr},
    {"_gcu_maybeExchangeDevice", THGPModule_maybeExchangeDevice, METH_O,
     nullptr},
    {"_gcu_getDevice", THGPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_gcu_getDeviceCount", THGPModule_getDeviceCount_wrap, METH_NOARGS,
     nullptr},
    {"_gcu_canDeviceAccessPeer", THGPModule_canDeviceAccessPeer_wrap,
     METH_VARARGS, nullptr},
    {"_gcu_getArchFlags", THGPModule_getArchFlags, METH_NOARGS, nullptr},
    {"_gcu_isInBadFork", THGPModule_isInBadFork, METH_NOARGS, nullptr},
    {"_gcu_getCurrentStream", THGPModule_getCurrentStream_wrap, METH_O,
     nullptr},
    {"_gcu_getCurrentRawStream", THGPModule_getCurrentStream_raw, METH_O,
     nullptr},
    {"_gcu_getDefaultStream", THGPModule_getDefaultStream_wrap, METH_O,
     nullptr},
    // {"_gcu_getCurrentBlasHandle", THGPModule_getCurrentBlasHandle_wrap,
    //  METH_NOARGS, nullptr},
    // {"_gcu_clearCublasWorkspaces", THGPModule_clearBlasWorkspaces_wrap,
    //  METH_NOARGS, nullptr},
    {"_gcu_isCurrentStreamCapturing", THGPModule_isCurrentStreamCapturing_wrap,
     METH_NOARGS, nullptr},
    {"_gcu_setStream", castPyCFunctionWithKeywords(THGPModule_setStream_wrap),
     METH_VARARGS | METH_KEYWORDS, nullptr},
    // {"_gcu_getCompiledVersion", THGPModule_getCompiledVersion, METH_NOARGS,
    //  nullptr},
    {"_gcu_hasPrimaryContext", THGPModule_hasPrimaryContext, METH_O, nullptr},
    {"_gcu_setMemoryFraction", THGPModule_setMemoryFraction, METH_VARARGS,
     nullptr},
    {"_gcu_emptyCache", THGPModule_emptyCache, METH_NOARGS, nullptr},
    {"_gcu_memoryStats", THGPModule_memoryStats, METH_O, nullptr},
    {"_gcu_resetAccumulatedMemoryStats", THGPModule_resetAccumulatedMemoryStats,
     METH_O, nullptr},
    {"_gcu_resetPeakMemoryStats", THGPModule_resetPeakMemoryStats, METH_O,
     nullptr},
    {"_gcu_memorySnapshot", THGPModule_memorySnapshot, METH_NOARGS, nullptr},
    {"_gcu_attach_out_of_memory_observer", THGPModule_attachOutOfMemoryObserver,
     METH_O, nullptr},
    // {"_gcu_gcuHostAllocator", THGPModule_gcuHostAllocator, METH_NOARGS,
    //  nullptr},
    {"_gcu_gcuCachingAllocator_raw_alloc",
     THGPModule_gcuCachingAllocator_raw_alloc, METH_VARARGS, nullptr},
    {"_gcu_gcuCachingAllocator_raw_delete",
     THGPModule_gcuCachingAllocator_raw_delete, METH_O, nullptr},
    {"_gcu_gcuCachingAllocator_set_allocator_settings",
     THGPModule_gcuCachingAllocator_set_allocator_settings, METH_O, nullptr},
    {"_gcu_getAllocatorBackend", THGPModule_getAllocatorBackend, METH_NOARGS,
     nullptr},
    {"_gcu_tops_malloc_host_accessible", THGPModule_topsMallocHostAccessible,
     METH_VARARGS, nullptr},
    {"_gcu_synchronize", THGPModule_gcuSynchronize, METH_NOARGS, nullptr},
    // {"_gcu_ipc_collect", THGPModule_gcuIPCCollect, METH_NOARGS, nullptr},
    // {"_gcu_sleep", THGPModule_gcuSleep, METH_O, nullptr},
    {"_gcu_lock_mutex", THGPModule_gcuLockMutex, METH_NOARGS, nullptr},
    {"_gcu_unlock_mutex", THGPModule_gcuUnlockMutex, METH_NOARGS, nullptr},
    {"_gcu_set_sync_debug_mode", THGPModule_gcuSetSyncDebugMode, METH_O,
     nullptr},
    {"_gcu_get_sync_debug_mode", THGPModule_gcuGetSyncDebugMode, METH_NOARGS,
     nullptr},
#ifdef USE_KINETO_GCU
    {"_autograd_init", THGPAutograd_initExtension, METH_NOARGS, nullptr},
#endif
    // {"_gcu_jiterator_compile_and_launch_kernel",
    //  THGPModule_gcuJiteratorCompileAndLaunchKernel, METH_VARARGS, nullptr},
    // {"_gcu_get_topsdnn_benchmark_limit", THGPModule_benchmarkLimitTopsDNN,
    //  METH_NOARGS, nullptr},
    // {"_gcu_set_topsdnn_benchmark_limit", THGPModule_setBenchmarkLimitTopsDNN,
    //  METH_O, nullptr},

    // #ifdef USE_ECCL
    //     {"_eccl_version", THGPModule_eccl_version, METH_NOARGS, nullptr},
    //     {"_eccl_unique_id", THGPModule_eccl_unique_id, METH_NOARGS, nullptr},
    //     {"_eccl_init_rank", THGPModule_eccl_init_rank, METH_VARARGS,
    //     nullptr},
    //     {"_eccl_reduce", THGPModule_eccl_reduce, METH_VARARGS, nullptr},
    //     {"_eccl_all_reduce", THGPModule_eccl_all_reduce, METH_VARARGS,
    //     nullptr},
    //     {"_eccl_broadcast", THGPModule_eccl_broadcast, METH_VARARGS,
    //     nullptr},
    //     {"_eccl_all_gather", THGPModule_eccl_all_gather, METH_VARARGS,
    //     nullptr},
    //     {"_eccl_reduce_scatter", THGPModule_eccl_reduce_scatter,
    //     METH_VARARGS,
    //      nullptr},
    // #endif
    {nullptr, nullptr, 0, nullptr},
};

PyMethodDef* THGPModule_get_methods() { return _THGPModule_methods; }

static void registerGcuPluggableAllocator(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<torch_gcu::GCUCachingAllocator::AllocatorState,
             std::shared_ptr<torch_gcu::GCUCachingAllocator::AllocatorState>>(
      m, "_gcu_GCUAllocator_AllocatorState");

  m.def("_gcu_getCheckpointState",
        [](c10::DeviceIndex device, torch_gcu::MempoolId_t id) {
          return torch_gcu::GCUCachingAllocator::getCheckpointState(device, id);
        });
  m.def("_free_And_Remove_DeleterFn", [](size_t storage_impl_ptr) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
    auto alloc = torch_gcu::GCUCachingAllocator::get();
    auto data_ptr = storage_impl->data_ptr().get();
    bool succeeded = storage_impl->mutable_data_ptr().compare_exchange_deleter(
        alloc->raw_deleter(), c10::detail::deleteNothing);
    TORCH_CHECK(succeeded, "Expected standard deleter");
    torch_gcu::GCUCachingAllocator::raw_delete(data_ptr);
  });
  m.def("_has_Standard_Deleter", [](size_t storage_impl_ptr) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
    auto alloc = torch_gcu::GCUCachingAllocator::get();
    return (storage_impl->data_ptr().get_deleter() == alloc->raw_deleter());
  });
  m.def("_gcu_beginAllocateCurrentStreamToPool",
        [](c10::DeviceIndex device, torch_gcu::MempoolId_t mempool_id) {
          auto stream = torch_gcu::getCurrentGCUStream(device);
          TORCH_CHECK(stream, "Expected stream capture to be under way");
          torch_gcu::GCUCachingAllocator::beginAllocateToPool(
              device, mempool_id,
              [stream](topsStream_t target) { return target == stream; });
        });
  m.def("_gcu_beginAllocateToPool",
        [](c10::DeviceIndex device, torch_gcu::MempoolId_t mempool_id) {
          torch_gcu::GCUCachingAllocator::beginAllocateToPool(
              device, mempool_id, [](topsStream_t) { return true; });
        });
  m.def("_gcu_beginAllocateCurrentThreadToPool",
        [](c10::DeviceIndex device, torch_gcu::MempoolId_t mempool_id) {
          auto tid = std::this_thread::get_id();
          torch_gcu::GCUCachingAllocator::beginAllocateToPool(
              device, mempool_id, [=](topsStream_t) {
                auto current_tid = std::this_thread::get_id();
                return current_tid == tid;
              });
        });
  m.def("_gcu_endAllocateCurrentStreamToPool",
        [](c10::DeviceIndex device, torch_gcu::MempoolId_t mempool_id) {
          torch_gcu::GCUCachingAllocator::endAllocateToPool(device, mempool_id);
        });
  m.def("_gcu_endAllocateToPool",
        [](c10::DeviceIndex device, torch_gcu::MempoolId_t mempool_id) {
          torch_gcu::GCUCachingAllocator::endAllocateToPool(device, mempool_id);
        });
  m.def("_gcu_releasePool",
        [](c10::DeviceIndex device, torch_gcu::MempoolId_t mempool_id) {
          torch_gcu::GCUCachingAllocator::releasePool(device, mempool_id);
        });
  m.def("_construct_GCU_Tensor_From_Storage_And_Metadata",
        [](py::dict& metadata, c10::Storage s) {
          auto dtype_arg = metadata["dtype"].ptr();
          auto meta = c10::scalarTypeToTypeMeta(torch::toScalarType(dtype_arg));

          constexpr c10::DispatchKeySet gcu_dks(c10::DispatchKey::PrivateUse1);
          at::Tensor tensor = at::detail::make_tensor_base<c10::TensorImpl>(
              std::move(s), gcu_dks, meta);

          tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
              metadata["size"].cast<std::vector<int64_t>>(),
              metadata["stride"].cast<std::vector<int64_t>>());
          tensor.unsafeGetTensorImpl()->set_storage_offset(
              metadata["storage_offset"].cast<int64_t>());
          return tensor;
        });
  m.def("_gcu_checkPoolLiveAllocations",
        [](c10::DeviceIndex device, torch_gcu::MempoolId_t mempool_id,
           const py::set& expected_live_allocations) {
          std::unordered_set<void*> allocations;
          allocations.reserve(expected_live_allocations.size());
          for (auto& elem : expected_live_allocations) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            allocations.insert(reinterpret_cast<void*>(py::cast<size_t>(elem)));
          }
          return torch_gcu::GCUCachingAllocator::checkPoolLiveAllocations(
              device, mempool_id, allocations);
        });
  m.def("_gcu_setCheckpointPoolState",
        [](c10::DeviceIndex device,
           std::shared_ptr<torch_gcu::GCUCachingAllocator::AllocatorState> pps,
           const std::vector<size_t>& stale_storages_ptr,
           const std::vector<size_t>& storages_to_add_deleters_to_ptr = {}) {
          std::unordered_set<c10::StorageImpl*> ptr_set;
          // iterate on std::vector for determinism
          std::vector<c10::StorageImpl*> ptrs;
          for (size_t ptr_int : stale_storages_ptr) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            c10::StorageImpl* ptr = (c10::StorageImpl*)ptr_int;
            if (!ptr_set.count(ptr)) {
              ptrs.push_back(ptr);
              ptr_set.insert(ptr);
            }
          }
          auto delta = torch_gcu::GCUCachingAllocator::setCheckpointPoolState(
              device, std::move(pps));
          auto& freed_pointers = delta.ptrs_freed;

          std::unordered_set<void*> allocd_set;
          for (auto& data_ptr : delta.dataptrs_allocd) {
            allocd_set.insert(data_ptr.get());
          }
          std::unordered_set<void*> freed_pointer_set;
          size_t definite_freed_count = 0;
          for (void* ptr : freed_pointers) {
            if (!allocd_set.count(ptr)) {
              definite_freed_count += 1;
            }
            freed_pointer_set.insert((ptr));
          }
          // that block has already been freed,
          // so even those this will error, so too will the allocator
          // when the corresponding tensor dies because there is no
          // live tensor corresponding to it
          TORCH_CHECK(ptr_set.size() >= definite_freed_count,
                      "Any stale tensors which are being manually freed"
                      " must be passed to set checkpoint");

          removeStorageDeleterFns(ptrs, freed_pointer_set);
          std::vector<c10::StorageImpl*> storages_to_add_deleters_to;
          storages_to_add_deleters_to.reserve(
              storages_to_add_deleters_to_ptr.size());
          for (size_t ptr_int : storages_to_add_deleters_to_ptr) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            storages_to_add_deleters_to.push_back((c10::StorageImpl*)ptr_int);
          }

          addStorageDeleterFns(storages_to_add_deleters_to, delta);
        });
}

namespace torch_gcu {

namespace shared {

void initGcurtBindings(PyObject* module);
void initTopstxBindings(PyObject* module);

}  // namespace shared

namespace python {

std::vector<c10::optional<torch_gcu::GCUStream>>
THGPUtils_PySequence_to_GCUStreamList(PyObject* obj) {
  if (!PySequence_Check(obj)) {
    throw std::runtime_error(
        "Expected a sequence in THGPUtils_PySequence_to_GCUStreamList");
  }
  THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, nullptr));
  if (seq.get() == nullptr) {
    throw std::runtime_error("expected PySequence, but got " +
                             std::string(THPUtils_typename(obj)));
  }

  std::vector<c10::optional<torch_gcu::GCUStream>> streams;
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject* stream = PySequence_Fast_GET_ITEM(seq.get(), i);

    if (PyObject_IsInstance(stream, THGPStreamClass)) {
      streams.emplace_back(torch_gcu::GCUStream::unpack3(
          (reinterpret_cast<THGPStream*>(stream))->stream_id,
          (reinterpret_cast<THGPStream*>(stream))->device_index,
          static_cast<c10::DeviceType>(
              (reinterpret_cast<THGPStream*>(stream))->device_type)));
    } else if (stream == Py_None) {
      streams.emplace_back();
    } else {
      std::runtime_error(
          "Unknown data type found in stream list. Need torch_npu.npu.Stream "
          "or None");
    }
  }
  return streams;
}

// gcu_common
void initCommMethods(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def(
       "_broadcast_coalesced",
       [](std::vector<at::Tensor>& tensors, std::vector<int64_t> devices,
          size_t buffer_size) {
         return torch_gcu::broadcast_coalesced(tensors, devices, buffer_size);
       },
       py::arg("tensors"), py::arg("devices"), py::arg("buffer_size"),
       py::call_guard<py::gil_scoped_release>())
      .def(
          "_broadcast",
          [](at::Tensor& tensor, std::vector<int64_t> devices) {
            return torch_gcu::broadcast(tensor, devices);
          },
          py::arg("tensor"), py::arg("devices"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_broadcast_out",
          [](at::Tensor& tensor, std::vector<at::Tensor>& out_tensors) {
            return torch_gcu::broadcast_out(tensor, out_tensors);
          },
          py::arg("tensor"), py::arg("out"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_scatter",
          [](at::Tensor& tensor, std::vector<int64_t>& devices,
             c10::optional<std::vector<int64_t>> chunk_sizes, int64_t dim,
             c10::optional<py::object> py_streams) {
            c10::optional<std::vector<c10::optional<torch_gcu::GCUStream>>>
                streams;
            if (py_streams) {
              py::handle handle = *py_streams;
              streams = THGPUtils_PySequence_to_GCUStreamList(handle.ptr());
            }
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return torch_gcu::scatter(tensor, devices, chunk_sizes, dim,
                                      streams);
          },
          py::arg("tensor"), py::arg("devices"), py::arg("chunk_sizes"),
          py::arg("dim"), py::arg("streams"))
      .def(
          "_scatter_out",
          [](at::Tensor& tensor, std::vector<at::Tensor>& out_tensors,
             int64_t dim, c10::optional<py::object> py_streams) {
            c10::optional<std::vector<c10::optional<torch_gcu::GCUStream>>>
                streams;
            if (py_streams) {
              py::handle handle = *py_streams;
              streams = THGPUtils_PySequence_to_GCUStreamList(handle.ptr());
            }
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return torch_gcu::scatter_out(tensor, out_tensors, dim, streams);
          },
          py::arg("tensor"), py::arg("out"), py::arg("dim"), py::arg("streams"))
      .def(
          "_gather",
          [](std::vector<at::Tensor>& tensors, int64_t dim,
             c10::optional<int32_t> destination_index) {
            return torch_gcu::gather(tensors, dim, destination_index);
          },
          py::arg("tensors"), py::arg("dim"), py::arg("destination_index"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_gather_out",
          [](std::vector<at::Tensor>& tensors, at::Tensor& out_tensor,
             int64_t dim) {
            return torch_gcu::gather_out(tensors, out_tensor, dim);
          },
          py::arg("tensors"), py::arg("out"), py::arg("dim"),
          py::call_guard<py::gil_scoped_release>());
}
}  // namespace python

void initModule(PyObject* module) {
  python::initCommMethods(module);  // torch/csrc/cuda/python_comm.cpp
  shared::initGcurtBindings(module);
  shared::initTopstxBindings(module);
  registerGcuDeviceProperties(module);
#ifdef USE_KINETO_GCU
  torch_gcu::profiler::initPythonBindings(module);
#endif
  registerGcuPluggableAllocator(module);
}

}  // namespace torch_gcu
