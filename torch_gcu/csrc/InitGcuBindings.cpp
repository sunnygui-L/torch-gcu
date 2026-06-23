#include <ATen/Parallel.h>
#include <ATen/autocast_mode.h>
#include <Python.h>
#include <c10/util/Optional.h>
#include <python/gcu_launch_host_func.h>
#include <topsaten/topsaten_ops.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/utils/device_lazy_init.h>

#include "aten/GCUNativeFunctions.h"
#include "aten/gcu_conv_determine_backend_memory_format.h"
#include "c10/core/DeviceType.h"
#include "gcu/cpp_frame_info.h"
#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_custom_api.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_macros.h"
#include "gcu/gcu_stream.h"
#include "gcu/python_frame_info.h"
#include "gcu/trace.h"
#include "python/cpp_frame.h"
#include "python/efficient_ops.h"
#include "python/extra_ops.h"
#include "python/gcu_fusion_op.h"
#include "python/gcu_module.h"
#include "python/gcu_py_graph.h"
#include "python/gcu_storage_sharing.h"
#include "python/gcu_tensor_type.h"
#include "python/op_debug_py_interface.h"
#include "python/python_frame.h"
#include "python/storage_narrow.h"
#include "python/torch_util.h"
#include "python/transfer_to_gcu_utils.h"

#ifdef USE_C10D_ECCL
#include "python/dist_py_init.h"
#endif

PyObject *module;

void AddPyMethodDefs(std::vector<PyMethodDef> &vector, PyMethodDef *methods) {
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

PyObject *THPModule_gcu_shutdown(PyObject * /* unused */) { Py_RETURN_NONE; }

static PyObject *is_any_autocast_enabled(PyObject *_unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_autocast_enabled(at::kCUDA) ||
      at::autocast::is_autocast_enabled(at::kCPU) ||
      at::autocast::is_autocast_enabled(at::kXPU) ||
      at::autocast::is_autocast_enabled(at::kIPU) ||
      at::autocast::is_autocast_enabled(at::kXLA) ||
      at::autocast::is_autocast_enabled(at::kHPU) ||
      at::autocast::is_autocast_enabled(at::kPrivateUse1)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject *set_autocast_enabled(PyObject *_unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw torch::TypeError(std::string("enabled must be a bool (got ") +
                           Py_TYPE(arg)->tp_name + ")");
  }
  at::autocast::set_autocast_enabled(at::kPrivateUse1, arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject *is_autocast_enabled(PyObject *_unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_autocast_enabled(at::kPrivateUse1)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject *set_autocast_dtype(PyObject *_unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (!THPDtype_Check(arg)) {
    throw torch::TypeError(std::string("dtype must be a torch.dtype (got ") +
                           Py_TYPE(arg)->tp_name + ")");
  }
  at::ScalarType targetType = reinterpret_cast<THPDtype *>(arg)->scalar_type;
  at::autocast::set_autocast_dtype(at::kPrivateUse1, targetType);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject *get_autocast_dtype(PyObject *_unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  at::ScalarType current_dtype =
      at::autocast::get_autocast_dtype(at::kPrivateUse1);
  auto dtype = (PyObject *)torch::getTHPDtype(current_dtype);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
static PyMethodDef TorchGcuMethods[] = {
    {"_gcu_shutdown", (PyCFunction)THPModule_gcu_shutdown, METH_NOARGS,
     nullptr},
    {"set_autocast_enabled", set_autocast_enabled, METH_O, nullptr},
    {"is_autocast_enabled", is_autocast_enabled, METH_NOARGS, nullptr},
    {"set_autocast_dtype", set_autocast_dtype, METH_O, nullptr},
    {"get_autocast_dtype", get_autocast_dtype, METH_NOARGS, nullptr},
    {"_is_any_autocast_enabled", is_any_autocast_enabled, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

namespace torch_gcu {

void RegisterOpDebugFunc(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_enter_dump_op_name_scope",
        []() { interface::enterDumpOpNameScope(); });
  m.def("_exit_print_cpp_name_scope",
        []() { interface::exitDimpOpNameScope(); });
  m.def("_enter_fallback_cpu_scope",
        []() { interface::enterFallbackCpuScope(); });
  m.def("_exit_fallback_cpu_scope",
        []() { interface::exitFallbackCpuScope(); });
  m.def("_enter_op_check_scope", []() { interface::enterOpCheckScope(); });
  m.def("_exit_op_check_scope", []() { interface::exitOpCheckScope(); });
  m.def("_get_fallback_cpu_scope_state",
        []() -> bool { return interface::getFallbackCpuScopeState(); });
  m.def("_get_dump_op_name_scope_state",
        []() -> bool { return interface::getDumpOpNameScopeState(); });
  m.def("_get_op_check_scope_state",
        []() -> bool { return interface::getOpCheckScopeState(); });
  m.def("_get_op_statistics_info",
        []() -> std::string { return interface::getOpStatisticsInfo(); });
  m.def("_dump_op_statistics_to_json",
        []() { interface::dumpOpStatisticsToJson(); });
  m.def("_clear_statistics", []() { interface::clearOpStatistics(); });
}

void InitOtherBindings(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_gcu_is_bf16_supported", []() { return true; });
  m.def("_gcu_is_support_inf_nan", []() { return true; });
  m.def("_conv_determine_backend_memory_format",
        [](const at::Tensor &input, const at::Tensor &weight,
           const at::native::ConvBackend backend) {
          return torch_gcu::_determine_backend_memory_format(input, weight,
                                                             backend);
        });
  m.def(
      "_user_trace_point",
      [](const std::string &trace_info) {
        // host trace info
        TorchTracepoint scoped_range(TorchTrace::GetTorchTrace().domain());
        scoped_range.enter(AOTOPS, "user_trace_point", trace_info.data());
        auto stream = getCurrentGCUStream();

        // device kernel trace info
        auto status = topsaten::topsatenFlushNan(trace_info.c_str(), stream);
        if (status != TOPSATEN_STATUS_SUCCESS) {
          PTCHECK(0) << "Call topsatenFlushNan fail, got error: " << status;
        }
        return;
      },
      py::arg("trace_info") = "");
  m.def(
      "_tops_host_trace_point",
      [](const std::string &name, const std::string &payload = "") {
        TorchTracepoint scoped_range(TorchTrace::GetTorchTrace().domain());
        if (payload.empty()) {
          scoped_range.enter(OTHERS, name.data());
        } else {
          scoped_range.enter(OTHERS, name.data(), payload.data());
        }
        return;
      },
      py::arg("name") = "", py::arg("payload") = "");
  m.def("_replace_device_new_method", []() { ReplaceDeviceNewMethod(); });
  m.def("_replace_generator_new_method", []() { ReplaceGeneratorNewMethod(); });
  m.def(
      "_compile_trace_start",
      [](const std::string &func_name, const std::string &payload = "") {
        if (payload.empty()) {
          COMPILE_PYTHON_TRACE_START(func_name);
        } else {
          COMPILE_PYTHON_TRACE_START(func_name, payload);
        }
      },
      py::arg("func_name") = "", py::arg("payload") = "");
  m.def(
      "_compile_trace_end",
      [](const std::string &func_name) { COMPILE_PYTHON_TRACE_END(func_name); },
      py::arg("func_name") = "");
  m.def(
      "_user_trace_start",
      [](const std::string &func_name, const std::string &payload = "") {
        if (payload.empty()) {
          OTHER_PYTHON_TRACE_START(func_name);
        } else {
          OTHER_PYTHON_TRACE_START(func_name, payload);
        }
      },
      py::arg("func_name") = "", py::arg("payload") = "");
  m.def(
      "_user_trace_end",
      [](const std::string &func_name) { OTHER_PYTHON_TRACE_END(func_name); },
      py::arg("func_name") = "");
  m.def("_support_kineto_gcu", []() {
#ifdef USE_KINETO_GCU
    return true;
#else
  return false;
#endif
  });
  m.def("_replace_StorageBase_methods", []() { ReplaceStorageBaseMethods(); });
  m.def("_get_gcu_data_ptr", [](const at::Tensor &tensor) -> uintptr_t {
    void *ptr = gcu_data_ptr(tensor);
    return reinterpret_cast<uintptr_t>(ptr);
  });
  m.def("_get_gcu_device_ptr", [](const at::Tensor &tensor) -> uintptr_t {
    void *ptr = gcu_device_ptr(tensor);
    return reinterpret_cast<uintptr_t>(ptr);
  });
}

}  // namespace torch_gcu

void THGPStream_init(PyObject *module);
void THGPEvent_init(PyObject *module);

PyMethodDef *THGPModule_get_methods();

static std::vector<PyMethodDef> methods;

extern "C" TORCH_GCU_API

    PyObject *
    initModule() {
  at::internal::lazy_init_num_threads();

  AddPyMethodDefs(methods, TorchGcuMethods);
  AddPyMethodDefs(methods, THGPModule_get_methods());
// AddPyMethodDefs(methods, torch_gcu::profiler::profiler_functions());
#ifdef USE_C10D_ECCL
  AddPyMethodDefs(methods, torch_gcu::distributed::python_functions());
#endif
  AddPyMethodDefs(methods, torch_gcu::tensors::gcu_tensor_type_functions());
  // AddPyMethodDefs(methods, torch_gcu::autocast::autocast_mode_functions());
  static struct PyModuleDef torchgcu_module = {
      PyModuleDef_HEAD_INIT, "torch_gcu._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&torchgcu_module);

  // This will only initialize base classes and attach them to library namespace
  // They won't be ready for real usage until importing gcu module, that will
  // complete the process (but it defines Python classes before calling back
  // into C, so these lines have to execute first)..
  THGPStream_init(module);
  THGPEvent_init(module);
  THGPGraph_init(module);
  THGPStorage_Sharing_methods(module);

  // torch_gcu::GCUNativeFunctions::InitializeAtenBindings();
  torch_gcu::InitOtherBindings(module);
  torch_gcu::initModule(module);  // gcu_module
  torch_gcu::RegisterFusionOp(module);
  torch_gcu::RegisterEfficientOps(module);
  torch_gcu::RegisterExtraOps(module);
  torch_gcu::RegisterOpDebugFunc(module);
  torch_gcu::RegisterLaunchHostFunc(module);

  torch_gcu::SetPythonCallstackFn(&torch_gcu::GetPythonCallstackStr);
  torch_gcu::SetCppCallstackFn(&torch_gcu::GetCppFrameImpl);
  return module;
}

PyMODINIT_FUNC PyInit__C(void) { return initModule(); }
