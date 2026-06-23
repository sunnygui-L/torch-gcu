#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/SequenceNumber.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/PythonFallbackKernel.h>
#include <ATen/record_function.h>
#include <c10/core/DeviceType.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/ScalarType.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/input_metadata.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_saved_variable_hooks.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/utils/python_arg_parsing.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_raii.h>
#include <torch/csrc/utils/python_torch_function_mode.h>

#include <set>
#include <unordered_set>
#include <utility>

#include "torch_gcu/csrc/profiler/autograd/profiler_kineto.h"
#include "torch_gcu/csrc/profiler/autograd/profiler_python.h"
#include "torch_gcu/csrc/profiler/kineto_shim.h"

PyObject* THGPAutograd_initExtension(PyObject* _unused, PyObject* unused) {
  using namespace torch_gcu::autograd::profiler;
  using namespace torch_gcu::profiler::impl;

  auto torch_gcu_C_module = THPObjectPtr(PyImport_ImportModule("torch_gcu._C"));
  if (!torch_gcu_C_module) return nullptr;
  auto _C_m = py::handle(torch_gcu_C_module).cast<py::module>();
  auto m = _C_m.def_submodule("_autograd", "autograd bindings");

  py::class_<KinetoEvent>(m, "_KinetoEvent")
      // name of the event
      .def("name", [](const KinetoEvent& e) { return e.name(); })
      // PyTorch thread id of the start callback
      .def("start_thread_id",
           [](const KinetoEvent& e) { return e.startThreadId(); })
      // PyTorch thread id of the end callback
      .def("end_thread_id",
           [](const KinetoEvent& e) { return e.endThreadId(); })
      // for events of scope BACKWARD_FUNCTION - PyTorch thread id
      // of the corresponding forward op
      .def("fwd_thread_id",
           [](const KinetoEvent& e) { return e.fwdThreadId(); })
      // together with fwd_thread_id, used to uniquely identify
      // the forward op
      .def("sequence_nr", [](const KinetoEvent& e) { return e.sequenceNr(); })
      // absolute start time (since unix epoch) in ns
      .def("start_ns", [](const KinetoEvent& e) { return e.startNs(); })
      // absolute end time (since unix epoch) in ns
      .def("end_ns", [](const KinetoEvent& e) { return e.endNs(); })
      // duration in ns
      .def("duration_ns", [](const KinetoEvent& e) { return e.durationNs(); })
      // used for correlation between high-level PyTorch events
      // and low-level device events
      .def("correlation_id",
           [](const KinetoEvent& e) { return e.correlationId(); })
      // shapes of input tensors
      .def("shapes", [](const KinetoEvent& e) { return e.shapes().vec(); })
      .def("dtypes", [](const KinetoEvent& e) { return e.dtypes().vec(); })
      .def("concrete_inputs",
           [](const KinetoEvent& e) {
             std::vector<py::object> as_pyobj;
             std::transform(
                 e.concreteInputs().begin(), e.concreteInputs().end(),
                 std::back_inserter(as_pyobj), [](const c10::IValue& val) {
                   return torch::jit::toPyObject(val);
                 });
             return as_pyobj;
           })
      .def("kwinputs",
           [](const KinetoEvent& e) {
             std::unordered_map<std::string, py::object> inputs;
             for (const auto& [key, value] : e.kwinputs()) {
               inputs[key] = torch::jit::toPyObject(value);
             }
             return inputs;
           })
      // stack traces of the PyTorch CPU events
      .def("stack", [](const KinetoEvent& e) { return e.stack().vec(); })
      // type of the RecordFunction that generated a PyTorch CPU event
      // (op, torchscript function, user label, etc)
      .def("scope", [](const KinetoEvent& e) { return e.scope(); })
      // device number, for CPU - process id
      .def("device_index", [](const KinetoEvent& e) { return e.deviceIndex(); })
      // for CUDA - stream id, for CPU - start thread id
      .def("device_resource_id",
           [](const KinetoEvent& e) { return e.deviceResourceId(); })
      // device type
      .def("device_type", [](const KinetoEvent& e) { return e.deviceType(); })
      // correlation id of a linked event
      .def("linked_correlation_id",
           [](const KinetoEvent& e) { return e.linkedCorrelationId(); })
      // compute flops
      .def("flops", [](const KinetoEvent& e) { return e.flops(); })
      // Whether this is async event or not
      .def("is_async", [](const KinetoEvent& e) { return e.isAsync(); })
      .def("gcu_elapsed_us", &KinetoEvent::gcuElapsedUs)
      .def("privateuse1_elapsed_us", &KinetoEvent::privateuse1ElapsedUs)
      .def("is_user_annotation",
           [](const KinetoEvent& e) {
             return e.activityType() ==
                        (uint8_t)libkineto_gcu::ActivityType::USER_ANNOTATION ||
                    e.activityType() ==
                        (uint8_t)
                            libkineto_gcu::ActivityType::GCU_USER_ANNOTATION;
           })
      .def("nbytes", [](const KinetoEvent& e) { return e.nBytes(); });

  py::class_<ProfilerResult>(m, "_ProfilerResult")
      .def("trace_start_ns", &ProfilerResult::trace_start_ns)
      .def("events", &ProfilerResult::events)
      .def("experimental_event_tree", &ProfilerResult::event_tree)
      .def("save", &ProfilerResult::save);

  m.def("_enable_profiler", &enableProfiler, py::arg("config"),
        py::arg("activities"),
        py::arg("scopes") = std::unordered_set<at::RecordScope>());
  m.def("_disable_profiler", disableProfiler);
  m.def("_prepare_profiler", prepareProfiler,
        py::call_guard<py::gil_scoped_release>());
  m.def("_toggle_collection_dynamic", toggleCollectionDynamic,
        py::call_guard<py::gil_scoped_release>());
  m.def("_add_metadata_json", addMetadataJson);  // Only if `USE_KINETO` is set
  m.def("_kineto_step", profilerStep);           // Only if `USE_KINETO` is set
  m.def("kineto_available",
        []() { return torch_gcu::profiler::kKinetoAvailable; });

  // TODO: remove this
  // NOTICE: These record functions are not torch operators and may not show up
  // in TorchScript tracing, FX transforms, or operator serialization. For these
  // use cases, please use `torch.profiler.record_function`.
  // Creates a new profiling scope using RecordFunction and invokes its starting
  // callbacks.
  m.def("_record_function_with_args_enter", [](const std::string& name,
                                               const py::args& args) {
    using torch::autograd::profiler::PythonRecordFunction;
    auto python_rec =
        c10::make_intrusive<PythonRecordFunction>(at::RecordScope::USER_SCOPE);
    auto* rec = &python_rec->record;
    if (rec->isActive()) {
      if (rec->needsInputs()) {
        auto iv_inputs = std::vector<c10::IValue>();
        for (const auto& arg : args) {
          iv_inputs.push_back(torch::jit::toTypeInferredIValue(arg));
        }
        rec->before(name, c10::ArrayRef<const c10::IValue>(iv_inputs.data(),
                                                           iv_inputs.size()));
      } else {
        rec->before(name);
      }
    }
    return torch::jit::toPyObject(std::move(python_rec));
  });

  // TODO: remove this
  // Ends the profiling scope created with record_function_with_param_enter.
  m.def("_record_function_with_args_exit", [](const py::object& obj) {
    using torch::autograd::profiler::PythonRecordFunction;
    auto python_record = torch::jit::toCustomClass<PythonRecordFunction>(obj);

    // We don't actually need to do anything with handle just need to persist
    // the lifetime until now.
    python_record->record.end();
  });

  // TODO: remove this
  m.def("_supported_activities", []() {
    std::set<torch_gcu::profiler::impl::ActivityType> activities{
        torch_gcu::profiler::impl::ActivityType::CPU};
    if (at::hasMTIA()) {
      activities.insert(torch_gcu::profiler::impl::ActivityType::MTIA);
    }
    if (at::getNumGPUs() > 0) {
      activities.insert(torch_gcu::profiler::impl::ActivityType::CUDA);
    }
    if (at::hasXPU()) {
      activities.insert(torch_gcu::profiler::impl::ActivityType::XPU);
    }
    if (at::hasMTIA()) {
      activities.insert(torch_gcu::profiler::impl::ActivityType::MTIA);
    }
    if (c10::get_privateuse1_backend() != "privateuseone") {
      activities.insert(torch_gcu::profiler::impl::ActivityType::PrivateUse1);
    }
    return activities;
  });

  m.def("_profiler_enabled", profilerEnabled);
  m.def("_profiler_type", torch_gcu::profiler::impl::profilerType);

  torch_gcu::autograd::profiler::python_tracer::init();
  Py_RETURN_TRUE;
}
