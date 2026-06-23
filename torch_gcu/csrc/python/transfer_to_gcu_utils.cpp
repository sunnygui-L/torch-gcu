#include "python/transfer_to_gcu_utils.h"

#include <ATen/core/GeneratorForPrivateuseone.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <structmember.h>
#include <tops/tops_runtime_api.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

#include "gcu/gcu_generator_impl.h"

// we will call THPPointer<THPGenerator>::free()
// reference to torch2.10.0 torch/csrc/utils.cpp add below code
template <>
void THPPointer<THPGenerator>::free() {
  if (ptr) Py_DECREF(ptr);
}

template class THPPointer<THPGenerator>;

namespace torch_gcu {

// Combines self and args into one tuple.
static auto combine_self_args(PyObject* self, PyObject* args) -> py::tuple {
  if (args == nullptr) {
    return py::make_tuple(py::handle(self));
  } else if (self == nullptr) {
    return py::reinterpret_borrow<py::tuple>(args);
  }

  auto py_args = py::reinterpret_borrow<py::tuple>(args);
  size_t n = py_args.size();
  auto args_ = py::tuple(n + 1);
  args_[0] = py::handle(self);
  for (const auto i : c10::irange(n)) {
    args_[i + 1] = py_args[i];
  }
  return args_;
}

auto handle_torch_function(
    torch::PythonArgs& r, PyObject* self, PyObject* args, PyObject* kwargs,
    PyObject* torch_api, const char* module_name,
    const char* func_name_override = nullptr) -> PyObject* {
  py::object torch_api_function = PyObject_FastGetAttrString(
      torch_api, (char*)(func_name_override ? func_name_override
                                            : r.get_func_name().c_str()));
  TORCH_INTERNAL_ASSERT(torch_api_function.ptr() != nullptr,
                        "torch API function must exist");
  py::tuple args_ = combine_self_args(self, args);
  return torch::handle_torch_function_no_python_arg_parser(
      r.overloaded_args, args_.ptr(), kwargs, r.get_func_name().c_str(),
      torch_api_function.ptr(), module_name);
}

PyObject* THGPDevice_pynew(PyTypeObject* type, PyObject* args,
                           PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"device(Device device)",
       "device(c10::string_view type, int64_t? index=-1)"});
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    PyObject* THPUpperModuleOfDevice =
        THPObjectPtr(PyImport_ImportModule("torch._C"));

    return ::torch_gcu::handle_torch_function(r, nullptr, args, kwargs,
                                              THPUpperModuleOfDevice, "torch");
  }
  if (r.idx == 0) {
    auto device = r.device(0);
    // NOTE: transfer cuda to gcu
    if (device.is_cuda()) {
      device = at::Device(at::DeviceType::PrivateUse1, device.index());
    }
    return THPDevice_New(device);
  } else if (r.idx == 1) {
    auto as_device =
        r.device(0);  // this works, because device can take strings
    if (as_device.has_index()) {
      auto device_type = r.string(0);
      throw std::runtime_error(
          "type (string) must not include an index because index "
          "was passed explicitly: " +
          device_type);
    }
    int64_t device_index = -1;
    if (!r.isNone(1)) {
      device_index = r.toInt64(1);
      // -1 is allowed in ATen/C++, to mean the default device, but not in
      // Python.
      TORCH_CHECK(device_index >= 0, "Device index must not be negative");
    }
    // NOTE: transfer cuda to gcu
    auto device_type = as_device.type();
    if (device_type == at::DeviceType::CUDA) {
      device_type = at::DeviceType::PrivateUse1;
    }
    at::Device device(device_type, static_cast<c10::DeviceIndex>(device_index));
    return THPDevice_New(device);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THGPGenerator_pynew(PyTypeObject* type, PyObject* args,
                              PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({"Generator(Device device=None)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.deviceWithDefault(0, at::Device(at::kCPU));

  THPGeneratorPtr self((THPGenerator*)type->tp_alloc(type, 0));

  c10::DeviceType device_type = device.type();
  if (device_type == at::kCPU) {
    self->cdata = at::make_generator<at::CPUGeneratorImpl>();
  } else if (device.type() == at::kCUDA) {
    self->cdata = at::globalContext()
                      .getAcceleratorHooksInterface(at::kPrivateUse1)
                      .getNewGenerator(device.index());
  } else {
    self->cdata = at::globalContext()
                      .getAcceleratorHooksInterface(device_type)
                      .getNewGenerator(device.index());
  }
  return (PyObject*)self.release();
  END_HANDLE_TH_ERRORS
}

}  // namespace torch_gcu