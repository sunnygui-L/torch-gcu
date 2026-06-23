#include <tops/tops_runtime_api.h>
#include <torch/csrc/utils/pybind.h>

#include "gcu/gcu_exception.h"
#include "gcu/gcu_guard.h"

namespace torch_gcu::shared {

void initGcurtBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto gcurt = m.def_submodule("_gcurt", "libtopsrt.so bindings");

  py::enum_<topsError_t>(gcurt, "topsError").value("success", topsSuccess);

  gcurt.def("topsGetErrorString", topsGetErrorString);
  // gcurt.def("topsProfilerStart", topsProfilerStart);
  // gcurt.def("topsProfilerStop", topsProfilerStop);
  gcurt.def("topsHostRegister",
            [](uintptr_t ptr, size_t size, unsigned int flags) -> topsError_t {
              return topsHostRegister((void*)ptr, size, flags);
            });
  gcurt.def("topsHostUnregister", [](uintptr_t ptr) -> topsError_t {
    return topsHostUnregister((void*)ptr);
  });
  gcurt.def("topsStreamCreate", [](uintptr_t ptr) -> topsError_t {
    return topsStreamCreate((topsStream_t*)ptr);
  });
  gcurt.def("topsStreamDestroy", [](uintptr_t ptr) -> topsError_t {
    return topsStreamDestroy((topsStream_t)ptr);
  });
  gcurt.def("topsMemGetInfo", [](int device) -> std::pair<size_t, size_t> {
    GCUGuard guard(device);
    size_t device_free = 0;
    size_t device_total = 0;
    C10_GCU_CHECK(topsMemGetInfo(&device_free, &device_total));
    return {device_free, device_total};
  });
}

}  // namespace torch_gcu::shared
