#include "python/gcu_py_graph.h"

#include <torch/csrc/Generator.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include "gcu/gcu_graph.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THGPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but pytorch CI linter and some builds prefer "module".
  auto torch_gcu_C_m = py::handle(module).cast<py::module>();

  torch_gcu_C_m.def("_graph_pool_handle", &torch_gcu::graph_pool_handle);

  shared_ptr_class_<torch_gcu::GCUGraph>(torch_gcu_C_m, "_GCUGraph")
      .def(py::init<>())
      .def(
          "capture_begin",
          [](torch_gcu::GCUGraph& self,
             c10::optional<torch_gcu::MempoolId_t> pool_opt,
             std::string capture_error_mode) {
            topsStreamCaptureMode capture_mode;
            torch_gcu::MempoolId_t pool = pool_opt.has_value()
                                              ? pool_opt.value()
                                              : torch_gcu::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = topsStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = topsStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = topsStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(false,
                          "Unknown capture error mode. Expected `global`, "
                          "`thread_local`, or `relaxed`, got ",
                          capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"), py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def("capture_end", torch::wrap_pybind_function_no_gil(
                              &torch_gcu::GCUGraph::capture_end))
      .def("replay",
           torch::wrap_pybind_function_no_gil(&torch_gcu::GCUGraph::replay))
      .def("reset",
           torch::wrap_pybind_function_no_gil(&torch_gcu::GCUGraph::reset))
      .def("pool",
           torch::wrap_pybind_function_no_gil(&torch_gcu::GCUGraph::pool))
      .def("debug_dump", torch::wrap_pybind_function_no_gil(
                             &::torch_gcu::GCUGraph::debug_dump))
      .def("enable_debug_mode", torch::wrap_pybind_function_no_gil(
                                    &::torch_gcu::GCUGraph::enable_debug_mode))
      .def("debug_dump",
           torch::wrap_pybind_function_no_gil(
               &::torch_gcu::GCUGraph::debug_dump),
           py::arg("debug_path"))
      .def(
          "register_generator_state",
          [](torch_gcu::GCUGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // GIL released after unwrapping the Python object.
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"));
}