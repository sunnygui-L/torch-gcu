#ifdef USE_C10D_ECCL
#include <ATen/core/functional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <pybind11/chrono.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/tensor_flatten.h>
#include <torch/custom_class.h>
#include <torch_gcu/csrc/python/dist_py_init.h>

#include <mutex>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch_gcu/csrc/distributed/ProcessGroupECCL.hpp>

#include "pybind11/pybind11.h"

namespace {

// Wrapper to ensure GIL is released before destructing ProcessGroupGloo
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_;

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) =
      default;
  IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept { return *impl_; }
  T* operator->() const noexcept { return impl_.get(); }
  C10_NODISCARD T* get() const noexcept { return impl_.get(); }
  void reset() noexcept { impl_.reset(); }
  operator bool() const noexcept { return impl_; }
};

}  // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true);

namespace torch_gcu {
namespace distributed {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>>;

void _c10d_gcu_init(PyObject* /*_unused*/, PyObject* /*noargs*/) {
  auto torch_gcu_C_module = THPObjectPtr(PyImport_ImportModule("torch_gcu._C"));
  if (!torch_gcu_C_module) {
    throw python_error();
  }
  auto torch_gcu_C_m = py::handle(torch_gcu_C_module).cast<py::module>();

  auto m = torch_gcu_C_m.def_submodule("_distributed_c10d",
                                       "distributed c10d bindings");
  auto module = py::handle(m).cast<py::module>();
  py::module_ dist = py::module_::import("torch._C._distributed_c10d");

  auto processGroupECCL =
      intrusive_ptr_no_gil_destructor_class_<::c10d_gcu::ProcessGroupECCL>(
          module, "ProcessGroupECCL", dist.attr("Backend"))
          .def(py::init<
                   const c10::intrusive_ptr<::c10d::Store>&, int, int,
                   c10::intrusive_ptr<::c10d_gcu::ProcessGroupECCL::Options>>(),
               py::call_guard<py::gil_scoped_release>())
          .def(py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                           int rank, int size,
                           const std::chrono::milliseconds& timeout) {
                 auto options = ::c10d_gcu::ProcessGroupECCL::Options::create();
                 options->is_high_priority_stream = false;
                 options->timeout = timeout;
                 return c10::make_intrusive<::c10d_gcu::ProcessGroupECCL>(
                     store, rank, size, options);
               }),
               py::arg("store"), py::arg("rank"), py::arg("size"),
               py::arg("timeout") = kProcessGroupDefaultTimeout,
               py::call_guard<py::gil_scoped_release>())
          .def("get_eccl_comm", &::c10d_gcu::ProcessGroupECCL::getECCLComm)
          .def("_group_start", &::c10d_gcu::ProcessGroupECCL::groupStart)
          .def("_group_end", &::c10d_gcu::ProcessGroupECCL::groupEnd)
          .def("alltoallv_d", &::c10d_gcu::ProcessGroupECCL::alltoallv_d,
               py::arg("outputTensor"), py::arg("inputTensor"),
               py::arg("outputSplitSizes"), py::arg("inputSplitSizes"),
               py::arg("flag"), py::arg("opts") = ::c10d::AllgatherOptions(),
               py::call_guard<py::gil_scoped_release>())
          .def("allreduce_outplace",
               &::c10d_gcu::ProcessGroupECCL::allreduce_outplace,
               py::arg("outputTensor"), py::arg("inputTensor"),
               py::arg("opts") = ::c10d::AllreduceOptions(),
               py::call_guard<py::gil_scoped_release>())
          .def("allgatherv", &::c10d_gcu::ProcessGroupECCL::allgatherv,
               py::arg("outputTensor"), py::arg("inputTensor"),
               py::arg("recvCounts"),
               py::arg("opts") = ::c10d::AllgatherOptions(),
               py::call_guard<py::gil_scoped_release>())
          .def("reduce_scatterv",
               &::c10d_gcu::ProcessGroupECCL::reduce_scatterv,
               py::arg("outputTensor"), py::arg("inputTensor"),
               py::arg("recvCounts"),
               py::arg("opts") = ::c10d::ReduceScatterOptions(),
               py::call_guard<py::gil_scoped_release>())
          .def("get_unique_id", &::c10d_gcu::ProcessGroupECCL::getUniqueID,
               py::call_guard<py::gil_scoped_release>())
          .def_property_readonly("options",
                                 &::c10d_gcu::ProcessGroupECCL::getOptions);

  intrusive_ptr_class_<::c10d_gcu::ProcessGroupECCL::Options>(
      processGroupECCL, "Options", dist.attr("Backend").attr("Options"))
      .def(py::init<>())
      .def_readwrite("op_timeout",
                     &::c10d_gcu::ProcessGroupECCL::Options::timeout);
}

std::once_flag init_flag;
PyObject* c10d_gcu_init(PyObject* _unused, PyObject* noargs) {
  std::call_once(init_flag, _c10d_gcu_init, _unused, noargs);
  Py_RETURN_TRUE;
}

static PyMethodDef methods[] = {  // NOLINT
    {"_c10d_gcu_init", c10d_gcu_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

TORCH_GCU_API PyMethodDef* python_functions() { return methods; }

}  // namespace distributed
}  // namespace torch_gcu

#endif  // USE_C10D_ECCL
