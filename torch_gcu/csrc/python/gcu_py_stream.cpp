#include "python/gcu_py_stream.h"

#include <structmember.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_numbers.h>

#include <iostream>

#include "gcu/gcu_functions.h"
#include "gcu/gcu_guard.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"
#include "topsaten/topsaten_define.h"

PyObject* THGPStreamClass = nullptr;

static PyObject* THGPStream_pynew(PyTypeObject* type, PyObject* args,
                                  PyObject* kwargs) {
  HANDLE_TH_ERRORS

  const auto current_device = torch_gcu::current_device();

  int priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;
  uint64_t stream_ptr = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {"priority",    "stream_id",  "device_index",
                                    "device_type", "stream_ptr", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "|iLLLK", const_cast<char**>(kwlist), &priority,
          &stream_id, &device_index, &device_type, &stream_ptr)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  if (stream_ptr) {
    TORCH_CHECK(priority == 0,
                "Priority was explicitly set for a external stream")
  }
  torch_gcu::GCUStream stream =
      (stream_id || device_index || device_type)
          ? torch_gcu::GCUStream::unpack3(
                stream_id, device_index,
                static_cast<c10::DeviceType>(device_type))
      : stream_ptr
          ? torch_gcu::getStreamFromExternal(
                reinterpret_cast<topsStream_t>(stream_ptr), current_device)
          : torch_gcu::getStreamFromPool(priority);

  THGPStream* self = (THGPStream*)ptr.get();
  self->stream_id = static_cast<int64_t>(stream.id());
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  new (&self->gcu_stream) torch_gcu::GCUStream(stream);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THGPStream_dealloc(THGPStream* self) {
  self->gcu_stream.~GCUStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// static PyObject* THGPStream_get_device(THGPStream* self, void* /*unused*/) {
//   HANDLE_TH_ERRORS
//   return THPDevice_New(self->gcu_stream.device());
//   END_HANDLE_TH_ERRORS
// }

static PyObject* THGPStream_get_gcu_stream(THGPStream* self, void* /*unused*/) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->gcu_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPStream_get_priority(THGPStream* self, void* /*unused*/) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->gcu_stream.priority());
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPStream_priority_range(PyObject* /*_unused*/,
                                           PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  auto [least_priority, greatest_priority] =
      torch_gcu::GCUStream::priority_range();
  return Py_BuildValue("(ii)", least_priority, greatest_priority);
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPStream_query(PyObject* _self, PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  auto self = (THGPStream*)_self;
  return PyBool_FromLong(self->gcu_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPStream_set_limit(PyObject* _self, PyObject* args,
                                      PyObject* kwargs) {
  HANDLE_TH_ERRORS
  size_t cluster_num = 0;
  size_t sip_num = 0;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {"cluster_num", "sip_num", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii",
                                   const_cast<char**>(kwlist), &cluster_num,
                                   &sip_num)) {
    Py_RETURN_FALSE;
  }
  auto self = (THGPStream*)_self;
  if (self->gcu_stream.stream() == nullptr) {
    TORCH_WARN(
        "Not support default stream,please export "
        "USE_GCU_DEFAULT_STREAM=false");
    Py_RETURN_FALSE;
  }
  C10_GCU_CHECK(
      topsStreamSetLaunchLimit(self->gcu_stream, cluster_num, sip_num));
  auto status = topsatenInit();
  if (status != TOPSATEN_STATUS_SUCCESS) {
    PTCHECK(0) << "Call topsatenInit fail, got error: " << status;
  }
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPStream_get_limit(PyObject* _self, PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  auto self = (THGPStream*)_self;
  if (self->gcu_stream.stream() == nullptr) {
    TORCH_WARN(
        "Not support default stream,please export "
        "USE_GCU_DEFAULT_STREAM=false");
    Py_RETURN_FALSE;
  }
  size_t cluster_num = 0;
  size_t sip_num = 0;
  auto ret = topsStreamGetLaunchLimit(self->gcu_stream, &cluster_num, &sip_num);
  if (ret != topsSuccess) {
    Py_RETURN_FALSE;
  }
  return Py_BuildValue("(KK)", cluster_num, sip_num);
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPStream_synchronize(PyObject* _self, PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    auto self = (THGPStream*)_self;
    self->gcu_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPStream_eq(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THGPStream*)_self;
  auto other = (THGPStream*)_other;
  return PyBool_FromLong(self->gcu_stream == other->gcu_stream);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyMemberDef THGPStream_members[] = {{nullptr}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyGetSetDef THGPStream_properties[] = {
    {"gcu_stream", (getter)THGPStream_get_gcu_stream, nullptr, nullptr,
     nullptr},
    {"priority", (getter)THGPStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static PyMethodDef THGPStream_methods[] = {
    {(char*)"query", THGPStream_query, METH_NOARGS, nullptr},
    {(char*)"set_limit", castPyCFunctionWithKeywords(THGPStream_set_limit),
     METH_VARARGS | METH_KEYWORDS, nullptr},
    {(char*)"get_limit", THGPStream_get_limit, METH_NOARGS, nullptr},
    {(char*)"synchronize", THGPStream_synchronize, METH_NOARGS, nullptr},
    {(char*)"priority_range", THGPStream_priority_range,
     METH_STATIC | METH_NOARGS, nullptr},
    {(char*)"__eq__", THGPStream_eq, METH_O, nullptr},
    {nullptr}};

// clang-format off
PyTypeObject THGPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch_gcu._C._GcuStreamBase",         /* tp_name */
    sizeof(THGPStream),                       /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)THGPStream_dealloc,           /* tp_dealloc */
    0,                                        /* tp_vectorcall_offset */
    nullptr,                                  /* tp_getattr */
    nullptr,                                  /* tp_setattr */
    nullptr,                                  /* tp_reserved */
    nullptr,                                  /* tp_repr */
    nullptr,                                  /* tp_as_number */
    nullptr,                                  /* tp_as_sequence */
    nullptr,                                  /* tp_as_mapping */
    nullptr,                                  /* tp_hash  */
    nullptr,                                  /* tp_call */
    nullptr,                                  /* tp_str */
    nullptr,                                  /* tp_getattro */
    nullptr,                                  /* tp_setattro */
    nullptr,                                  /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr,                                  /* tp_doc */
    nullptr,                                  /* tp_traverse */
    nullptr,                                  /* tp_clear */
    nullptr,                                  /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    nullptr,                                  /* tp_iter */
    nullptr,                                  /* tp_iternext */
    THGPStream_methods,                       /* tp_methods */
    THGPStream_members,                       /* tp_members */
    THGPStream_properties,                    /* tp_getset */
    nullptr,                                  /* tp_base */
    nullptr,                                  /* tp_dict */
    nullptr,                                  /* tp_descr_get */
    nullptr,                                  /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    nullptr,                                  /* tp_init */
    nullptr,                                  /* tp_alloc */
    THGPStream_pynew,                         /* tp_new */
};
// clang-format on

void THGPStream_init(PyObject* module) {
  Py_INCREF(THPStreamClass);
  THGPStreamType.tp_base = THPStreamClass;
  THGPStreamClass = (PyObject*)&THGPStreamType;
  if (PyType_Ready(&THGPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THGPStreamType);
  if (PyModule_AddObject(module, "_GcuStreamBase", (PyObject*)&THGPStreamType) <
      0) {
    throw python_error();
  }
}
