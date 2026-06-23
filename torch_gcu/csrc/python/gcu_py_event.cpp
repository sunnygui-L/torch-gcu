#include "python/gcu_py_event.h"

#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "gcu/gcu_functions.h"
#include "gcu/gcu_guard.h"
#include "python/gcu_py_stream.h"

PyObject* THGPEventClass = nullptr;

static PyObject* THGPEvent_pynew(PyTypeObject* type, PyObject* args,
                                 PyObject* kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {"enable_timing", "blocking", "interprocess",
                                    nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|bbb",
                                   const_cast<char**>(kwlist), &enable_timing,
                                   &blocking, &interprocess)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THGPEvent* self = (THGPEvent*)ptr.get();
  unsigned int flags =
      (blocking ? topsEventBlockingSync : topsEventDefault) |
      (enable_timing ? topsEventDefault : topsEventDisableTiming) |
      (interprocess ? topsEventInterprocess : topsEventDefault);

  new (&self->gcu_event) torch_gcu::GCUEvent(flags);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPEvent_from_ipc_handle(PyObject* _type, PyObject* args,
                                           PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto type = (PyTypeObject*)_type;

  static torch::PythonArgParser parser({
      "from_ipc_handle(Device device, std::string ipc_handle)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Device device = r.device(0);
  std::string handle_string = r.string(1);

  TORCH_CHECK(handle_string.size() == sizeof(topsIpcEventHandle_t),
              "topsIpcEventHandle_t expects byte-like object of size ",
              sizeof(topsIpcEventHandle_t), ", but got ", handle_string.size());
  TORCH_CHECK(device.type() == at::kPrivateUse1,
              "Event can only be created on "
              "GCU devices, but got device type ",
              device.type())

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }
  THGPEvent* self = (THGPEvent*)ptr.get();

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  topsIpcEventHandle_t handle;
  std::memcpy(&handle, handle_string.c_str(), handle_string.size());
  new (&self->gcu_event) torch_gcu::GCUEvent(device.index(), &handle);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THGPEvent_dealloc(THGPEvent* self) {
  self->gcu_event.~GCUEvent();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THGPEvent_get_gcu_event(THGPEvent* self, void* /*unused*/) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->gcu_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPEvent_get_device(THGPEvent* self, void* /*unused*/) {
  HANDLE_TH_ERRORS
  at::optional<at::Device> device = self->gcu_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPEvent_record(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS
  auto self = (THGPEvent*)_self;
  auto stream = (THGPStream*)_stream;
  self->gcu_event.record(stream->gcu_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPEvent_wait(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS {
    auto self = (THGPEvent*)_self;
    auto stream = (THGPStream*)_stream;
    pybind11::gil_scoped_release no_gil{};
    self->gcu_event.block(stream->gcu_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPEvent_query(PyObject* _self, PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  auto self = (THGPEvent*)_self;
  return PyBool_FromLong(self->gcu_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPEvent_elapsed_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THGPEvent*)_self;
  auto other = (THGPEvent*)_other;
  return PyFloat_FromDouble(self->gcu_event.elapsed_time(other->gcu_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPEvent_synchronize(PyObject* _self, PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS {
    auto self = (THGPEvent*)_self;
    pybind11::gil_scoped_release no_gil{};
    self->gcu_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THGPEvent_ipc_handle(PyObject* _self, PyObject* /*noargs*/) {
  HANDLE_TH_ERRORS
  auto self = (THGPEvent*)_self;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  topsIpcEventHandle_t handle;
  self->gcu_event.ipc_handle(&handle);
  return PyBytes_FromStringAndSize((const char*)&handle, sizeof(handle));
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables, modernize-avoid-c-arrays)
static struct PyGetSetDef THGPEvent_properties[] = {
    {"device", (getter)THGPEvent_get_device, nullptr, nullptr, nullptr},
    {"gcu_event", (getter)THGPEvent_get_gcu_event, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables, modernize-avoid-c-arrays)
static PyMethodDef THGPEvent_methods[] = {
    {(char*)"from_ipc_handle",
     castPyCFunctionWithKeywords(THGPEvent_from_ipc_handle),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS, nullptr},
    {(char*)"record", THGPEvent_record, METH_O, nullptr},
    {(char*)"wait", THGPEvent_wait, METH_O, nullptr},
    {(char*)"query", THGPEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", THGPEvent_elapsed_time, METH_O, nullptr},
    {(char*)"synchronize", THGPEvent_synchronize, METH_NOARGS, nullptr},
    {(char*)"ipc_handle", THGPEvent_ipc_handle, METH_NOARGS, nullptr},
    {nullptr}};

PyTypeObject THGPEventType = {
    PyVarObject_HEAD_INIT(nullptr,
                          0) "torch_gcu._C._GcuEventBase", /* tp_name */
    sizeof(THGPEvent),                                     /* tp_basicsize */
    0,                                                     /* tp_itemsize */
    (destructor)THGPEvent_dealloc,                         /* tp_dealloc */
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
    THGPEvent_methods,                        /* tp_methods */
    nullptr,                                  /* tp_members */
    THGPEvent_properties,                     /* tp_getset */
    nullptr,                                  /* tp_base */
    nullptr,                                  /* tp_dict */
    nullptr,                                  /* tp_descr_get */
    nullptr,                                  /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    nullptr,                                  /* tp_init */
    nullptr,                                  /* tp_alloc */
    THGPEvent_pynew,                          /* tp_new */
};

void THGPEvent_init(PyObject* module) {
  THGPEventClass = (PyObject*)&THGPEventType;
  if (PyType_Ready(&THGPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THGPEventType);
  if (PyModule_AddObject(module, "_GcuEventBase", (PyObject*)&THGPEventType) <
      0) {
    throw python_error();
  }
}
