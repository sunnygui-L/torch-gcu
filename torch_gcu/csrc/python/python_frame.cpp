/*
 * Copyright 2021-2023 Enflame. All Rights Reserved.
 */
#include "python/python_frame.h"

#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include <sstream>

#include "csrc/gcu/python_frame_info.h"
#include "python/pythoncapi_compat.h"
namespace torch_gcu {

// NOTE:torch/csrc/utils/python_strings.h
// Unpacks PyBytes (PyString) or PyUnicode as std::string
// PyBytes are unpacked as-is. PyUnicode is unpacked as UTF-8.
// NOTE: this method requires the GIL
inline std::string THPUtils_unpackString(PyObject* obj) {
  if (PyBytes_Check(obj)) {
    size_t size = PyBytes_GET_SIZE(obj);
    return std::string(PyBytes_AS_STRING(obj), size);
  }
  if (PyUnicode_Check(obj)) {
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
    if (!data) {
      throw std::runtime_error("error unpacking string as utf-8");
    }
    return std::string(data, (size_t)size);
  }
  throw std::runtime_error("unpackString: expected bytes or unicode object");
}

// Python interpreter retrieval routine adapted from
// https://stackoverflow.com/a/8706144
std::vector<torch::jit::StackEntry> _pythonCallstack() {
  pybind11::gil_scoped_acquire gil;
  PyFrameObject* frame = PyEval_GetFrame();
  Py_XINCREF(frame);
  std::vector<torch::jit::StackEntry> entries;

  while (nullptr != frame) {
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    size_t line = PyCode_Addr2Line(code.get(), PyFrame_GetLasti(frame));
    std::string filename = THPUtils_unpackString(code->co_filename);
    std::string funcname = THPUtils_unpackString(code->co_name);
    auto source =
        std::make_shared<torch::jit::Source>(funcname, filename, line);
    entries.emplace_back(torch::jit::StackEntry{
        funcname, torch::jit::SourceRange(source, 0, funcname.size())});
    auto new_frame = PyFrame_GetBack(frame);
    Py_DECREF(frame);
    frame = new_frame;
  }
  return entries;
}

std::string GetPythonCallstackStr() {
  auto stacks = _pythonCallstack();
  std::stringstream ss;
  ss << "\nTraceback (most recent call last):\n";
  for (auto& stack : stacks) {
    auto info = stack.range.file_line_col();
    if (info) {
      auto filename = std::get<0>(*info);
      auto line = std::get<1>(*info);
      auto funcname = stack.filename;
      ss << "  File \"" << filename << "\", line " << line << ", in "
         << funcname << "\n";
    } else {
      ss << "  File \"<unknown>\", line <unknown>, in <unknown>\n";
    }
  }
  return ss.str();
}

}  // namespace torch_gcu
