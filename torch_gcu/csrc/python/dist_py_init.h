#pragma once

#include <torch/csrc/python_headers.h>

namespace torch_gcu {
namespace distributed {

PyMethodDef* python_functions();

}  // namespace distributed
}  // namespace torch_gcu