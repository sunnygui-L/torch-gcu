#include <torch/csrc/utils/pybind.h>

#include "topstx/topstx.h"

namespace torch_gcu::shared {

void initTopstxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto topstx = m.def_submodule("_topstx", "libtopstx.so bindings");
  topstx.def("rangePushA", [](std::string msg) {
    topstxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = TOPSTX_VERSION;
    eventAttrib.size = TOPSTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = TOPSTX_MESSAGE_TYPE_STRING;
    eventAttrib.message.str = msg.data();
    topstxRangePush(&eventAttrib);
  });
  topstx.def("rangePop", topstxRangePop);
  topstx.def("rangeStartA", [](std::string msg) -> int {
    topstxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = TOPSTX_VERSION;
    eventAttrib.size = TOPSTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = TOPSTX_MESSAGE_TYPE_STRING;
    eventAttrib.message.str = msg.data();
    return topstxRangeStart(&eventAttrib);
  });
  topstx.def("rangeEnd", topstxRangeEnd);
  topstx.def("markA", [](std::string msg) {
    topstxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = TOPSTX_VERSION;
    eventAttrib.size = TOPSTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = TOPSTX_MESSAGE_TYPE_STRING;
    eventAttrib.message.str = msg.data();
    topstxMark(&eventAttrib);
  });
}

}  // namespace torch_gcu::shared
