#include "gcu/gcu_exception.h"

#include <c10/util/Exception.h>

class IgnoreWarningHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning& /*warning*/) {
    // pass
  }
};

c10::WarningHandler* getIgnoreHandler() {
  static IgnoreWarningHandler handler_ = IgnoreWarningHandler();
  return &handler_;
};

int warning_ingore(bool ingore) {
  if (ingore) {
    c10::WarningUtils::set_warning_handler(getIgnoreHandler());
    return 1;
  } else {
    c10::WarningUtils::set_warning_handler(nullptr);
    return 0;
  }
}
