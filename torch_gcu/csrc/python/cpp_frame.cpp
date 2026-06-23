#include "python/cpp_frame.h"

#ifdef USE_KINETO_GCU
#include "profiler/combined_traceback.h"
#endif

namespace torch_gcu {
TORCH_GCU_API std::string GetCppFrameImpl(c10::GatheredContext* context) {
#ifdef USE_KINETO_GCU
  if (!context) {
    return "";
  }

  std::ostringstream os;
  auto frames = dynamic_cast<torch_gcu::CapturedTraceback*>(context);
  auto s = symbolize({frames});
  for (auto f : s.all_frames) {
    os << f.filename << ":" << f.lineno << " in " << f.funcname << " "
       << "\n";
  }

  return os.str();
#else
  return "";
#endif
}

}  // namespace torch_gcu