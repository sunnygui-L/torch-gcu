#include "aten/aot_ops/gcu_ops.h"
#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

namespace aotops {

void record_stream(at::Tensor& self, c10::Stream stream) {
  // clang-format off
  PTDLOG(OP) << "record_stream" << ": {\n"
             << tensorArgsToString({self}, {})
             << "stream: " << stream << "\n"
             << "}\n";
  // clang-format on

  struct c10::StreamData3 data = stream.pack3();
  GCUCachingAllocator::recordStream(
      self.storage().data_ptr(),
      GCUStream::unpack3(data.stream_id, data.device_index, data.device_type));
}

}  // namespace aotops

}  // namespace torch_gcu
