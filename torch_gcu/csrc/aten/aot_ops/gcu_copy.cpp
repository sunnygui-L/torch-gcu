#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorIterator.h>
#include <c10/core/DeviceGuard.h>
#include <topsaten/topsaten_ops.h>
#include <torch/library.h>

#include <limits>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/gcu_resize.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/shape_infer_func.h"
#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_caching_host_allocator.h"
#include "gcu/gcu_event.h"
#include "gcu/gcu_guard.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_peer_to_peer_access.h"
#include "gcu/gcu_stream.h"
#include "gcu/gcu_utils.h"
#include "gcu/runtime_wrapper.h"

namespace torch_gcu {

namespace aotops {

namespace {

// device-to-device copy, does type conversion
void copy_device_to_device(at::TensorIterator& iter, bool non_blocking,
                           bool p2p_enabled) {
  int64_t numel = iter.numel();

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
  bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();
  bool memcpy_eligible =
      same_type && same_conj && same_neg && iter.is_contiguous();
  if (!memcpy_eligible) {
    PTDLOG(TORCH_GCU) << "memcpy_eligible false";
    PTDLOG(TORCH_GCU) << "same_type: " << same_type
                      << ", same_conj: " << same_conj
                      << ", same_neg: " << same_neg;
    auto out_n = iter.noutputs();
    for (int i = 0; i < out_n; ++i) {
      PTDLOG(TORCH_GCU) << "output tensor " << i
                        << " is_contiguous: " << iter.output(i).is_contiguous();
    }
    auto in_n = iter.ninputs();
    for (int i = 0; i < in_n; ++i) {
      PTDLOG(TORCH_GCU) << "input tensor " << i
                        << " is_contiguous: " << iter.input(i).is_contiguous();
    }
  }

  at::Device dst_device = iter.device(0);
  at::Device src_device = iter.device(1);

  // See NOTE [workaround torch.as_tensor]
  if (src_device.is_cpu()) {
    src_device = dst_device;
  }

  GCUGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy. We have to explicitly do this
  // for non-contig copies. This mimics the behavior of cross-device
  // topsMemcpyAsync on the default stream.
  GCUStream copy_stream = getCurrentGCUStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    // GCUEvent dst_ready;
    // device_guard.set_device(dst_device);
    // dst_ready.record(getCurrentGCUStream(dst_device.index()));

    // device_guard.set_device(src_device);
    // dst_ready.block(copy_stream);
    device_guard.set_device(src_device);
    topsEvent_t event_src{nullptr};
    C10_GCU_CHECK(topsEventCreateWithFlags(
        &event_src, topsEventInterprocess | topsEventStrongOrder |
                        topsEventDisableQueryAndSync));
    device_guard.set_device(dst_device);
    topsEvent_t event_dst{nullptr};
    C10_GCU_CHECK(topsOpenEventHandle(&event_dst, event_src));
    auto dst_stream = getCurrentGCUStream(dst_device.index());
    C10_GCU_CHECK(topsEventRecord(event_dst, dst_stream));

    device_guard.set_device(src_device);
    C10_GCU_CHECK(topsStreamWaitEvent(copy_stream, event_src, 0));
    C10_GCU_CHECK(topsEventDestroy(event_src));
    C10_GCU_CHECK(topsEventDestroy(event_dst));
  }

  auto call_topsaten = [&]() {
    auto dst = iter.tensor(0);
    auto src = iter.tensor(1);
    auto xsrc = createTopsatenTensor(src);
    auto xdst = createTopsatenTensor(dst);
    auto op_info = [&]() -> std::string {
      std::stringstream ss;
      // clang-format off
      ss << "topsatenCopy: {\n"
          << tensorArgsToString({src}, {dst})
          << "non_blocking: " << non_blocking << "\n"
          << "copy_stream: " << copy_stream << "\n"
          << "}\n";
      // clang-format on
      return ss.str();
    };
    PTDLOG(OP) << op_info();
    // for d2d non_blocking has no effect and topsatenCopy
    // not support non_blocking is true.
    CHECK_TOPSATEN_CALL(topsaten::topsatenCopy(xdst, xsrc, false, copy_stream),
                        op_info);
    maybeGCUStreamSynchronize(copy_stream);
  };

  if (memcpy_eligible) {
    void* dst = gcu_data_ptr(iter.tensor(0));
    void* src = gcu_data_ptr(iter.tensor(1));
    size_t size = numel * at::elementSize(get_gcu_scalar_type(iter.dtype(0)));
    if (src_device != dst_device) {
      // Due to bizarre gcu driver intricacies, copies of
      // topsMallocAsynced memory between devices that aren't
      // peer-to-peer-capable need "topsMemcpyPeerAsync".
      // So we let the allocator implement the correct call
      // (either topsMemcpyAsync or topsMemcpyPeerAsync)
      C10_GCU_CHECK(GCUCachingAllocator::memcpyAsync(
          dst, dst_device.index(), src, src_device.index(), size, copy_stream,
          p2p_enabled));
    } else {
      if (src != dst) {
        // Use the topsatenCopy operator, instead of runtime memcpy, in
        // exchange for higher performance
        call_topsaten();
      }
    }
  } else {
    if (same_neg) {
      if (!same_conj) {
        PTCHECK(false) << "GCU not support conj_kernel yet";
      } else {
        call_topsaten();
      }
    } else {
      if (!same_conj) {
        PTCHECK(false) << "GCU not support neg_conj_kernel yet";
      } else {
        PTCHECK(false) << "GCU not support neg_kernel yet";
      }
    }
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on src_device, record stream event
    // GCUEvent src_ready;
    // src_ready.record(copy_stream);
    // device_guard.set_device(dst_device);
    // src_ready.block(getCurrentGCUStream(dst_device.index()));

    device_guard.set_device(dst_device);
    topsEvent_t event_dst{nullptr};
    C10_GCU_CHECK(topsEventCreateWithFlags(
        &event_dst, topsEventInterprocess | topsEventStrongOrder |
                        topsEventDisableQueryAndSync));

    device_guard.set_device(src_device);
    topsEvent_t event_src{nullptr};
    C10_GCU_CHECK(topsOpenEventHandle(&event_src, event_dst));
    C10_GCU_CHECK(topsEventRecord(event_src, copy_stream));

    device_guard.set_device(dst_device);
    auto dst_stream = getCurrentGCUStream(dst_device.index());
    C10_GCU_CHECK(topsStreamWaitEvent(dst_stream, event_dst, 0));
    C10_GCU_CHECK(topsEventDestroy(event_dst));
    C10_GCU_CHECK(topsEventDestroy(event_src));
    device_guard.set_device(src_device);
  }

  C10_GCU_CHECK(topsGetLastError());
}

static bool copy_requires_temporaries(at::TensorIterator& iter,
                                      bool p2p_enabled) {
  at::Device dst_device = iter.device(0);
  at::Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GCU.
    TORCH_INTERNAL_ASSERT(dst_device.is_privateuseone() &&
                          src_device.is_privateuseone());
    return false;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use topsMemcpyAsync
    return false;
  } else if (dst_device.is_privateuseone() && src_device.is_privateuseone()) {
    // Copies between GCUs can use the copy kernel if P2P is supported
    return !p2p_enabled;
  } else {
    // The remaining cases require temporaries. For example, this includes
    // non-contiguous copies between CPU and GCU.
    return true;
  }
}

static bool maybe_enable_p2p_access(at::Device dst_device,
                                    at::Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }
  return get_p2p_access(src_device.index(), dst_device.index());
}

static void copy_kernel(at::TensorIterator& iter, bool non_blocking) {
  TORCH_CHECK(iter.ntensors() == 2);

  at::Device dst_device = iter.device(0);
  at::Device src_device = iter.device(1);

  // Enable p2p access between devices. (No-op if it involves the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  if (copy_requires_temporaries(iter, p2p_enabled)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    at::Tensor dst_contig;
    at::Tensor src_contig;

    // If non_blocking is true - type conversions are performed on the GCU
    // for CPU-GCU copies, otherwise type conversions are performed on the CPU.
    // Type conversions are performed on the src device for GCU-GCU copies.
    if (iter.device_type(0) == at::kPrivateUse1 || non_blocking) {
      dst_contig = dst.is_contiguous()
                       ? dst
                       : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type)
                       ? dst
                       : at::empty_like(dst, iter.dtype(1),
                                        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // propagate the correct conjugate bit
    dst_contig._set_conj(dst.is_conj());
    src_contig._set_conj(iter.tensor(1).is_conj());

    dst_contig._set_neg(dst.is_neg());
    src_contig._set_neg(iter.tensor(1).is_neg());

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }
  // NOTE: [workaround torch.as_tensor]
  // torch.as_tensor cannot specify src device,
  // If the src is on the device will incorrect triggering of H2D,
  // So we corrected it and used D2D.
  void* dst = gcu_data_ptr(iter.tensor(0));
  void* src = gcu_data_ptr(iter.tensor(1));
  bool src_is_gcu = false;
  if (src_device.is_cpu()) {
    uint32_t data = -1;
    auto ret =
        topsPointerGetAttribute(&data, TOPS_POINTER_ATTRIBUTE_MEMORY_TYPE, src);
    if (ret == topsSuccess && data == topsMemoryTypeDevice) {
      // for torch.as_tensor
      src_is_gcu = true;
      PTDLOG(OP)
          << "The H2D above is actually D2D without stream sync, because "
             "torch.as_sensor cannot specify "
             "src device";
    }
  }
  // Copy on GCU (or between GCUs)
  if (dst_device.is_privateuseone() &&
      (src_device.is_privateuseone() || src_is_gcu)) {
    copy_device_to_device(iter, non_blocking, p2p_enabled);
    return;
  }

  // Copy between CPU and GCU, with same dtype && contiguous
  OptionalGCUGuard device_guard;
  topsMemcpyKind kind;
  at::Tensor host_tensor;
  if (dst_device.is_privateuseone() && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = topsMemcpyHostToDevice;
    host_tensor = iter.tensor(1);
  } else if (dst_device.is_cpu() && src_device.is_privateuseone()) {
    device_guard.set_device(src_device);
    kind = topsMemcpyDeviceToHost;
    host_tensor = iter.tensor(0);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GCU copy_()");
  }

  // for narrow dtype
  auto dtype = iter.dtype(0);
  bool is_narrow = is_narrow_type(dtype);
  if (is_narrow) {
    warn_type_narrow(dtype);
    bool host_is_pinned = host_tensor.is_pinned();
    auto converted_type = get_gcu_scalar_type(dtype);
    at::Tensor converted_tensor;
    if (non_blocking && host_is_pinned) {
      converted_tensor = at::empty_like(
          host_tensor,
          at::TensorOptions().dtype(converted_type).pinned_memory(true));
      converted_tensor.copy_(host_tensor);
    } else {
      converted_tensor = host_tensor.to(converted_type);
    }
    host_tensor = converted_tensor;
    if (src_device.is_cpu()) {
      src = host_tensor.data_ptr();
    } else {
      dst = host_tensor.data_ptr();
    }
  }
  int64_t nbytes = iter.numel() * host_tensor.element_size();
  GCUStream stream = getCurrentGCUStream();

  // NOTE: only use async copy when the source is a CPU tensor and non_blocking
  // is true
  if (non_blocking && (!is_narrow || src_device.is_cpu())) {
    C10_GCU_CHECK(topsMemcpyAsync(dst, src, nbytes, kind, stream));
    // we use both the storage context and the tensor data pointer as the key
    // for the caching host allocator. This allows us to better attribute the
    // events to the original tensor allocation correctly. The cases we seek to
    // handle are:

    // 1: a user can pass a pinned memory tensor with an alternative
    // context, for example if allocating memory directly from the pinned memory
    // allocator and constructing a tensor with torch::from_blob.

    // 2: a user can pass a tensor with a different base pointer to the original
    // allocation (via slicing).
    auto* ptr = (dst_device == at::kCPU ? dst : src);
    auto* ctx = host_tensor.storage().data_ptr().get_context();
    // NOTE: non pinned memory with non blocking copy, topsMemcpyAsync not
    // support, do stream synchronize.
    bool recorded = CachingHostAllocator_recordEvent(ptr, ctx, stream);
    if (!recorded) {
      // Only S60 needs stream synchronization for non-pinned memory
      if (HardwareType::GetInstance().getHardware() == BackendType::kS60) {
        TORCH_WARN_ONCE(
            "non blocking copy with non-pinned host memory, trigger a stream "
            "synchronization.");
        StreamSynchronize(stream);
      }
    }
  } else {
    memcpy_and_sync(dst, src, nbytes, kind, stream);
  }

  if (is_narrow && kind == topsMemcpyDeviceToHost) {
    iter.tensor(0).copy_(host_tensor);
  }

  if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
    iter.tensor(0).conj_physical_();
  }
  if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
    iter.tensor(0).neg_();
  }
}

}  // namespace

at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst,
                      bool non_blocking) {
  // clang-format off
  PTDLOG(OP) << "_copy_from" << ": {\n"
             << tensorArgsToString({self}, {dst})
             << "non_blocking: " << non_blocking << "\n"
             << "}\n";
  // clang-format on

  // Exit early if self and dst are views of the same data
  const bool is_same_data =
      (dst.is_alias_of(self) && dst.storage_offset() == self.storage_offset() &&
       dst.strides().equals(self.strides()) &&
       dst.sizes().equals(self.sizes()) &&
       dst.scalar_type() == self.scalar_type() &&
       dst.is_conj() == self.is_conj() && dst.is_neg() == self.is_neg());
  if (is_same_data) {
    return dst;
  }

  auto iter = at::TensorIteratorConfig()
                  .add_output(dst)
                  .add_input(self)
                  .resize_outputs(false)
                  .check_all_same_dtype(false)
                  .check_all_same_device(false)
                  .build();

  if (iter.numel() == 0) {
    return dst;
  }

  if (!dst.is_complex() && self.is_complex()) {
    TORCH_WARN_ONCE(
        "Casting complex values to real discards the imaginary part");
  }

  copy_kernel(iter, non_blocking);

  return dst;
}

// for cpu fallback(h2d) op or xxxxx.out op
at::Tensor _copy_from_and_resize(const at::Tensor& self,
                                 const at::Tensor& dst) {
  // clang-format off
  PTDLOG(OP) << "_copy_from_and_resize" << ": {\n"
             << tensorArgsToString({self}, {dst})
             << "}\n";
  // clang-format on

  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(dst.defined(), "dst is undefined");

  aotops::_resize_output_(dst, self.sizes(), dst.device());

  if (self.numel() == 0 || dst.is_same(self)) {
    return dst;
  }

  aotops::_copy_from(self, dst, false);

  return dst;
}

at::Tensor& arange_out(const at::Scalar& start, const at::Scalar& end,
                       const at::Scalar& step, at::Tensor& result) {
  arange_out_shape_infer(start, end, step, result);
  auto dtype = scalarTypeToTopsatenDataType(result.scalar_type());
  auto layout = topsatenLayoutType_t::TOPSATEN_LAYOUT_STRIDED;
  auto pinMemory = false;
  bridge_topsatenArange_out1(result, start, end, step, dtype, layout,
                             pinMemory);
  return result;
}

}  // namespace aotops

}  // namespace torch_gcu
