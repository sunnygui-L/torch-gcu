import torch
import torch_gcu
import multiprocessing
from multiprocessing.reduction import ForkingPickler
from torch.multiprocessing.reductions import (
    shared_cache,
    rebuild_storage_filename,
    rebuild_storage_empty,
    rebuild_storage_fd,
    StorageWeakRef,
    fd_id,
    rebuild_cuda_tensor,
    rebuild_meta_tensor,
    rebuild_tensor,
    storage_from_cache,
)
from torch._namedtensor_internals import check_serializing_named_tensor


def _reduce_storage(storage):
    from torch.multiprocessing import get_sharing_strategy

    if storage.is_cuda:
        raise RuntimeError(
            "Cannot pickle CUDA storage; try pickling a CUDA tensor instead")
    elif storage.is_gcu:
        raise RuntimeError(
            "Cannot pickle GCU storage; try pickling a GCU tensor instead")
    elif storage.device.type == "meta":
        raise RuntimeError(
            "Cannot pickle meta storage; try pickling a meta tensor instead")
    elif get_sharing_strategy() == "file_system":
        metadata = storage._share_filename_cpu_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        if isinstance(storage, torch.TypedStorage):
            metadata += (storage.dtype, )
        storage._shared_incref()
    elif storage.size() == 0:
        # This is special cased because Empty tensors
        # (with size 0) cannot be mmapped.
        return (rebuild_storage_empty, (type(storage), ))
    else:
        fd, size = storage._share_fd_cpu_()
        df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd  # type: ignore[assignment]

    shared_cache[cache_key] = StorageWeakRef(storage)
    return (rebuild, (type(storage), ) + metadata)


def rebuild_meta_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    dtype,
    storage_size_bytes,
    requires_grad,
):
    untyped_storage = torch.UntypedStorage(storage_size_bytes, device="meta")

    typed_storage = torch.TypedStorage(wrap_storage=untyped_storage,
                                       dtype=dtype,
                                       _internal=True)

    t = torch._utils._rebuild_tensor(
        typed_storage,
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t


def rebuild_gcu_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    # If storage_handle is None, storage points to nullptr.
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0,
                              dtype=dtype,
                              device=storage_device,
                              _internal=True)
    else:
        storage = storage_from_cache(storage_cls,
                                     (storage_handle, storage_offset_bytes))
        if storage is None:
            torch.gcu._lazy_init()
            storage = storage_cls._new_shared_gcu(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )
            shared_cache[(storage_handle,
                          storage_offset_bytes)] = StorageWeakRef(storage)
        else:
            # We already ref counting this Storage, but producer needs new ref-counters to be released.
            storage_cls._release_ipc_counter(ref_counter_handle,
                                             ref_counter_offset,
                                             device=storage_device)

    _storage = (storage if isinstance(storage, torch.UntypedStorage) else
                storage._untyped_storage)

    t = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=_storage,
                                   dtype=dtype,
                                   _internal=True),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t


def _reduce_tensor(tensor):
    storage = tensor._typed_storage()

    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
            "since autograd does not support crossing process boundaries.  "
            "If you just want to transfer the data, call detach() on the tensor "
            "before serializing (e.g., putting it on the queue).")

    check_serializing_named_tensor(tensor)
    torch.utils.hooks.warn_if_has_hooks(tensor)

    from torch.nested._internal.nested_tensor import NestedTensor

    if tensor.is_nested and not isinstance(tensor, NestedTensor):
        # return reduce_nested_tensor(tensor)
        raise NotImplementedError(
            "Nested tensors on GCU are not supported yet")

    if tensor.layout in {
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_bsr,
            torch.sparse_csc,
            torch.sparse_bsc,
    }:
        # return reduce_sparse_tensor(tensor)
        raise NotImplementedError(
            "Sparse tensors on GCU are not supported yet")

    storage = tensor._typed_storage()

    if storage._untyped_storage.device.type == "cuda":
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        # _backward_hooks purposely omitted here, see
        # Note [Don't serialize hooks]
        return (
            rebuild_cuda_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor_offset,  # tensor offset in its storage
                type(storage),
                tensor.dtype,
                device,
                handle,  # identifier which CUDA allocation is the storage in.
                storage_size_bytes,  # size(in bytes) of the storage
                storage_offset_bytes,  # offset(in bytes) of the storage in the CUDA allocation
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )
    elif storage._untyped_storage.device.type == "gcu":
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_gcu_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        # _backward_hooks purposely omitted here, see
        # Note [Don't serialize hooks]
        return (
            rebuild_gcu_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor_offset,  # tensor offset in its storage
                type(storage),
                tensor.dtype,
                device,
                handle,  # identifier which GCU allocation is the storage in.
                storage_size_bytes,  # size(in bytes) of the storage
                storage_offset_bytes,  # offset(in bytes) of the storage in the GCU allocation
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )
    elif storage._untyped_storage.device.type == "meta":
        return (
            rebuild_meta_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.dtype,
                tensor.untyped_storage().size(),
                tensor.requires_grad,
            ),
        )

    # _backward_hooks purposely omitted here, see Note [Don't serialize hooks]
    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        tensor.requires_grad,
    )
    return (rebuild_tensor, (type(tensor), storage, metadata))


def rebuild_event_gcu(device, handle):
    return torch.gcu.Event.from_ipc_handle(device, handle)


def reduce_event_gcu(event):
    handle = event.ipc_handle()
    return (rebuild_event_gcu, (event.device, handle))


def apply_reductions_patch():
    ForkingPickler.register(torch.gcu.Event, reduce_event_gcu)
    torch.multiprocessing.reductions.reduce_storage = _reduce_storage
    torch.multiprocessing.reductions.reduce_tensor = _reduce_tensor

    torch.multiprocessing.reductions.init_reductions()
