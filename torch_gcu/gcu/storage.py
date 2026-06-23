import torch
import torch_gcu
from typing import Union


def _share_gcu_(self, *args, **kwargs):
    return torch_gcu._C._share_gcu_(self, *args, **kwargs)


def _typed_storage_share_gcu_(self, *args, **kwargs):
    return self._untyped_storage._share_gcu_(*args, **kwargs)


def _new_shared_gcu(*args, **kwargs):
    return torch_gcu._C._new_shared_gcu(*args, **kwargs)


def _typed_storage_new_shared_gcu(*args, **kwargs):
    return torch.UntypedStorage._new_shared_gcu(*args, **kwargs)


def _release_ipc_counter_gcu(*args, **kwargs):
    return torch_gcu._C._release_ipc_counter_gcu(*args, **kwargs)


def _typed_storage_release_ipc_counter_gcu(
    *args, device: Union[str, torch.device] = "gcu", **kwargs
):
    return torch.UntypedStorage._release_ipc_counter_gcu(*args, **kwargs)


def apply_storage_patch():
    # torch.UntypedStorage.is_shared = _is_shared
    # torch.UntypedStorage.is_pinned = _untyped_storage_is_pinned
    # torch.UntypedStorage.pin_memory = _untyped_storage_pin_memory
    # torch.UntypedStorage.copy_ = _untyped_storage_copy
    setattr(torch.UntypedStorage, "_share_gcu_", _share_gcu_)
    setattr(torch.UntypedStorage, "_new_shared_gcu", _new_shared_gcu)
    setattr(torch.UntypedStorage, "_release_ipc_counter_gcu", _release_ipc_counter_gcu)
    setattr(torch.UntypedStorage, "_release_ipc_counter_cuda", _release_ipc_counter_gcu)
    # torch.TypedStorage.__init__ = _typed_storage_init
    # torch.TypedStorage.copy_ = _typed_storage_copy
    # torch.TypedStorage.cpu = _cpu
    # torch.TypedStorage.is_pinned = _typed_storage_is_pinned
    # torch.TypedStorage.pin_memory = _typed_storage_pin_memory
    setattr(
        torch.TypedStorage,
        "_release_ipc_counter_gcu",
        _typed_storage_release_ipc_counter_gcu,
    )
    # torch.serialization._save = _save
    # torch.serialization._legacy_save = _legacy_save
    # torch.serialization._load = _load
    # torch.serialization._legacy_load = _legacy_load
    # setattr(torch.TypedStorage, "gcu", typedstorage_to_gcu)
    setattr(torch.TypedStorage, "_share_gcu_", _typed_storage_share_gcu_)
    setattr(torch.TypedStorage, "_new_shared_gcu", _typed_storage_new_shared_gcu)

    # torch.package.PackageExporter._persistent_id = _package_exporter_persistent_id
