r"""
This module exposes a TunableOp interface.

TorchGCU does NOT support tunable for now.

"""
import warnings
from typing import Optional, Tuple

import torch

__all__ = [
    "enable",
    "is_enabled",
    "tuning_enable",
    "tuning_is_enabled",
    "set_max_tuning_duration",
    "get_max_tuning_duration",
    "set_max_tuning_iterations",
    "get_max_tuning_iterations",
    "set_filename",
    "get_filename",
    "get_results",
    "get_validators",
    "write_file_on_exit",
    "write_file",
    "read_file",
]


def not_support_warning():
    warnings.warn("torch_gcu does NOT support tunable for now!")
    return None


def enable(val: bool = True) -> None:
    r"""This is the big on/off switch for all TunableOp implementations."""
    not_support_warning()


def is_enabled() -> bool:
    r"""Returns whether the TunableOp feature is enabled."""
    return False


def tuning_enable(val: bool = True) -> None:
    r"""Enable tuning of TunableOp implementations.

    When enabled, if a tuned entry isn't found, run the tuning step and record
    the entry.
    """
    not_support_warning()


def tuning_is_enabled() -> bool:
    r"""Returns whether TunableOp implementations can be tuned."""
    return False


def set_max_tuning_duration(duration: int) -> None:
    r"""Set max time in milliseconds to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
    not_support_warning()


def get_max_tuning_duration() -> int:
    r"""Get max time to spend tuning a given solution."""
    not_support_warning()
    return 0


def set_max_tuning_iterations(iterations: int) -> None:
    r"""Set max number of iterations to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
    not_support_warning()


def get_max_tuning_iterations() -> int:
    r"""Get max iterations to spend tuning a given solution."""
    not_support_warning()
    return 0


def set_filename(filename: str, insert_device_ordinal: bool = False) -> None:
    r"""Set the filename to use for input/output of tuning results.

    If :attr:`insert_device_ordinal` is ``True`` then the current device ordinal
    will be added to the given filename automatically. This can be used in a
    1-process-per-gcu scenario to ensure all processes write to a separate file.
    """
    not_support_warning()


def get_filename() -> str:
    r"""Get the results filename."""
    not_support_warning()
    return ""


def get_results() -> Tuple[str, str, str, float]:
    r"""Return all TunableOp results."""
    not_support_warning()
    return ("", "", "", 0.0)


def get_validators() -> Tuple[str, str]:
    r"""Return the TunableOp validators."""
    not_support_warning()
    return ("", "")


def write_file_on_exit(val: bool) -> None:
    r"""During Tuning Context destruction, write file to disk.

    This is useful as a final flush of your results to disk if your application
    terminates as result of normal operation or an error. Manual flushing of
    your results can be achieved by manually calling ``write_file()``."""
    not_support_warning()


def write_file(filename: Optional[str] = None) -> bool:
    r"""Write results to a CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
    not_support_warning()
    return False


def read_file(filename: Optional[str] = None) -> bool:
    r"""Read results from a TunableOp CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
    not_support_warning()
    return False
