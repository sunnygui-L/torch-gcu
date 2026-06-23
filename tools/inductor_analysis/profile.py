from typing import Any, Callable, List, Set
from dataclasses import field
import itertools
import logging
import time

import torch
from torch._inductor.compile_fx import CompiledFxGraph


logger = logging.getLogger(__name__)


GLOBAL_PROFILE_INFO = {}
GLOBAL_PROFILE_ENABLE = True


class ProfiledFxGraphEager(torch.nn.Module):
    """Class holding a FX graph for eager profiling"""

    def __init__(self, gm_name, gm):
        super().__init__()
        self.gm_name = gm_name
        self.gm = gm
        self.gm_exec_counter = itertools.count(1)
        self.gm_exec_ms = 0

    def __call__(self, *args) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        outputs = self.gm(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        global GLOBAL_PROFILE_ENABLE
        if GLOBAL_PROFILE_ENABLE:
            counter = next(self.gm_exec_counter)
            t_ms = (time.time() - t_start) * 1000
            self.gm_exec_ms += t_ms
            logger.debug(
                f"{self.gm_name} run {counter} with current {t_ms}ms and avg {self.gm_exec_ms/counter}ms")
            GLOBAL_PROFILE_INFO[self.gm_name] = {
                "counter": counter, "time": self.gm_exec_ms}
        return outputs


class ProfiledCompiledFxGraph(CompiledFxGraph):
    def __init__(self,
                 graph_name,
                 compiled_artifact: Callable = None,
                 current_callable: Callable = None,
                 cache_key: str = None,
                 artifact_path: str = None,
                 cache_linemap: List = None,
                 device_types: Set[str] = field(default_factory=set),
                 device_idxs: Set[int] = field(default_factory=set),
                 mutated_inputs: Set[str] = field(default_factory=set),
                 mutated_input_idxs: Set[int] = field(default_factory=list)):
        super().__init__(compiled_artifact,
                         current_callable,
                         cache_key,
                         artifact_path,
                         cache_linemap,
                         device_types,
                         device_idxs,
                         mutated_inputs,
                         mutated_input_idxs)
        self.gm_name = graph_name
        self.gm_exec_ms = 0
        self.gm_exec_counter = itertools.count(1)

    def __call__(self, inputs) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        outputs = self.get_current_callable()(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        global GLOBAL_PROFILE_ENABLE
        if GLOBAL_PROFILE_ENABLE:
            counter = next(self.gm_exec_counter)
            t_ms = (time.time() - t_start) * 1000
            self.gm_exec_ms += t_ms
            logger.debug(
                f"{self.gm_name} run {counter} with current {t_ms}ms and avg {self.gm_exec_ms/counter}ms")
            GLOBAL_PROFILE_INFO[self.gm_name] = {
                "counter": counter, "time": self.gm_exec_ms}
        return outputs


def get_profile_infos():
    return GLOBAL_PROFILE_INFO


def enable_profile(enabled: bool = True):
    global GLOBAL_PROFILE_ENABLE
    GLOBAL_PROFILE_ENABLE = enabled


def dump_profile_infos():
    profile_infos = get_profile_infos()
    total = sum([value["time"] for value in profile_infos.values()])
    sorted_profile_infos = sorted(
        profile_infos.items(), key=lambda x: x[1]["time"], reverse=True)
    for key, val in sorted_profile_infos:
        counter, time_ms = val["counter"], val["time"]
        avg = time_ms / counter
        pct = (time_ms / total) * 100
        print(f"{key}: counter {counter}, time {time_ms}ms, avg {avg}ms, {pct}")
