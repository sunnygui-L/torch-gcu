from typing import Any, Dict, List, Optional
import itertools
import copy

import torch
from torch._dynamo.backends.registry import register_backend


EAGER_PROFILE_MODE = "eager_profile_mode"
GRAPH_COUNTER = itertools.count(0)


@register_backend(name="inductor_analysis")
def compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    options: Optional[Dict[str, Any]] = None,
):
    """
    Compile a given FX graph with inductor for perf analysis.

    Args:
        gm: The FX graph to compile.
        example_inputs:  List of tensor inputs.
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Callable with same behavior as gm but faster.
    """

    graph_id = next(GRAPH_COUNTER)

    if options is not None:
        options = copy.deepcopy(options)
        profile = False
        if EAGER_PROFILE_MODE in options:
            profile = options[EAGER_PROFILE_MODE]
            del options[EAGER_PROFILE_MODE]
            if profile:
                from .profile import ProfiledFxGraphEager
                return ProfiledFxGraphEager(gm_name=f"FxGraph-{graph_id}", gm=gm)

    from torch._inductor.compile_fx import compile_fx
    from .compile_fx import compile_fx_inner

    return compile_fx(gm, example_inputs, inner_compile=compile_fx_inner, config_patches=options)
