from typing import FrozenSet, List, Optional
from contextlib import contextmanager
import itertools
import logging
import sys

import torch
from torch._subclasses import FakeTensor
from torch.fx.node import Node
import torch._prims_common as utils
from torch._inductor.compile_fx import (
    BoxedBool,
    BoxedDeviceIndex,
    CompiledFxGraph,
    is_tf32_warning_applicable,
    _warn_tf32_disabled,
    V,
    _shape_env_from_inputs,
    view_to_reshape,
    fake_tensor_prop,
    post_grad_passes,
    GraphLowering,
    fx_codegen_and_compile as inductorc_fx_codegen_and_compile,
    compile_fx_inner as inductor_compile_fx_inner,
)
from .profile import ProfiledCompiledFxGraph


logger = logging.getLogger(__name__)


@contextmanager
def format_fx_node():
    old_format = Node.format_node

    # @compatibility(is_backward_compatible=True)
    def new_format(self,
                   placeholder_names: Optional[List[str]] = None,
                   maybe_return_typename: Optional[List[str]] = None) -> Optional[str]:

        def format_fake_tensor(t: FakeTensor):
            return f"FakeTensor({t.device}, {t.dtype}, {t.size()}, {t.stride()}, {utils.suggest_memory_format(t)})"

        node_info = old_format(self, placeholder_names, maybe_return_typename)
        if "val" in self.meta:
            val = self.meta["val"]
            if isinstance(val, (tuple, list)):
                node_info += " - > ["
                for i, v in enumerate(val):
                    if isinstance(val, FakeTensor):
                        node_info += f"{i}: {format_fake_tensor(val)}, "
                    else:
                        node_info += f"{i}: {v}, "
                node_info += "]"
            elif isinstance(val, FakeTensor):
                node_info += f" -> {format_fake_tensor(val)}"
            else:
                node_info += f" -> meta info {val}"
        return node_info

    Node.format_node = new_format

    try:
        yield
    finally:
        Node.format_node = old_format


def fx_codegen_and_compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs: Optional[BoxedBool] = None,
    num_fixed: int = 0,
    is_backward: bool = False,
    graph_id: Optional[int] = None,
    cpp_wrapper: bool = False,
    aot_mode: bool = False,
    is_inference: bool = False,
    user_visible_outputs: FrozenSet[str] = frozenset(),
    layout_opt: Optional[bool] = None,
) -> CompiledFxGraph:
    if is_tf32_warning_applicable(gm):
        _warn_tf32_disabled()

    # lift the maximum depth of the Python interpreter stack
    # to adapt large/deep models
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

    V.debug.fx_graph(gm, example_inputs)

    shape_env = _shape_env_from_inputs(example_inputs)

    # Convert view to reshape in the graph. This is necessary primarily for
    # layout optimization. Do it unconditionally for uniformity.
    #
    # It's needed because when we do layout optimization, an contiguous tensor
    # in eager mode may becomes a channels last tensor. A view op previously
    # can be applied to the contiguous tensor may not be able to be applied
    # on the channels tensor any more. An error like
    #   RuntimeError: view size is not compatible with input tensor's size and stride
    #   (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
    # will be printed.
    #
    # Replace view op to reshape op in this case.
    # As an example, timm_resnest/botnet26t_256/convnext_base etc. will fail if we don't do this.
    #
    # Also this has to be done before FakeTensorProp below to avoid the failed
    # .view() call.
    view_to_reshape(gm)

    fake_mode = fake_tensor_prop(gm, example_inputs)

    # pattern matcher passes might not preserve striding information
    # on node.meta["val"]. if in the future we rely on these being
    # correct we will need to fix.

    with V.set_fake_mode(fake_mode):  # type: ignore[call-arg]
        # has some issues with memory in training
        post_grad_passes(gm, is_inference=is_inference)
        V.debug.fx_graph_transformed(gm, example_inputs)

    with format_fx_node():
        logger.debug("Post-AOT Autograd {} graph {}:\n{}"
                     .format("BACKWARDS" if is_backward else "FORWARDS", graph_id, gm.graph))

    with V.set_fake_mode(fake_mode):  # type: ignore[call-arg]
        graph = GraphLowering(
            gm,
            shape_env=shape_env,
            num_static_inputs=num_fixed,
            graph_id=graph_id,
            cpp_wrapper=cpp_wrapper,
            aot_mode=aot_mode,
            user_visible_outputs=user_visible_outputs,
        )
        with V.set_graph_handler(graph):  # type: ignore[call-arg]
            graph.run(*example_inputs)
            context = torch._guards.TracingContext.get()
            if context is not None and context.output_strides is not None:
                # Return the output strides to the caller via TracingContext
                assert len(context.output_strides) == 0
                assert graph.graph_outputs is not None
                for out in graph.graph_outputs:
                    if hasattr(out, "layout"):
                        context.output_strides.append(
                            tuple(  # type: ignore[arg-type]
                                V.graph.sizevars.size_hint(s) for s in out.layout.stride
                            )
                        )
                    else:
                        context.output_strides.append(None)
            compiled_fn = graph.compile_to_fn()

            if graph.disable_cudagraphs:
                BoxedBool.disable(cudagraphs)

            compiled_graph = ProfiledCompiledFxGraph(
                f"FxGraph-{graph_id}",
                compiled_artifact=compiled_fn,
                cache_key=graph.cache_key,
                artifact_path=graph.cache_path,
                cache_linemap=graph.cache_linemap,
                device_types=graph.device_types,
                device_idxs=graph.device_idxs,
                mutated_inputs=graph.mutated_inputs,
                mutated_input_idxs=set(graph.mutated_input_idxs),
            )
    return compiled_graph


def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs: Optional[BoxedBool] = None,
    num_fixed: int = 0,
    is_backward: bool = False,
    graph_id: Optional[int] = None,
    cpp_wrapper: bool = False,
    aot_mode: bool = False,
    is_inference: bool = False,
    boxed_forward_device_index: Optional[BoxedDeviceIndex] = None,
    user_visible_outputs: FrozenSet[str] = frozenset(),
    layout_opt: Optional[bool] = None,
):
    torch._inductor.compile_fx.fx_codegen_and_compile = fx_codegen_and_compile
    compiled_graph: CompiledFxGraph = inductor_compile_fx_inner(
        gm,
        example_inputs,
        cudagraphs=cudagraphs,
        num_fixed=num_fixed,
        is_backward=is_backward,
        graph_id=graph_id,
        cpp_wrapper=cpp_wrapper,
        aot_mode=aot_mode,
        is_inference=is_inference,
        boxed_forward_device_index=boxed_forward_device_index,
        user_visible_outputs=user_visible_outputs,
        layout_opt=layout_opt
    )
    return compiled_graph
