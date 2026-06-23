import yaml

from typing import List, Dict, Sequence, Union

from torchgen.model import (
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    SchemaKind,
)
from torchgen.utils import concatMap
from torchgen.packaged.autograd.gen_inplace_or_view_type import ALL_VIEW_FUNCTIONS

from gen_aot_ops import RUNTIME_OP, NONE_KERNEL_OP, SPLIT_TOPSATEN_MULTI_OUT_OP, aten_to_kernel, get_multi_outputs

exclude = list(
    set(list(ALL_VIEW_FUNCTIONS.keys())
        + RUNTIME_OP
        + NONE_KERNEL_OP
    )
)

def gen_topsaten_bridge(
    backend_yaml_path: str,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]]
) -> List[str]:

    with open(backend_yaml_path, "r") as f:
        yaml_values = yaml.safe_load(f)
    assert isinstance(yaml_values, dict)

    supported_cfg = yaml_values.pop("supported", [])
    if supported_cfg is None:
        supported_cfg = []  # Allow an empty list of supported ops

    supported = [op["name"] for op in supported_cfg]

    assert isinstance(
        supported, list
    ), f'expected "supported" to be a list, but got: {supported} (of type {type(supported)})'

    optional_cfg = yaml_values.pop("optional", [])
    if optional_cfg is None:
        optional_cfg = []  # Allow an empty list of optional ops

    optional = [op["name"] for op in optional_cfg]

    assert isinstance(
        optional, list
    ), f'expected "optional" to be a list, but got: {optional} (of type {type(optional)})'

    op_list = [OperatorName.parse(op) for op in supported + optional]
    op_cfg_map = {
        op_name: op_cfg
        for op_name, op_cfg in zip(op_list, supported_cfg + optional_cfg)
    }

    impls = []
    one_out = set()
    two_out = set()
    three_out = set()
    four_out = set()
    multi_out = set()

    for g in grouped_native_functions:
        native_functions_map: Dict[OperatorName, NativeFunction] = {
            f.func.name: f
            for f in concatMap(
                lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
                [g],
            )
        }
        for f in native_functions_map:
            if (f in op_list) and (f"{f}" not in exclude):
                fun_cfg = op_cfg_map[f]
                if (fun_cfg.get("gen_topsaten_bridge", True)):
                    func = native_functions_map[f]
                    out_args = get_multi_outputs(func)
                    out_num = len(out_args)
                    if out_num == 1:
                        one_out.add(aten_to_kernel(f"{f}"))
                    elif f"{f}" in SPLIT_TOPSATEN_MULTI_OUT_OP:
                        if out_num == 2:
                            two_out.add(aten_to_kernel(f"{f}"))
                        elif out_num == 3:
                            three_out.add(aten_to_kernel(f"{f}"))
                        elif out_num == 4:
                            four_out.add(aten_to_kernel(f"{f}"))
                    else:
                        multi_out.add(aten_to_kernel(f"{f}"))


    one_out = sorted(one_out)
    two_out = sorted(two_out)
    three_out = sorted(three_out)
    four_out = sorted(four_out)
    multi_out = sorted(multi_out)

    for op in one_out:
        impls.append(f"DEFINE_BRIDGE_TOPSATENOP({op})")

    for op in two_out:
        impls.append(f"DEFINE_BRIDGE_TOPSATENOP_OUT2({op})")

    for op in three_out:
        impls.append(f"DEFINE_BRIDGE_TOPSATENOP_OUT3({op})")

    for op in four_out:
        impls.append(f"DEFINE_BRIDGE_TOPSATENOP_OUT4({op})")

    for op in multi_out:
        impls.append(f"DEFINE_BRIDGE_TOPSATENOP_MULTI_OUT({op})")

    return impls
