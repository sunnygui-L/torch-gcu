import os
import yaml

from typing import List, Dict, Sequence, Union

from torchgen.model import (
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.utils import concatMap

space = " " * 2

hardware_list = ["s60", "l600"]

def capitalize_with_underline(s:str):
    str_list = s.split("_")
    return "_".join([x.capitalize() for x in str_list])

def get_namespace(f:Union[NativeFunction, NativeFunctionsGroup]):
    if isinstance(f, NativeFunction):
        return f.namespace
    elif isinstance(f, NativeFunctionsGroup):
        return f.functional.namespace
    else:
        return "aten"

def get_op_list(backend_yaml_path: str):
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
    op_cfg = supported_cfg + optional_cfg

    op_cfg_map = {
        op_name: op_cfg
        for op_name, op_cfg in zip(op_list, op_cfg)
    }
    return op_cfg_map

def get_fallback_op_list(fallback_yaml_path: str):
    if fallback_yaml_path == None:
        return {}

    with open(fallback_yaml_path, "r") as f:
        yaml_values = yaml.safe_load(f)
    assert isinstance(yaml_values, dict)

    fallback_op_cfg = yaml_values.pop("fallback", [])
    if fallback_op_cfg is None:
        fallback_op_cfg = []  # Allow an empty list of fallback op

    op_name = [op["name"] for op in fallback_op_cfg]

    op_list = [OperatorName.parse(op) for op in op_name]

    op_cfg_map = {
        op_name: op_cfg
        for op_name, op_cfg in zip(op_list, fallback_op_cfg)
    }

    return op_cfg_map

def to_args_map_value(args: dict):
    args_list = []
    for k, v in args.items():
        if isinstance(v, str):
            values = v.split(",")
            for value in values:
                args_list.append(f"\"{k}::{value}\"")
        else:
            args_list.append(f"\"{k}::{v}\"")
    return ", ".join(args_list)

def register_one_op(hardware:str, f:NativeFunction, cfg:dict = None):
    body = []
    if cfg is None:
        body.append(f'{space}registerFallbackOps("{f.namespace}::{f.func.name}", {{}});')
    else:
        if (hardware in cfg):
            h_cfg = cfg[hardware]
            if h_cfg == "all":
                body.append(f'{space}registerFallbackOps(\"{f.namespace}::{cfg["name"]}\", {{"all"}});')
            else:
                assert isinstance(h_cfg, list), f'expected "{hardware}" to be a list, but got: {h_cfg} (of type {type(h_cfg)})'
                args_list = ",".join([to_args_map_value(args) for args in h_cfg])
                body.append(f'{space}registerFallbackOps(\"{f.namespace}::{cfg["name"]}\", {{{args_list}}});')
    return body

def register_fallback_ops(op_cfg_map_list: List[dict], fallback_yaml_path_list: List[str],
                          grouped_native_functions_list: List[Sequence[Union[NativeFunction, NativeFunctionsGroup]]]):

    register_funcs = []
    for h in hardware_list:
        defn = f"void RegisterFallBackOps::init{capitalize_with_underline(h)}Register() {{\n"
        body = []

        for op_cfg_map, fallback_yaml_path, grouped_native_functions in zip(op_cfg_map_list, fallback_yaml_path_list, grouped_native_functions_list):
            fallback_op_cfg_map = get_fallback_op_list(fallback_yaml_path)
            for g in grouped_native_functions:
                native_functions_map: Dict[OperatorName, NativeFunction] = {
                    f.func.name: f
                    for f in concatMap(
                        lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
                        [g],
                    )
                }

                for f in native_functions_map:
                    if (f in op_cfg_map):
                        fun_cfg = op_cfg_map[f]
                        func = native_functions_map[f]
                        if (fun_cfg.get("fallback_cpu", True)) and f in fallback_op_cfg_map:
                            cfg = fallback_op_cfg_map[f]
                            body.extend(register_one_op(h, func, cfg))
                        elif fun_cfg.get("fallback_cpu", True):
                            body.extend(register_one_op(h, func))

        func = defn + "\n".join(body) + "\n}"
        register_funcs.append(func)

    return "\n".join(register_funcs)

def get_limited_args(cfg:dict = None):
    limited_args = []

    for hardware in hardware_list:
        if (cfg is not None) and (hardware in cfg):
            h_cfg = cfg[hardware]
            if (h_cfg != "all"):
                assert isinstance(h_cfg, list), f'expected "{hardware}" to be a list, but got: {h_cfg} (of type {type(h_cfg)})'
                args_list = [k for args in h_cfg for k in args]
                limited_args.extend(args_list)
    if len(limited_args) == 0:
        return None
    else:
        args_str = ", ".join([f'std::make_tuple(std::string("{arg}"), {arg})' for arg in list(set(limited_args))])
        return args_str


def gen_limited_fallback(f:NativeFunction):
    codegen_root = os.path.dirname(os.path.abspath(__file__))
    fallback_yaml_path = os.path.join(codegen_root, "gcu_op_fallback.yaml")
    fallback_op_cfg_map = get_fallback_op_list(fallback_yaml_path)
    op_name = f.func.name
    if op_name in fallback_op_cfg_map:
        limited_args = get_limited_args(fallback_op_cfg_map[op_name])
    else:
        limited_args = None
    args = [f'"{f.namespace}::{op_name}"']
    if limited_args is not None:
        args.append(limited_args)
    body = f'bool limited_fallback = isFallback({", ".join(args)});'
    return body