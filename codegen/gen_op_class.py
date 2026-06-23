import yaml

from typing import List, Dict, Sequence, Union

from torchgen.model import (
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.api.types import DispatcherSignature

from torchgen.context import native_function_manager
from torchgen.gen import cpp_string
from torchgen.utils import concatMap

def gen_one_op_class(f:NativeFunction):
    namespace = f.namespace
    sig = DispatcherSignature.from_schema(f.func)
    name = f.func.name.unambiguous_name()
    return f"""
struct {name} {{
  using schema = {sig.type()};
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "{namespace}::{f.func.name.name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "{f.func.name.overload_name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, {cpp_string(str(f.func))})
  static {sig.defn(name="call", is_redispatching_fn=False)};
  static {sig.defn(name="redispatch", is_redispatching_fn=True)};
}};"""

def gen_op_class(
    backend_yaml_path: str,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
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

    op_list = [OperatorName.parse(op) for op in supported]

    ops = []
    for g in grouped_native_functions:
        native_functions_map: Dict[OperatorName, NativeFunction] = {
            f.func.name: f
            for f in concatMap(
                lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
                [g],
            )
        }
        for f in native_functions_map:
            if (f in op_list):
                func = native_functions_map[f]
                with native_function_manager(func):
                    op_decl = gen_one_op_class(func)
                    ops.append({"func": f"{f}", "decl": op_decl})

    ops.sort(key=lambda x: x['func'])

    return "\n".join(op["decl"] for op in ops)