import yaml

from typing import List, Dict, Sequence, Union

from torchgen.api import cpp
from torchgen.context import native_function_manager
from torchgen.model import (
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    SchemaKind,
    ListType,
)
from torchgen.api.types import DispatcherSignature, Binding
from torchgen.utils import concatMap
from gen_op_fallback_register import gen_limited_fallback


def all_arguments(f: NativeFunction) -> str:
    all_args = cpp.arguments(
        f.func.arguments,
        faithful=True,
        symint=False,
        method=False,
        cpp_no_default_args=f.cpp_no_default_args,
    )
    cpp_args = [a.name for a in all_args]
    cpp_args_str = ", ".join(cpp_args)
    return cpp_args_str

def get_owned_name(arg: Binding) -> str:
    if arg.nctype.cpp_type() == "at::TensorList":
        return arg.name + ".vec()"
    elif arg.nctype.cpp_type() == "const at::ITensorListRef &":
        return f"listref_to_vector({arg.name})"
    else:
        return arg.name

def all_owned_arguments(f: NativeFunction) -> str:
    all_args = cpp.arguments(
        f.func.arguments,
        faithful=True,
        symint=False,
        method=False,
        cpp_no_default_args=f.cpp_no_default_args,
    )

    cpp_args = [get_owned_name(a) for a in all_args]
    cpp_args_str = ", ".join(cpp_args)
    return cpp_args_str

def opcheck_argument(f: NativeFunction, tuple_name: str = "clone_args") -> str:
    num_args = len(f.func.arguments.all)
    opcheck_args = [f"std::get<{i}>({tuple_name})" for i in range(num_args)]
    opcheck_args_str = ", ".join(opcheck_args)
    return opcheck_args_str

def self_argument(f: NativeFunction) -> str:
    arguments = f.func.arguments
    self_arg = arguments.self_arg.argument
    out_args = [
        r.no_default()
        for r in cpp.argument(
            self_arg,
            faithful=True,
            symint=False,
            method=False,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=f.cpp_no_default_args,
        )
    ]
    cpp_args = [
        a.name + ".vec()" if isinstance(a.argument.type, ListType) else a.name
        for a in out_args
    ]
    cpp_args_str = ", ".join(cpp_args)
    return cpp_args_str

def opcheck_self_argument(f: NativeFunction, result_name: str = "result") -> str:
    self_num = 0 + len(f.func.arguments.pre_self_positional)
    cpp_args_str = f"std::get<{self_num}>({result_name})"
    return cpp_args_str

def out_arguments(f: NativeFunction) -> str:
    arguments = f.func.arguments
    args = arguments.out
    out_args = [
        r.no_default()
        for a in args
        for r in cpp.argument(
            a,
            faithful=True,
            symint=False,
            method=False,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=f.cpp_no_default_args,
        )
    ]
    cpp_args = [
        a.name + ".vec()" if isinstance(a.argument.type, ListType) else a.name
        for a in out_args
    ]
    cpp_args_str = ", ".join(cpp_args)
    return cpp_args_str

def opcheck_out_argument(f: NativeFunction, result_name: str = "result") -> str:
    out_args = f.func.arguments.out
    out_args_start = len(f.func.arguments.non_out)

    cpp_args = [
        f"std::get<{out_args_start + i}>({result_name})" for i, arg in enumerate(out_args)
    ]

    cpp_args_str = ", ".join(cpp_args)
    return cpp_args_str

def non_out_arguments(f: NativeFunction) -> str:
    arguments = f.func.arguments
    args = arguments.non_out
    non_out_args = [
        r.no_default()
        for a in args
        for r in cpp.argument(
            a,
            faithful=True,
            symint=False,
            method=False,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=f.cpp_no_default_args,
        )
    ]
    cpp_args = [a.name for a in non_out_args]
    cpp_args_str = ", ".join(cpp_args)
    return cpp_args_str

def get_aot_op(f: NativeFunction) -> str:
    return cpp.name(f.func)

def get_fallback_op(f: NativeFunction) -> str:
    return f.func.name.unambiguous_name()

def get_check_op(f: NativeFunction) -> str:
    return f.func.name.unambiguous_name()

def only_call_body(f: NativeFunction) -> list:
    sig_body = []
    sig_body.append(f"  return aotops::{get_aot_op(f)}({all_arguments(f)});")
    return sig_body

def cpu_fallback_body(f: NativeFunction) -> list:
    sig_body = []
    name_space = f.namespace.upper()
    sig_body.append(f"  {gen_limited_fallback(f)}")
    sig_body.append(f"  GCU_CPU_FALLBACK_WITH_LIMITED(cfg, {name_space}, {get_fallback_op(f)}, limited_fallback, {all_arguments(f)})")
    return sig_body

def op_calltrace_body(f: NativeFunction) -> list:
    sig_body = []
    sig_body.append(f"  OP_CALLTRACE(cfg, {get_aot_op(f)})")
    return sig_body

def op_check_body(f: NativeFunction) -> list:
    name_space = f.namespace.upper()
    aot_op = get_aot_op(f)
    all_args = all_arguments(f)
    all_owned_args = all_owned_arguments(f)
    out_args = out_arguments(f)
    opcheck_clone_args = "clone_op_check_input"

    sig_body = []

    sig_body.append(f"""  static bool enable_op_check = cfg.enableOpCheck(__func__);
  static bool disable_op_check = cfg.disableOpCheck(__func__);
  bool op_check_scope = cfg.inOpCheckScope();
  bool op_check = (enable_op_check || op_check_scope) && (!disable_op_check);
  if (op_check) {'{'}
    OP_CHECK_INPUT_INFO_RECOED({all_args})
    auto clone_input = clone_args({all_owned_args});""")
    k = f.func.kind()
    if k == SchemaKind.mutable:
        if aot_op.endswith("outf"):
            k = SchemaKind.out
        else:
            k = SchemaKind.functional

    sig_body.append(f"    auto {opcheck_clone_args} = clone_args({all_owned_args});")

    opcheck_input = opcheck_argument(f, opcheck_clone_args)

    if k is SchemaKind.functional:
        opcheck_return = "    auto&& xdevice_out =\n    "
    else:
        opcheck_return = ""

    sig_body.append(f"""{opcheck_return}    at::native::call_fallback_fn<&torch_gcu::gcu_opcheck_run,
                                 {name_space}_OP({get_check_op(f)})>::call({opcheck_input});""")

    if k is SchemaKind.inplace:
        sig_body.append(f"    auto&& xdevice_out = {opcheck_self_argument(f, opcheck_clone_args)};")
    elif k is SchemaKind.out:
        if len(f.func.arguments.out) > 1:
            sig_body.append(f"    auto xdevice_out = std::forward_as_tuple({opcheck_out_argument(f, opcheck_clone_args)});")
        else:
            sig_body.append(f"    auto&& xdevice_out = {opcheck_out_argument(f, opcheck_clone_args)};")

    if k is SchemaKind.functional:
        sig_body.append(f"    auto gcu_out = aotops::{aot_op}({all_args});")
    else:
        sig_body.append(f"    aotops::{aot_op}({all_args});")

        if k is SchemaKind.inplace:
            sig_body.append(f"    auto&& gcu_out = {self_argument(f)};")
        elif k is SchemaKind.out:
            if len(f.func.arguments.out) > 1:
                sig_body.append(f"    auto gcu_out = std::forward_as_tuple({out_args});")
            else:
                sig_body.append(f"    auto&& gcu_out = {out_args};")

    sig_body.append(f"""    auto result =
        gcu_out_check(gcu_out, xdevice_out, std::string(__func__));
    if (result.acc_pass && !cfg.enableTestMode()) {'{'}
      PTDLOG(OP) << result.check_info;
    {"} else {"}
      OP_CHECK_DEBUG_INFO(cfg, ss, gcu_out, xdevice_out, clone_input)
    {'}'}""")

    if len(f.func.returns) > 0:
        sig_body.append(f"    return gcu_out;")
    else:
        sig_body.append(f"    return;")

    sig_body.append(f"""  {"} else {"}
    return aotops::{aot_op}({all_args});
  {'}'}""")

    return sig_body

def gen_only_call(g: Union[NativeFunction, NativeFunctionsGroup], f: NativeFunction) -> str:
    sig_body = []

    all_inputs = all_arguments(f)
    sig_body.append(f"OP_COMMON_MACRO({all_inputs})")

    sig_body.extend(only_call_body(f))

    sig_body_str = "\n".join(sig_body)

    definition = DispatcherSignature.from_schema(f.func, prefix="GCUNativeFunctions::", symint=False).defn()
    impl = f"""
{definition} {{
  {sig_body_str}
}}
"""
    return impl

def gen_call_and_fallback(g: Union[NativeFunction, NativeFunctionsGroup], f: NativeFunction) -> str:
    sig_body = []

    sig_body.append(f"auto& cfg = torch_gcu::OpDebugConfig::GetInstance();")

    sig_body.extend(op_calltrace_body(f))

    sig_body.append(f"  PRINT_OP_NAME_FALLBACK(cfg)")

    sig_body.extend(cpu_fallback_body(f))

    all_inputs = all_arguments(f)
    sig_body.append(f"  OP_COMMON_MACRO({all_inputs})")

    sig_body.extend(only_call_body(f))

    sig_body_str = "\n".join(sig_body)

    definition = DispatcherSignature.from_schema(f.func, prefix="GCUNativeFunctions::", symint=False).defn()
    impl = f"""
{definition} {{
  {sig_body_str}
}}
"""
    return impl

def gen_call_and_check(g: Union[NativeFunction, NativeFunctionsGroup], f: NativeFunction) -> str:
    sig_body = []

    sig_body.append(f"auto& cfg = torch_gcu::OpDebugConfig::GetInstance();")

    sig_body.extend(op_calltrace_body(f))

    sig_body.append(f"  PRINT_OP_NAME_CHECK(cfg)")

    all_inputs = all_arguments(f)
    sig_body.append(f"  OP_COMMON_MACRO({all_inputs})")

    sig_body.extend(op_check_body(f))

    sig_body_str = "\n".join(sig_body)

    definition = DispatcherSignature.from_schema(f.func, prefix="GCUNativeFunctions::", symint=False).defn()
    impl = f"""
{definition} {{
  {sig_body_str}
}}
"""
    return impl

def gen_call_fallback_and_check(g: Union[NativeFunction, NativeFunctionsGroup], f: NativeFunction) -> str:
    sig_body = []

    sig_body.append(f"auto& cfg = torch_gcu::OpDebugConfig::GetInstance();")

    sig_body.extend(op_calltrace_body(f))

    sig_body.append(f"  PRINT_OP_NAME_ALL(cfg)")

    sig_body.extend(cpu_fallback_body(f))

    all_inputs = all_arguments(f)
    sig_body.append(f"  OP_COMMON_MACRO({all_inputs})")

    sig_body.extend(op_check_body(f))

    sig_body_str = "\n".join(sig_body)

    definition = DispatcherSignature.from_schema(f.func, prefix="GCUNativeFunctions::", symint=False).defn()
    impl = f"""
{definition} {{
  {sig_body_str}
}}
"""
    return impl

def gen_GCUNativeFunctions_methods(
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
                fun_cfg = op_cfg_map[f]
                if (not fun_cfg.get("disable", False) and fun_cfg.get("gen_native_function", True)):
                    func = native_functions_map[f]

                    if (not fun_cfg.get("op_check", True)):
                        if (not fun_cfg.get("fallback_cpu", True)):
                            gen_one = gen_only_call
                        else:
                            gen_one = gen_call_and_fallback
                    else:
                        if (not fun_cfg.get("fallback_cpu", True)):
                            gen_one = gen_call_and_check
                        else:
                            gen_one = gen_call_fallback_and_check

                    with native_function_manager(func):
                        op_impl = gen_one(g, func)
                        ops.append({"func": f"{f}", "impl": op_impl})

    ops.sort(key=lambda x: x['func'])

    return [op['impl'] for op in ops]
