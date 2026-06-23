import itertools
import torchgen.dest as dest
from typing import List, Optional
from torchgen.api.translate import translate
from torchgen.api import cpp
from torchgen.api.types import (
    CppSignature,
    CppSignatureGroup,
    Expr,
    kernel_signature,
)
from torchgen.context import native_function_manager
from torchgen.model import (
    Argument,
    DeviceCheckType,
    DispatchKey,
    gets_generated_out_inplace_wrapper,
    is_cuda_dispatch_key,
    NativeFunction,
    NativeFunctionsGroup,
    SchemaKind,
    TensorOptionsArguments,
)
from torchgen.utils import assert_never, Target
from torchgen.packaged.autograd.gen_inplace_or_view_type import ALL_VIEW_FUNCTIONS
from gen_aot_ops import get_extra_outputs

def gen_out_inplace_wrapper_with_using_int(
    self, f: NativeFunction, g: Optional[NativeFunctionsGroup]
) -> Optional[str]:
    if g is None:
        return None
    k = f.func.kind()
    if k is SchemaKind.inplace:
        copy_op = "at::_copy_from"
    elif k is SchemaKind.out:
        copy_op = "at::_copy_from_and_resize"
    else:
        raise AssertionError("gen_out_inplace_wrapper called on a functional op")

    sig = self.wrapper_kernel_sig(f)
    name = sig.name()

    func_res = f"{name}_tmp"
    return_names = cpp.return_names(f)
    if len(return_names) > 1:
        updates = "\n  ".join(
            f"{copy_op}(std::get<{i}>({func_res}), {ret_name});"
            for i, ret_name in enumerate(return_names)
        )
        returns = f'{sig.returns_type().cpp_type()}({", ".join(return_names)})'
    elif len(return_names) == 1:
        ret_name = return_names[0]
        updates = f"{copy_op}({func_res}, {ret_name});"
        returns = ret_name
    else:
        assert len(f.func.arguments.out) == 1
        returns = ""
        out_arg = f.func.arguments.out[0]
        if out_arg.type.is_list_like():
            updates = f"""\
for (size_t i = 0; i < {func_res}.size(); ++i) {{
    {copy_op}({func_res}[i], {out_arg.name}[i]);
}}"""
        else:
            updates = f"{copy_op}({func_res}, {out_arg.name});"

    functional_sig = self.wrapper_kernel_sig(g.functional)
    wrapper_name = sig.name()

    return f"""\
{sig.defn(name=wrapper_name)} {{
  auto {func_res} = {functional_sig.name()}({", ".join(e.expr for e in translate(sig.arguments(), functional_sig.arguments()))});
  {updates}
  return {returns};
}}
"""

setattr(dest.RegisterDispatchKey, "gen_out_inplace_wrapper", gen_out_inplace_wrapper_with_using_int)


no_device_check_list = list(ALL_VIEW_FUNCTIONS.keys())

no_device_gurad_list = list(ALL_VIEW_FUNCTIONS.keys())

def gen_device_check(
     type: DeviceCheckType, args: List[Argument], method_name: str
 ) -> str:
     if type == DeviceCheckType.NoCheck:
         return "// No device check\n"

     device_check = "c10::optional<at::Device> common_device = c10::nullopt;\n"
     device_check += "  (void)common_device; // Suppress unused variable warning\n"
     for arg in args:
         # Only tensor like arguments are eligible
         if arg.type.is_tensor_like():
             device_check += f"""
  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");"""
     return device_check

def input_ivalue_arguments(args_exprs_list: List[Expr]) -> str:
    cpp_args = [f"at::IValue({a.expr})" for a in args_exprs_list]
    cpp_args_str = ", ".join(cpp_args)
    return cpp_args_str

def return_ivalue_argument(f: NativeFunction, returns_name: str) -> str:
    returns_num = len(f.func.returns)
    return_list = []
    if returns_num == 1:
        return_list = [returns_name]
    elif returns_num > 1:
        return_list = [f"std::get<{i}>({returns_name})" for i in range(returns_num)]

    return_list = [f"at::IValue({name})" for name in return_list]
    return_list_str = ", ".join(return_list)
    return return_list_str

def op_statistics_input_body(args_exprs_list: List[Expr]) -> list:
    sig_body = []
    sig_body.append(f"  torch_gcu::Params params;")
    sig_body.append(f"  if (cfg.enableOpStatistics()) {{")
    sig_body.append(f"    std::vector<at::IValue> input_arguments = {{{input_ivalue_arguments(args_exprs_list)}}};")
    sig_body.append(f"    params = torch_gcu::op_record_input(input_arguments);")
    sig_body.append(f"  }}")
    return sig_body

def op_statistics_output_body(f: NativeFunction, returns_name: str) -> list:
    sig_body = []
    namespace = f.namespace
    sig_body.append(f"  if (cfg.enableOpStatistics()) {{")
    sig_body.append(f"""    auto op_schema = c10::Dispatcher::singleton()
                         .findSchemaOrThrow({namespace.upper()}_OP({f.func.name.unambiguous_name()})::name,
                                            {namespace.upper()}_OP({f.func.name.unambiguous_name()})::overload_name)
                         .schema();""")
    sig_body.append(f"    std::vector<at::IValue> return_arguments = {{{return_ivalue_argument(f, returns_name)}}};")
    sig_body.append(f"    torch_gcu::op_record_output(op_schema, return_arguments, params);")
    sig_body.append(f"  }}")
    return sig_body

def op_statistics_body(f: NativeFunction, args_exprs_list: List[Expr], returns_name: str) -> list:
    sig_body = []
    namespace = f.namespace
    sig_body.append(f"  if (cfg.enableOpStatistics()) {{")
    sig_body.append(f"""    auto op_schema = c10::Dispatcher::singleton()
                         .findSchemaOrThrow({namespace.upper()}_OP({f.func.name.unambiguous_name()})::name,
                                            {namespace.upper()}_OP({f.func.name.unambiguous_name()})::overload_name)
                         .schema();""")
    sig_body.append(f"    std::vector<at::IValue> input_arguments = {{{input_ivalue_arguments(args_exprs_list)}}};")
    sig_body.append(f"    std::vector<at::IValue> return_arguments = {{{return_ivalue_argument(f, returns_name)}}};")
    sig_body.append(f"    torch_gcu::op_record(op_schema, input_arguments, return_arguments);")
    sig_body.append(f"  }}")
    return sig_body

CANT_DUMP_LIST = [
    "_copy_from"
]

def cant_dump_args(f: NativeFunction) -> bool:
    if f"{f.func.name}" in CANT_DUMP_LIST:
        return True
    return False

def do_op_dump_body(f: NativeFunction, kernel: str) -> list:
    sig_body = []
    if cant_dump_args(f):
        return sig_body
    sig_body.append(f"  static bool enable_dump_input = cfg.enableDumpInput(\"{kernel}\");")
    sig_body.append(f"  static bool disable_dump_input = cfg.disableDumpInput(\"{kernel}\");")
    sig_body.append(f"  static bool enable_dump_output = cfg.enableDumpOutput(\"{kernel}\");")
    sig_body.append(f"  static bool disable_dump_output = cfg.disableDumpOutput(\"{kernel}\");")
    sig_body.append(f"  bool dump_input = enable_dump_input && (!disable_dump_input);")
    sig_body.append(f"  bool dump_output = enable_dump_output && (!disable_dump_output);")
    sig_body.append(f"  std::string op_path;")
    return sig_body

def op_dump_input_body(f: NativeFunction, kernel: str, args_exprs_str: str) -> list:
    sig_body = []
    if cant_dump_args(f):
        return sig_body
    sig_body.append(f"  if (dump_input) {{")
    sig_body.append(f"    auto file_path = cfg.getOpDumpPath();")
    sig_body.append(f"    auto time_stamp = torch_gcu::util::GetTimeStampMillis();")
    sig_body.append(f"    op_path = file_path + \"/\" + std::string(\"{kernel}\") + \"_\" + time_stamp + \"_\";")
    sig_body.append(f"    std::string input_path = op_path + \"input\";")
    sig_body.append(f"    torch_gcu::dump_args(input_path, 0, {args_exprs_str});")
    sig_body.append(f"  }}")
    return sig_body

def op_dump_output_body(f: NativeFunction, kernel: str, return_name: str) -> list:
    sig_body = []
    if cant_dump_args(f):
        return sig_body
    extra_outputs = get_extra_outputs(f)
    returns_num = len(f.func.returns)
    return_args = [args.name for args in extra_outputs]
    if returns_num > 0:
        return_args = [return_name] + return_args
    return_args_str = ", ".join(return_args)

    sig_body.append(f"  if (dump_output) {{")
    sig_body.append(f"    if (op_path.empty()) {{")
    sig_body.append(f"      auto file_path = cfg.getOpDumpPath();")
    sig_body.append(f"      auto time_stamp = torch_gcu::util::GetTimeStampMillis();")
    sig_body.append(f"      op_path = file_path + \"/\" + std::string(\"{kernel}\") + \"_\" + time_stamp + \"_\";")
    sig_body.append(f"    }}")
    sig_body.append(f"    std::string output_path = op_path + \"output\";")
    sig_body.append(f"    torch_gcu::dump_args(output_path, 0, {return_args_str});")
    sig_body.append(f"  }}")
    return sig_body

def call_with_statistics_body(f: NativeFunction, impl_name: str, args_exprs_str: str, args_exprs_list: List[Expr]) -> list:
    sig_body = []
    kernel = impl_name.split("::")[-1]
    return_num = len(f.func.returns)
    return_name = "gcu_out"
    k = f.func.kind()

    return_type = ""
    if k is SchemaKind.functional or return_num > 1:
        return_type = "auto"
    else:
        return_type = "auto&"

    sig_body.append("auto& cfg = torch_gcu::OpDebugConfig::GetInstance();")

    is_inplace = k is SchemaKind.inplace

    return_type_name = ""
    return_value = ""
    if is_inplace:
        sig_body.extend(op_statistics_input_body(args_exprs_list))

    sig_body.extend(do_op_dump_body(f, kernel))
    sig_body.extend(op_dump_input_body(f, kernel, args_exprs_str))

    if return_num > 0:
        return_type_name = f"{return_type} {return_name} = "
        return_value = f" {return_name}"

    sig_body.append(f"  {return_type_name}{impl_name}({args_exprs_str});")
    sig_body.extend(op_statistics_output_body(f, return_name) if is_inplace else op_statistics_body(f, args_exprs_list, return_name))
    sig_body.extend(op_dump_output_body(f, kernel, return_name))

    sig_body.append(f"  return{return_value};")

    return sig_body

def gen_unstructured_gcu(
    self, f: NativeFunction, g: Optional[NativeFunctionsGroup] = None
) -> Optional[str]:
    with native_function_manager(f):
        inplace_meta = False
        gets_out_inplace_wrapper = False
        if not self.backend_index.has_kernel(f):
            if (
                self.backend_index.dispatch_key == DispatchKey.Meta
                and f.func.kind() is SchemaKind.inplace
                and
                # Defer to composites for meta implementation
                not f.has_composite_kernel
                and
                # Inplace list operations are not supported
                len(f.func.returns) == 1
            ):
                inplace_meta = True
            elif (
                not self.backend_index.use_out_as_primary
                and g is not None
                and gets_generated_out_inplace_wrapper(f, g, self.backend_index)
            ):
                # We want to generate inplace/out wrappers, that don't have a kernel for the backend.
                gets_out_inplace_wrapper = True
            else:
                return None
        if f.manual_kernel_registration:
            return None

        if (
            self.target is Target.REGISTRATION
            and not self.selector.is_native_function_selected(f)
        ):
            return None

        sig = self.wrapper_kernel_sig(f)

        name = sig.name()
        returns_type = sig.returns_type().cpp_type()
        args = sig.arguments()
        args_str = ", ".join(a.defn() for a in args)

        # See Note [Direct dispatch bindings]
        cpp_sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=False
        )

        # TODO: dedupe this with the structured codegen
        if self.target is Target.NAMESPACED_DECLARATION:
            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += f"TORCH_API {cpp_sig.decl()};\n"
            return result
        elif self.target is Target.NAMESPACED_DEFINITION:

            def generate_defn(cpp_sig: CppSignature) -> str:
                return f"""
{cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""

            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += generate_defn(cpp_sig)
            return result

        elif self.target is Target.ANONYMOUS_DEFINITION:
            # short circuit for inplace_meta
            if inplace_meta:
                assert f.func.arguments.self_arg is not None
                self_arg_name = f.func.arguments.self_arg.argument.name
                # TODO: handle in place on tensor list
                return f"""
{returns_type} {name}({args_str}) {{
  TORCH_CHECK_NOT_IMPLEMENTED({self_arg_name}.is_meta(),
    "Cannot inplace into non-meta tensor with meta tensor argument");
  return {self_arg_name};
}}
"""

            # short circuit for generated inplace/out wrappers
            if gets_out_inplace_wrapper:
                return self.gen_out_inplace_wrapper(f, g)

            metadata = self.backend_index.get_kernel(f)
            if metadata is None:
                return None
            if self.class_method_name is None:
                impl_name = f"{metadata.cpp_namespace}::{metadata.kernel}"
            else:
                impl_name = f"{metadata.cpp_namespace}::{self.class_method_name}::{metadata.kernel}"

            kernel_sig = kernel_signature(f, self.backend_index)

            args_exprs_list = translate(sig.arguments(), kernel_sig.arguments(), method=False)
            args_exprs_str = ", ".join(
                e.expr
                for e in args_exprs_list
            )

            device_check = "  // No device check\n"

            is_privateuse1 = (
                self.backend_index.dispatch_key == DispatchKey.PrivateUse1
                or self.backend_index.dispatch_key == DispatchKey.AutogradPrivateUse1
            )
            gcu_device_check = is_privateuse1 and f"{f.func.name}" not in no_device_check_list
            # Backends that require device guards presumably also require device checks.
            if self.backend_index.device_guard or gcu_device_check:
                device_check_args = itertools.chain(
                    f.func.arguments.out, f.func.arguments.flat_positional
                )
                device_check = gen_device_check(
                    f.device_check, list(device_check_args), name
                )

            device_guard = "// DeviceGuard omitted"  # default
            gcu_enable_guard = is_privateuse1 and not f.structured and f"{f.func.name}" not in no_device_gurad_list
            if (f.device_guard and self.backend_index.device_guard) or gcu_enable_guard:
                has_tensor_options = any(
                    isinstance(a, TensorOptionsArguments)
                    for a in f.func.arguments.non_out
                )
                if has_tensor_options:
                    # kernel is creating a tensor
                    device_guard = """
  const DeviceGuard device_guard(device_or_default(device));"""

                    # CUDA requires special handling
                    if is_cuda_dispatch_key(self.backend_index.dispatch_key):
                        device_guard = (
                            f"globalContext().lazyInitCUDA();\n{device_guard}"
                        )
                    elif is_privateuse1:
                        device_guard = (
                            f"at::globalContext().lazyInitDevice(at::kPrivateUse1);\n{device_guard}"
                        )
                else:
                    # kernel is operating on existing tensors

                    # There is precedence for which argument we use to do
                    # device guard.  This describes the precedence order.
                    self_arg = (
                        [f.func.arguments.self_arg.argument]
                        if f.func.arguments.self_arg is not None
                        else []
                    )
                    candidate_args = itertools.chain(
                        self_arg,
                        f.func.arguments.out,
                        f.func.arguments.flat_positional,
                    )

                    # Only tensor like arguments are eligible
                    device_of = next(
                        (
                            f"{a.name}"
                            for a in candidate_args
                            if a.type.is_tensor_like()
                        ),
                        None,
                    )
                    if device_of is not None:
                        device_guard = f"const OptionalDeviceGuard device_guard(device_of({device_of}));"

            if is_privateuse1:
                call_body = "\n".join(call_with_statistics_body(f, impl_name, args_exprs_str, args_exprs_list))
            else:
                call_body = f"return {impl_name}({args_exprs_str});"
            return f"""\
namespace {{

{returns_type} {name}({args_str}) {{
  {device_check}

  {device_guard}
  {call_body}
}}

}} // anonymous namespace
"""

        elif self.target is Target.REGISTRATION:
            if f.manual_kernel_registration or self.skip_dispatcher_op_registration:
                return None
            else:
                payload = f"TORCH_FN({name})"
                if f.namespace == "aten":
                    return f'm.impl("{f.func.name}",\n{payload});\n'
                else:
                    return f'm.impl(TORCH_SELECTIVE_NAME("{f.namespace}::{f.func.name}"),\n{payload});\n'
        else:
            assert_never(self.target)

setattr(dest.RegisterDispatchKey, "gen_unstructured", gen_unstructured_gcu)
