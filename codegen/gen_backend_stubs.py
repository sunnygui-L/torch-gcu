import argparse
import os
import pathlib
import re
from collections import Counter, defaultdict, namedtuple
from typing import Dict, List, Optional, Sequence, Set, Union, Callable

import yaml
import gen_structured
import gen_topsaten_bridge
import gen_aot_ops
import gen_GCUNativeFunctions_methods
import gen_op_class
import gen_op_fallback_register

import torchgen
from torchgen.api import cpp
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
import torchgen.dest as dest
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import native_function_manager
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    SchemaKind,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
    concatMap,
    context,
    FileManager,
    NamespaceHelper,
    Target,
)

import wrapper_torch_codegen

def rename_privateuse1_dispatch_key():
    # rename DispatcherKey about PrivateUse1
    custom_backend = "GCU"
    def PrivateUse1Str(self):
        return self.name.replace("PrivateUse1", custom_backend)

    @staticmethod
    def parse(value: str) -> "DispatchKey":
        for k, v in DispatchKey.__members__.items():
            if k == value.replace(custom_backend, "PrivateUse1"):
                return v
        raise AssertionError(f"unknown dispatch key {value}")

    DispatchKey.__str__ = PrivateUse1Str
    DispatchKey.parse = parse

def get_namespace(f:Union[NativeFunction, NativeFunctionsGroup]):
    if isinstance(f, NativeFunction):
        return f.namespace
    elif isinstance(f, NativeFunctionsGroup):
        return f.functional.namespace
    else:
        return "aten"

# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping)
ParsedExternalYaml = namedtuple(
    "ParsedExternalYaml",
    ['true_backend', "backend_key", "autograd_key", "class_name", "cpp_namespace", "backend_indices"],
)


def parse_backend_yaml(
    backend_yaml_path: str,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_indices: Dict[DispatchKey, BackendIndex],
) -> ParsedExternalYaml:

    native_functions_map: Dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(
            lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
            grouped_native_functions,
        )
    }

    with open(backend_yaml_path, "r") as f:
        yaml_values = yaml.safe_load(f)
    assert isinstance(yaml_values, dict)

    valid_keys = [
        "backend",
        "class_name",
        "cpp_namespace",
        "extra_headers",
        "supported",
        "autograd",
        "full_codegen",
        "non_native",
        "ir_gen",
        "symint",
        "optional",
    ]

    # backend = yaml_values.pop("backend", None)
    # assert backend is not None, 'You must provide a value for "backend"'
    yaml_backend = yaml_values.pop('backend', None)
    true_backend = 'PrivateUse1' if yaml_backend == 'GCU' else yaml_backend
    assert true_backend is not None, 'You must provide a value for "backend"'
    backend = "GCU"

    class_name = yaml_values.pop("class_name", None)

    cpp_namespace = yaml_values.pop("cpp_namespace", None)
    assert cpp_namespace is not None, 'You must provide a value for "cpp_namespace"'

    # Mostly just defaulting to false to stick with LazyTensor convention.
    use_out_as_primary = yaml_values.pop("use_out_as_primary", False)
    assert isinstance(
        use_out_as_primary, bool
    ), f"You must provide either True or False for use_out_as_primary. Provided: {use_out_as_primary}"

    use_device_guard = yaml_values.pop("device_guard", False)
    assert isinstance(
        use_device_guard, bool
    ), f"You must provide either True or False for device_guard. Provided: {use_device_guard}"

    supported_cfg = yaml_values.pop("supported", [])
    if supported_cfg is None:
        supported_cfg = []  # Allow an empty list of supported ops

    supported = [op["name"] for op in supported_cfg if not op.get("disable", False)]

    for op in supported_cfg:
        if op.get("topsaten_name", "") != "":
            gen_aot_ops.special_name_transfer.update({op["name"]: op["topsaten_name"]})
        if not op.get("device_check", True):
            wrapper_torch_codegen.no_device_check_list.append(op["name"])
        if not op.get("device_guard", True):
            wrapper_torch_codegen.no_device_gurad_list.append(op["name"])
        if op.get("support_deterministic_algorithms", False):
            gen_aot_ops.support_deterministic_algorithms_list.append(op["name"])

    assert isinstance(
        supported, list
    ), f'expected "supported" to be a list, but got: {supported} (of type {type(supported)})'

    symint = yaml_values.pop("symint", [])
    if symint is None:
        symint = []  # Allow an empty list of symint ops
    assert isinstance(
        symint, list
    ), f'expected "symint" to be a list, but got: {supported} (of type {type(supported)})'
    symint_set = set(symint)

    supported_autograd = yaml_values.pop("autograd", [])
    assert isinstance(
        supported_autograd, list
    ), f'expected "autograd" to be a list, but got: {supported_autograd}'

    # full_codegen is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    full_codegen = yaml_values.pop("full_codegen", [])
    supported.extend(full_codegen)

    # non_native is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    non_native = yaml_values.pop("non_native", {})

    # ir_gen is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    _ = yaml_values.pop("ir_gen", {})

    optional_cfg = yaml_values.pop("optional", [])
    if optional_cfg is None:
        optional_cfg = []  # Allow an empty list of optional ops
    optional = [op["name"] for op in optional_cfg if not op.get("disable", False)]

    for op in optional_cfg:
        if op.get("topsaten_name", "") != "":
            gen_aot_ops.special_name_transfer.update({op["name"]: op["topsaten_name"]})
        if not op.get("device_check", True):
            wrapper_torch_codegen.no_device_check_list.append(op["name"])
        if not op.get("device_guard", True):
            wrapper_torch_codegen.no_device_gurad_list.append(op["name"])
        if op.get("support_deterministic_algorithms", False):
            gen_aot_ops.support_deterministic_algorithms_list.append(op["name"])

    assert isinstance(
        optional, list
    ), f'expected "optional" to be a list, but got: {optional}'

    assert (
        len(yaml_values.keys()) == 0
    ), f'{backend_yaml_path} contains unexpected keys: {", ".join(yaml_values.keys())}. \
Only the following keys are supported: {", ".join(valid_keys)}'

    def create_backend_index(
        backend_ops: List[str],
        symint_ops: Set[str],
        dispatch_key: DispatchKey,
        *,
        use_out_as_primary: bool,
        use_device_guard: bool,
    ) -> BackendIndex:
        metadata: Dict[OperatorName, BackendMetadata] = {}
        for op in backend_ops:
            op_name = OperatorName.parse(op)
            assert (
                op_name in native_functions_map
            ), f"Found an invalid operator name: {op_name}"
            # See Note [External Backends Follow Dispatcher API]
            kernel_name = dispatcher.name(native_functions_map[op_name].func)
            if op in symint_ops:
                kernel_name += "_symint"
            # TODO: allow structured external backends later.
            m = BackendMetadata(
                kernel=kernel_name, structured=False, cpp_namespace=cpp_namespace
            )
            metadata[op_name] = m
        return BackendIndex(
            dispatch_key=dispatch_key,
            use_out_as_primary=use_out_as_primary,
            external=True,
            device_guard=use_device_guard,
            index=metadata,
        )

    backend_key: Optional[DispatchKey] = None
    if len(supported) > 0:
        with context(
            lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'
        ):
            backend_key = DispatchKey.parse(backend)

        backend_idx = create_backend_index(
            supported,
            symint_set,
            backend_key,
            use_out_as_primary=use_out_as_primary,
            use_device_guard=use_device_guard,
        )
        assert backend_key not in backend_indices
        backend_indices[backend_key] = backend_idx

    autograd_key: Optional[DispatchKey] = None
    if len(supported_autograd) > 0:
        with context(
            lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'
        ):
            autograd_key = DispatchKey.parse(f"Autograd{backend}")

        autograd_idx = create_backend_index(
            supported_autograd,
            symint_set,
            autograd_key,
            use_out_as_primary=use_out_as_primary,
            use_device_guard=use_device_guard,
        )
        assert autograd_key not in backend_indices
        backend_indices[autograd_key] = autograd_idx

    optional_key: Optional[DispatchKey] = None
    if len(optional) > 0:
        with context(
            lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'
        ):
            optional_key = DispatchKey.parse(backend)

        optional_idx = create_backend_index(
            optional,
            symint_set,
            optional_key,
            use_out_as_primary=use_out_as_primary,
            use_device_guard=use_device_guard,
        )

        assert f"Optional{backend}" not in backend_indices
        backend_indices[f"Optional{backend}"] = optional_idx

    for g in grouped_native_functions:
        if isinstance(g, NativeFunction):
            forward_kernels = (
                []
                if backend_key is None
                else [
                    m
                    for m in [backend_indices[backend_key].get_kernel(g)]
                    if m is not None
                ]
            )
            backward_kernels = (
                []
                if autograd_key is None
                else [
                    m
                    for m in [backend_indices[autograd_key].get_kernel(g)]
                    if m is not None
                ]
            )
        else:
            forward_kernels = (
                []
                if backend_key is None
                else [
                    m
                    for m in [
                        backend_indices[backend_key].get_kernel(f)
                        for f in g.functions()
                    ]
                    if m is not None
                ]
            )
            backward_kernels = (
                []
                if autograd_key is None
                else [
                    m
                    for m in [
                        backend_indices[autograd_key].get_kernel(f)
                        for f in g.functions()
                    ]
                    if m is not None
                ]
            )

        forward_kernels = [f for f in forward_kernels if f is not None]
        backward_kernels = [f for f in backward_kernels if f is not None]
        assert (
            len(forward_kernels) == 0 or len(backward_kernels) == 0
        ), f'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s \
autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! \
{forward_kernels[0].kernel} is listed under "supported", but {backward_kernels[0].kernel} is listed under "autograd".'

    return ParsedExternalYaml(
        true_backend, backend_key, autograd_key, class_name, cpp_namespace, backend_indices
    )


def error_on_missing_kernels(
    native_functions: Sequence[NativeFunction],
    backend_indices: Dict[DispatchKey, BackendIndex],
    backend_key: DispatchKey,
    autograd_key: Optional[DispatchKey],
    class_name: str,
    kernel_defn_file_path: str,
    kernel_defn_gen_file_path: List[str],
    full_codegen: Optional[List[OperatorName]] = None,
    extra_native_functions: List[Sequence[NativeFunction]] = [],
    extra_backend_indices: List[Dict[DispatchKey, BackendIndex]] = [],
    extra_backend_keys: List[DispatchKey] = [],
    extra_autograd_keys: List[Optional[DispatchKey]] = [],
) -> None:
    try:
        with open(kernel_defn_file_path, "r") as f:
            backend_defns = f.read()
        for path in kernel_defn_gen_file_path:
            with open(path, "r") as f:
                backend_gen_defns = f.read()
            backend_defns += backend_gen_defns
    except IOError as e:
        raise AssertionError(
            f"Unable to read from the specified impl_path file: {kernel_defn_file_path}"
        ) from e

    if full_codegen is None:
        full_codegen = []

    indices = [backend_indices[backend_key].index] + (
        [] if autograd_key is None else [backend_indices[autograd_key].index]
    )

    for ex_backend_indices, ex_backend_keys, ex_autograd_keys in zip(
        extra_backend_indices, extra_backend_keys, extra_autograd_keys
    ):
        indices = (
            indices
            + [ex_backend_indices[ex_backend_keys].index]
            + (
                []
                if ex_autograd_keys not in ex_backend_indices
                else [ex_backend_indices[ex_autograd_keys].index]
            )
        )

    for extra_native_function in extra_native_functions:
        native_functions = native_functions + extra_native_function

    # Quick mapping from each OperatorName used by the external backend
    # to its backend kernel name
    expected_backend_op_names: Dict[OperatorName, str] = dict(
        list(
            concatMap(
                lambda index: [
                    (op_name, metadata.kernel) for op_name, metadata in index.items()
                ],
                indices,
            )
        )
    )
    expected_backend_native_funcs: List[NativeFunction] = [
        f
        for f in native_functions
        if f.func.name in expected_backend_op_names.keys()
        and f.func.name not in full_codegen
    ]
    expected_backend_kernel_name_counts: Dict[str, List[NativeFunction]] = defaultdict(
        list
    )
    for native_f in expected_backend_native_funcs:
        expected_backend_kernel_name_counts[
            expected_backend_op_names[native_f.func.name]
        ].append(native_f)

    # This just looks for lines containing "foo(", and assumes that the kernel foo has been implemented.
    # It might cause false negatives (we won't catch all cases), but that's ok - if we catch a missing kernel
    # here, then we get a nicer error message. If we miss it, you get a linker error.
    kernel_defn_regex = rf"(.*){class_name}::\s*([\w\d]*)\([^\)]*\)\s*{{"
    actual_backend_kernel_name_counts = Counter(
        # A bit unwieldy (this could probably be moved into regex),
        # but we don't want to include kernel names that come from function calls,
        # like "return torch_gcu::GCUNativeFunctions::empty_strided_symint(...)".
        # Easy check is to ignore any lines with colons before the class name.
        [
            y
            for (x, y) in re.findall(kernel_defn_regex, backend_defns)
            if not x.endswith(":")
        ]
    )

    missing_kernels_err_msg = ""
    for expected_name, funcs in expected_backend_kernel_name_counts.items():
        expected_overload_count = len(funcs)
        actual_overload_count = actual_backend_kernel_name_counts[expected_name]
        if expected_overload_count != actual_overload_count:

            def create_decl(f: NativeFunction) -> str:
                with native_function_manager(f):
                    return DispatcherSignature.from_schema(f.func).decl()

            expected_schemas_str = "\n".join([create_decl(f) for f in funcs])
            missing_kernels_err_msg += f"""
{class_name} is missing a kernel definition for {expected_name}. We found {actual_overload_count} kernel(s) with that name,
but expected {expected_overload_count} kernel(s). The expected function schemas for the missing operator are:
{expected_schemas_str}

"""
    assert missing_kernels_err_msg == "", missing_kernels_err_msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backend stub files")
    parser.add_argument(
        "-s",
        "--source-yaml",
        "--source_yaml",
        help="path to source yaml file containing operator external definitions",
    )
    parser.add_argument("-o", "--output-dir", "--output_dir", help="output directory")
    parser.add_argument(
        "--dry-run", "--dry_run", type=bool, default=False, help="output directory"
    )
    parser.add_argument(
        "--impl-path",
        "--impl_path",
        type=str,
        default=None,
        help="path to the source C++ file containing kernel definitions",
    )
    parser.add_argument(
        "--vision-yaml-path",
        "--vision_yaml_path",
        type=str,
        default=None,
        help="path to torch vision yaml file containing operator external definitions",
    )
    parser.add_argument(
        "--vision-source-yaml",
        "--vision_source_yaml",
        type=str,
        default=None,
        help="path to torch vision gcu supported yaml file",
    )
    options = parser.parse_args()

    run(options.source_yaml, options.output_dir, options.dry_run, options.impl_path,
        vision_native_yaml=options.vision_yaml_path, vision_source_yaml=options.vision_source_yaml)


def gen_dispatchkey_nativefunc_headers(
    fm: FileManager,
    class_name: str,
    cpp_namespace: str,
    backend_indices: Dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_dispatch_key: DispatchKey,
    autograd_dispatch_key: Optional[DispatchKey],
    backend_name: str = "",
    extra_backend_indices: List[Dict[DispatchKey, BackendIndex]] = [],
    extra_grouped_functions: List[Sequence[Union[NativeFunction, NativeFunctionsGroup]]] = [],
) -> None:
    assert class_name is not None
    generated_comment = (
        "Autogenerated file by gen_backend_stubs.py. Do not edit directly!"
    )

    # Convert to a set first to remove duplicate kernel names.
    # Backends are allowed to repeat kernel names; only generate the declaration once!
    # Sort for deterministic output.
    backend_declarations = sorted(
        set(
            concatMap(
                lambda f: dest.compute_native_function_declaration(
                    f, backend_indices[backend_dispatch_key]
                ),
                grouped_native_functions,
            )
        )
    )
    autograd_declarations = sorted(
        set(
            concatMap(
                lambda f: []
                if autograd_dispatch_key is None
                else dest.compute_native_function_declaration(
                    f, backend_indices[autograd_dispatch_key]
                ),
                grouped_native_functions,
            )
        )
    )
    optional_declarations = sorted(
        set(
            concatMap(
                lambda f: []
                if f"Optional{backend_dispatch_key}" not in backend_indices
                else dest.compute_native_function_declaration(
                    f, backend_indices[f"Optional{backend_dispatch_key}"]
                ),
                grouped_native_functions,
            )
        )
    )

    extra_declarations=[]
    for backend_indice, grouped_function in zip(extra_backend_indices, extra_grouped_functions):
        declarations = sorted(
            set(
                concatMap(
                    lambda f: []
                    if backend_dispatch_key not in backend_indice
                    else dest.compute_native_function_declaration(
                        f, backend_indice[backend_dispatch_key]
                    ),
                    grouped_function,
                )
            )
        )
        extra_declarations.extend(declarations)


    ns_helper = NamespaceHelper(cpp_namespace)
    fm.write_with_template(
        f"{backend_dispatch_key}NativeFunctions.h",
        "DispatchKeyNativeFunctions.h",
        lambda: {
            "generated_comment": generated_comment,
            "namespace_prologue": ns_helper.prologue,
            "class_name": class_name,
            "namespace_epilogue": ns_helper.epilogue,
            "dispatch_declarations": backend_declarations + optional_declarations + autograd_declarations + extra_declarations,
            "BackendName": backend_name,
            "DispatchKey": backend_dispatch_key,
        },
    )


def gen_dispatcher_registrations(
    fm: FileManager,
    output_dir: str,
    class_name: str,
    backend_indices: Dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_dispatch_key: DispatchKey,
    dispatch_key: DispatchKey,
    selector: "SelectiveBuilder",
    # build_in_tree is true for lazy TS backend and affects include paths, not used for external backends
    dispatch_key_name: str,
    register_dispatch_key_func: Callable,
    build_in_tree: bool = False,
    per_operator_headers: bool = False,
    backend_name: str = "",
    eager_registration: bool = True,
    optional_op: bool = False,
    namespace: str = "aten",
    external_backend_headers_str = ""
) -> None:
    headers = [
        f"{output_dir}/{backend_dispatch_key}NativeFunctions.h",
    ]
    if build_in_tree:
        external_backend_headers_str = f"{external_backend_headers_str}\n" + "\n".join(f"#include <{h}>" for h in headers)
    else:
        external_backend_headers_str = f"{external_backend_headers_str}\n" + "\n".join(f'#include "{h}"' for h in headers)

    assert class_name is not None
    backend_index = (
        backend_indices[f"Optional{backend_dispatch_key}"]
        if optional_op
        else backend_indices[dispatch_key]
    )

    dispatch_registrations_body = list()
    if optional_op:
        for grouped_native_function in grouped_native_functions:
            dispatch_registrations_functions = dest.RegisterDispatchKey(
                backend_index,
                Target.REGISTRATION,
                selector,
                rocm=False,
                symint=True,
                class_method_name=f"{class_name}",
                skip_dispatcher_op_registration=False,
            )(grouped_native_function)

            if len(dispatch_registrations_functions) > 0:
                if isinstance(grouped_native_function, NativeFunctionsGroup):
                    root_name = grouped_native_function.functional.root_name
                elif isinstance(grouped_native_function, NativeFunction):
                    root_name = grouped_native_function.root_name

                NoRegister = f'if (!torch_gcu::OpDebugConfig::GetInstance().isDeregister("{root_name}")) '

                for i in range(len(dispatch_registrations_functions)):
                    fun = dispatch_registrations_functions[i]
                    dispatch_registrations_functions[i] = NoRegister + " {\n" + fun + "}"

                dispatch_registrations_body = (
                    dispatch_registrations_body + dispatch_registrations_functions
                )
    else:
        dispatch_registrations_body = list(
            concatMap(
                dest.RegisterDispatchKey(
                    backend_index,
                    Target.REGISTRATION,
                    selector,
                    rocm=False,
                    symint=True,
                    class_method_name=f"{class_name}",
                    skip_dispatcher_op_registration=False,
                ),
                grouped_native_functions,
            )
        )

    newline = "\n"
    ns_helper = NamespaceHelper(namespace_str="at")
    deferred_dispatch_registrations = ""
    static_init_dispatch_registrations = ""
    if eager_registration:
        static_template = CodeTemplate(
            f"""\
TORCH_LIBRARY_IMPL({namespace.lower()}, $dispatch_key, m) {{
    $dispatch_registrations_body
}};"""
        )
        static_init_dispatch_registrations = static_template.substitute(
            dispatch_key=dispatch_key_name,
            dispatch_registrations_body=dispatch_registrations_body,
        )
    else:
        deferred_template = CodeTemplate(
            f"""\
TORCH_API void Register${{backend_name}}${{dispatch_key}}NativeFunctions() {{
    static auto m = MAKE_TORCH_LIBRARY_IMPL({namespace.lower()}, $dispatch_key);
    $dispatch_registrations_body
}}"""
        )
        deferred_dispatch_registrations = deferred_template.substitute(
            backend_name=backend_name,
            dispatch_key=dispatch_key,
            dispatch_registrations_body=dispatch_registrations_body,
        )

    optional = "Optional" if optional_op else ""
    external_backend_headers_str = (
        external_backend_headers_str +
        '\n#include <ATen/core/dispatch/Dispatcher.h>' +
        '\n#include <ATen/Operators.h>' +
        '\n#include "aten/aot_ops/gcu_opcheck_utils.h"' +
        '\n#include "aten/op_statistics.h"' +
        '\n#include "aten/op_debug_config.h"' +
        '\n#include "gcu/sys_util.h"'
    )
    filename = f"Register{optional}{dispatch_key}.cpp" if namespace == "aten" else f"Register{namespace.title()}{dispatch_key}.cpp"

    fm.write_with_template(
        filename,
        "RegisterDispatchKey.cpp",
        lambda: {
            "extra_cuda_headers": "",
            "external_backend_headers": external_backend_headers_str,
            "ops_headers": "",
            "DispatchKey": dispatch_key,
            "dispatch_namespace": dispatch_key.lower(),
            "dispatch_headers": dest.gen_registration_headers(
                backend_index, per_operator_headers=per_operator_headers, rocm=False
            ),
            "dispatch_helpers": dest.gen_registration_helpers(backend_index),
            "dispatch_definitions": fm.substitute_with_template(
                "RegisterDispatchDefinitions.ini",
                lambda: {
                    "ns_prologue": ns_helper.prologue,
                    "ns_epilogue": ns_helper.epilogue,
                    "static_init_dispatch_registrations": static_init_dispatch_registrations,
                    "deferred_dispatch_registrations": deferred_dispatch_registrations,
                    "dispatch_namespace": dispatch_key.lower(),
                    "dispatch_namespaced_definitions": "",
                    "dispatch_anonymous_definitions": list(
                        concatMap(
                            dest.RegisterDispatchKey(
                                backend_index,
                                Target.ANONYMOUS_DEFINITION,
                                selector,
                                rocm=False,
                                symint=True,
                                class_method_name=f"{class_name}",
                                skip_dispatcher_op_registration=False,
                            ),
                            grouped_native_functions,
                        )
                    ),
                },
            ).split(newline),
        },
    )


def run(
    source_yaml: str, output_dir: str, dry_run: bool, impl_path: Optional[str] = None,
    vision_native_yaml: str = None, vision_source_yaml: str = None
) -> None:
    rename_privateuse1_dispatch_key()

    torchgen_root = pathlib.Path(torchgen.__file__).parent.absolute()
    template_dir = os.path.join(torchgen_root, "packaged/ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(
        torchgen_root, "packaged/ATen/native/native_functions.yaml"
    )
    tags_yaml_path = os.path.join(torchgen_root, "packaged/ATen/native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    grouped_native_functions = get_grouped_native_functions(native_functions)
    parsed_backend_yaml = parse_backend_yaml(
        source_yaml, grouped_native_functions, backend_indices
    )
    true_backend = parsed_backend_yaml.true_backend
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    class_name = parsed_backend_yaml.class_name
    backend_indices = parsed_backend_yaml.backend_indices
    selector = SelectiveBuilder.get_nop_selector()
    codegen_root = pathlib.Path(gen_GCUNativeFunctions_methods.__file__).parent.absolute()
    codegen_template_dir = os.path.join(codegen_root, "templates")
    limited_fallback_op_list = []
    limited_grouped_functions_list = []
    fallback_yaml_path_list = []
    limited_grouped_functions_list.append(grouped_native_functions)
    limited_fallback_op_list.append(gen_op_fallback_register.get_op_list(source_yaml))
    fallback_yaml_path_list.append(os.path.join(codegen_root, "gcu_op_fallback.yaml"))

    if backend_key is None:
        # This could be useful if a backend wants to quickly set up a noop yaml file but doesn't have any kernels ready yet.
        return

    if class_name is None:
        # class_name is an optional argument to backend yaml file.
        # if specified it allows an external backend to override
        # the name of the class that all generated kernel definitions live under.
        # if not specified, its value is given as native_function_class_name.
        class_name = backend_indices[backend_key].native_function_class_name()
    assert class_name is not None

    GCUNativeFunctions_methods_fm = FileManager(
        install_dir=output_dir, template_dir=codegen_template_dir, dry_run=dry_run
    )

    op_impls = gen_GCUNativeFunctions_methods.gen_GCUNativeFunctions_methods(source_yaml, grouped_native_functions)

    GCUNativeFunctions_methods_fm.write_with_template(
        "GCUNativeFunctions_gen.cpp",
        "GCUNativeFunctions.cpp",
        lambda: {
            "extra_headers": "",
            "op_impl": "".join(op_impl for op_impl in op_impls),
        },
    )

    impl_gen_path = os.path.join(output_dir, "GCUNativeFunctions_gen.cpp")
    impl_gen_path_list = [impl_gen_path]

    extra_source_yamls=[]
    extra_native_functions=[]
    extra_grouped_functions=[]
    extra_backend_indices=[]
    extra_backend_keys=[]
    extra_autograd_keys=[]
    extra_op_class_headers=[]
    if vision_native_yaml and vision_source_yaml:
        vision_yaml = parse_native_yaml(vision_native_yaml, tags_yaml_path)
        vision_native_functions, vision_backend_indices = (
            vision_yaml.native_functions,
            vision_yaml.backend_indices,
        )

        grouped_vision_functions = get_grouped_native_functions(vision_native_functions)
        parsed_vision_backend_yaml = parse_backend_yaml(
            vision_source_yaml, grouped_vision_functions, vision_backend_indices)

        vision_backend_key = parsed_backend_yaml.backend_key
        vision_autograd_key = parsed_backend_yaml.autograd_key
        vision_class_name = parsed_vision_backend_yaml.class_name
        vision_backend_indices = parsed_vision_backend_yaml.backend_indices
        extra_source_yamls.append(vision_source_yaml)
        extra_grouped_functions.append(grouped_vision_functions)
        extra_native_functions.append(vision_native_functions)
        extra_backend_indices.append(vision_backend_indices)
        extra_backend_keys.append(vision_backend_key)
        extra_autograd_keys.append(vision_autograd_key)

        if bool(vision_backend_indices[vision_backend_key].index):

            if vision_class_name is None:
                # class_name is an optional argument to backend yaml file.
                # if specified it allows an external backend to override
                # the name of the class that all generated kernel definitions live under.
                # if not specified, its value is given as native_function_class_name.
                vision_class_name = vision_backend_indices[vision_backend_key].native_function_class_name()
            assert vision_class_name is not None
            vision_namespace = get_namespace(grouped_vision_functions[0])

            op_class = gen_op_class.gen_op_class(vision_source_yaml, grouped_vision_functions)

            op_class_file_name = f"{vision_namespace.title()}Operators.h"
            GCUNativeFunctions_methods_fm.write_with_template(
                op_class_file_name,
                "Operators.h",
                lambda: {
                    "Operators_includes": "",
                    "namespace_defn": vision_namespace.upper(),
                    "namespace": vision_namespace.lower(),
                    "Operators_declarations": op_class,
                },
            )

            vision_op_impls = gen_GCUNativeFunctions_methods.gen_GCUNativeFunctions_methods(vision_source_yaml, grouped_vision_functions)

            extra_vision_op_headers = f'#include "aten/{op_class_file_name}"'
            extra_op_class_headers.append(extra_vision_op_headers)
            gen_file_name = f"GCU{vision_namespace.title()}Functions_gen.cpp"
            GCUNativeFunctions_methods_fm.write_with_template(
                gen_file_name,
                "GCUNativeFunctions.cpp",
                lambda: {
                    "extra_headers": extra_vision_op_headers,
                    "op_impl": "".join(op_impl for op_impl in vision_op_impls),
                },
            )
            vision_impl_gen_path = os.path.join(output_dir, gen_file_name)
            impl_gen_path_list.append(vision_impl_gen_path)

            limited_grouped_functions_list.append(grouped_vision_functions)
            limited_fallback_op_list.append(gen_op_fallback_register.get_op_list(vision_source_yaml))
            fallback_yaml_path_list.append(None)

    if impl_path is not None:
        error_on_missing_kernels(
            native_functions,
            backend_indices,
            backend_key,
            autograd_key,
            class_name,
            impl_path,
            impl_gen_path_list,
            extra_native_functions=extra_native_functions,
            extra_backend_indices=extra_backend_indices,
            extra_backend_keys=extra_backend_keys,
            extra_autograd_keys=extra_autograd_keys
        )

    gen_dispatchkey_nativefunc_headers(
        fm,
        class_name,
        cpp_namespace,
        backend_indices,
        grouped_native_functions,
        backend_key,
        autograd_key,
        extra_backend_indices=extra_backend_indices,
        extra_grouped_functions=extra_grouped_functions
    )

    for dispatch_key in (
        [backend_key] if autograd_key is None else [backend_key, autograd_key]
    ):
        gen_dispatcher_registrations(
            fm,
            output_dir,
            class_name,
            backend_indices,
            grouped_native_functions,
            backend_key,
            dispatch_key,
            selector,
            dispatch_key_name=dispatch_key.name.replace("GCU", true_backend),
            register_dispatch_key_func=dest.RegisterDispatchKey,
        )

    if f"Optional{backend_key}" in backend_indices:
        class_name = backend_indices[f"Optional{backend_key}"].native_function_class_name()

        gen_dispatcher_registrations(
            fm,
            output_dir,
            class_name,
            backend_indices,
            grouped_native_functions,
            backend_key,
            backend_key,
            selector,
            dispatch_key_name=dispatch_key.name.replace("GCU", true_backend),
            register_dispatch_key_func=dest.RegisterDispatchKey,
            optional_op=True
        )

    extra_topsaten_bridge_impls = []
    extra_op_defns = []
    extra_op_impls = []
    extra_unstructured_shape_infer = []
    extra_foreach_op_check_slow_path_str = []
    extra_foreach_op_slow_path_str = []
    for ex_source_yaml, ex_grouped_functions, ex_backend_indices, ex_backend_key, ex_autograd_key, extra_op_class_header in zip(extra_source_yamls, extra_grouped_functions, extra_backend_indices, extra_backend_keys, extra_autograd_keys, extra_op_class_headers):
        for dispatch_key in (
            [ex_backend_key] if ex_autograd_key is None else [ex_autograd_key, ex_autograd_key]
        ):
            namespace = get_namespace(ex_grouped_functions[0])
            gen_dispatcher_registrations(
                fm,
                output_dir,
                class_name,
                ex_backend_indices,
                ex_grouped_functions,
                ex_backend_key,
                dispatch_key,
                selector,
                dispatch_key_name=dispatch_key.name.replace("GCU", true_backend),
                register_dispatch_key_func=dest.RegisterDispatchKey,
                namespace=namespace,
                external_backend_headers_str=extra_op_class_header
            )

        ex_topsaten_bridge_impls = gen_topsaten_bridge.gen_topsaten_bridge(ex_source_yaml, ex_grouped_functions)
        extra_topsaten_bridge_impls.extend(ex_topsaten_bridge_impls)
        ex_op_defns, ex_op_impls, ex_unstructured_shape_infer, ex_foreach_op_check_slow_path_str, ex_foreach_op_slow_path_str = gen_aot_ops.gen_aot_ops(
            ex_source_yaml, ex_grouped_functions)
        extra_op_defns.extend(ex_op_defns)
        extra_op_impls.extend(ex_op_impls)
        extra_unstructured_shape_infer.extend(ex_unstructured_shape_infer)
        extra_foreach_op_check_slow_path_str.extend(ex_foreach_op_check_slow_path_str)
        extra_foreach_op_slow_path_str.extend(ex_foreach_op_slow_path_str)

    shape_infer_fm = FileManager(
        install_dir=os.path.join(output_dir, "shape_inference"), template_dir=codegen_template_dir, dry_run=dry_run
    )

    class_list = gen_structured.gen_backend_shape_infer_class(grouped_native_functions)

    extra_structured_headers = '#include "aten/shape_inference/gcu_structured.h"'
    shape_infer_fm.write_with_template(
        f"{dispatch_key.lower()}_structured_shape_infer.h",
        "structured_shape_infer.h",
        lambda: {
            "extra_headers": extra_structured_headers,
            "structured_definitions": "\n".join(structured + "\n" for structured in class_list),
        },
    )

    topsaten_bridge_fm = FileManager(
        install_dir=os.path.join(output_dir, "aot_ops"), template_dir=codegen_template_dir, dry_run=dry_run
    )

    topsaten_bridge_impls = gen_topsaten_bridge.gen_topsaten_bridge(source_yaml, grouped_native_functions)
    topsaten_bridge_impls.extend(extra_topsaten_bridge_impls)
    topsaten_bridge_fm.write_with_template(
        "topsaten_bridge.h",
        "topsaten_bridge.h",
        lambda: {
            "extra_headers": "",
            "op_impls": "".join(op_impl + "\n" for op_impl in topsaten_bridge_impls),
        },
    )

    aotops_fm = FileManager(
        install_dir=os.path.join(output_dir, "aot_ops"), template_dir=codegen_template_dir, dry_run=dry_run
    )

    op_defns, op_impls, unstructured_shape_infer, foreach_op_check_slow_path_str, foreach_op_slow_path_str = gen_aot_ops.gen_aot_ops(
        source_yaml, grouped_native_functions)
    op_defns = op_defns + extra_op_defns
    op_impls = op_impls + extra_op_impls
    unstructured_shape_infer = unstructured_shape_infer + extra_unstructured_shape_infer
    foreach_op_check_slow_path_str = foreach_op_check_slow_path_str + "\n\n".join(extra_foreach_op_check_slow_path_str)
    foreach_op_slow_path_str = foreach_op_slow_path_str + "\n\n".join(extra_foreach_op_slow_path_str)
    GCUNativeFunctions_methods_fm.write_with_template(
        "foreach_op_utils.h",
        "foreach_op_utils.h",
        lambda: {
            "extra_headers": "",
            "op_definitions": foreach_op_check_slow_path_str + "\n\n" + foreach_op_slow_path_str,
        },
    )

    #fallback register
    fallback_register = gen_op_fallback_register.register_fallback_ops(
        limited_fallback_op_list,
        fallback_yaml_path_list, limited_grouped_functions_list)
    GCUNativeFunctions_methods_fm.write_with_template(
        "register_fallback_ops_gen.cpp",
        "empty_template.cpp",
        lambda: {
            "extra_headers": '#include "aten/register_fallback_ops.h"',
            "namespace": "torch_gcu",
            "body": fallback_register,
        },
    )

    aotops_fm.write_with_template(
        f"{dispatch_key.lower()}_aot_ops.h",
        "aot_ops.h",
        lambda: {
            "extra_headers": "",
            "op_definitions": "\n".join(op_defn + "\n" for op_defn in op_defns),
        },
    )

    extra_body="""
#define BINARY_KERNEL_LAUNCH_LHS_IS_SCALAR(topsatenop, out, lhs, rhs, \\
                                           alpha...)                  \\
  if (is_cpu_scalar(lhs) && !is_cpu_scalar(rhs)) {                    \\
    auto xlhs = scalarTensorToTopsatenScalar(lhs);                    \\
    bridge_##topsatenop##_out1(out, xlhs, rhs, ##alpha);              \\
  } else

#define BINARY_KERNEL_LAUNCH_RHS_IS_SCALAR(topsatenop, out, lhs, rhs, \\
                                           alpha...)                  \\
  if (!is_cpu_scalar(lhs) && is_cpu_scalar(rhs)) {                    \\
    auto xrhs = scalarTensorToTopsatenScalar(rhs);                    \\
    bridge_##topsatenop##_out1(out, lhs, xrhs, ##alpha);              \\
  } else

#define BINARY_KERNEL_LAUNCH_END(topsatenop, out, lhs, rhs, alpha...) \\
  { bridge_##topsatenop##_out1(out, lhs, rhs, ##alpha); }
"""

    extra_aotops_headers="""
#include "aten/aot_ops/gcu_aot_ops.h"

#include <ATen/core/op_registration/adaption.h>
#include <iostream>

#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/gcu_structured_shape_infer.h"
#include "aten/shape_inference/shape_infer_func.h"
#include "aten/foreach_op_utils.h"
"""

    aotops_fm.write_with_template(
        f"{dispatch_key.lower()}_aot_ops.cpp",
        "aot_ops.cpp",
        lambda: {
            "extra_headers": extra_aotops_headers,
            "extra_body": extra_body,
            "op_impl": "\n".join(op_impl + "\n" for op_impl in op_impls),
        },
    )

    structured_shape_infer_op_defns, structured_shape_infer_impls = gen_structured.gen_structured_shape_infer(grouped_native_functions)

    shape_infer_op_defns = structured_shape_infer_op_defns + unstructured_shape_infer

    shape_infer_fm.write_with_template(
        "aotops_shape_infer_func.h",
        "aot_ops.h",
        lambda: {
            "extra_headers": "#include <ATen/core/Reduction.h>",
            "op_definitions": "\n".join(defn + "\n" for defn in shape_infer_op_defns),
        },
    )

    structured_shape_infer_headers="""
#include "aten/shape_inference/shape_infer_func.h"
#include <ATen/core/op_registration/adaption.h>
#include "aten/shape_inference/gcu_structured_shape_infer.h"
"""

    shape_infer_fm.write_with_template(
        "structured_shape_infer_func.cpp",
        "aot_ops.cpp",
        lambda: {
            "extra_headers": structured_shape_infer_headers,
            "extra_body": "",
            "op_impl": "\n".join(defn + "\n" for defn in structured_shape_infer_impls),
        },
    )

    exclude_header = ["shape_infer_func.h", "gcu_structured.h"]
    shape_infer_dir=os.path.join(output_dir, "shape_inference")
    head_files_list = [file_name for file_name in os.listdir(shape_infer_dir) if file_name.endswith(".h") and file_name not in exclude_header]
    head_files_list.sort()
    head_files_list = [f"#include \"csrc/aten/shape_inference/{file_name}\"" for file_name in head_files_list]

    shape_infer_fm.write_with_template(
        "shape_infer_func.h",
        "shape_infer_func.h",
        lambda: {
            "headers": "\n".join(head for head in head_files_list),
        },
    )

if __name__ == "__main__":
    main()
