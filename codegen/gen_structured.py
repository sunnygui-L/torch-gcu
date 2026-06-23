import textwrap

from typing import List, Dict, Sequence, Tuple, Union
from gen_aot_ops import gen_one_structured
import torchgen.api.meta as meta
from torchgen.model import (
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    SchemaKind,
)

from torchgen.utils import (
    concatMap,
    assert_never,
)

from torchgen.context import native_function_manager

def gen_class_set_output_body(dispatch_key: DispatchKey, k: SchemaKind, maybe_create_proxy: bool) -> str:
    if dispatch_key in [
        DispatchKey.PrivateUse1,
        DispatchKey.CompositeExplicitAutogradNonFunctional,
    ]:
        maybe_set_guard = """
auto current_device = guard_.current_device();
if (C10_UNLIKELY(current_device.has_value())) {
  TORCH_INTERNAL_ASSERT(*current_device == options.device(),
    "structured kernels don't support multi-device outputs");
} else {
  guard_.reset_device(options.device());
}
"""
        maybe_set_guard_line = maybe_set_guard + "\n"
    else:
        maybe_set_guard_line = maybe_set_guard = ""

    if maybe_create_proxy:
        create_proxy = """
auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
if (C10_UNLIKELY(maybe_proxy.has_value())) {
    proxy_outputs_[output_idx] = std::move(maybe_proxy).value();
}
"""
    else:
        create_proxy = ""

    if k is SchemaKind.functional:
        assert dispatch_key in (
            DispatchKey.PrivateUse1,
            DispatchKey.Meta,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
        )
        return f"""{maybe_set_guard_line}
outputs_[output_idx] = create_out(sizes, strides, options);"""
    elif k is SchemaKind.inplace:
        return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
check_inplace(out, sizes, options);
{create_proxy}"""
    elif k is SchemaKind.out:
        return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
resize_out(out, sizes, strides, options);
{create_proxy}"""
    elif k is SchemaKind.mutable or k is SchemaKind.scratch:
        raise AssertionError(
            f"{k} structured operators are currently not supported"
        )
    else:
        assert_never(k)

def gen_class_set_output_functions(
        dispatch_key: DispatchKey, k: SchemaKind, parent_class: str, generate_super: bool
) -> str:
    if generate_super:
        set_output_super = f"{parent_class}::set_output_raw_strided(output_idx, sizes, strides, options, names);"
    else:
        set_output_super = ""

    def gen_set_output_function(name: str, maybe_create_proxy: bool) -> str:
        return f"""
void set_output_{name}(
    int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
    at::TensorOptions options, at::DimnameList names
) override {{
{textwrap.indent(gen_class_set_output_body(dispatch_key, k, maybe_create_proxy), "    ")}
    if (!names.empty()) {{
      at::namedinference::propagate_names(outputs_[output_idx], names);
    }}
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
{textwrap.indent(set_output_super, "    ")}
}}
"""

    return f"""
{gen_set_output_function("strided", maybe_create_proxy=True)}
{gen_set_output_function("raw_strided", maybe_create_proxy=False)}
"""

# returns the definition of a ctor, as well as how to construct
# this class to a variable named op
def gen_class_ctor(k: SchemaKind, class_name: str, returns: int) -> str:
    if k is SchemaKind.functional:
        return ""
    elif k is SchemaKind.inplace:
        # TODO: Make sure out argument is guaranteed to be self
        return f"{class_name}(at::Tensor& self) : outputs_{{std::ref(self)}} {{}}"
    elif k is SchemaKind.out:
        out_args = ", ".join(f"at::Tensor& out{i}" for i in range(returns))
        out_refs = ", ".join(f"std::ref(out{i})" for i in range(returns))
        return f"{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}"
    elif k is SchemaKind.mutable or k is SchemaKind.scratch:
        raise AssertionError(
            f"{k} structured operators are currently not supported"
        )
    else:
        assert_never(k)

def gen_class(
    dispatch_key: DispatchKey,
    f: NativeFunction,
    k: SchemaKind,
    *,
    class_name: str,
    parent_class: str,
    generate_super: bool,
) -> str:
    # if k is SchemaKind.functional:
    #     output_type = "at::Tensor"
    #     output_value = "outputs_[output_idx]"
    #     proxy_field = ""
    # elif k is SchemaKind.inplace:
    #     output_type = "std::reference_wrapper<at::Tensor>"
    #     output_value = "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"
    #     proxy_field = f"std::array<c10::optional<at::Tensor>, {len(f.func.returns)}> proxy_outputs_;"
    # elif k is SchemaKind.out:
    #     output_type = "std::reference_wrapper<at::Tensor>"
    #     output_value = "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"
    #     proxy_field = f"std::array<c10::optional<at::Tensor>, {len(f.func.returns)}> proxy_outputs_;"
    if k is SchemaKind.functional:
        output_type = "at::Tensor"
        output_value = "outputs_[output_idx]"
        proxy_field = ""
    elif k is SchemaKind.inplace:
        output_type = "std::reference_wrapper<at::Tensor>"
        output_value = "outputs_[output_idx].get()"
        proxy_field = f"std::array<c10::optional<at::Tensor>, {len(f.func.returns)}> proxy_outputs_;"
    elif k is SchemaKind.out:
        output_type = "std::reference_wrapper<at::Tensor>"
        output_value = "outputs_[output_idx].get()"
        proxy_field = f"std::array<c10::optional<at::Tensor>, {len(f.func.returns)}> proxy_outputs_;"

    if dispatch_key == DispatchKey.PrivateUse1:
        guard_field = "torch_gcu::OptionalGCUGuard guard_;"
    elif (
        dispatch_key
        == DispatchKey.CompositeExplicitAutogradNonFunctional
    ):
        guard_field = "c10::OptionalDeviceGuard guard_;"
    else:
        guard_field = ""

    indent = " " * 4
    class_ctor_str = gen_class_ctor(k, class_name, len(f.func.returns))
    lines = (
        f"struct {class_name} final : public {parent_class} {{",
        f"{textwrap.indent(class_ctor_str, indent)}",
        f"{textwrap.indent(gen_class_set_output_functions(dispatch_key, k, parent_class, generate_super), indent)}",
        "    const at::Tensor& maybe_get_output(int64_t output_idx) override {",
        f"      return {output_value};\n",
        "    }",
        f"    std::array<{output_type}, {len(f.func.returns)}> outputs_;",
        f"{textwrap.indent(proxy_field, indent)}",
        f"{textwrap.indent(guard_field, indent)}",
        "};",
    )
    struct = "\n".join(line for line in lines if line)
    return "\n".join(line for line in struct.split('\n') if line.strip() != '')

def gen_backend_shape_infer_class(
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
) -> List[str]:

    class_list = []
    for g in grouped_native_functions:
        native_functions_map: Dict[OperatorName, NativeFunction] = {
            f.func.name: f
            for f in concatMap(
                lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
                [g],
            )
        }
        for f in native_functions_map:
            if g.structured and isinstance(g, NativeFunctionsGroup):
                k = native_functions_map[f].func.kind()
                class_name = f"structured_{meta.name(g)}_gcu_{k.name}"
                parent_class = f"at::meta::structured_{meta.name(g)}"
                f_class = gen_class(DispatchKey.PrivateUse1, native_functions_map[f], k, class_name=class_name, parent_class=parent_class, generate_super=g.out.structured_inherits is not None)
                class_list.append({"func": f"{f}", "class": f_class})

    class_list.sort(key=lambda x: x['func'])

    return [c['class'] for c in class_list]

def gen_structured_shape_infer(
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
) -> Tuple[List[str], List[str]]:

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
            if g.structured and isinstance(g, NativeFunctionsGroup):
                func = native_functions_map[f]
                with native_function_manager(func):
                    op_defn, op_impl = gen_one_structured(g, func, True)
                    ops.append(
                        {"func": f"{f}", "op_defn": op_defn, "impl": op_impl}
                    )

    ops.sort(key=lambda x: x["func"])
    return [op["op_defn"] for op in ops], [op["impl"] for op in ops],
