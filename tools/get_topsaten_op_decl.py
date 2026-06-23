import argparse
import os
import re
import pathlib
import torchgen

from dataclasses import dataclass
from typing import Optional, Sequence, Union

from torchgen.gen import parse_native_yaml
from torchgen.model import (
    Argument,
    Arguments,
    BaseTy,
    NativeFunction,
    Return,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.utils import FileManager

# A type representing a topsaten class
@dataclass(frozen=True)
class TopsAtenType(Type):
    class_name: str

    def __str__(self) -> str:
        """
        Return the class name will prefix __torch__.torch.classes
        """
        return f"{self.class_name}"

    def is_base_ty_like(self, base_ty: BaseTy) -> bool:
        return False

    def is_symint_like(self) -> bool:
        return False

    def is_nullable(self) -> bool:
        """
        Assume a custom class is not nullable.
        """
        return False

    def is_list_like(self) -> Optional["ListType"]:
        return None

ATEN_TO_TOPATEN_TYPE = {
    "Tensor":"topsatenTensor",
    "Tensor?":"topsatenTensor",
    "Tensor[]":"std::vector<topsatenTensor>",
    "Tensor?[]":"std::vector<topsatenTensor>",
    "Scalar":"topsatenScalar_t",
    "Scalar?":"topsatenScalar_t",
    "Scalar[]":"std::vector<topsatenScalar_t>",
    "ScalarType":"topsatenDataType_t",
    "ScalarType?":"topsatenDataType_t",
    "Layout":"topsatenLayoutType_t",
    "Layout?":"topsatenLayoutType_t",
    "Device":"NoNeed",
    "Device?":"NoNeed",
    "DeviceIndex":"int8_t",
    "MemoryFormat":"topsatenMemoryFormat_t",
    "MemoryFormat?":"topsatenMemoryFormat_t",
    "Generator?":"topsatenPhiloxState_t",
    "QScheme":"uint8_t",
    "str":"const char *const",
    "str?":"const char *const",
    "Dimname":"int64_t",
    "Dimname[]?":"topsatenSize_t",
    "Dimname[]":"topsatenSize_t",
    "Dimname[1]":"topsatenSize_t",
    "bool":"bool",
    "bool?":"topsatenScalar_t",
    "bool[2]":"std::array<bool,2>",
    "bool[3]":"std::array<bool,3>",
    "bool[4]":"std::array<bool,4>",
    "int":"int64_t",
    "int?":"topsatenScalar_t",
    "int[]":"topsatenSize_t",
    "int[1]":"topsatenSize_t",
    "int[2]":"topsatenSize_t",
    "int[3]":"topsatenSize_t",
    "int[]?":"topsatenSize_t",
    "int[1]?":"topsatenSize_t",
    "int[2]?":"topsatenSize_t",
    "SymInt":"int64_t",
    "SymInt?":"topsatenScalar_t",
    "SymInt[]":"topsatenSize_t",
    "SymInt[1]":"topsatenSize_t",
    "SymInt[2]":"topsatenSize_t",
    "SymInt[3]":"topsatenSize_t",
    "SymInt[4]":"topsatenSize_t",
    "SymInt[5]":"topsatenSize_t",
    "SymInt[6]":"topsatenSize_t",
    "SymInt[]?":"topsatenSize_t",
    "SymInt[1]?":"topsatenSize_t",
    "float":"double",
    "float?":"topsatenScalar_t",
    "float[]?":"std::vector<double>",
}

BASIC_TYPE = [
    "int8_t",
    "uint8_t",
    "const char *const",
    "int64_t",
    "bool",
    "double",
    "std::array<bool,2>",
    "std::array<bool,3>",
    "std::array<bool,4>",
]

TOPS_ATEN_TENSOR_TYPE = [
    "Tensor",
    "Tensor?",
    "Tensor[]",
    "Tensor?[]",
]

TOPS_ATEN_INT_TYPE = [
    "DeviceIndex",
    "QScheme",
    "Dimname",
    "Dimname[]?",
    "Dimname[]",
    "Dimname[1]",
    "int",
    "int?",
    "int[]",
    "int[1]",
    "int[2]",
    "int[3]",
    "int[]?",
    "int[1]?",
    "int[2]?",
    "SymInt",
    "SymInt?",
    "SymInt[]",
    "SymInt[1]",
    "SymInt[2]",
    "SymInt[3]",
    "SymInt[4]",
    "SymInt[5]",
    "SymInt[6]",
    "SymInt[]?",
    "SymInt[1]?",
]

TOPS_ATEN_FLOAT_TYPE = [
    "float",
    "float?",
    "float[]?",
]

TOPS_ATEN_BOOL_TYPE = [
    "bool",
    "bool?",
    "bool[2]",
    "bool[3]",
    "bool[4]",
]

TORCH_ATEN_SCALAR_TYPE = [
    "Scalar",
    "Scalar?",
    "Scalar[]",
]

ATEN_TO_CPP_BOOL = {
    "False": "false",
    "True": "true",
}

ATEN_TO_TOPATEN_MEMFORMAT = {
    "None":"topsatenMemoryFormat_t::TOPSATEN_MEMORY_NONE",
    "contiguous_format":"topsatenMemoryFormat_t::TOPSATEN_MEMORY_CONTIGUOUS",
    "channelsLast_format":"topsatenMemoryFormat_t::TOPSATEN_MEMORY_NHWC",
    "channelsLast3d_format":"topsatenMemoryFormat_t::TOPSATEN_MEMORY_NDHWC",
    "preserve_format":"topsatenMemoryFormat_t::TOPSATEN_MEMORY_PRESERVE",
}

def skip_func(f:NativeFunction) -> bool:
    args = f.func.arguments.flat_all
    for arg in args:
        if f"{arg.type}" == "Storage" or f"{arg.type}" == "Stream":
            return True
    if "cudnn" in f"{f.func.name}" or "cpu" in f"{f.func.name}" or "mkldnn" in f"{f.func.name}":
        return True
    return False

def aten_to_topsaten_default(torch_type: str, tops_type: str, default: str):
    vec_pattern = r'std::vector<[^>]+>'
    if tops_type == "bool":
         return ATEN_TO_CPP_BOOL[default]
    elif tops_type == "const char *const":
         if default == "None":
             return f"\"{default.lower()}\""
    elif default == "Mean":
         return "1"
    elif tops_type == "topsatenScalar_t":
        if default == "None":
            return "{TOPSATEN_DATA_NONE, {}}"
        elif torch_type in TOPS_ATEN_INT_TYPE:
            return f"{{TOPSATEN_DATA_I32, {{{default}}}}}"
        elif torch_type in TOPS_ATEN_FLOAT_TYPE or torch_type in TORCH_ATEN_SCALAR_TYPE:
            return f"{{TOPSATEN_DATA_FP32, {{{default}}}}}"
        elif torch_type in TOPS_ATEN_BOOL_TYPE:
            return f"{{TOPSATEN_DATA_PRED, {{{default.lower()}}}}}"
    elif tops_type == "topsatenTensor":
        if default == "None":
            return "{}"
    elif tops_type == "topsatenSize_t":
        if default == "None" or default == "[]":
            return "{}"
    elif tops_type == "topsatenMemoryFormat_t":
        return ATEN_TO_TOPATEN_MEMFORMAT[default]
    elif tops_type == "topsatenDataType_t":
        if default == "None":
            return "topsatenDataType_t::TOPSATEN_DATA_NONE"
    elif tops_type == "topsatenLayoutType_t":
        return "topsatenLayoutType_t::TOPSATEN_LAYOUT_STRIDED"
    elif tops_type == "topsatenGenerator_t":
        if default == "None":
            return "{}"
    elif tops_type == "topsatenPhiloxState_t":
        if default == "None":
            return "{}"
    elif re.findall(vec_pattern, tops_type):
        if default == "None":
            return "{}"
    return default

def aten_to_topsaten_argument(arg: Argument) -> Argument:
    type = ATEN_TO_TOPATEN_TYPE[f"{arg.type}"]
    default = aten_to_topsaten_default(f"{arg.type}", type, arg.default) if arg.default is not None else arg.default
    return Argument(name=arg.name, type=TopsAtenType(type), default=default, annotation=arg.annotation)

def aten_to_topsaten_selfarg(arg: SelfArgument) -> SelfArgument:
    argument = aten_to_topsaten_arg(arg.argument)
    return SelfArgument(argument)

def aten_to_topsaten_tensor_opt(arg: TensorOptionsArguments) -> TensorOptionsArguments:
    dtype = aten_to_topsaten_arg(arg.dtype)
    layout = aten_to_topsaten_arg(arg.layout)
    device = aten_to_topsaten_arg(arg.device)
    pin_memory = aten_to_topsaten_arg(arg.pin_memory)
    return TensorOptionsArguments(dtype, layout, device, pin_memory)

def get_input_decl(arg: Argument):
    if arg.default is not None:
        return f"const {arg.type}& {arg.name} = {arg.default}"
    else:
        return f"const {arg.type}& {arg.name}"

def get_input_decl(arg: SelfArgument):
    self_arg = arg.argument
    return get_input_decl(self_arg)

def aten_to_topsaten_return(ret: Return) -> Return:
    type = ATEN_TO_TOPATEN_TYPE[f"{ret.type}"]
    return Return(name=ret.name, type=TopsAtenType(type), annotation=ret.annotation)

def aten_to_topsaten_arg(arg: Union[Argument, SelfArgument, TensorOptionsArguments, Return]):
    if isinstance(arg, Argument):
        return aten_to_topsaten_argument(arg)
    elif isinstance(arg, SelfArgument):
        return aten_to_topsaten_selfarg(arg)
    elif isinstance(arg, TensorOptionsArguments):
        return aten_to_topsaten_tensor_opt(arg)
    elif isinstance(arg, Return):
        return aten_to_topsaten_return(arg)

def make_input_const(input:Union[Argument, SelfArgument, TensorOptionsArguments], with_name=True, with_default=True):
    if isinstance(input, SelfArgument):
        return make_input_const(input.argument, with_name, with_default)
    elif isinstance(input, TensorOptionsArguments):
        dtype = make_input_const(input.dtype, with_name, with_default)
        layout = make_input_const(input.layout, with_name, with_default)
        return ", ".join([dtype, layout])
    else:
        if f"{input.type}" in BASIC_TYPE:
            if with_name:
                decl = f"{input.type} {input.name}"
            else:
                decl = f"{input.type}"
        elif f"{input.type}" == "NoNeed":
            return ""
        elif f"{input.name}" == "pin_memory":
            return ""
        else:
            if with_name:
                decl = f"const {input.type} &{input.name}"
            else:
                decl = f"const {input.type} &"

        if input.default is not None and with_default:
            return f"{decl} = {input.default}"
        else:
            return decl

def aten_to_topsaten_name(name) -> str:
    kernel = ""
    names = name.split(".")
    base_name = names[0]
    names = "".join(e.capitalize() for e in base_name.split("_"))
    kernel = "topsaten" + names
    return kernel

def get_topsaten_inputs(f:NativeFunction) -> Sequence[Union[Argument, SelfArgument, TensorOptionsArguments]]:
    aten_args = f.func.arguments.flat_non_out
    topsaten_args = []
    for aten_arg in aten_args:
        topsaten_args.append(aten_to_topsaten_arg(aten_arg))
    return topsaten_args

def flat_non_self(args: Arguments) -> Sequence[Union[Argument, SelfArgument, TensorOptionsArguments]]:
    ret: list[Union[Argument, SelfArgument, TensorOptionsArguments]] = []
    ret.extend(args.pre_self_positional)
    ret.extend(args.post_self_positional)
    ret.extend(args.kwarg_only)
    ret.extend(args.out)
    return ret

def get_multi_outputs(f:NativeFunction) -> Sequence[Return]:
    outputs = []
    k = f.func.kind()
    if k is SchemaKind.functional:
        for _return in f.func.returns:
            outputs.append(aten_to_topsaten_arg(_return))
        for arg in f.func.arguments.flat_all:
            if arg.is_write:
                outname = arg.name
                if arg.name == "self":
                    outname = "output"
                outputs.append(aten_to_topsaten_arg(Return(outname, arg.type, arg.annotation)))
    elif k is SchemaKind.inplace:
        self_arg = f.func.arguments.self_arg
        outputs.append(aten_to_topsaten_arg(Return("output", self_arg.argument.type, self_arg.argument.annotation)))
        for arg in flat_non_self(f.func.arguments):
            if arg.is_write:
                outputs.append(aten_to_topsaten_arg(Return(arg.name, arg.type, arg.annotation)))
    elif k is SchemaKind.out:
        for out in f.func.arguments.out:
            outputs.append(aten_to_topsaten_arg(Return(out.name, out.type, out.annotation)))
        for arg in f.func.arguments.flat_non_out:
            if arg.is_write:
                outputs.append(aten_to_topsaten_arg(Return(arg.name, arg.type, arg.annotation)))
    elif k is SchemaKind.mutable:
        return_num = len(f.func.returns)
        if return_num == 0:
            for arg in f.func.arguments.flat_non_out:
                if arg.is_write:
                    outputs.append(aten_to_topsaten_arg(Return(arg.name, arg.type, arg.annotation)))
        else:
            for _return in f.func.returns:
                outputs.append(aten_to_topsaten_arg(_return))
            for arg in f.func.arguments.flat_non_out:
                if arg.is_write:
                    outputs.append(aten_to_topsaten_arg(Return(arg.name, arg.type, arg.annotation)))
    return outputs

def get_topsaten_op_decl(f:NativeFunction):
    is_random = True if "nondeterministic_seeded" in f.tags else False

    topsaten_name = aten_to_topsaten_name(f"{f.func.name}")
    inputs = get_topsaten_inputs(f)
    outputs = get_multi_outputs(f)
    # TODO where extra_random_state add and what it is
    extra_random_state = []
    if is_random:
        for arg in inputs:
            if f"{arg.type}" == "topsatenPhiloxState_t" or f"{arg.type}" == "topsatenGenerator_t":
                break
        else:
            type = TopsAtenType("topsatenPhiloxState_t")
            extra_random_state.append(Argument(name="philox_state", type=type, default="{}", annotation=None))

    if len(outputs) == 0:
        output_str = ""
        output_str_type_only = ""
    elif len(outputs) == 1:
        output = outputs[0]
        output_name = output.name if output.name is not None else "output"
        output_str = f"{output.type} &{output_name}, "
        output_str_type_only = f"{output.type} &, "
    else:
        output_types = ",".join([f"{output.type} &" for output in outputs ])
        output_str = f"std::tuple<{output_types}> output, "
        output_str_type_only = f"std::tuple<{output_types}>, "

    inputs_with_default = [make_input_const(input) for input in inputs + extra_random_state if make_input_const(input) != ""]
    inputs_type_only = [make_input_const(input, False, False) for input in inputs if make_input_const(input) != ""]

    input_str = ", ".join(inputs_with_default)
    input_str_type_only = ", ".join(inputs_type_only)

    decl = f"topsatenStatus_t TOPSATEN_EXPORT {topsaten_name}({output_str}{input_str}, topsStream_t stream = nullptr);"
    key = f"topsatenStatus_t TOPSATEN_EXPORT {topsaten_name}({output_str_type_only}{input_str_type_only}, topsStream_t stream = nullptr);"
    return key, decl

def gen_topsaten_decls():
    torchgen_root = pathlib.Path(torchgen.__file__).parent.absolute()

    native_yaml_path = os.path.join(
        torchgen_root, "packaged/ATen/native/native_functions.yaml"
    )
    tags_yaml_path = os.path.join(torchgen_root, "packaged/ATen/native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)

    native_functions = parsed_yaml.native_functions
    func_decls = dict()
    for native_function in native_functions:
        if skip_func(native_function):
            continue
        key, decl = get_topsaten_op_decl(native_function)
        if key in func_decls:
            pre_decl = func_decls[key]
            if len(decl) < len(pre_decl):
                continue
        func_decls[key] = decl

    return func_decls

def run(output_dir: str, template_dir = None, dry_run: bool=False) -> None:
    if template_dir == None:
        codegen_root = pathlib.Path(os.path.realpath(__file__)).parent.parent.absolute()
        template_dir = os.path.join(codegen_root, "codegen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    fm = make_file_manager(output_dir)

    topsaten_decls = gen_topsaten_decls()

    brief ="""
/**
 * @file topsaten_ops.h
 * @brief topsflame common aten ops api definitions.
 */
"""

    headers = """
#ifndef TOPSATEN_OPS_H_
#define TOPSATEN_OPS_H_

#include <tuple>
#include <vector>

#include "tops/tops_runtime.h"
#include "topsaten/topsaten_define.h"

#if defined(__cplusplus)
"""

    namespace = "topsaten"

    extra_end = """
#endif

#endif /* TOPSATEN_OPS_H_ */
"""
    fm.write_with_template(
        f"topsaten_ops.h",
        "topsaten_ops.h",
        lambda: {
            "brief": brief,
            "headers": headers,
            "namespace": namespace,
            "decl": "\n".join(decl + "\n" for key, decl in topsaten_decls.items()),
            "extra_end": extra_end,
        },
    )

# example python3.8 get_topsaten_op_decl.py -o ./
# torch must be install.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate aotops head file")
    parser.add_argument("-o", "--output-dir", "--output_dir", default="./topsaten_ops", help="output directory")
    options = parser.parse_args()
    run(options.output_dir)