from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torchgen.api.meta as meta
import torchgen.api.structured as structured
import yaml
from torchgen.api import cpp
from torchgen.api.translate import translate

from torchgen.api.types import (
    ArrayCType,
    ArrayRefCType,
    BaseCType,
    Binding,
    ConstRefCType,
    CType,
    DispatcherSignature,
    Expr,
    ListCType,
    MutRefCType,
    NamedCType,
    OptionalCType,
    scalarT,
    tensorT,
    TupleCType,
    VectorCType,
    VectorizedCType,
    voidT,
)
from torchgen.context import native_function_manager
from torchgen.model import (
    Argument,
    Arguments,
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    OptionalType,
    Return,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)

from torchgen.packaged.autograd.gen_inplace_or_view_type import ALL_VIEW_FUNCTIONS

from torchgen.utils import assert_never, concatMap

special_name_transfer = dict()

# List of operators that support deterministic algorithms
support_deterministic_algorithms_list = []

ALL_AMP_FUNCTIONS = [
    "_amp_foreach_non_finite_check_and_unscale_",
    "_amp_foreach_non_finite_check_and_unscale",
    "_amp_foreach_non_finite_check_and_unscale.out",
    "_amp_update_scale_",
    "_amp_update_scale",
    "_amp_update_scale.out",
]

RUNTIME_OP = [
    "_copy_from",
    "_copy_from_and_resize",
    "_local_scalar_dense",
    "_pin_memory",
    "_resize_output_",
    "empty.memory_format",
    "empty_strided",
    "record_stream",
    "resize_",
    "set_",
    "set_.source_Storage",
    "set_.source_Storage_storage_offset",
    "set_.source_Tensor",
]

NONE_KERNEL_OP = [
    "_has_compatible_shallow_copy_type",
    "is_pinned",
    "is_set_to",
    "zero_",
]

ATEN_DYANIMIC_OP = [
    "bincount",
    "bincount.out",
    "_ctc_loss",
    "_ctc_loss.out",
    "_ctc_loss.Tensor",
    "_ctc_loss.Tensor_out",
    "index.Tensor",
    "index.Tensor_out",
    "repeat_interleave.Tensor",
    "repeat_interleave.Tensor_out",
    "repeat_interleave.self_Tensor",
    "repeat_interleave.self_int",
    "one_hot",
    "unique_dim",
    "unique_dim.out",
    "unique_consecutive",
    "unique_consecutive.out",
    "unique_dim_consecutive",
    "unique_dim_consecutive.out",
    "_unique",
    "_unique.out",
    "_unique2",
    "_unique2.out",
    "masked_select",
    "masked_select.out",
    "nonzero",
    "nonzero.out",
    "argwhere",
    "linalg_lstsq",
    "linalg_lstsq.out",
]

TORCHVISION_DYANIMIC_OP = ["nms"]

ALL_DYANIMIC_OP = ATEN_DYANIMIC_OP + TORCHVISION_DYANIMIC_OP

# gelu only support tanh for approximate
# clamp need self.to(out.scalar_type())
# scatter aten string_view->enum topsaten
# sort aten c10::optional<bool>->bool topsaten
# native_dropout aten arg train is optional bool not bool topsaten
# argsort.stable need maybe_wrap_dim
# copysign.out do not have scalar topsaten interface
# cumprod.out not support without given dtype
TOPSATEN_WORKAROUND = [
    "clamp.Tensor_out",
    "clamp.out",
    "index.Tensor",
    "index.Tensor_out",
    "scatter.reduce_out",
    "scatter.src_out",
    "scatter.value_out",
    "scatter.value_reduce_out",
    "sort.values_stable",
    "argsort.stable",
    "copysign.out",
    "cumprod.out",
]

SPLIT_TOPSATEN_MULTI_OUT_OP = [
    "_linalg_solve_ex.result",
    "_cummax_helper",
    "frexp.Tensor_out",
    "max.dim_max",
    "max_pool2d_with_indices.out",
    "max_pool3d_with_indices",
    "max_pool3d_with_indices.out",
    "min.dim_min",
    "mode.values",
    "sort.values_stable",
    "topk.values",
    "_unique2",
    "native_batch_norm",
    "native_group_norm",
    "native_layer_norm",
    "_weight_norm_interface",
    "mode",
    "native_dropout",
    "nll_loss2d_forward",
    "nll_loss2d_forward.output",
    "std_mean.correction",
    "linalg_lu_factor_ex.out",
]

exclude = list(
    set(
        TOPSATEN_WORKAROUND
        + list(ALL_VIEW_FUNCTIONS.keys())
        + RUNTIME_OP
        + NONE_KERNEL_OP
        + ALL_DYANIMIC_OP
    )
)

# copy from torch/csrc/utils/python_arg_parser.cpp
ALLOW_NUMBERS_AS_TENSORS_LIST = [
    "add",
    "add_",
    "add_out",
    "div",
    "div_",
    "div_out",
    "divide",
    "divide_",
    "divide_out",  ## alias of div
    "mul",
    "mul_",
    "mul_out",
    "multiply",
    "multiply_",
    "multiply_out",  ## alias of mul
    "sub",
    "sub_",
    "sub_out",
    "subtract",
    "subtract_",
    "subtract_out",  ## alias of sub
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "to",
    "_to_copy",
    "copy_",
    "floor_divide",
    "floor_divide_",
    "floor_divide_out",
]

# assemble from aten/src/ATen/native/BinaryOps.cpp
BINARY_KERNEL_LHS_IS_SCALAR = [
    "special_chebyshev_polynomial_t",
    "special_chebyshev_polynomial_t.out",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_u.out",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_v.out",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w.out",
    "special_hermite_polynomial_h",
    "special_hermite_polynomial_h.out",
    "special_hermite_polynomial_he",
    "special_hermite_polynomial_he.out",
    "special_laguerre_polynomial_l",
    "special_laguerre_polynomial_l.out",
    "special_legendre_polynomial_p",
    "special_legendre_polynomial_p.out",
    "special_shifted_chebyshev_polynomial_t",
    "special_shifted_chebyshev_polynomial_t.out",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_u.out",
    "special_shifted_chebyshev_polynomial_v",
    "special_shifted_chebyshev_polynomial_v.out",
    "special_shifted_chebyshev_polynomial_w",
    "special_shifted_chebyshev_polynomial_w.out",
    "special_xlog1py",
    "special_xlog1py.out",
    "special_zeta",
    "special_zeta.out",
    "remainder.Tensor",
    "remainder.Tensor_out",
    "bitwise_and.Tensor",
    "bitwise_and.Tensor_out",
    "bitwise_or.Tensor",
    "bitwise_or.Tensor_out",
    "bitwise_xor.Tensor",
    "bitwise_xor.Tensor_out",
    "bitwise_left_shift.Tensor",
    "bitwise_left_shift.Tensor_out",
    "bitwise_right_shift.Tensor",
    "bitwise_right_shift.Tensor_out",
    "xlogy.OutTensor",
    "xlogy.Tensor",
]

# assemble from aten/src/ATen/native/BinaryOps.cpp
BINARY_KERNEL_RHS_IS_SCALAR = [
    "special_chebyshev_polynomial_t",
    "special_chebyshev_polynomial_t.out",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_u.out",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_v.out",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w.out",
    "special_hermite_polynomial_h",
    "special_hermite_polynomial_h.out",
    "special_hermite_polynomial_he",
    "special_hermite_polynomial_he.out",
    "special_laguerre_polynomial_l",
    "special_laguerre_polynomial_l.out",
    "special_legendre_polynomial_p",
    "special_legendre_polynomial_p.out",
    "special_shifted_chebyshev_polynomial_t",
    "special_shifted_chebyshev_polynomial_t.out",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_u.out",
    "special_shifted_chebyshev_polynomial_v",
    "special_shifted_chebyshev_polynomial_v.out",
    "special_shifted_chebyshev_polynomial_w",
    "special_shifted_chebyshev_polynomial_w.out",
    "special_xlog1py",
    "special_xlog1py.out",
    "special_zeta",
    "special_zeta.out",
    "_add_relu.Tensor",
    "_add_relu_.Tensor",
    "copysign.out",
    "copysign.Tensor",
    "copysign_.Tensor",
    "div.Tensor",
    "div.Tensor_mode",
    "div_.Tensor",
    "div_.Tensor_mode",
    "mul.out",
    "mul.Tensor",
    "sub.Tensor",
    "sub_.Tensor",
    "add.Tensor",
    "add_.Tensor",
    "remainder.Tensor",
    "remainder.Tensor_out",
    "remainder_.Tensor",
    "rsub.Tensor",
    "bitwise_and.Tensor",
    "bitwise_and.Tensor_out",
    "bitwise_and_.Tensor",
    "bitwise_or.Tensor",
    "bitwise_or.Tensor_out",
    "bitwise_or_.Tensor",
    "bitwise_xor.Tensor",
    "bitwise_xor.Tensor_out",
    "bitwise_xor_.Tensor",
    "__lshift__.Tensor",
    "bitwise_left_shift.Tensor",
    "bitwise_left_shift.Tensor_out",
    "bitwise_left_shift_.Tensor",
    "__rshift__.Tensor",
    "bitwise_right_shift.Tensor",
    "bitwise_right_shift.Tensor_out",
    "bitwise_right_shift_.Tensor",
    "logical_and",
    "logical_and.out",
    "logical_and_",
    "logical_or",
    "logical_or.out",
    "logical_or_",
    "logical_xor",
    "logical_xor.out",
    "logical_xor_",
    "floor_divide",
    "floor_divide.out",
    "fmod.Tensor",
    "fmod.Tensor_out",
    "fmod_.Tensor",
    "xlogy.OutTensor",
    "xlogy.Tensor",
    "xlogy_.Tensor",
]

GENERATOR_OP_WITH_TWO_OUT = [
    "_fused_dropout",
    "_fused_dropout.out",
]

GENERATOR_INPLACE_CREAT_OP = [
    "bernoulli_.Tensor",
    "bernoulli_.float",
    "random_.from",
    "random_.to",
    "random_",
    "uniform_",
    "cauchy_",
    "log_normal_",
    "exponential_",
    "geometric_",
    "normal_",
]

GENERATOR_OP_WITH_ONE_OUT = [
    *GENERATOR_INPLACE_CREAT_OP,
    "bernoulli",
    "bernoulli.out",
    "bernoulli.p",
    "rand.generator_with_names",
    "rand.generator",
    "rand.generator_out",
    "randint.generator",
    "randint.low_generator",
    "randint.generator_out",
    "randint.low_generator_out",
    "randn.generator",
    "randn.generator_with_names",
    "randn.generator_out",
    "randperm.generator",
    "randperm.generator_out",
    "rrelu",
    "rrelu_",
    "_standard_gamma",
    "_sample_dirichlet",
    "poisson",
    "binomial",
    "multinomial.out",
    "multinomial",
    "normal_functional",
    "normal.Tensor_float_out",
    "normal.Tensor_float",
    "normal.float_Tensor_out",
    "normal.float_Tensor",
    "normal.Tensor_Tensor_out",
    "normal.Tensor_Tensor",
    "normal.float_float",
    "normal.float_float_out",
    "rrelu_with_noise.out",
    "rrelu_with_noise",
    "rrelu_with_noise_",
    "bernoulli.Tensor_out",
    "bernoulli.Tensor",
    "bernoulli.float_out",
    "rand.generator_with_names_out",
    "randn.generator_with_names_out",
    "_standard_gamma.out",
    "_sample_dirichlet.out",
    "poisson.out",
    "binomial.out",
    "random.from_out",
    "random.from",
    "random.to_out",
    "random.to",
    "random.out",
    "random",
    "uniform.out",
    "uniform",
    "cauchy.out",
    "cauchy",
    "log_normal.out",
    "log_normal",
    "exponential.out",
    "exponential",
    "geometric.out",
    "geometric",
    "normal.out",
]

INPLACE_OP_UNCARE_INPUT = [
    *GENERATOR_INPLACE_CREAT_OP,
    "fill_.Scalar",
    "fill_.Tensor",
]

RETURN_HOST_TYPE_LIST = [
    "uint8_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "double",
    "float",
    "bool",
    "SymInt",
]


def is_return_host_type(rtype: CType) -> bool:
    if isinstance(rtype, BaseCType):
        if rtype.type.name in RETURN_HOST_TYPE_LIST:
            return True
        return False
    if isinstance(rtype, ConstRefCType):
        return is_return_host_type(rtype.elem)
    if isinstance(rtype, VectorCType):
        return is_return_host_type(rtype.elem)
    if isinstance(rtype, ArrayCType):
        return is_return_host_type(rtype.elem)
    if isinstance(rtype, TupleCType):
        for elem in rtype.elems:
            if is_return_host_type(elem):
                return True
        return False
    if isinstance(rtype, MutRefCType):
        return is_return_host_type(rtype.elem)
    if isinstance(rtype, OptionalCType):
        return is_return_host_type(rtype.elem)
    if isinstance(rtype, ListCType):
        return is_return_host_type(rtype.elem)
    if isinstance(rtype, ArrayRefCType):
        return is_return_host_type(rtype.elem)
    if isinstance(rtype, VectorizedCType):
        return is_return_host_type(rtype.elem)
    assert False, f"unsupported type for return type {type(rtype)}"


def has_host_returns_value(f: NativeFunction) -> str:
    returns_type = cpp.returns_type(f.func.returns)
    if is_return_host_type(returns_type):
        return "true"
    return "false"


def get_multi_out_input_output(aotops_args_names: str, out_num: int) -> Tuple[str, str]:
    args_list = aotops_args_names.split(", ")
    output = args_list[:out_num]
    input = args_list[out_num:]
    return ", ".join(input), f'std::forward_as_tuple({", ".join(output)})'


def shape_infer_decl(
    f: NativeFunction,
    args: Sequence[Binding],
    name: str,
) -> str:
    returns_type = cpp.returns_type(f.func.returns).cpp_type()
    cpp_args = [a.decl() for a in args]
    cpp_args_str = ", ".join(cpp_args)
    return f"{returns_type} {name}({cpp_args_str})"


def argument(a: Union[Argument, SelfArgument, TensorOptionsArguments]) -> List[Binding]:
    if isinstance(a, Argument):
        return [
            Binding(
                nctype=structured.argument_type(a, binds=a.name),
                name=a.name,
                default=None,
                argument=a,
            )
        ]
    elif isinstance(a, SelfArgument):
        return argument(a.argument)
    elif isinstance(a, TensorOptionsArguments):
        # raise AssertionError("structured kernels don't support TensorOptions yet")
        dtype = argument(a.dtype)
        layout = argument(a.layout)
        device = argument(a.device)
        pin_memory = argument(a.pin_memory)
        return dtype + layout + device + pin_memory
    else:
        assert_never(a)


tensor_arg_type = ["Tensor", "Tensor[]"]


def get_type(
    arg: Union[Argument, SelfArgument, TensorOptionsArguments, Return],
) -> Type:
    if isinstance(arg, Argument):
        return arg.type
    elif isinstance(arg, SelfArgument):
        return arg.argument.type
    elif isinstance(arg, TensorOptionsArguments):
        assert arg.dtype.type
    elif isinstance(arg, Return):
        return arg.type
    pass


def get_name(arg: Union[Argument, SelfArgument, TensorOptionsArguments, Return]) -> str:
    if isinstance(arg, Argument):
        return arg.name
    elif isinstance(arg, SelfArgument):
        return arg.argument.name
    elif isinstance(arg, TensorOptionsArguments):
        assert "TensorOptions"
    elif isinstance(arg, Return):
        return arg.name
    pass


def get_first_tensor_out(
    f: NativeFunction, structured: bool = False
) -> Tuple[str, str]:
    k = f.func.kind()
    device_arg = None
    if k is SchemaKind.functional:
        for i, arg in enumerate(f.func.returns):
            if f"{arg.type}" in tensor_arg_type:
                if structured:
                    return_name = f"op.outputs_[{i}]"
                else:
                    if len(f.func.returns) == 1:
                        return_name = f"result"
                    else:
                        return_name = f"result{i}"
                device_arg = Return(
                    name=return_name,
                    type=arg.type,
                    annotation=arg.annotation,
                )
                break
    elif k is SchemaKind.inplace:
        for arg in f.func.arguments.all:
            if isinstance(arg, SelfArgument):
                device_arg = arg.argument
                break
    elif k is SchemaKind.out:
        for arg in f.func.arguments.out:
            if f"{arg.type}" in tensor_arg_type:
                device_arg = arg
                break

    return device_arg.type, device_arg.name


def unstructured_aotops_arguments(f: NativeFunction) -> List[str]:
    args: List[str] = []
    out_names = []
    out_ops = get_multi_outputs(f)
    out_names = [out.name for out in out_ops]
    non_out_args = [r for arg in f.func.arguments.non_out for r in argument(arg)]
    opt_generator = (
        "OptionalCType(elem=BaseCType(type=BaseCppType(ns='at', name='Generator')))"
    )
    non_out_names = [
        r.name if f"{r.nctype.type}" != opt_generator else "xgen" for r in non_out_args
    ]
    if f"{f.func.name}" in INPLACE_OP_UNCARE_INPUT:
        non_out_names.remove("self")
    args.extend(out_names)
    args.extend(non_out_names)
    return args


def structured_aotops_arguments(
    g: Union[NativeFunction, NativeFunctionsGroup],
) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if isinstance(g, NativeFunctionsGroup):
        out_args = g.out.func.arguments.out
        non_out_args = g.out.func.arguments.non_out
    else:
        out_args = g.func.arguments.out
        non_out_args = g.func.arguments.non_out

    args.extend(out_args)
    args.extend(non_out_args)

    return [r for arg in args for r in structured.argument(arg)]


def out_arguments(g: Union[NativeFunction, NativeFunctionsGroup]) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if isinstance(g, NativeFunctionsGroup):
        args.extend(g.out.func.arguments.out)
    else:
        args.extend(g.func.arguments.out)
    return [r for arg in args for r in structured.argument(arg)]


def aten_to_kernel(name) -> str:
    kernel = ""
    if name in special_name_transfer:
        return special_name_transfer[name]

    names = name.split(".")
    base_name = names[0]
    names = "".join(e.capitalize() for e in base_name.split("_"))
    kernel = "topsaten" + names
    return kernel


###########for c++ aten ops return###########
@dataclass(frozen=True)
class MutRefVectorCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::vector<{self.elem.cpp_type()}> &"

    def cpp_type_registration_declarations(self) -> str:
        return f"::std::vector<{self.elem.cpp_type_registration_declarations()}> &"

    def remove_const_ref(self) -> CType:
        return MutRefVectorCType(self.elem.remove_const_ref())


# Translation of a (non-multi) return type from JIT to C++
# N.B: returntype_type returns a CType, not a NamedCType.
# This is mostly because of the mismatch between return types and return names.
# e.g. a function with a return type of 'void' has 0 return names,
# and a function with a return type of 'std::tuple' has >1 return name.
def atenops_outtype_type(t: Type, *, mutable: bool, symint: bool = False) -> CType:
    # placeholder is ignored
    # NB: symint is ALWAYS respected for return types.  So symint argument
    # here is IGNORED
    r = cpp.valuetype_type(t, binds="__placeholder__", mutable=mutable, symint=True)
    if r is not None:
        return r.type

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                # if local.use_const_ref_for_mutable_tensors():
                #     return ConstRefCType(BaseCType(tensorT))
                # else:
                return MutRefCType(BaseCType(tensorT))
            else:
                # Note [Tensor Copy Returns]
                # Currently, we use "Argument.is_write" to determine
                # whether or not Tensor return types should be copies or references.
                # If that ever changes, take a look at other locations of this note!
                return BaseCType(tensorT)
        elif t.name == BaseTy.Scalar:
            return BaseCType(scalarT)
            # return MutRefCType(BaseCType(tensorT))
    elif isinstance(t, ListType):
        elem = atenops_outtype_type(t.elem, mutable=False)
        assert t.size is None, f"fixed size list returns not supported: {t}"
        return MutRefVectorCType(elem)
    elif isinstance(t, OptionalType):
        elem = atenops_outtype_type(t.elem, mutable=mutable)
        if str(t.elem) == "Tensor":
            return OptionalCType(elem)

    raise AssertionError(f"unrecognized return type {t}")


# Translation of a single aten out to its C++ type
def atenops_out_type(r: Return, *, symint: bool = False) -> CType:
    return atenops_outtype_type(r.type, mutable=r.is_write, symint=symint)


# Translation of a full (possibly multi) aten out from JIT to its C++ type
def atenops_out_types(rs: Sequence[Return], *, symint: bool = False) -> CType:
    if len(rs) == 0:
        return BaseCType(voidT)
    elif len(rs) == 1:
        return atenops_out_type(rs[0], symint=symint)
    else:
        return TupleCType([atenops_out_type(r, symint=symint) for r in rs])


def flat_non_self(
    args: Arguments,
) -> Sequence[Union[Argument, SelfArgument, TensorOptionsArguments]]:
    ret: list[Union[Argument, SelfArgument, TensorOptionsArguments]] = []
    ret.extend(args.pre_self_positional)
    ret.extend(args.post_self_positional)
    ret.extend(args.kwarg_only)
    ret.extend(args.out)
    return ret


def get_extra_outputs(f: NativeFunction) -> Sequence[Return]:
    all_returns = get_multi_outputs(f)
    return_num = len(f.func.returns)
    return all_returns[return_num:]


def get_multi_outputs(f: NativeFunction) -> Sequence[Return]:
    outputs = []
    k = f.func.kind()
    if k is SchemaKind.functional:
        return_names = cpp.return_names(f)
        for i, _return in enumerate(f.func.returns):
            return_name = return_names[i]
            outputs.append(Return(return_name, _return.type, _return.annotation))
        for arg in f.func.arguments.flat_all:
            if arg.is_write:
                outputs.append(Return(arg.name, arg.type, arg.annotation))
    elif k is SchemaKind.inplace:
        self_arg = f.func.arguments.self_arg
        outputs.append(
            Return("self", self_arg.argument.type, self_arg.argument.annotation)
        )
        for arg in flat_non_self(f.func.arguments):
            if arg.is_write:
                outputs.append(Return(arg.name, arg.type, arg.annotation))
    elif k is SchemaKind.out:
        for out in f.func.arguments.out:
            outputs.append(Return(out.name, out.type, out.annotation))
        for arg in f.func.arguments.flat_non_out:
            if arg.is_write:
                outputs.append(Return(arg.name, arg.type, arg.annotation))
    elif k is SchemaKind.mutable:
        return_num = len(f.func.returns)
        if return_num == 0:
            for arg in f.func.arguments.flat_non_out:
                if arg.is_write:
                    outputs.append(Return(arg.name, arg.type, arg.annotation))
        else:
            return_names = cpp.return_names(f)
            for i, _return in enumerate(f.func.returns):
                return_name = return_names[i]
                outputs.append(Return(return_name, _return.type, _return.annotation))
            for arg in f.func.arguments.flat_non_out:
                if arg.is_write:
                    outputs.append(Return(arg.name, arg.type, arg.annotation))
    return outputs


#############################################


def gen_one_unstructured(
    f: NativeFunction,
) -> Tuple[str, str]:
    sig = DispatcherSignature.from_schema(
        f.func,
        prefix="",
        symint=False,
    )

    k = f.func.kind()
    aten_kernel = aten_to_kernel(f"{f.func.name}")

    args = [a.name for a in sig.arguments()]
    args_str = ", ".join(args)
    shape_infer = sig.name() + "_shape_infer"
    return_names = cpp.return_names(f)

    is_foreach_op = f"{f.func.name}".startswith("_foreach")

    sig_body = []

    if is_foreach_op:
        ss = f"  auto use_slow = {sig.name()}_check_slow_path({args_str});\n"
        ss += (
            f"  if (use_slow) {{\n    return {sig.name()}_slow_path({args_str});\n  }}"
        )
        sig_body.append(ss)

    if k is SchemaKind.inplace or k is SchemaKind.out or len(f.func.returns) == 0:
        sig_body.append(f"  {shape_infer}({args_str});")
    elif len(f.func.returns) == 1:
        sig_body.append(f"  auto {return_names[0]} = {shape_infer}({args_str});")
    else:
        sig_body.append(f"  auto outputs = {shape_infer}({args_str});")
        for i, r in enumerate(return_names):
            sig_body.append(f"  auto {r} = std::get<{i}>(outputs);")

    aotops_names = ", ".join(n for n in unstructured_aotops_arguments(f))

    tensor_arg_type, tensor_arg_name = get_first_tensor_out(f)

    out_args = get_multi_outputs(f)
    out_num = len(out_args)
    if out_num == 1 and f"{tensor_arg_type}" == "Tensor":
        sig_body.append(
            f"  if ({tensor_arg_name}.numel() == 0) return {return_names[0]};"
        )

    # Sync global deterministic mode to operator library before bridge call
    if f"{f.func.name}" in support_deterministic_algorithms_list:
        sig_body.append("  // Sync global deterministic mode to operator library")
        sig_body.append("  auto& ctx = at::globalContext();")
        sig_body.append("  bool deterministic_mode = ctx.deterministicAlgorithms();")
        sig_body.append(
            f'  PTDLOG(OP) << "[GCU_DETERMINISTIC] {f.func.name}: Setting deterministic mode to " << deterministic_mode;'
        )
        sig_body.append("  topsaten::topsatenSetDeterministicMode(deterministic_mode);")
        sig_body.append("")

    if (
        sig.name() in ALLOW_NUMBERS_AS_TENSORS_LIST
        or f"{f.func.name}" in BINARY_KERNEL_LHS_IS_SCALAR + BINARY_KERNEL_RHS_IS_SCALAR
    ):
        if (
            sig.name() in ALLOW_NUMBERS_AS_TENSORS_LIST
            or f"{f.func.name}" in BINARY_KERNEL_LHS_IS_SCALAR
        ):
            sig_body.append(
                f"  BINARY_KERNEL_LAUNCH_LHS_IS_SCALAR({aten_kernel}, {aotops_names})"
            )
        if (
            sig.name() in ALLOW_NUMBERS_AS_TENSORS_LIST
            or f"{f.func.name}" in BINARY_KERNEL_RHS_IS_SCALAR
        ):
            sig_body.append(
                f"  BINARY_KERNEL_LAUNCH_RHS_IS_SCALAR({aten_kernel}, {aotops_names})"
            )
        sig_body.append(f"  BINARY_KERNEL_LAUNCH_END({aten_kernel}, {aotops_names})")
    elif f"{f.func.name}" in GENERATOR_OP_WITH_ONE_OUT:
        sig_body.append(
            f"  auto xgen = optionalGeneratorToTopsatenGenerator({return_names[0]}, generator);"
        )
        sig_body.append(f"  bridge_{aten_kernel}_out{out_num}({aotops_names});")
    elif out_num == 1 or f"{f.func.name}" in SPLIT_TOPSATEN_MULTI_OUT_OP:
        sig_body.append(f"  bridge_{aten_kernel}_out{out_num}({aotops_names});")
    else:
        out_type = atenops_out_types(out_args, symint=False).cpp_type()
        need_sync = has_host_returns_value(f)
        input_args, output_args = get_multi_out_input_output(aotops_names, out_num)
        sig_body.append(
            f"  bridge_{aten_kernel}_multi_out(std::make_index_sequence<std::tuple_size<{out_type}>::value>{{}}, {need_sync}, {output_args}, {input_args});"
        )

    if len(f.func.returns) == 1:
        ret_expr = return_names[0]
    elif len(f.func.returns) == 0:
        ret_expr = ""
    elif k == SchemaKind.out:
        ret_expr = f'std::forward_as_tuple({", ".join(return_names)})'
    else:
        moved = ", ".join(f"std::move({name})" for name in return_names)
        ret_expr = f"std::make_tuple({moved})"

    sig_body.append(f"  return {ret_expr};")

    definition = sig.defn()
    defn_shape_infer = sig.defn(sig.name() + "_shape_infer") + ";\n"
    defn = definition + ";\n"
    sig_body_str = "\n".join(sig_body)

    impl = f"""
{definition} {{
{sig_body_str}
}}
"""
    return defn, impl, defn_shape_infer


def gen_one_structured(
    g: Union[NativeFunction, NativeFunctionsGroup],
    f: NativeFunction,
    is_gen_shape_infer: bool = False,
) -> Tuple[str, str]:
    sig = DispatcherSignature.from_schema(
        f.func,
        prefix="",
        symint=False,
    )

    sig_body = []
    context: List[Union[Binding, Expr]] = list(sig.arguments())

    k = f.func.kind()

    class_name = f"structured_{meta.name(g)}_gcu_{k.name}"

    if k is SchemaKind.functional:
        sig_body.append(f"  {class_name} op;")
    elif k is SchemaKind.inplace:
        sig_body.append(f"  {class_name} op(self);")
    elif k is SchemaKind.out:
        out_args_str = ", ".join(a.name for a in f.func.arguments.out)
        sig_body.append(f"  {class_name} op({out_args_str});")

    meta_exprs = ", ".join(
        e.expr for e in translate(context, structured.meta_arguments(g), method=False)
    )

    sig_body.append(f"  op.meta({meta_exprs});")

    aten_kernel = aten_to_kernel(f"{f.func.name}")

    out_args = structured.out_arguments(g)
    for i, out_arg in enumerate(out_args):
        assert ConstRefCType(BaseCType(tensorT)) == out_arg.nctype.type
        # get non-const outputs
        sig_body.append("  // get non-const outputs")
        expr = f"out{i}"
        if k is SchemaKind.out:
            sig_body.append(f"  auto {expr} = op.maybe_get_output({i});")
        else:
            sig_body.append(f"  auto {expr} = op.outputs_[{i}];")

        context.append(
            Expr(
                expr=expr,
                # TODO: Stop hardcoding that the output type is a Tensor.  Note
                # that for the codegen here this is fine because outputs_ is
                # hardcoded to be tensor already
                type=NamedCType(out_arg.nctype.name, MutRefCType(BaseCType(tensorT))),
            )
        )

    aotops_exprs = ", ".join(
        e.expr for e in translate(context, structured_aotops_arguments(g), method=False)
    )

    device_arg_type, device_arg_name = get_first_tensor_out(f, True)

    out_num = 1 if k is SchemaKind.inplace else 0
    out_num = max(out_num, len(f.func.returns), len(f.func.arguments.out))

    # Destructively return the final tensors
    # TODO: Do this in translate instead
    if k is SchemaKind.functional:
        if len(f.func.returns) == 1:
            ret_expr = "std::move(op.outputs_[0])"  # small optimization
        else:
            moved = ", ".join(
                f"std::move(op.outputs_[{i}])" for i in range(len(f.func.returns))
            )
            ret_expr = f"std::make_tuple({moved})"
    elif k is SchemaKind.inplace:
        ret_expr = "self"
    elif k is SchemaKind.out:
        if len(f.func.returns) == 1:
            ret_expr = f.func.arguments.out[0].name
        else:
            refs = ", ".join(a.name for a in f.func.arguments.out)
            ret_expr = f"std::forward_as_tuple({refs})"

    if not is_gen_shape_infer:
        if out_num == 1 and f"{device_arg_type}" == "Tensor":
            sig_body.append(f"  if ({device_arg_name}.numel() == 0) return {ret_expr};")

        # Sync global deterministic mode to operator library before bridge call
        if f"{f.func.name}" in support_deterministic_algorithms_list:
            sig_body.append("  // Sync global deterministic mode to operator library")
            sig_body.append("  auto& ctx = at::globalContext();")
            sig_body.append(
                "  bool deterministic_mode = ctx.deterministicAlgorithms();"
            )
            sig_body.append(
                f'  PTDLOG(OP) << "[GCU_DETERMINISTIC] {f.func.name}: Setting deterministic mode to " << deterministic_mode;'
            )
            sig_body.append(
                "  topsaten::topsatenSetDeterministicMode(deterministic_mode);"
            )
            sig_body.append("")

        if (
            sig.name() in ALLOW_NUMBERS_AS_TENSORS_LIST
            or f"{f.func.name}"
            in BINARY_KERNEL_LHS_IS_SCALAR + BINARY_KERNEL_RHS_IS_SCALAR
        ):
            if (
                sig.name() in ALLOW_NUMBERS_AS_TENSORS_LIST
                or f"{f.func.name}" in BINARY_KERNEL_LHS_IS_SCALAR
            ):
                sig_body.append(
                    f"  BINARY_KERNEL_LAUNCH_LHS_IS_SCALAR({aten_kernel}, {aotops_exprs})"
                )
            if (
                sig.name() in ALLOW_NUMBERS_AS_TENSORS_LIST
                or f"{f.func.name}" in BINARY_KERNEL_RHS_IS_SCALAR
            ):
                sig_body.append(
                    f"  BINARY_KERNEL_LAUNCH_RHS_IS_SCALAR({aten_kernel}, {aotops_exprs})"
                )
            sig_body.append(
                f"  BINARY_KERNEL_LAUNCH_END({aten_kernel}, {aotops_exprs})"
            )
        elif out_num == 1 or f"{f.func.name}" in SPLIT_TOPSATEN_MULTI_OUT_OP:
            sig_body.append(f"  bridge_{aten_kernel}_out{out_num}({aotops_exprs});")
        else:
            out_type = cpp.returns_type(f.func.returns, symint=False).cpp_type()
            need_sync = has_host_returns_value(f)
            input_args, output_args = get_multi_out_input_output(aotops_exprs, out_num)
            sig_body.append(
                f"  bridge_{aten_kernel}_multi_out(std::make_index_sequence<std::tuple_size<{out_type}>::value>{{}}, {need_sync}, {output_args}, {input_args});"
            )

    sig_body.append(f"  return {ret_expr};")

    sig_body_str = "\n".join(sig_body)

    if not is_gen_shape_infer:
        name = sig.name()
        definition = sig.defn()
    else:
        name = sig.name() + "_shape_infer"
        if k is SchemaKind.out:
            definition = sig.defn(sig.name() + "_shape_infer")
        else:
            cpp_args = cpp.arguments(
                f.func.arguments,
                faithful=False,
                symint=False,
                method=False,
                cpp_no_default_args=f.cpp_no_default_args,
            )
            definition = shape_infer_decl(f, cpp_args, name)

    defn = definition + ";\n"
    impl = f"""
{sig.defn(name)} {{
{sig_body_str}
}}
"""
    return defn, impl


def gen_aotops_defn(
    f: NativeFunction,
) -> str:
    sig = DispatcherSignature.from_schema(
        f.func,
        prefix="",
        symint=False,
    )
    definition = sig.defn()
    defn = definition + ";\n"
    return defn


def gen_aot_ops(
    backend_yaml_path: str,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
) -> Tuple[List[str], List[str], List[str], str, str]:
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
        optional_cfg = []  # Allow an empty list of supported ops

    optional = [op["name"] for op in optional_cfg]

    assert isinstance(
        optional, list
    ), f'expected "optional" to be a list, but got: {optional} (of type {type(optional)})'

    supported_op_list = [OperatorName.parse(op) for op in supported] + [
        OperatorName.parse(op) for op in optional
    ]
    op_cfg_map = {
        op_name: op_cfg
        for op_name, op_cfg in zip(supported_op_list, supported_cfg + optional_cfg)
    }

    ops = []
    ops_defn = []
    unstructured_shape_infer = []
    foreach_op_check_slow_path = []
    foreach_op_slow_path = []
    for g in grouped_native_functions:
        native_functions_map: Dict[OperatorName, NativeFunction] = {
            f.func.name: f
            for f in concatMap(
                lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
                [g],
            )
        }
        for f in native_functions_map:
            if f in supported_op_list:
                fun_cfg = op_cfg_map[f]
                if (
                    (fun_cfg.get("gen_aot_ops", True))
                    and (not fun_cfg.get("dynamic_output_shape", False))
                    and f"{f}" not in exclude
                ):
                    if g.structured and isinstance(g, NativeFunctionsGroup):
                        func = native_functions_map[f]
                        with native_function_manager(func):
                            op_defn, op_impl = gen_one_structured(g, func)
                            ops.append(
                                {"func": f"{f}", "op_defn": op_defn, "impl": op_impl}
                            )
                            ops_defn.append({"func": f"{f}", "op_defn": op_defn})
                    else:
                        func = native_functions_map[f]
                        with native_function_manager(func):
                            op_defn, op_impl, defn_shape_infer = gen_one_unstructured(
                                func
                            )
                            ops.append(
                                {"func": f"{f}", "op_defn": op_defn, "impl": op_impl}
                            )
                            ops_defn.append({"func": f"{f}", "op_defn": op_defn})
                            unstructured_shape_infer.append(
                                {"func": f"{f}", "shap_infer": defn_shape_infer}
                            )
                elif not fun_cfg.get("disable", False) and fun_cfg.get(
                    "gen_native_function", True
                ):
                    func = native_functions_map[f]
                    with native_function_manager(func):
                        op_defn = gen_aotops_defn(func)
                        ops_defn.append({"func": f"{f}", "op_defn": op_defn})
                # gen foreach op info
                func = native_functions_map[f]
                if f"{func.func.name}".startswith("_foreach"):
                    with native_function_manager(func):
                        sig = DispatcherSignature.from_schema(
                            func.func,
                            prefix="",
                            symint=False,
                        )
                        args = [a.defn() for a in sig.arguments()]
                        args_str = ", ".join(args)
                        foreach_check_slow_path_decl_str = (
                            f"bool {sig.name()}_check_slow_path({args_str});"
                        )
                        foreach_slow_path_decl_str = f"{sig.returns_type().cpp_type()} {sig.name()}_slow_path({args_str});"
                        foreach_op_check_slow_path.append(
                            foreach_check_slow_path_decl_str
                        )
                        foreach_op_slow_path.append(foreach_slow_path_decl_str)

    foreach_op_check_slow_path_str = "\n\n".join(foreach_op_check_slow_path)
    foreach_op_slow_path_str = "\n\n".join(foreach_op_slow_path)
    ops.sort(key=lambda x: x["func"])
    ops_defn.sort(key=lambda x: x["func"])
    unstructured_shape_infer.sort(key=lambda x: x["func"])

    return (
        [op_defn["op_defn"] for op_defn in ops_defn],
        [op["impl"] for op in ops],
        [defn["shap_infer"] for defn in unstructured_shape_infer],
        foreach_op_check_slow_path_str,
        foreach_op_slow_path_str,
    )
