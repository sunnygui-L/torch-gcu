#
# Copyright 2020-2023 Enflame. All Rights Reserved.
#
import os

import json
import argparse
import time
import warnings

import torch
import torch_gcu


# result:
# {
#     "xx_op":{
#                 "aten_op_name": ,
#                 "dynamic_op": ,
#                 "fail_count": ,
#                 "fail_cases":
#                     [
#                         {"params": params, "acc_msg": acc_msg},
#                     ],
#                 "fail_case": {},
#                 "params_count" : ,
#                 "schema": ,
#                 "topsaten_name": ,
#                 "use_count":
#             },
# }
results = {}
config = {}

def prepare_args(params):

    def creat_tensor(tensor_info):

        def creat_contiguous_tensor(dtype, shape):
            float_map = {"Double": torch.float32, "Float": torch.float32, "Half": torch.float16}
            int_map = {"Long": torch.int64, "Int": torch.int32}
            bool_map = {"Bool": torch.bool}

            if dtype in float_map.keys():
                return torch.randn(size=shape, dtype=float_map[dtype])
            elif dtype in int_map.keys():
                return torch.randint(high=100, size=shape, dtype=int_map[dtype])
            elif dtype in bool_map.keys():
                return torch.randint(low=0, high=2, size=shape, dtype=bool_map[dtype])
            else:
                raise NotImplementedError(f"{dtype} tensor not support!")

        dtype = tensor_info["dtype"]
        shape = tensor_info["size"]

        if tensor_info["is_contiguous"]:
            return creat_contiguous_tensor(dtype, shape)
        else:
            strides = tensor_info["strides"]
            max_size = -1
            for size_i, stride_i in zip(shape, strides):
                max_size = max(max_size, size_i * stride_i)
            result = creat_contiguous_tensor(dtype, (max_size, ))
            return torch.as_strided(result, shape, strides)

    # TODO
    # not support tensorlist, support later
    inputs_cpu = {}
    inputs_gcu = {}
    for param_i in params["inputs"]:
        if param_i["args_type"] == "Tensor":
            tmp = creat_tensor(param_i["value"])
            inputs_cpu[param_i["name"]] = tmp
            inputs_gcu[param_i["name"]] = tmp.gcu()
        elif param_i["args_type"] == "Optional[Tensor]":
            if param_i["value"] != "None":
                tmp = creat_tensor(param_i["value"])
                inputs_cpu[param_i["name"]] = tmp
                inputs_gcu[param_i["name"]] = tmp.gcu()
        else:
            inputs_cpu[param_i["name"]] = param_i["value"]
            inputs_gcu[param_i["name"]] = param_i["value"]

    return inputs_cpu, inputs_gcu


def record_msg_init(op_params):
    results[op_params["aten_op_name"]] = {
        "aten_op_name": op_params["aten_op_name"],
        "dynamic_op": op_params["dynamic_op"],
        "fail_count": 0,
        "fail_cases": [],
        "params_count": op_params["params_count"],
        "schema": op_params["schema"],
        "topsaten_name": op_params["topsaten_name"],
        "use_count": op_params["use_count"],
    }


def dump_file():
    global results
    results_list = list(results.values())

    dump_path = os.path.join(config["dump_path"], 'debug_from_op_statistic.json')
    with open(dump_path, "w", encoding='utf8')as f:
        f.write(json.dumps(results_list, indent=4))


def dump_tensor(op_name, idx, out_cpu, out_gcu):
    dump_path = os.path.join(config["dump_path"], op_name)
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    out_cpu_path = os.path.join(dump_path, op_name + f'_{idx}_out_cpu.pt')
    torch.save(out_cpu.clone(), out_cpu_path)

    out_gcu_path = os.path.join(dump_path, op_name + f'_{idx}_out_gcu.pt')
    torch.save(out_gcu.clone().cpu(), out_gcu_path)


def check_out(op_name, idx, out_cpu, out_gcu):
    out_gcu = out_gcu.cpu()
    mask = torch.isclose(out_cpu, out_gcu, rtol=config["rtol"],
                         atol=config["atol"], equal_nan=config["equal_nan"])
    acc_pass = mask.all().item()

    acc_msg = None
    if not acc_pass:
        diff_out_cpu = out_cpu.masked_select(mask)
        diff_out_gcu = out_gcu.masked_select(mask)

        acc_msg = f"test {op_name} params idx: {idx} case acc check failed!\n"

        left = " " * 8

        acc_msg += left + "out_cpu:" + " " * 16 + "out_gcu:\n"

        print_num = min(100, diff_out_cpu.numel())
        for i in range(print_num):
            acc_msg += left + f"{diff_out_cpu[i]:<24.12}{diff_out_gcu[i]:<.12}\n"

    return acc_pass, acc_msg


def op_test(op):

    aten_op_name = op["aten_op_name"]
    case_num = op["params_count"]

    _, torch_op = aten_op_name.split("::")

    check_pass = True
    for idx in range(case_num):
        case_i_params = op["params"][idx]
        assert case_i_params["idx"] == idx + 1

        inputs_cpu, inputs_gcu = prepare_args(case_i_params)

        out_cpu = eval(f"torch.ops.aten.{torch_op}(**inputs_cpu)")
        out_gcu = eval(f"torch.ops.aten.{torch_op}(**inputs_gcu)")

        acc_pass, acc_msg = check_out(aten_op_name, case_i_params["idx"], out_cpu, out_gcu)

        if not acc_pass:
            check_pass = False

            if aten_op_name not in results.keys():
                record_msg_init(op)

            fail_case = {
                "params": case_i_params,
                "acc_msg": acc_msg,
            }
            results[aten_op_name]["fail_cases"].append(fail_case)
            results[aten_op_name]["fail_count"] = results[aten_op_name]["fail_count"] + 1

            if config["op_check_dump"]:
                dump_tensor(torch_op, case_i_params["idx"], out_cpu, out_gcu)

            if not config["op_check_no_break"]:
                if config["op_check_dump"]:
                    del results[aten_op_name]["fail_count"]
                    results[aten_op_name]["fail_case"] = results[aten_op_name]["fail_cases"][0]
                    del results[aten_op_name]["fail_cases"]
                    del results[aten_op_name]["params_count"]
                    del results[aten_op_name]["use_count"]
                    dump_file()
                raise AssertionError(acc_msg)
    if check_pass:
        print(f"{aten_op_name} check pass!")


def prepare_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--atol", default=1e-08, type=float, help="absolute tolerance.")
    parser.add_argument("--rtol", default=1e-05, type=float, help="relative tolerance.")
    parser.add_argument("--equal_nan", default='False', type=str, help="if True, then two NaNs will be considered equal.")
    parser.add_argument('--op_check_no_break', default='True', type=str, help="if True, the test will no break when case fail.")
    parser.add_argument('--op_check_dump', default='False', type=str, help="if true, the test result will dump.")
    parser.add_argument('--op_check', default='all_ops', type=str, help="which op need to check.")
    parser.add_argument('--op_statistic_path', type=str, required=True, help="op statistic dump file path.")
    parser.add_argument('--dump_path', default='~', type=str, help="the path where result to dump.")

    args = parser.parse_args()

    config["atol"] = args.atol
    config["rtol"] = args.rtol
    config["equal_nan"] = True if args.equal_nan == 'True' else False
    config["op_check_no_break"] = True if args.op_check_no_break == 'True' else False
    config["op_check_dump"] = True if args.op_check_dump == 'True' else False
    config["op_check"] = args.op_check

    base_path = os.getcwd()
    config["op_statistic_path"] = os.path.join(base_path, os.path.relpath(args.op_statistic_path, base_path))
    if config["op_check_dump"]:
        dump_path = args.dump_path
        if dump_path == '~':
            dump_path = base_path
        dump_path = os.path.join(base_path, os.path.relpath(dump_path, base_path))
        current_time = time.strftime('%Y%m%d', time.localtime())
        pid = os.getpid().__str__()
        config["dump_path"] = os.path.join(dump_path, 'debug_from_op_statistic_' + current_time + '_' + pid)
        if not os.path.exists(config["dump_path"]):
            os.makedirs(config["dump_path"])

    print("debug from op_statistic config: {")
    print("  atol:", config["atol"])
    print("  rtol:", config["rtol"])
    print("  equal_nan:", config["equal_nan"])
    print("  op_check_no_break:", config["op_check_no_break"])
    print("  op_check_dump:", config["op_check_dump"])
    print("  op_check:", config["op_check"])
    print("  op_statistic_path:", config["op_statistic_path"])
    if config["op_check_dump"]:
        print("  dump_path:", config["dump_path"])
    print("}")


if __name__ == '__main__':
    prepare_config()

    # TODO
    # auto generate later
    can_not_op_check = [
        "aten::_copy_from",
        "aten::_copy_from_and_resize",
        "aten::_local_scalar_dense",
        "aten::_pin_memory",
        "aten::_reshape_alias",
        "aten::_resize_output_",
        "aten::_scaled_dot_product_flash_attention",
        "aten::as_strided",
        "aten::binomial",
        "aten::empty.memory_format",
        "aten::empty_strided",
        "aten::equal",
        "aten::exponential_",
        "aten::is_pinned",
        "aten::is_set_to",
        "aten::multinomial",
        "aten::multinomial.out",
        "aten::native_dropout",
        "aten::native_dropout_backward",
        "aten::normal_",
        "aten::poisson",
        "aten::randperm.generator_out",
        "aten::record_stream",
        "aten::resize_",
        "aten::set_",
        "aten::set_.source_Storage",
        "aten::set_.source_Storage_storage_offset",
        "aten::set_.source_Tensor",
        "aten::uniform_",
        "aten::view",
        "aten::view_as_complex",
        "aten::view_as_real",
        "aten::_amp_foreach_non_finite_check_and_unscale_",
        "aten::_amp_update_scale_",
        "aten::_has_compatible_shallow_copy_type",
        "aten::convolution_backward",
    ]

    # TODO
    # remove after everything is ok
    support = [
        "aten::_softmax.out",
        "aten::add.out",
        "aten::arange.start_out",
        "aten::bmm.out",
        "aten::argmax.out",
        "aten::clamp.out",
        "aten::convolution",
        "aten::cos.out",
        "aten::div.out",
        "aten::eq.Tensor_out",
        "aten::exp.out",
        "aten::fill_.Scalar",
        "aten::gelu.out",
        "aten::linear",
        "aten::lt.Tensor_out",
        "aten::masked_fill_.Scalar",
        "aten::mul.out",
    ]

    # TODO
    # remove after everything is ok, just record
    # not_support = [
    #     "aten::cat.out", # List[Tensor]
    #     "aten::index.Tensor", # List[Optional[Tensor]]
    #     "aten::index_select", # param 'index' can't use random
    #     "aten::native_group_norm", # return Tuple(Tensor)
    #     "aten::native_layer_norm", # return Tuple(Tensor)
    # ]

    include_list = []
    exclude_list = []
    is_all_ops = False
    for op in config["op_check"].split(","):
        op = op.strip()
        if op == "all_ops":
            is_all_ops = True
        elif op.startswith("-"):
            exclude_list.append(op[1:])
        else:
            include_list.append(op)

    with open(config["op_statistic_path"], 'r', encoding='utf8') as f:
        op_statistics = json.load(f)

        for op in op_statistics:
            if op["aten_op_name"] not in exclude_list \
               and (is_all_ops or op["aten_op_name"] in include_list):
                if op["aten_op_name"] in can_not_op_check:
                    warnings.warn(f'{op["aten_op_name"]} can not op_check!')
                elif op["aten_op_name"] not in support:
                    warnings.warn(f'{op["aten_op_name"]} not support!')
                else:
                    op_test(op)

        if config["op_check_dump"]:
            dump_file()

        if len(results.keys()) == 0:
            print("check success, no fail!!!")
        else:
            for op in results.keys():
                fail_num = results[op]["fail_count"]
                fail_list = []
                for fail_case in results[op]["fail_cases"]:
                    fail_list.append(fail_case["params"]["idx"])
                print(f"test {op} has {fail_num} params cases failed, failed params id: {fail_list}")
                for fail_case in results[op]["fail_cases"]:
                    print(fail_case["acc_msg"])
