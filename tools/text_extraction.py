
## function 1: only first pass ir
# command 1: python3.6 -u text_extraction.py -i="input.log" -o="save_first_ir.log" -f

## function 2: only last pass ir
# command 2: python3.6 -u text_extraction.py -i="input.log" -o="save_last_ir.log" -l

## function 3: first pass and last pass ir
# command 3: python3.6 -u text_extraction.py -i="input.log" -o="save_first_last_ir.log" -fl

## function 4: all the line with 'LAZYNODE', and python stack
# command 4: python3.6 -u text_extraction.py -i="input.log" -o="save_lazy_node.log" -ln

## function 5: specify the str list needed
# command 5: python3.6 -u text_extraction.py -i="input.log" -o="save_lazy_node.log" -ss "str1" "str2" "str3"

## function 6: first and last ir, and str1 && str2 && str3
# command 6: python3.6 -u text_extraction.py -i="input.log" -o="save_lazy_node.log" -fl -ss "str1" "str2" "str3"

## function 7: first and last ir, and lazynode log
# command 7: python3.6 -u text_extraction.py -i="input.log" -o="save_lazy_node.log" -fl -ln

## ......
# ......

import argparse
import sys
import fileinput

def parse_args():
    parser = argparse.ArgumentParser(description='==> Log Extraction Function')
    parser.add_argument('--input_log_file', '-i', help='the file of the log you want to exctract.')
    parser.add_argument('--output_file_to_save', '-o', help='the file name you want to save.')
    mul_exec_group = parser.add_mutually_exclusive_group() # -f, -l, -fl: Only specify one, not any two or more
    mul_exec_group.add_argument('--only_first_pass', '-f', action='store_true', help='only exctract hlir first pass ir.')
    mul_exec_group.add_argument('--only_last_pass', '-l', action='store_true', help='only exctract hlir last pass ir.')
    mul_exec_group.add_argument('--first_and_last_pass', '-fl', action='store_true', help='Exctract all the first and last hlir pass ir.')
    parser.add_argument('--lazy_node', '-ln', action='store_true', help='Exctract all the log in torch_gcu namespace.')
    parser.add_argument('--specific_string', '-ss', default=None, nargs='+', type=str,
                        help='Specifies a specific set of strings that you want to extract contains.')
    args = parser.parse_args()
    return args


def extract_contain(args, strings):
    read_file_name = args.input_log_file
    save_file_name = args.output_file_to_save

    f = open(save_file_name, "w")
    for line in fileinput.input(read_file_name):
        for s in strings:
            if (s in line):
                f.write(line)


def extract_line(f, line, start_flag, end_flag, start_strs, end_strs, print_end_str):
    for ss in start_strs:
        if (ss in line):
            start_flag[0] = True

    for es in end_strs:
        if (es in line):
            end_flag[0] = True
            if print_end_str:
                f.write(line)

    if (start_flag[0] and end_flag[0]):
        start_flag[0] = False
        end_flag[0] = False

    if start_flag[0] and (not end_flag[0]):
        f.write(line)


def extract_between_with(args, start_strs=[], end_strs=[]):
    read_file_name = args.input_log_file
    save_file_name = args.output_file_to_save

    ir_start = [False]
    ir_end = [False]

    lazy_node_star = [False]
    lazy_node_end = [False]

    f = open(save_file_name, "w")

    for line in fileinput.input(read_file_name):
        extract_line(f, line, ir_start, ir_end, start_strs, end_strs, False)

        # with special strings, such as: -ss "str1" "str2" "str3"
        if args.specific_string != None:
            special_str_list = args.specific_string
            for str in special_str_list:
                if (str in line):
                    f.write(line)

        if args.lazy_node: # -ln
            # write with python stack message
            lazy_stack_starts = [">>>> Python stack start"]
            lazy_stack_ends = [">>>> Python stack end"]
            extract_line(f, line, lazy_node_star, lazy_node_end, lazy_stack_starts, lazy_stack_ends, True)

            # write with LAZYNODE message
            if ("LAZYNODE" in line):
                f.write(line)


def main(args):
    if args.only_first_pass:
        start_strs = ["/// IR Dump Before hlir::HlirFirstPass ///"]
        end_strs = ["============ HLIR Compile Begin ============"]
        extract_between_with(args, start_strs, end_strs)
        return
    elif args.only_last_pass:
        start_strs = ["/// IR Dump Before hlir::HlirLastPass ///"]
        end_strs = ["============ HLIR Compile Finish ============"]
        extract_between_with(args, start_strs, end_strs)
        return
    elif args.first_and_last_pass:
        start_strs = ["/// IR Dump Before hlir::HlirFirstPass ///"]
        start_strs.append("/// IR Dump Before hlir::HlirLastPass ///")
        end_strs = ["============ HLIR Compile Begin ============"]
        end_strs.append("============ HLIR Compile Finish ============")
        extract_between_with(args, start_strs, end_strs)
        return
    elif args.lazy_node:
        extract_between_with(args)
        return
    elif args.specific_string:
        contains = args.specific_string
        extract_contain(args, contains)
        return
    else:
        print("!!!! Not avalible use. Please use as: !!!!\n" +
              "usage: text_extraction.py [-h] [--input_log_file/-i INPUT_LOG_FILE] \n\
                          [--output_file_to_save/-o OUTPUT_FILE_TO_SAVE] \n\
                          [--only_first_pass/-f | --only_last_pass/-l | --first_and_last_pass/-fl] \n\
                          [--lazy_node/-ln] \n\
                          [--specific_string/-ss SPECIFIC_STRING [SPECIFIC_STRING ...]]")


if __name__ == '__main__':
    args = parse_args()
    main(args)
