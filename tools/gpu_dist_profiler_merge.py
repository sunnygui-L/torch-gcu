
import os
import json

import argparse
import sys
import fileinput

# 检查输入是否为目录
def is_valid_directory(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    return path

def parse_args():
    parser = argparse.ArgumentParser(description='==> Multi Profiler json merge Function')
    parser.add_argument('--directory', '-d', type=is_valid_directory, help='Input directory path')
    parser.add_argument('--output', '-o', help='The output json file name you want to save.')

    args = parser.parse_args()
    return args

def main(args):
    if (not args.directory) or (not args.output):
        print("!!!! Not avalible use. Please use as: !!!!\n" +
                      "usage: gpu_dist_profiler_merge.py [-d] [--directory] /patch/to/merge [-o] merged_file.json")
    else:
        directory = args.directory
        out_json = args.output

        for i, filename in enumerate(os.listdir(directory)):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                if i == 0:
                    with open(file_path, 'r') as file:
                        data1 = json.load(file)
                        rank_info = data1['distributedInfo']['rank']
                        for event in data1["traceEvents"]:
                            if "args" in event:
                                # 在"args"中添加"rank"字段
                                event["args"]["rank"] = rank_info

                else:
                    with open(file_path, 'r') as file:
                        data2 = json.load(file)
                        rank_info = data2['distributedInfo']['rank']
                        for event in data2["traceEvents"]:
                            if "args" in event:
                                # 在"args"中添加"rank"字段
                                event["args"]["rank"] = rank_info
                        data1['traceEvents'].extend(data2['traceEvents'])
                print(f"[Json Merge] Merge of {file_path} finished.")

        # 将合并后的数据写入新的 JSON 文件
        with open(out_json, 'w') as file:
            json.dump(data1, file, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
