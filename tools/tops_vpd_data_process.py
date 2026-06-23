
import os
import json

import argparse
import sys
import fileinput
from decimal import Decimal

try:
    import sqlite3
    import openpyxl
except ModuleNotFoundError:
    raise ImportError('sqlite3 or openpyxl not found! Please install with cmd: apt-get install sqlite3 or pip install openpyxl') from e

import shutil
import os

# cmd 1:
# python3.10 -u tops_vpd_data_process.py \
#   --func 1 \
#   --vpd_path /path/to/xxx.vpd \
#   --start_timestamp 1731314108870366326 \
#   --end_timestamp 1731314120926574759 \
#   --save_name top_vpd_export

# cmd 2:
# python3.10 -u tops_vpd_data_process.py \
#   --func 2 \
#   --vpd_path /path/to/xxx.vpd \
#   --dummy_kernel_name xxx_gcu_kernel \
#   --save_name export_dummy_kernel_split

def parse_args():
    parser = argparse.ArgumentParser(description='==> topsprof vpd file for data process, function 1: gcu kernel statistic; function 2: split GCU Kernels to two lines, while Dummy Kernels exist.')
    parser.add_argument('--func', type=int, help='When func=1: gcu kernel statistic; and func=2 is split GCU Kernels to two lines, while Dummy Kernels exist.')
    parser.add_argument('--vpd_path', help='Path to the vpd file you want to parse')
    parser.add_argument('--start_timestamp', help='The start timestamp you want to parse in vpd.')
    parser.add_argument('--end_timestamp', help='The end timestamp you want to parse in vpd.')
    parser.add_argument('--dummy_kernel_name', help='Specify the dummy kernel name of topsaten op.')
    parser.add_argument('--save_name', default="top_vpd_export", help='The name of the file you want to save.')

    args = parser.parse_args()
    return args


def copy_and_rename_file(src_path, dst_path):
    """
    :param src_path
    :param dst_path
    """
    # 检查源文件是否存在
    if not os.path.exists(src_path):
        print(f"Failed： src file '{src_path}' not found")
        return

    if os.path.exists(dst_path):
        print(f"Failed： dst file '{dst_path}' has exist, if you had remove first, and re-run.")
        return

    # 复制文件
    shutil.copy2(src_path, dst_path)
    print(f"文件已复制并重命名为: {dst_path}")


def gcu_kernel_statistic(args):
    vpd_path = args.vpd_path
    start_time = args.start_timestamp
    end_time = args.end_timestamp

    conn = sqlite3.connect(vpd_path)
    cursor = conn.cursor()
    cursor.execute('''
SELECT * FROM gcu_op
WHERE start_timestamp >= ?
AND end_timestamp <= ?
ORDER BY `name` ASC;
''', (start_time, end_time))

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "sheet1"

    ws.append(["Op名称", "持续时间(ms)", "百分比", "数量"])

    rows = cursor.fetchall()
    content_row = []
    total_time = 0
    total_num = 0
    for i, res in enumerate(rows):
        name = res[1]

        # 查找 '<' 和 '(' 的位置, 截取kernel名称，避免过长
        index_less_than = name.find('<')
        index_parenthesis = name.find('(')

        if index_less_than == -1:
            index_less_than = float('inf')
        if index_parenthesis == -1:
            index_parenthesis = float('inf')

        index = min(index_less_than, index_parenthesis)

        if index != float('inf'):
            name = name[:index]

        duration = res[7]
        total_time = total_time + duration
        total_num = total_num + 1

        print(f"[{name}, {duration}],")
        content_row.append([name, duration, 0.0, 1])

    merged_dict = {}
    for row in content_row:
        name = row[0]
        if name.startswith("void matmul_kernel_"):
            name = "void matmul_kernel_xxxx"
        if name.startswith("void sdp_flash_attention_"):
            name = "void sdp_flash_attention_xxxx"
        if name in merged_dict:
            merged_dict[name][0] += row[1]
            merged_dict[name][1] += row[2]
            merged_dict[name][2] += row[3]
        else:
            merged_dict[name] = row[1:]

    merged_list = [[name] + values for name, values in merged_dict.items()]

    content_row.clear()

    for row in merged_list:
        name = row[0]
        duration = row[1]
        percentage = duration / total_time
        percentage = f"{percentage:.6f}"
        percentage = f"{Decimal(percentage):.2%}"
        number = row[3]
        content_row.append([name, f"{duration / 1000 / 1000:.6f}", percentage, number])

    ws.append(["总计", f"{total_time / 1000 / 1000:.6f}", "100.00%", total_num])
    ws.append(["", "", "", ""])
    for r in content_row:
        ws.append(r)

    wb.save(args.save_name + ".xlsx")

    conn.commit()
    conn.close()

def dummy_kernel_split(args):
    vpd_path = args.vpd_path
    kernel_name = args.dummy_kernel_name

    export_vpd_path = args.save_name + ".vpd"
    copy_and_rename_file(vpd_path, export_vpd_path)

    conn = sqlite3.connect(export_vpd_path)
    cursor = conn.cursor()
    cursor.execute('''
-- 步骤 1: 从 gcu_op 表中提取 view_id 中的 x 值
CREATE TABLE extracted_x AS
SELECT SUBSTR(view_id, INSTR(view_id, 'dev:') + 4, INSTR(view_id, ',user:') - INSTR(view_id, 'dev:') - 4) AS x
FROM gcu_op
WHERE name LIKE '%' || ? || '%';
''', (kernel_name,))

    cursor.execute('''
-- 步骤 2: 从 dev 表中找到 row_name 包含 'GCU Kernels' 的行，并复制这些行
-- 先找到需要复制的行
CREATE TABLE to_copy AS
SELECT *
FROM dev
WHERE row_name LIKE '%GCU Kernels%'
  AND id IN (SELECT x FROM extracted_x);
''')

    cursor.execute('''
-- 复制这些行，并修改 row_name 为 'Dummy Kernels', id 自增
INSERT INTO dev (id, row_name, node, category)  -- 替换为实际的列名
SELECT (SELECT MAX(id) FROM dev) + ROW_NUMBER() OVER (ORDER BY id), 'Dummy Kernels', node, category
FROM to_copy;
''')

    cursor.execute('''
-- 步骤 3: 建立 x 值与新 id 的映射关系
CREATE TABLE x_to_new_id AS
SELECT x, new_id
FROM (
    SELECT x, ROW_NUMBER() OVER (ORDER BY x) AS rn
    FROM extracted_x
) AS x_values
JOIN (
    SELECT id AS new_id, ROW_NUMBER() OVER (ORDER BY id) AS rn
    FROM dev
    WHERE row_name = 'Dummy Kernels'
) AS new_ids
ON x_values.rn = new_ids.rn;
''')

    cursor.execute('''
-- 步骤 4: 更新 gcu_op 表中的 view_id 列
UPDATE gcu_op
SET view_id = (
    SELECT REPLACE(gcu_op.view_id, 'dev:' || x, 'dev:' || new_id)
    FROM x_to_new_id
    WHERE gcu_op.view_id LIKE '%dev:' || x || ',user:%'
)
WHERE name LIKE '%' || ? || '%'
  AND EXISTS (
      SELECT 1
      FROM x_to_new_id
      WHERE gcu_op.view_id LIKE '%dev:' || x || ',user:%'
  );
''', (kernel_name,))

    cursor.execute('''
-- 清理临时表
DROP TABLE extracted_x;
''')

    cursor.execute('''
DROP TABLE to_copy;
''')

    cursor.execute('''
DROP TABLE x_to_new_id;
''')

    conn.commit()
    conn.close()

def main(args):
    if args.func == 1:
        print(f'Function1: You are use the tools to statistic vpd kernels for performance data.')
        gcu_kernel_statistic(args)
    elif args.func == 2:
        print(f'Function2: You are use the tools to split Dummy Kernels to two lines in GCU Kernels.')
        dummy_kernel_split(args)
    else:
        print(f'Function: {args.func} is not supported.')
        return

if __name__ == '__main__':
    args = parse_args()
    main(args)
