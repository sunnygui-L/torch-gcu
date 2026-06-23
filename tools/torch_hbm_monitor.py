import datetime
import linecache
import torch
import socket
import os

# Configs
TRUE_VAL = ('true', '1', 't')
PRINT_TENSOR_INFO = os.getenv(
    'PRINT_TENSOR_INFO', 'True').lower() in TRUE_VAL
USE_INCREMENTAL = os.getenv(
    'USE_INCREMENTAL', 'False').lower() in TRUE_VAL
DUMP_TO_FILE = os.getenv('DUMP_TO_FILE', '0').lower() in TRUE_VAL
PRINT_TO_TERNIMAL = os.getenv(
    'PRINT_TO_TERNIMAL', '1').lower() in TRUE_VAL
print(
    f"Configs:\n \
    PRINT_TENSOR_INFO:{PRINT_TENSOR_INFO}\n \
    USE_INCREMENTAL:{USE_INCREMENTAL}\n \
    DUMP_TO_FILE:{DUMP_TO_FILE}\n \
    PRINT_TO_TERNIMAL:{PRINT_TO_TERNIMAL}\n")

USE_GPU = False
USE_GCU = False
dump_file = None
if torch.cuda.is_available():
    gpu_profile_fn = f"GPU_memprof_pid_{os.getpid()}_{datetime.datetime.now():%d-%b-%y-%H-%M-%S}.prof.txt"
    if DUMP_TO_FILE:
        print('profiling GPU usage to ', gpu_profile_fn)
    dump_file = gpu_profile_fn
    USE_GPU = True
else:
    try:
        import torch_gcu
    except ModuleNotFoundError:
        print('torch_gcu not found!')
    else:
        gcu_profile_fn = f"GCU_memprof_pid_{os.getpid()}_{datetime.datetime.now():%d-%b-%y-%H-%M-%S}.prof.txt"
        if DUMP_TO_FILE:
            print('profiling GCU usage to ', gcu_profile_fn)
        dump_file = gcu_profile_fn
        USE_GCU = True

# Global variables
last_tensor_sizes = set()
last_meminfo_used = 0
cur_lineno = None
last_lineno = None
last_module_name = None
cur_module_name = None
last_filename = None
cur_filename = None
last_func_name = None
cur_func_name = None


def enable_line(module_name, filename, func_name, lineno):
    # Example
    # 'module_name' include all module level, just like a.b.c
    # 'filename' include directory, use os.path.basename(filename) to get file name only
    # 'a' in filename                                       # Enable code in file xxx/a.py
    # 'b' in os.path.dirname(os.path.abspath(filename))     # Enable code in directory xxx/b/xxx
    # 'c' in func_name                                      # Enable code in function b
    # 'd' in module_name                                    # Enable code in python module xxx.d.xxx
    # 10 == lineno                                          # Enable code in line 10
    # Use 'and' 'or' to concat different condition
    return True


def get_frame_info(frame):
    func_name = frame.f_code.co_name
    module_name = frame.f_globals["__name__"]
    lineno = frame.f_lineno
    filename = frame.f_globals["__file__"]
    if (filename.endswith(".pyc") or
            filename.endswith(".pyo")):
        filename = filename[:-1]
    return module_name, filename, func_name, lineno


def hbm_monitor(frame, event, arg):
    global last_tensor_sizes
    global last_meminfo_used
    global cur_lineno, last_lineno
    global last_module_name, cur_module_name
    global last_filename, cur_filename
    global last_func_name, cur_func_name
    global dump_file

    if event == 'line':
        try:
            cur_module_name, cur_filename, cur_func_name, cur_lineno = get_frame_info(
                frame)
            if not enable_line(cur_module_name, cur_filename, cur_func_name, cur_lineno):
                cur_lineno = None
            # about _previous_ line (!)
            if cur_lineno and last_lineno:
                line = linecache.getline(last_filename, last_lineno)
                where_str = last_module_name + ' ' + \
                    last_func_name + ':' + str(last_lineno)
                if USE_GPU:
                    new_meminfo_used = torch.cuda.memory_allocated()
                elif USE_GCU:
                    new_meminfo_used = torch_gcu.get_hbm_alloc_size()
                else:
                    print('Unsupported device!')
                    exit(1)
                mem_display = new_meminfo_used - \
                    last_meminfo_used if USE_INCREMENTAL else new_meminfo_used
                if not DUMP_TO_FILE:
                    dump_file = '/dev/null'
                with open(dump_file, 'a+') as f:
                    def out(info):
                        if DUMP_TO_FILE:
                            f.write(info)
                        if PRINT_TO_TERNIMAL:
                            print(info)

                    info = f"{where_str:<50}"\
                        f":{(mem_display)/1024**2:<10.10f}Mb "\
                        f"{line.rstrip()}\n"
                    out(info)
                    last_meminfo_used = new_meminfo_used
                    if PRINT_TENSOR_INFO is True:
                        for tensor in get_tensors():
                            if not hasattr(tensor, 'dbg_alloc_where'):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {(id(x), type(x), tuple(x.size()), x.dtype, x.dbg_alloc_where)
                                            for x in get_tensors()}
                        for xid, t, s, d, loc in new_tensor_sizes - last_tensor_sizes:
                            info = f' + {loc:<50} {str(xid):<30} {str(s):<20} {str(d):<20} {str(t):<10}\n'
                            out(info)
                        for xid, t, s, d, loc in last_tensor_sizes - new_tensor_sizes:
                            info = f' - {loc:<50} {str(xid):<30} {str(s):<20} {str(d):<20} {str(t):<10}\n'
                            out(info)
                        last_tensor_sizes = new_tensor_sizes
            if not last_lineno or cur_lineno:
                last_lineno = cur_lineno
                last_module_name = cur_module_name
                last_filename = cur_filename
                last_func_name = cur_func_name

            return hbm_monitor

        except (KeyError, AttributeError):
            pass

    return hbm_monitor


def get_tensors():
    import gc
    for obj in gc.get_objects():
        tensor = []
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue
            if (USE_GPU and tensor.is_cuda) or (USE_GCU and tensor.is_gcu()):
                yield tensor
        except Exception:
            pass
