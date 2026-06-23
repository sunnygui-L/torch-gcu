
import subprocess
from typing import Any, Dict, Tuple
import torch
import sys,os
from torch_gcu import _C


torch._C._conv_determine_backend_memory_format = _C._conv_determine_backend_memory_format
torch._C._is_any_autocast_enabled = _C._is_any_autocast_enabled

# let the logs of each GCU in a separate file
def _popen_gcu(self, args: Tuple, env: Dict[str, str]) -> subprocess.Popen:
    kwargs: Dict[str, Any] = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    log_dir = args.log_dir if ("log_dir" in args and args.log_dir is not None) else os.getcwd()
    log_name = os.path.join(log_dir,'{}_log'.format(env['RANK']))
    f = open(log_name,'w')
    print("the device {} log in {}".format(env['RANK'], log_name))
    return subprocess.Popen(
        args=args,
        env=env,
        stdout=f,
        stderr=f,
        **kwargs,
    )

if os.getenv('SAVE_LOG_BY_DEVICE'):
  from torch.distributed.elastic.multiprocessing.subprocess_handler import SubprocessHandler
  SubprocessHandler._popen = _popen_gcu