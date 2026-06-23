import os
import subprocess
import glob
import stat
import sys
import traceback
import platform
from pathlib import Path
from typing import Union
from looseversion import LooseVersion

UNKNOWN = "unknown"


def get_tag(base_dir, tops_version) -> str:
    PACKAGE_VERSION = os.getenv('PACKAGE_VERSION', default='')
    if PACKAGE_VERSION == '' or PACKAGE_VERSION == '123.456':
        PACKAGE_VERSION = tops_version + "." + subprocess.check_output(
            ["git", "show", "-s", "--date=format:'%Y%m%d'", "--format=%cd"],
            cwd=base_dir).decode("ascii").strip().replace("'", "")
    return PACKAGE_VERSION

def get_commit(base_dir) -> str:
    return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=base_dir).decode("ascii").strip()

def get_tops_version(version_file_path):
    tops_version = UNKNOWN
    with open(version_file_path, 'r') as file:
        tops_version = file.read().strip()
    return tops_version

def get_sha(pytorch_root: Union[str, Path]) -> str:
    try:
        return (subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        cwd=pytorch_root)  # Compliant
                .decode("ascii").strip())
    except Exception:
        return UNKNOWN


def generate_torch_gcu_version_file(torch_gcu_version, tops_version,
                                    debug_mode):
    torch_gcu_root = Path(__file__).parent
    version_path = torch_gcu_root / "torch_gcu" / "version.py"
    if version_path.exists():
        version_path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    sha = get_sha(torch_gcu_root)
    with os.fdopen(os.open(version_path, flags, modes), 'w') as f:
        f.write("__all__ = ['__version__', 'git_version', 'gcu', 'debug']\n")
        f.write(
            "__version__ = '{version}'\n".format(version=torch_gcu_version))
        f.write("git_version = {}\n".format(repr(sha)))
        f.write("gcu = '{}'\n".format(str(tops_version)))
        f.write("debug = {}\n".format(debug_mode))


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def get_cmake_command():

    def _get_version(cmd):
        for line in subprocess.check_output([cmd, '--version'
                                             ]).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')

    "Returns cmake command."
    cmake_command = 'cmake'
    if platform.system() == 'Windows':
        return cmake_command
    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.18.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.18.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.18.0 found')


def get_build_type():
    build_type = 'Release'
    if os.getenv('DEBUG',
                 default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
        build_type = 'Debug'

    if os.getenv('REL_WITH_DEB_INFO',
                 default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
        build_type = 'RelWithDebInfo'

    return build_type


def get_coverage_flag():
    coverage_flag = False
    if os.getenv('USE_COVERAGE',
                 default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
        coverage_flag = True

    return coverage_flag


def get_build_type_dir(base_dir):
    build_dir = ""
    if os.getenv("USE_CI") == "1":
        build_dir = os.getenv("BUILD_DIR", default=os.path.join(base_dir, "../cmake_build"))
    else:
        build_dir = os.path.join(base_dir, "build")

    build_type = get_build_type()

    if build_type == "Release":
        build_type_dir = os.path.join(build_dir, "Release")
    elif build_type == "Debug":
        build_type_dir = os.path.join(build_dir, "Debug")
    elif build_type == "RelWithDebInfo":
        build_type_dir = os.path.join(build_dir, "RelWithDebInfo")
    else:
        raise RuntimeError(f'unknown build type: {build_type} !')

    return build_type_dir


def _get_build_mode():
    for i in range(1, len(sys.argv)):
        if not sys.argv[i].startswith('-'):
            return sys.argv[i]

    raise RuntimeError("Run setup.py without build mode.")


def get_pytorch_dir():
    try:
        import torch
        return os.path.dirname(os.path.abspath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


def generate_bindings_code(base_dir, torch_version):
    python_execute = sys.executable
    generate_code_cmd = [
        "bash",
        os.path.join(base_dir, 'generate_code.sh'), python_execute,
        torch_version
    ]
    if subprocess.call(generate_code_cmd) != 0:  # Compliant
        print('Failed to generate ATEN bindings: {}'.format(generate_code_cmd),
              file=sys.stderr)
        sys.exit(1)


def get_src_py_and_dst(base_dir):

    py_src_dir = os.path.join(base_dir, "torch_gcu")
    py_build_dir = f"{get_build_type_dir(base_dir)}/packages/torch_gcu"

    ret = []
    generated_python_files = glob.glob(os.path.join(py_src_dir, '**/*.py'),
                                       recursive=True)
    for src in generated_python_files:
        dst = os.path.join(py_build_dir, os.path.relpath(src, py_src_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))
    return ret



def is_build_c10d_eccl():
    return not (os.getenv("USE_C10D_ECCL") in ['OFF', '0', 'NO', 'FALSE', 'N'])

def is_build_kineto_gcu():
    return not (os.getenv("USE_KINETO_GCU") in ['OFF', '0', 'NO', 'FALSE', 'N'])