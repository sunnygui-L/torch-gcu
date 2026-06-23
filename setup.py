import multiprocessing
import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import re
import shutil
import subprocess

from sysconfig import get_paths
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools import setup, Extension
from setuptools.command.build_clib import build_clib
from wheel.bdist_wheel import bdist_wheel
import torch
from build_utils import (
    get_tag,
    get_pytorch_dir,
    get_cmake_command,
    get_build_type,
    get_build_type_dir,
    get_src_py_and_dst,
    _get_build_mode,
    get_coverage_flag,
    generate_bindings_code,
    generate_torch_gcu_version_file,
    get_tops_version,
    is_build_c10d_eccl,
    get_commit,
    is_build_kineto_gcu,
)
import platform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_GIT_URL = os.getenv("PROJECT_GIT_URL", default="")
CAPE_BUILD = os.getenv("CAPE_BUILD", default="OFF").upper() in ["ON", "1", "YES", "TRUE", "Y"]

def is_arm_platform():
    machine = platform.machine().lower()
    return "arm" in machine or "aarch" in machine


def CppExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for C++.
    """
    pytorch_dir = get_pytorch_dir()
    temp_include_dirs = kwargs.get("include_dirs", [])
    temp_include_dirs.append(os.path.join(pytorch_dir, "include"))
    temp_include_dirs.append(
        os.path.join(pytorch_dir, "include/torch/csrc/api/include")
    )
    kwargs["include_dirs"] = temp_include_dirs

    temp_library_dirs = kwargs.get("library_dirs", [])
    temp_library_dirs.append(os.path.join(pytorch_dir, "lib"))
    temp_library_dirs.append("/opt/tops/lib")

    kwargs["library_dirs"] = temp_library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    libraries.append("torch_python")

    kwargs["libraries"] = libraries
    kwargs["language"] = "c++"
    return Extension(name, sources, *args, **kwargs)


FETCH_DEPS = os.getenv("FETCH_DEPS")

if os.getenv("PACKAGE_COMMITID") != "":
    PACKAGE_COMMITID = os.getenv("PACKAGE_COMMITID")
else:
    PACKAGE_COMMITID = get_commit(BASE_DIR)


class CPPLibBuild(build_clib, object):

    def run(self):

        cmake = get_cmake_command()

        if cmake is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )
        self.cmake = cmake

        if os.getenv("USE_CI") == "1":
            build_dir = os.getenv(
                "BUILD_DIR", default=os.path.join(BASE_DIR, "../cmake_build")
            )
        else:
            build_dir = os.path.join(BASE_DIR, "build")

        build_type = get_build_type()
        build_type_dir = get_build_type_dir(BASE_DIR)

        output_lib_path = os.path.join(build_type_dir, "packages/torch_gcu/lib")

        btabbr = "dbg" if build_type == "Debug" else "rel"
        arch = platform.machine().lower()
        install_path = os.path.join(build_dir, f"{arch}-linux-{btabbr}")

        os.makedirs(build_type_dir, exist_ok=True)
        os.makedirs(output_lib_path, exist_ok=True)
        os.makedirs(install_path, exist_ok=True)

        self.build_lib = os.path.relpath(
            os.path.join(build_type_dir, "packages/torch_gcu")
        )
        self.build_temp = os.path.relpath(build_type_dir)

        cmake_args = [
            "-DCMAKE_BUILD_TYPE=" + get_build_type(),
            "-DCMAKE_INSTALL_PREFIX=" + os.path.abspath(install_path),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + os.path.abspath(output_lib_path),
            "-DTORCHGCU_INSTALL_LIBDIR=" + os.path.abspath(output_lib_path),
            "-DPYTHON_INCLUDE_DIR=" + get_paths()["include"],
            "-DPYTORCH_INSTALL_DIR=" + get_pytorch_dir(),
            "-B " + build_type_dir,
            "-G Ninja",
        ]
        subprocess.check_call(
            [self.cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ
        )

        if os.getenv("CPU_COUNT"):
            cpu_count = int(os.getenv("CPU_COUNT"))
        else:
            cpu_count = multiprocessing.cpu_count()
        build_args = ["-j", str(cpu_count)]

        subprocess.check_call(["ninja"] + ["install"] + build_args, cwd=build_type_dir)

        subprocess.check_call(["ninja"] + build_args, cwd=build_type_dir)


class Build(build_ext, object):

    def run(self):
        self.run_command("build_clib")
        self.build_lib = os.path.relpath(f"{get_build_type_dir(BASE_DIR)}/packages")
        self.build_temp = os.path.relpath(f"{get_build_type_dir(BASE_DIR)}")
        self.library_dirs.append(
            os.path.relpath(f"{get_build_type_dir(BASE_DIR)}/packages/torch_gcu/lib")
        )
        super(Build, self).run()


class InstallCmd(install):

    def finalize_options(self) -> None:
        self.build_lib = os.path.relpath(f"{get_build_type_dir(BASE_DIR)}/packages")
        if os.getenv("USE_COVERAGE") == "1":
            os.chdir(BASE_DIR + "/../cmake_build")
            cmd = "cd Release && find -name '*.gcno' ! -path './third_party/*' ! -path './_deps/*' ! -path './x86_64-linux-rel/*' | xargs -i tar rf gcno.tar {}"
            subprocess.call(cmd, shell=True)
            os.chdir(BASE_DIR + "/../cmake_build")
            cmd = "cp Release/gcno.tar x86_64-linux-rel/coverage"
            subprocess.call(cmd, shell=True)
        return super(InstallCmd, self).finalize_options()


class PythonPackageBuild(build_py, object):

    def run(self) -> None:
        self.build_lib = os.path.relpath(f"{get_build_type_dir(BASE_DIR)}/packages")
        ret = get_src_py_and_dst(BASE_DIR)
        for src, dst in ret:
            self.copy_file(src, dst)
        super(PythonPackageBuild, self).run()


class BdistWheelBuild(bdist_wheel):

    def run(self):
        bdist_wheel.run(self)


# check .git/hooks directory has pre-commit file, if not exist, create soft link from githools/pre-commit to .git/hooks/pre-commit
if not os.path.exists(".git/hooks/pre-commit"):
    if os.path.exists("githooks/pre-commit"):
        os.symlink("../../githooks/pre-commit", ".git/hooks/pre-commit")
        print("create soft link from githooks/pre-commit to .git/hooks/pre-commit")

# check pytorch version
TORCH_VERSION = torch.__version__.split("+")[0]
first_v, second_v, _ = TORCH_VERSION.split(".")
assert (
    first_v == "2" and second_v == "10"
), f"torch_gcu must be compiled with torch == 2.10.x, but your PyTorch version is {torch.__version__}."

TOPS_VERSION = get_tops_version(f"{BASE_DIR}/.version")
if str(os.getenv("FIX_VERSION")) == "1":
    TORCH_GCU_VERSION = f"{TORCH_VERSION}"
else:
    TORCH_GCU_VERSION = f"{TORCH_VERSION}+{get_tag(BASE_DIR, TOPS_VERSION)}"

DEBUG = os.getenv("DEBUG", default="").upper() in ["ON", "1", "YES", "TRUE", "Y"]

generate_torch_gcu_version_file(TORCH_GCU_VERSION, TOPS_VERSION, DEBUG)

build_mode = _get_build_mode()
if build_mode not in ["clean"]:
    # Generate bindings code, including RegisterGCU.cpp, RegisterOptionalGCU.cpp & GCUNativeFunctions.h.
    generate_bindings_code(BASE_DIR, TORCH_VERSION)

topsaten_include = f"{get_build_type_dir(BASE_DIR)}/topsaten_binary/usr/include/gcu"

# Setup include directories folders.
include_directories = [
    os.path.join(BASE_DIR),
    os.path.join(BASE_DIR, "torch_gcu"),
    os.path.join(BASE_DIR, "torch_gcu", "csrc"),
    f"/opt/tops/include/logging",
]

if os.getenv("USE_CI") == "1":
    include_directories.append(f"{get_build_type_dir(BASE_DIR)}/opt/tops/include/")
    include_directories.append(topsaten_include)
else:
    include_directories.append("/opt/tops/include/")
    include_directories.append("/usr/include/gcu/")

extra_link_args = []

extra_compile_args = [
    "-std=c++17",
    "-Wno-sign-compare",
    "-Wno-deprecated-declarations",
    "-Wno-return-type",
    "-U_GLIBCXX_USE_CXX11_ABI",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
]
os.environ["_GLIBCXX_USE_CXX11_ABI"] = "1"

if is_build_c10d_eccl():
    extra_compile_args.append("-DUSE_C10D_ECCL")

if is_build_kineto_gcu():
    extra_compile_args.append("-DUSE_KINETO_GCU")

RUNTIME_1_5_1_1016 = os.getenv("RUNTIME_1_5_1_1016", default="ON")
if RUNTIME_1_5_1_1016 == "ON":
    extra_compile_args.append("-DRUNTIME_1_5_1_1016=ON")
else:
    extra_compile_args.append("-DRUNTIME_1_5_1_1016=OFF")

if re.match(r"clang", os.getenv("CC", "")):
    extra_compile_args += [
        "-Wno-macro-redefined",
        "-Wno-return-std-move",
    ]

sanitizer = os.getenv("SANITIZER")

if DEBUG:
    extra_compile_args += ["-O0", "-g"]
    extra_link_args += ["-O0", "-g", "-Wl,-z,now"]
else:
    if sanitizer:
        extra_link_args += ["-Wl,-z,now"]
    else:
        extra_compile_args += ["-DNDEBUG"]
        extra_link_args += ["-Wl,-z,now,-s"]

if get_coverage_flag():
    extra_compile_args += ["-fprofile-arcs", "-ftest-coverage"]
    extra_link_args += ["-lgcov"]

if sanitizer:
    if sanitizer == "address":
        extra_compile_args += ["-fsanitize=address", "-fno-omit-frame-pointer"]
        extra_link_args += ["-fsanitize=address"]
    elif sanitizer == "thread":
        extra_compile_args += ["-fsanitize=thread", "-fno-omit-frame-pointer"]
        extra_link_args += ["-fsanitize=thread"]

classifiers = [
    "Development Status :: 1 - Beta",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "License :: OSI Approved ::",
    "Programming Language :: Python :: 3",
    "Topic :: Software Development :: AI",
    "Topic :: Software Development :: Libraries :: Python Modules",
]


# remove old dependencies
def remove_folder_if_exists(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"removing old {folder_path}...")


if os.getenv("USE_CI") != "1" and FETCH_DEPS in ["ON", "1", "YES", "TRUE", "Y"]:
    deps_dir = os.path.join(get_build_type_dir(BASE_DIR), "_deps")
    remove_folder_if_exists(deps_dir)

LONG_DESC = "Backend for torch2.x to run on gcu."


def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


install_requires = read_requirements()

_packages = ["torch_gcu"]
_package_dir: dict[str, str] = {}

setup(
    name="torch_gcu",
    version=TORCH_GCU_VERSION,
    description="Backend for pytorch to run on gcu. CommitID=({})".format(
        PACKAGE_COMMITID
    ),
    long_description=LONG_DESC,
    classifiers=classifiers,
    keywords="torch_gcu",
    url="",
    author="",
    author_email="",
    license="MIT",
    packages=_packages,
    **({"package_dir": _package_dir} if _package_dir else {}),
    libraries=[("torch_gcu", {"sources": list()})],
    install_requires=install_requires,
    ext_modules=[
        CppExtension(
            "torch_gcu._C",
            sources=["torch_gcu/csrc/InitGcuBindings.cpp"],
            libraries=["torch_gcu_python"],
            include_dirs=include_directories,
            extra_compile_args=extra_compile_args + ["-fstack-protector-all"],
            library_dirs=["lib"],
            extra_link_args=extra_link_args + ["-Wl,-rpath,$ORIGIN/lib"],
        ),
    ],
    package_data={
        "torch_gcu": [
            "*.so",
            "lib/*.so*",
        ],
    },
    entry_points={
        "torch.backends": [
            "torch_gcu = torch_gcu:_autoload",
        ],
    },
    data_files=[],
    cmdclass={
        "build_clib": CPPLibBuild,
        "build_ext": Build,
        "bdist_wheel": BdistWheelBuild,
        "build_py": PythonPackageBuild,
        "install": InstallCmd,
    },
)
