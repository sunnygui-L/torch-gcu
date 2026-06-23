#!/bin/bash
# Copyright 2024 Enflame. All Rights Reserved.
#
set -eu -o pipefail
BUILD_ROOT_DIR=$(pwd)
set -x
ARCH=$(uname -m)
BUILD_TORCH_VERSION=2.10.0

function ci_build() {
  cd /home/pypi_packages
  sudo python3 -m pip install torch-2.10.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl torchvision-0.25.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
  cd -
  cd ${project_name}
  sudo python3 -m pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com
  python3 setup.py bdist_wheel
}

function main() {
  $build_job_name
}

export project_name=${project_name:-"torch_gcu"}
export cpu=${process_num:-"10"}

if [ "$#" -eq 1 ]; then
  build_job_name=$1
else
  echo "donot support this build job"
  exit 1
fi

main "$@"
exit $?
         