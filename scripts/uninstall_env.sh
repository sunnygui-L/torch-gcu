#!/bin/bash

ROOT_DIR=`pwd`
echo "Current dir: $ROOT_DIR"
PROJECT_GIT_URL="${PROJECT_GIT_URL:-}"
# Function to pull files using git lfs
pull_from_repo() {
    local repo_name=$1
    local fetch_file_name=$2

    if [ -z "$repo_name" ] || [ -z "$fetch_file_name" ]; then
        echo "Error: Missing arguments. Usage: pull_from_repo <repo_name> <fetch_file_name>"
        return 1
    fi
    cd ${ROOT_DIR}/cmake_build/Release
    if [ -d "${repo_name}" ]; then
        echo "${repo_name} is already exist"
    else
        GIT_LFS_SKIP_SMUDGE=1 git clone ${PROJECT_GIT_URL}/${repo_name}.git
    fi

    echo "Pulling from repo: $repo_name, file: $fetch_file_name"
    cd ${repo_name} && chmod -x ${fetch_file_name} && git lfs pull --include=${fetch_file_name}
    if [ $? -ne 0 ]; then
        echo "Error: Failed to pull file $fetch_file_name from repo $repo_name"
        return 1
    fi
    cd ${ROOT_DIR}
}

set -x
# find TopsPlatform
CAPS_COMMIT=$(grep 'set(PREBUILD_CAPS_COMMIT' ${ROOT_DIR}/torch_gcu/CMakeLists.txt | sed -E 's/.*set\(PREBUILD_CAPS_COMMIT ([a-f0-9]+)\).*/\1/')
CAPS_COMMIT_SHORT=$(echo "$CAPS_COMMIT" | cut -c1-6)
echo "Found CAPS_COMMIT is: $CAPS_COMMIT"

CAPS_PACKAGE_VERSION=$(grep "set(PREBUILD_CAPS_VERSION_BIG" ${ROOT_DIR}/torch_gcu/CMakeLists.txt | sed -E 's/.*set\(PREBUILD_CAPS_VERSION_BIG ([0-9.]+)\).*/\1/')
echo "Found CAPS_PACKAGE_VERSION is: $CAPS_PACKAGE_VERSION"

TOPSPLATFORM_FILE=cmake_build/Release/caps_binary/${CAPS_COMMIT}/TopsPlatform_cape_${CAPS_PACKAGE_VERSION}-${CAPS_COMMIT_SHORT}_rpm_x86_64.run
pull_from_repo "caps_binary" "${CAPS_COMMIT}/TopsPlatform_cape_${CAPS_PACKAGE_VERSION}-${CAPS_COMMIT_SHORT}_rpm_x86_64.run"

if [ ! -f "$TOPSPLATFORM_FILE" ]; then
    echo "Error: TopsPlatform file still not found after pulling"
    exit 1
fi
echo "Found TOPSPLATFORM_FILE is: $TOPSPLATFORM_FILE"

chmod +x ${TOPSPLATFORM_FILE}
sudo ./${TOPSPLATFORM_FILE} -y --uninstall --container -v

sudo dpkg -r --force-depends --force-overwrite tccl topsaten || true
sudo rpm -e --nodeps tccl topsaten || true
