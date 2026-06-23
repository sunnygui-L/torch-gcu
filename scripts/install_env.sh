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
sudo ./${TOPSPLATFORM_FILE} -y --container -v

# find topsaten
TOPSOP_COMMITID=$(grep 'set(TOPSOP_COMMITID' ${ROOT_DIR}/torch_gcu/cmake/fetch_dependences.cmake | sed -E 's/.*set\(TOPSOP_COMMITID "([^"]+)"\).*/\1/')
echo "Found TOPSOP_COMMITID is: $TOPSOP_COMMITID"

TOPSOP_PACKAGE_VERSION=$(grep 'set(TOPSOP_PACKAGE_VERSION' ${ROOT_DIR}/torch_gcu/cmake/fetch_dependences.cmake | sed -E 's/.*set\(TOPSOP_PACKAGE_VERSION "([^"]+)"\).*/\1/')
echo "Found TOPSOP_PACKAGE_VERSION is: $TOPSOP_PACKAGE_VERSION"

TOPSATEN_FILE=cmake_build/Release/topsaten_binary/${TOPSOP_COMMITID}/topsaten_cape_${TOPSOP_PACKAGE_VERSION}-1_amd64.deb
pull_from_repo "topsaten_binary" "${TOPSOP_COMMITID}/topsaten_cape_${TOPSOP_PACKAGE_VERSION}-1_amd64.deb"

if [ ! -f "$TOPSATEN_FILE" ]; then
    echo "Error: Topsaten file still not found after pulling"
    exit 1
fi
echo "Found TOPSATEN_FILE is: $TOPSATEN_FILE"

# find tccl
PCALS_COMMITID=$(grep 'set(PCALS_COMMITID' ${ROOT_DIR}/torch_gcu/cmake/fetch_dependences.cmake | sed -E 's/.*set\(PCALS_COMMITID "([^"]+)"\).*/\1/')
echo "Found PCALS_COMMITID is: $PCALS_COMMITID"

PCALS_ECCL_PACKAGE_VERSION=$(grep 'set(PCALS_ECCL_PACKAGE_VERSION' ${ROOT_DIR}/torch_gcu/cmake/fetch_dependences.cmake | sed -E 's/.*set\(PCALS_ECCL_PACKAGE_VERSION "([^"]+)"\).*/\1/')
echo "Found PCALS_ECCL_PACKAGE_VERSION is: $PCALS_ECCL_PACKAGE_VERSION"

TCCL_FILE=cmake_build/Release/pcals_binary/${PCALS_COMMITID}/tccl_${PCALS_ECCL_PACKAGE_VERSION}-1_amd64.deb
pull_from_repo "pcals_binary" "${PCALS_COMMITID}/tccl_${PCALS_ECCL_PACKAGE_VERSION}-1_amd64.deb"

if [ ! -f "$TCCL_FILE" ]; then
    echo "Error: TCCL file still not found after pulling"
    exit 1
fi
echo "Found TCCL_FILE is: $TCCL_FILE"

sudo dpkg -i --force-overwrite ${TCCL_FILE} ${TOPSATEN_FILE}
