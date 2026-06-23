# CAI_E2:Added FetchContent to download libtorch_archive repository during cmake
# configuration step CAI_E2:Corrected URL and URL_HASH to use linux version of
# libtorch_archive
include(FetchContent)
# set(FETCHCONTENT_QUIET FALSE)
set(libtorch_archive_header "")

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    set(torch_link_x86 "module_package/libtorch/libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip")
endif ()
set(PACKAGE_CMDS "unzip /FILE/")
set(PACKAGE_FILES ${CMAKE_FPKG_LIBDIR}/libtorch)
if (NOT PROJECT_GIT_URL)
    fetchFromArtifactory(torch_x86
        FILE ${torch_link_x86}
        EXTRACT
        PKG_COMMAND ${PACKAGE_CMDS}
        PKG_FILES ${PACKAGE_FILES}
    )
else()
    set(git_name "torch_binary")

    string(REGEX REPLACE "(.*)/(.*)" "\\2" torch_binary_file_name "${torch_link_x86}")
    set(fetch_file_name "*")

    message("factor fetch_file_name: ${fetch_file_name}")
    execute_process(
        COMMAND bash -c "cd ${CMAKE_CURRENT_BINARY_DIR};GIT_LFS_SKIP_SMUDGE=1 git clone ${PROJECT_GIT_URL}/${git_name}.git;cd ${git_name};git lfs pull --include=${fetch_file_name}"
    )

    message("torch_binary_file_name: ${CMAKE_CURRENT_BINARY_DIR}/${git_name}/${torch_binary_file_name}")

    file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_BINARY_DIR}/${git_name}/${torch_binary_file_name}"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${git_name}")


    file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_BINARY_DIR}/${git_name}/${torch_binary_file_name}"
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${git_name}/libtorch")

    set(torch_x86_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${git_name}/libtorch" CACHE STRING "")
    message("torch libtorch_archive_SOURCE_DIR: ${torch_x86_SOURCE_DIR}")

endif()