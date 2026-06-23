#
# Copyright 2020-2022 Enflame. All Rights Reserved.
#

###########################################
####### Integrate BFLOAT16 LIBRARY ########
###########################################

###################################################
# Get bfloat16 project in                         #
# cmake_build/_deps/sw_enflame.open_bfloat16-src  #
# source dir and build dir are                    #
# sw_enflame.open_bfloat16_SOURCE_DIR and         #
# sw_enflame.open_bfloat16_BINARY_DIR             #
# they are effective just in this file.           #
###################################################
include(FetchContent)

FetchContent_Declare (sw_enflame.open_bfloat16
    GIT_REPOSITORY git@git.enflame.cn:sw/enflame.open/bfloat16.git
    GIT_TAG 1.4.0
)
FetchContent_MakeAvailable(sw_enflame.open_bfloat16)

# For target needing bfloat16 and/or float16 libraries
if(BFLOAT16_TEST)
  add_executable(bfloat16_test ${CMAKE_CURRENT_SOURCE_DIR}/bfloat16_test.cc)
  target_link_libraries(bfloat16_test libbfloat16_static)
  # target_link_libraries(bfloat16_test logging_lib_logging logging_lib_util logging_include dtu_config_lib dtu_config_proto)

  add_executable(float16_test ${CMAKE_CURRENT_SOURCE_DIR}/float16_test.cc)
  target_link_libraries(float16_test libfloat16_static)
  # target_link_libraries(float16_test logging_lib_logging logging_lib_util logging_include dtu_config_lib dtu_config_proto)
endif()
